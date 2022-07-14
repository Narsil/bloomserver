#![feature(async_closure)]
use actix_web::middleware::Logger;
use actix_web::{
    http::header::ContentType, http::StatusCode, post, web, web::Bytes, App, HttpResponse,
    HttpServer, ResponseError,
};
use crossbeam_channel::{bounded, unbounded, Receiver, Select, Sender};
use memmap::MmapOptions;
use safetensors::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tch::{kind, Device, IndexOp, Tensor};
use thiserror::Error;
use tokenizers::Tokenizer;

const PADDING_IDX: i64 = 3;
const EPS: f64 = 1e-5;

#[derive(Clone)]
struct Config {
    n_head: i64,
    hidden_size: i64,
    n_layer: i64,
    kind: kind::Kind,
    slow_but_exact: bool,
    pretraining_tp: usize,
}

impl Config {
    fn new() -> Self {
        Self {
            n_head: 112,
            hidden_size: 14336,
            n_layer: 70,
            kind: kind::Kind::BFloat16,
            slow_but_exact: true,
            pretraining_tp: 4,
        }
    }
    fn new350m() -> Self {
        Self {
            n_head: 16,
            hidden_size: 1024,
            n_layer: 24,
            kind: kind::Kind::Half,
            slow_but_exact: false,
            pretraining_tp: 1,
        }
    }

    fn new_testing() -> Self {
        Self {
            n_head: 8,
            hidden_size: 64,
            n_layer: 2,
            kind: kind::Kind::BFloat16,
            slow_but_exact: true,
            pretraining_tp: 2,
        }
    }
}

#[derive(Clone)]
struct LayoutConfig {
    layers_first_thread: usize,
    layers_per_thread: usize,
    layers_last_thread: usize,
    n_threads: usize,
    embeddings_filename: String,
    final_filename: String,
    layer_template_filename: String,
}

impl LayoutConfig {
    fn new() -> Self {
        Self {
            layers_first_thread: 0,
            layers_per_thread: 5,
            layers_last_thread: 0,
            n_threads: 14,
            embeddings_filename: "./bloom-embedding.bin".to_string(),
            final_filename: "./bloom-final.bin".to_string(),
            layer_template_filename: "./bloom-h.{}.bin".to_string(),
        }
    }

    fn new_testing() -> Self {
        Self {
            layers_first_thread: 0,
            layers_per_thread: 1,
            layers_last_thread: 0,
            n_threads: 2,
            embeddings_filename: "./bloom-testing.bin".to_string(),
            final_filename: "./bloom-testing.bin".to_string(),
            layer_template_filename: "./bloom-testing.bin".to_string(),
        }
    }

    fn new_dgx() -> Self {
        Self {
            layers_first_thread: 0,
            layers_per_thread: 5,
            layers_last_thread: 0,
            n_threads: 2,
            embeddings_filename: "./bloom-embedding.bin".to_string(),
            final_filename: "./bloom-final.bin".to_string(),
            layer_template_filename: "./bloom-h.1.bin".to_string(),
        }
    }

    fn new350m() -> Self {
        Self {
            layers_first_thread: 0,
            layers_per_thread: 12,
            layers_last_thread: 0,
            n_threads: 2,
            embeddings_filename: "./bloom-350m.bin".to_string(),
            final_filename: "./bloom-350m.bin".to_string(),
            layer_template_filename: "./bloom-350m.bin".to_string(),
        }
    }
}

type PastLayer = (Tensor, Tensor);
type Past = Vec<PastLayer>;
/// Size of batch and response channel
type Ack = (i64, Sender<(Tensor, Past)>);
type Msg = (Tensor, Past, Ack);
type Msg2 = ((Tensor, Tensor, Tensor, Past), Vec<Ack>);
type InChan = Sender<Msg>;
type RChan1 = Receiver<Msg>;
type RChan = Receiver<Msg2>;
type SChan = Sender<Msg2>;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Sampling {
    top_p: Option<f64>,
    top_k: Option<usize>,
    #[serde(default = "default_temperature")]
    temperature: f64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct BeamSearch {
    num_beams: usize,
}

fn default_temperature() -> f64 {
    1.0
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Greedy {}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum GenerationMode {
    BeamSearch(BeamSearch),
    Sampling(Sampling),
    Greedy,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(try_from = "IntermediateParameters")]
struct Parameters {
    generation_mode: GenerationMode,
    max_new_tokens: usize,
}

impl TryFrom<IntermediateParameters> for Parameters {
    // TODO: impl proper error type instead of `String`
    type Error = String;

    fn try_from(data: IntermediateParameters) -> Result<Self, Self::Error> {
        let generation_mode = match (data.do_sample, data.num_beams) {
            (Some(do_sample), _) if do_sample => GenerationMode::Sampling(Sampling {
                temperature: data.temperature.unwrap_or(1.0),
                top_k: data.top_k,
                top_p: data.top_p,
            }),
            (_, Some(num_beams)) => GenerationMode::BeamSearch(BeamSearch { num_beams }),
            _ => {
                if let Some(temperature) = data.temperature {
                    GenerationMode::Sampling(Sampling {
                        temperature: data.temperature.unwrap_or(1.0),
                        top_k: data.top_k,
                        top_p: data.top_p,
                    })
                } else if let Some(top_k) = data.top_k {
                    GenerationMode::Sampling(Sampling {
                        temperature: data.temperature.unwrap_or(1.0),
                        top_k: data.top_k,
                        top_p: data.top_p,
                    })
                } else if let Some(top_p) = data.top_p {
                    GenerationMode::Sampling(Sampling {
                        temperature: data.temperature.unwrap_or(1.0),
                        top_k: data.top_k,
                        top_p: data.top_p,
                    })
                } else {
                    GenerationMode::Greedy
                }
            }
        };
        Ok(Parameters {
            generation_mode,
            max_new_tokens: data.max_new_tokens.unwrap_or(20),
        })
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            generation_mode: GenerationMode::Greedy,
            max_new_tokens: 20,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct IntermediateParameters {
    do_sample: Option<bool>,
    num_beams: Option<usize>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    temperature: Option<f64>,
    max_new_tokens: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Generation {
    inputs: String,
    #[serde(default)]
    parameters: Parameters,
}

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("Queue is full")]
    QueueFull,
}

impl ResponseError for GenerationError {
    fn status_code(&self) -> StatusCode {
        match self {
            GenerationError::QueueFull => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

fn empty_past(config: &Config) -> Past {
    let kind = (config.kind, Device::Cuda(0));
    let p = config.n_head;
    let q = config.hidden_size / config.n_head;
    let past_key = Tensor::zeros(&[1, 0, p, q], kind);
    let past_value = Tensor::zeros(&[1, 0, p, q], kind);
    let past_key_values: Vec<_> = (0..config.n_layer)
        .map(|_| (past_key.copy(), past_value.copy()))
        .collect();
    past_key_values
}

// struct Stream {
//     input_ids: Tensor,
//     new_generated_tokens: usize,
//     payload: Generation,
//     in_channel: InChan,
//     prio_channel: InChan,
//     tokenizer: Arc<Tokenizer>,
//     start: std::time::Instant,
//     sent: bool,
//     sx: Sender<(Tensor, Past)>,
//     rx: Receiver<(Tensor, Past)>,
//     beam_scores: Option<Tensor>,
// }
//
// impl Stream {
//     fn new(
//         payload: Generation,
//         in_channel: InChan,
//         prio_channel: InChan,
//         tokenizer: Arc<Tokenizer>,
//     ) -> Self {
//         let encoded = tokenizer.encode(payload.inputs.clone(), false).unwrap();
//         let mut ids: Vec<_> = encoded.get_ids().iter().map(|&i| i as i64).collect();
//         // TODO The model is not able to handle empty input ids
//         if ids.len() == 0 {
//             ids.push(0);
//         }
//         let start = std::time::Instant::now();
//         let (sx, rx) = bounded::<(Tensor, Past)>(0);
//         let kind = (kind::Kind::Int64, Device::Cuda(0));
//         let input_ids = Tensor::of_slice(ids.as_slice())
//             .to_kind(kind.0)
//             .to_device(kind.1)
//             .view((1, -1));
//         Self {
//             payload,
//             in_channel,
//             prio_channel,
//             tokenizer,
//             new_generated_tokens: 0,
//             input_ids,
//             start,
//             sent: false,
//             sx,
//             rx,
//             beam_scores: None,
//         }
//     }
//
//     fn is_finished(&self) -> bool {
//         match &self.payload.parameters {
//             Parameters::Greedy(params) => self.new_generated_tokens >= params.max_new_tokens,
//             Parameters::BeamSearch(params) => self.new_generated_tokens >= params.max_new_tokens,
//             Parameters::Sampling(params) => self.new_generated_tokens >= params.max_new_tokens,
//         }
//     }
//
//     fn get_ids(&self) -> Vec<u32> {
//         // TODO handle batching maybe
//         // first row should be ~ok.
//         self.input_ids
//             .i((0,))
//             .iter::<i64>()
//             .unwrap()
//             .map(|i| i as u32)
//             .collect()
//     }
//     }
// }
//
// impl futures::Stream for Stream {
//     type Item = Result<Bytes, GenerationError>;
//
//     fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
//         let this = self.get_mut();
//         if this.is_finished() {
//             return Poll::Ready(None);
//         }
//
//         // TODO This config does not really matter
//         // as we're not using past right now
//         let config = Config::new350m();
//
//         let past_key_values = empty_past(&config);
//
//         let ack = (1, this.sx.clone());
//         if this.new_generated_tokens == 0 {
//             this.in_channel
//                 .try_send((this.input_ids.copy(), past_key_values, ack))
//                 .map_err(|_| {
//                     println!("Queue was full {:?}", this.in_channel.len());
//                     GenerationError::QueueFull
//                 })?;
//         } else {
//             this.prio_channel
//                 .send((this.input_ids.copy(), past_key_values, ack))
//                 .expect("This send should always work");
//         }
//         this.sent = true;
//         let start = Instant::now();
//         let (logits, _r_past_key_values) = this.rx.recv().unwrap();
//         println!(
//             "Wasted {:?} on thread {:?}",
//             start.elapsed(),
//             std::thread::current().name()
//         );
//         this.sent = false;
//         this.add_next_id(&logits);
//         // past_key_values = r_past_key_values;
//         // input_ids = Tensor::f_cat(&[input_ids, new_id.copy()], 1).unwrap();
//         this.new_generated_tokens += 1;
//
//         if this.is_finished() {
//             let n = this.new_generated_tokens;
//             println!(
//                 "Inference generated {n} tokens in {:?} ({:?}/tok)",
//                 this.start.elapsed(),
//                 this.start.elapsed().div_f32(n as f32)
//             );
//             let full_ids = this.get_ids();
//             let string = this.tokenizer.decode(full_ids, false).unwrap();
//             let result = serde_json::to_string(&json!([{ "generated_text": string }])).unwrap();
//             Poll::Ready(Some(Ok(Bytes::copy_from_slice(result.as_bytes()))))
//         } else {
//             Poll::Ready(Some(Ok(Bytes::copy_from_slice(b""))))
//         }
//         // }
//         // }
//     }
// }
//
// #[post("/generate")]
// async fn generate(payload: web::Json<Generation>, state: web::Data<AppState>) -> HttpResponse {
//     let state = state.into_inner();
//     let stream = Stream::new(
//         payload.into_inner(),
//         state.in_channel.clone(),
//         state.prio_channel.clone(),
//         state.tokenizer.clone(),
//     );
//     HttpResponse::Ok()
//         .content_type(ContentType::json())
//         .streaming(stream)
// }

fn filter_top_p(scored_logits: &Tensor, top_p: f64) -> Tensor {
    let B = scored_logits.size()[0];
    let filter_value = f64::NEG_INFINITY;
    let descending = true;
    let (sorted_logits, sorted_indices) = scored_logits.sort(-1, descending);
    // cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    let cumulative_probs = sorted_logits
        .softmax(-1, kind::Kind::Float)
        .cumsum(-1, kind::Kind::Float);

    // # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    // sorted_indices_to_remove = cumulative_probs > self.top_p
    let mut sorted_indices_to_remove = cumulative_probs.gt(top_p);
    // if self.min_tokens_to_keep > 1:
    //     # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
    //     sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
    // # Shift the indices to the right to keep also the first token above the threshold
    // sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    // sorted_indices_to_remove[..., 0] = 0

    // sorted_indices_to_remove = sorted_indices_to_remove
    //     .i((0..B, 1..S))
    //     .fill_tensor(&sorted_indices_to_remove.i((0..B, 0..S - 1)).view((-1)));
    sorted_indices_to_remove.i((0..B, -1)).f_fill_(0).unwrap();
    sorted_indices_to_remove = sorted_indices_to_remove.roll(&[1], &[1]);

    // # scatter sorted tensors to original indexing
    // indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    let indices_to_remove = sorted_indices_to_remove
        .f_scatter(1, &sorted_indices, &sorted_indices_to_remove)
        .unwrap();
    scored_logits
        .f_masked_fill(&indices_to_remove, filter_value)
        .unwrap()
    // scores = scores.masked_fill(indices_to_remove, self.filter_value)
    // return scores
}

fn add_next_id(input_ids: &Tensor, params: &Parameters, logits: &Tensor) -> Tensor {
    // TODO handle batching
    match &params.generation_mode {
        GenerationMode::Greedy => {
            let S = logits.size()[1];
            let new_ids = logits
                .i((0..1, S - 1..S))
                .argmax(-1, false)
                .to_device(input_ids.device());
            Tensor::f_cat(&[input_ids.copy(), new_ids.copy()], 1).unwrap()
        }
        GenerationMode::Sampling(params) => {
            let S = logits.size()[1];
            let B = logits.size()[0];
            let filter_value = f64::NEG_INFINITY;
            let last_logits = logits.i((0, S - 1..S)).to_device(input_ids.device());

            let mut scored_logits = last_logits / params.temperature;

            if let Some(top_k) = params.top_k {
                // indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
                let largest = true;
                let sorted = true;
                let top_ks = scored_logits.topk(top_k as i64, -1, largest, sorted).0;
                let size = top_ks.size();
                let top_k = top_ks.i((0..size[0], -1));

                let indices_to_remove = scored_logits.le_tensor(&top_k);
                scored_logits = scored_logits.masked_fill(&indices_to_remove, filter_value);
            }

            if let Some(top_p) = params.top_p {
                // sorted_logits, sorted_indices = torch.sort(scores, descending=True)

                scored_logits = filter_top_p(&scored_logits, top_p);
            }

            let probs = scored_logits.f_softmax(-1, kind::Kind::Float).unwrap();
            let new_ids = probs.f_multinomial(1, false).unwrap();
            Tensor::f_cat(&[input_ids.copy(), new_ids.copy()], 1).unwrap()
        }
        _ => todo!(),
        // Parameters::BeamSearch(params) => {
        //     let num_beams = params.num_beams as i64;
        //     if self.new_generated_tokens == 0 {
        //         // We're in the first step it's rather easy.
        //         let S = logits.size()[1];
        //         let last_logits = logits.i((0, S - 1..S)).to_device(self.input_ids.device());
        //         // Actually cast to logits so we can save the scores
        //         let last_logits = last_logits.f_log_softmax(-1, kind::Kind::Float).unwrap();
        //         let largest = true;
        //         let sorted = true;
        //         let top_ks = last_logits.topk(num_beams, -1, largest, sorted);
        //         let values = top_ks.0;
        //         let indices = top_ks.1;

        //         // repeat input_ids to fit the new tokens
        //         let input_ids = self.input_ids.repeat(&[num_beams, 1]);
        //         let new_ids = indices.to_device(self.input_ids.device());

        //         // new_ids is now of shape [1, num_beams]
        //         // input_ids is of shape [num_beams, seq_length]
        //         // So transposing to we can concatenate the indices within input_ids
        //         let new_ids = new_ids.transpose(1, 0);
        //         println!("Input ids {:?}", input_ids);
        //         println!("New ids {:?}", new_ids);
        //         self.input_ids = Tensor::f_cat(&[input_ids.copy(), new_ids.copy()], 1).unwrap();
        //         // Save the current scores in logits form.
        //         self.beam_scores = Some(values);
        //         println!(
        //             "After beam search first step we have input_ids {:?}",
        //             self.input_ids
        //         );
        //     } else {
        //         // Now the tricky part.
        //         let size = logits.size();
        //         let last_logits = logits
        //             .i((0..size[0], size[1] - 1..size[1]))
        //             .to_device(self.input_ids.device());
        //         // Actually cast to logits so we can save the scores
        //         let last_logits = last_logits.f_log_softmax(-1, kind::Kind::Float).unwrap();
        //         panic!("We haven't handled that part !");
        //     }
        // }
    }
}
#[post("/generate")]
async fn generate(
    payload: web::Json<Generation>,
    state: web::Data<AppState>,
) -> actix_web::Result<HttpResponse> {
    let state = state.into_inner();
    let encoded = state
        .tokenizer
        .encode(payload.inputs.clone(), false)
        .unwrap();
    let mut ids: Vec<_> = encoded.get_ids().iter().map(|&i| i as i64).collect();
    // TODO The model is not able to handle empty input ids
    if ids.len() == 0 {
        ids.push(0);
    }
    let start = std::time::Instant::now();
    let (sx, rx) = bounded::<(Tensor, Past)>(2);
    let kind = (kind::Kind::Int64, Device::Cuda(0));
    let mut input_ids = Tensor::of_slice(ids.as_slice())
        .to_kind(kind.0)
        .to_device(kind.1)
        .view((1, -1));
    let max_new_tokens = payload.parameters.max_new_tokens;
    // TODO This config does not really matter
    // as we're not using past right now
    let config = Config::new350m();
    let start = Instant::now();

    for i in 0..max_new_tokens {
        // let start_loop = Instant::now();
        let ack = (input_ids.size()[0], sx.clone());
        let past_key_values = empty_past(&config);
        if i == 0 {
            state
                .in_channel
                .try_send((input_ids.copy(), past_key_values, ack))
                .map_err(|_| {
                    // println!("Queue was full {:?}", state.in_channel.len());
                    GenerationError::QueueFull
                })?;
        } else {
            state
                .prio_channel
                .send((input_ids.copy(), past_key_values, ack))
                .expect("This send should always work");
        }
        let (logits, _r_past_key_values) = rx.recv().unwrap();
        input_ids = add_next_id(&input_ids, &payload.parameters, &logits);
    }
    let n = max_new_tokens;
    let full_ids = input_ids
        .i((0,))
        .iter::<i64>()
        .unwrap()
        .map(|i| i as u32)
        .collect();
    let string = state.tokenizer.decode(full_ids, false).unwrap();
    Ok(HttpResponse::Ok()
        .content_type(ContentType::json())
        .json(json!([{ "generated_text": string }])))
}

#[derive(Clone)]
struct AppState {
    in_channel: Sender<Msg>,
    prio_channel: Sender<Msg>,
    tokenizer: Arc<Tokenizer>,
}

fn convert(view: TensorView, device: Device) -> Tensor {
    let kind = match view.get_dtype() {
        Dtype::F16 => kind::Kind::Half,
        Dtype::BF16 => kind::Kind::BFloat16,
        _ => {
            todo!("Need to implement that");
        }
    };
    let t = Tensor::of_data_size(
        &view.get_data(),
        &view
            .get_shape()
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>(),
        kind,
    )
    .to_device(device);
    t
}

fn get_slopes_power_of_2(n: usize) -> Vec<f64> {
    let start = 2.0f64.powf(-(2.0f64.powf(-((n as f64).log2() - 3.0f64))));
    let ratio = start;
    return (0..n).map(|i| start * ratio.powi(i as i32)).collect();
}

fn next_pow2(mut x: usize) -> usize {
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return (x + 1) >> 1;
}

fn get_slopes(n: usize) -> Vec<f64> {
    let is_power_of_2 = n == 0 || n & (n - 1) == 0;
    // println!("Slopes {:?}", n);
    // println!("Is power {:?}", is_power_of_2);
    // println!("Is power {:?}", n & (n - 1));
    if is_power_of_2 {
        return get_slopes_power_of_2(n);
    } else {
        let closest_power_of_2 = next_pow2(n);
        // println!("Closest power of 2 {:?}", closest_power_of_2);
        return get_slopes_power_of_2(closest_power_of_2)
            .into_iter()
            .chain(
                get_slopes(closest_power_of_2 << 1)
                    .into_iter()
                    .step_by(2)
                    .take(n - closest_power_of_2),
            )
            .collect();
    }
}

fn build_alibi_tensor(
    attention_mask: &Tensor,
    n_head: i64,
    kind: kind::Kind,
    device: Device,
) -> Tensor {
    let slopes = get_slopes(n_head as usize);
    //println!("Slopes {:?}", slopes);
    let slopes = Tensor::of_slice(&slopes).to_kind(kind).to_device(device);
    // debug("slopes", &slopes);
    let A = attention_mask.f_cumsum(-1, kind).unwrap().unsqueeze(1) - 1;
    // debug("A", &A);
    let B = attention_mask.unsqueeze(1);
    // debug("B", &B);
    let arange_tensor = A * B;
    // debug("A range tensor", &arange_tensor);
    let slopes = slopes.unsqueeze(-1);
    // debug("Slopes", &slopes);
    let mut alibi = slopes * arange_tensor;
    // debug("alibi1 ", &alibi);
    alibi = alibi * attention_mask.unsqueeze(1);
    // debug("alibi2 ", &alibi);

    let size = attention_mask.size();
    let batch_size = size[0];
    let seq_length = size[1];
    alibi = alibi.reshape(&[batch_size * n_head, 1, seq_length]);
    // debug("alibi", &alibi);
    return alibi;
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    hidden_size: i64,
}

impl LayerNorm {
    fn new(hidden_size: i64, name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let weight_name = format!("{name}.weight");
        let weight = convert(
            model
                .tensor(&weight_name)
                .expect(&format!("Failed to load {weight_name} with name {name}")),
            device,
        );
        let bias_name = format!("{name}.bias");
        let bias = convert(
            model
                .tensor(&bias_name)
                .expect(&format!("Failed to load {bias_name}")),
            device,
        );
        Self {
            hidden_size,
            weight,
            bias,
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        // println!(
        //     "Layer norm {:?} - {:?} - {:?}",
        //     self.weight.size(),
        //     self.bias.size(),
        //     xs.size()
        // );
        xs.f_layer_norm(
            &[self.hidden_size],
            Some(&self.weight),
            Some(&self.bias),
            EPS,
            true,
        )
        .unwrap()
    }
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn new(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let tname = format!("{name}.weight");
        let weight = convert(
            model
                .tensor(&tname)
                .expect(&format!("Could not find {tname}")),
            device,
        );

        let bias_name = format!("{name}.bias");
        let bias = convert(
            model
                .tensor(&bias_name)
                .expect(&format!("Could not find {bias_name}")),
            device,
        );

        Self { weight, bias }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.f_linear(&self.weight, Some(&self.bias)).unwrap()
    }
}

fn attention_mask_func(
    attention_scores: &mut Tensor,
    attention_mask: &Tensor,
    causal_mask: &Tensor,
) -> (Tensor, Tensor) {
    // TODO
    debug("in attention_scores", attention_scores);
    debug("in attention_mask", attention_mask);
    debug("in causal_mask", causal_mask);
    let attention_mask_bool = attention_mask
        .to_kind(kind::Kind::Bool)
        .f_logical_not()
        .unwrap();
    debug("not attention_mask", &attention_mask_bool);

    let query_length = attention_scores.size()[2];
    let key_length = attention_scores.size()[3];
    let n_heads = attention_scores.size()[1];

    let a_attention_mask_bool = attention_mask_bool.unsqueeze(1).unsqueeze(-1);
    let size = a_attention_mask_bool.size();
    let a_attention_mask_bool = a_attention_mask_bool.i((
        0..size[0],
        0..size[1],
        key_length - query_length..key_length,
    ));
    let b_causal_mask = causal_mask.f_logical_not().unwrap();

    let size = b_causal_mask.size();
    let b_causal_mask = b_causal_mask.i((
        0..size[0],
        0..size[1],
        key_length - query_length..key_length,
    ));
    debug("A attention_mask_bool", &a_attention_mask_bool);
    debug("b causal mask", &b_causal_mask);
    let mut padded_causal_mask = a_attention_mask_bool.f_logical_or(&b_causal_mask).unwrap();
    debug("padded causal_mask", &padded_causal_mask);
    let size = attention_mask_bool.size();
    let c_attention_mask_bool = &attention_mask_bool
        .i((0..size[0], 0..key_length))
        .unsqueeze(1)
        .unsqueeze(1);
    padded_causal_mask = padded_causal_mask
        .f_logical_or(c_attention_mask_bool)
        .unwrap();
    // TODO
    // padded_causal_mask = attention_mask_bool.logical_or(
    //     attention_mask_bool[:, None, key_length - query_length : key_length, None],
    //     ~causal_mask[:, :, key_length - query_length : key_length, :key_length].bool(),
    // )
    // padded_causal_mask = torch.logical_or(padded_causal_mask, attention_mask_bool[:, None, None, :key_length])
    // (
    //     attention_scores.masked_fill_(padded_causal_mask.expand(-1, n_heads, -1, -1), -10000.0),
    //     padded_causal_mask,
    // )
    let masked_fill = &padded_causal_mask
        .f_expand(&[-1, n_heads, -1, -1], true)
        .unwrap();
    debug("Masked fill", &masked_fill);
    let masked_output = attention_scores
        .f_masked_fill_(masked_fill, -10000.0)
        .unwrap();
    debug("Masked output", &masked_output);
    (masked_output, padded_causal_mask)
}

struct BloomScaledSoftmax {
    scale: i64,
    kind: kind::Kind,
}

impl BloomScaledSoftmax {
    fn new(scale: i64, kind: kind::Kind) -> Self {
        Self { scale, kind }
    }

    fn forward(&self, input: &Tensor, attention_mask: &Tensor, max_positions: i64) -> Tensor {
        debug("input", input);
        let mut scaled_input = self.scale * input;
        debug("scaled input", &scaled_input);
        let seq_ids = Tensor::f_arange(max_positions, (kind::Kind::Int64, input.device())).unwrap();

        let a = seq_ids.unsqueeze(0);
        let b = seq_ids.unsqueeze(-1);
        let causal_mask = a.f_le_tensor(&b).unwrap().to_kind(kind::Kind::Bool).view((
            1,
            1,
            max_positions,
            max_positions,
        ));

        debug("Causal mask", &causal_mask);

        // TODO Padded causal mask
        let (mask_output, padded_causal_mask) =
            attention_mask_func(&mut scaled_input, &attention_mask, &causal_mask);
        debug("mask output", &mask_output);

        // TODO dtype float16 ?
        let probs = mask_output.f_softmax(-1, kind::Kind::Float).unwrap()
            * padded_causal_mask.f_logical_not().unwrap();
        debug("Probs", &probs);

        let out_probs = probs.to_kind(self.kind);
        debug("Out Probs", &out_probs);

        out_probs
    }
}

struct BloomAttention {
    dense: Linear,
    query_key_value: Linear,
    num_attention_heads: i64,
    scaled_softmax: BloomScaledSoftmax,
    head_dim: i64,
    layer_number: usize,
    real_layer_number: usize,
    norm_factor: f64,
    n_head: i64,
    hidden_size: i64,
    pretraining_tp: i64,
    slow_but_exact: bool,
}

impl BloomAttention {
    fn new(
        config: &Config,
        name: &str,
        model: &SafeTensors<'_>,
        layer_number: usize,
        device: Device,
    ) -> Self {
        let dense = Linear::new(&format!("{name}.dense"), model, device);
        let query_key_value = Linear::new(&format!("{name}.query_key_value"), model, device);
        let head_dim = (config.hidden_size / config.n_head) as i64;
        let num_attention_heads = config.n_head as i64;
        let real_layer_number = layer_number;
        let layer_number = std::cmp::max(1, layer_number);
        let norm_factor = (head_dim as f64).sqrt() * layer_number as f64;
        let scaled_softmax = BloomScaledSoftmax::new(layer_number as i64, config.kind);
        Self {
            dense,
            query_key_value,
            num_attention_heads,
            head_dim,
            layer_number,
            real_layer_number,
            norm_factor,
            scaled_softmax,
            n_head: config.n_head as i64,
            hidden_size: config.hidden_size as i64,
            slow_but_exact: config.slow_but_exact,
            pretraining_tp: config.pretraining_tp as i64,
        }
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        residual: &Tensor,
        attention_mask: &Tensor,
        alibi: &Tensor,
        _layer_past: &mut PastLayer,
    ) -> Tensor {
        let layer_number = self.layer_number;
        let mut mixed_x_layer = self.query_key_value.forward(hidden_states);
        debug(&format!("Mixed_x_layer {layer_number}"), &mixed_x_layer);
        let new_tensor_shape = [
            mixed_x_layer.size()[0],
            mixed_x_layer.size()[1],
            self.num_attention_heads,
            3 * self.head_dim,
        ];
        let layer_number = self.layer_number;
        debug(&format!("Mixed x layer {layer_number}"), &mixed_x_layer);
        mixed_x_layer = mixed_x_layer.f_view(new_tensor_shape).unwrap();
        // Split.
        let tensor_list = mixed_x_layer.split(self.head_dim, 3);
        let (query, key, value) = match &tensor_list[..] {
            [query, key, value] => (query, key, value),
            _ => unreachable!(),
        };
        // let (past_key, past_value) = layer_past;

        // let mut key_layer = Tensor::f_cat(&[past_key.as_ref(), key], 1).unwrap();
        // let mut value_layer = Tensor::f_cat(&[past_value.as_ref(), value], 1).unwrap();
        let mut key_layer = key.copy();

        // Update past for next loops
        // *layer_past = (key_layer.copy(), value_layer.copy());

        // [batch_size, head_dim, q_length, k_length, num_heads]
        let B = query.size()[0];
        let H = query.size()[2];
        let Q = query.size()[1];
        let K = key_layer.size()[1];
        // let NH = value_layer.size()[2];
        let NH = self.hidden_size / self.n_head;
        let output_size = (B, H, Q, K);

        let query_layer = query.transpose(1, 0).reshape(&[Q, B * H, -1]);
        key_layer = key_layer.transpose(1, 0).reshape(&[K, B * H, -1]);

        // let sliced_alibi = alibi[: output_size[0] * output_size[1], :, : output_size[3]]
        let size = alibi.size();
        let sliced_alibi = alibi.i((0..B * H, 0..size[1], 0..K));
        let beta = 1.0 / (self.layer_number as f64);
        let alpha = 1.0 / self.norm_factor;

        let a = sliced_alibi;
        let b = query_layer.transpose(1, 0);
        let c = key_layer.transpose(1, 0).transpose(1, 2);
        debug("Sliced alibi", &a);
        debug("query layer", &b);
        debug("key layer", &c);

        save_layer_to_disk(
            &a,
            &format!("rust_baddbmm_sliced_alibi_{}.npy", self.real_layer_number,),
        );
        save_layer_to_disk(
            &b,
            &format!("rust_baddbmm_query_layer_{}.npy", self.real_layer_number,),
        );
        save_layer_to_disk(
            &c,
            &format!("rust_baddbmm_key_layer_{}.npy", self.real_layer_number,),
        );
        save_layer_to_disk(
            &Tensor::of_slice(&[beta]),
            &format!("rust_baddbmm_beta_{}.npy", self.real_layer_number,),
        );
        save_layer_to_disk(
            &Tensor::of_slice(&[alpha]),
            &format!("rust_baddbmm_alpha_{}.npy", self.real_layer_number,),
        );
        let matmul_result = a.f_baddbmm(&b, &c, beta, alpha).unwrap();

        save_layer_to_disk(
            &matmul_result,
            &format!("rust_baddbmm_matmul_result_{}.npy", self.real_layer_number,),
        );

        let attention_scores = matmul_result.f_view(output_size).unwrap();
        debug(
            &format!("Attention baddbmm {layer_number}"),
            &attention_scores,
        );
        let shape = attention_scores.size();
        let max_positions = std::cmp::max(shape[shape.len() - 1], shape[shape.len() - 2]);
        let attention_probs =
            self.scaled_softmax
                .forward(&attention_scores, attention_mask, max_positions);
        save_layer_to_disk(
            &attention_probs,
            &format!(
                "rust_softmax_attention_probs_{}.npy",
                self.real_layer_number,
            ),
        );
        let value_layer = value.transpose(1, 0).reshape(&[K, B * H, -1]);
        let attention_probs_reshaped = attention_probs.f_view((B * H, Q, -1)).unwrap();
        let bmm = Tensor::bmm(&attention_probs_reshaped, &value_layer.transpose(0, 1));
        save_layer_to_disk(
            &bmm,
            &format!("rust_bmm_context_layer_{}.npy", self.real_layer_number,),
        );
        debug("Bmm", &bmm);
        let context_layer_r = bmm.f_view((B, H, Q, -1)).unwrap();
        let context_layer_p = context_layer_r.permute(&[2, 0, 1, 3]).contiguous();
        let context = context_layer_p.f_view((Q, B, H * NH)).unwrap();
        save_layer_to_disk(
            &context,
            &format!("rust_bmm_context_layer2_{}.npy", self.real_layer_number,),
        );
        let output_tensor = if self.slow_but_exact {
            let slices = context.size().last().unwrap() / self.pretraining_tp;
            let mut output_tensor = Tensor::zeros_like(&context);
            for i in 0..self.pretraining_tp {
                let context_tp = context.i((.., .., i * slices..(i + 1) * slices));
                let dense_tp = self.dense.weight.i((.., i * slices..(i + 1) * slices));
                output_tensor += context_tp.linear::<Tensor>(&dense_tp, None);
            }
            output_tensor
        } else {
            self.dense.forward(&context)
        };
        save_layer_to_disk(
            &output_tensor,
            &format!("rust_bmm_dense_{}.npy", self.real_layer_number,),
        );
        save_layer_to_disk(
            &residual,
            &format!("rust_bmm_residual_{}.npy", self.real_layer_number,),
        );
        let mut output = output_tensor.transpose(1, 0);
        output += residual;
        save_layer_to_disk(
            &output,
            &format!("rust_bmm_dropout_{}.npy", self.real_layer_number,),
        );
        output
    }
}

struct BloomMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    real_layer_number: usize,
    slow_but_exact: bool,
    pretraining_tp: i64,
}

fn bloom_gelu(x: &Tensor) -> Tensor {
    let y: Tensor = 0.79788456 * x * (1.0 + 0.044715 * x * x);
    return x * 0.5 * (1.0 + y.tanh());
}

impl BloomMlp {
    fn new(
        config: &Config,
        name: &str,
        model: &SafeTensors<'_>,
        device: Device,
        real_layer_number: usize,
    ) -> Self {
        let dense_h_to_4h = Linear::new(&format!("{name}.dense_h_to_4h"), model, device);
        let dense_4h_to_h = Linear::new(&format!("{name}.dense_4h_to_h"), model, device);
        Self {
            dense_h_to_4h,
            dense_4h_to_h,
            real_layer_number,
            slow_but_exact: config.slow_but_exact,
            pretraining_tp: config.pretraining_tp as i64,
        }
    }

    fn forward(&self, hidden_states: &Tensor, residual: &Tensor) -> Tensor {
        save_layer_to_disk(
            &hidden_states,
            &format!("rust_mlp_init_{}.npy", self.real_layer_number,),
        );
        debug("hidden_states", hidden_states);
        debug("INCOMING residual", residual);
        let hidden_states = self.dense_h_to_4h.forward(hidden_states);
        debug("hidden_states h to 4h", &hidden_states);
        let hidden_states = bloom_gelu(&hidden_states);
        save_layer_to_disk(
            &hidden_states,
            &format!("rust_mlp_gelu_{}.npy", self.real_layer_number,),
        );
        debug("hidden_states gelu", &hidden_states);
        let hidden_states = if self.slow_but_exact {
            let mut intermediate_output = Tensor::zeros_like(&residual);
            let slices = self.dense_4h_to_h.weight.size().last().unwrap() / self.pretraining_tp;
            for i in 0..self.pretraining_tp {
                let i = i as i64;
                let tp = hidden_states.i((.., .., i * slices..(i + 1) * slices));
                let dense_size = self.dense_4h_to_h.weight.size();
                let dense_tp = self
                    .dense_4h_to_h
                    .weight
                    .i((.., i * slices..(i + 1) * slices));
                intermediate_output += tp.linear::<Tensor>(&dense_tp, None);
            }
            intermediate_output
        } else {
            self.dense_4h_to_h.forward(&hidden_states)
        };
        debug("hidden_states 4h to h", &hidden_states);
        let hidden_states = hidden_states + residual;
        debug("hidden_states residual", &hidden_states);
        save_layer_to_disk(
            &hidden_states,
            &format!("rust_mlp_output_{}.npy", self.real_layer_number,),
        );
        hidden_states
    }
}

struct BloomBlock {
    input_layernorm: LayerNorm,
    self_attention: BloomAttention,
    post_attention_layernorm: LayerNorm,
    mlp: BloomMlp,
    layer_number: usize,
}

impl BloomBlock {
    fn new(
        config: &Config,
        prefix: &str,
        model: &SafeTensors<'_>,
        layer_number: usize,
        device: Device,
    ) -> Self {
        // attention
        let input_layernorm = LayerNorm::new(
            config.hidden_size,
            &format!("{prefix}.input_layernorm"),
            model,
            device,
        );
        let post_attention_layernorm = LayerNorm::new(
            config.hidden_size,
            &format!("{prefix}.post_attention_layernorm"),
            model,
            device,
        );
        let self_attention = BloomAttention::new(
            config,
            &format!("{prefix}.self_attention"),
            model,
            layer_number,
            device,
        );
        let mlp = BloomMlp::new(
            config,
            &format!("{prefix}.mlp"),
            model,
            device,
            layer_number,
        );
        Self {
            input_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
            layer_number,
        }
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        alibi: &Tensor,
        layer_past: &mut PastLayer,
    ) -> Tensor {
        debug(
            &format!(
                "============================Layer {}===========================",
                self.layer_number
            ),
            hidden_states,
        );
        let layernorm_output = self.input_layernorm.forward(hidden_states);
        let layer_number = self.layer_number;
        debug(
            &format!("Block layernorm {layer_number}"),
            &layernorm_output,
        );
        // Depends on apply_residual_connection_post_layernorm
        let residual = hidden_states;
        debug(&format!("Residual {layer_number}"), residual);
        let attention_output = self.self_attention.forward(
            &layernorm_output,
            residual,
            attention_mask,
            alibi,
            layer_past,
        );
        debug(
            &format!("Attention output {layer_number}"),
            &attention_output,
        );
        let layernorm_output = self.post_attention_layernorm.forward(&attention_output);
        debug(
            &format!("Post attention layer norm {layer_number}"),
            &layernorm_output,
        );
        // Depends on apply_residual_connection_post_layernorm
        let residual = attention_output;
        let output = self.mlp.forward(&layernorm_output, &residual);
        debug(&format!("MLP output {layer_number}"), &output);
        output
    }
}

struct Embedding {
    weight: Tensor,
}

impl Embedding {
    fn new(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let weight = convert(model.tensor(&format!("{name}.weight")).unwrap(), device);
        Self { weight }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        debug("Embedding weights", &self.weight);
        let result = Tensor::embedding(&self.weight, xs, PADDING_IDX, false, false);
        debug("Embedding", &result);
        result
    }
}

fn save_layer_to_disk(tensor: &Tensor, filename: &str) {
    tensor
        .to_device(Device::Cpu)
        .to_kind(kind::Kind::Float)
        .write_npy(filename)
        .unwrap();
}

fn debug_force(prefix: &str, x: &Tensor) {
    let size = x.size();
    let B = size[0];
    let S = if size.len() > 1 { size[1] } else { B };
    println!(
        "{prefix} - {:?} - Values: {:?}",
        size,
        x.reshape(&[-1,])
            .iter::<f64>()
            .unwrap()
            .take(std::cmp::min(S as usize, 10))
            .collect::<Vec<_>>()
    );
    if B > 1 {
        println!(
            "                          {:?}",
            x.i(1)
                .reshape(&[-1,])
                .iter::<f64>()
                .unwrap()
                .take(30)
                .collect::<Vec<_>>()
        );
    }
}

fn debug(prefix: &str, x: &Tensor) {
    // debug_force(prefix, x);
}

struct BloomModel {
    word_embeddings: Embedding,
    word_embeddings_layernorm: LayerNorm,
    h: Vec<BloomBlock>,
    ln_f: LayerNorm,
}

impl BloomModel {
    fn new(config: &Config, model: &SafeTensors<'_>, device: Device) -> Self {
        let word_embeddings = Embedding::new(&format!("word_embeddings"), model, device);
        let word_embeddings_layernorm = LayerNorm::new(
            config.hidden_size,
            &format!("word_embeddings_layernorm"),
            model,
            device,
        );
        let ln_f = LayerNorm::new(config.hidden_size, &format!("ln_f"), model, device);
        let h = (0..config.n_layer)
            .map(|i| BloomBlock::new(config, &format!("h.{i}"), model, i as usize, device))
            .collect();
        Self {
            word_embeddings,
            word_embeddings_layernorm,
            h,
            ln_f,
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        alibi: &Tensor,
        past_key_values: &mut Past,
    ) -> Tensor {
        let inputs_embeds = self.word_embeddings.forward(input_ids);
        let mut hidden_states = self.word_embeddings_layernorm.forward(&inputs_embeds);

        debug(
            "First layer norm weights",
            &self.word_embeddings_layernorm.weight,
        );

        debug("First layer norm", &hidden_states);

        debug("Alibi", &alibi);

        for (i, (block, layer_past)) in self.h.iter().zip(past_key_values.iter_mut()).enumerate() {
            hidden_states = block.forward(&hidden_states, &attention_mask, &alibi, layer_past);
            debug(&format!("Block {i}"), &hidden_states);
        }
        hidden_states = self.ln_f.forward(&hidden_states);
        debug(&format!("ln_f"), &hidden_states);
        hidden_states
    }
}

struct InvertedEmbedding {
    weight: Tensor,
}

impl InvertedEmbedding {
    fn new(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let weight = convert(model.tensor(&format!("{name}.weight")).unwrap(), device);
        Self { weight }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        debug("InvertedEmbedding weights", &self.weight);
        debug("Incoming tensor ", xs);
        let logits = xs.f_linear::<Tensor>(&self.weight, None).unwrap();
        let max = logits.max_dim(-1, false);
        debug("lm logits (max)", &max.0);
        debug("lm logits (max indices)", &max.1);
        debug("lm logits", &logits);
        logits
    }
}

struct BloomForCausalLM {
    transformer: BloomModel,
    lm_head: InvertedEmbedding,
}

impl BloomForCausalLM {
    fn new(config: &Config, model: &SafeTensors<'_>, device: Device) -> Self {
        let transformer = BloomModel::new(config, model, device);
        let lm_head = InvertedEmbedding::new("word_embeddings", model, device);
        Self {
            transformer,
            lm_head,
        }
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        alibi: &Tensor,
        past_key_values: &mut Past,
    ) -> Tensor {
        let hidden_states =
            self.transformer
                .forward(input_ids, attention_mask, alibi, past_key_values);
        let lm_logits = self.lm_head.forward(&hidden_states);
        lm_logits
    }
}
fn padding_with_ack(
    config: &Config,
    mut items: Vec<(Tensor, Past, Ack)>,
) -> ((Tensor, Tensor, Tensor, Past), Vec<Ack>) {
    let mut tensors = vec![];
    let mut acks = vec![];
    for item in items {
        tensors.push((item.0, item.1));
        acks.push(item.2);
    }
    (padding(&config, tensors), acks)
}

fn padding(config: &Config, mut items: Vec<(Tensor, Past)>) -> (Tensor, Tensor, Tensor, Past) {
    // TODO
    let max_length = items.iter().map(|(ids, _)| ids.size()[1]).max().unwrap();
    let batch_size: i64 = items.iter().map(|(ids, _)| ids.size()[0]).sum::<i64>();
    let kind = (kind::Kind::Int64, Device::Cuda(0));
    let device = items[0].0.device();
    let kind2 = (kind::Kind::Int64, device);
    let mut all_input_ids = Tensor::zeros(&[batch_size, max_length], kind2) + PADDING_IDX;
    let mut attention_mask = Tensor::zeros(&[batch_size, max_length], kind2);

    let mut total_ids = 0;

    let mut current_batch = 0;
    for (input_ids, past_key_values) in items {
        let seq_length = input_ids.size()[1];
        let mini_batch_size = input_ids.size()[0];
        total_ids += mini_batch_size as usize * seq_length as usize;
        // all_input_ids[i:i+mini_batch_size, max_length - seq_length:seq_length] =
        // input_ids

        let mut batch_indices = vec![];
        let mut id_indices = vec![];
        for i in 0..mini_batch_size {
            for j in 0..seq_length {
                let batch_index = i + current_batch;
                let id_index = j + max_length - seq_length;
                batch_indices.push(batch_index);
                id_indices.push(id_index);
            }
        }
        let batch_index = Tensor::of_slice(batch_indices.as_slice())
            .to_kind(kind.0)
            .to_device(kind.1);
        let id_index = Tensor::of_slice(id_indices.as_slice())
            .to_kind(kind.0)
            .to_device(kind.1);

        // input_put requires 1-d tensor ?
        let id_row = input_ids.view((-1,));
        all_input_ids = all_input_ids
            .f_index_put_(&[Some(&batch_index), Some(&id_index)], &id_row, false)
            .unwrap();

        let attn = input_ids.fill(1).view((-1,));
        attention_mask = attention_mask
            .f_index_put_(&[Some(&batch_index), Some(&id_index)], &attn, false)
            .unwrap();
        current_batch += mini_batch_size;
    }

    let p = config.n_head;
    let q = config.hidden_size / config.n_head;
    let past_key = Tensor::zeros(&[batch_size, 0, p, q], kind);
    let past_value = Tensor::zeros(&[batch_size, 0, p, q], kind);
    let past_key_values: Vec<_> = (0..config.n_layer)
        .map(|_| (past_key.copy(), past_value.copy()))
        .collect();

    let alibi = build_alibi_tensor(&attention_mask, config.n_head, config.kind, device);

    let total = std::cmp::max(1, batch_size as usize * max_length as usize);
    // if batch_size > 4 {
    //     println!(
    //         "Running on batch of size {:?} - Fillrate {:?}%",
    //         batch_size,
    //         (total_ids * 100) / total
    //     );
    // }
    (all_input_ids, attention_mask, alibi, past_key_values)
}

fn thread1(
    rx: RChan1,
    prio_rx: RChan1,
    s2: SChan,
    thread_number: usize,
    config: Config,
    layout_config: LayoutConfig,
) {
    println!("Starting thread {thread_number}");
    let start = std::time::Instant::now();
    let device = Device::Cuda(thread_number);

    let file = std::fs::File::open(&layout_config.embeddings_filename).unwrap();
    // SAFETY: This is actually unsafe.
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let embedding_model = SafeTensors::deserialize(&mmap).unwrap();

    // println!("Embedding {:?}", embedding_model.names());
    // println!("Layer {:?}", model.names());

    let word_embeddings = Embedding::new(&format!("word_embeddings"), &embedding_model, device);
    let word_embeddings_layernorm = LayerNorm::new(
        config.hidden_size,
        &format!("word_embeddings_layernorm"),
        &embedding_model,
        device,
    );

    let layers: Vec<BloomBlock> = (0..layout_config.layers_first_thread)
        .map(|i| {
            let file = std::fs::File::open(
                layout_config
                    .layer_template_filename
                    .replace("{}", &i.to_string()),
            )
            .unwrap();
            // SAFETY: This is actually unsafe.
            let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
            let model = SafeTensors::deserialize(&mmap).unwrap();
            BloomBlock::new(
                &config,
                &if std::env::var("BLOOM").unwrap_or("".to_string()) == "bloom-dgx" {
                    "h.0".to_string()
                } else {
                    format!("h.{i}")
                },
                &model,
                i,
                device,
            )
        })
        .collect();
    println!(
        "{:?} : Loaded thread {thread_number} in {:?}",
        std::time::Instant::now(),
        start.elapsed()
    );

    let mut last_loop = Instant::now();
    loop {
        let mut sel = Select::new();
        let oper1 = sel.recv(&rx);
        let oper2 = sel.recv(&prio_rx);
        let oper = sel.select();
        let mut all_items = match oper.index() {
            i if i == oper1 => {
                vec![oper.recv(&rx).unwrap()]
            }
            i if i == oper2 => {
                vec![oper.recv(&prio_rx).unwrap()]
            }
            _ => unreachable!(),
        };

        let max_batch_size = 4;

        let now = Instant::now();
        let deadline = std::cmp::max(
            last_loop + Duration::from_millis(10),
            now + Duration::from_millis(1),
        );
        // let deadline = now + Duration::from_millis(1);
        while let Ok(item) = prio_rx.recv_deadline(deadline) {
            all_items.push(item);
        }

        if all_items.len() < max_batch_size {
            while let Ok(item) = rx.recv_deadline(deadline) {
                all_items.push(item);
            }
        }

        while let Ok(oper) = sel.select_deadline(deadline) {
            match oper.index() {
                i if i == oper1 => {
                    all_items.push(oper.recv(&rx).unwrap());
                }
                i if i == oper2 => {
                    all_items.push(oper.recv(&prio_rx).unwrap());
                }
                _ => unreachable!(),
            }
        }
        // let start = Instant::now();
        let ((input_ids, attention_mask, alibi, mut past_key_values), acks) =
            padding_with_ack(&config, all_items);
        // println!("Padding took {:?}", start.elapsed());
        let inputs_embeds = word_embeddings.forward(&input_ids);
        let mut hidden_states = word_embeddings_layernorm.forward(&inputs_embeds);

        for (layer, layer_past) in layers.iter().zip(past_key_values.iter_mut()) {
            hidden_states = layer.forward(&hidden_states, &attention_mask, &alibi, layer_past);
        }
        // println!("Thread1 took {:?}", start.elapsed());
        s2.send((
            (hidden_states, attention_mask, alibi, past_key_values),
            acks,
        ))
        .unwrap();
        last_loop = Instant::now();
    }
}

fn thread2(rx: RChan, s: SChan, thread_number: usize, config: Config, layout_config: LayoutConfig) {
    println!("Starting thread {thread_number}");
    let start = std::time::Instant::now();
    let device = Device::Cuda(thread_number);

    let layers: Vec<BloomBlock> = (0..layout_config.layers_per_thread)
        .map(|i| {
            let layer_number = i
                + layout_config.layers_first_thread
                + layout_config.layers_per_thread * (thread_number - 1);
            println!("Loading layer {layer_number} on thread2 ({thread_number})");
            let file = std::fs::File::open(
                layout_config
                    .layer_template_filename
                    .replace("{}", &layer_number.to_string()),
            )
            .unwrap();
            // SAFETY: This is actually unsafe.
            let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
            let model = SafeTensors::deserialize(&mmap).unwrap();
            BloomBlock::new(
                &config,
                &if std::env::var("BLOOM").unwrap_or("".to_string()) == "bloom-dgx" {
                    "h.0".to_string()
                } else {
                    format!("h.{layer_number}")
                },
                &model,
                layer_number,
                device,
            )
        })
        .collect();
    println!(
        "{:?} : Loaded thread {thread_number} in {:?}",
        std::time::Instant::now(),
        start.elapsed()
    );
    loop {
        // Receive 1 item
        // println!("start loop  thread {thread_number}");
        // let start = Instant::now();
        let mut all_items = vec![rx.recv().unwrap()];
        // if start.elapsed() > Duration::from_millis(200) {
        //     println!(
        //         "Got stuck on RECEIVE thread {thread_number} for {:?}",
        //         start.elapsed()
        //     );
        // }

        let start = Instant::now();
        let deadline = start + Duration::from_millis(1);
        while let Ok(item) = rx.recv_deadline(deadline) {
            all_items.push(item);
        }
        all_items.sort_by(|a, b| {
            let a_size = a.0 .0.size();
            let b_size = b.0 .0.size();
            let a = a_size[0] * a_size[1];
            let b = b_size[0] * b_size[1];
            a.cmp(&b)
        });

        for item in all_items {
            let ((mut hidden_states, mut attention_mask, mut alibi, mut past_key_values), rq) =
                item;
            hidden_states = hidden_states.to_device(device);
            attention_mask = attention_mask.to_device(device);
            alibi = alibi.to_device(device);

            for (layer, layer_past) in layers.iter().zip(past_key_values.iter_mut()) {
                debug(&format!("past_key thread2"), &layer_past.0);
                debug(&format!("past_values thread2"), &layer_past.1);
                hidden_states = layer.forward(&hidden_states, &attention_mask, &alibi, layer_past);
            }
            // println!(
            //     "Thread2 {thread_number} took {:?} on batch size {:?}",
            //     start.elapsed(),
            //     hidden_states.size()[0]
            // );
            let start = Instant::now();
            s.send(((hidden_states, attention_mask, alibi, past_key_values), rq))
                .unwrap();
            if start.elapsed() > Duration::from_millis(200) {
                println!(
                    "Got stuck on SEND thread {thread_number} for {:?}",
                    start.elapsed()
                );
            }
        }
    }
}

fn thread3(rx: RChan, thread_number: usize, config: Config, layout_config: LayoutConfig) {
    println!("Starting thread {thread_number}");
    let start = std::time::Instant::now();
    let device = Device::Cuda(thread_number);

    let file = std::fs::File::open(layout_config.embeddings_filename).unwrap();
    // SAFETY: This is actually unsafe.
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let embedding_model = SafeTensors::deserialize(&mmap).unwrap();

    let file = std::fs::File::open(layout_config.final_filename).unwrap();
    // SAFETY: This is actually unsafe.
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let final_model = SafeTensors::deserialize(&mmap).unwrap();

    let ln_f = LayerNorm::new(config.hidden_size, &format!("ln_f"), &final_model, device);
    let lm_head = InvertedEmbedding::new(&format!("word_embeddings"), &embedding_model, device);

    let layers: Vec<BloomBlock> = (0..layout_config.layers_last_thread)
        .map(|i| {
            let layer_number = layout_config.layers_first_thread
                + layout_config.layers_per_thread * layout_config.n_threads
                + i;
            println!("Loading layer {layer_number} on thread3 ({thread_number})");
            let file = std::fs::File::open(
                layout_config
                    .layer_template_filename
                    .replace("{}", &layer_number.to_string()),
            )
            .unwrap();
            // SAFETY: This is actually unsafe.
            let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
            let model = SafeTensors::deserialize(&mmap).unwrap();
            BloomBlock::new(
                &config,
                &if std::env::var("BLOOM").unwrap_or("".to_string()) == "bloom-dgx" {
                    "h.0".to_string()
                } else {
                    format!("h.{i}")
                },
                &model,
                layer_number,
                device,
            )
        })
        .collect();

    println!(
        "{:?} : Loaded thread {thread_number} in {:?}",
        std::time::Instant::now(),
        start.elapsed()
    );

    loop {
        let ((mut hidden_states, mut attention_mask, mut alibi, mut past_key_values), rqs) = rx
            .recv()
            .expect("You probably want to handle this case, but I'm too lazy");

        hidden_states = hidden_states.to_device(device);
        attention_mask = attention_mask.to_device(device);
        alibi = alibi.to_device(device);
        for (layer, layer_past) in layers.iter().zip(past_key_values.iter_mut()) {
            debug(&format!("past_key thread3"), &layer_past.0);
            debug(&format!("past_values thread3"), &layer_past.1);
            hidden_states = layer.forward(&hidden_states, &attention_mask, &alibi, layer_past);
        }
        debug(&format!("last_hidden_states"), &hidden_states);
        hidden_states = ln_f.forward(&hidden_states);
        debug(&format!("After ln_f"), &hidden_states);
        let lm_logits = lm_head.forward(&hidden_states);

        let mut current_batch = 0;
        for (mini_batch_size, rq) in rqs {
            let simple_logits = lm_logits.i(current_batch..current_batch + mini_batch_size);
            let past = empty_past(&config);
            rq.send((simple_logits, past)).unwrap();
            current_batch += mini_batch_size;
        }
    }
}

#[actix_web::main] // or #[tokio::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    tch::maybe_init_cuda();

    let start = std::time::Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file("./tokenizer.json").unwrap());
    println!("Loaded tokenizer in {:?}", start.elapsed());
    println!("Starting threads {:?}", std::time::Instant::now());

    let (tx, rx) = bounded::<Msg>(1);
    let (prio_tx, prio_rx) = unbounded::<Msg>();

    let (config, layout_config) = if let Ok(env) = std::env::var("BLOOM") {
        match env.as_ref() {
            "bigscience-small-testing" => (Config::new_testing(), LayoutConfig::new_testing()),
            "bloom-350m" => (Config::new350m(), LayoutConfig::new350m()),
            "bloom" => (Config::new(), LayoutConfig::new()),
            "bloom-dgx" => (Config::new(), LayoutConfig::new_dgx()),
            other => panic!("Model {other} is not known"),
        }
    } else {
        (Config::new(), LayoutConfig::new())
    };

    let channels: Vec<_> = (0..layout_config.n_threads + 1)
        .map(|_| unbounded::<Msg2>())
        .collect();

    let s = channels[0].0.clone();
    let config_ = config.clone();
    let layout_config_ = layout_config.clone();
    tokio::task::spawn_blocking(move || {
        thread1(rx, prio_rx, s, 0, config_, layout_config_);
    });

    for i in 0..layout_config.n_threads {
        let (r, s) = (channels[i].1.clone(), channels[i + 1].0.clone());
        let config_ = config.clone();
        let layout_config_ = layout_config.clone();
        tokio::task::spawn_blocking(move || {
            thread2(r, s, i + 1, config_, layout_config_);
        });
    }
    let r = channels[layout_config.n_threads].1.clone();
    let config_ = config.clone();
    let layout_config_ = layout_config.clone();
    tokio::task::spawn_blocking(move || {
        thread3(r, layout_config.n_threads + 1, config_, layout_config_);
    });

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(web::Data::new(AppState {
                tokenizer: tokenizer.clone(),
                in_channel: tx.clone(),
                prio_channel: prio_tx.clone(),
            }))
            .service(generate)
    })
    .bind(("127.0.0.1", 8001))?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padding() {
        let config = Config::new350m();
        let device = Device::Cuda(0);
        let input_ids = Tensor::of_slice(&[3, 4, 5])
            .view((1, 3))
            .to_kind(kind::Kind::Int64)
            .to_device(device);
        let input_ids2 = Tensor::of_slice(&[8, 1, 3, 4, 5, 6])
            .view((1, 6))
            .to_kind(kind::Kind::Int64)
            .to_device(device);
        let past = vec![];
        let past2 = vec![];

        let items = vec![(input_ids, past), (input_ids2, past2)];

        let (all_input_ids, _, _, _) = padding(&config, items);

        assert_eq!(all_input_ids.size(), vec![2, 6]);
        assert_eq!(
            Vec::<i64>::from(all_input_ids),
            vec![3, 3, 3, 3, 4, 5, 8, 1, 3, 4, 5, 6]
        );
    }

    #[test]
    fn test_bloom_350m() {
        // Batched input (2, 4)
        // First input is not full
        let model = BLOOM_350M.lock().unwrap();
        let config = Config::new350m();
        let p = config.n_head;
        let q = config.hidden_size / config.n_head;
        let device = Device::Cuda(0);
        let kind = (config.kind, device);
        let past_key = Tensor::zeros(&[1, 0, p, q], kind);
        let past_value = Tensor::zeros(&[1, 0, p, q], kind);
        let mut past_key_values: Vec<_> = (0..config.n_layer)
            .map(|_| (past_key.copy(), past_value.copy()))
            .collect();
        let input_ids = Tensor::of_slice(&[2, 2, 34, 54, 132, 225, 532, 342])
            .view((2, 4))
            .to_device(Device::Cuda(0));
        let attention_mask = Tensor::of_slice(&[0, 0, 1, 1, 1, 1, 1, 1])
            .view((2, 4))
            .to_device(Device::Cuda(0));

        let alibi = build_alibi_tensor(&attention_mask, config.n_head, config.kind, device);

        let logits = model.forward(&input_ids, &attention_mask, &alibi, &mut past_key_values);

        assert_eq!(logits.size(), vec![2, 4, 250880]);
        let expected = Tensor::of_slice(&[
            640.5, 668.5, 671.0, 102.375, 670.5, 676.5, 680.5, 676.0, 671.0, 670.0,
        ])
        .to_kind(config.kind)
        .to_device(device);
        assert_all_close(&expected, &logits.i((0, 0, 0..10)));
        let ids = logits.argmax(-1, false);
        assert_eq!(ids.size(), vec![2, 4]);
        assert_eq!(
            Vec::<i64>::from(ids),
            // Original implem output in comment
            // Most likely linked to odd `baddbmm` kernel for alpha + beta fusion
            // Which leads to small drifts.
            vec![4141, 4141, 2, 17, 64530, 15, 15, 100] // vec![235149, 235149, 2, 17, 64530, 15, 15, 100]
        );
    }

    fn assert_all_close(left: &Tensor, right: &Tensor) {
        if !left.allclose(right, 1e-7, 1e-7, false) {
            left.print();
            right.print();
            panic!("{left:?} is not close to {right:?}");
        }
    }

    #[test]
    fn test_alibi() {
        let config = Config::new350m();
        let device = Device::Cuda(0);
        let attention_mask = Tensor::of_slice(&[0, 0, 1, 1, 1, 1, 1, 1])
            .view((2, 4))
            .to_device(device);

        assert_eq!(attention_mask.size(), vec![2, 4]);
        let alibi = build_alibi_tensor(&attention_mask, config.n_head, config.kind, device);
        assert_eq!(alibi.size(), vec![32, 1, 4]);
        assert_eq!(
            Vec::<f64>::from(alibi)
                .into_iter()
                .take(8)
                .collect::<Vec<_>>(),
            vec![-0.0, -0.0, 0.0, 0.70703125, -0.0, -0.0, 0.0, 0.5]
        );
    }

    #[test]
    fn test_alibi2() {
        // let config = Config::new();
        let device = Device::Cuda(0);
        let kind = kind::Kind::BFloat16;
        let n_head = 5;
        let attention_mask = Tensor::of_slice(&[1, 1, 1]).view((1, 3)).to_device(device);

        assert_eq!(attention_mask.size(), vec![1, 3]);
        let alibi = build_alibi_tensor(&attention_mask, n_head, kind, device);
        assert_eq!(alibi.size(), vec![5, 1, 3]);
        assert_eq!(
            Vec::<f64>::from(alibi)
                .into_iter()
                .take(15)
                .collect::<Vec<_>>(),
            vec![
                0.0, 0.25, 0.5, 0.0, 0.0625, 0.125, 0.0, 0.015625, 0.03125, 0.0, 0.00390625,
                0.0078125, 0.0, 0.5, 1.0
            ]
        );

        let attention_mask = Tensor::of_slice(&[1, 1, 1, 1])
            .view((1, 4))
            .to_device(device);

        assert_eq!(attention_mask.size(), vec![1, 4]);
        let alibi = build_alibi_tensor(&attention_mask, n_head, kind, device);
        assert_eq!(alibi.size(), vec![5, 1, 4]);
        assert_eq!(
            Vec::<f64>::from(alibi)
                .into_iter()
                .take(20)
                .collect::<Vec<_>>(),
            vec![
                0.0, 0.25, 0.5, 0.75, 0.0, 0.0625, 0.125, 0.1875, 0.0, 0.015625, 0.03125, 0.046875,
                0.0, 0.00390625, 0.0078125, 0.01171875, 0.0, 0.5, 1.0, 1.5
            ]
        );
    }

    fn test_generate(
        input: &[&str],
        config: &Config,
        tokenizer: &Tokenizer,
        model: &BloomForCausalLM,
        max_new_tokens: usize,
    ) -> Vec<String> {
        // Taken directly from https://github.com/huggingface/transformers/blob/main/tests/models/bloom/test_modeling_bloom.py#L379
        let mut all_items = vec![];
        for input_string in input {
            let encoded = tokenizer.encode(input_string.to_string(), false).unwrap();
            let ids: Vec<_> = encoded.get_ids().iter().map(|&i| i as i64).collect();
            let input_ids = Tensor::of_slice(ids.as_slice())
                .to_kind(kind::Kind::Int64)
                .to_device(Device::Cuda(0))
                .view((1, -1));
            let past = empty_past(&config);

            // Not necessary, but want to reuse the real code
            let item = (input_ids, empty_past(&config));
            all_items.push(item);
        }

        let (mut input_ids, mut attention_mask, mut alibi, mut past_key_values) =
            padding(&config, all_items);

        for _ in 0..max_new_tokens {
            debug("Input ids", &input_ids);
            let logits = model.forward(&input_ids, &attention_mask, &alibi, &mut past_key_values);
            let size = logits.size();
            let new_ids = logits
                .i((0..size[0], size[1] - 1..size[1]))
                .argmax(-1, false);
            let ones = new_ids.ones_like();
            input_ids = Tensor::cat(&[input_ids, new_ids], 1);
            attention_mask = Tensor::cat(&[attention_mask, ones], 1);
            alibi = build_alibi_tensor(
                &attention_mask,
                config.n_head,
                config.kind,
                attention_mask.device(),
            );
        }

        let mut all_strings = vec![];
        for i in 0..input.len() {
            let output_ids: Vec<_> = input_ids
                .i(i as i64)
                .reshape(&[-1])
                .iter::<i64>()
                .unwrap()
                .map(|i| i as u32)
                .collect();
            // Do skip special tokens
            let string = tokenizer.decode(output_ids.clone(), true).unwrap();
            all_strings.push(string);
        }
        all_strings
    }

    use once_cell::sync::Lazy;
    use std::sync::{Arc, Mutex};

    static BLOOM_350M: Lazy<Arc<Mutex<BloomForCausalLM>>> =
        Lazy::new(|| Arc::new(Mutex::new(bloom_350m())));
    static BLOOM_TESTING: Lazy<Arc<Mutex<BloomForCausalLM>>> =
        Lazy::new(|| Arc::new(Mutex::new(bloom_testing())));

    fn bloom_350m() -> BloomForCausalLM {
        let config = Config::new350m();
        let file = std::fs::File::open("./bloom-350m.bin").unwrap();
        // SAFETY: This is actually unsafe.
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let model_file = SafeTensors::deserialize(&mmap).unwrap();

        let device = Device::Cuda(0);

        let model = BloomForCausalLM::new(&config, &model_file, device);
        model
    }

    fn bloom_testing() -> BloomForCausalLM {
        let config = Config::new_testing();
        let file = std::fs::File::open("./bloom-testing.bin").unwrap();
        // SAFETY: This is actually unsafe.
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let model_file = SafeTensors::deserialize(&mmap).unwrap();

        let device = Device::Cuda(0);

        let model = BloomForCausalLM::new(&config, &model_file, device);
        model
    }

    #[test]
    fn test_simple_generation() {
        let config = Config::new350m();
        let model = BLOOM_350M.lock().unwrap();
        let tokenizer = Tokenizer::from_file("./tokenizer.json").unwrap();

        let input_sentence = "I enjoy walking with my cute dog";
        let input_sentence2 = "Hello my name is";

        let output = test_generate(&[input_sentence], &config, &tokenizer, &model, 43);
        assert_eq!(output[0], "I enjoy walking with my cute dog, and I love to watch the kids play. I am a very active person, and I am a very good listener. I am a very good person, and I am a very good person. I am a");

        let output = test_generate(&[input_sentence2], &config, &tokenizer, &model, 40);
        assert_eq!(output[0], "Hello my name is Aya, I am a beautiful, sexy, and very hot girl. I am a very good, very good, very good, very good, very good, very good, very good, very");

        let output = test_generate(
            &[input_sentence, input_sentence2],
            &config,
            &tokenizer,
            &model,
            43,
            // 21,
        );
        assert_eq!(output[0], "I enjoy walking with my cute dog, and I love to watch the kids play. I am a very active person, and I am a very good listener. I am a very good person, and I am a very good person. I am a");
        // TODO This is different from the single generation for some reason
        // This bug doesn't seem to exist on torch==1.11.0
        // **but** we need 1.12.0 for cumsum on bfloat16.
        // This bug is also present in `transformers` where the values where taken from.
        assert_eq!(output[1],  "Hello my name is Aya, I am a beautiful, sexy, and very hot girl. I am a very good and very good man, I am very good at my job, I am very good at my job, I am");
    }

    #[test]
    fn test_logits_testing() {
        let config = Config::new_testing();
        let model = BLOOM_TESTING.lock().unwrap();
        let tokenizer = Tokenizer::from_file("./tokenizer.json").unwrap();
        let device = Device::Cuda(0);

        let example_ids = &[
            3478, 368, 109586, 35433, 2, 77, 132619, 3478, 368, 109586, 35433, 2, 2175, 23714,
            73173, 144252, 2, 77, 132619, 3478,
        ];
        let tensor_ids = Tensor::of_slice(example_ids)
            .view((1, -1))
            .to_kind(kind::Kind::Int64)
            .to_device(device);
        let past = empty_past(&config);
        let (mut input_ids, mut attention_mask, mut alibi, mut past_key_values) =
            padding(&config, vec![(tensor_ids, past)]);

        let logits = model.forward(&input_ids, &attention_mask, &alibi, &mut past_key_values);
        let splits = logits.split(125440, -1);
        assert_eq!(splits.len(), 2);
        let output_gpu_1 = &splits[0];
        let output_gpu_2 = &splits[1];

        assert_all_close(
            &output_gpu_1.mean(logits.kind()),
            &Tensor::of_slice(&[-1.823902130126953e-05])
                .to_kind(config.kind)
                .to_device(device),
        );
        assert_all_close(
            &output_gpu_2.mean(logits.kind()),
            &Tensor::of_slice(&[1.9431114196777344e-05])
                .to_kind(config.kind)
                .to_device(device),
        );
    }

    #[test]
    fn test_embeddings_testing() {
        let config = Config::new_testing();
        let model = BLOOM_TESTING.lock().unwrap();
        let tokenizer = Tokenizer::from_file("./tokenizer.json").unwrap();
        let device = Device::Cuda(0);

        let example_ids = &[
            3478, 368, 109586, 35433, 2, 77, 132619, 2175, 23714, 73173, 144252,
        ];
        let tensor_ids = Tensor::of_slice(example_ids)
            .view((1, -1))
            .to_kind(kind::Kind::Int64)
            .to_device(device);
        let past = empty_past(&config);
        let (mut input_ids, mut attention_mask, mut alibi, mut past_key_values) =
            padding(&config, vec![(tensor_ids, past)]);

        let embeddings = model.transformer.word_embeddings.forward(&input_ids);

        assert_all_close(
            &embeddings.mean_dim(&[-1], false, embeddings.kind()),
            &Tensor::of_slice(&[
                0.0002307891845703125,
                -0.000568389892578125,
                -0.0003910064697265625,
                -0.000194549560546875,
                0.0004138946533203125,
                0.000659942626953125,
                -0.00031280517578125,
                0.000457763671875,
                0.000263214111328125,
                -0.000286102294921875,
                0.00052642822265625,
            ])
            .view((1, -1))
            .to_kind(config.kind)
            .to_device(device),
        );

        println!("Mean OK");
        assert_all_close(
            &embeddings.min_dim(-1, false).0,
            &Tensor::of_slice(&[
                -0.00921630859375,
                -0.010009765625,
                -0.01031494140625,
                -0.01177978515625,
                -0.0074462890625,
                -0.00848388671875,
                -0.009521484375,
                -0.0074462890625,
                -0.0145263671875,
                -0.007415771484375,
                -0.01007080078125,
            ])
            .view((1, -1))
            .to_kind(config.kind)
            .to_device(device),
        );
        println!("Min OK");

        assert_all_close(
            &embeddings.max_dim(-1, false).0,
            &Tensor::of_slice(&[
                0.0128173828125,
                0.01214599609375,
                0.0111083984375,
                0.01019287109375,
                0.0157470703125,
                0.0174560546875,
                0.0078125,
                0.0113525390625,
                0.0146484375,
                0.01116943359375,
                0.01141357421875,
            ])
            .view((1, -1))
            .to_kind(config.kind)
            .to_device(device),
        );

        let embeddings_ln = model
            .transformer
            .word_embeddings_layernorm
            .forward(&embeddings);

        assert_all_close(
            &embeddings_ln.mean_dim(&[-1], false, embeddings_ln.kind()),
            &Tensor::of_slice(&[
                -6.580352783203125e-05,
                0.0001316070556640625,
                -0.00030517578125,
                4.00543212890625e-05,
                -7.2479248046875e-05,
                -8.96453857421875e-05,
                0.0001583099365234375,
                2.1219253540039062e-05,
                -0.000247955322265625,
                -0.00021839141845703125,
                -0.0001430511474609375,
            ])
            .view((1, -1))
            .to_kind(config.kind)
            .to_device(device),
        );

        assert_all_close(
            &embeddings_ln.min_dim(-1, false).0,
            &Tensor::of_slice(&[
                -1.6953125, -1.6875, -1.6875, -2.125, -1.390625, -1.5390625, -1.875, -1.4609375,
                -2.296875, -1.3515625, -1.78125,
            ])
            .view((1, -1))
            .to_kind(config.kind)
            .to_device(device),
        );
        assert_all_close(
            &embeddings_ln.max_dim(-1, false).0,
            &Tensor::of_slice(&[
                2.265625, 2.28125, 1.953125, 1.90625, 2.703125, 2.828125, 1.65625, 2.015625,
                2.234375, 2.171875, 1.828125,
            ])
            .view((1, -1))
            .to_kind(config.kind)
            .to_device(device),
        );
    }

    #[test]
    fn test_embeddings_full() {
        let device = Device::Cuda(0);
        let config = Config::new();
        let file = std::fs::File::open("bloom-embedding.bin").unwrap();
        // SAFETY: This is actually unsafe.
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let embedding_model = SafeTensors::deserialize(&mmap).unwrap();

        let word_embeddings = Embedding::new("word_embeddings", &embedding_model, device);
        let word_embeddings_layernorm = LayerNorm::new(
            config.hidden_size,
            "word_embeddings_layernorm",
            &embedding_model,
            device,
        );
        let input_ids = Tensor::of_slice(&[0, 1, 2, 3, 4, 5])
            .view((1, -1))
            .to_device(device);

        let inputs_embeds = word_embeddings.forward(&input_ids);

        let mut hidden_states = word_embeddings_layernorm.forward(&inputs_embeds);
    }

    #[test]
    fn test_unpacking_params() {
        let parameters: Parameters = serde_json::from_str(r#"{"do_sample": true}"#).unwrap();
        assert_eq!(
            parameters,
            Parameters {
                generation_mode: GenerationMode::Sampling(Sampling {
                    top_p: None,
                    top_k: None,
                    temperature: 1.0,
                }),
                max_new_tokens: 20
            }
        );
        let parameters: Parameters = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(
            parameters,
            Parameters {
                generation_mode: GenerationMode::Greedy,
                max_new_tokens: 20
            }
        );
        let parameters: Parameters = serde_json::from_str(r#"{"do_sample": false}"#).unwrap();
        assert_eq!(
            parameters,
            Parameters {
                generation_mode: GenerationMode::Greedy,
                max_new_tokens: 20
            }
        );
        let parameters: Parameters = serde_json::from_str(r#"{"top_k": 20}"#).unwrap();
        assert_eq!(
            parameters,
            Parameters {
                generation_mode: GenerationMode::Sampling(Sampling {
                    top_p: None,
                    top_k: Some(20),
                    temperature: 1.0,
                }),
                max_new_tokens: 20
            }
        );
    }
    #[test]
    fn test_sampling_top_p() {
        // dist = torch.log(
        //     torch.tensor([[0.3, 0.1, 0.1, 0.5], [0.15, 0.3, 0.3, 0.25]], device=torch_device, dtype=torch.float)
        // )
        let top_p = 0.7;
        let dist = Tensor::of_slice(&[0.3, 0.1, 0.1, 0.5, 0.15, 0.3, 0.3, 0.25]).view((2, 4));

        // top_p_warp = TopPLogitsWarper(0.7)
        // filtered_dist = torch.exp(top_p_warp(input_ids, dist))
        let filtered_dist = filter_top_p(&dist.log(), top_p).exp();

        // # dist should be filtered to keep min num values so that sum is >= 0.7
        // # exp (-inf) => 0
        // EXPECTED_FILTERED_DIST = torch.tensor(
        //     [[0.3, 0.0, 0.0, 0.5], [0.0, 0.3, 0.3, 0.25]], device=torch_device, dtype=torch.float
        // )
        let expected = Tensor::of_slice(&[0.3, 0.0, 0.0, 0.5, 0.0, 0.3, 0.3, 0.25]).view((2, 4));

        // self.assertTrue(torch.allclose(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3))
        let rtol = 1e-05;
        let atol = 1e-03;
        let equal_nan = false;
        assert!(filtered_dist.allclose(&expected, rtol, atol, equal_nan));
    }
}
