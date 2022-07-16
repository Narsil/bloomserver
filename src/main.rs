#![feature(async_closure)]
use crate::utils::debug;
use actix_web::middleware::Logger;
use actix_web::{
    http::header::ContentType, http::StatusCode, post, web, App, HttpResponse, HttpServer,
    ResponseError,
};
use crossbeam_channel::{bounded, unbounded, Receiver, Select, Sender};
use memmap::MmapOptions;
use safetensors::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tch::{kind, Device, IndexOp, Tensor};
use thiserror::Error;
use tokenizers::Tokenizer;

pub mod model;
#[cfg(test)]
mod test;
pub mod utils;

use crate::model::{
    build_alibi_tensor, BloomBlock, Config, Embedding, InvertedEmbedding, LayerNorm, Past,
};

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

/// Size of batch and response channel
type Ack = (i64, Sender<(Tensor, Past)>);
type Msg = (Tensor, Past, Ack);
type Msg2 = ((Tensor, Tensor, Tensor, Past), Vec<Ack>);
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
                        temperature,
                        top_k: data.top_k,
                        top_p: data.top_p,
                    })
                } else if let Some(_top_k) = data.top_k {
                    GenerationMode::Sampling(Sampling {
                        temperature: data.temperature.unwrap_or(1.0),
                        top_k: data.top_k,
                        top_p: data.top_p,
                    })
                } else if let Some(_top_p) = data.top_p {
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

fn filter_top_p(scored_logits: &Tensor, top_p: f64) -> Tensor {
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
    let _ = sorted_indices_to_remove.i((.., -1)).f_fill_(0).unwrap();
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
            let seq_length = logits.size()[1];
            let new_ids = logits
                .i((0..1, seq_length - 1..seq_length))
                .argmax(-1, false)
                .to_device(input_ids.device());
            Tensor::f_cat(&[input_ids.copy(), new_ids.copy()], 1).unwrap()
        }
        GenerationMode::Sampling(params) => {
            let seq_length = logits.size()[1];
            let filter_value = f64::NEG_INFINITY;
            let last_logits = logits
                .i((0, seq_length - 1..seq_length))
                .to_device(input_ids.device());

            let mut scored_logits = last_logits / params.temperature;

            if let Some(top_k) = params.top_k {
                // indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
                let largest = true;
                let sorted = true;
                let top_ks = scored_logits.topk(top_k as i64, -1, largest, sorted).0;
                let top_k = top_ks.i((.., -1));

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
    if ids.is_empty() {
        ids.push(0);
    }
    let (sx, rx) = bounded::<(Tensor, Past)>(2);
    let kind = (kind::Kind::Int64, Device::Cuda(0));
    let mut input_ids = Tensor::of_slice(ids.as_slice())
        .to_kind(kind.0)
        .to_device(kind.1)
        .view((1, -1));
    let max_new_tokens = payload.parameters.max_new_tokens;

    for i in 0..max_new_tokens {
        // let start_loop = Instant::now();
        let ack = (input_ids.size()[0], sx.clone());
        let past_key_values = empty_past(&state.config);
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
    config: Config,
}

pub fn convert(view: TensorView, device: Device) -> Tensor {
    let kind = match view.get_dtype() {
        Dtype::F16 => kind::Kind::Half,
        Dtype::BF16 => kind::Kind::BFloat16,
        _ => {
            todo!("Need to implement that");
        }
    };
    let t = Tensor::of_data_size(
        view.get_data(),
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

fn padding_with_ack(
    config: &Config,
    items: Vec<(Tensor, Past, Ack)>,
) -> ((Tensor, Tensor, Tensor, Past), Vec<Ack>) {
    let mut tensors = vec![];
    let mut acks = vec![];
    for item in items {
        tensors.push((item.0, item.1));
        acks.push(item.2);
    }
    (padding(config, tensors), acks)
}

fn padding(config: &Config, items: Vec<(Tensor, Past)>) -> (Tensor, Tensor, Tensor, Past) {
    // TODO
    let max_length = items.iter().map(|(ids, _)| ids.size()[1]).max().unwrap();
    let batch_size: i64 = items.iter().map(|(ids, _)| ids.size()[0]).sum::<i64>();
    let kind = (kind::Kind::Int64, Device::Cuda(0));
    let device = items[0].0.device();
    let kind2 = (kind::Kind::Int64, device);
    let mut all_input_ids = Tensor::zeros(&[batch_size, max_length], kind2) + config.padding_idx;
    let mut attention_mask = Tensor::zeros(&[batch_size, max_length], kind2);

    // let mut total_ids = 0;

    let mut current_batch = 0;
    for (input_ids, _past_key_values) in items {
        let seq_length = input_ids.size()[1];
        let mini_batch_size = input_ids.size()[0];
        // total_ids += mini_batch_size as usize * seq_length as usize;
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

    // let total = std::cmp::max(1, batch_size as usize * max_length as usize);
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

    let word_embeddings = Embedding::new("word_embeddings", &embedding_model, device);
    let word_embeddings_layernorm = LayerNorm::new(
        config.hidden_size,
        "word_embeddings_layernorm",
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
                &if std::env::var("BLOOM").unwrap_or_else(|_| "".to_string()) == "bloom-dgx" {
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
                &if std::env::var("BLOOM").unwrap_or_else(|_| "".to_string()) == "bloom-dgx" {
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
                debug("past_key thread2", &layer_past.0);
                debug("past_values thread2", &layer_past.1);
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

    let ln_f = LayerNorm::new(config.hidden_size, "ln_f", &final_model, device);
    let lm_head = InvertedEmbedding::new("word_embeddings", &embedding_model, device);

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
                &if std::env::var("BLOOM").unwrap_or_else(|_| "".to_string()) == "bloom-dgx" {
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
            debug("past_key thread3", &layer_past.0);
            debug("past_values thread3", &layer_past.1);
            hidden_states = layer.forward(&hidden_states, &attention_mask, &alibi, layer_past);
        }
        debug("last_hidden_states", &hidden_states);
        hidden_states = ln_f.forward(&hidden_states);
        debug("After ln_f", &hidden_states);
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
                config: config.clone(),
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
    use crate::model::tests::{BLOOM_350M, BLOOM_TESTING};
    use crate::model::BloomForCausalLM;
    use crate::test::assert_all_close;

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
            let item = (input_ids, past);
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

    #[test]
    fn test_simple_generation() {
        let config = Config::new350m();
        let model = BLOOM_350M.lock().unwrap();
        let tokenizer = Tokenizer::from_file("./tokenizer.json").unwrap();

        let input_sentence = "I enjoy walking with my cute dog";
        let input_sentence2 = "Hello my name is";

        let output = test_generate(&[input_sentence], &config, &tokenizer, &model, 43);
        assert_eq!(output[0], "I enjoy walking with my cute dog, and I love to watch the kids play. I am a very active person, and I am very active. I am a very good listener, and I am very good at listening. I am a very good");

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
        assert_eq!(output[0], "I enjoy walking with my cute dog, and I love to watch the kids play. I am a very active person, and I am very active. I am a very good listener, and I am very good at listening. I am a very good");
        // TODO This is different from the single generation for some reason
        // This bug doesn't seem to exist on torch==1.11.0
        // **but** we need 1.12.0 for cumsum on bfloat16.
        // This bug is also present in `transformers` where the values where taken from.
        assert_eq!(output[1],  "Hello my name is Aya, I am a beautiful, sexy, and very hot girl. I am a very good and very good man, I am very good at my job, I am very good at my job, I am");
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

    #[test]
    fn test_logits_testing() {
        let config = Config::new_testing();
        let model = BLOOM_TESTING.lock().unwrap();
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
        let (input_ids, attention_mask, alibi, mut past_key_values) =
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
        let device = Device::Cuda(0);

        let example_ids = &[
            3478, 368, 109586, 35433, 2, 77, 132619, 2175, 23714, 73173, 144252,
        ];
        let tensor_ids = Tensor::of_slice(example_ids)
            .view((1, -1))
            .to_kind(kind::Kind::Int64)
            .to_device(device);
        let past = empty_past(&config);
        let (input_ids, _attention_mask, _alibi, _past_key_values) =
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
}
