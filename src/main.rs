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

const PADDING_IDX: i64 = 2;
const EPS: f64 = 1e-5;

#[cfg(not(feature = "bloom-350m"))]
const N_HEAD: usize = 112;
#[cfg(not(feature = "bloom-350m"))]
const HIDDEN_SIZE: usize = 14336;
#[cfg(not(feature = "bloom-350m"))]
const N_LAYER: usize = 70;
#[cfg(not(feature = "bloom-350m"))]
const MODEL_KIND: kind::Kind = kind::Kind::BFloat16;

#[cfg(feature = "bloom-350m")]
const N_HEAD: usize = 16;
#[cfg(feature = "bloom-350m")]
const HIDDEN_SIZE: usize = 1024;
#[cfg(feature = "bloom-350m")]
const N_LAYER: usize = 24;
#[cfg(feature = "bloom-350m")]
const MODEL_KIND: kind::Kind = kind::Kind::Half;

const LAYERS_FIRST_THREAD: usize = 0;
const LAYERS_PER_THREAD: usize = 5;
const N_THREADS: usize = 14;
const LAYERS_LAST_THREAD: usize = 0;

type PastLayer = (Tensor, Tensor);
type Past = Vec<PastLayer>;
type Msg = (Tensor, Past, Sender<(Tensor, Past)>);
type Msg2 = (Tensor, Tensor, Tensor, Past, Vec<Sender<(Tensor, Past)>>);
type InChan = Sender<Msg>;
type RChan1 = Receiver<Msg>;
type RChan = Receiver<Msg2>;
type SChan = Sender<Msg2>;

#[derive(Deserialize, Serialize)]
struct Parameters {
    top_k: Option<usize>,
    max_new_tokens: usize,
    #[serde(default)]
    stream: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            top_k: None,
            max_new_tokens: 20,
            stream: false,
        }
    }
}

#[derive(Deserialize, Serialize)]
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

struct Stream {
    new_generated_tokens: usize,
    ids: Vec<i64>,
    payload: Generation,
    in_channel: InChan,
    prio_channel: InChan,
    tokenizer: Arc<Tokenizer>,
    start: std::time::Instant,
    sent: bool,
    sx: Sender<(Tensor, Past)>,
    rx: Receiver<(Tensor, Past)>,
}

impl Stream {
    fn new(
        payload: Generation,
        in_channel: InChan,
        prio_channel: InChan,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        let encoded = tokenizer.encode(payload.inputs.clone(), false).unwrap();
        let mut ids: Vec<_> = encoded.get_ids().iter().map(|&i| i as i64).collect();
        // TODO The model is not able to handle empty input ids
        if ids.len() == 0 {
            ids.push(0);
        }
        let start = std::time::Instant::now();
        let (sx, rx) = unbounded::<(Tensor, Past)>();
        Self {
            payload,
            in_channel,
            prio_channel,
            tokenizer,
            new_generated_tokens: 0,
            ids,
            start,
            sent: false,
            sx,
            rx,
        }
    }
}

impl futures::Stream for Stream {
    type Item = Result<Bytes, GenerationError>;

    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        // if this.sent {
        if this.new_generated_tokens >= this.payload.parameters.max_new_tokens {
            return Poll::Ready(None);
        }
        let kind = (kind::Kind::Int, Device::Cuda(0));
        let input_ids = Tensor::of_slice(this.ids.as_slice())
            .to_kind(kind.0)
            .to_device(kind.1)
            .view((1, -1));

        let p = N_HEAD as i64;
        let q = (HIDDEN_SIZE / N_HEAD) as i64;
        let past_key = Tensor::zeros(&[1, 0, p, q], kind);
        let past_value = Tensor::zeros(&[1, 0, p, q], kind);
        let past_key_values: Vec<_> = (0..N_LAYER)
            .map(|_| (past_key.copy(), past_value.copy()))
            .collect();

        if this.new_generated_tokens == 0 {
            this.in_channel
                .try_send((input_ids.copy(), past_key_values, this.sx.clone()))
                .map_err(|_| {
                    println!("Queue was full {:?}", this.in_channel.len());
                    GenerationError::QueueFull
                })?;
        } else {
            this.prio_channel
                .send((input_ids.copy(), past_key_values, this.sx.clone()))
                .expect("This send should always work");
        }
        this.sent = true;
        // } else {
        // if this.rx.is_empty() {
        //     println!("Pending waiting on channel");
        //     Poll::Pending
        // } else {
        let (logits, _r_past_key_values) = this.rx.recv().unwrap();
        this.sent = false;
        let S = logits.size()[1];
        let new_id = logits.slice(1, S - 1, S, 1).argmax(-1, false);
        // past_key_values = r_past_key_values;
        // input_ids = Tensor::f_cat(&[input_ids, new_id.copy()], 1).unwrap();
        let received_ids: Vec<i64> = new_id.try_into().unwrap();
        this.ids.push(received_ids[0]);
        this.new_generated_tokens += 1;

        if this.payload.parameters.stream {
            let received_ids: Vec<_> = received_ids.iter().map(|&i| i as u32).collect();
            let string = this.tokenizer.decode(received_ids.clone(), false).unwrap();
            let result = serde_json::to_string(&json!([{ "text": string }])).unwrap();
            Poll::Ready(Some(Ok(Bytes::copy_from_slice(result.as_bytes()))))
        } else {
            if this.new_generated_tokens == this.payload.parameters.max_new_tokens {
                let n = this.new_generated_tokens;
                println!(
                    "Inference generated {n} tokens in {:?} ({:?}/tok)",
                    this.start.elapsed(),
                    this.start.elapsed().div_f32(n as f32)
                );
                let full_ids: Vec<_> = this.ids.iter().map(|&i| i as u32).collect();
                let string = this.tokenizer.decode(full_ids, false).unwrap();
                let result = serde_json::to_string(&json!([{ "generated_text": string }])).unwrap();
                Poll::Ready(Some(Ok(Bytes::copy_from_slice(result.as_bytes()))))
            } else {
                Poll::Ready(Some(Ok(Bytes::copy_from_slice(b""))))
            }
        }
        // }
        // }
    }
}

#[post("/generate")]
async fn generate(payload: web::Json<Generation>, state: web::Data<AppState>) -> HttpResponse {
    let state = state.into_inner();
    let stream = Stream::new(
        payload.into_inner(),
        state.in_channel.clone(),
        state.prio_channel.clone(),
        state.tokenizer.clone(),
    );
    HttpResponse::Ok()
        .content_type(ContentType::json())
        .streaming(stream)
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
    return x + 1;
}

fn get_slopes(n: usize) -> Vec<f64> {
    let is_power_of_2 = n == 0 || n & (n - 1) != 0;
    if is_power_of_2 {
        return get_slopes_power_of_2(n);
    } else {
        let closest_power_of_2 = next_pow2(n);
        return get_slopes_power_of_2(closest_power_of_2)
            .into_iter()
            .chain(
                get_slopes(closest_power_of_2 >> 1)
                    .into_iter()
                    .step_by(2)
                    .take(n - closest_power_of_2),
            )
            .collect();
    }
}

fn build_alibi_tensor(attention_mask: &Tensor, n_head: usize, device: Device) -> Tensor {
    let slopes = get_slopes(n_head);
    let kind = MODEL_KIND;
    let slopes = Tensor::of_slice(&slopes).to_kind(kind).to_device(device);
    debug("slopes", &slopes);
    let A = attention_mask.f_cumsum(-1, kind).unwrap().unsqueeze(1) - 1;
    debug("A", &A);
    let B = attention_mask.unsqueeze(1);
    debug("B", &B);
    let arange_tensor = A * B;
    debug("A range tensor", &arange_tensor);
    let slopes = slopes.unsqueeze(-1);
    debug("Slopes", &slopes);
    let mut alibi = slopes * arange_tensor;
    debug("alibi1 ", &alibi);
    alibi = alibi * attention_mask.unsqueeze(1);
    debug("alibi2 ", &alibi);

    let size = attention_mask.size();
    let batch_size = size[0];
    let seq_length = size[1];
    alibi = alibi.reshape(&[batch_size * (N_HEAD as i64), 1, seq_length]);
    debug("alibi", &alibi);
    return alibi;
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn new(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let weight_name = format!("{name}.weight");
        let weight = convert(
            model
                .tensor(&weight_name)
                .expect(&format!("Failed to load {weight_name}")),
            device,
        );
        let bias_name = format!("{name}.bias");
        let bias = convert(
            model
                .tensor(&bias_name)
                .expect(&format!("Failed to load {bias_name}")),
            device,
        );
        Self { weight, bias }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        // println!(
        //     "Layer norm {:?} - {:?} - {:?}",
        //     self.weight.size(),
        //     self.bias.size(),
        //     xs.size()
        // );
        xs.f_layer_norm(
            &[HIDDEN_SIZE as i64],
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

    let a_attention_mask_bool = attention_mask_bool
        .unsqueeze(1)
        .unsqueeze(-1)
        .f_slice(2, key_length - query_length, key_length, 1)
        .unwrap();
    let b_causal_mask = causal_mask
        .f_logical_not()
        .unwrap()
        .f_slice(2, key_length - query_length, key_length, 1)
        .unwrap()
        .f_slice(3, 0, key_length, 1)
        .unwrap();
    debug("A attention_mask_bool", &a_attention_mask_bool);
    debug("b causal mask", &b_causal_mask);
    let mut padded_causal_mask = a_attention_mask_bool.f_logical_or(&b_causal_mask).unwrap();
    debug("padded causal_mask", &padded_causal_mask);
    padded_causal_mask = padded_causal_mask
        .f_logical_or(
            &attention_mask_bool
                .unsqueeze(1)
                .unsqueeze(1)
                .f_slice(3, 0, key_length, 1)
                .unwrap(),
        )
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
}

impl BloomScaledSoftmax {
    fn new(scale: i64) -> Self {
        Self { scale }
    }

    fn forward(&self, input: &Tensor, attention_mask: &Tensor, max_positions: i64) -> Tensor {
        debug("input", input);
        let mut scaled_input = self.scale * input;
        debug("scaled input", &scaled_input);
        let seq_ids = Tensor::f_arange(max_positions, (kind::Kind::Int, input.device())).unwrap();

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

        let out_probs = probs.to_kind(MODEL_KIND);
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
    norm_factor: f64,
}

impl BloomAttention {
    fn new(name: &str, model: &SafeTensors<'_>, layer_number: usize, device: Device) -> Self {
        let dense = Linear::new(&format!("{name}.dense"), model, device);
        let query_key_value = Linear::new(&format!("{name}.query_key_value"), model, device);
        let head_dim = (HIDDEN_SIZE / N_HEAD) as i64;
        let num_attention_heads = N_HEAD as i64;
        let layer_number = std::cmp::max(1, layer_number);
        let norm_factor = (head_dim as f64).sqrt() * layer_number as f64;
        let scaled_softmax = BloomScaledSoftmax::new(layer_number as i64);
        Self {
            dense,
            query_key_value,
            num_attention_heads,
            head_dim,
            layer_number,
            norm_factor,
            scaled_softmax,
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
        let NH = (HIDDEN_SIZE / N_HEAD) as i64;
        let output_size = (B, H, Q, K);

        let query_layer = query.transpose(1, 0).reshape(&[Q, B * H, -1]);
        key_layer = key_layer.transpose(1, 0).reshape(&[K, B * H, -1]);

        // let sliced_alibi = alibi[: output_size[0] * output_size[1], :, : output_size[3]]
        let sliced_alibi = alibi.slice(0, 0, B * H, 1).slice(2, 0, K, 1);
        let beta = 1.0 / (self.layer_number as f64);
        let alpha = 1.0 / self.norm_factor;
        let b = beta * sliced_alibi;
        let a = alpha * query_layer.transpose(1, 0);
        let c = key_layer.transpose(1, 0).transpose(1, 2);
        debug("A", &a);
        debug("B", &b);
        debug("C", &c);
        let matmul_result = b
            .f_baddbmm(
                &a, &c,
                // beta=beta,
                // alpha=(1.0 / self.norm_factor),
            )
            .unwrap();
        let attention_scores = matmul_result.f_view(output_size).unwrap();
        debug(
            &format!("Attention baddbmm {layer_number}"),
            &attention_scores,
        );
        let max_positions = std::cmp::max(attention_scores.size()[3], attention_scores.size()[2]);
        let attention_probs =
            self.scaled_softmax
                .forward(&attention_scores, attention_mask, max_positions);
        let value_layer = value.transpose(1, 0).reshape(&[K, B * H, -1]);
        let attention_probs_reshaped = attention_probs.f_view((B * H, Q, -1)).unwrap();
        let context_layer = Tensor::bmm(&attention_probs_reshaped, &value_layer.transpose(0, 1));
        let context_layer_r = context_layer.f_view((B, H, Q, -1)).unwrap();
        let context_layer_p = context_layer_r.permute(&[2, 0, 1, 3]).contiguous();
        let context = context_layer_p.f_view((Q, B, H * NH)).unwrap();
        let output_tensor = self.dense.forward(&context);
        let mut output = output_tensor.transpose(1, 0);
        output += residual;
        output
    }
}

struct BloomMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

fn bloom_gelu(x: &Tensor) -> Tensor {
    let y: Tensor = 0.79788456 * x * (1.0 + 0.044715 * x * x);
    return x * 0.5 * (1.0 + y.tanh());
}

impl BloomMlp {
    fn new(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let dense_h_to_4h = Linear::new(&format!("{name}.dense_h_to_4h"), model, device);
        let dense_4h_to_h = Linear::new(&format!("{name}.dense_4h_to_h"), model, device);
        Self {
            dense_h_to_4h,
            dense_4h_to_h,
        }
    }

    fn forward(&self, hidden_states: &Tensor, residual: &Tensor) -> Tensor {
        debug("hidden_states", hidden_states);
        debug("INCOMING residual", residual);
        let hidden_states = self.dense_h_to_4h.forward(hidden_states);
        debug("hidden_states h to 4h", &hidden_states);
        let hidden_states = bloom_gelu(&hidden_states);
        debug("hidden_states gelu", &hidden_states);
        let hidden_states = self.dense_4h_to_h.forward(&hidden_states);
        debug("hidden_states 4h to h", &hidden_states);
        let hidden_states = hidden_states + residual;
        debug("hidden_states residual", &hidden_states);
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
    fn new(prefix: &str, model: &SafeTensors<'_>, layer_number: usize, device: Device) -> Self {
        // attention
        let input_layernorm = LayerNorm::new(&format!("{prefix}.input_layernorm"), model, device);
        let post_attention_layernorm =
            LayerNorm::new(&format!("{prefix}.post_attention_layernorm"), model, device);
        let self_attention = BloomAttention::new(
            &format!("{prefix}.self_attention"),
            model,
            layer_number,
            device,
        );
        let mlp = BloomMlp::new(&format!("{prefix}.mlp"), model, device);
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

fn debug(prefix: &str, x: &Tensor) {
    // println!(
    //     "{prefix} - {:?} - Values: {:?}",
    //     x.size(),
    //     x.reshape(&[-1,])
    //         .iter::<f64>()
    //         .unwrap()
    //         .take(10)
    //         .collect::<Vec<_>>()
    // );
}

pub struct BloomModel {
    word_embeddings: Embedding,
    word_embeddings_layernorm: LayerNorm,
    h: Vec<BloomBlock>,
    ln_f: LayerNorm,
}

impl BloomModel {
    pub fn new(model: &SafeTensors<'_>, device: Device) -> Self {
        let word_embeddings = Embedding::new(&format!("word_embeddings"), model, device);
        let word_embeddings_layernorm =
            LayerNorm::new(&format!("word_embeddings_layernorm"), model, device);
        let ln_f = LayerNorm::new(&format!("ln_f"), model, device);
        let h = (0..N_LAYER)
            .map(|i| BloomBlock::new(&format!("h.{i}"), model, i, device))
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
        let result = xs.f_linear::<Tensor>(&self.weight, None).unwrap();
        debug("Outgoing tensor ", &result);
        result
    }
}

struct BloomForCausalLM {
    transformer: BloomModel,
    lm_head: InvertedEmbedding,
}

impl BloomForCausalLM {
    fn new(model: &SafeTensors<'_>, device: Device) -> Self {
        let transformer = BloomModel::new(model, device);
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

fn padding(
    mut items: Vec<(Tensor, Past, Sender<(Tensor, Past)>)>,
) -> (Tensor, Tensor, Tensor, Past, Vec<Sender<(Tensor, Past)>>) {
    // TODO
    let max_length = items.iter().map(|(ids, _, _)| ids.size()[1]).max().unwrap();
    let batch_size = items.len() as i64;
    let kind = (kind::Kind::Int64, Device::Cuda(0));
    let device = items[0].0.device();
    let kind2 = (kind::Kind::Int, device);
    let mut all_input_ids = Tensor::zeros(&[batch_size, max_length], kind2) + PADDING_IDX;
    let mut attention_mask = Tensor::zeros(&[batch_size, max_length], kind2);
    let mut alibi = Tensor::zeros(&[batch_size, max_length], kind2);
    let mut rqs = vec![];

    let mut total_ids = 0;

    for (i, (input_ids, past_key_values, rq)) in items.into_iter().enumerate() {
        let seq_length = input_ids.size()[1];
        total_ids += seq_length as usize;
        // all_input_ids[i:i+1, max_length - seq_length:seq_length] = input_ids[0]
        //
        let batch_index = Tensor::of_slice(vec![i as i64; seq_length as usize].as_slice())
            .to_kind(kind.0)
            .to_device(kind.1);
        let id_index = Tensor::arange(seq_length, kind) + max_length - seq_length;

        let id_row = input_ids.i((0,));
        all_input_ids = all_input_ids
            .f_index_put_(&[Some(&batch_index), Some(&id_index)], &id_row, false)
            .unwrap();

        let attn = input_ids.fill(1);
        attention_mask = attention_mask
            .f_index_put(&[Some(&batch_index), Some(&id_index)], &attn.i((0,)), false)
            .unwrap();
        rqs.push(rq);
    }

    let p = N_HEAD as i64;
    let q = (HIDDEN_SIZE / N_HEAD) as i64;
    let past_key = Tensor::zeros(&[batch_size, 0, p, q], kind);
    let past_value = Tensor::zeros(&[batch_size, 0, p, q], kind);
    let past_key_values: Vec<_> = (0..N_LAYER)
        .map(|_| (past_key.copy(), past_value.copy()))
        .collect();

    let alibi = build_alibi_tensor(&attention_mask, N_HEAD, device);

    let total = std::cmp::max(1, batch_size as usize * max_length as usize);
    // println!(
    //     "Running on batch of size {:?} - Fillrate {:?}%",
    //     batch_size,
    //     (total_ids * 100) / total
    // );
    (all_input_ids, attention_mask, alibi, past_key_values, rqs)
}

fn thread1(rx: RChan1, prio_rx: RChan1, s2: SChan, thread_number: usize) {
    println!("Starting thread {thread_number}");
    let start = std::time::Instant::now();
    let device = Device::Cuda(thread_number);

    let file = std::fs::File::open("./bloom-embedding.bin").unwrap();
    // SAFETY: This is actually unsafe.
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let embedding_model = SafeTensors::deserialize(&mmap).unwrap();

    // println!("Embedding {:?}", embedding_model.names());
    // println!("Layer {:?}", model.names());

    let word_embeddings = Embedding::new(&format!("word_embeddings"), &embedding_model, device);
    let word_embeddings_layernorm = LayerNorm::new(
        &format!("word_embeddings_layernorm"),
        &embedding_model,
        device,
    );

    let layers: Vec<BloomBlock> = (0..LAYERS_FIRST_THREAD)
        .map(|i| {
            let file_number = i + 1;
            let file = std::fs::File::open(&format!("./bloom-h.{file_number}.bin")).unwrap();
            // SAFETY: This is actually unsafe.
            let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
            let model = SafeTensors::deserialize(&mmap).unwrap();
            BloomBlock::new(&format!("h.{i}"), &model, i, device)
        })
        .collect();
    println!(
        "{:?} : Loaded thread {thread_number} in {:?}",
        std::time::Instant::now(),
        start.elapsed()
    );

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
        let instant = Instant::now();
        let deadline = instant + Duration::from_millis(1);

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
        if rx.len() != 0 {
            println!("After deadline in queue {:?}", rx.len());
        }
        if prio_rx.len() != 0 {
            println!("After deadline prio queue {:?}", prio_rx.len());
        }

        let (input_ids, attention_mask, alibi, mut past_key_values, rqs) = padding(all_items);
        let inputs_embeds = word_embeddings.forward(&input_ids);
        let mut hidden_states = word_embeddings_layernorm.forward(&inputs_embeds);

        for (layer, layer_past) in layers.iter().zip(past_key_values.iter_mut()) {
            hidden_states = layer.forward(&hidden_states, &attention_mask, &alibi, layer_past);
        }
        s2.send((hidden_states, attention_mask, alibi, past_key_values, rqs))
            .unwrap();
    }
}

fn thread2(rx: RChan, s: SChan, thread_number: usize) {
    println!("Starting thread {thread_number}");
    let start = std::time::Instant::now();
    let device = Device::Cuda(thread_number);

    let layers: Vec<BloomBlock> = (0..LAYERS_PER_THREAD)
        .map(|i| {
            let layer_number = i + LAYERS_FIRST_THREAD + LAYERS_PER_THREAD * (thread_number - 1);
            println!("Loading layer {layer_number} on thread2 ({thread_number})");
            let file_number = layer_number + 1;
            let file = std::fs::File::open(&format!("./bloom-h.{file_number}.bin")).unwrap();
            // SAFETY: This is actually unsafe.
            let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
            let model = SafeTensors::deserialize(&mmap).unwrap();
            BloomBlock::new(&format!("h.{layer_number}"), &model, layer_number, device)
        })
        .collect();
    println!(
        "{:?} : Loaded thread {thread_number} in {:?}",
        std::time::Instant::now(),
        start.elapsed()
    );

    loop {
        let (mut hidden_states, mut attention_mask, mut alibi, mut past_key_values, rq) = rx
            .recv()
            .expect("You probably want to handle this case, but I'm too lazy");
        hidden_states = hidden_states.to_device(device);
        attention_mask = attention_mask.to_device(device);
        alibi = alibi.to_device(device);
        for (layer, layer_past) in layers
            .iter()
            .zip(past_key_values.iter_mut().skip(5 + 7 * (thread_number - 1)))
        {
            debug(&format!("past_key thread2"), &layer_past.0);
            debug(&format!("past_values thread2"), &layer_past.1);
            hidden_states = layer.forward(&hidden_states, &attention_mask, &alibi, layer_past);
        }
        s.send((hidden_states, attention_mask, alibi, past_key_values, rq))
            .unwrap();
    }
}

fn thread3(rx: RChan, thread_number: usize) {
    println!("Starting thread {thread_number}");
    let start = std::time::Instant::now();
    let device = Device::Cuda(thread_number);

    let file = std::fs::File::open("./bloom-embedding.bin").unwrap();
    // SAFETY: This is actually unsafe.
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let embedding_model = SafeTensors::deserialize(&mmap).unwrap();

    let file = std::fs::File::open("./bloom-final.bin").unwrap();
    // SAFETY: This is actually unsafe.
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let final_model = SafeTensors::deserialize(&mmap).unwrap();

    let ln_f = LayerNorm::new(&format!("ln_f"), &final_model, device);
    let lm_head = InvertedEmbedding::new(&format!("word_embeddings"), &embedding_model, device);

    let layers: Vec<BloomBlock> = (0..LAYERS_LAST_THREAD)
        .map(|i| {
            let layer_number = LAYERS_FIRST_THREAD + LAYERS_PER_THREAD * N_THREADS + i;
            println!("Loading layer {layer_number} on thread3 ({thread_number})");
            let file_number = layer_number + 1;
            let file = std::fs::File::open(&format!("./bloom-h.{file_number}.bin")).unwrap();
            // SAFETY: This is actually unsafe.
            let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
            let model = SafeTensors::deserialize(&mmap).unwrap();
            BloomBlock::new(&format!("h.{layer_number}"), &model, layer_number, device)
        })
        .collect();

    println!(
        "{:?} : Loaded thread {thread_number} in {:?}",
        std::time::Instant::now(),
        start.elapsed()
    );

    loop {
        let (mut hidden_states, mut attention_mask, mut alibi, mut past_key_values, rqs) = rx
            .recv()
            .expect("You probably want to handle this case, but I'm too lazy");

        hidden_states = hidden_states.to_device(device);
        attention_mask = attention_mask.to_device(device);
        alibi = alibi.to_device(device);
        for (layer, layer_past) in layers
            .iter()
            .zip(past_key_values.iter_mut().skip(5 + 7 * 9))
        {
            debug(&format!("past_key thread3"), &layer_past.0);
            debug(&format!("past_values thread3"), &layer_past.1);
            hidden_states = layer.forward(&hidden_states, &attention_mask, &alibi, layer_past);
        }
        debug(&format!("last_hidden_states"), &hidden_states);
        hidden_states = ln_f.forward(&hidden_states);
        debug(&format!("After ln_f"), &hidden_states);
        let logits = lm_head.forward(&hidden_states);

        for (_i, rq) in rqs.into_iter().enumerate() {
            let simple_logits = logits.f_slice(0, 0, logits.size()[0], 1).unwrap();

            let p = N_HEAD as i64;
            let q = (HIDDEN_SIZE / N_HEAD) as i64;
            let kind = (kind::Kind::Half, device);
            let past_key = Tensor::zeros(&[1, 0, p, q], kind);
            let past_value = Tensor::zeros(&[1, 0, p, q], kind);
            let simple_past_key_values: Vec<_> = (0..N_LAYER)
                .map(|_| (past_key.copy(), past_value.copy()))
                .collect();
            rq.send((simple_logits, simple_past_key_values)).unwrap();
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

    let (tx, rx) = bounded::<Msg>(256);
    let (prio_tx, prio_rx) = unbounded::<Msg>();
    let (s0, r0) = bounded::<Msg2>(1);
    let (s1, r1) = bounded::<Msg2>(1);
    let (s2, r2) = bounded::<Msg2>(1);
    let (s3, r3) = bounded::<Msg2>(1);
    let (s4, r4) = bounded::<Msg2>(1);
    let (s5, r5) = bounded::<Msg2>(1);
    let (s6, r6) = bounded::<Msg2>(1);
    let (s7, r7) = bounded::<Msg2>(1);
    let (s8, r8) = bounded::<Msg2>(1);
    let (s9, r9) = bounded::<Msg2>(1);
    let (s10, r10) = bounded::<Msg2>(1);
    let (s11, r11) = bounded::<Msg2>(1);
    let (s12, r12) = bounded::<Msg2>(1);
    let (s13, r13) = bounded::<Msg2>(1);
    let (s14, r14) = bounded::<Msg2>(1);

    std::thread::spawn(move || {
        thread1(rx, prio_rx, s0, 0);
    });
    std::thread::spawn(move || {
        thread2(r0, s1, 1);
    });
    std::thread::spawn(move || {
        thread2(r1, s2, 2);
    });
    std::thread::spawn(move || {
        thread2(r2, s3, 3);
    });
    std::thread::spawn(move || {
        thread2(r3, s4, 4);
    });
    std::thread::spawn(move || {
        thread2(r4, s5, 5);
    });
    std::thread::spawn(move || {
        thread2(r5, s6, 6);
    });
    std::thread::spawn(move || {
        thread2(r6, s7, 7);
    });
    std::thread::spawn(move || {
        thread2(r7, s8, 8);
    });
    std::thread::spawn(move || {
        thread2(r8, s9, 9);
    });
    std::thread::spawn(move || {
        thread2(r9, s10, 10);
    });
    std::thread::spawn(move || {
        thread2(r10, s11, 11);
    });
    std::thread::spawn(move || {
        thread2(r11, s12, 12);
    });
    std::thread::spawn(move || {
        thread2(r12, s13, 13);
    });
    std::thread::spawn(move || {
        thread2(r13, s14, 14);
    });
    std::thread::spawn(move || {
        thread3(r14, 15);
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
        let device = Device::Cuda(0);
        let input_ids = Tensor::of_slice(&[3, 4, 5])
            .view((1, 3))
            .to_kind(kind::Kind::Int)
            .to_device(device);
        let input_ids2 = Tensor::of_slice(&[8, 1, 3, 4, 5, 6])
            .view((1, 6))
            .to_kind(kind::Kind::Int)
            .to_device(device);
        let past = vec![];
        let past2 = vec![];
        let (tx, rx) = bounded::<(Tensor, Past)>(10);

        let items = vec![(input_ids, past, tx.clone()), (input_ids2, past2, tx)];

        let (all_input_ids, _, _, _, _) = padding(items);

        assert_eq!(all_input_ids.size(), vec![2, 6]);
        assert_eq!(
            Vec::<i64>::from(all_input_ids),
            vec![2, 2, 2, 3, 4, 5, 8, 1, 3, 4, 5, 6]
        );
    }

    #[test]
    #[cfg(feature = "bloom-350m")]
    fn test_bloom_350m() {
        let file = std::fs::File::open("./bloom-350m.bin").unwrap();
        // SAFETY: This is actually unsafe.
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let model_file = SafeTensors::deserialize(&mmap).unwrap();

        let device = Device::Cuda(0);

        let model = BloomForCausalLM::new(&model_file, device);

        // Batched input (2, 4)
        // First input is not full
        let p = N_HEAD as i64;
        let q = (HIDDEN_SIZE / N_HEAD) as i64;
        let kind = (kind::Kind::Half, device);
        let past_key = Tensor::zeros(&[1, 0, p, q], kind);
        let past_value = Tensor::zeros(&[1, 0, p, q], kind);
        let mut past_key_values: Vec<_> = (0..N_LAYER)
            .map(|_| (past_key.copy(), past_value.copy()))
            .collect();
        let input_ids = Tensor::of_slice(&[2, 2, 34, 54, 132, 225, 532, 342])
            .view((2, 4))
            .to_device(Device::Cuda(0));
        let attention_mask = Tensor::of_slice(&[0, 0, 1, 1, 1, 1, 1, 1])
            .view((2, 4))
            .to_device(Device::Cuda(0));

        let alibi = build_alibi_tensor(&attention_mask, N_HEAD, device);

        let logits = model.forward(&input_ids, &attention_mask, &alibi, &mut past_key_values);

        assert_eq!(logits.size(), vec![2, 4, 250880]);
        assert_eq!(
            Vec::<f64>::from(logits.copy())
                .into_iter()
                .take(10)
                .collect::<Vec<_>>(),
            vec![640.5, 668.5, 671.0, 102.375, 670.5, 676.5, 680.5, 676.0, 671.0, 670.0]
        );
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

    #[test]
    #[cfg(feature = "bloom-350m")]
    fn test_alibi() {
        let device = Device::Cuda(0);
        let attention_mask = Tensor::of_slice(&[0, 0, 1, 1, 1, 1, 1, 1])
            .view((2, 4))
            .to_device(device);

        assert_eq!(attention_mask.size(), vec![2, 4]);
        let alibi = build_alibi_tensor(&attention_mask, N_HEAD, device);
        assert_eq!(alibi.size(), vec![32, 1, 4]);
        assert_eq!(
            Vec::<f64>::from(alibi)
                .into_iter()
                .take(8)
                .collect::<Vec<_>>(),
            vec![-0.0, -0.0, 0.0, 0.70703125, -0.0, -0.0, 0.0, 0.5]
        );
    }
}
