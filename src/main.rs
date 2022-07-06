#![feature(async_closure)]
use actix_web::middleware::Logger;
use actix_web::{post, web, App, HttpServer, Responder};
use memmap::MmapOptions;
use safetensors::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tch::{kind, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

const PADDING_IDX: i64 = 2;
const EPS: f64 = 1e-5;
const N_HEAD: usize = 16;
const HIDDEN_SIZE: usize = 1024;
const N_LAYER: usize = 24;

type PastLayer = (Tensor, Tensor);
type Past = Vec<PastLayer>;

#[derive(Deserialize, Serialize)]
struct Parameters {
    top_k: Option<usize>,
}

impl Default for Parameters {
    fn default() -> Self {
        Self { top_k: None }
    }
}

#[derive(Deserialize, Serialize)]
struct Generation {
    inputs: String,
    #[serde(default)]
    parameters: Parameters,
}

#[post("/generate")]
//async fn generate(payload: web::Json<Generation>, state: web::Data<AppState>) -> impl Responder {
async fn generate(payload: web::Json<Generation>, state: web::Data<AppState>) -> impl Responder {
    let inputs = &payload.inputs;

    let encoded = state.tokenizer.encode(inputs.clone(), false).unwrap();
    let ids = encoded.get_ids();
    let n = ids.len();
    let kind = (kind::Kind::Int, Device::Cuda(0));
    let t = Tensor::zeros(&[1, n as i64], kind);
    // println!("Ids {ids:?}");
    for (i, id) in ids.into_iter().enumerate() {
        let _ = t.i((0, i as i64)).fill_((*id) as i64);
    }
    // t.print();
    let (tx, rx) = std::sync::mpsc::channel::<Tensor>();

    let past_key = Tensor::zeros(&[1, 0, 16, 64], kind);
    let past_value = Tensor::zeros(&[1, 0, 16, 64], kind);
    let past_key_values: Vec<_> = (0..N_LAYER)
        .map(|_| (past_key.copy(), past_value.copy()))
        .collect();
    state
        .in_channel
        .send((t, past_key_values, tx))
        .expect("Can send");

    let received = rx.recv().unwrap();
    let received_ids: Vec<i64> = received.try_into().unwrap();
    let received_ids: Vec<u32> = received_ids.into_iter().map(|i| i as u32).collect();

    let string = state.tokenizer.decode(received_ids, false).unwrap();
    format!("Hello {string}!")
}

#[derive(Clone)]
struct AppState {
    in_channel: std::sync::mpsc::Sender<(Tensor, Past, std::sync::mpsc::Sender<Tensor>)>,
    tokenizer: Arc<Tokenizer>,
}

fn convert(view: TensorView, device: Device) -> Tensor {
    let kind = match view.get_dtype() {
        Dtype::F16 => kind::Kind::Half,
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

fn build_alibi_tensor(max_seq_len: usize, n_head: usize) -> Tensor {
    let slopes = get_slopes(n_head);
    let device = Device::Cuda(0);
    let kind = kind::Kind::Half;
    let slopes = Tensor::of_slice(&slopes)
        .unsqueeze(1)
        .unsqueeze(1)
        .to_kind(kind)
        .to_device(device);
    let arange_tensor = Tensor::arange(max_seq_len as i64, (kind, device))
        .unsqueeze(0)
        .unsqueeze(0);
    let alibi = slopes
        * arange_tensor
            .f_expand(&[n_head as i64, -1, -1], true)
            .unwrap();

    return alibi;
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn new(name: &str, model: &SafeTensors<'_>) -> Self {
        let weight = convert(
            model.tensor(&format!("{name}.weight")).unwrap(),
            Device::Cuda(0),
        );
        let bias = convert(
            model.tensor(&format!("{name}.weight")).unwrap(),
            Device::Cuda(0),
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
        xs.f_layer_norm(&[1024], Some(&self.weight), Some(&self.bias), EPS, true)
            .unwrap()
    }
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn new(name: &str, model: &SafeTensors<'_>) -> Self {
        let weight = convert(
            model.tensor(&format!("{name}.weight")).unwrap(),
            Device::Cuda(0),
        );
        let name = format!("{name}.bias");
        let bias = convert(model.tensor(&name).unwrap(), Device::Cuda(0));
        Self { weight, bias }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.f_linear(&self.weight, Some(&self.bias)).unwrap()
    }
}

struct BloomScaledSoftmax {}

impl BloomScaledSoftmax {
    fn new() -> Self {
        Self {}
    }

    fn forward(&self, input: &Tensor, _max_positions: i64) -> Tensor {
        // TODO actually implement this
        input.copy()
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
    fn new(name: &str, model: &SafeTensors<'_>, layer_number: usize) -> Self {
        let dense = Linear::new(&format!("{name}.dense"), model);
        let query_key_value = Linear::new(&format!("{name}.query_key_value"), model);
        let head_dim = (HIDDEN_SIZE / N_HEAD) as i64;
        let num_attention_heads = N_HEAD as i64;
        let layer_number = std::cmp::max(1, layer_number);
        let norm_factor = (head_dim as f64).sqrt() * layer_number as f64;
        let scaled_softmax = BloomScaledSoftmax::new();
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
        layer_past: &mut PastLayer,
        alibi: &Tensor,
    ) -> Tensor {
        let mut mixed_x_layer = self.query_key_value.forward(hidden_states);
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
        let (past_key, past_value) = layer_past;

        let mut key_layer = Tensor::f_cat(&[past_key.as_ref(), key], 1).unwrap();
        let mut value_layer = Tensor::f_cat(&[past_value.as_ref(), value], 1).unwrap();

        debug(&format!("Key layer {layer_number}"), &key_layer);
        debug(&format!("Value layer {layer_number}"), &value_layer);

        // Update past for next loops
        *layer_past = (key_layer.copy(), value_layer.copy());

        // [batch_size, head_dim, q_length, k_length, num_heads]
        let B = query.size()[0];
        let H = query.size()[2];
        let Q = query.size()[1];
        let K = key_layer.size()[1];
        // let NH = value_layer.size()[2];
        let NH = 64;
        let output_size = (B, H, Q, K);

        let query_layer = query.transpose(1, 0).reshape(&[Q, B * H, -1]);
        key_layer = key_layer.transpose(1, 0).reshape(&[K, B * H, -1]);

        // let sliced_alibi = alibi[: output_size[0] * output_size[1], :, : output_size[3]]
        let sliced_alibi = alibi.slice(0, 0, B * H, 1).slice(2, 0, K, 1);
        let beta = 1.0 / (self.layer_number as f64);
        let alpha = 1.0 / self.norm_factor;
        // TODO Check why no alpha,beta in this operator.
        let b = beta * sliced_alibi;
        let a = alpha * query_layer.transpose(1, 0);
        debug(&format!("b {layer_number}"), &b);
        debug(&format!("a {layer_number}"), &a);
        let matmul_result = b
            .f_baddbmm(
                &a,
                &key_layer.transpose(1, 0).transpose(1, 2),
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
        let attention_probs = self
            .scaled_softmax
            .forward(&attention_scores, max_positions);
        value_layer = value.transpose(1, 0).reshape(&[K, B * H, -1]);
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
    return x * 0.5 * (1.0 + (y).tanh());
}

impl BloomMlp {
    fn new(name: &str, model: &SafeTensors<'_>) -> Self {
        let dense_h_to_4h = Linear::new(&format!("{name}.dense_h_to_4h"), model);
        let dense_4h_to_h = Linear::new(&format!("{name}.dense_4h_to_h"), model);
        Self {
            dense_h_to_4h,
            dense_4h_to_h,
        }
    }

    fn forward(&self, hidden_states: &Tensor, _residual: &Tensor) -> Tensor {
        let hidden_states = self.dense_h_to_4h.forward(hidden_states);
        let hidden_states = bloom_gelu(&hidden_states);
        let hidden_states = self.dense_4h_to_h.forward(&hidden_states);
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
    fn new(prefix: &str, model: &SafeTensors<'_>, layer_number: usize) -> Self {
        // attention
        let input_layernorm = LayerNorm::new(&format!("{prefix}.input_layernorm"), model);
        let post_attention_layernorm =
            LayerNorm::new(&format!("{prefix}.post_attention_layernorm"), model);
        let self_attention =
            BloomAttention::new(&format!("{prefix}.self_attention"), model, layer_number);
        let mlp = BloomMlp::new(&format!("{prefix}.mlp"), model);
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
        layer_past: &mut PastLayer,
        alibi: &Tensor,
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
        let attention_output =
            self.self_attention
                .forward(&layernorm_output, residual, layer_past, alibi);
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
    fn new(name: &str, model: &SafeTensors<'_>) -> Self {
        let weight = convert(
            model.tensor(&format!("{name}.weight")).unwrap(),
            Device::Cuda(0),
        );
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
    //     "{prefix} - Values: {:?}",
    //     x.reshape(&[-1,])
    //         .iter::<f64>()
    //         .unwrap()
    //         .take(5)
    //         .collect::<Vec<_>>()
    // );
}

struct BloomModel {
    word_embeddings: Embedding,
    word_embeddings_layernorm: LayerNorm,
    h: Vec<BloomBlock>,
    ln_f: LayerNorm,
}

impl BloomModel {
    fn new(model: &SafeTensors<'_>) -> Self {
        let word_embeddings = Embedding::new(&format!("word_embeddings"), model);
        let word_embeddings_layernorm =
            LayerNorm::new(&format!("word_embeddings_layernorm"), model);
        let ln_f = LayerNorm::new(&format!("ln_f"), model);
        let h = (0..24)
            .map(|i| BloomBlock::new(&format!("h.{i}"), model, i))
            .collect();
        Self {
            word_embeddings,
            word_embeddings_layernorm,
            h,
            ln_f,
        }
    }

    fn forward(&self, input_ids: &Tensor, past_key_values: &mut Past) -> Tensor {
        let n_head = N_HEAD;
        let inputs_embeds = self.word_embeddings.forward(input_ids);
        let mut hidden_states = self.word_embeddings_layernorm.forward(&inputs_embeds);

        debug("First layer norm", &hidden_states);

        let current_sequence_length =
            (hidden_states.size()[1] + past_key_values[0].0.size()[1]) as usize;
        let alibi = build_alibi_tensor(current_sequence_length, n_head);

        debug("Alibi", &alibi);

        for (i, (block, layer_past)) in self.h.iter().zip(past_key_values.iter_mut()).enumerate() {
            hidden_states = block.forward(
                &hidden_states,
                layer_past,
                &alibi,
                // head_mask=head_mask[i],
            );
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
    fn new(name: &str, model: &SafeTensors<'_>) -> Self {
        let weight = convert(
            model.tensor(&format!("{name}.weight")).unwrap(),
            Device::Cuda(0),
        );
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
    fn new(model: &SafeTensors<'_>) -> Self {
        let transformer = BloomModel::new(model);
        let lm_head = InvertedEmbedding::new("word_embeddings", model);
        Self {
            transformer,
            lm_head,
        }
    }

    fn forward(&self, input_ids: &Tensor, past_key_values: &mut Past) -> Tensor {
        let hidden_states = self.transformer.forward(input_ids, past_key_values);
        let lm_logits = self.lm_head.forward(&hidden_states);
        lm_logits
    }
}

#[actix_web::main] // or #[tokio::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    tch::maybe_init_cuda();

    let start = std::time::Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file("./tokenizer.json").unwrap());
    println!("Loaded tokenizer in {:?}", start.elapsed());

    let (tx, rx) = std::sync::mpsc::channel::<(Tensor, Past, std::sync::mpsc::Sender<Tensor>)>();

    std::thread::spawn(move || {
        // println!("Start thread");
        let file = std::fs::File::open("./out_bloom-350m.bin").unwrap();
        // SAFETY: This is actually unsafe.
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let weights = SafeTensors::deserialize(&mmap).unwrap();

        let names = weights.names();
        println!("Metadata {names:?}");
        let start = std::time::Instant::now();
        let bloom = BloomForCausalLM::new(&weights);
        println!("Loaded model in {:?}", start.elapsed());

        loop {
            let (input_ids, mut past_key_values, rq) = rx
                .recv()
                .expect("You probably want to handle this case, but I'm too lazy");

            let start = std::time::Instant::now();
            let logits = bloom.forward(&input_ids, &mut past_key_values);

            // println!("Logits {:?}", logits);
            let S = logits.size()[1];
            let new_id = logits.slice(1, S - 1, S, 1).argmax(-1, false);
            // new_id.print();
            // println!("New id {:?}", new_id);

            println!("Inference ran in {:?}", start.elapsed());
            rq.send(new_id).unwrap();
        }
    });

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(web::Data::new(AppState {
                tokenizer: tokenizer.clone(),
                in_channel: tx.clone(),
            }))
            .service(generate)
    })
    .bind(("127.0.0.1", 8001))?
    .run()
    .await
}
