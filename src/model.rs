use crate::convert;
use crate::utils::{debug, save_layer_to_disk};
use safetensors::SafeTensors;
use tch::{kind::Kind, Device, IndexOp, Tensor};

pub type PastLayer = (Tensor, Tensor);
pub type Past = Vec<PastLayer>;

const PADDING_IDX: i64 = 3;
const EPS: f64 = 1e-5;

#[derive(Clone)]
pub struct Config {
    pub n_head: i64,
    pub hidden_size: i64,
    pub n_layer: i64,
    pub kind: Kind,
    pub slow_but_exact: bool,
    pub pretraining_tp: usize,
    pub padding_idx: i64,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    pub fn new() -> Self {
        Self {
            n_head: 112,
            hidden_size: 14336,
            n_layer: 70,
            kind: Kind::BFloat16,
            slow_but_exact: false,
            pretraining_tp: 4,
            padding_idx: PADDING_IDX,
        }
    }
    pub fn new350m() -> Self {
        Self {
            n_head: 16,
            hidden_size: 1024,
            n_layer: 24,
            kind: Kind::Half,
            slow_but_exact: false,
            pretraining_tp: 1,
            padding_idx: PADDING_IDX,
        }
    }

    pub fn new_testing() -> Self {
        Self {
            n_head: 8,
            hidden_size: 64,
            n_layer: 2,
            kind: Kind::BFloat16,
            slow_but_exact: true,
            pretraining_tp: 2,
            padding_idx: PADDING_IDX,
        }
    }
}
pub fn get_slopes_power_of_2(n: usize) -> Vec<f64> {
    let start = 2.0f64.powf(-(2.0f64.powf(-((n as f64).log2() - 3.0f64))));
    let ratio = start;
    (0..n).map(|i| start * ratio.powi(i as i32)).collect()
}

fn next_pow2(mut x: usize) -> usize {
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    (x + 1) >> 1
}

fn get_slopes(n: usize) -> Vec<f64> {
    let is_power_of_2 = n == 0 || n & (n - 1) == 0;
    // println!("Slopes {:?}", n);
    // println!("Is power {:?}", is_power_of_2);
    // println!("Is power {:?}", n & (n - 1));
    if is_power_of_2 {
        get_slopes_power_of_2(n)
    } else {
        let closest_power_of_2 = next_pow2(n);
        // println!("Closest power of 2 {:?}", closest_power_of_2);
        get_slopes_power_of_2(closest_power_of_2)
            .into_iter()
            .chain(
                get_slopes(closest_power_of_2 << 1)
                    .into_iter()
                    .step_by(2)
                    .take(n - closest_power_of_2),
            )
            .collect()
    }
}

pub fn build_alibi_tensor(
    attention_mask: &Tensor,
    n_head: i64,
    kind: Kind,
    device: Device,
) -> Tensor {
    let slopes = get_slopes(n_head as usize);
    //println!("Slopes {:?}", slopes);

    // IMPORTANT, use f32 for this tensor
    let slopes = Tensor::of_slice(&slopes)
        .to_kind(Kind::Float)
        .to_device(device);
    // debug("slopes", &slopes);
    let a = attention_mask
        .f_cumsum(-1, Kind::Float)
        .unwrap()
        .unsqueeze(1)
        - 1;
    // debug("A", &A);
    let b = attention_mask.unsqueeze(1);
    // debug("B", &B);
    let arange_tensor = a * b;
    // debug("A range tensor", &arange_tensor);
    let slopes = slopes.unsqueeze(-1);
    // debug("Slopes", &slopes);
    let mut alibi = slopes * arange_tensor;
    // debug("alibi1 ", &alibi);
    alibi *= attention_mask.unsqueeze(1);
    // debug("alibi2 ", &alibi);

    let size = attention_mask.size();
    let batch_size = size[0];
    let seq_length = size[1];
    alibi = alibi.reshape(&[batch_size * n_head, 1, seq_length]);

    // Cast back to what model expects
    alibi = alibi.to_kind(kind);
    alibi
}

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    hidden_size: i64,
}

impl LayerNorm {
    pub fn new(hidden_size: i64, name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let weight_name = format!("{name}.weight");
        let weight = convert(
            model
                .tensor(&weight_name)
                .unwrap_or_else(|_| panic!("Failed to load {weight_name} with name {name}")),
            device,
        );
        let bias_name = format!("{name}.bias");
        let bias = convert(
            model
                .tensor(&bias_name)
                .unwrap_or_else(|_| panic!("Failed to load {bias_name}")),
            device,
        );
        Self {
            hidden_size,
            weight,
            bias,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
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

pub struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    pub fn new(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let tname = format!("{name}.weight");
        let weight = convert(
            model
                .tensor(&tname)
                .unwrap_or_else(|_| panic!("Could not find {tname}")),
            device,
        );

        let bias_name = format!("{name}.bias");
        let bias = convert(
            model
                .tensor(&bias_name)
                .unwrap_or_else(|_| panic!("Could not find {bias_name}")),
            device,
        );

        Self { weight, bias }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
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
    let attention_mask_bool = attention_mask.to_kind(Kind::Bool).f_logical_not().unwrap();
    debug("not attention_mask", &attention_mask_bool);

    let query_length = attention_scores.size()[2];
    let key_length = attention_scores.size()[3];
    let n_heads = attention_scores.size()[1];

    let a_attention_mask_bool = attention_mask_bool.unsqueeze(1).unsqueeze(-1);
    let a_attention_mask_bool =
        a_attention_mask_bool.i((.., .., key_length - query_length..key_length));
    let b_causal_mask = causal_mask.f_logical_not().unwrap();

    let b_causal_mask = b_causal_mask.i((.., .., key_length - query_length..key_length));
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
    debug("Masked fill", masked_fill);
    let masked_output = attention_scores
        .f_masked_fill_(masked_fill, -10000.0)
        .unwrap();
    debug("Masked output", &masked_output);
    (masked_output, padded_causal_mask)
}
pub struct BloomScaledSoftmax {
    scale: i64,
    kind: Kind,
}

impl BloomScaledSoftmax {
    pub fn new(scale: i64, kind: Kind) -> Self {
        Self { scale, kind }
    }

    pub fn forward(&self, input: &Tensor, attention_mask: &Tensor, max_positions: i64) -> Tensor {
        debug("input", input);
        let mut scaled_input = self.scale * input;
        debug("scaled input", &scaled_input);
        let seq_ids = Tensor::f_arange(max_positions, (Kind::Int64, input.device())).unwrap();

        let a = seq_ids.unsqueeze(0);
        let b = seq_ids.unsqueeze(-1);
        let causal_mask = a.f_le_tensor(&b).unwrap().to_kind(Kind::Bool).view((
            1,
            1,
            max_positions,
            max_positions,
        ));

        debug("Causal mask", &causal_mask);

        // TODO Padded causal mask
        let (mask_output, padded_causal_mask) =
            attention_mask_func(&mut scaled_input, attention_mask, &causal_mask);
        debug("mask output", &mask_output);

        // TODO dtype float16 ?
        let probs = mask_output.f_softmax(-1, Kind::Float).unwrap()
            * padded_causal_mask.f_logical_not().unwrap();
        debug("Probs", &probs);

        let out_probs = probs.to_kind(self.kind);
        debug("Out Probs", &out_probs);

        out_probs
    }
}

pub struct BloomAttention {
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
    pub fn new(
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

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual: &Tensor,
        attention_mask: &Tensor,
        alibi: &Tensor,
        layer_past: &mut PastLayer,
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
        let (past_key, past_value) = layer_past;

        let device = key.device();

        println!("Key layer {:?}", key.size());
        let mut key_layer =
            Tensor::f_cat(&[past_key.as_ref().to_device(device), key.copy()], 1).unwrap();
        let value_layer =
            Tensor::f_cat(&[past_value.as_ref().to_device(device), value.copy()], 1).unwrap();
        println!("Key layer after {:?}", key_layer.size());

        // Update past for next loops
        *layer_past = (key_layer.copy(), value_layer.copy());

        let batch_size = query.size()[0];
        let n_head = query.size()[2];
        let query_length = query.size()[1];
        let key_length = key_layer.size()[1];
        let hidden_per_head = self.hidden_size / self.n_head;
        let output_size = (batch_size, n_head, query_length, key_length);

        let query_layer = query
            .transpose(1, 0)
            .reshape(&[query_length, batch_size * n_head, -1]);
        key_layer = key_layer
            .transpose(1, 0)
            .reshape(&[key_length, batch_size * n_head, -1]);

        // let sliced_alibi = alibi[: output_size[0] * output_size[1], :, : output_size[3]]
        println!("Alibi {:?}", alibi.size());
        println!("Key length {:?}", key_length);
        let sliced_alibi = alibi.i((0..batch_size * n_head, .., 0..key_length));
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
        let value_layer =
            value_layer
                .transpose(1, 0)
                .reshape(&[key_length, batch_size * n_head, -1]);
        let attention_probs_reshaped = attention_probs
            .f_view((batch_size * n_head, query_length, -1))
            .unwrap();
        let bmm = Tensor::bmm(&attention_probs_reshaped, &value_layer.transpose(0, 1));
        save_layer_to_disk(
            &bmm,
            &format!("rust_bmm_context_layer_{}.npy", self.real_layer_number,),
        );
        debug("Bmm", &bmm);
        let context_layer_r = bmm.f_view((batch_size, n_head, query_length, -1)).unwrap();
        let context_layer_p = context_layer_r.permute(&[2, 0, 1, 3]).contiguous();
        let context = context_layer_p
            .f_view((query_length, batch_size, n_head * hidden_per_head))
            .unwrap();
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
            residual,
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

pub struct BloomMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    real_layer_number: usize,
    slow_but_exact: bool,
    pretraining_tp: i64,
}

fn bloom_gelu(x: &Tensor) -> Tensor {
    let y: Tensor = 0.79788456 * x * (1.0 + 0.044715 * x * x);
    x * 0.5 * (1.0 + y.tanh())
}

impl BloomMlp {
    pub fn new(
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

    pub fn forward(&self, hidden_states: &Tensor, residual: &Tensor) -> Tensor {
        save_layer_to_disk(
            hidden_states,
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
            let mut intermediate_output = Tensor::zeros_like(residual);
            let slices = self.dense_4h_to_h.weight.size().last().unwrap() / self.pretraining_tp;
            for i in 0..self.pretraining_tp {
                let i = i as i64;
                let tp = hidden_states.i((.., .., i * slices..(i + 1) * slices));
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

pub struct BloomBlock {
    input_layernorm: LayerNorm,
    self_attention: BloomAttention,
    post_attention_layernorm: LayerNorm,
    mlp: BloomMlp,
    layer_number: usize,
}

impl BloomBlock {
    pub fn new(
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

    pub fn forward(
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

pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    pub fn new(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let weight = convert(model.tensor(&format!("{name}.weight")).unwrap(), device);
        Self { weight }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        debug("Embedding weights", &self.weight);
        let result = Tensor::embedding(&self.weight, xs, PADDING_IDX, false, false);
        debug("Embedding", &result);
        result
    }
}

pub struct BloomModel {
    pub word_embeddings: Embedding,
    pub word_embeddings_layernorm: LayerNorm,
    pub h: Vec<BloomBlock>,
    pub ln_f: LayerNorm,
}

impl BloomModel {
    pub fn new(config: &Config, model: &SafeTensors<'_>, device: Device) -> Self {
        let word_embeddings = Embedding::new("word_embeddings", model, device);
        let word_embeddings_layernorm = LayerNorm::new(
            config.hidden_size,
            "word_embeddings_layernorm",
            model,
            device,
        );
        let ln_f = LayerNorm::new(config.hidden_size, "ln_f", model, device);
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

        debug("Alibi", alibi);

        for (i, (block, layer_past)) in self.h.iter().zip(past_key_values.iter_mut()).enumerate() {
            hidden_states = block.forward(&hidden_states, attention_mask, alibi, layer_past);
            debug(&format!("Block {i}"), &hidden_states);
        }
        hidden_states = self.ln_f.forward(&hidden_states);
        debug("ln_f", &hidden_states);
        hidden_states
    }
}

pub struct InvertedEmbedding {
    weight: Tensor,
}

impl InvertedEmbedding {
    pub fn new(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let weight = convert(model.tensor(&format!("{name}.weight")).unwrap(), device);
        Self { weight }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
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

pub struct BloomForCausalLM {
    pub transformer: BloomModel,
    pub lm_head: InvertedEmbedding,
}

impl BloomForCausalLM {
    pub fn new(config: &Config, model: &SafeTensors<'_>, device: Device) -> Self {
        let transformer = BloomModel::new(config, model, device);
        let lm_head = InvertedEmbedding::new("word_embeddings", model, device);
        Self {
            transformer,
            lm_head,
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        alibi: &Tensor,
        past_key_values: &mut Past,
    ) -> Tensor {
        let hidden_states =
            self.transformer
                .forward(input_ids, attention_mask, alibi, past_key_values);
        self.lm_head.forward(&hidden_states)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::test::assert_all_close;
    use memmap::MmapOptions;
    use once_cell::sync::Lazy;
    use std::sync::{Arc, Mutex};

    pub static BLOOM_350M: Lazy<Arc<Mutex<BloomForCausalLM>>> =
        Lazy::new(|| Arc::new(Mutex::new(bloom_350m())));
    pub static BLOOM_TESTING: Lazy<Arc<Mutex<BloomForCausalLM>>> =
        Lazy::new(|| Arc::new(Mutex::new(bloom_testing())));

    fn bloom_350m() -> BloomForCausalLM {
        let config = Config::new350m();
        let file = std::fs::File::open("./weights/bloom-350m.bin").unwrap();
        // SAFETY: This is actually unsafe.
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let model_file = SafeTensors::deserialize(&mmap).unwrap();

        let device = Device::Cuda(0);

        let model = BloomForCausalLM::new(&config, &model_file, device);
        model
    }

    fn bloom_testing() -> BloomForCausalLM {
        let config = Config::new_testing();
        let file = std::fs::File::open("./weights/bloom-testing.bin").unwrap();
        // SAFETY: This is actually unsafe.
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let model_file = SafeTensors::deserialize(&mmap).unwrap();

        let device = Device::Cuda(0);

        let model = BloomForCausalLM::new(&config, &model_file, device);
        model
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
        let kind = Kind::BFloat16;
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

    #[test]
    fn test_alibi3() {
        // let config = Config::new();
        let device = Device::Cuda(0);
        let kind = Kind::BFloat16;
        let n_head = 112;
        let attention_mask = Tensor::of_slice(&[1, 1, 1, 1, 1, 1, 1])
            .view((1, 7))
            .to_device(device);

        assert_eq!(attention_mask.size(), vec![1, 7]);
        let alibi = build_alibi_tensor(&attention_mask, n_head, kind, device);
        assert_eq!(alibi.size(), vec![112, 1, 7]);
        assert_eq!(
            Vec::<f64>::from(alibi.i(0)).into_iter().collect::<Vec<_>>(),
            vec![0., 0.91796875, 1.8359375, 2.75, 3.671875, 4.59375, 5.5]
        );
        assert_eq!(
            Vec::<f64>::from(alibi.i(1)).into_iter().collect::<Vec<_>>(),
            vec![0., 0.83984375, 1.6796875, 2.515625, 3.359375, 4.21875, 5.03125]
        );
    }

    #[test]
    fn test_alibi4() {
        // let config = Config::new();
        let device = Device::Cuda(0);
        let kind = Kind::BFloat16;
        let n_head = 112;
        let attention_mask = Tensor::of_slice(vec![1; 385].as_slice())
            .view((1, 385))
            .to_device(device);

        assert_eq!(attention_mask.size(), vec![1, 385]);
        let alibi = build_alibi_tensor(&attention_mask, n_head, kind, device);
        assert_eq!(alibi.size(), vec![112, 1, 385]);
        assert_eq!(
            Vec::<f64>::from(alibi.i(0))
                .into_iter()
                .take(10)
                .collect::<Vec<_>>(),
            vec![0., 0.91796875, 1.8359375, 2.75, 3.671875, 4.59375, 5.5, 6.40625, 7.34375, 8.25]
        );
        // Still first row but towards the end
        assert_eq!(
            Vec::<f64>::from(alibi.i(0))
                .into_iter()
                .skip(375)
                .collect::<Vec<_>>(),
            vec![344., 344., 346., 346., 348., 348., 350., 350., 352., 352.]
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
        let past_key = Tensor::zeros(&[2, 0, p, q], kind);
        let past_value = Tensor::zeros(&[2, 0, p, q], kind);
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
}
