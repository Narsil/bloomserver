use crate::convert;
use crate::utils::{debug, save_layer_to_disk};
use safetensors::SafeTensors;
use tch::{kind::Kind, Device, IndexOp, Tensor};

#[derive(Debug)]
pub struct PastLayer {
    pub key: Tensor,
    pub value: Tensor,
}

impl PastLayer {
    pub fn seq_length(&self) -> i64 {
        return self.key.size()[2];
    }
}

pub fn empty_past(config: &Config, batch_size: i64) -> Past {
    non_empty_past(config, batch_size, 0, 0.0, 0.0)
}

pub fn non_empty_past(
    config: &Config,
    batch_size: i64,
    length_past: i64,
    key: f64,
    value: f64,
) -> Past {
    let device = Device::Cuda(0);
    let p = config.n_head;
    let q = config.hidden_size / config.n_head;
    let past_key_template =
        Tensor::zeros(&[batch_size * p, q, length_past], (config.kind, device)) + key;
    let past_value_template =
        Tensor::zeros(&[batch_size * p, length_past, q], (config.kind, device)) + value;
    let all_past_key_values = (0..config.n_layer as usize)
        .map(|_| PastLayer {
            key: past_key_template.copy(),
            value: past_value_template.copy(),
        })
        .collect::<Vec<_>>();
    all_past_key_values
}
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
    if is_power_of_2 {
        get_slopes_power_of_2(n)
    } else {
        let closest_power_of_2 = next_pow2(n);
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

/// Make causal mask used for self-attention.
fn make_causal_mask(
    input_ids_size: Vec<i64>,
    device: Device,
    past_key_values_length: i64,
) -> Tensor {
    // batch_size, target_length = input_ids_shape
    let batch_size = input_ids_size[0];
    let target_length = input_ids_size[1];
    // mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    let mask = Tensor::zeros(
        &[target_length, target_length + past_key_values_length],
        (Kind::Bool, device),
    );
    // # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    // seq_ids = torch.arange(target_length, device=device)
    let seq_ids = Tensor::arange(target_length, (Kind::Int, device));
    // mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]
    let causal_mask = seq_ids
        .unsqueeze(1)
        .f_lt_tensor(&seq_ids.unsqueeze(0))
        .unwrap();
    _ = mask
        .i((.., past_key_values_length..))
        .f_copy_(&causal_mask)
        .unwrap();

    // if past_key_values_length > 0:
    //     mask[:, :past_key_values_length] = False
    if past_key_values_length > 0 {
        _ = mask.i((.., ..past_key_values_length)).f_fill_(0).unwrap();
    }

    // expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    let expanded_mask = mask
        .unsqueeze(0)
        .f_expand(
            &[
                batch_size,
                target_length,
                target_length + past_key_values_length,
            ],
            true,
        )
        .unwrap();
    // return expanded_mask
    expanded_mask
}

/// Expands attention_mask from `[batch_size, src_length]` to `[batch_size, tgt_length, src_length]`.
fn expand_mask(mask: &Tensor, tgt_length: i64) -> Tensor {
    // batch_size, src_length = mask.shape
    let batch_size = mask.size()[0];
    let src_length = mask.size()[1];
    // tgt_length = tgt_length if tgt_length is not None else src_length

    // expanded_mask = ~(mask[:, None, :].to(torch.bool))
    let expanded_mask = mask
        .unsqueeze(1)
        .to_kind(Kind::Bool)
        // @TODO @thomas21: Make in-place
        .f_logical_not()
        .unwrap();
    // return expanded_mask.expand(batch_size, 1, tgt_length, src_length)
    expanded_mask
        .f_expand(&[batch_size, tgt_length, src_length], true)
        .unwrap()
}

pub fn prepare_attn_mask(
    attention_mask: &Tensor,
    input_size: Vec<i64>,
    past_key_values_length: i64,
    num_attention_heads: i64,
) -> Tensor {
    let device = attention_mask.device();
    // _, src_length = input_shape
    let batch_size = input_size[0];
    let src_length = input_size[1];

    // [batch_size, seq_length] -> [batch_size, tgt_length, src_length]
    // expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
    let expanded_attn_mask = expand_mask(attention_mask, src_length);
    // combined_attention_mask = (
    //     expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
    // )
    let combined_attention_mask = if src_length > 1 {
        let causal_mask = make_causal_mask(input_size, device, past_key_values_length);
        expanded_attn_mask.f_logical_or(&causal_mask).unwrap()
    } else {
        expanded_attn_mask
    };

    // [batch_size, seq_length] -> [batch_size * num_attention_heads, tgt_length, src_length]
    let result = combined_attention_mask
        .f_repeat_interleave_self_int(num_attention_heads, 0, batch_size * num_attention_heads)
        .unwrap();

    result
}

pub fn build_alibi_tensor(
    attention_mask: &Tensor,
    n_head: i64,
    kind: Kind,
    device: Device,
) -> Tensor {
    let slopes = get_slopes(n_head as usize);

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

fn finfo_min(kind: Kind) -> f64 {
    match kind {
        Kind::Float => f32::MIN as f64,
        Kind::Half => -65504.0,
        Kind::BFloat16 => -3.3028235e38, // TODO use real bfloat16 min if possible.
        _ => todo!(),
    }
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

/// Different from usual Linear in order to remove transpose operation:
/// weight: [in_features, out_features]
/// bias: [out_features]
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
        ).f_transpose_copy(1,0).unwrap();

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
        let in_features = self.weight.size()[0];
        let out_features = self.weight.size()[1];
        let mut out_size = xs.size();
        if let Some(last) = out_size.last_mut() {
            *last = out_features;
        }

        let flatten_xs = xs.view((-1, in_features));

        self.bias
            .f_addbmm(&flatten_xs, &self.weight)
            .unwrap()
            .f_view(out_size.as_slice())
            .unwrap()
    }
}

pub struct BloomAttention {
    dense: Linear,
    query_key_value: Linear,
    num_attention_heads: i64,
    head_dim: i64,
    layer_number: usize,
    real_layer_number: usize,
    norm_factor: f64,
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
        let norm_factor = 1.0 / (head_dim as f64).sqrt();
        Self {
            dense,
            query_key_value,
            num_attention_heads,
            head_dim,
            layer_number,
            real_layer_number,
            norm_factor,
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
        let mixed_x_layer = self.query_key_value.forward(hidden_states);
        let (batch_size, q_length) = if let [batch_size, q_length,] = mixed_x_layer.size()[..2] { (batch_size, q_length) } else { todo!() };
        debug(&format!("Mixed_x_layer {layer_number}"), &mixed_x_layer);
        let new_tensor_shape = [
            batch_size,
            q_length,
            self.num_attention_heads,
            3 * self.head_dim,
        ];
        let mixed_x_layer_view = mixed_x_layer.f_view(new_tensor_shape).unwrap();
        // Split.
        let tensor_list = mixed_x_layer_view.split(self.head_dim, 3);
        let (query, key, value) = match &tensor_list[..] {
            [query, key, value] => (query, key, value),
            _ => unreachable!(),
        };

        let query_layer = query
            .transpose(1, 2)
            .reshape(&[batch_size * self.num_attention_heads, q_length, self.head_dim]);
        let key_layer = key
            .permute(&[0, 2, 3, 1])
            .reshape(&[batch_size * self.num_attention_heads, self.head_dim, q_length]);
        let value_layer = value
            .transpose(1, 2)
            .reshape(&[batch_size * self.num_attention_heads, q_length, self.head_dim]);

        // TODO @thomasw21: Figure out why we need to cast to device here.
        let device = query_layer.device();
        let key_layer = Tensor::f_cat(
            &[&layer_past.key.to_device(device), &key_layer],
            2,
        )
        .unwrap();
        let value_layer = Tensor::f_cat(
            &[&layer_past.value.to_device(device), &value_layer],
            1,
        )
        .unwrap();

        // Update past for next loops
        let past_key_values_length = layer_past.seq_length();

        let beta = 1.0;
        let alpha = self.norm_factor;

        save_layer_to_disk(
            &alibi,
            &format!("rust_baddbmm_sliced_alibi_{}.npy", self.real_layer_number,),
        );
        save_layer_to_disk(
            &key_layer,
            &format!("rust_baddbmm_key_layer_{}.npy", self.real_layer_number,),
        );
        save_layer_to_disk(
            &query_layer,
            &format!("rust_baddbmm_query_layer_{}.npy", self.real_layer_number,),
        );
        save_layer_to_disk(
            &value_layer,
            &format!("rust_baddbmm_value_layer_{}.npy", self.real_layer_number,),
        );
        let mut attention_scores = alibi
            .f_baddbmm_s(&query_layer, &key_layer, beta, alpha)
            .unwrap();

        save_layer_to_disk(
            &attention_scores,
            &format!("rust_baddbmm_matmul_result_{}.npy", self.real_layer_number,),
        );

        // cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        let input_dtype = attention_scores.kind();
        // `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == Kind::Half {
            attention_scores = attention_scores.to_kind(Kind::Float);
        }

        // let causal_mask = prepare_attn_mask(
        //     &attention_mask,
        //     hidden_states.size(),
        //     past_key_values_length,
        //     self.num_attention_heads
        // );
        // save_layer_to_disk(
        //     &causal_mask,
        //     &format!("rust_softmax_attention_mask_{}.npy", self.real_layer_number,),
        // );
        save_layer_to_disk(
            &attention_scores,
            &format!(
                "rust_softmax_attention_scores_{}.npy",
                self.real_layer_number,
            ),
        );
        let attn_weights =
            attention_scores.masked_fill(&attention_mask, finfo_min(attention_scores.kind()));
        save_layer_to_disk(
            &attn_weights,
            &format!("rust_softmax_attn_weights_{}.npy", self.real_layer_number,),
        );
        let attention_probs = attn_weights
            .f_softmax(-1, Kind::Float)
            .unwrap()
            .to_kind(input_dtype);
        save_layer_to_disk(
            &attention_probs,
            &format!(
                "rust_softmax_attention_probs_{}.npy",
                self.real_layer_number,
            ),
        );

        save_layer_to_disk(
            &value_layer,
            &format!("rust_bmm_value_layer_{}.npy", self.real_layer_number,),
        );
        let context_layer = Tensor::f_bmm(&attention_probs, &value_layer).unwrap();

        let context_layer_r = context_layer
            .f_view((batch_size, self.num_attention_heads, q_length, self.head_dim))
            .unwrap();
        let context_layer_p = context_layer_r.permute(&[0, 2, 1, 3]).contiguous();
        let context = context_layer_p
            .f_view((batch_size, q_length, self.num_attention_heads * self.head_dim))
            .unwrap();
        let mut output = if self.slow_but_exact {
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
            &output,
            &format!("rust_bmm_dense_{}.npy", self.real_layer_number,),
        );
        save_layer_to_disk(
            residual,
            &format!("rust_bmm_residual_{}.npy", self.real_layer_number,),
        );
        output += residual;
        save_layer_to_disk(
            &output,
            &format!("rust_bmm_dropout_{}.npy", self.real_layer_number,),
        );

        // update layer_past
        *layer_past = PastLayer {
            key: key_layer,
            value: value_layer,
        };

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
    num_heads: i64
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
        let num_heads = config.n_head;
        Self {
            word_embeddings,
            word_embeddings_layernorm,
            h,
            ln_f,
            num_heads
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


        let input_size = input_ids.size();
        let past_key_values_length = past_key_values[0].seq_length();
        let causal_mask = prepare_attn_mask(
            &attention_mask, input_size, past_key_values_length, self.num_heads
        );

        for (i, (block, layer_past)) in self.h.iter().zip(past_key_values.iter_mut()).enumerate() {
            hidden_states = block.forward(&hidden_states, &causal_mask, alibi, layer_past);
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
    use crate::empty_past;
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
        let device = Device::Cuda(0);
        let mut past_key_values = empty_past(&config, 2);
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
