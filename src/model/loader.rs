use crate::model::model::{
    BloomAttention, BloomBlock, BloomForCausalLM, BloomMlp, BloomModel, Config, Embedding,
    FakeTpLinear, InvertedEmbedding, LayerNorm, Linear, ParallelLinear,
};
use crate::model::tp_layers::{TensorParallelColumnLinear, TensorParallelRowLinear};
use log::debug;
use nccl_rs::ThreadGroup;
use safetensors::{Dtype, SafeTensors, TensorView};
use std::rc::Rc;
use tch::{kind::Kind, Device, Tensor};

pub fn convert(view: TensorView, device: Device) -> Tensor {
    let kind = match view.get_dtype() {
        Dtype::F16 => Kind::Half,
        Dtype::BF16 => Kind::BFloat16,
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

impl FakeTpLinear {
    pub fn load(
        name: &str,
        model: &SafeTensors<'_>,
        device: Device,
        pretraining_tp: usize,
    ) -> Self {
        let linear = Linear::load(name, model, device);
        Self {
            linear,
            pretraining_tp,
        }
    }
}

impl Linear {
    pub fn load(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let tname = format!("{name}.weight");
        let weight = convert(
            model
                .tensor(&tname)
                .unwrap_or_else(|_| panic!("Could not find {tname}")),
            device,
        )
        .f_transpose_copy(1, 0)
        .unwrap();

        let bias_name = format!("{name}.bias");
        let bias = convert(
            model
                .tensor(&bias_name)
                .unwrap_or_else(|_| panic!("Could not find {bias_name}")),
            device,
        );

        Self { weight, bias }
    }
}

impl BloomAttention {
    pub fn load(
        config: &Config,
        name: &str,
        model: &SafeTensors<'_>,
        layer_number: usize,
        device: Device,
    ) -> Self {
        let dense = if config.slow_but_exact {
            ParallelLinear::FakeTp(FakeTpLinear::load(
                &format!("{name}.dense"),
                model,
                device,
                config.pretraining_tp,
            ))
        } else {
            ParallelLinear::Linear(Linear::load(&format!("{name}.dense"), model, device))
        };
        let query_key_value = ParallelLinear::Linear(Linear::load(
            &format!("{name}.query_key_value"),
            model,
            device,
        ));
        Self::new(query_key_value, dense, config, layer_number)
    }

    pub fn load_tp(
        config: &Config,
        name: &str,
        model: &SafeTensors<'_>,
        layer_number: usize,
        device: Device,
        group: Rc<ThreadGroup>,
    ) -> Self {
        let query_key_value = ParallelLinear::TensorParallelColumnLinear(
            TensorParallelColumnLinear::load(&format!("{name}.query_key_value"), model, device),
        );
        let dense = ParallelLinear::TensorParallelRowLinear(TensorParallelRowLinear::load(
            &format!("{name}.dense"),
            model,
            device,
            group,
        ));
        let head_dim = (config.hidden_size / config.n_head) as i64;
        let num_attention_heads = config.n_head as i64;
        let layer_number = layer_number;
        let norm_factor = 1.0 / (head_dim as f64).sqrt();
        Self {
            dense,
            query_key_value,
            num_attention_heads,
            head_dim,
            layer_number,
            norm_factor,
        }
    }
}

impl BloomMlp {
    pub fn load(
        config: &Config,
        name: &str,
        model: &SafeTensors<'_>,
        device: Device,
        layer_number: usize,
    ) -> Self {
        let dense_h_to_4h = Linear::load(&format!("{name}.dense_h_to_4h"), model, device);
        let dense_4h_to_h = Linear::load(&format!("{name}.dense_4h_to_h"), model, device);
        Self::new(dense_h_to_4h, dense_4h_to_h, config, layer_number)
    }
    pub fn load_tp(
        config: &Config,
        name: &str,
        model: &SafeTensors<'_>,
        device: Device,
        layer_number: usize,
        _group: Rc<ThreadGroup>,
    ) -> Self {
        // let dense_h_to_4h =
        //     TensorParallelColumnLinear::new(&format!("{name}.dense_h_to_4h"), model, device);
        // let dense_4h_to_h =
        //     TensorParallelRowLinear::new(&format!("{name}.dense_4h_to_h"), model, device, group);
        let dense_h_to_4h = Linear::load(&format!("{name}.dense_h_to_4h"), model, device);
        let dense_4h_to_h = Linear::load(&format!("{name}.dense_4h_to_h"), model, device);
        Self::new(dense_h_to_4h, dense_4h_to_h, config, layer_number)
    }
}

impl LayerNorm {
    pub fn load(hidden_size: i64, name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
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
}
impl BloomBlock {
    pub fn load(
        config: &Config,
        prefix: &str,
        model: &SafeTensors<'_>,
        layer_number: usize,
        device: Device,
    ) -> Self {
        // attention
        let input_layernorm = LayerNorm::load(
            config.hidden_size,
            &format!("{prefix}.input_layernorm"),
            model,
            device,
        );
        let post_attention_layernorm = LayerNorm::load(
            config.hidden_size,
            &format!("{prefix}.post_attention_layernorm"),
            model,
            device,
        );
        let self_attention = BloomAttention::load(
            config,
            &format!("{prefix}.self_attention"),
            model,
            layer_number,
            device,
        );
        let mlp = BloomMlp::load(
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

    pub fn load_tp(
        config: &Config,
        prefix: &str,
        model: &SafeTensors<'_>,
        layer_number: usize,
        device: Device,
        group: Rc<ThreadGroup>,
    ) -> Self {
        // attention
        let input_layernorm = LayerNorm::load(
            config.hidden_size,
            &format!("{prefix}.input_layernorm"),
            model,
            device,
        );
        let post_attention_layernorm = LayerNorm::load(
            config.hidden_size,
            &format!("{prefix}.post_attention_layernorm"),
            model,
            device,
        );
        let self_attention = BloomAttention::load_tp(
            config,
            &format!("{prefix}.self_attention"),
            model,
            layer_number,
            device,
            Rc::clone(&group),
        );
        let mlp = BloomMlp::load_tp(
            config,
            &format!("{prefix}.mlp"),
            model,
            device,
            layer_number,
            group,
        );
        Self {
            input_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
            layer_number,
        }
    }
}

impl BloomModel {
    pub fn load(config: &Config, model: &SafeTensors<'_>, device: Device) -> Self {
        let word_embeddings = Embedding::load("word_embeddings", model, device);
        let word_embeddings_layernorm = LayerNorm::load(
            config.hidden_size,
            "word_embeddings_layernorm",
            model,
            device,
        );
        let ln_f = LayerNorm::load(config.hidden_size, "ln_f", model, device);
        let h = (0..config.n_layer)
            .map(|i| BloomBlock::load(config, &format!("h.{i}"), model, i as usize, device))
            .collect();
        Self::new(word_embeddings, word_embeddings_layernorm, h, ln_f, config)
    }
}

impl Embedding {
    pub fn load(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let weight = convert(model.tensor(&format!("{name}.weight")).unwrap(), device);
        Self::new(weight)
    }
}

impl InvertedEmbedding {
    pub fn load(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let weight = convert(model.tensor(&format!("{name}.weight")).unwrap(), device);
        Self::new(weight)
    }
}

impl BloomForCausalLM {
    pub fn load(config: &Config, model: &SafeTensors<'_>, device: Device) -> Self {
        let transformer = BloomModel::load(config, model, device);
        let lm_head = InvertedEmbedding::load("word_embeddings", model, device);
        Self::new(transformer, lm_head)
    }
}

impl TensorParallelColumnLinear {
    pub fn load(name: &str, model: &SafeTensors<'_>, device: Device) -> Self {
        let tname = format!("{name}.weight");
        let weight = convert(
            model
                .tensor(&tname)
                .unwrap_or_else(|_| panic!("Could not find {tname}")),
            device,
        )
        .f_transpose_copy(1, 0)
        .unwrap();

        let bias_name = format!("{name}.bias");
        let bias = convert(
            model
                .tensor(&bias_name)
                .unwrap_or_else(|_| panic!("Could not find {bias_name}")),
            device,
        );

        Self::new(weight, bias)
    }
}

impl TensorParallelRowLinear {
    pub fn load(
        name: &str,
        model: &SafeTensors<'_>,
        device: Device,
        group: Rc<ThreadGroup>,
    ) -> Self {
        let tname = format!("{name}.weight");
        let weight = convert(
            model
                .tensor(&tname)
                .unwrap_or_else(|_| panic!("Could not find {tname}")),
            device,
        )
        .f_transpose_copy(1, 0)
        .unwrap();
        debug!("Row linear {:?}", weight);

        let bias_name = format!("{name}.bias");
        let bias = convert(
            model
                .tensor(&bias_name)
                .unwrap_or_else(|_| panic!("Could not find {bias_name}")),
            device,
        );

        Self::new(weight, bias, group)
    }
}
