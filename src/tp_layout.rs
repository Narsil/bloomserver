use crate::layout::{padding_with_ack, receive, send, RChan1};
use crate::model::{
    prepare_attn_mask, BloomBlock, Config, Embedding, InvertedEmbedding, LayerNorm, Past, PastLayer,
};
use crate::utils::debug;
use crossbeam_channel::{Receiver, Select, Sender};
use log::info;
use memmap::MmapOptions;
use nccl_rs::ThreadGroup;
use safetensors::SafeTensors;
use tch::{kind::Kind, Device, IndexOp};

pub fn thread(group: ThreadGroup, config: Config, channels: Option<(RChan1, RChan1)>) {
    // TODO load the weights
    // let thread_number = group.rank() as usize;
    // info!("Starting thread {thread_number}");
    // let start = std::time::Instant::now();
    // let device = Device::Cuda(thread_number);

    // let filename = "weights/bloom-testing.bin";

    // let file = std::fs::File::open(filename).unwrap();
    // // SAFETY: This is actually unsafe.
    // let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    // let model = SafeTensors::deserialize(&mmap).unwrap();

    // let word_embeddings = Embedding::new("word_embeddings", &model, device);
    // let word_embeddings_layernorm = LayerNorm::new(
    //     config.hidden_size,
    //     "word_embeddings_layernorm",
    //     &model,
    //     device,
    // );
    // let layers: Vec<BloomBlock> = (0..config.n_layer as usize)
    //     .map(|i| BloomBlock::new_tp(&config, &format!("h.{i}"), &model, i, device, &group))
    //     .collect();

    // let ln_f = LayerNorm::new(config.hidden_size, "ln_f", &model, device);
    // let lm_head = InvertedEmbedding::new("word_embeddings", &model, device);

    // if let Some((rx, prio_rx)) = channels {
    //     loop {
    //         let ((input_ids, causal_mask, attention_mask, alibi, mut past_key_values), acks) =
    //             receive(&rx, &prio_rx, &config);
    //         let inputs_embeds = word_embeddings.forward(&input_ids);
    //         let mut hidden_states = word_embeddings_layernorm.forward(&inputs_embeds);

    //         let seq_lengths = Vec::<i64>::from(
    //             attention_mask
    //                 .f_sum_dim_intlist(&[-1], false, Kind::Int)
    //                 .unwrap(),
    //         );

    //         group.broadcast((hidden_states, past_key_values, causal_mask, alibi));

    //         for (layer, layer_past) in layers.iter().zip(past_key_values.iter_mut()) {
    //             hidden_states = layer.forward(&hidden_states, &causal_mask, &alibi, layer_past);
    //         }
    //         debug("last_hidden_states", &hidden_states);
    //         hidden_states = ln_f.forward(&hidden_states);
    //         debug("After ln_f", &hidden_states);
    //         let lm_logits = lm_head.forward(&hidden_states);

    //         send(
    //             lm_logits,
    //             &past_key_values,
    //             &acks,
    //             &seq_lengths,
    //             causal_mask,
    //             &config,
    //         );
    //     }
    // } else {
    //     let (hidden_states, past_key_values, causal_mask, alibi) = group.broadcast_recv();

    //     for (layer, layer_past) in layers.iter().zip(past_key_values.iter_mut()) {
    //         hidden_states = layer.forward(&hidden_states, &causal_mask, &alibi, layer_past);
    //     }
    // }
}
