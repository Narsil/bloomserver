use crate::generation::padding;
use crate::model::{
    prepare_attn_mask, BloomBlock, Config, Embedding, InvertedEmbedding, LayerNorm, Past, PastLayer,
};
use crate::utils::debug;
use crossbeam_channel::{Receiver, Select, Sender};
use log::info;
use memmap::MmapOptions;
use safetensors::SafeTensors;
use std::time::{Duration, Instant};
use tch::{kind::Kind, Device, IndexOp, Tensor};

/// Size of batch and response channel
pub type Ack = (i64, Sender<(Tensor, Past)>);
pub type Msg = (Tensor, Past, Ack);
pub type Msg2 = ((Tensor, Tensor, Tensor, Past, Vec<i64>), Vec<Ack>);
pub type RChan1 = Receiver<Msg>;
pub type SChan1 = Sender<Msg>;
pub type RChan = Receiver<Msg2>;
pub type SChan = Sender<Msg2>;

#[derive(Clone)]
pub struct LayoutConfig {
    pub layers_first_thread: usize,
    pub layers_per_thread: usize,
    pub layers_last_thread: usize,
    pub n_threads: usize,
    pub embeddings_filename: String,
    pub final_filename: String,
    pub layer_template_filename: String,
}

impl LayoutConfig {
    pub fn new() -> Self {
        Self {
            layers_first_thread: 0,
            layers_per_thread: 5,
            layers_last_thread: 0,
            n_threads: 14,
            embeddings_filename: "./weights/bloom-embedding.bin".to_string(),
            final_filename: "./weights/bloom-final.bin".to_string(),
            layer_template_filename: "./weights/bloom-h.{}.bin".to_string(),
        }
    }

    pub fn new_testing() -> Self {
        Self {
            layers_first_thread: 0,
            layers_per_thread: 1,
            layers_last_thread: 0,
            n_threads: 2,
            embeddings_filename: "./weights/bloom-testing.bin".to_string(),
            final_filename: "./weights/bloom-testing.bin".to_string(),
            layer_template_filename: "./weights/bloom-testing.bin".to_string(),
        }
    }

    pub fn new_dgx() -> Self {
        Self {
            layers_first_thread: 0,
            layers_per_thread: 5,
            layers_last_thread: 0,
            n_threads: 2,
            embeddings_filename: "./weights/bloom-embedding.bin".to_string(),
            final_filename: "./weights/bloom-final.bin".to_string(),
            layer_template_filename: "./weights/bloom-h.1.bin".to_string(),
        }
    }

    pub fn new350m() -> Self {
        Self {
            layers_first_thread: 0,
            layers_per_thread: 3,
            layers_last_thread: 0,
            n_threads: 8,
            embeddings_filename: "./weights/bloom-350m.bin".to_string(),
            final_filename: "./weights/bloom-350m.bin".to_string(),
            layer_template_filename: "./weights/bloom-350m.bin".to_string(),
        }
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self::new()
    }
}
pub fn padding_with_ack(
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

pub fn receive(
    rx: &RChan1,
    prio_rx: &RChan1,
    config: &Config,
) -> ((Tensor, Tensor, Tensor, Tensor, Past), Vec<Ack>) {
    let mut sel = Select::new();
    let oper1 = sel.recv(rx);
    let oper2 = sel.recv(prio_rx);
    let oper = sel.select();
    let (mut all_items, no_past) = match oper.index() {
        i if i == oper1 => (vec![oper.recv(rx).unwrap()], true),
        i if i == oper2 => (vec![oper.recv(prio_rx).unwrap()], false),
        _ => unreachable!(),
    };

    if no_past {
        let max_batch_size = 4;
        while let Ok(item) = prio_rx.recv() {
            all_items.push(item);
            if all_items.len() >= max_batch_size {
                break;
            }
        }
    }

    let ((input_ids, attention_mask, alibi, past_key_values), acks) =
        padding_with_ack(config, all_items);
    let input_size = input_ids.size();
    let past_key_values_length = past_key_values[0].seq_length();
    let causal_mask = prepare_attn_mask(
        &attention_mask,
        input_size,
        past_key_values_length,
        config.n_head,
    );
    (
        (
            input_ids,
            causal_mask,
            attention_mask,
            alibi,
            past_key_values,
        ),
        acks,
    )
}

pub fn send(
    lm_logits: Tensor,
    past_key_values: &[PastLayer],
    acks: &[Ack],
    seq_lengths: &[i64],
    causal_mask: Tensor,
    config: &Config,
) {
    let mut current_batch = 0i64;
    for (mini_batch_size, rq) in acks {
        // XXX actually clean the padded values of past so that subsequent
        // calls can get a chance to have a better padding (+ correct attention mask).
        let seq_length = seq_lengths[current_batch as usize];
        let total_seq_length = causal_mask.size()[2];
        let start_batch_size_times_num_heads = current_batch * config.n_head;
        let end_batch_size_times_num_heads =
            start_batch_size_times_num_heads + mini_batch_size * config.n_head;
        let past: Vec<_> = past_key_values
            .iter()
            .map(|layer_past| PastLayer {
                key: layer_past.key.i((
                    start_batch_size_times_num_heads..end_batch_size_times_num_heads,
                    ..,
                    total_seq_length - seq_length..,
                )),
                value: layer_past.value.i((
                    start_batch_size_times_num_heads..end_batch_size_times_num_heads,
                    total_seq_length - seq_length..,
                )),
            })
            .collect();
        let simple_logits = lm_logits.i(current_batch..current_batch + mini_batch_size);
        rq.send((simple_logits, past)).unwrap();
        current_batch += mini_batch_size;
    }
}

pub fn thread1(
    rx: RChan1,
    prio_rx: RChan1,
    s2: SChan,
    thread_number: usize,
    config: Config,
    layout_config: LayoutConfig,
) {
    info!("Starting thread {thread_number}");
    let start = std::time::Instant::now();
    let device = Device::Cuda(thread_number);

    let file = std::fs::File::open(&layout_config.embeddings_filename).unwrap();
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
    info!(
        "{:?} : Loaded thread {thread_number} in {:?}",
        std::time::Instant::now(),
        start.elapsed()
    );

    let mut last_loop = Instant::now();
    loop {
        // let start = Instant::now();
        let ((input_ids, causal_mask, attention_mask, alibi, mut past_key_values), acks) =
            receive(&rx, &prio_rx, &config);
        let inputs_embeds = word_embeddings.forward(&input_ids);
        let mut hidden_states = word_embeddings_layernorm.forward(&inputs_embeds);

        let seq_lengths = Vec::<i64>::from(
            attention_mask
                .f_sum_dim_intlist(&[-1], false, Kind::Int)
                .unwrap(),
        );

        for (layer, layer_past) in layers.iter().zip(past_key_values.iter_mut()) {
            hidden_states = layer.forward(&hidden_states, &causal_mask, &alibi, layer_past);
        }
        s2.send((
            (
                hidden_states,
                causal_mask,
                alibi,
                past_key_values,
                seq_lengths,
            ),
            acks,
        ))
        .unwrap();
        last_loop = Instant::now();
    }
}

pub fn thread2(
    rx: RChan,
    s: SChan,
    thread_number: usize,
    config: Config,
    layout_config: LayoutConfig,
) {
    info!("Starting thread {thread_number}");
    let start = std::time::Instant::now();
    let device = Device::Cuda(thread_number);

    let offset =
        layout_config.layers_first_thread + layout_config.layers_per_thread * (thread_number - 1);
    let layers: Vec<BloomBlock> = (0..layout_config.layers_per_thread)
        .map(|i| {
            let layer_number = i + offset;
            info!("Loading layer {layer_number} on thread2 ({thread_number})");
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
    info!(
        "{:?} : Loaded thread {thread_number} in {:?}",
        std::time::Instant::now(),
        start.elapsed()
    );
    loop {
        // Receive 1 item
        let mut all_items = vec![rx.recv().unwrap()];

        while let Ok(item) = rx.recv_timeout(Duration::from_millis(0)) {
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
            let (
                (mut hidden_states, mut causal_mask, mut alibi, mut past_key_values, seq_lengths),
                rq,
            ) = item;
            hidden_states = hidden_states.to_device(device);
            causal_mask = causal_mask.to_device(device);
            alibi = alibi.to_device(device);

            for (layer, layer_past) in layers.iter().zip(past_key_values.iter_mut().skip(offset)) {
                hidden_states = layer.forward(&hidden_states, &causal_mask, &alibi, layer_past);
            }
            s.send((
                (
                    hidden_states,
                    causal_mask,
                    alibi,
                    past_key_values,
                    seq_lengths,
                ),
                rq,
            ))
            .unwrap();
        }
    }
}

pub fn thread3(rx: RChan, thread_number: usize, config: Config, layout_config: LayoutConfig) {
    info!("Starting thread {thread_number}");
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

    let offset = layout_config.layers_first_thread
        + layout_config.layers_per_thread * layout_config.n_threads;
    let layers: Vec<BloomBlock> = (0..layout_config.layers_last_thread)
        .map(|i| {
            let layer_number = offset + i;
            info!("Loading layer {layer_number} on thread3 ({thread_number})");
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

    info!(
        "{:?} : Loaded thread {thread_number} in {:?}",
        std::time::Instant::now(),
        start.elapsed()
    );

    loop {
        let (
            (mut hidden_states, mut causal_mask, mut alibi, mut past_key_values, seq_lengths),
            acks,
        ) = rx
            .recv()
            .expect("You probably want to handle this case, but I'm too lazy");

        hidden_states = hidden_states.to_device(device);
        causal_mask = causal_mask.to_device(device);
        alibi = alibi.to_device(device);
        for (layer, layer_past) in layers.iter().zip(past_key_values.iter_mut().skip(offset)) {
            debug("past_key thread3", &layer_past.key);
            debug("past_values thread3", &layer_past.value);
            hidden_states = layer.forward(&hidden_states, &causal_mask, &alibi, layer_past);
        }
        debug("last_hidden_states", &hidden_states);
        hidden_states = ln_f.forward(&hidden_states);
        debug("After ln_f", &hidden_states);
        let lm_logits = lm_head.forward(&hidden_states);

        send(
            lm_logits,
            &past_key_values,
            &acks,
            &seq_lengths,
            causal_mask,
            &config,
        );
    }
}
