use crate::empty_past;
use crate::generation::padding;
use crate::model::{BloomBlock, Config, Embedding, InvertedEmbedding, LayerNorm, Past};
use crate::utils::debug;
use crossbeam_channel::{Receiver, Select, Sender};
use memmap::MmapOptions;
use safetensors::SafeTensors;
use std::time::{Duration, Instant};
use tch::{Device, IndexOp, Tensor};

/// Size of batch and response channel
pub type Ack = (i64, Sender<(Tensor, Past)>);
pub type Msg = (Tensor, Past, Ack);
pub type Msg2 = ((Tensor, Tensor, Tensor, Past), Vec<Ack>);
pub type RChan1 = Receiver<Msg>;
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
            embeddings_filename: "./bloom-embedding.bin".to_string(),
            final_filename: "./bloom-final.bin".to_string(),
            layer_template_filename: "./bloom-h.{}.bin".to_string(),
        }
    }

    pub fn new_testing() -> Self {
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

    pub fn new_dgx() -> Self {
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

    pub fn new350m() -> Self {
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

pub fn thread1(
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
            last_loop + Duration::from_millis(20),
            now + Duration::from_millis(1),
        );

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
            if all_items.len() >= max_batch_size {
                break;
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

pub fn thread2(
    rx: RChan,
    s: SChan,
    thread_number: usize,
    config: Config,
    layout_config: LayoutConfig,
) {
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
        // println!(
        //     "Got stuck on RECEIVE thread {thread_number} for {:?}",
        //     start.elapsed()
        // );

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
            s.send(((hidden_states, attention_mask, alibi, past_key_values), rq))
                .unwrap();
        }
    }
}

pub fn thread3(rx: RChan, thread_number: usize, config: Config, layout_config: LayoutConfig) {
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
