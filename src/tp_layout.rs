use crate::generation::{next_ids, padding, Parameters};
use crate::model::{
    build_alibi_tensor, prepare_attn_mask, BloomBlock, Config, Embedding,
    empty_past,
    InvertedEmbedding, LayerNorm, Past,
};
use crossbeam_channel::{Receiver, Sender};
use log::{debug, info};
use memmap::MmapOptions;
use nccl_rs::ThreadGroup;
use safetensors::SafeTensors;
use std::rc::Rc;
use tch::{Device, IndexOp, Tensor};

type Ack = (Tensor, Parameters, Sender<Tensor>);
pub type Msg = (Tensor, Parameters, Sender<Tensor>);
pub type TpReceiver = Receiver<Msg>;
pub type TpSender = Sender<Msg>;

pub fn padding_with_ack(
    config: &Config,
    items: Vec<(Tensor, Parameters, Sender<Tensor>)>,
    device: Device,
    group: &Rc<ThreadGroup>,
) -> ((Tensor, Tensor, Tensor, Past), Vec<Ack>) {
    let world_size = group.ranks() as usize;
    let mut tensors = vec![];
    let mut acks = vec![];
    let batch_size = items.len() as i64;
    for item in items {
        let past = empty_past(&config, batch_size, world_size);
        tensors.push((item.0.copy(), past));
        acks.push((item.0.to_device(device), item.1, item.2));
    }
    (padding(config, tensors, world_size), acks)
}

#[derive(Clone)]
pub struct LayoutConfig {
    pub world_size: usize,
    name: String
}

impl LayoutConfig {
    pub fn new() -> Self {
        Self {
            world_size: 16,
            name: "bloom".to_string()
        }
    }

    pub fn new_testing() -> Self {
        Self {
            world_size: 2,
            name: "bigscience-small-testing".to_string()
        }
    }

    pub fn new350m() -> Self {
        Self {
            world_size: 4,
            name: "bloom-350m".to_string()
        }
    }
}

pub fn thread(
    group: ThreadGroup,
    config: Config,
    layout_config: LayoutConfig,
    channel: TpReceiver,
    init_channel: Sender<bool>,
) {
    let thread_number = group.rank() as usize;
    let world_size= group.ranks() as i32;
    info!("Starting thread {thread_number}");
    let start = std::time::Instant::now();
    let device = Device::Cuda(thread_number);
    let group = Rc::new(group);

    // TODO support large model
    let filename = format!("./weights/bigscience/{}_tp-rank-{thread_number}-of-{world_size}.bin", layout_config.name);
    debug!("Loading {filename}");


    let file = std::fs::File::open(filename).unwrap();
    // SAFETY: This is actually unsafe.
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let model = SafeTensors::deserialize(&mmap).unwrap();

    let word_embeddings = Embedding::load("transformer.word_embeddings", &model, device);


    let word_embeddings_layernorm = LayerNorm::load(
        config.hidden_size,
        "transformer.word_embeddings_layernorm",
        &model,
        device,
    );
    let layers: Vec<BloomBlock> = (0..config.n_layer as usize)
        .map(|i| {
            let block = BloomBlock::load_tp(
                &config,
                &format!("transformer.h.{i}"),
                &model,
                i,
                device,
                Rc::clone(&group),
            );
            // info!("Loaded layer {i} on thread {thread_number}");
            block
        })
        .collect();

    let ln_f = LayerNorm::load(config.hidden_size, "transformer.ln_f", &model, device);
    let lm_head = InvertedEmbedding::load("transformer.word_embeddings", &model, device);
    let elapsed = start.elapsed();
    info!("Loaded thread {thread_number} in {elapsed:?}");

    let max_batch_size = 16;
    let num_heads = config.n_head / group.ranks() as i64;
    init_channel.send(true).unwrap();
    loop {
        debug!("[{thread_number}] Waiting on channel");
        // let mut all_items = vec![channel.recv().unwrap()];
        // TODO remove timeout
        let mut all_items = vec![channel.recv_timeout(std::time::Duration::from_secs(120)).unwrap()];
        while let Ok(item) = channel.recv_timeout(std::time::Duration::from_millis(0)) {
            all_items.push(item);
            if all_items.len() > max_batch_size {
                break;
            }
        }

        let ((mut input_ids, mut attention_mask, _alibi, mut past_key_values), mut acks) =
            padding_with_ack(&config, all_items, device, &group);
        input_ids = input_ids.to_device(device);
        attention_mask = attention_mask.to_device(device);
        past_key_values.iter_mut().for_each(|past|{
            past.key = past.key.to_device(device);
            past.value = past.value.to_device(device);
        });

        let initial_length = input_ids.size()[1];

        while !acks.is_empty() {
            let start = std::time::Instant::now();
            debug!("On rank {thread_number}");
            let alibi = build_alibi_tensor(&attention_mask, num_heads, config.kind, device);
            debug!("alibi rank {thread_number}");
            let input_size = input_ids.size();
            let past_key_values_length = past_key_values[0].seq_length();
            let causal_mask = prepare_attn_mask(
                &attention_mask,
                input_size,
                past_key_values_length,
                num_heads
            );
            debug!("attn mask rank {thread_number}");
            let inputs_embeds = word_embeddings.forward(&input_ids);
            debug!("embeding rank {thread_number}");
            let mut hidden_states = word_embeddings_layernorm.forward(&inputs_embeds);
            debug!("layer norm {thread_number}");

            debug!("Layers {:?} past {:?}", layers.len(), past_key_values.len());
            for (i, (layer, layer_past)) in layers.iter().zip(past_key_values.iter_mut()).enumerate() {
                hidden_states = layer.forward(&hidden_states, &causal_mask, &alibi, layer_past);
                debug!("layer {i} on thread {thread_number}");
            }
            debug!("layers {thread_number}");
            hidden_states = ln_f.forward(&hidden_states);
            let lm_logits = lm_head.forward(&hidden_states);
            debug!("Logits {:?} - rank {thread_number}", lm_logits);

            let mut keep_ids = vec![];
            let mut keep_head_ids = vec![];

            let mut all_new_ids = vec![];
            let mut new_acks = vec![];
            for (i, (mut all_ids, parameters, rq)) in (0..lm_logits.size()[0]).zip(acks.into_iter())
            {
                let new_ids =
                    next_ids(&parameters, &lm_logits.i(i..i + 1)).to_device(input_ids.device());
                all_ids = Tensor::f_cat(&[all_ids, new_ids.copy()], 1).unwrap();
                if all_ids.size()[1] - initial_length >= parameters.max_new_tokens as i64 {
                    if group.rank() == 0 {
                        rq.send_timeout(all_ids.copy(), std::time::Duration::from_millis(0)).unwrap();
                    }
                } else {
                    new_acks.push((all_ids, parameters, rq));
                    all_new_ids.push(new_ids);
                    keep_ids.push(i);
                    keep_head_ids.extend(i * num_heads..(i + 1) * num_heads);
                }
            }

            acks = new_acks;
            if acks.is_empty() {
                break;
            }
            input_ids = Tensor::f_cat(&all_new_ids, 0).unwrap();
            if thread_number == 0{
            debug!("Reindexing past {:?} - {:?}" ,keep_head_ids, past_key_values[0].key.size());
           }
            past_key_values.iter_mut().for_each(|past| {
                past.key = past.key.i(keep_head_ids.as_slice());
                past.value = past.value.i(keep_head_ids.as_slice());
            });
            let ones = Tensor::ones(&[input_ids.size()[0], 1], (config.kind, device));
            attention_mask = Tensor::f_cat(&[attention_mask, ones], 1).unwrap();

            if thread_number == 0{
            info!("Loop took {:?}", start.elapsed());
            }
        }
    }
}
