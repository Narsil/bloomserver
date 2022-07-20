use bloomserver::generation::{add_next_id, GenerationMode, Parameters};
use bloomserver::layout::{thread1, thread2, thread3, LayoutConfig, Msg, Msg2};
use bloomserver::model::{Config, Past};
use bloomserver::{empty_past, Generation, GenerationError};
use crossbeam_channel::{bounded, unbounded, Sender};
use tch::{kind, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

fn generate(
    tokenizer: &Tokenizer,
    payload: Generation,
    config: &Config,
    in_channel: &Sender<Msg>,
    prio_channel: &Sender<Msg>,
) -> String {
    let encoded = tokenizer.encode(payload.inputs.clone(), false).unwrap();
    let mut ids: Vec<_> = encoded.get_ids().iter().map(|&i| i as i64).collect();
    // TODO The model is not able to handle empty input ids
    if ids.is_empty() {
        ids.push(0);
    }
    println!("Ids {:?}", ids);
    println!("Ids length {:?}", ids.len());
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
        let past_key_values = empty_past(config);
        std::env::set_var("GENERATION_STEP", i.to_string());
        if i == 0 {
            in_channel
                .try_send((input_ids.copy(), past_key_values, ack))
                .map_err(|_| {
                    // println!("Queue was full {:?}", in_channel.len());
                    GenerationError::QueueFull
                })
                .unwrap();
        } else {
            prio_channel
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
    let string = tokenizer.decode(full_ids, false).unwrap();
    return string;
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    tch::maybe_init_cuda();

    let start = std::time::Instant::now();
    let tokenizer = Tokenizer::from_file("./tokenizer2.json").unwrap();
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

    let payloads = vec![
        // Generation {
        //     inputs: "I enjoy walking my cute dog".to_string(),
        //     parameters: Parameters {
        //         max_new_tokens: 20,
        //         generation_mode: GenerationMode::Greedy,
        //     },
        // },
        Generation {
            inputs: "Math exercise - answers:\n34+10=44\n54+20=".to_string(),
            parameters: Parameters {
                max_new_tokens: 20,
                generation_mode: GenerationMode::Greedy,
            },
        },
        // Generation {
        //     inputs: "I enjoy walking my cute dog".to_string(),
        //     parameters: Parameters {
        //         max_new_tokens: 20,
        //         generation_mode: GenerationMode::Greedy,
        //     },
        // },
    ];
    for payload in payloads {
        let result = generate(&tokenizer, payload, &config, &tx, &prio_tx);
        println!("{:?}", result);
    }

    Ok(())
}
