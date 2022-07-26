use actix_web::middleware::Logger;
use actix_web::{http::header::ContentType, post, web, App, HttpResponse, HttpServer};
use crossbeam_channel::{bounded, unbounded, Sender};
use log::info;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tch::{kind, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

use bloomserver::generation::add_next_id;
use bloomserver::layout::{thread1, thread2, thread3, LayoutConfig, Msg, Msg2};
use bloomserver::model::{Config, Past};
use bloomserver::{empty_past, Generation, GenerationError};

#[derive(Clone)]
struct AppState {
    in_channel: Sender<Msg>,
    prio_channel: Sender<Msg>,
    tokenizer: Arc<Tokenizer>,
    config: Config,
}

#[post("/generate")]
async fn generate(
    payload: web::Json<Generation>,
    state: web::Data<AppState>,
) -> actix_web::Result<HttpResponse> {
    let start = Instant::now();
    let input_string = payload.inputs.clone();
    let parameters = payload.parameters.clone();
    let string = actix_web::rt::task::spawn_blocking(move || -> Result<String, GenerationError> {
        let state = state.into_inner();
        let encoded = state
            .tokenizer
            .encode(payload.inputs.clone(), false)
            .unwrap();
        let mut ids: Vec<_> = encoded.get_ids().iter().map(|&i| i as i64).collect();
        // TODO The model is not able to handle empty input ids
        if ids.is_empty() {
            ids.push(0);
        }

        if ids.len() > 512 {
            return Err(GenerationError::InputTooLong);
        }
        if payload.parameters.max_new_tokens > 384 {
            return Err(GenerationError::TooManyNewTokens);
        }

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
            let past_key_values = empty_past(&state.config);
            if i == 0 {
                state
                    .in_channel
                    .try_send((input_ids.copy(), past_key_values, ack))
                    .map_err(|_| {
                        // println!("Queue was full {:?}", state.in_channel.len());
                        GenerationError::QueueFull
                    })?;
            } else {
                state
                    .prio_channel
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
        let string = state.tokenizer.decode(full_ids, true).unwrap();
        Ok(string)
    })
    .await
    .unwrap()?;
    info!(
        r#"Ran query with:
    input: {:?}
    parameters {:?}
    output {:?}
    ran in {:?}"#,
        input_string,
        parameters,
        string,
        start.elapsed()
    );
    Ok(HttpResponse::Ok()
        .content_type(ContentType::json())
        .json(json!([{ "generated_text": string }])))
}
#[actix_web::main] // or #[tokio::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    tch::maybe_init_cuda();

    let start = std::time::Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file("./tokenizer.json").unwrap());
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

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(web::Data::new(AppState {
                tokenizer: tokenizer.clone(),
                in_channel: tx.clone(),
                prio_channel: prio_tx.clone(),
                config: config.clone(),
            }))
            .service(generate)
    })
    .bind(("127.0.0.1", 8001))?
    .run()
    .await
}
