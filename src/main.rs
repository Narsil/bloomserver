#![feature(async_closure)]
use actix_web::middleware::Logger;
use actix_web::{http::header::ContentType, post, web, App, HttpResponse, HttpServer};
use crossbeam_channel::{bounded, unbounded, Sender};
use log::info;
use nccl_rs::ThreadGroup;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tch::{kind, Cuda, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

use bloomserver::generation::next_ids;
use bloomserver::layout::{thread1, thread2, thread3, LayoutConfig, Msg, Msg2, SChan1};
use bloomserver::model::{Config, Past};
use bloomserver::tp_layout::thread as tp_thread;
use bloomserver::utils::SAVE;
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
    let max_new_tokens = payload.parameters.max_new_tokens;
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

        if ids.len() > 384 {
            return Err(GenerationError::InputTooLong);
        }
        if payload.parameters.max_new_tokens > 364 {
            return Err(GenerationError::TooManyNewTokens);
        }

        let (sx, rx) = bounded::<(Tensor, Past)>(2);
        let kind = (kind::Kind::Int64, Device::Cuda(0));
        let mut input_ids = Tensor::of_slice(ids.as_slice())
            .to_kind(kind.0)
            .to_device(kind.1)
            .view((1, -1));

        let mut all_ids = input_ids.copy();
        let max_new_tokens = payload.parameters.max_new_tokens;

        let mut past_key_values = empty_past(&state.config, 1);
        for i in 0..max_new_tokens {
            // let start_loop = Instant::now();
            if SAVE {
                std::env::set_var("GENERATION_STEP", &format!("{}", i));
            }
            let ack = (input_ids.size()[0], sx.clone());
            if i == 0 {
                state
                    .in_channel
                    .try_send((input_ids.copy(), past_key_values, ack))
                    .map_err(|_| GenerationError::QueueFull)?;
            } else {
                state
                    .prio_channel
                    .send((input_ids.copy(), past_key_values, ack))
                    .expect("This send should always work");
            }
            let received = rx.recv().unwrap();
            let logits = received.0;
            past_key_values = received.1;
            let new_ids = next_ids(&payload.parameters, &logits).to_device(input_ids.device());
            all_ids = Tensor::f_cat(&[all_ids, new_ids.copy()], 1).unwrap();
            input_ids = new_ids;
        }
        let full_ids = all_ids
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
    ran in {:?} ({:?}/token)"#,
        input_string,
        parameters,
        string,
        start.elapsed(),
        start.elapsed().div_f64(max_new_tokens as f64)
    );
    Ok(HttpResponse::Ok()
        .content_type(ContentType::json())
        .json(json!([{ "generated_text": string }])))
}

fn init_threads_tp(model_name: &str) -> (Arc<Tokenizer>, SChan1, SChan1, Config) {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    tch::maybe_init_cuda();
    let start = std::time::Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file("./tokenizer.json").unwrap());
    info!("Loaded tokenizer in {:?}", start.elapsed());
    info!("Starting threads {:?}", std::time::Instant::now());

    let (tx, rx) = bounded::<Msg>(1);
    let (prio_tx, prio_rx) = unbounded::<Msg>();

    let world_size = Cuda::device_count() as i32;

    let config = match model_name {
        "bigscience-small-testing" => Config::new_testing(),
        "bloom-350m" => Config::new350m(),
        "bloom" => Config::new(),
        "bloom-dgx" => Config::new(),
        other => panic!("Model {other} is not known"),
    };

    let id = ThreadGroup::new_id().unwrap();
    let config_ = config.clone();
    tokio::task::spawn_blocking(move || {
        let group = ThreadGroup::new(world_size, 0, id).unwrap();
        tp_thread(group, config_, Some((rx, prio_rx)));
    });
    for rank in 1..world_size {
        let config_ = config.clone();
        tokio::task::spawn_blocking(move || {
            let group = ThreadGroup::new(world_size, rank, id).unwrap();
            tp_thread(group, config_, None);
        });
    }

    (tokenizer, tx, prio_tx, config)
}

fn init_threads(model_name: &str) -> (Arc<Tokenizer>, SChan1, SChan1, Config) {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    tch::maybe_init_cuda();
    let start = std::time::Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file("./tokenizer.json").unwrap());
    info!("Loaded tokenizer in {:?}", start.elapsed());
    info!("Starting threads {:?}", std::time::Instant::now());

    let (tx, rx) = bounded::<Msg>(1);
    let (prio_tx, prio_rx) = unbounded::<Msg>();

    let (config, layout_config) = match model_name {
        "bigscience-small-testing" => (Config::new_testing(), LayoutConfig::new_testing()),
        "bloom-350m" => (Config::new350m(), LayoutConfig::new350m()),
        "bloom" => (Config::new(), LayoutConfig::new()),
        "bloom-dgx" => (Config::new(), LayoutConfig::new_dgx()),
        other => panic!("Model {other} is not known"),
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
    (tokenizer, tx, prio_tx, config)
}

#[actix_web::main] // or #[tokio::main]
async fn main() -> std::io::Result<()> {
    let model_name = std::env::var("BLOOM").unwrap_or_else(|_| "bloom".to_string());
    let (tokenizer, in_channel, prio_channel, config) = match std::env::var("TP") {
        Ok(s) if s == *"1" => init_threads_tp(&model_name),
        _ => init_threads(&model_name),
    };
    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(web::Data::new(AppState {
                tokenizer: tokenizer.clone(),
                in_channel: in_channel.clone(),
                prio_channel: prio_channel.clone(),
                config: config.clone(),
            }))
            .service(generate)
    })
    .bind(("127.0.0.1", 8001))?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::test;

    #[actix_web::test]
    async fn test_generation_testing() {
        let (tokenizer, in_channel, prio_channel, config) =
            init_threads("bigscience-small-testing");
        let app = test::init_service(
            App::new()
                .wrap(Logger::default())
                .app_data(web::Data::new(AppState {
                    tokenizer,
                    in_channel,
                    prio_channel,
                    config,
                }))
                .service(generate),
        )
        .await;
        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "I enjoy walking my cute dog"}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        assert_eq!(bytes.unwrap(), web::Bytes::from_static(b"[{\"generated_text\":\"I enjoy walking my cute dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog\"}]"));
    }

    #[actix_web::test]
    async fn test_generation_350m() {
        let (tokenizer, in_channel, prio_channel, config) = init_threads("bloom-350m");
        info!("Started");
        let app = test::init_service(
            App::new()
                .wrap(Logger::default())
                .app_data(web::Data::new(AppState {
                    tokenizer,
                    in_channel,
                    prio_channel,
                    config,
                }))
                .service(generate),
        )
        .await;
        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "I enjoy walking my cute dog", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        assert_eq!(
            bytes.unwrap(),
            web::Bytes::from_static(b"[{\"generated_text\":\"I enjoy walking my cute dog, but I also love to play with my cat. I am a very active person and I love\"}]")
        );

        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "A \"whatpu\" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a \"farduddle\" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        assert_eq!(
            bytes.unwrap(),
            web::Bytes::from_static(b"[{\"generated_text\":\"A \\\"whatpu\\\" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a \\\"farduddle\\\" means to jump up and down really fast. An example of a sentence that uses the word farduddle is: We were traveling in Africa and we saw these very cute whatpus. To do a \\\"fardudd\"}]")
        );
    }

    #[actix_web::test]
    async fn test_generation_full() {
        if std::env::var("RUN_SLOW") != Ok("1".to_string()) {
            return;
        }
        let (tokenizer, in_channel, prio_channel, config) = init_threads("bloom");
        let app = test::init_service(
            App::new()
                .wrap(Logger::default())
                .app_data(web::Data::new(AppState {
                    tokenizer,
                    in_channel,
                    prio_channel,
                    config,
                }))
                .service(generate),
        )
        .await;

        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "test", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        assert_eq!(
            bytes.unwrap(),
            web::Bytes::from_static(b"[{\"generated_text\":\"test.mark.parametrize('stdin, stdout', [\\n    ({'username': 'foo'\"}]")
        );

        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "I enjoy walking my cute dog", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        assert_eq!(
            bytes.unwrap(),
            web::Bytes::from_static(b"[{\"generated_text\":\"I enjoy walking my cute dog, reading, and spending time with my family. I am a very active person and love to be\"}]")
        );

        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "Math exercise - answers:\n34+10=44\n54+20=", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        assert_eq!(
            bytes.unwrap(),
            web::Bytes::from_static(b"[{\"generated_text\":\"Math exercise - answers:\\n34+10=44\\n54+20=74\\n74+20=94\\n94+20=114\\n114+20=134\\n\"}]")
        );
        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "A \"whatpu\" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a \"farduddle\" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        assert_eq!(
            bytes.unwrap(),
            web::Bytes::from_static(b"[{\"generated_text\":\"A \\\"whatpu\\\" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a \\\"farduddle\\\" means to jump up and down really fast. An example of a sentence that uses the word farduddle is: The kids were jumping up and down on the trampoline and doing a farduddle. To\"}]")
        );
    }
}
