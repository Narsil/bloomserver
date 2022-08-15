#![feature(async_closure)]
use actix_web::middleware::Logger;
use actix_web::{http::header::ContentType, post, web, App, HttpResponse, HttpServer};
use crossbeam_channel::{bounded, Sender};
use log::{debug, info};
use nccl_rs::ThreadGroup;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tch::{kind, Cuda, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

use bloomserver::generation::Parameters;
use bloomserver::model::Config;
use bloomserver::tp_layout::thread;
use bloomserver::{Generation, GenerationError};

#[derive(Clone)]
struct AppState {
    channel: Sender<(Tensor, Parameters, Sender<Tensor>)>,
    tokenizer: Arc<Tokenizer>,
}

#[post("/generate")]
async fn generate(
    payload: web::Json<Generation>,
    state: web::Data<AppState>,
) -> actix_web::Result<HttpResponse> {
    debug!("Received request {:?}", payload);
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

        let (sx, rx) = bounded::<Tensor>(1);
        let kind = (kind::Kind::Int64, Device::Cuda(0));
        let input_ids = Tensor::of_slice(ids.as_slice())
            .to_kind(kind.0)
            .to_device(kind.1)
            .view((1, -1));

        debug!("Sending request {:?}", input_ids);
        state
            .channel
            .send((input_ids, payload.parameters.clone(), sx))
            .map_err(|_| GenerationError::QueueFull)?;

        let all_ids = rx
            .recv()
            .map_err(|_| GenerationError::CouldNotReceiveAnswer)?;
        debug!("Received answer {:?}", all_ids);

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

fn init_threads(
    model_name: &str,
) -> (Arc<Tokenizer>, Sender<(Tensor, Parameters, Sender<Tensor>)>) {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    tch::maybe_init_cuda();
    let start = std::time::Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file("./tokenizer.json").unwrap());
    info!("Loaded tokenizer in {:?}", start.elapsed());
    info!("Starting threads {:?}", std::time::Instant::now());

    let (tx, rx) = bounded::<(Tensor, Parameters, Sender<Tensor>)>(1);

    let world_size = Cuda::device_count() as i32;

    let config = match model_name {
        "bigscience-small-testing" => Config::new_testing(),
        "bloom-350m" => Config::new350m(),
        "bloom" => Config::new(),
        "bloom-dgx" => Config::new(),
        other => panic!("Model {other} is not known"),
    };

    let id = ThreadGroup::new_id().unwrap();
    for rank in 0..world_size {
        let config_ = config.clone();
        let rx = rx.clone();
        tokio::task::spawn_blocking(move || {
            let group = ThreadGroup::new(world_size, rank, id).unwrap();
            thread(group, config_, rx);
        });
    }

    (tokenizer, tx)
}

#[actix_web::main] // or #[tokio::main]
async fn main() -> std::io::Result<()> {
    let model_name = std::env::var("BLOOM").unwrap_or_else(|_| "bloom".to_string());
    let (tokenizer, channel) = init_threads(&model_name);
    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(web::Data::new(AppState {
                tokenizer: tokenizer.clone(),
                channel: channel.clone(),
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
    async fn test_tp_generation_testing() {
        let (tokenizer, channel) = init_threads("bigscience-small-testing");
        let app = test::init_service(
            App::new()
                .wrap(Logger::default())
                .app_data(web::Data::new(AppState { tokenizer, channel }))
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
}
