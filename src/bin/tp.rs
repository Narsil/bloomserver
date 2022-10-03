#![feature(async_closure)]
use actix_web::middleware::Logger;
use actix_web::{http::header::ContentType, post, web, App, HttpResponse, HttpServer};
use crossbeam_channel::{bounded, Sender};
use log::{debug, info};
use nccl_rs::ThreadGroup;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tch::{kind, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

use bloomserver::generation::Parameters;
use bloomserver::model::Config;
use bloomserver::tp_layout::{thread, LayoutConfig};
use bloomserver::{Generation, GenerationError};

#[derive(Clone)]
struct AppState {
    channels: Vec<Sender<(Tensor, Parameters, Sender<Tensor>)>>,
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

        for (i, channel) in state.channels.iter().enumerate(){
            debug!("Sending request {:?} - {:?}", input_ids, i);
            channel
                .send((input_ids.copy(), payload.parameters.clone(), sx.clone()))
                // .send_timeout((input_ids.copy(), payload.parameters.clone(), sx.clone()), std::time::Duration::from_millis(0))
                .map_err(|_| GenerationError::QueueFull)?;
        }

        // TODO remove timeout
        let all_ids = rx
            .recv_timeout(std::time::Duration::from_secs(120))
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
) -> (Arc<Tokenizer>, Vec<Sender<(Tensor, Parameters, Sender<Tensor>)>>) {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    tch::maybe_init_cuda();
    let start = std::time::Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file("./tokenizer.json").unwrap());
    info!("Loaded tokenizer in {:?}", start.elapsed());
    info!("Starting threads");



    let (config, layout_config) = match model_name {
        "bigscience-small-testing" => (Config::new_testing(), LayoutConfig::new_testing()),
        "bloom-350m" => (Config::new350m(), LayoutConfig::new350m()),
        "bloom" => (Config::new(), LayoutConfig::new()),
        other => panic!("Model {other} is not known"),
    };

    let world_size = layout_config.world_size as i32;
    let id = ThreadGroup::new_id().unwrap();
    let mut channels = vec![];
    debug!("Spawning the threads");
    let (init_tx, init_rx) = bounded::<bool>(world_size as usize);
    for rank in 0..world_size {
        let config = config.clone();
        let layout_config = layout_config.clone();
        let (tx, rx) = bounded::<(Tensor, Parameters, Sender<Tensor>)>(1);
        channels.push(tx);
        let init_tx = init_tx.clone();
        debug!("Spawning the thread {rank}");
        tokio::task::spawn_blocking(move || {
            let group = ThreadGroup::new(world_size, rank, id).unwrap();
            debug!("Within blocking {rank}");
            thread(group, config, layout_config, rx, init_tx);
        });
    }
    for _ in 0..world_size {
        let ok = init_rx.recv().unwrap();
    }

    (tokenizer, channels)
}

#[actix_web::main] // or #[tokio::main]
async fn main() -> std::io::Result<()> {
    let model_name = std::env::var("BLOOM").unwrap_or_else(|_| "bloom".to_string());
    let (tokenizer, channels) = init_threads(&model_name);
    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(web::Data::new(AppState {
                tokenizer: tokenizer.clone(),
                channels: channels.clone(),
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
        let (tokenizer, channels) = init_threads("bigscience-small-testing");
        let app = test::init_service(
            App::new()
                .wrap(Logger::default())
                .app_data(web::Data::new(AppState { tokenizer, channels }))
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
            web::Bytes::from_static(b"[{\"generated_text\":\"testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest\"}]")
        );
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
    async fn test_tp_generation_350m() {
        let (tokenizer, channels) = init_threads("bloom-350m");
        info!("Started");
        let app = test::init_service(
            App::new()
                .wrap(Logger::default())
                .app_data(web::Data::new(AppState { tokenizer, channels }))
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
        // TODO
        // assert_eq!(
        //     bytes.unwrap(),
        //     web::Bytes::from_static(b"[{\"generated_text\":\"I enjoy walking my cute dog, but I also love to play with my cat. I am a very active person and I love\"}]")
        // );

        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "A \"whatpu\" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a \"farduddle\" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        // TODO
        // assert_eq!(
        //     bytes.unwrap(),
        //     web::Bytes::from_static(b"[{\"generated_text\":\"A \\\"whatpu\\\" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a \\\"farduddle\\\" means to jump up and down really fast. An example of a sentence that uses the word farduddle is: We were traveling in Africa and we saw these very cute whatpus. To do a \\\"fardudd\"}]")
        // );
    }

    #[actix_web::test]
    async fn test_tp_generation_full() {
        if std::env::var("RUN_SLOW") != Ok("1".to_string()) {
            return;
        }
        let (tokenizer, channels) = init_threads("bloom");
        info!("Started");
        let app = test::init_service(
            App::new()
                .wrap(Logger::default())
                .app_data(web::Data::new(AppState { tokenizer, channels }))
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
        // TODO
        // assert_eq!(
        //     bytes.unwrap(),
        //     web::Bytes::from_static(b"[{\"generated_text\":\"test.mark.parametrize('stdin, stdout', [\\n    ({'username': 'foo'\"}]")
        // );

        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "I enjoy walking my cute dog", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        // TODO
        // assert_eq!(
        //     bytes.unwrap(),
        //     web::Bytes::from_static(b"[{\"generated_text\":\"I enjoy walking my cute dog, reading, and spending time with my family. I am a very active person and love to be\"}]")
        // );

        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "Math exercise - answers:\n34+10=44\n54+20=", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        // TODO
        // assert_eq!(
        //     bytes.unwrap(),
        //     web::Bytes::from_static(b"[{\"generated_text\":\"Math exercise - answers:\\n34+10=44\\n54+20=74\\n74+20=94\\n94+20=114\\n114+20=134\\n\"}]")
        // );
        let req = test::TestRequest::post()
            .uri("/generate")
            .insert_header(ContentType::json())
            .set_json(serde_json::json!({"inputs": "A \"whatpu\" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a \"farduddle\" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:", "parameters": {"max_new_tokens": 20}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let body = resp.into_body();
        let bytes = actix_web::body::to_bytes(body).await;
        // TODO
        // assert_eq!(
        //     bytes.unwrap(),
        //     web::Bytes::from_static(b"[{\"generated_text\":\"A \\\"whatpu\\\" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a \\\"farduddle\\\" means to jump up and down really fast. An example of a sentence that uses the word farduddle is: The kids were jumping up and down on the trampoline and doing a farduddle. To\"}]")
        // );
    }
}
