#![feature(async_closure)]
pub mod generation;
pub mod layout;
pub mod model;
#[cfg(test)]
mod test;
pub mod tp_layers;
pub mod tp_layout;
pub mod utils;

pub use crate::model::{empty_past, non_empty_past};
use actix_web::{http::StatusCode, ResponseError};
use safetensors::{Dtype, TensorView};
use serde::{Deserialize, Serialize};
use tch::{kind, Device, Tensor};
use thiserror::Error;

use crate::generation::Parameters;
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Generation {
    pub inputs: String,
    #[serde(default)]
    pub parameters: Parameters,
}

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("{{\"error\": \"Queue is full\"}}")]
    QueueFull,
    #[error(
        "{{\"error\": \"Input is too long (128 tokens). We're disabling long prompts temporarily\"}}"
    )]
    InputTooLong,
    #[error("{{\"error\": \"We can't generate more than 64 tokens at a time. We're disabling long generations temporarily\"}}")]
    TooManyNewTokens,
}

impl ResponseError for GenerationError {
    fn status_code(&self) -> StatusCode {
        match self {
            GenerationError::QueueFull => StatusCode::SERVICE_UNAVAILABLE,
            GenerationError::InputTooLong => StatusCode::BAD_REQUEST,
            GenerationError::TooManyNewTokens => StatusCode::BAD_REQUEST,
        }
    }
}

pub fn convert(view: TensorView, device: Device) -> Tensor {
    let kind = match view.get_dtype() {
        Dtype::F16 => kind::Kind::Half,
        Dtype::BF16 => kind::Kind::BFloat16,
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
