#![feature(async_closure)]
pub mod generation;
pub mod layout;
pub mod model;
#[cfg(test)]
mod test;
pub mod utils;

use crate::model::{Config, Past};
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
    #[error("Queue is full")]
    QueueFull,
}

impl ResponseError for GenerationError {
    fn status_code(&self) -> StatusCode {
        match self {
            GenerationError::QueueFull => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

pub fn empty_past(config: &Config) -> Past {
    let kind = (config.kind, Device::Cuda(0));
    let p = config.n_head;
    let q = config.hidden_size / config.n_head;
    let past_key = Tensor::zeros(&[1, 0, p, q], kind);
    let past_value = Tensor::zeros(&[1, 0, p, q], kind);
    let past_key_values: Vec<_> = (0..config.n_layer)
        .map(|_| (past_key.copy(), past_value.copy()))
        .collect();
    past_key_values
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
