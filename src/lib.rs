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
use serde::{Deserialize, Serialize};
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
    #[error("{{\"error\": \"Could not receive any answer an internal error occurred\"}}")]
    CouldNotReceiveAnswer,
}

impl ResponseError for GenerationError {
    fn status_code(&self) -> StatusCode {
        match self {
            GenerationError::QueueFull => StatusCode::SERVICE_UNAVAILABLE,
            GenerationError::InputTooLong => StatusCode::BAD_REQUEST,
            GenerationError::TooManyNewTokens => StatusCode::BAD_REQUEST,
            GenerationError::CouldNotReceiveAnswer => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
