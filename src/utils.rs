use log::debug as log_debug;
use tch::{kind::Kind, Device, IndexOp, Tensor};

const DEBUG: bool = false;
pub const SAVE: bool = false;

pub fn debug_force(prefix: &str, x: &Tensor) {
    let size = x.size();
    let batch_size = size[0];
    let second_size = if size.len() > 1 { size[1] } else { batch_size };
    log_debug!(
        "{prefix} - {:?} - Values: {:?}",
        size,
        x.reshape(&[-1,])
            .iter::<f64>()
            .unwrap()
            .take(std::cmp::min(second_size as usize, 10))
            .collect::<Vec<_>>()
    );
    if batch_size > 1 {
        log_debug!(
            "                          {:?}",
            x.i(1)
                .reshape(&[-1,])
                .iter::<f64>()
                .unwrap()
                .take(30)
                .collect::<Vec<_>>()
        );
    }
}

pub fn debug(prefix: &str, x: &Tensor) {
    if DEBUG {
        debug_force(prefix, x);
    }
}
pub fn save_layer_to_disk(tensor: &Tensor, filename: &str) {
    if SAVE {
        let step = std::env::var("GENERATION_STEP").unwrap_or_else(|_| "nostep".to_string());
        let tensor = if tensor.kind() == Kind::BFloat16 || tensor.kind() == Kind::Bool {
            tensor.to_device(Device::Cpu).to_kind(Kind::Float)
        } else {
            tensor.to_device(Device::Cpu)
        };
        tensor
            .write_npy(&format!("tensors/{}_{}", step, filename))
            .unwrap();
    }
}
