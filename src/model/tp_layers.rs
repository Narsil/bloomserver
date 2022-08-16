use nccl_rs::ThreadGroup;
use std::rc::Rc;
use tch::Tensor;

/// Different from usual TensorParallelColumnLinear in order to remove transpose operation:
/// weight: [in_features, out_features]
/// bias: [out_features]
pub struct TensorParallelColumnLinear {
    weight: Tensor,
    bias: Tensor,
}

impl TensorParallelColumnLinear {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        let in_features = self.weight.size()[0];
        let out_features = self.weight.size()[1];
        let mut out_size = xs.size();
        if let Some(last) = out_size.last_mut() {
            *last = out_features;
        }

        let flatten_xs = xs.view((-1, in_features));

        self.bias
            .f_addmm(&flatten_xs, &self.weight)
            .unwrap()
            .f_view(out_size.as_slice())
            .unwrap()
    }
}

/// Different from usual TensorParallelRowLinear in order to remove transpose operation:
/// weight: [in_features, out_features]
/// bias: [out_features]
pub struct TensorParallelRowLinear {
    weight: Tensor,
    bias: Tensor,
    group: Rc<ThreadGroup>,
}

impl TensorParallelRowLinear {
    pub fn new(weight: Tensor, bias: Tensor, group: Rc<ThreadGroup>) -> Self {
        Self {
            weight,
            bias,
            group,
        }
    }
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        let in_features = self.weight.size()[0];
        let out_features = self.weight.size()[1];
        let mut out_size = xs.size();
        if let Some(last) = out_size.last_mut() {
            *last = out_features;
        }

        let flatten_xs = xs.view((-1, in_features));

        let out = self
            .bias
            .f_addmm(&flatten_xs, &self.weight)
            .unwrap()
            .f_view(out_size.as_slice())
            .unwrap();
        self.group.all_reduce(out).unwrap()
    }
}
