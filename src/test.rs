use tch::Tensor;

pub fn assert_all_close(left: &Tensor, right: &Tensor) {
    if !left.allclose(right, 1e-7, 1e-7, false) {
        left.print();
        right.print();
        panic!("{left:?} is not close to {right:?}");
    }
}
