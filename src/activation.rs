use ndarray::NdFloat;

/// An activation function in a Neural Network defines whether a neuron will
/// send a signal to its outputs or not.
pub trait Activation<F: NdFloat> {
    fn compute(&self, x: F) -> F;
}

pub struct Sigmoid;
impl<F: NdFloat> Activation<F> for Sigmoid {
    fn compute(&self, x: F) -> F {
        F::one() / (F::one() + (-x).exp())
    }
}
