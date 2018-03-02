use ndarray::NdFloat;

/// An activation function in a Neural Network defines whether a neuron will
/// send a signal to its outputs or not.
pub trait Activation<F: NdFloat> {
    fn compute(&self, x: F) -> F;
}

/// The Sigmoid function squashes a real value into the ]0, 1[ range.
pub struct Sigmoid;
impl<F: NdFloat> Activation<F> for Sigmoid {
    fn compute(&self, x: F) -> F {
        F::one() / (F::one() + (-x).exp())
    }
}

/// The Hyperbolic tangent squashes a real value into the ]-1, 1[ range.
pub struct TanH;
impl<F: NdFloat> Activation<F> for TanH {
    fn compute(&self, x: F) -> F {
        x.tanh()
    }
}

/// The Rectified Linear Unit (ReLU) functions replaces negative values with 0.
pub struct Rectifier;
impl<F: NdFloat> Activation<F> for Rectifier {
    fn compute(&self, x: F) -> F {
        if x < F::zero() { F::zero() } else { x }
    }
}
