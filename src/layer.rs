use ndarray::{NdFloat, Array1, Array2};

use activation::Activation;

/// A layer of artificial Neurons within an artificial Neural Network.
///
/// A ```Layer``` has multiple inputs, each one with its own weight that will
/// determine its influence on the outputs.
pub struct Layer<F: NdFloat> {
    activation: Box<Activation<F>>,
    weights: Array2<F>,
    output: Array1<F>,
}

impl<F: NdFloat> Layer<F> {
    pub fn new<A: 'static>(activation: A, weights: Array2<F>) -> Self
    where A: Activation<F> {
        let (dim, _) = weights.dim();
        Layer {
            activation: Box::new(activation),
            weights,
            output: Array1::zeros(dim),
        }
    }
}
