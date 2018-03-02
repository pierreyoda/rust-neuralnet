use rand::Rng;
use rand::distributions::Range;
use ndarray::{Array1, Array2, NdFloat};

use super::Float;
use activation::Activation;
use utils::NdArrayRandomizer;

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
    where
        A: Activation<F>,
    {
        let (dim, _) = weights.dim();
        Layer {
            activation: Box::new(activation),
            weights,
            output: Array1::zeros(dim),
        }
    }
}

impl Layer<Float> {
    pub fn with_random_weights<A: 'static, R>(
        activation: A,
        dim_inputs: usize,
        dim_outputs: usize,
        rng: &mut R,
    ) -> Self
    where
        A: Activation<Float>,
        R: Rng,
    {
        let weights =
            Array2::<Float>::random((dim_inputs, dim_outputs), Range::new(-1.0, 1.0), rng);
        Layer::new(activation, weights)
    }
}
