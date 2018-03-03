use ndarray::ArrayView2;

use super::{Float, ResultString};
use layer::Layer;

/// An Artificial Neural Network mimics the behavior of real nervous systems
/// by simulating Neurons (grouped by ```Layer```).
///
/// The Neural Network is composed of several ```Layer```s.
pub struct NeuralNetwork {
    layers: Vec<Layer<Float>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer<Float>>) -> Self {
        NeuralNetwork { layers }
    }

    /// Perform simple forward propagation accross the layers and return an
    /// ```ÀrrayView``` to the last layer's output.
    pub fn run_forward(&mut self, inputs: ArrayView2<Float>) -> ResultString<ArrayView2<Float>> {
        let mut layer_result = Err("NeuralNetwork.run_foward : no layers defined.".into());
        for layer in &mut self.layers {
            layer_result = layer.forward(inputs);
        }
        layer_result
    }
}
