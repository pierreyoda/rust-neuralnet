use ndarray::NdFloat;

use layer::Layer;

type Float = f64;

/// An Artificial Neural Network mimics the behavior of real nervous systems
/// by simulating Neurons (grouped by ```Layer```).
///
/// The Neural Network is composed of several ```Layer```s.
pub struct NeuralNetwork {
    layers: Vec<Layer<Float>>,
}
