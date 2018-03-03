//! Facilities meant to make the definition of common Artificial Neural Network
//! topologies easier.

use rand::Rng;
use ndarray::Ix2;

use super::Float;
use layer::Layer;
use network::NeuralNetwork;
use activation::Activation;

pub struct NeuralNetworkBuilder {
    /// Number of outputs of the current last layer.
    last_layer_outputs: usize,
    layers: Vec<Layer<Float>>,
}

impl NeuralNetworkBuilder {
    pub fn with_inputs(inputs: usize) -> Self {
        assert!(inputs > 0, "An ANN requires at least 1 input.");
        NeuralNetworkBuilder {
            last_layer_outputs: inputs,
            layers: Vec::new(),
        }
    }

    /// Add a hidden layer with the specified topology and activation function.
    pub fn layer<A: 'static, R>(mut self, neurons: usize, activation: A, rng: &mut R) -> Self
    where
        A: Activation<Float, Ix2>,
        R: Rng,
    {
        debug_assert!(self.last_layer_outputs > 0);
        let layer =
            Layer::with_random_weights(activation, self.last_layer_outputs, neurons, neurons, rng);
        self.layers.push(layer);
        self.last_layer_outputs = neurons;
        self
    }

    pub fn output<A: 'static, R>(
        mut self,
        neurons: usize,
        outputs: usize,
        activation: A,
        rng: &mut R,
    ) -> NeuralNetwork
    where
        A: Activation<Float, Ix2>,
        R: Rng,
    {
        assert!(self.layers.len() > 0, "NeuralNetworkBuilder : no output ");
        debug_assert!(self.last_layer_outputs > 0);
        let last_layer =
            Layer::with_random_weights(activation, self.last_layer_outputs, neurons, outputs, rng);
        self.layers.push(last_layer);
        self.last_layer_outputs = outputs;
        NeuralNetwork::new(self.layers)
    }
}
