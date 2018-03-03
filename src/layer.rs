use rand::Rng;
use rand::distributions::Range;
use ndarray::{Array2, ArrayView2, Ix2, NdFloat};

use super::{Float, ResultString};
use activation::Activation;
use utils::NdArrayRandomizer;

/// A layer of artificial Neurons within an Artificial Neural Network.
///
/// A ```Layer``` has multiple inputs, each one with its own weight that will
/// determine its influence on the outputs.
///
/// Example with a single hidden layer with one output:
///
/// w[x][y] = weight of the input n°x for neuron n°y
/// wo[z]   = weight of the output for neuron n°z
///
/// [Input 1] --> (w11) --> Hidden 1
///           --> (w12) --> Hidden 2
///           --> (w13) --> Hidden 3
/// [Input 2] --> (w21) --> Hidden 1
///           --> (w22) --> Hidden 2
///           --> (w23) --> Hidden 3
/// > [Output] => (wo1, wo2, wo3)
///
/// The layer's input weights can be described as a 2*3 matrix (inputs * neurons):
/// [
///     w11, w12, w13,
///     w21, w22, w23,
/// ]
/// The layer's outputs weights can be described as the 3*1 matrix (neurons * outputs) :
/// [
///     w01,
///     w02,
///     w03,
/// ]
///
///
pub struct Layer<F: NdFloat> {
    activation: Box<Activation<F, Ix2>>,
    inputs_weights: Array2<F>,
    outputs: Array2<F>,
    outputs_weights: Array2<F>,
}

impl<F: NdFloat> Layer<F> {
    pub fn new<A: 'static>(
        activation: A,
        inputs_weights: Array2<F>,
        outputs_weights: Array2<F>,
    ) -> Self
    where
        A: Activation<F, Ix2>,
    {
        assert_eq!(inputs_weights.cols(), outputs_weights.rows());
        let dim = inputs_weights.dim();
        Layer {
            activation: Box::new(activation),
            inputs_weights,
            outputs: Array2::zeros(dim),
            outputs_weights: outputs_weights,
        }
    }

    /// Compute the outputs of the layer using forward propagation.
    /// The output vector will be stored within the layer and a read-only
    /// ```ArrayView``` of it will be returned.
    ///
    /// The output matrix is given by the values of the activation function
    /// evaluated for each neuron at the weighted sum of the layer:
    /// weighted_sum = [input matrix] * [input weights matrix]
    ///              : example with 3 neurons and 4 samples for both of two outputs
    ///              = [i11 i21          [
    ///                 i21 i22     *     w11, w12, w13,
    ///                 i31 i32
    ///                 i41 i42           w21, w22, w23
    ///                ]                 ]
    ///              : (4 samples * 2 inputs) * (2 inputs * 3 neurons)
    ///              : (4 samples * 3 neurons) matrix
    /// result       = activation_function(layer_sum)
    /// output_sum   = [result matrix] * [output weights matrix]
    ///              : (4 samples * 3 neurons) * (3 neurons * 1 output)
    ///              : (4 samples * 1 output) matrix
    /// output       = activation_function(output_sum)
    ///              : (4 samples * 1 output) matrix
    pub fn forward(&mut self, inputs: ArrayView2<F>) -> ResultString<ArrayView2<F>> {
        if inputs.cols() != self.inputs_weights.rows() {
            return Err(format!(
                "Layer.forward : inputs size mismatch (inputs cols = {} != {} = weights rows)",
                inputs.cols(),
                self.inputs_weights.rows(),
            ));
        }
        let layer_sum = inputs.dot(&self.inputs_weights);
        let layer_result = self.activation.compute(&layer_sum);
        let output_sum = layer_result.dot(&self.outputs_weights);
        self.outputs = self.activation.compute(&output_sum);
        Ok(self.outputs.view())
    }
}

impl Layer<Float> {
    pub fn with_random_weights<A: 'static, R>(
        activation: A,
        dim_inputs: usize,
        dim_neurons: usize,
        dim_outputs: usize,
        rng: &mut R,
    ) -> Self
    where
        A: Activation<Float, Ix2>,
        R: Rng,
    {
        let inputs_weights =
            Array2::<Float>::random((dim_inputs, dim_neurons), Range::new(0.0, 1.0), rng);
        let outputs_weights =
            Array2::<Float>::random((dim_neurons, dim_outputs), Range::new(0.0, 1.0), rng);
        Layer::new(activation, inputs_weights, outputs_weights)
    }
}
