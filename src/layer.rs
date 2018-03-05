use rand::Rng;
use rand::distributions::Range;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Ix2, NdFloat, Zip};

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
    // cached results
    layer_inputs_sum: Array2<F>,
    layer_inputs_sum_activated: Array2<F>,
    layer_outputs_sum: Array2<F>,
    backprop_error_1: Array2<F>,
    backprop_error_2: Array2<F>,
    costs: Array1<F>,
    cost_d_inputs: Array2<F>,
    cost_d_outputs: Array2<F>,
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
            layer_inputs_sum: Array2::zeros((0, 0)),
            layer_inputs_sum_activated: Array2::zeros((0, 0)),
            layer_outputs_sum: Array2::zeros((0, 0)),
            backprop_error_1: Array2::zeros((0, 0)),
            backprop_error_2: Array2::zeros((0, 0)),
            costs: Array1::zeros(0),
            cost_d_inputs: Array2::zeros((0, 0)),
            cost_d_outputs: Array2::zeros((0, 0)),
        }
    }

    /// Compute and store the outputs of the layer using forward propagation.
    /// The output vector will be stored within the layer and a read-only
    /// ```ArrayView``` of it will be returned.
    ///
    /// The output matrix is given by the values of the activation function
    /// evaluated for each neuron at the weighted sum of the layer.
    ///
    /// ## Input
    ///
    /// `inputs`: ([samples] * [inputs])
    ///
    /// ## Intermediate results
    ///
    /// - layer_inputs_sum
    ///   : ([samples] * [inputs]) * ([inputs] * [neurons])
    ///   : ([samples] * [neurons])
    ///   = inputs * inputs_weights
    ///
    /// - layer_inputs_sum_activated = activation(layer_sum)
    ///
    /// - layer_outputs_sum
    ///   : ([samples * [neurons]) * ([neurons] * [outputs])
    ///   : ([samples] * [outputs])
    ///   = layers_inputs_sum_activated * outputs_weights
    ///
    ///
    /// ## Output
    ///
    /// Returns a view to the outputs as estimated by the Layer for the given `inputs`.
    /// outputs
    /// : ([samples] * [output])
    /// = activation(outputs_sum)
    ///
    pub fn forward_propagation(&mut self, inputs: &ArrayView2<F>) -> ResultString<ArrayView2<F>> {
        if inputs.cols() != self.inputs_weights.rows() {
            return Err(format!(
                "Layer.forward : inputs size mismatch (inputs cols = {} != {} = weights rows)",
                inputs.cols(),
                self.inputs_weights.rows(),
            ));
        }
        println!("rezrezrez\n{}\n{}\n\n\n", inputs, self.inputs_weights);
        self.layer_inputs_sum = inputs.dot(&self.inputs_weights);
        self.layer_inputs_sum_activated = self.activation.compute(&self.layer_inputs_sum);
        self.layer_outputs_sum = self.layer_inputs_sum_activated.dot(&self.outputs_weights);
        self.outputs = self.activation.compute(&self.layer_outputs_sum);
        Ok(self.outputs.view())
    }

    /// Compute and store the gradient of the Mean Squared Error cost function
    /// for the current ```Layer```.
    ///
    /// ## Input
    ///
    /// - `inputs`: ([samples] * [inputs])
    ///
    /// - `expected_outputs`: ([samples] * [outputs])
    ///
    /// ## Intermediate results
    ///
    /// .: = element-wise multiplication
    ///
    /// - `backprop_error_1`
    ///   : ([samples] * [outputs])
    ///   = - (self.outputs - expected_outputs) .* activation_derivative(self.layer_outputs_sum)
    ///
    /// - `cost_d_outputs`: partial derivative of the cost with respect to the outputs weights
    ///   : ([neurons] * [samples]) * ([samples] * [outputs]) = ([neurons] * [outputs])
    ///   = self.layer_inputs_sum_activated.transposed() * backprop_error_1
    ///
    /// - `backprop_error_2`
    ///   : ([samples] * [outputs]) * ([outputs] * [neurons]) = ([samples] * [neurons])
    ///   = (backprop_error_1 * outputs_weights.transposed()) .* activation_derivative(self.layer_inputs_sum)
    ///
    /// - `cost_d_inputs`: partial derivative of the cost with respect to the inputs weights
    ///   : ([inputs] * [samples]) * ([samples] * [neurons]) = ([inputs] * [neurons])
    ///   = inputs.transposed() * backprop_error_2
    ///
    /// ## Output
    ///
    /// Returns a view to the gradient of the cost function.
    ///
    pub fn cost_gradient_mse(
        &mut self,
        inputs: &ArrayView2<F>,
        expected_outputs: &ArrayView2<F>,
    ) -> (ArrayView2<F>, ArrayView2<F>) {
        let outputs_derivative = self.activation.compute_derivative(&self.layer_outputs_sum);
        let outputs_delta = expected_outputs - &self.outputs;
        self.backprop_error_1 = outputs_delta * outputs_derivative;
        self.cost_d_outputs = self.layer_inputs_sum_activated
            .t()
            .dot(&self.backprop_error_1);

        let inputs_derivative = self.activation.compute_derivative(&self.layer_inputs_sum);
        self.backprop_error_2 =
            self.backprop_error_1.dot(&self.outputs_weights.t()) * inputs_derivative;
        self.cost_d_inputs = inputs.t().dot(&self.backprop_error_2);

        (self.cost_d_inputs.view(), self.cost_d_outputs.view())
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

    /// Compute and store the "score" of our current outputs evaluation compared
    /// to the expected outputs using the Mean Squared Error cost function.
    ///
    /// ## Input
    ///
    /// `expected_outputs`: ([samples] * [outputs])
    ///
    /// ## Output
    /// Returns a view to the evaluated cost vector.
    ///
    /// costs
    /// : (1 * [ouputs])
    /// = 1/2 * sum((expected_output - output) ^ 2)
    pub fn cost_mse(
        &mut self,
        inputs: &ArrayView2<Float>,
        expected_outputs: &ArrayView2<Float>,
    ) -> ArrayView1<Float> {
        let mut squared_diffs = Array2::zeros(expected_outputs.dim());
        Zip::from(&mut squared_diffs)
            .and(&self.outputs)
            .and(expected_outputs)
            .apply(|d, expected, approx| *d = (expected - approx).powi(2));

        self.costs = Array1::zeros(expected_outputs.cols());
        Zip::from(&mut self.costs)
            .and(squared_diffs.gencolumns())
            .apply(|c, d_row| *c = 0.5 * d_row.scalar_sum());
        self.costs.view()
    }
}
