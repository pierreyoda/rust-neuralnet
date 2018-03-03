extern crate ndarray;
extern crate rand;
extern crate rust_neuralnet;

use ndarray::arr2;
use rand::thread_rng;

use rust_neuralnet::activation::Sigmoid;
use rust_neuralnet::builder::NeuralNetworkBuilder;
use rust_neuralnet::training::Sample;

/// Train a Neural Network to replicate the XOR (exclusive) function with
/// a single hidden layer.
///
/// XOR :
/// Input A | Input B => Output = A XOR B
/// 0         0          0
/// 0         1          1
/// 1         0          1
/// 1         1          0
#[test]
fn xor() {
    let mut rng = rand::thread_rng();
    let (t, f) = (1.0, 0.0);
    let dataset = vec![
        Sample::dataset(vec![f, f], vec![f]),
        Sample::dataset(vec![f, t], vec![t]),
        Sample::dataset(vec![t, f], vec![t]),
        Sample::dataset(vec![t, t], vec![f]),
    ];

    let mut neural_network = NeuralNetworkBuilder::with_inputs(2)
        .layer(2, Sigmoid, &mut rng)
        .output(1, 1, Sigmoid, &mut rng);
    let inputs = arr2(&[[t, f]]);

    // feed-forward propagation test (temporary)
    let result = neural_network.run_forward(inputs.view());
    assert!(result.is_ok());
}
