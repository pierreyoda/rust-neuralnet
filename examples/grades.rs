//! Given a ((hours of sleep), (hours of study) => (grade)) dataset,
//! predict a grade for any ((hours of sleep), (hours of stury)) input tuple
//! using an Artificial Neural Network.

extern crate ndarray;
extern crate rand;
extern crate rust_neuralnet;

use ndarray::arr2;
use rand::thread_rng;

use rust_neuralnet::activation::Sigmoid;
use rust_neuralnet::builder::NeuralNetworkBuilder;
use rust_neuralnet::training::{Sample, prepare_dataset};

fn main() {
    let mut rng = rand::thread_rng();

    let dataset = vec![
        Sample::dataset(vec![3.0, 5.0], vec![75.0]),
        Sample::dataset(vec![5.0, 1.0], vec![82.0]),
        Sample::dataset(vec![10.0, 2.0], vec![93.0]),
    ];
    let (inputs, expected_outputs) = prepare_dataset(&dataset).unwrap();

    let mut neural_network = NeuralNetworkBuilder::with_inputs(2)
        .output(3, 1, Sigmoid, &mut rng);

    {
        let outputs = neural_network.run_forward(inputs.view());
        println!("ANN : forward_propag outputs=\n{:?}", outputs);
    }

    println!("expected outputs = {:?}", expected_outputs);

    neural_network.backward_propagation(inputs.view(), expected_outputs.view());
}
