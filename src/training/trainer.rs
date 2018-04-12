use ndarray::Array2;

use super::super::{Float, ResultString};
use network::NeuralNetwork;
use super::{prepare_dataset, Sample};

pub enum TrainerHaltCondition {
    Epochs(u32),
}

pub struct Trainer {
    inputs: Array2<Float>,
    outputs: Array2<Float>,
    network: NeuralNetwork,
    halt_condition: TrainerHaltCondition,
}

impl Trainer {
    pub fn with_dataset(network: NeuralNetwork, dataset: &Vec<Sample>) -> ResultString<Self> {
        match prepare_dataset(dataset) {
            Ok((inputs, outputs)) => Ok(Trainer {
                inputs,
                outputs,
                network,
                halt_condition: TrainerHaltCondition::Epochs(1),
            }),
            Err(why) => Err(why),
        }
    }

    pub fn halt_condition(mut self, halt_condition: TrainerHaltCondition) -> Option<Self> {
        use self::TrainerHaltCondition::*;
        match halt_condition {
            Epochs(epochs) => if epochs == 0 {
                None
            } else {
                Some(self)
            },
        }
    }
}
