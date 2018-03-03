use ndarray::Array1;

use super::Float;

/// A Sample contains the vector of the observed values of all the inputs
/// of an Artificial Neural Network.
/// If the Sample is part of the training dataset, it must also contain the
/// outputs vector. Otherwise, we can use it to predict the output(s).
#[derive(Debug)]
pub struct Sample {
    inputs: Array1<Float>,
    outputs: Option<Array1<Float>>,
}

impl Sample {
    pub fn dataset<V>(inputs: V, outputs: V) -> Self
    where
        V: Into<Array1<Float>>,
    {
        Sample {
            inputs: inputs.into(),
            outputs: Some(outputs.into()),
        }
    }

    pub fn predict<V>(inputs: V) -> Self
    where
        V: Into<Array1<Float>>,
    {
        Sample {
            inputs: inputs.into(),
            outputs: None,
        }
    }
}
