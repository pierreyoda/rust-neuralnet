use ndarray::{Array1, Array2};

use super::{Float, ResultString};

/// A Sample contains the vector of the observed values of all the inputs
/// of an Artificial Neural Network.
/// If the Sample is part of the training dataset, it must also contain the
/// outputs vector. Otherwise, we can use it to predict the output(s).
#[derive(Clone, Debug)]
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

// TODO: refactor using ndarray's Zip
pub fn prepare_dataset(dataset: &Vec<Sample>) -> ResultString<(Array2<Float>, Array2<Float>)> {
    if dataset.is_empty() {
        return Err("empty dataset".into());
    }
    let maybe_sample = dataset.first().unwrap().clone(); // TODO: avoid clone ?
    if maybe_sample.outputs.is_none() {
        return Err("dataset error : no observed output for the sample of index 0".into());
    }
    let inputs_number = maybe_sample.inputs.len();
    let outputs_number = maybe_sample.outputs.unwrap().len();

    let mut inputs = Array2::zeros((dataset.len(), inputs_number));
    let mut observed_outputs = Array2::zeros((dataset.len(), outputs_number));
    for i in 0..dataset.len() {
        let sample_outputs = match &dataset[i].outputs {
            &Some(ref o) => o,
            &None => {
                return Err(format!(
                    "dataset error : no observed output for the sample of index {}",
                    i,
                ))
            }
        };

        let sample_inputs = &dataset[i].inputs;
        if sample_inputs.len() != inputs_number {
            return Err(format!(
                "dataset error for sample of index {}: inputs count mismatch
                ({} instead of the expected {})",
                i,
                sample_inputs.len(),
                inputs_number,
            ));
        }
        for j in 0..inputs_number {
            let e = inputs.get_mut((i, j)).unwrap();
            *e = sample_inputs[j];
        }

        if sample_outputs.len() != outputs_number {
            return Err(format!(
                "dataset error for sample of index {}: outputs count mismatch
                ({} instead of the expected {})",
                i,
                sample_outputs.len(),
                outputs_number,
            ));
        }
        for j in 0..outputs_number {
            let e: &mut Float = observed_outputs.get_mut((i, j)).unwrap();
            *e = sample_outputs[j];
        }
    }
    Ok((inputs, observed_outputs))
}
