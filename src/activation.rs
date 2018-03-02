use ndarray::NdFloat;

/// An activation function in a Neural Network defines whether a neuron will
/// send a signal to its outputs or not.
pub trait Activation<F: NdFloat> {
    #[inline]
    fn compute(&self, x: F) -> F;
}

/// The Identity function.
pub struct Identity;
impl<F: NdFloat> Activation<F> for Identity {
    #[inline]
    fn compute(&self, x: F) -> F {
        x
    }
}

/// The Sigmoid function squashes a real value into the ]0, 1[ range.
pub struct Sigmoid;
impl<F: NdFloat> Activation<F> for Sigmoid {
    #[inline]
    fn compute(&self, x: F) -> F {
        F::one() / (F::one() + (-x).exp())
    }
}

/// The Hyperbolic tangent squashes a real value into the ]-1, 1[ range.
pub struct TanH;
impl<F: NdFloat> Activation<F> for TanH {
    #[inline]
    fn compute(&self, x: F) -> F {
        x.tanh()
    }
}

/// The Rectified Linear Unit (ReLU) functions replaces negative values with 0.
pub struct Rectifier;
impl<F: NdFloat> Activation<F> for Rectifier {
    #[inline]
    fn compute(&self, x: F) -> F {
        if x < F::zero() {
            F::zero()
        } else {
            x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Float;
    use super::*;

    fn test_numerical_function<A>(function: A, inputs: Vec<f64>, outputs: Vec<f64>)
    where
        A: Activation<Float>,
    {
        assert_eq!(inputs.len(), outputs.len());
        for i in 0..inputs.len() {
            let result = function.compute(inputs[i]);
            assert_relative_eq!(result, outputs[i]);
        }
    }

    #[test]
    fn identity() {
        let inputs = vec![-23.0, -7.0, 0.0, 3.0, 10.0];
        test_numerical_function(Identity, inputs.clone(), inputs);
    }

    #[test]
    fn sigmoid() {
        test_numerical_function(
            Sigmoid,
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![
                0.1192029220221175,
                0.2689414213699951,
                0.5,
                0.7310585786300048,
                0.8807970779778824,
            ],
        );
    }

    #[test]
    fn tanh() {
        test_numerical_function(
            TanH,
            vec![-1.0, -0.5, 0.0, 0.5, 1.0],
            vec![
                -0.7615941559557649,
                -0.4621171572600097,
                0.0,
                0.4621171572600097,
                0.7615941559557649,
            ],
        );
    }

    #[test]
    fn relu() {
        test_numerical_function(
            Rectifier,
            vec![-150.0, -7.0, 0.0, 3.0, 10.0],
            vec![0.0, 0.0, 0.0, 3.0, 10.0],
        );
    }
}
