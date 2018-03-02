use ndarray::{Array1, NdFloat};

/// An activation function in a Neural Network defines whether a neuron will
/// send a signal to its outputs or not.
pub trait Activation<F: NdFloat> {
    #[inline]
    fn compute(&self, x: &Array1<F>) -> Array1<F>;

    #[inline]
    fn compute_derivative(&self, x: &Array1<F>) -> Array1<F>;
}

/// The Identity function.
pub struct Identity;
impl<F: NdFloat> Activation<F> for Identity {
    #[inline]
    fn compute(&self, x: &Array1<F>) -> Array1<F> {
        x.clone()
    }
    #[inline]
    fn compute_derivative(&self, x: &Array1<F>) -> Array1<F> {
        Array1::from_vec(vec![F::one(); x.dim()])
    }
}

/// The Sigmoid function squashes a real value into the ]0, 1[ range.
pub struct Sigmoid;
impl<F: NdFloat> Activation<F> for Sigmoid {
    #[inline]
    fn compute(&self, x: &Array1<F>) -> Array1<F> {
        x.map(|v: &F| F::one() / (F::one() + (-*v).exp()))
    }
    #[inline]
    fn compute_derivative(&self, x: &Array1<F>) -> Array1<F> {
        x.map(|v: &F| {
            let y = F::one() / (F::one() + (-*v).exp());
            y * (F::one() - y)
        })
    }
}

/// The Hyperbolic tangent squashes a real value into the ]-1, 1[ range.
pub struct TanH;
impl<F: NdFloat> Activation<F> for TanH {
    #[inline]
    fn compute(&self, x: &Array1<F>) -> Array1<F> {
        x.map(|v| v.tanh())
    }
    #[inline]
    fn compute_derivative(&self, x: &Array1<F>) -> Array1<F> {
        x.map(|v| F::one() - v.tanh().powi(2))
    }
}

/// The Rectified Linear Unit (ReLU) functions replaces negative values with 0.
pub struct Rectifier;
impl<F: NdFloat> Activation<F> for Rectifier {
    #[inline]
    fn compute(&self, x: &Array1<F>) -> Array1<F> {
        let zero = F::zero();
        x.map(|v| if *v < zero { zero } else { *v })
    }
    #[inline]
    fn compute_derivative(&self, x: &Array1<F>) -> Array1<F> {
        let (zero, one) = (F::zero(), F::one());
        x.map(|v| if *v < zero { zero } else { one })
    }
}

#[cfg(test)]
mod tests {
    use super::super::Float;
    use super::*;

    fn test_numerical_function<A>(
        function: A,
        inputs: Array1<Float>,
        values: Array1<Float>,
        derivatives: Array1<Float>,
    ) where
        A: Activation<Float>,
    {
        assert_eq!(inputs.len(), values.len());
        assert_eq!(inputs.len(), derivatives.len());
        let computed_values = function.compute(&inputs);
        let computed_derivatives = function.compute_derivative(&inputs);
        for i in 0..inputs.len() {
            assert_relative_eq!(computed_values[i], values[i]); // 16 digits precision by default
            assert_relative_eq!(computed_derivatives[i], derivatives[i]);
        }
    }

    #[test]
    fn identity() {
        let inputs = array![-23.0, -7.0, 0.0, 3.0, 10.0];
        test_numerical_function(
            Identity,
            inputs.clone(),
            inputs,
            Array1::from_vec(vec![1.0; 5]),
        );
    }

    #[test]
    fn sigmoid() {
        test_numerical_function(
            Sigmoid,
            array![-2.0, -1.0, 0.0, 1.0, 2.0],
            array![
                0.1192029220221175,
                0.2689414213699951,
                0.5,
                0.7310585786300048,
                0.8807970779778824,
            ],
            array![
                0.1049935854035065,
                0.1966119332414819,
                0.25,
                0.1966119332414819,
                0.1049935854035066,
            ],
        );
    }

    #[test]
    fn tanh() {
        test_numerical_function(
            TanH,
            array![-1.0, -0.5, 0.0, 0.5, 1.0],
            array![
                -0.7615941559557649,
                -0.4621171572600097,
                0.0,
                0.4621171572600097,
                0.7615941559557649,
            ],
            array![
                0.4199743416140261,
                0.7864477329659274,
                1.0,
                0.7864477329659274,
                0.4199743416140261,
            ],
        );
    }

    #[test]
    fn relu() {
        test_numerical_function(
            Rectifier,
            array![-150.0, -7.0, 0.0, 3.0, 10.0],
            array![0.0, 0.0, 0.0, 3.0, 10.0],
            array![0.0, 0.0, 1.0, 1.0, 1.0],
        );
    }
}
