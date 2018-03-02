use ndarray::NdFloat;

/// An activation function in a Neural Network defines whether a neuron will
/// send a signal to its outputs or not.
pub trait Activation<F: NdFloat> {
    #[inline]
    fn compute(&self, x: F) -> F;

    #[inline]
    fn compute_derivative(&self, x: F) -> F;
}

/// The Identity function.
pub struct Identity;
impl<F: NdFloat> Activation<F> for Identity {
    #[inline]
    fn compute(&self, x: F) -> F {
        x
    }
    #[inline]
    fn compute_derivative(&self, _: F) -> F {
        F::one()
    }
}

/// The Sigmoid function squashes a real value into the ]0, 1[ range.
pub struct Sigmoid;
impl<F: NdFloat> Activation<F> for Sigmoid {
    #[inline]
    fn compute(&self, x: F) -> F {
        F::one() / (F::one() + (-x).exp())
    }
    #[inline]
    fn compute_derivative(&self, x: F) -> F {
        let y = self.compute(x);
        y * (F::one() - y)
    }
}

/// The Hyperbolic tangent squashes a real value into the ]-1, 1[ range.
pub struct TanH;
impl<F: NdFloat> Activation<F> for TanH {
    #[inline]
    fn compute(&self, x: F) -> F {
        x.tanh()
    }
    #[inline]
    fn compute_derivative(&self, x: F) -> F {
        let y = self.compute(x);
        F::one() - y.powi(2)
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
    #[inline]
    fn compute_derivative(&self, x: F) -> F {
        if x < F::zero() {
            F::zero()
        } else {
            F::one()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Float;
    use super::*;

    fn test_numerical_function<A>(
        function: A,
        inputs: Vec<f64>,
        values: Vec<f64>,
        derivatives: Vec<f64>,
    ) where
        A: Activation<Float>,
    {
        assert_eq!(inputs.len(), values.len());
        assert_eq!(inputs.len(), derivatives.len());
        for i in 0..inputs.len() {
            let input = inputs[i];
            let value = function.compute(input);
            let derivative = function.compute_derivative(input);
            assert_relative_eq!(value, values[i]); // 16 digits precision by default
            assert_relative_eq!(derivative, derivatives[i]);
        }
    }

    #[test]
    fn identity() {
        let inputs = vec![-23.0, -7.0, 0.0, 3.0, 10.0];
        test_numerical_function(Identity, inputs.clone(), inputs, vec![1.0; 5]);
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
            vec![
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
            vec![-1.0, -0.5, 0.0, 0.5, 1.0],
            vec![
                -0.7615941559557649,
                -0.4621171572600097,
                0.0,
                0.4621171572600097,
                0.7615941559557649,
            ],
            vec![
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
            vec![-150.0, -7.0, 0.0, 3.0, 10.0],
            vec![0.0, 0.0, 0.0, 3.0, 10.0],
            vec![0.0, 0.0, 1.0, 1.0, 1.0],
        );
    }
}
