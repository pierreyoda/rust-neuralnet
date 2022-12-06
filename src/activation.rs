use ndarray::{Array, Dimension, NdFloat};

/// An activation function in a Neural Network defines whether a neuron will
/// send a signal to its outputs or not.
pub trait Activation<F, D>
where
    F: NdFloat,
    D: Dimension,
{
    fn compute(&self, x: &Array<F, D>) -> Array<F, D>;

    fn compute_derivative(&self, x: &Array<F, D>) -> Array<F, D>;
}

/// The Identity function.
pub struct Identity;
impl<F: NdFloat, D: Dimension> Activation<F, D> for Identity {
    #[inline]
    fn compute(&self, x: &Array<F, D>) -> Array<F, D> {
        x.clone()
    }
    #[inline]
    fn compute_derivative(&self, x: &Array<F, D>) -> Array<F, D> {
        let one = F::one();
        x.map(|_| one)
    }
}

/// The Sigmoid function squashes a real value into the ]0, 1[ range.
pub struct Sigmoid;
impl<F: NdFloat, D: Dimension> Activation<F, D> for Sigmoid {
    #[inline]
    fn compute(&self, x: &Array<F, D>) -> Array<F, D> {
        let one = F::one();
        x.map(|v: &F| one / (one + (-*v).exp()))
    }
    #[inline]
    fn compute_derivative(&self, x: &Array<F, D>) -> Array<F, D> {
        let one = F::one();
        x.map(|v: &F| {
            let y = one / (one + (-*v).exp());
            y * (one - y)
        })
    }
}

/// The Hyperbolic tangent squashes a real value into the ]-1, 1[ range.
pub struct TanH;
impl<F: NdFloat, D: Dimension> Activation<F, D> for TanH {
    #[inline]
    fn compute(&self, x: &Array<F, D>) -> Array<F, D> {
        x.map(|v| v.tanh())
    }
    #[inline]
    fn compute_derivative(&self, x: &Array<F, D>) -> Array<F, D> {
        let one = F::one();
        x.map(|v| one - v.tanh().powi(2))
    }
}

/// The Rectified Linear Unit (ReLU) functions replaces negative values with 0.
pub struct Rectifier;
impl<F: NdFloat, D: Dimension> Activation<F, D> for Rectifier {
    #[inline]
    fn compute(&self, x: &Array<F, D>) -> Array<F, D> {
        let zero = F::zero();
        x.map(|v| if *v < zero { zero } else { *v })
    }
    #[inline]
    fn compute_derivative(&self, x: &Array<F, D>) -> Array<F, D> {
        let (zero, one) = (F::zero(), F::one());
        x.map(|v| if *v < zero { zero } else { one })
    }
}

#[cfg(test)]
mod tests {
    use super::super::Float;
    use super::*;
    use ndarray::{Array1, Ix1};

    fn test_numerical_function<A, V>(
        function: A,
        inputs: V,
        values: Vec<Float>,
        derivatives: Vec<Float>,
    ) where
        A: Activation<Float, Ix1>,
        V: Into<Array1<Float>>,
    {
        let inputs_array: Array1<Float> = inputs.into();
        assert_eq!(inputs_array.len(), values.len());
        assert_eq!(inputs_array.len(), derivatives.len());
        let computed_values = function.compute(&inputs_array);
        let computed_derivatives = function.compute_derivative(&inputs_array);
        for i in 0..inputs_array.len() {
            assert_relative_eq!(computed_values[i], values[i]); // 16 digits precision by default
            assert_relative_eq!(computed_derivatives[i], derivatives[i]);
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
