use rand::Rng;
use rand::distributions::IndependentSample;
use ndarray::{Array1, Array2, ArrayBase, DataOwned, Dimension, NdFloat, ShapeBuilder, Zip};

/// Extends ndarray's ```ArrayBase``` to provide random arrays of any shape
/// and any data type.
pub trait NdArrayRandomizer<F, S, D>
where
    F: NdFloat,
    S: DataOwned<Elem = F>,
    D: Dimension,
{
    fn random<ArrayShape, Sampler, R>(
        shape: ArrayShape,
        distribution: Sampler,
        rng: &mut R,
    ) -> ArrayBase<S, D>
    where
        ArrayShape: ShapeBuilder<Dim = D>,
        Sampler: IndependentSample<F>,
        R: Rng;
}

impl<F, S, D> NdArrayRandomizer<F, S, D> for ArrayBase<S, D>
where
    F: NdFloat,
    S: DataOwned<Elem = F>,
    D: Dimension,
{
    fn random<ArrayShape, Sampler, R>(
        shape: ArrayShape,
        distribution: Sampler,
        rng: &mut R,
    ) -> ArrayBase<S, D>
    where
        ArrayShape: ShapeBuilder<Dim = D>,
        Sampler: IndependentSample<F>,
        R: Rng,
    {
        Self::from_shape_fn(shape, |_| distribution.ind_sample(rng))
    }
}

/// Returns a flattened version of the given array.
/// Similar to numpy's ```ravel``` function.
pub fn flatten_array<F: NdFloat>(array: Array2<F>) -> Option<Array1<F>> {
    let dim = array.len();
    match array.into_shape(dim) {
        Ok(array_flat) => Some(array_flat),
        Err(shape_error) => None,
    }
}

/// Concatenates two ndarray vectors into a single one.
/// TODO: there is probably a better, more idiomatic way
pub fn concat_vectors<F: NdFloat>(left: &Array1<F>, right: &Array1<F>) -> Array1<F> {
    let left_len = left.len();
    let mut concat: Array1<F> = Array1::zeros(left_len + right.len());
    Zip::indexed(&mut concat).apply(|i, v| {
        *v = if i < left_len {
            *left.get(i).unwrap()
        } else {
            *right.get(i - left_len).unwrap()
        };
    });
    concat
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, arr1, arr2};
    use super::{concat_vectors, flatten_array};
    use super::super::Float;

    fn test_vector(vector: &Array1<Float>, values: Vec<Float>) {
        assert!(vector.len() >= values.len());
        for i in 0..values.len() {
            assert_relative_eq!(vector.get(i).unwrap(), &values[i]);
        }
    }

    #[test]
    fn flatten_array_function() {
        let array = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let maybe_array_flat = flatten_array(array);
        assert!(maybe_array_flat.is_some());
        let array_flat = maybe_array_flat.unwrap();
        test_vector(&array_flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concat_vectors_function() {
        let (vector_left, vector_right) = (arr1(&[1.0, 2.0]), arr1(&[3.0, 4.0]));
        let vector_concat = concat_vectors(&vector_left, &vector_right);
        test_vector(&vector_concat, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
