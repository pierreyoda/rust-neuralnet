use rand::Rng;
use rand::distributions::IndependentSample;
use ndarray::{Array1, Array2, ArrayBase, DataOwned, Dimension, NdFloat, ShapeBuilder};

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

#[cfg(test)]
mod tests {
    use ndarray::{Array2, arr2};
    use super::flatten_array;

    #[test]
    fn flatten_array_function() {
        let array = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let maybe_array_flat = flatten_array(array);
        assert!(maybe_array_flat.is_some());
        let array_flat = maybe_array_flat.unwrap();
        assert_relative_eq!(array_flat.get(0).unwrap(), &1.0);
        assert_relative_eq!(array_flat.get(1).unwrap(), &2.0);
        assert_relative_eq!(array_flat.get(2).unwrap(), &3.0);
        assert_relative_eq!(array_flat.get(3).unwrap(), &4.0);
        assert_relative_eq!(array_flat.get(4).unwrap(), &5.0);
        assert_relative_eq!(array_flat.get(5).unwrap(), &6.0);
    }
}
