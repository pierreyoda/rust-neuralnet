use ndarray::{ArrayBase, DataOwned, Dimension, NdFloat, ShapeBuilder};
use rand::distributions::IndependentSample;
use rand::Rng;

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
