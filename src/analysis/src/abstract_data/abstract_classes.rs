use std::{fmt::Debug, ops::Div};

use ndarray::{
    Array, Array1, Array2, Data, DataMut, DataOwned, DimMax, Dimension, Ix2, OwnedRepr, RawData,
    RemoveAxis,
};
use ndarray_linalg::*;
use polars::{
    export::num::{Float, FromPrimitive, Zero},
    prelude::*,
};

pub trait AnalysisToolKit {
    fn calculate_num_rows(&self, data: &DataFrame) -> usize;
    fn diagonalize_array<T>(&self, data: &Array1<T>) -> Array2<T>
    where
        T: Clone + Zero;
    fn calculate_std<T, D>(&self, data: &Array<T, D>, axis: usize, ddof: T) -> Array<T, D::Smaller>
    where
        T: Clone + Zero + FromPrimitive + Float,
        D: Dimension + RemoveAxis;

    fn divide_matrices<T, D>(&self, lhs: &Array<T, D>, rhs: &Array<T, D>) -> Array<T, D>
    where
        T: Clone + Zero + FromPrimitive + Float + Debug,
        D: Dimension + RemoveAxis;

    fn calculate_svd<T>(&self, matrix: &Array<T, Ix2>)
    where
        T: FromPrimitive + Scalar + Lapack + Clone + Zero + Float + Debug;
}
