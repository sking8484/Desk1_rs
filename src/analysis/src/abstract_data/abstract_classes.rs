use std::{error::Error, fmt::Debug};

use ndarray::{
    Array, Array1, Array2, Dimension, Ix1, Ix2,
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

    fn calculate_svd<T>(&self, matrix: &Array<T, Ix2>) -> Result<SVD<T>, Box<dyn Error>>
    where
        T: FromPrimitive
            + Scalar
            + Lapack
            + Clone
            + Zero
            + Float
            + Debug
            + ndarray_linalg::Scalar<Real = T>
            + nalgebra::ComplexField;
    //fn filter_svd_matrices<T>(&self, elementaryMatrices: Vec<&Array<T, Ix2>>, singularValues: Vec<&Array<T, Ix2>>, informationThreshold: f32) -> todo!();
}

#[derive(Debug)]
pub struct DecompData<T> {
    pub singular_value: T,
    pub decomp_matrix: Array<T, Ix2>
}

#[derive(Debug)]
pub struct SVD<T> {
    pub u: Array<T, Ix2>,
    pub s: Array<T, Ix1>,
    pub vt: Array<T, Ix2>,
    pub decomposition: Option<Vec<DecompData<T>>>
}
