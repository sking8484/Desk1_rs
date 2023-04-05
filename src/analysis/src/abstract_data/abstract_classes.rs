use std::{error::Error, fmt::Debug};

use nalgebra::{ComplexField, DMatrix, DVector, RealField, RowOVector, Dyn};
use num::Float;
use num_traits::identities::Zero;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::*;

pub trait AnalysisToolKit {
    fn calculate_num_rows<T>(&self, data: &DMatrix<T>) -> usize;
    fn diagonalize_array<T>(&self, data: &DVector<T>) -> DMatrix<T>
    where
        T: Zero + nalgebra::Scalar;
    fn calculate_row_std<T>(&self, data: &DMatrix<T>) -> RowOVector<T, Dyn>
    where
        T: RealField + Float;
    fn divide_matrices<T>(&self, data1: &DMatrix<T>, data2: &DMatrix<T>) -> DMatrix<T>
    where
        T: Float + RealField;

    fn calculate_svd<T>(&self, matrix: &DMatrix<T>) -> Result<SVD<T>, Box<dyn Error>>
    where
        T: nalgebra::ComplexField<RealField = T> + Copy;
    fn filter_svd_matrices<T>(&self, matrices: &Vec<DMatrix<T>>, values: &DVector<T::RealField>, informationThreshold: f64) -> Option<DMatrix<T>>
    where
        T: Float + RealField;
    fn run_regression<T>(&self, independent_variables: &DMatrix<T>, dependent_variables: &DVector<T>, eps:T)
        where T: Float + RealField;
}

#[derive(Debug)]
pub struct DecompData<T> {
    pub singular_value: T,
    pub decomp_matrix: DMatrix<T>,
}

#[derive(Debug)]
pub struct SVD<T>
where
    T: ComplexField,
{
    pub u: DMatrix<T>,
    pub s: DVector<T::RealField>,
    pub vt: DMatrix<T>,
    pub decomposition: Option<Vec<DecompData<T>>>,
}
