use std::{error::Error, fmt::Debug};

use crate::abstract_data::abstract_classes::{AnalysisToolKit, DecompData, SVD};
pub mod abstract_data;
use nalgebra::*;
use ndarray::{s, Array, Array1, Array2, Axis, Dimension, Ix2, RemoveAxis};
use ndarray_linalg::*;
use polars::{
    export::num::{Float, FromPrimitive, Zero},
    prelude::*,
};

pub struct AnalysisMethods {}

impl AnalysisToolKit for AnalysisMethods {
    fn calculate_num_rows<T>(&self, data: &DMatrix<T>) -> usize {
        data.nrows()
    }
    fn diagonalize_array<T>(&self, data: &Array1<T>) -> Array2<T>
    where
        T: Clone + Zero,
    {
        let diagonal_matrix: Array2<T> = Array2::from_diag(&data);
        diagonal_matrix
    }
    fn calculate_std<T, D>(&self, data: &Array<T, D>, axis: usize, ddof: T) -> Array<T, D::Smaller>
    where
        T: Clone + Zero + FromPrimitive + Float,
        D: Dimension + RemoveAxis,
    {
        return data.std_axis(Axis(axis), ddof);
    }
    fn divide_matrices<T, D>(&self, lhs: &Array<T, D>, rhs: &Array<T, D>) -> Array<T, D>
    where
        T: Clone + Zero + FromPrimitive + Float + Debug,
        D: Dimension + RemoveAxis,
    {
        let mut iter1 = lhs.indexed_iter();
        let mut iter2 = rhs.indexed_iter();
        let mut return_matrix: Array<T, D> = Array::zeros(lhs.raw_dim());
        let return_iter = return_matrix.iter_mut();

        for val in return_iter {
            let val1 = iter1.next().unwrap().1;
            let val2 = *iter2.next().unwrap().1;
            *val = *val1 / val2;
        }
        return return_matrix;
    }

    fn calculate_svd<T>(&self, matrix: &Array<T, Ix2>) -> Result<SVD<T>, Box<dyn Error>>
    where
        T: Clone
            + Zero
            + FromPrimitive
            + Float
            + Debug
            + ndarray_linalg::Lapack
            + ndarray_linalg::Scalar<Real = T>
            + nalgebra::ComplexField<RealField = T>,
    {
        let cloned_data = matrix.clone().into_iter();
        let nalgebra_mat =
            nalgebra::DMatrix::from_iterator(matrix.nrows(), matrix.ncols(), cloned_data);
        let data = nalgebra::linalg::SVD::new(nalgebra_mat, true, true);

        let mut svd_output;
        match data.u {
            None => panic!("We don't have any SVD Data"),
            Some(..) => {
                svd_output = SVD {
                    u: data.u.unwrap(),
                    s: data.singular_values,
                    vt: data.v_t.unwrap(),
                    decomposition: None,
                };
            }
        }
        let s_iterator = svd_output.s.iter();
        let mut decomposition_vec = vec![];
        let mut i = 0;
        for val in s_iterator {
            let new_matrix = (svd_output.u.column(i) * svd_output.vt.row(i)) * *val;
            println!("New Matrix: {:}", new_matrix);
            decomposition_vec.push(DecompData {
                singular_value: val.clone(),
                decomp_matrix: new_matrix.clone(),
            });
            i += 1;
        }
        svd_output.decomposition = Some(decomposition_vec);

        return Ok(svd_output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, aview1};

    #[test]
    fn calc_num_rows() {
        let methods = AnalysisMethods {};
        let matrix = DMatrix::from_vec(3, 3, vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]);
        assert_eq!(methods.calculate_num_rows(&matrix), 3)
    }

    #[test]
    fn diagonalize_array() {
        let methods = AnalysisMethods {};
        let array = arr1(&[1, 2]);
        let diagonals = methods.diagonalize_array(&array);
        assert_eq!(diagonals, arr2(&[[1, 0], [0, 2]]))
    }

    #[test]
    fn calculate_std() {
        let methods = AnalysisMethods {};
        let a = arr2(&[[1., 2.], [3., 4.], [5., 6.]]);
        let stddev = methods.calculate_std(&a, 0, 1.);
        assert_eq!(stddev, aview1(&[2., 2.]));
    }

    #[test]
    fn divide_matrices() {
        let methods = AnalysisMethods {};
        let a = arr2(&[[2., 2., 2.], [4., 4., 4.], [6., 6., 6.]]);
        let b = arr2(&[[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]]);
        let divided = methods.divide_matrices(&a, &b);
        let expected = arr2(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        assert_eq!(divided, expected)
    }
    #[test]
    fn calculate_svd() {
        let methods = AnalysisMethods {};
        let a = arr2(&[[2., 2., 2.], [4., 4., 4.], [6., 6., 6.]]);
        let result = methods.calculate_svd(&a);
        match result {
            Ok(..) => assert!(true),
            Err(..) => assert!(false),
        }
    }
}
