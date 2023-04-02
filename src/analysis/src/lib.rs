use std::{error::Error, fmt::Debug};

use crate::abstract_data::abstract_classes::{AnalysisToolKit, DecompData, SVD};
pub mod abstract_data;
use nalgebra::*;

use num::Float;
use num_traits::identities::Zero;

pub struct AnalysisMethods {}

impl AnalysisToolKit for AnalysisMethods {
    fn calculate_num_rows<T>(&self, data: &DMatrix<T>) -> usize {
        data.nrows()
    }
    fn diagonalize_array<T>(&self, data: &DVector<T>) -> DMatrix<T>
    where T: nalgebra::Scalar + Zero
    {
        let diagonal_matrix = DMatrix::from_diagonal(&data);
        diagonal_matrix
    }
    fn calculate_row_std<T>(&self, data: &DMatrix<T>) -> RowOVector<T, Dyn>
    where
        T: RealField + Float
    {
        data.row_variance().map(|x| Float::sqrt(x))
    }
    fn divide_matrices<T>(&self, data1: &DMatrix<T>, data2: &DMatrix<T>) -> DMatrix<T> 
    where
        T: Float + RealField
    {
        data1.zip_map(data2, |i, j| i/j)
    }

    fn calculate_svd<T>(&self, matrix: &DMatrix<T>) -> Result<SVD<T>, Box<dyn Error>>
    where
        T: ComplexField<RealField = T> + Copy
    {
        let cloned_data = matrix.clone();
        let data = nalgebra::linalg::SVD::new(cloned_data, true, true);

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

    #[test]
    fn calc_num_rows() {
        let methods = AnalysisMethods {};
        let matrix = DMatrix::from_vec(3, 3, vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]);
        assert_eq!(methods.calculate_num_rows(&matrix), 3)
    }

    #[test]
    fn diagonalize_array() {
        let methods = AnalysisMethods {};
        let array = DVector::from_fn(3, |i, _| i+1);
        let diagonals = methods.diagonalize_array(&array);
        assert_eq!(diagonals, DMatrix::from_vec(3, 3, vec![
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        ]));
    }

    #[test]
    fn calculate_std() {
        let methods = AnalysisMethods {};
        let matrix = DMatrix::from_vec(3, 3, vec![
            1., 1., 1., 
            1., 1., 1.,
            1., 1., 1.
        ]);
        let stddev = methods.calculate_row_std(&matrix);
        assert_eq!(stddev, DVector::from_fn(3, |i, _| 0.0).transpose());
    }

    #[test]
    fn divide_matrices() {
        let methods = AnalysisMethods {};
        let a = DMatrix::from_vec(3, 3, vec![
            6., 6., 6.,
            4., 4., 4.,
            2., 2., 2.
        ]);
        let b = DMatrix::from_vec(3, 3, vec![
            2., 3., 1.,
            1., 1., 1.,
            1., 1., 1.
        ]);
        let divided = methods.divide_matrices(&a, &b);
        let expected = DMatrix::from_vec(3, 3, vec![
            3., 2., 6.,
            4., 4., 4.,
            2., 2., 2.
        ]);
        assert_eq!(divided, expected)
    }
    #[test]
    fn calculate_svd() {
        let methods = AnalysisMethods {};
        let mat = DMatrix::from_vec(3, 3, vec![
            2., 2., 2.,
            4., 4., 4.,
            6., 6., 6.
        ]);
        let result = methods.calculate_svd(&mat);
        println!("{:?}", result);
        match result {
            Ok(..) => assert!(true),
            Err(..) => assert!(false),
        }
    }
}
