use std::{fmt::Debug, ops::Div};

use crate::abstract_data::abstract_classes::AnalysisToolKit;
pub mod abstract_data;
use ndarray::{Array, Array1, Array2, Axis, DataMut, DataOwned, DimMax, Dimension, RemoveAxis};
use polars::{
    export::num::{Float, FromPrimitive, Zero},
    prelude::*,
};

pub struct AnalysisMethods {}

impl AnalysisToolKit for AnalysisMethods {
    fn calculate_num_rows(&self, data: &DataFrame) -> usize {
        data.height()
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, aview1};

    #[test]
    fn calc_num_rows() {
        let methods = AnalysisMethods {};
        let frame: DataFrame = df!("data" => &["1", "2", "3"]).expect("We should see a df");
        assert_eq!(methods.calculate_num_rows(&frame), 3)
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
}
