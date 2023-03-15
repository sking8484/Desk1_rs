use crate::abstract_data::abstract_classes::AnalysisToolKit;
pub mod abstract_data;
use polars::{prelude::*, export::num::Zero};
use ndarray::{Array1, Array2};

pub struct AnalysisMethods {}

impl AnalysisToolKit for AnalysisMethods {
    fn calculate_num_rows(&self, data: &DataFrame) -> usize {
        data.height()
    }
    fn diagonalize_array<T>(&self, data: &Array1<T>) -> Array2<T> 
    where 
        T: Clone + Zero 
    {
        let diagonal_matrix: Array2<T> = Array2::from_diag(&data);
        diagonal_matrix
    }
}



#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use super::*;

    #[test]
    fn calc_num_rows() {
        let methods = AnalysisMethods{};
        let frame: DataFrame = df!("data" => &["1", "2", "3"])
            .expect("We should see a df");
        assert_eq!(methods.calculate_num_rows(&frame), 3)
    }

    #[test]
    fn diagonalize_array() {
        let methods = AnalysisMethods{};
        let array = arr1(&[1, 2]);
        let diagonals = methods.diagonalize_array(&array);
        assert_eq!(diagonals, arr2(&[[1, 0], [0, 2]]))
    }
}
