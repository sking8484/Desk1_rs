
use polars::prelude::*;
use polars::frame::*;


extern crate polars;
use crate::predictor::api::{Predictor};
use crate::predictor::analysis::api::Analysis;
use ndarray::*;

use super::analysis::analysis::AnalysisMethods;

pub struct MssaPredictor{
    pub num_days_per_col: i64
}


impl Predictor for MssaPredictor {
    fn test(&self) -> bool {
        return true;
    }
    fn retrieve_formatted_data(&self) -> Result<DataFrame, PolarsError> {
        let df = DataFrame::new(vec![
            Series::new("col1", &[1.0, 2.0, 3.0, 4.0]),
            Series::new("col2", &[5.0, 6.0, 7.0, 8.0]),
            Series::new("col3", &[9.0, 10.0, 11.0, 12.0]),
            ]
        )?;
        return Ok(df)
    }
    fn create_predictions(&self) -> bool {
        todo!()
    }
    fn build_prediction_data(&self) -> Result<DataFrame, PolarsError> {
        let cleaned_data = self.retrieve_formatted_data()?;
        let page_matrix = self.create_page_matrix(cleaned_data);
    }
}

impl MssaPredictor {
    fn create_page_matrix(&self, cleaned_data: DataFrame) -> Result<DataFrame, PolarsError> {
        let array_data = cleaned_data.to_ndarray::<Float64Type>(IndexOrder::Fortran);

        let column_names = cleaned_data.get_columns();
    }

    fn build_page_matrix_data<A: std::clone::Clone>(&self, array: Array2<A>) -> Array2<A> {
        let column_data = array.axis_iter(Axis(0));
        let mut return_array : Array2<A>;
        let next_col_vec: Array1<A>;
        let k = 0;
        for col in column_data {
            let j = 0;
            let inner_column_data = col.iter();
            for data_point in inner_column_data {
                if j == 0 {
                    let mut next_col_vec = array![data_point];
                } else {
                    stack![Axis(0), next_col_vec.view(), array![*data_point]];
                }
                j += 1;
                if j == self.num_days_per_col - 1 {
                    if k == 0 {
                        return_array = stack![Axis(1), next_col_vec, next_col_vec];
                    } else {
                        concatenate![Axis(1), return_array, next_col_vec.insert_axis(Axis(1))];
                    }
                    j = 0;
                }
                k += 1;
            }
        }
        return return_array;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_invoke_retrieve_data() {
        let predictor = MssaPredictor{num_days_per_col:10};
        predictor.retrieve_formatted_data();
        assert!(true)
    }

    #[test]
    fn test_invoke_build_prediction_data(){
        let predictor = MssaPredictor{num_days_per_col:10};
        predictor.build_prediction_data();
        assert!(true)
    }

    #[test]
    fn test_assert_build_prediction_data_correct() {
        let predictor = MssaPredictor{num_days_per_col: 2};
        let results = predictor.build_prediction_data();
        let expected_df = DataFrame::new(vec![
            Series::new("col1_1", &[1.0, 2.0]),
            Series::new("col1_2", &[3.0, 4.0]),
            Series::new("col2_1", &[5.0, 6.0]),
            Series::new("col2_2", &[7.0, 8.0]),
            Series::new("col3_1", &[9.0, 10.0]),
            Series::new("col3_2", &[11.0, 12.0]),
        ]).expect("Why didn't this work");
        match results {
           Ok(A) => {
                assert_eq!(A, expected_df)
            } 
            Err(B) => {
                assert!(false)
            }
        }
    }
}
