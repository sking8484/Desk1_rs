
use polars::prelude::*;
use polars::frame::*;


extern crate polars;
use crate::predictor::api::{Predictor};
use crate::predictor::analysis::api::Analysis;
use ndarray::*;

use super::analysis::analysis::AnalysisMethods;

pub struct MssaPredictor{
    pub num_days_per_col: i64,
    pub total_trailing_days: i64
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
        let page_matrix = self.create_page_matrix(cleaned_data)?;
        return Ok(page_matrix);
    }
}

impl MssaPredictor {
    fn create_page_matrix(&self, cleaned_data: DataFrame) -> Result<DataFrame, PolarsError> {
        let array_data = cleaned_data.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;

        let raw_column_names = cleaned_data.get_column_names();
        let formatted_col_names = self.create_column_names(raw_column_names);
        let page_matrix = self.build_page_matrix_data(array_data)?;
        return Ok(self.convert_ndarray_to_polars(page_matrix, formatted_col_names)?);
    }

    fn create_column_names(&self, columns: Vec<&str>) -> Vec<String> {
        let mut names_vec: Vec<String> = Vec::new();
        for elem in columns.iter() {
            for i in 0..self.total_trailing_days/self.num_days_per_col {
                names_vec.push(format!("{}_{}", elem, i+1))
            }
        }
        return names_vec
    }

    fn build_page_matrix_data<A: std::clone::Clone>(&self, data_matrix: Array2<A>) -> Result<Array2<A>, PolarsError> {
        let mut column_data = data_matrix.axis_iter(Axis(1));
        let mut return_array: Option<Array2<A>> = None;
        let mut next_col_vec: Option<Array1<A>> = None;
        for col in column_data {
            let mut j = 0;
            let inner_column_data = col.iter();
            for data_point in inner_column_data {
                if j == 0 {
                    next_col_vec = Some(array![data_point.to_owned()]);
                } else {
                    next_col_vec = Some(concatenate![Axis(0), next_col_vec.clone().expect("Next Col Vec must be instantiated").view(), array![data_point.to_owned()]]);
                }
                j += 1;
                if j == self.num_days_per_col {
                    match &return_array {
                        Some(value) => {
                            return_array = concatenate![Axis(1), value.clone(), next_col_vec.clone().expect("Better be good").insert_axis(Axis(1))].into();
                        },
                        None => {
                            return_array = Some(next_col_vec.clone().expect("Next Col Vec should be instantiated").insert_axis(Axis(1)));
                        }
                    }
                    
                    j = 0;
                }
            }
        }
        match return_array {
            Some(value) => return Ok(value),
            None => return Err(PolarsError::ComputeError("Error".into())),
        }
    }

    fn convert_ndarray_to_polars(&self, array: Array2<f64>, column_names: Vec<String>) -> Result<DataFrame, PolarsError> {
        let mut df: DataFrame = DataFrame::new(
        array.axis_iter(ndarray::Axis(1))
            .into_iter()
            .enumerate()
            .map(|(i, col)| {
                Series::new(
                    &column_names[i],
                    col.to_vec()
                )
            })
            .collect::<Vec<Series>>()
        ).unwrap();
        return Ok(df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_invoke_retrieve_data() {
        let predictor = MssaPredictor{num_days_per_col:10, total_trailing_days: 10};
        predictor.retrieve_formatted_data();
        assert!(true)
    }

    #[test]
    fn test_invoke_build_prediction_data(){
        let predictor = MssaPredictor{num_days_per_col:10, total_trailing_days: 10};
        predictor.build_prediction_data();
        assert!(true)
    }

    #[test]
    fn test_assert_build_prediction_data_correct() {
        let predictor = MssaPredictor{num_days_per_col: 2, total_trailing_days: 4};
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
