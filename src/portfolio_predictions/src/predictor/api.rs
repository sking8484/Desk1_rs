extern crate polars;

use polars::prelude::*;

#[derive(Debug)]
pub enum PredictionError {
    TrainModelError,
    BuildDataError

}

pub trait Predictor {
    fn test(&self) -> bool; 
    fn retrieve_formatted_data(&self) -> Result<DataFrame, PolarsError>;
    fn build_prediction_data(&self, cleaned_data: DataFrame) -> Result<DataFrame, PolarsError>;
    fn create_predictions(&self) -> DataFrame;
}
