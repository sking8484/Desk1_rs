extern crate polars;

use polars::prelude::*;

use super::mssa::{MssaError, MssaPredictor};

#[derive(Debug)]
pub enum PredictionError {
    TrainModelError,
    BuildDataError

}

pub trait Predictor {
    fn new(num_days_per_col: i64, total_trailing_days: i64, num_assets: i64, hsvt_threshold: f64) -> Result<MssaPredictor, MssaError>;
    fn retrieve_formatted_data(&self) -> Result<DataFrame, PolarsError>;
    fn build_prediction_data(&self, cleaned_data: DataFrame) -> Result<DataFrame, PolarsError>;
    fn create_predictions(&self, prediction_data: DataFrame) -> DataFrame;
}
