extern crate polars;

use polars::prelude::*;


pub trait Predictor {
    fn test(&self) -> bool; 
    fn retrieve_formatted_data(&self) -> Result<DataFrame, PolarsError>;
    fn build_prediction_data(&self) -> Result<DataFrame, PolarsError>;
    fn create_predictions(&self) -> bool;
}
