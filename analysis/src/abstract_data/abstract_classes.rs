use polars::prelude::*;

pub trait AnalysisToolKit {
    fn calculate_num_rows(&self, data: DataFrame) -> usize;
}
