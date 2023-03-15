use polars::{prelude::*, export::num::Zero};
use ndarray::{Array1, Array2};

pub trait AnalysisToolKit {
    fn calculate_num_rows(&self, data: &DataFrame) -> usize;
    fn diagonalize_array<T>(&self, data: &Array1<T>) -> Array2<T>
    where
        T: Clone + Zero;
}
