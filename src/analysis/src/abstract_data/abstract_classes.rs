use std::{ops::Div, fmt::Debug};

use polars::{prelude::*, export::num::{Zero, FromPrimitive, Float}}; use ndarray::{Array1, Array2, RemoveAxis, Array, OwnedRepr, Dimension, DimMax, DataOwned, DataMut};

pub trait AnalysisToolKit {
    fn calculate_num_rows(&self, data: &DataFrame) -> usize;
    fn diagonalize_array<T>(&self, data: &Array1<T>) -> Array2<T>
        where
            T: Clone + Zero;
    fn calculate_std<T, D>(&self, data: &Array<T, D>, axis: usize, ddof: T) -> Array<T, D::Smaller>
        where 
            T: Clone + Zero + FromPrimitive + Float,
            D: Dimension + RemoveAxis;

    fn divide_matrices<T, D>(&self, lhs: &Array<T, D>, rhs: &Array<T, D>)
        where
            T: Clone + Zero + FromPrimitive + Float + Debug,
            D: Dimension + RemoveAxis;
            
}

