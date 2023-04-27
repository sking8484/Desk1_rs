use crate::abstract_data::abstract_classes::{
    AnalysisToolKit, DataSettings, DecompData, MSSA, SVD,
};
use smartstring::alias::String;
use std::{error::Error, fmt::Debug};
pub mod abstract_data;
use nalgebra::*;

use num::Float;
use num_traits::identities::Zero;
use polars::prelude::*;

pub struct AnalysisMethods {}

impl AnalysisToolKit for AnalysisMethods {
    fn calculate_num_rows<T>(&self, data: &DMatrix<T>) -> usize {
        data.nrows()
    }
    fn diagonalize_array<T>(&self, data: &DVector<T>) -> DMatrix<T>
    where
        T: nalgebra::Scalar + Zero,
    {
        let diagonal_matrix = DMatrix::from_diagonal(&data);
        diagonal_matrix
    }
    fn calculate_row_std<T>(&self, data: &DMatrix<T>) -> RowOVector<T, Dyn>
    where
        T: RealField + Float,
    {
        data.row_variance().map(|x| Float::sqrt(x))
    }
    fn divide_matrices<T>(&self, data1: &DMatrix<T>, data2: &DMatrix<T>) -> DMatrix<T>
    where
        T: Float + RealField,
    {
        data1.zip_map(data2, |i, j| i / j)
    }

    fn calculate_svd<T>(&self, matrix: &DMatrix<T>) -> Result<SVD<T>, Box<dyn Error>>
    where
        T: ComplexField<RealField = T> + Copy,
    {
        let cloned_data = matrix.clone();
        let data = nalgebra::linalg::SVD::new(cloned_data, true, true);

        let mut svd_output;
        match data.u {
            None => panic!("We don't have any SVD Data"),
            Some(..) => {
                svd_output = SVD {
                    u: data.u.unwrap(),
                    s: data.singular_values,
                    vt: data.v_t.unwrap(),
                    decomposition: None,
                };
            }
        }
        let s_iterator = svd_output.s.iter();
        let mut decomposition_vec = vec![];
        let mut i = 0;
        for val in s_iterator {
            let new_matrix = (svd_output.u.column(i) * svd_output.vt.row(i)) * *val;
            decomposition_vec.push(DecompData {
                singular_value: val.clone(),
                decomp_matrix: new_matrix.clone(),
            });
            i += 1;
        }
        svd_output.decomposition = Some(decomposition_vec);

        return Ok(svd_output);
    }
    fn filter_svd_matrices<T>(
        &self,
        matrices: &Vec<DMatrix<T>>,
        singular_values: &DVector<T::RealField>,
        information_threshold: f64,
    ) -> Option<DMatrix<T>>
    where
        T: Float + RealField + Scalar,
    {
        let sum_squared = singular_values.map(|i| i * i).sum();
        let scaled_values = singular_values.map(|i| (i * i) / sum_squared);

        let mut filtered_matrix: Option<DMatrix<T>> = None;

        let mut information = 0.0;
        let mut index = 0;
        for matrix in matrices.iter() {
            if information > information_threshold {
                break;
            }
            match index {
                0 => {
                    filtered_matrix = Some(matrix.clone());
                }
                _ => {
                    filtered_matrix = Some(filtered_matrix.unwrap() + matrix);
                }
            };
            information += num_traits::cast::ToPrimitive::to_f64(&scaled_values[index]).unwrap();
            index += 1;
        }
        return filtered_matrix;
    }
    fn fit_regression<T>(
        &self,
        independent_variables: &DMatrix<T>,
        dependent_variables: &DVector<T>,
        eps: T,
    ) -> DVector<f64>
    where
        T: Float + RealField,
    {
        let A = independent_variables.clone();
        let b = dependent_variables.clone();
        let x = (A.transpose() * A.clone()).pseudo_inverse(eps).unwrap() * (A.transpose() * b);
        return x
            .map(|i| Float::round(num_traits::ToPrimitive::to_f64(&i).unwrap() * 100.0) / 100.0);
    }
    fn create_predictions<T>(
        &self,
        coefficients: &DVector<T>,
        independent_variables: &DMatrix<T>,
    ) -> DVector<T>
    where
        T: Float + RealField,
    {
        let output = coefficients.transpose() * (independent_variables);
        return output.transpose();
    }
    fn clean_data(&self, df: DataFrame, settings: DataSettings) -> DataFrame {
        let mut return_df = df;
        if settings.look_back != 0 {
            return_df = return_df.tail(Some(
                num_traits::ToPrimitive::to_usize(&settings.look_back).unwrap(),
            ));
        }

        if settings.remove_date_col {
            let date_col = return_df.drop_in_place("date").unwrap();
        }
        return return_df;
    }
}

// Spectrum Analysis Implementation

pub struct SpectrumAnalysis {
    pub data: DataFrame,
    pub columns: DVector<String>,
    pub l: i64,
    pub look_back: i64,
    pub information_threshold: f64,
}

impl SpectrumAnalysis {
    fn new(data: DataFrame, l: i64, mut look_back: i64, information_threshold: f64) -> Self {
        if look_back == 0 {
            look_back = num_traits::ToPrimitive::to_i64(&data.shape().0).unwrap();
        }
        if look_back % l != 0 {
            panic!("Lookback and l are not comapitable")
        }
        Self {
            l,
            look_back,
            information_threshold,
            columns: DVector::from_vec(data.get_column_names_owned()),
            data,
        }
    }
}

impl MSSA for SpectrumAnalysis {
    fn create_prediction_features<T>(&self, data: &DMatrix<T>)
    where
        T: Float + RealField,
    {
        let columns_per_series = self.look_back / self.l;
        let temp_cols = self.columns.clone();
        let mut iter = temp_cols.iter().zip(
            ((columns_per_series - 1)..num_traits::ToPrimitive::to_i64(&data.shape().1).unwrap())
                .step_by(columns_per_series.try_into().unwrap())
        );
        while let Some(data) = iter.next() {
            println!("{}, {}", data.0, data.1)
        }
        
    }
}

#[cfg(test)]
mod test_analysis {
    use super::*;

    #[test]
    fn calc_num_rows() {
        let methods = AnalysisMethods {};
        let matrix = DMatrix::from_vec(3, 3, vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]);
        assert_eq!(methods.calculate_num_rows(&matrix), 3)
    }

    #[test]
    fn diagonalize_array() {
        let methods = AnalysisMethods {};
        let array = DVector::from_fn(3, |i, _| i + 1);
        let diagonals = methods.diagonalize_array(&array);
        assert_eq!(
            diagonals,
            DMatrix::from_vec(3, 3, vec![1, 0, 0, 0, 2, 0, 0, 0, 3])
        );
    }

    #[test]
    fn calculate_std() {
        let methods = AnalysisMethods {};
        let matrix = DMatrix::from_vec(3, 3, vec![1., 1., 1., 1., 1., 1., 1., 1., 1.]);
        let stddev = methods.calculate_row_std(&matrix);
        assert_eq!(stddev, DVector::from_fn(3, |_i, _| 0.0).transpose());
    }

    #[test]
    fn divide_matrices() {
        let methods = AnalysisMethods {};
        let a = DMatrix::from_vec(3, 3, vec![6., 6., 6., 4., 4., 4., 2., 2., 2.]);
        let b = DMatrix::from_vec(3, 3, vec![2., 3., 1., 1., 1., 1., 1., 1., 1.]);
        let divided = methods.divide_matrices(&a, &b);
        let expected = DMatrix::from_vec(3, 3, vec![3., 2., 6., 4., 4., 4., 2., 2., 2.]);
        assert_eq!(divided, expected)
    }
    #[test]
    fn calculate_svd() {
        let methods = AnalysisMethods {};
        let mat = DMatrix::from_vec(3, 3, vec![2., 2., 2., 4., 4., 4., 6., 6., 6.]);
        let result = methods.calculate_svd(&mat);
        match result {
            Ok(..) => assert!(true),
            Err(..) => assert!(false),
        }
    }
    #[test]
    fn filter_matrices() {
        let methods = AnalysisMethods {};
        let matrices = vec![
            DMatrix::from_vec(3, 3, vec![1., 1., 1., 1., 1., 1., 1., 1., 1.]),
            DMatrix::from_vec(3, 3, vec![1., 1., 1., 1., 1., 1., 1., 1., 1.]),
        ];

        let singular_values = DVector::from_vec(vec![0.1, 0., 0.]);
        let threshold = 0.5;

        let result = methods.filter_svd_matrices(&matrices, &singular_values, threshold);
        assert_eq!(
            result.unwrap(),
            DMatrix::from_vec(3, 3, vec![1., 1., 1., 1., 1., 1., 1., 1., 1.])
        );
        let matrices = vec![
            DMatrix::from_vec(3, 3, vec![1., 1., 1., 1., 1., 1., 1., 1., 1.]),
            DMatrix::from_vec(3, 3, vec![1., 1., 1., 1., 1., 1., 1., 1., 1.]),
        ];

        let singular_values = DVector::from_vec(vec![0.1, 0.2, 0.]);
        let threshold = 0.8;

        let result = methods.filter_svd_matrices(&matrices, &singular_values, threshold);
        assert_eq!(
            result.unwrap(),
            DMatrix::from_vec(3, 3, vec![2., 2., 2., 2., 2., 2., 2., 2., 2.])
        );
    }
    #[test]
    fn test_linregress() {
        let methods = AnalysisMethods {};
        let indep = DMatrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]).transpose();
        let dep = DVector::from_vec(vec![5., 11., 17.]);
        let regression_out = methods.fit_regression(&indep, &dep, 0.001);
        assert_eq!(regression_out, DVector::from_vec(vec![1., 2.]));
    }
    #[test]
    fn test_clean_data() {
        let settings = DataSettings {
            look_back: 1,
            remove_null_cols: true,
            remove_date_col: true,
        };
        let methods = AnalysisMethods {};
        let data = df!("date" => &["12/21/2021", "12/21/2022"],
        "price" => &["12.21", "12.22"])
        .unwrap();
        let result = methods.clean_data(data, settings);
        assert_eq!(result, df!("price" => &["12.22"]).unwrap())
    }
}

#[cfg(test)]
mod test_spectrum {
    use super::*;

    #[test]
    fn test_spectrum_initialization() {
        let data = df!("date" => &["12/21/2021", "12/21/2022"],
        "price" => &["12.21", "12.22"])
        .unwrap();
        let spectrum = SpectrumAnalysis::new(data.clone(), 100, 1000, 0.95);
        assert_eq!(spectrum.l, 100);
        let spectrum = SpectrumAnalysis::new(data.clone(), 2, 0, 0.95);
        assert_eq!(
            spectrum.look_back,
            num_traits::ToPrimitive::to_i64(&data.shape().0).unwrap()
        );
    }

    #[test]
    fn test_create_prediction_features() {

        let data = df!("date" => &["12/21/2021", "12/21/2022"],
        "price" => &["12.21", "12.22"])
        .unwrap();
        let data_mat = DMatrix::from_vec(2, 2, vec![2022.0, 2022.1, 12.21, 12.22]);
        println!("{}", data_mat);
        let spectrum = SpectrumAnalysis::new(data.clone(), 2, 0, 0.95);
        spectrum.create_prediction_features(&data_mat)
    }
}
