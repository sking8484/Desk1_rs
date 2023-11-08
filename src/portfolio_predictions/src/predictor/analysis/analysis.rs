use std::{error::Error, fmt::Debug};
use nalgebra::*;
use ndarray::*;

use num::Float;
use num_traits::identities::Zero;
use polars::prelude::*;

#[derive(Debug)]
pub enum AnalysisError {
    SVDError,
    ConversionError,
    ShapeError,
    ModelError
}

pub struct AnalysisMethods{}

pub struct NDArrayMath{}
pub struct SvdResponse{
    pub u: Array2<f64>,
    pub sigma: Array2<f64>,
    pub vt: Array2<f64>
}
impl NDArrayMath {
    pub fn svd(&self, array: Array2<f64>) -> Result<SvdResponse, AnalysisError> {
        
        let nalgebra_matrix = NDArrayHelper{}.convert_ndarray_matrix_to_nalgebra(array);
        let svd = nalgebra_matrix.svd(true, true);
        let svd_output = NDArrayHelper{}.build_svd_return_object(svd);

        if let Ok(response) = svd_output {
            return Ok(response)
        }
        return Err(AnalysisError::SVDError)
    }

    pub fn filter_svd_matrices(&self, svd_response: SvdResponse, min_significant_val: f64) -> SvdResponse {
        let sigma_sum = svd_response.sigma.sum();
        let new_sigmas = svd_response.sigma.map(|&a| {
            if a/sigma_sum > min_significant_val {
                return a;
            } else {
                return 0.0;
            }
        });

        return SvdResponse { u: svd_response.u.clone(), sigma: new_sigmas, vt: svd_response.vt.clone() }
    }

    pub fn rebuild_matrix_from_svd(&self, svd_response: SvdResponse) -> Array2<f64> {
        let u_sigma = svd_response.u.dot(&svd_response.sigma);
        let u_sigma_vt = u_sigma.dot(&svd_response.vt);
        return u_sigma_vt
    }

    pub fn build_linear_model(&self, A: Array2<f64>, b: Array1<f64>) -> Result<Array1<f64>, AnalysisError> {

        let na_design_matrix = NDArrayHelper{}.convert_ndarray_matrix_to_nalgebra(A);
        let na_observations = NDArrayHelper{}.convert_ndarray_vector_to_nalgebra(b);

        let a_transpose_a = na_design_matrix.clone().transpose() * na_design_matrix.clone();
        let a_transpose_b = na_design_matrix.clone().transpose() * na_observations;

        print!("{}", a_transpose_a);
        let a_transpose_a_inv = a_transpose_a.try_inverse();
        if let Some(inv) = a_transpose_a_inv {
            let model =  inv*a_transpose_b;
            let ndarray_model = NDArrayHelper{}.convert_nalgebra_to_ndarray(model);
            if let Ok(nd_model) = ndarray_model.clone() {
                let vec = nd_model.clone().into_shape(nd_model.nrows());
                if let Ok(v) = vec {
                    return Ok(v);
                } else {
                    return Err(AnalysisError::ModelError)
                }
            } else {
                return Err(AnalysisError::ModelError)
            }
        } else {
            return Err(AnalysisError::ModelError)
        }
    }
}

pub struct NDArrayHelper {}

impl NDArrayHelper {

    pub fn build_svd_return_object(&self, svd_response: nalgebra::SVD<f64, nalgebra::Dyn, nalgebra::Dyn>) -> Result<SvdResponse, ShapeError>{
        let u: Array2<f64> = self.convert_nalgebra_to_ndarray(svd_response.u.expect("U Matrix should not be empty"))?.to_owned();
        let sigma: Array2<f64> = Array2::from_diag(&self.convert_nalgebra_vec_to_ndarray(svd_response.singular_values)?);
        let vt: Array2<f64> = self.convert_nalgebra_to_ndarray(svd_response.v_t.expect("V_T matrix should not be empty"))?.t().to_owned();


        return Ok(SvdResponse{
            u,
            sigma,
            vt
        })
    }

    pub fn convert_ndarray_matrix_to_nalgebra(&self, array: Array2<f64>) -> DMatrix<f64> {
        let (nrows, ncols) = array.dim();
        let nalgebra_matrix = DMatrix::from_row_iterator(nrows, ncols, array.into_iter());
        return nalgebra_matrix;
    }

    pub fn convert_ndarray_vector_to_nalgebra(&self, array: Array1<f64>) -> DMatrix<f64> {
        let nrows = array.dim();
        let nalgebra_matrix = DMatrix::from_row_iterator(nrows, 1, array.into_iter());
        return nalgebra_matrix;
        
    }

    pub fn convert_ndarray_to_polars(&self, array: Array2<f64>, column_names: Vec<String>) -> Result<DataFrame, PolarsError> {
        let df: DataFrame = DataFrame::new(
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
     pub fn convert_nalgebra_to_ndarray(&self, matrix: nalgebra::DMatrix<f64>) -> Result<Array2<f64>, ShapeError> {
        let (nrows, ncols) = matrix.shape();
        let ndarray = Array2::from_shape_vec((nrows, ncols), matrix.data.as_vec().clone())?;
        return Ok(ndarray)
    }
    
    pub fn convert_nalgebra_vec_to_ndarray(&self, vector: nalgebra::DVector<f64>) -> Result<Array1<f64>, ShapeError> {
        let ndarray_array = Array1::from(
            vector
                .as_slice()
                .iter()
                .map(|&x| x)
                .collect::<Vec<f64>>(),
        );
        return Ok(ndarray_array)
    }

}

#[cfg(test)]
mod test_ndrray_helper {
    use super::*;

    #[test]
    fn test_polars_from_ndarray() {
        let array_helpers = NDArrayHelper{};
        let column_names: Vec<String> = vec!["One".to_string(), "Two".to_string()];
        let array = array![[1.0, 2.0,], [3.0, 4.0]];
        let df = array_helpers.convert_ndarray_to_polars(array, column_names);
        let expected = DataFrame::new(vec![
            Series::new("One", &[1.0, 3.0]),
            Series::new("Two", &[2.0, 4.0])
        ]).expect("DF Just created should not be empty");
        if let Ok(val) = df {
            assert_eq!(val, expected)
        } else {
            assert!(false);
        }
    }
}

#[cfg(test)]
mod test_ndarray_math {
    use super::*;

    #[test]
    fn test_individual_svd_elements() {
        let array = array![[4.0, 0.0], [3.0, -5.0]];
        let svd_output = NDArrayMath{}.svd(array);
        let U = array![
            [-1.0/5.0_f64.powf(0.5), 2.0/5.0_f64.powf(0.5)],
            [-2.0/5.0_f64.powf(0.5), -1.0/5.0_f64.powf(0.5)]
        ];
        let sigma = array![
            [40.0_f64.powf(0.5), 0.0],
            [0.0, 10.0_f64.powf(0.5)]
        ];
        
        let vt = array![
            [-1.0/2.0_f64.powf(0.5), 1.0/2.0_f64.powf(0.5)],
            [1.0/2.0_f64.powf(0.5), 1.0/2.0_f64.powf(0.5)]
        ];
        if let Ok(val) = svd_output {
            assert_eq!(val.u.map(|&x| ((x*1000.0).round()/1000.0).abs()), U.map(|&x| ((x*1000.0).round()/1000.0).abs()));
            assert_eq!(val.sigma.map(|&x| (x*1000.0).round()/1000.0), sigma.map(|&x| (x*1000.0).round()/1000.0));
            assert_eq!(val.vt.map(|&x| ((x*1000.0).round()/1000.0).abs()), vt.map(|&x| ((x*1000.0).round()/1000.0).abs()));
        }
    }

    #[test]
    fn test_svd_combines_to_original() {
        let array = array![[4.0, 0.0], [3.0, -5.0]];
        let svd_output = NDArrayMath{}.svd(array.clone());
        if let Ok(val) = svd_output {
            let recombined_matrix = NDArrayMath{}.rebuild_matrix_from_svd(val);

            assert_eq!(recombined_matrix.map(|&x| {(x*1000.0).round()/1000.0}), array);
        }

    }

    #[test]
    fn test_filter_all_svd() {
        let array = array![[4.0, 0.0], [3.0, -5.0]];
        let svd_output = NDArrayMath{}.svd(array.clone());
        let expected_sigma = array![
            [0.0, 0.0],
            [0.0, 0.0]
        ];
        let filtered_output = NDArrayMath{}.filter_svd_matrices(svd_output.expect("SVD Output should be ok"), 1.0);
        assert_eq!(filtered_output.sigma, expected_sigma);
    }

    #[test]
    fn test_filter_none_svd() {
        let array = array![[4.0, 0.0], [3.0, -5.0]];
        let svd_output = NDArrayMath{}.svd(array.clone());

        let sigma = array![
            [40.0_f64.powf(0.5), 0.0],
            [0.0, 10.0_f64.powf(0.5)]
        ];
        
        let filtered_output = NDArrayMath{}.filter_svd_matrices(svd_output.expect("SVD Output should be ok"), 0.0);
        assert_eq!(filtered_output.sigma.map(|&a| (a*1000.0).round()/1000.0), sigma.map(|&a| (a*1000.0).round()/1000.0));
    }

    #[test]
    fn test_filter_some_svd() {
        let array = array![[4.0, 0.0], [3.0, -5.0]];
        let svd_output = NDArrayMath{}.svd(array.clone());

        let sigma = array![
            [40.0_f64.powf(0.5), 0.0],
            [0.0, 0.0]
        ];
        
        let filtered_output = NDArrayMath{}.filter_svd_matrices(svd_output.expect("SVD Output should be ok"), 0.5);
        assert_eq!(filtered_output.sigma.map(|&a| (a*1000.0).round()/1000.0), sigma.map(|&a| (a*1000.0).round()/1000.0));
    }
    
    #[test]
    fn test_recombinded_svd() {
        let array = array![[4.0, 0.0], [3.0, -5.0]];
        let svd_output = NDArrayMath{}.svd(array.clone());

        let U = array![
            [-1.0/5.0_f64.powf(0.5), 2.0/5.0_f64.powf(0.5)],
            [-2.0/5.0_f64.powf(0.5), -1.0/5.0_f64.powf(0.5)]
        ];
        let sigma = array![
            [40.0_f64.powf(0.5), 0.0],
            [0.0, 0.0]
        ];
        
        let vt = array![
            [-1.0/2.0_f64.powf(0.5), 1.0/2.0_f64.powf(0.5)],
            [1.0/2.0_f64.powf(0.5), 1.0/2.0_f64.powf(0.5)]
        ];

        let expected_matrix = U.dot(&sigma).dot(&vt);
        
        let filtered_output = NDArrayMath{}.filter_svd_matrices(svd_output.expect("SVD Output should be ok"), 0.5);
        let recombined_maxtrix = NDArrayMath{}.rebuild_matrix_from_svd(filtered_output);
        assert_eq!(recombined_maxtrix.map(|&a| (a*1000.0).round()/1000.0), expected_matrix.map(|&a| (a*1000.0).round()/1000.0));
    }

    #[test]
    fn test_build_linear_model() {
        let input_a = array![
            [1.0, 2.0],
            [1.0, 5.0],
            [1.0, 7.0],
            [1.0, 8.0]

        ];

        let input_b  = array![1.0, 2.0, 3.0, 3.0];
        let expected = array![2.0/7.0, 5.0/14.0];

        let model = NDArrayMath{}.build_linear_model(input_a, input_b);

        assert_eq!(model.expect("Should build model").map(|&a| (a*1000.0).round()/1000.0), expected.map(|&a| num_traits::Float::round(a*1000.0)/1000.0))

    }
}
