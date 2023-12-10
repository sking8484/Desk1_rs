use num::Integer;
use num::ToPrimitive;
use polars::frame::*;
use polars::prelude::*;

extern crate polars;
use crate::predictor::analysis::analysis::{NDArrayHelper, NDArrayMath};
use crate::predictor::api::{PredictionError, Predictor};
use ndarray::*;

pub struct MssaPredictor {
    pub num_days_per_col: i64,
    pub total_trailing_days: i64,
    pub num_assets: i64,
    pub hsvt_threshold: f64,
}

pub struct SeperatedTrainingData {
    pub design_matrix: Array2<f64>,
    pub observation_array: Array1<f64>,
}

#[derive(Debug)]
pub enum MssaError {
    IncorrectParameterInput(String),
}

impl Predictor for MssaPredictor {
    fn new(
        num_days_per_col: i64,
        total_trailing_days: i64,
        num_assets: i64,
        hsvt_threshold: f64,
    ) -> Result<MssaPredictor, MssaError> {
        if total_trailing_days.mod_floor(&num_days_per_col) != 0 {
            return Err(MssaError::IncorrectParameterInput(
                "Paramaters Do Not Fit".to_string(),
            ));
        }
        Ok(MssaPredictor {
            num_days_per_col,
            total_trailing_days,
            num_assets,
            hsvt_threshold,
        })
    }
    fn retrieve_formatted_data(&self) -> Result<DataFrame, PolarsError> {
        let df = DataFrame::new(vec![Series::new("col1", &[4.0, 3.0, 0.0, -5.0])])?;
        Ok(df)
    }
    fn create_predictions(&self, prediction_data: DataFrame) -> DataFrame {
        let column_names = self.create_return_columns(&prediction_data);
        let model = self.train_data(prediction_data.clone());
        if let Ok(linear_model) = model {
            self.predict(linear_model, column_names, prediction_data)
        } else {
            DataFrame::empty()
        }
    }

    fn build_prediction_data(&self, cleaned_data: DataFrame) -> Result<DataFrame, PolarsError> {
        let page_matrix = self.create_page_matrix(cleaned_data)?;
        let hsvt_matrix = self.create_hsvt_matrix(page_matrix)?;
        Ok(hsvt_matrix)
    }
}

impl MssaPredictor {
    fn predict(
        &self,
        linear_model: Array1<f64>,
        column_names: Vec<String>,
        prediction_data: DataFrame,
    ) -> DataFrame {
        let mut predictions = DataFrame::empty();
        let (nrows, _ncols) = prediction_data.shape();
        let shortened_df = prediction_data.tail(Some(nrows - 1));
        let max_col_num = self.calculate_num_cols_per_block();
        for col in column_names {
            let current_asset_data = shortened_df
                .select([format!("{}_{}", col, max_col_num)])
                .expect("STUFF");
            let prediction = current_asset_data
                .clone()
                .to_ndarray::<Float64Type>(IndexOrder::Fortran)
                .expect("Stuff")
                .t()
                .dot(&linear_model)
                .map(|&a| (a * 1000.0).round() / 1000.0)
                .to_vec();
            let last_price = current_asset_data
                .to_ndarray::<Float64Type>(IndexOrder::Fortran)
                .expect("Stuff")
                .last_mut()
                .expect("Should not be empry")
                .to_owned();
            let mut prediction_vec: Vec<f64> = Vec::new();
            for i in prediction {
                prediction_vec.push((i / last_price) - 1.0);
            }
            predictions = predictions
                .hstack(&[Series::new(&col, prediction_vec)])
                .expect("GOOD");
        }
        predictions
    }

    fn create_return_columns(&self, prediction_data: &DataFrame) -> Vec<String> {
        let max_col_num = self.calculate_num_cols_per_block();
        let mut return_names: Vec<String> = Vec::new();
        for col_name in self.convert_col_names_to_string(prediction_data.get_column_names()) {
            if col_name.ends_with(&max_col_num.to_string()) {
                return_names.push(col_name.clone().split("_").collect::<Vec<_>>()[0].to_string());
            }
        }
        return_names
    }
    fn train_data(&self, prediction_data: DataFrame) -> Result<Array1<f64>, PredictionError> {
        let training_data = self.build_training_data(prediction_data);
        if let Ok(sep_train_data) = training_data {
            let model = NDArrayMath {}.build_linear_model(
                sep_train_data.design_matrix.t().to_owned(),
                sep_train_data.observation_array,
            );
            if let Ok(trained_model) = model {
                Ok(trained_model)
            } else {
                Err(PredictionError::TrainModelError)
            }
        } else {
            Err(PredictionError::BuildDataError)
        }
    }

    fn build_training_data(
        &self,
        prediction_data: DataFrame,
    ) -> Result<SeperatedTrainingData, PolarsError> {
        let (nrows, ncols) = prediction_data.shape();
        let design_df = prediction_data.head(Some(nrows - 1));
        let observations_df = prediction_data.tail(Some(1));

        let design_array = design_df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
        let obs_array = observations_df
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)?
            .into_shape(ncols);
        if let Ok(vec) = obs_array {
            Ok(SeperatedTrainingData {
                design_matrix: design_array,
                observation_array: vec,
            })
        } else {
            Err(PolarsError::ShapeMismatch("Bum".into()))
        }
    }

    fn create_hsvt_matrix(&self, page_matrix: DataFrame) -> Result<DataFrame, PolarsError> {
        let num_cols_per_block = self.calculate_num_cols_per_block();
        let mut return_df = DataFrame::empty();

        for x in self.calculate_cols_to_index_on(&num_cols_per_block) {
            return_df = self.append_partial_hsvt_matrix(
                x,
                page_matrix.clone(),
                return_df,
                num_cols_per_block,
            )?;
        }

        Ok(return_df)
    }

    fn calculate_num_cols_per_block(&self) -> i64 {
        self.total_trailing_days / self.num_days_per_col
    }

    fn calculate_cols_to_index_on(
        &self,
        &num_cols_per_block: &i64,
    ) -> std::iter::StepBy<std::ops::Range<i64>> {
        (0..(num_cols_per_block) * (self.num_assets) - 1)
            .step_by(num_cols_per_block.to_usize().expect(""))
    }

    fn append_partial_hsvt_matrix(
        &self,
        x: i64,
        page_matrix: DataFrame,
        mut return_df: DataFrame,
        num_cols_per_block: i64,
    ) -> Result<DataFrame, PolarsError> {
        let current_asset_page_matrix =
            self.select_page_matrix_for_single_asset(page_matrix, x, num_cols_per_block)?;
        let col_names = current_asset_page_matrix.get_column_names();
        let array_data =
            current_asset_page_matrix.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;

        let svd_data = NDArrayMath {}.svd(array_data);
        if let Ok(data) = svd_data {
            let hsvt_asset = self.rebuild_hsvt_matrix(data, col_names)?;
            for col in hsvt_asset.iter() {
                return_df = return_df.hstack(&[col.clone()])?;
            }
        }
        Ok(return_df)
    }

    fn select_page_matrix_for_single_asset(
        &self,
        page_matrix: DataFrame,
        x: i64,
        num_cols_per_block: i64,
    ) -> Result<DataFrame, PolarsError> {
        page_matrix.select_by_range(
            x.to_usize().expect("")..(x + num_cols_per_block).to_usize().expect(""),
        )
    }

    fn rebuild_hsvt_matrix(
        &self,
        data: super::analysis::analysis::SvdResponse,
        col_names: Vec<&str>,
    ) -> Result<DataFrame, PolarsError> {
        let filtered_svd = NDArrayMath {}.filter_svd_matrices(data, self.hsvt_threshold);
        let rebuilt_matrix = NDArrayMath {}
            .rebuild_matrix_from_svd(filtered_svd)
            .map(|&a| ((a * 1000.0).round() / 1000.0));
        let df = NDArrayHelper {}.convert_ndarray_to_polars(
            rebuilt_matrix,
            self.convert_col_names_to_string(col_names),
        )?;
        Ok(df)
    }
    fn create_page_matrix(&self, cleaned_data: DataFrame) -> Result<DataFrame, PolarsError> {
        let array_data = cleaned_data.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;

        let raw_column_names = cleaned_data.get_column_names();
        let formatted_col_names = self.create_column_names(raw_column_names);
        let page_matrix = self.build_page_matrix_data(array_data)?;
        Ok(NDArrayHelper {}.convert_ndarray_to_polars(page_matrix, formatted_col_names)?)
    }

    fn convert_col_names_to_string(&self, columns: Vec<&str>) -> Vec<String> {
        let mut names_vec: Vec<String> = Vec::new();
        for elem in columns.iter() {
            names_vec.push(format!("{}", elem))
        }
        names_vec
    }

    fn create_column_names(&self, columns: Vec<&str>) -> Vec<String> {
        let mut names_vec: Vec<String> = Vec::new();
        for elem in columns.iter() {
            for i in 0..self.total_trailing_days / self.num_days_per_col {
                names_vec.push(format!("{}_{}", elem, i + 1))
            }
        }
        return names_vec;
    }

    fn build_page_matrix_data<A: std::clone::Clone>(
        &self,
        data_matrix: Array2<A>,
    ) -> Result<Array2<A>, PolarsError> {
        let column_data = data_matrix.axis_iter(Axis(1));
        let mut return_array: Option<Array2<A>> = None;
        let mut next_col_vec: Option<Array1<A>> = None;

        for col in column_data {
            let mut j = 0;
            let inner_column_data = col.iter();
            for data_point in inner_column_data {
                next_col_vec = self.build_column_vector(j, data_point, next_col_vec);

                j += 1;
                if j == self.num_days_per_col {
                    return_array = self.build_return_array(return_array, &next_col_vec);
                    j = 0;
                }
            }
        }

        if let Some(array) = return_array {
            return Ok(array);
        } else {
            return Err(PolarsError::ComputeError("Error".into()));
        }
    }

    fn build_column_vector<A: std::clone::Clone>(
        &self,
        j: i64,
        data_point: &A,
        mut next_col_vec: Option<Array1<A>>,
    ) -> Option<Array1<A>> {
        if j == 0 {
            next_col_vec = Some(array![data_point.to_owned()]);
        } else {
            next_col_vec = Some(concatenate![
                Axis(0),
                next_col_vec
                    .clone()
                    .expect("Next Col Vec must be instantiated")
                    .view(),
                array![data_point.to_owned()]
            ]);
        }

        return next_col_vec;
    }

    fn build_return_array<A: std::clone::Clone>(
        &self,
        mut return_array: Option<Array2<A>>,
        next_col_vec: &Option<Array1<A>>,
    ) -> Option<Array2<A>> {
        if let Some(value) = return_array {
            return_array = concatenate![
                Axis(1),
                value.clone(),
                next_col_vec
                    .clone()
                    .expect("Better be good")
                    .insert_axis(Axis(1))
            ]
            .into();
        } else {
            return_array = Some(
                next_col_vec
                    .clone()
                    .expect("Next Col Vec should be instantiated")
                    .insert_axis(Axis(1)),
            );
        }

        return return_array;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars_io::prelude::*;
    use std::fs::File;

    #[test]
    fn test_invoke_retrieve_data() {
        let predictor = MssaPredictor {
            num_days_per_col: 10,
            total_trailing_days: 10,
            num_assets: 10,
            hsvt_threshold: 0.0,
        };
        predictor.retrieve_formatted_data();
        assert!(true)
    }

    #[test]
    fn test_invoke_build_prediction_data() {
        let predictor = MssaPredictor {
            num_days_per_col: 10,
            total_trailing_days: 10,
            num_assets: 10,
            hsvt_threshold: 0.0,
        };
        predictor.build_prediction_data(
            predictor
                .retrieve_formatted_data()
                .expect("Should have data"),
        );
        assert!(true)
    }

    #[test]
    fn test_assert_build_prediction_data_correct() {
        let predictor = MssaPredictor {
            num_days_per_col: 2,
            total_trailing_days: 4,
            num_assets: 1,
            hsvt_threshold: 0.0,
        };
        let results = predictor.build_prediction_data(
            predictor
                .retrieve_formatted_data()
                .expect("Should have data"),
        );
        let expected_df = DataFrame::new(vec![
            Series::new("col1_1", &[4.0, 3.0]),
            Series::new("col1_2", &[0.0, -5.0]),
        ])
        .expect("Why didn't this work");
        match results {
            Ok(A) => {
                assert_eq!(A, expected_df)
            }
            Err(B) => {
                assert!(false)
            }
        }
    }

    #[test]
    fn test_assert_build_prediction_data_correct_more_data() {
        let predictor = MssaPredictor {
            num_days_per_col: 2,
            total_trailing_days: 4,
            num_assets: 2,
            hsvt_threshold: 1.0,
        };
        let input = DataFrame::new(vec![
            Series::new("col1", &[4.0, 0.0, 3.0, -5.0]),
            Series::new("col2", &[5.0, 6.0, 7.0, 8.0]),
        ])
        .expect("STUFFFF");
        let expected_df = DataFrame::new(vec![
            Series::new("col1_1", &[0.0, 0.0]),
            Series::new("col1_2", &[0.0, 0.0]),
            Series::new("col2_1", &[0.0, 0.0]),
            Series::new("col2_2", &[0.0, 0.0]),
        ])
        .expect("Stuffffffff");

        let results = predictor.build_prediction_data(input);
        if let Ok(data) = results {
            assert_eq!(data, expected_df)
        } else {
            assert!(false)
        }
    }

    #[test]
    fn test_build_model() {
        let input_a = DataFrame::new(vec![
            Series::new("Constant", &[1.0, 1.0, 1.0, 1.0]),
            Series::new("X_Val", &[2.0, 5.0, 7.0, 8.0]),
            Series::new("Y_Val", &[1.0, 2.0, 3.0, 3.0]),
        ])
        .expect("Dataframe shouldn't be null");

        let expected = array![2.0 / 7.0, 5.0 / 14.0];

        let predictor = MssaPredictor {
            num_days_per_col: 2,
            total_trailing_days: 4,
            num_assets: 1,
            hsvt_threshold: 0.0,
        };
        let predictions = predictor.train_data(input_a.transpose(None, None).expect("Good!"));

        assert_eq!(
            predictions
                .expect("Should build model")
                .map(|&a| (a * 1000.0).round() / 1000.0),
            expected.map(|&a| num_traits::Float::round(a * 1000.0) / 1000.0)
        )
    }

    #[test]
    fn create_predictions() {
        let input_a = DataFrame::new(vec![
            Series::new("col1_1", &[1.0, 2.0, 1.0]),
            Series::new("col1_2", &[1.0, 5.0, 2.0]),
            Series::new("col2_1", &[1.0, 7.0, 3.0]),
            Series::new("col2_2", &[1.0, 8.0, 3.0]),
        ])
        .expect("Dataframe shouldn't be null");

        let expected = DataFrame::new(vec![
            Series::new("col1", &[2.143 / 2.0]),
            Series::new("col2", &[3.357 / 3.0]),
        ])
        .expect("Expected shouldn't be null");

        let predictor = MssaPredictor {
            num_days_per_col: 3,
            total_trailing_days: 6,
            num_assets: 2,
            hsvt_threshold: 0.0,
        };
        let predictions = predictor.create_predictions(input_a);

        assert_eq!(predictions, expected);
    }

    #[test]
    fn test_create_column_names() {
        let input = DataFrame::new(vec![
            Series::new("col1_1", &[1.0, 1.0]),
            Series::new("col1_2", &[1.0, 1.0]),
            Series::new("col2_1", &[1.0, 1.0]),
            Series::new("col2_2", &[1.0, 1.0]),
        ])
        .expect("Not Null DF");
        let expected = vec!["col1", "col2"];

        let predictor = MssaPredictor {
            num_days_per_col: 2,
            total_trailing_days: 4,
            num_assets: 2,
            hsvt_threshold: 0.0,
        };
        let output = predictor.create_return_columns(&input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_create_predictor_using_builder() {
        let predictor = MssaPredictor::new(3, 6, 2, 0.0);
    }

    #[test]
    fn test_actual_data_example() {
        let data = CsvReader::from_path("/Users/darstking/Desktop/Data/Desk1_rs/aapl_test.csv")
            .expect("should have data")
            .has_header(true)
            .truncate_ragged_lines(true)
            .finish()
            .expect("data");

        let mssa = MssaPredictor::new(10, 100, 1, 0.8).expect("stuff");
        let prediction_data = mssa.build_prediction_data(data).expect("data");
        let predictions = mssa.create_predictions(prediction_data);
        assert!(true);
    }
}
