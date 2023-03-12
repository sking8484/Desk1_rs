use crate::abstract_data::abstract_classes::AnalysisToolKit;
pub mod abstract_data;
use polars::prelude::*;

pub struct AnalysisMethods {}

impl AnalysisToolKit for AnalysisMethods {
    fn calculate_num_rows(&self, data: DataFrame) -> usize {
        data.height()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_num_rows() {
        let methods = AnalysisMethods{};
        let frame: DataFrame = df!("data" => &["1", "2", "3"])
            .expect("We should see a df");
        assert_eq!(methods.calculate_num_rows(frame), 3)
    }
}
