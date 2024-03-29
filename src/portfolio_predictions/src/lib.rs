pub mod predictor;
use crate::predictor::api::Predictor as Predictor;

struct PortfolioPredictor<T: Predictor> {
    prediction_methods: T,
}

impl<T: Predictor> PortfolioPredictor<T> {
    fn new(methods: T) -> Self {
        return PortfolioPredictor{prediction_methods:methods};
    } 

    fn update_predictions(&self) -> bool {
        return true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::mssa::MssaPredictor;

    #[test]
    fn test_instance() {
        let _portfolio_predictor = PortfolioPredictor::new(MssaPredictor{num_days_per_col: 2, total_trailing_days: 2, num_assets: 10, hsvt_threshold: 0.0});
        assert!(true)
    }
}
