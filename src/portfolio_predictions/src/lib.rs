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
        self.prediction_methods.create_predictions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::mssa::MssaPredictor;

    #[test]
    fn test_instance() {
        let _portfolio_predictor = PortfolioPredictor::new(MssaPredictor{num_days_per_col: 2, total_trailing_days: 2});
        assert!(true)
    }

    #[test]
    fn test_predictor_impl() {
        let portfolio_predictor = PortfolioPredictor::new(MssaPredictor{num_days_per_col:2, total_trailing_days: 2});
        assert!(portfolio_predictor.prediction_methods.test());
    }
}
