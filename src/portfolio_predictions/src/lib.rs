pub mod analysis;

struct PortfolioPredictor<T: predictor, K:datalink> {
    prediction_methods: T,
    data_link: K,
}

impl<T, K> PortfolioPredictor<T, K> {
    fn new(methods: T, link: K) -> Self {
        return PortfolioPredictor{prediction_methods:methods, data_link:link};
    } 
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance() {
        let portfolio_predictor = PortfolioPredictor::new(1, 0);
        assert!(true)

    }

    #[test]
    fn test_trait_impls() {
        let portfolio_predictor = PortfolioPredictor::new(1, 0);
        assert!(portfolio_predictor.prediction_methods.test())
        assert!(portfolio_predictor.data_link.test())
    }
}
