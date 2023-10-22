Potential Design Pattern

```
struct PortfolioPricePredictor<T: PricePredictor, S: DataLink,...> {
  pricePredictor: T
  dataLink: S
  ...
}

impl PortfolioPricePredictor {
  fn new() -> Self:
    return self{predictorClass, dataLinkClass}
}
```
