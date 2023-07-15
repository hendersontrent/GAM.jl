"""
    GAMModel(ModelFormula, data, model, covariateFits)
Holds information relevant to the final fitted GAM model.

Usage:
```julia-repl
GAMModel(ModelFormula, data, model, covariateFits)
```
Arguments:
- `ModelFormula` : `String` containing the expression of the model.
- `data` : `DataFrame` containing the covariates and response variable to use.
- `model` : `Struct` containing the final generalised additive model.
- `covariateFits` : `Union{SmoothData, NoSmoothData}` containing the coefficient, spline, and penalty information for each covariate.
"""
struct GAMModel
    ModelFormula::String
    data::DataFrame
    model
    covariateFits::Union{SmoothData, NoSmoothData}
end