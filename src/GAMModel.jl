"""
    GAMModel(ModelFormula, y_var, data, model, covariateFits)
Holds information relevant to the final fitted GAM model.

Usage:
```julia-repl
GAMModel(ModelFormula, y_var, data, model, covariateFits)
```
Arguments:
- `ModelFormula` : `String` containing the expression of the model.
- `y_var` : `Symbol` denoting the response variable column name.
- `data` : `DataFrame` containing the covariates and response variable to use.
- `covariateFits` : `Vector{Union{SmoothData, NoSmoothData}}` containing the coefficient, spline, and penalty information for each covariate.
"""
struct GAMModel
    ModelFormula::String
    y_var::Symbol
    data::DataFrame
    covariateFits::Vector{Union{SmoothData, NoSmoothData}}
end