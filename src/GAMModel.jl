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
- `model` : `` TO ADD INFO HERE!!!
"""
struct GAMModel
    ModelFormula::String
    y_var::Symbol
    data::DataFrame
    model
end