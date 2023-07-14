"""
    GAMModel(formula, data, λ_opt, Xp_opt, yp_opt, Spline_opt)
Holds information relevant to the GAM model.

Usage:
```julia-repl
GAMModel(formula, data, λ_opt, Xp_opt, yp_opt, Spline_opt)
```
Arguments:
- `formula` : `String` containing the expression of the model.
- `data` : `DataFrame` containing the covariates and response variable to use.
- `λ_opt` : `Float64` containing the optimised λ value.
- `Xp_opt` : `AbstractMatrix` containing the penalty design matrix.
- `yp_opt` : `AbstractVector` containing the penalty response variable.
- `Spline_opt` : `Spline` containing the optimised spline object.
"""
struct GAMModel
    formula::String
    data::DataFrame
    λ_opt::Float64
    Xp_opt::AbstractMatrix
    yp_opt::AbstractVector
    Spline_opt::Spline
end