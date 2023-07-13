"""
    GAMModel(x, y, λ_opt, Xp_opt, yp_opt, Spline_opt)
Holds information relevant to the GAM model.

Usage:
```julia-repl
GAMModel(x, y, λ_opt, Xp_opt, yp_opt, Spline_opt)
```
Arguments:
- `x` : `AbstractVector` containing the predictor variable.
- `y` : `AbstractVector` containing the response variable.
- `λ_opt` : `Float64` containing the optimised λ value.
- `Xp_opt` : `AbstractMatrix` containing the penalty design matrix.
- `yp_opt` : `AbstractVector` containing the penalty response variable.
- `Spline_opt` : `Spline` containing the optimised spline object.
"""
struct GAMModel
    x::AbstractVector
    y::AbstractVector
    λ_opt::Float64
    Xp_opt::AbstractMatrix
    yp_opt::AbstractVector
    Spline_opt::Spline
end