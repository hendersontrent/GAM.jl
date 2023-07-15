"""
    SmoothData(variable, β_opt, β_opt_confint, λ_opt, Spline_opt, Spline_opt_lower, Spline_opt_upper)
Holds optimised spline and coefficient information for a given smooth.

Usage:
```julia-repl
SmoothData(variable, β_opt, β_opt_confint, λ_opt, Spline_opt, Spline_opt_lower, Spline_opt_upper)
```
Arguments:
- `variable` : `Symbol` denoting the name of the covariate.
- `β_opt` : `Vector{Float64}` denoting the coefficient.
- `β_opt_confint` : `Array{Float64,2}` denoting the confidence interval for the coefficient β.
- `λ_opt` : `Float64` denoting the optimised penalty value.
- `Spline_opt` : `Spline` containing the optimised spline object.
- `Spline_opt_lower` : `Spline` containing the optimised spline object at the lower confidence interval bound.
- `Spline_opt_upper` : `Spline` containing the optimised spline object at the upper confidence interval bound.
"""
struct SmoothData
    variable::Symbol
    β_opt::Vector{Float64}
    β_opt_confint::Matrix{Float64}
    λ_opt::Float64
    Spline_opt::Spline
    Spline_opt_lower::Spline
    Spline_opt_upper::Spline
end

"""
    NoSmoothData(variable, β_opt, β_opt_confint)
Holds optimised coefficient information for a given categorical (or non-smooth) covariate.

Usage:
```julia-repl
NoSmoothData(variable, β_opt, β_opt_confint)
```
Arguments:
- `variable` : `Symbol` denoting the name of the covariate.
- `β_opt` : `β_opt` denoting the coefficient.
- `β_opt_confint` : `Array{Float64,2}` denoting the confidence interval for the coefficient β.
"""
struct NoSmoothData
    variable::Symbol
    β_opt::Float64
    β_opt_confint::Matrix{Float64}
end