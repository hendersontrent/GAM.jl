"""
    SmoothData(variable, β_opt, λ_opt, Spline_opt)
Holds optimised spline and coefficient information for a given smooth.

Usage:
```julia-repl
SmoothData(variable, β_opt, λ_opt, Spline_opt)
```
Arguments:
- `variable` : `Symbol` denoting the name of the covariate.
- `β_opt` : `Vector{Float64}` denoting the coefficient.
- `λ_opt` : `Float64` denoting the optimised penalty value.
- `Spline_opt` : `Spline` containing the optimised spline object.
"""
struct SmoothData
    variable::Symbol
    β_opt::Vector{Float64}
    λ_opt::Float64
    Spline_opt::Spline
end

"""
    NoSmoothData(variable, β_opt)
Holds optimised coefficient information for a given categorical (or non-smooth) covariate.

Usage:
```julia-repl
NoSmoothData(variable, β_opt)
```
Arguments:
- `variable` : `Symbol` denoting the name of the covariate.
- `β_opt` : `β_opt` denoting the coefficient.
"""
struct NoSmoothData
    variable::Symbol
    β_opt::Float64
end