"""
    OptimizeGCVLambda(Basis, x, y)
Actual optimisation algorithm called within the generalised cross validation procedure.
Usage:
```julia-repl
OptimizeGCVLambda(Basis, x, y)
```
Arguments:
- `Basis` : `BSplineBasis` containing the quantile B-spline basis.
- `x` : `AbstractVector` containing the predictor variable.
- `y` : `AbstractVector` containing the response variable.
"""

function OptimizeGCVLambda(Basis::BSplineBasis{Vector{Float64}}, x::AbstractVector, y::AbstractVector)

    # Optimization bounds 

    lower = [0]
    upper = [Inf]
    initial_lambda = [1.0]

    # Run Optimization

    inner_optimizer = GradientDescent()
    res = Optim.optimize(
        lambda -> GCV(lambda, Basis, x, y), 
        lower, upper, initial_lambda, 
        Fminbox(inner_optimizer)
    )
    return Optim.minimizer(res)[1]
end