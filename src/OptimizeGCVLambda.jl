"""
    OptimizeGCVLambda(BasisMatrices, Differences, y, optimizer)
Actual optimisation algorithm called within the generalised cross validation procedure.
Usage:
```julia-repl
OptimizeGCVLambda(BasisMatrices, Differences, y, optimizer)
```
Arguments:
- `BasisMatrices` : `BSplineBasis` containing the quantile B-spline basis.
- `Differences` : `AbstractVector` containing the difference matrices.
- `y` : `AbstractVector` containing the response variable.
- `optimizer` : `Optim.jl` optimizer to use. Defaults to `GradientDescent()`. Other common choices might be `BFGS()` or `LBFGS()`.
"""

function OptimizeGCVLambda(BasisMatrices::AbstractVector, Differences::AbstractVector, y::AbstractVector, optimizer=GradientDescent())
    k = length(BasisMatrices)
    lower = zeros(k)
    upper = fill(Inf, k)
    initial_lambda = fill(1.0, k)

    res = Optim.optimize(
        lambdas -> GCV(lambdas, BasisMatrices, y, Differences), 
        lower, upper, initial_lambda, 
        Fminbox(optimizer)
    )

    return Optim.minimizer(res)
end
