"""
    GCV(param, BasisMatrices, y, Differences)
Perform generalised cross validation to optimise λ (penalty).
Usage:
```julia-repl
GCV(param, BasisMatrices, y, Differences)
```
Arguments:
- `param` : `AbstractVector` denoting the parameter to be optimised using GCV.
- `BasisMatrices` : `Vector` containing the quantile B-spline basis matrices.
- `y` : `AbstractVector` containing the response variable.
- `Differences` : `AbstractVector` containing the difference matrices.
"""

function GCV(param::AbstractVector, BasisMatrices::AbstractVector, y::AbstractVector, Differences)
    n = length(y)
    k = length(BasisMatrices)
    lambdas = param[1:k]
    penalties = sum([λ .* D for (λ, D) in zip(lambdas, Differences)])
    X = hcat(BasisMatrices...)
    XtX = X'X
    XtY = X'y
    H = X*inv(XtX + penalties)*X' # Hat matrix
    trF = sum(diag(H))
    y_hat = X*inv(XtX + penalties)*XtY
    rss = sum((y - y_hat).^2) # Residual sums of squares
    gcv = n*rss/(n - trF)^2
    return gcv
end