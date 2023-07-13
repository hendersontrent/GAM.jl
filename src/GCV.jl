"""
    GCV(param, Basis, x, y)
Perform generalised cross validation to optimise λ (penalty).
Usage:
```julia-repl
GCV(param, Basis, x, y)
```
Arguments:
- `param` : `AbstractVector` denoting the parameter to be optimised using GCV.
- `Basis` : `BSplineBasis` containing the quantile B-spline basis.
- `x` : `AbstractVector` containing the predictor variable.
- `y` : `AbstractVector` containing the response variable.
"""

function GCV(param::AbstractVector, Basis::BSplineBasis{Vector{Float64}}, x::AbstractVector, y::AbstractVector)
    n = length(Basis.breakpoints)
    Xp, yp = PenaltyMatrix(Basis, param[1], x, y)
    β = coef(lm(Xp,yp))
    H = Xp*inv(Xp'Xp)Xp' # Hat matrix
    trF = sum(diag(H)[1:n])
    y_hat = Xp*β
    rss = sum((yp-y_hat)[1:n].^2) # Residual sums of squares
    gcv = n*rss/(n-trF)^2
    return gcv
end