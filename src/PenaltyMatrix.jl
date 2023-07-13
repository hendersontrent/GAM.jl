"""
    PenaltyMatrix(Basis, λ, x, y)
Compute a matrix of the data augmented with a penalty.
Usage:
```julia-repl
PenaltyMatrix(Basis, λ, x, y)
```
Arguments:
- `Basis` : `BSplineBasis` containing the quantile B-spline basis.
- `λ` : `Float64` denoting the penalty value.
- `x` : `AbstractVector` containing the predictor variable.
- `y` : `AbstractVector` containing the response variable.
"""

function PenaltyMatrix(Basis::BSplineBasis{Vector{Float64}}, λ::Float64, x::AbstractVector, y::AbstractVector)

    X = BasisMatrix(Basis, x) # Basis Matrix
    D = DifferenceMatrix(Basis) # D penalty matrix
    Xp = vcat(X, sqrt(λ)*D) # augment model matrix with penalty
    yp = vcat(y, repeat([0],size(D)[1])) # augment data with penalty

    return Xp, yp
end