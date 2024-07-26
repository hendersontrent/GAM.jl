"""
    FitOLS(y, x, sp, Basis)
Fit ordinary least squares model.

Usage:
```julia-repl
FitOLS(y, x, sp, Basis)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `x` : `Vector` of input data.
- `sp` : `xxx` xx.
- `Basis` : `AbstractArray` containing the basis matrix.
"""
function FitOLS(y, x, sp, Basis)

    n = length(y)
    X, Y, D, ColMeans, CoefIndex = BuildPenaltyMatrix(y, x, sp, Basis)

    Xp = vcat(X, D)
    Yp = vcat(y, repeat([0], size(D,1)))
    B = Xp \ Yp
    fitted = (Xp * B)[1:n]
    diagnostics = ModelDiagnostics(y, X, D, Diagonal(ones(n)), B)

    return GAMData(
        y,
        x,
        Basis,
        Dists[:Normal],
        Links[:Identity],
        B,
        ColMeans,
        CoefIndex,
        fitted,
        diagnostics
    )
end
