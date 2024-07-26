"""
    FitWPS(y, x, sp, Basis, w)
Fit WPS model.

Usage:
```julia-repl
FitWPS(y, x, sp, Basis, w)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `x` : `Vector` of input data.
- `sp` : `xxx` xx.
- `Basis` : `AbstractArray` containing the basis matrix.
"""
function FitWPS(y, x, sp, Basis, w = ones(length(y)))

    X, Y, D, ColMeans, CoefIndex = BuildPenaltyMatrix(y, x, sp, Basis)
    W = Diagonal(w)
    # Left-hand side (LHS) matrix
    LHS = X' * W * X + D' * D
    # Right-hand side (RHS) vector
    RHS = X' * W * Y
    # Solve for beta
    B = LHS \ RHS
    # Fitted values
    fitted = (X * B)
    # Run Diagnostics
    diagnostics = ModelDiagnostics(y, X, D, W, B)

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
