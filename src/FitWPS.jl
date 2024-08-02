"""
    FitWPS(y, x, sp, Basis, Dist, Link, w)
Fit WPS model.

Usage:
```julia-repl
FitWPS(y, x, sp, Basis, Dist, Link, w)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `x` : `Vector` of input data.
- `sp` : `Float` of the optimised smoothing parameter.
- `Dist` : Likelihood distribution.
- `Link` : Link function.
- `w` : `Vector` of weights.
"""
function FitWPS(y, x, sp, Basis, Dist, Link, w = ones(length(y)))

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
        Dist,
        Link,
        B,
        ColMeans,
        CoefIndex,
        fitted,
        diagnostics
    )
end
