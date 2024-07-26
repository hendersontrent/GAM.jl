"""
    ModelDiagnostics(y, X, D, W, B)
Computes and stores model diagnostic information.

Usage:
```julia-repl
ModelDiagnostics(y, X, D, W, B)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `X` : `Matrix` of input data.
- `D` : `xxx` xx.
- `w` : `xxx` xx.
- `B` : `xxx` xx.
"""
function ModelDiagnostics(y, X, D, W, B)
    n = length(y)
    H = HatMatrix(X, D, W) # hat matrix
    trF = sum(diag(H)[1:n]) # EDF
    rsd = y - (X * B) # residuals
    rss = sum(rsd.^2) # residual SS
    sig_hat = rss/(n-trF) # residual variance
    gcv = sig_hat*n/(n-trF) # GCV score

    return Dict(
        :RSS => rss,
        :EDF => trF,
        :GCV => gcv
    )
end