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

function GCV(param::AbstractVector, BasisMatrices::AbstractVector, y::AbstractVector, Differences::AbstractVector)
    n = length(y)
    k = length(BasisMatrices)
    lambdas = param[1:k]

    # Build Design Matrix

    X_int = vcat(
            repeat([1], n),
            repeat([0], sum(size.(Differences,1)))
        )
    # Initialize list of Design Matrix Elements

    X_elements = []

    for i in 1:k
        #  Get Basis Matrix For Predictor
        X_i = BasisMatrices[i]
        # Build List of Empty Difference Matrices
        D_i = [zeros(size(D,1), size(Differences[i],2)) for D in Differences]
        # Add in Penalized Difference Matrix
        D_i[i] = sqrt(lambdas[i]) * Differences[i]
        # Concatenate Penalized Difference Matrix
        D_i = vcat(D_i...)
        # Append to Design Matrix
        push!(X_elements, vcat(X_i, D_i))
    end

    # Could add more non-smooth predictors here...

    X = Matrix(hcat(X_int, X_elements...))
    Y = vcat(y, repeat([0],sum(size.(Differences,1))))

    # Compute GCV

    b = X \ Y # Coefficients
    H = X * inv(X' * X) * X' # Hat Matrix
    trA = sum(diag(H)[1:n]) # EDF
    y_hat = X * b # Fitted values
    rsd = y - y_hat[1:n] # Residuals
    rss = sum(rsd.^2) # Residual SS
    sig_hat = rss/(n-trA) # Residual variance
    gcv = sig_hat*n/(n-trA) # GCV score
    return gcv
end