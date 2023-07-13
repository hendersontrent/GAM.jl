"""
    FitGAM(x, y, n_knots, degree)
Computes a basic generalized additive model (GAM) on input data.
Usage:
```julia-repl
FitGAM(X, y, n_knots, degree)
```
Arguments:
- `x` : `AbstractVector` of the predictor variable.
- `y` : `AbstractVector` of the response variable.
- `n_knots` : `Int64` denoting the number of knots to use in the spline. Defaults to half the length of `x`.
- `degree` : `Int64` denoting the polynomial degree of the spline. Defaults to `3` for a cubic spline.
"""

function FitGAM(x::AbstractVector, y::AbstractVector; n_knots::Int64=floor(Int64, size(x)[1]), degree::Int64=3)

    # Add 1 to degree to get the degree needed to fit what was specified

    degree = degree + 1

    # Compute basis

    # NOTE: For the future version where X is a matrix input and we have multiple variables,
    # we could for loop over the creation of Basis through to Spline_opt and then sum
    # to get our "additive model". This might need a column of 1s prepended to X for the intercept

    Basis = QuantileBasis(x, n_knots, degree);

    # Compute optimised λ

    λ_opt = OptimizeGCVLambda(Basis, x, y);

    # Build penalised design matrix

    Xp_opt, yp_opt = PenaltyMatrix(Basis, λ_opt, x, y);

    # Fit Optimized Spline
    
    β_opt = coef(lm(Xp_opt, yp_opt));
    
    # Define Optimized Spline Object
    
    Spline_opt = Spline(Basis, β_opt);

    # Return final object

    outs = GAMModel(x, y, λ_opt, Xp_opt, yp_opt, Spline_opt)
    return(outs)
end