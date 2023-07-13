"""
    QuantileBasis(x, n_knots, order)
Compute a B-spline basis matrix for a predictor variable over a given number of knots and polynomial degree.
Usage:
```julia-repl
QuantileBasis(x, n_knots, order)
```
Arguments:
- `x` : `AbstractVector` containing the predictor variable.
- `n_knots` : `Int64` denoting the number of knots to use in the spline. Defaults to half the length of `x`.
- `degree` : `Int64` denoting the polynomial degree of the spline. Defaults to `3` for a cubic spline.
"""

function QuantileBasis(x::AbstractVector, n_knots::Int64, order::Int64)

    # Build a list of the Knots (breakpoints)

    KnotsList = quantile(x, range(0, 1; length=n_knots));

    # Define the Basis Object
    
    Basis = BSplineBasis(order, KnotsList);

    return Basis
end