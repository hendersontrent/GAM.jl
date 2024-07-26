"""
    BuildDifferenceMatrix(Basis)
Computes a matrix of second differences on a basis matrix to penalise the second derivative at the knots of the function.
Usage:
```julia-repl
BuildDifferenceMatrix(Basis, x)
```
Arguments:
- `Basis` : `BSplineBasis` containing the quantile B-spline basis.
"""

function BuildDifferenceMatrix(Basis::BSplineBasis)
    #nk = length(Basis.breakpoints)
    D = diffm(
        diagm(0 => ones(length(Basis))),
        1, # matrix dimension
        2  # number of differences
    )
    return D
end