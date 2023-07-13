"""
    BasisMatrix(Basis, x)
Computes a basis matrix for a given predictor variable using B-splines.
Usage:
```julia-repl
BasisMatrix(Basis, x)
```
Arguments:
- `Basis` : `BSplineBasis` containing the quantile B-spline basis.
- `x` : `AbstractVector` containing the predictor variable.
"""

function BasisMatrix(Basis::BSplineBasis{Vector{Float64}}, x::AbstractVector)
    splines = vec(
        mapslices(
            x -> Spline(Basis,x), 
            diagm(ones(length(Basis))),
            dims=1
        )
    );
    X = hcat([s.(x) for s in splines]...)
    return X 
end