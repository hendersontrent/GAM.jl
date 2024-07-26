"""
    FitGAM(y, x, Dist, Link, BasisArgs; Optimizer, maxIter, tol)
Fit generalised additive model.

Usage:
```julia-repl
FitGAM(y, x, Dist, Link, BasisArgs; Optimizer, maxIter, tol)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `x` : `Array` of input data.
- `Dist` : Likelihood distribution.
- `Link` : Link function.
- `BasisArgs` : `Array` denoting the number of knots and polynomial order for each spline. For example, `BasisArgs` would be of the form `[(10, 2), (10,2)]` if there were two covariates where each has 10 knots and a polynomial order of 2.
- `Optimizer` : Algorithm to use for optimisation. Defaults to `NelderMead()`.
- `maxIter` : Maximum number of iterations for algorithm.
- `tol` : Tolerance for solver.
"""
function FitGAM(y, x, Dist, Link, BasisArgs; Optimizer = NelderMead(), maxIter = 25, tol = 1e-6)

    Basis = map((xi, argi) -> BuildUniformBasis(xi, argi[1], argi[2]), x, BasisArgs)
    
    gam = OptimPIRLS(y, x, Basis, Dist, Link; Optimizer, maxIter, tol)
    return gam
end
