"""
    FitGAM(y, x, sp, Basis, Dist, Link; Optimizer, maxIter, tol)
Fit generalised additive model.

Usage:
```julia-repl
FitGAM(y, x, sp, Basis, Dist, Link; Optimizer, maxIter, tol)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `x` : `Vector` of input data.
- `sp` : `Float` of the optimised smoothing parameter.
- `Basis` : `AbstractArray` containing the basis matrix.
- `Dist` : Likelihood distribution.
- `Link` : Link function.
- `Optimizer` : Algorithm to use for optimisation. Defaults to `NelderMead()`.
- `maxIter` : Maximum number of iterations for algorithm.
- `tol` : Tolerance for solver.
"""
function FitGAM(y, x, Basis, Dist, Link; Optimizer = NelderMead(), maxIter = 25, tol = 1e-6)

    gam = OptimPIRLS(y, x, Basis, Dist, Link; Optimizer, maxIter, tol)
    return gam
end
