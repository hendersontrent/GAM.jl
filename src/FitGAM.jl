"""
    gam(ModelFormula, Data; Dist, Link, Optimizer, maxIter, tol)
Fit generalised additive model.

Usage:
```julia-repl
gam(ModelFormula, Data; Dist, Link, Optimizer, maxIter, tol)
```
Arguments:
- `ModelFormula` : `String` containing the expression of the model. Continuous covariates are wrapped in s() like `mgcv` in R, where `s()` has 3 parts: name of column, `k`` (integer denoting number of knots), and `degree` (polynomial degree of the spline). An example expression is `"Y ~ s(MPG, k=5, degree=3) + WHT + s(TRL, k=5, degree=2)"`
- `data` : `DataFrame` containing the covariates and response variable to use.
- `Family` : `String` specifying Likelihood distribution. Should be one of "Normal", "Poisson", or "Gamma". Defaults to "Normal"
- `Link` : `String` specifying link function distribution. Should be one of "Identity" or "Log". Defaults to "Identity"
- `Optimizer` : Algorithm to use for optimisation. Defaults to `NelderMead()`.
- `maxIter` : Maximum number of iterations for algorithm.
- `tol` : Tolerance for solver.
"""
function gam(ModelFormula::String, Data::DataFrame; Family="Normal", Link="Identity", Optimizer = NelderMead(), maxIter = 25, tol = 1e-6)

    family_name = Dist_Map[Family]
    family_name = Dists[family_name]
    link_name = Link_Map[Link]
    link_name = Links[link_name]

    # Parse formula and generate design matrix and response variable vector

    #ModelFormula = split(ModelFormula, " ~ ")[1] * split(ModelFormula, " ~ ")[2]
    GAMForm = ParseFormula(ModelFormula)
    y = Data[!, GAMForm.y]
    x = Data[!, GAMForm.covariates.variable]
    BasisArgs = [(GAMForm.covariates.k[i], GAMForm.covariates.degree[i]) for i in 1:nrow(GAMForm.covariates)]
    x = [x[!, col] for col in names(x)]

    # Build basis

    Basis = map((xi, argi) -> BuildUniformBasis(xi, argi[1], argi[2]), x, BasisArgs)

    # Fit PIRLS procedure
    
    gam = OptimPIRLS(y, x, Basis, family_name, link_name; Optimizer, maxIter, tol)
    return gam
end
