"""
    gam(ModelFormula, data; family, link, optimizer)
Computes a basic generalized additive model (GAM) on input data. An intercept is added by default.
Usage:
```julia-repl
gam(formula, data; family, link, optimizer)
```
Arguments:
- `ModelFormula` : `String` containing the expression of the model. Continuous covariates are wrapped in s() like `mgcv` in R, where `s()` has 3 parts: name of column, `k`` (integer denoting number of knots), and `degree` (polynomial degree of the spline). An example expression is `"Y ~ s(MPG, k=5, degree=3) + WHT + s(TRL, k=5, degree=2)"`
- `data` : `DataFrame` containing the covariates and response variable to use.
- `family` : `Distribution` denoting the likelihood to use. Must be one of the options in `GLM.jl`. Defaults to `Normal()`.
- `link` : denoting the link function to use for `family`. Defaults to the canonical link of `family`.
- `optimizer` : `Optim.jl` optimizer to use. Defaults to `GradientDescent()`. Other common choices might be `BFGS()` or `LBFGS()`.
"""

function gam(ModelFormula::String, data::DataFrame; family=Normal(), link=canonicallink(family), optimizer=GradientDescent())

    # Add a column of ones to the dataframe for the intercept term and add to formula

    intercept = DataFrame(Intercept = ones(size(data, 1)))
    data = hcat(intercept, data)
    ModelFormula = split(ModelFormula, " ~ ")[1] * " ~ :Intercept + " * split(ModelFormula, " ~ ")[2]
    GAMForm = ParseFormula(ModelFormula)
    y_var = GAMForm.y
    y = data[!, y_var]

    #---------------- Statistical calculations ---------------

    # Set up smooth covariates

    smooth_covariates = GAMForm.covariates[GAMForm.covariates[:, :smooth] .== true, :]
    smooth_cols = smooth_covariates[:, :variable]
    smooth_df = select(data, smooth_cols)
    smooth_x = Vector{Vector{Float64}}(undef, length(smooth_cols))

    for (i, col) in enumerate(smooth_cols)
        smooth_x[i] = data[!, col]
    end

    # Set up non-smooth covariates

    non_smooth_covariates = GAMForm.covariates[GAMForm.covariates[:, :smooth] .== false, :]
    non_smooth_cols = smooth_covariates[:, :variable]
    non_smooth_df = select(data, non_smooth_cols)

    #--------------------
    # Handle smooth terms
    #--------------------

    BasisCalcs = BSplineBasis[]

    for i in 1:nrow(smooth_covariates)
        variable = smooth_covariates[i, :variable]
        x_basis = data[!, variable]
        k = smooth_covariates[i, :k]
        degree = smooth_covariates[i, :degree] + 1 # Add 1 to degree to get the degree needed to fit what was specified
        Basis = QuantileBasis(x_basis, k, degree)
        push!(BasisCalcs, Basis)
    end

    # Compute basis and difference matrix

    X = BasisMatrix.(BasisCalcs, smooth_x) # Basis Matrix
    D = DifferenceMatrix.(BasisCalcs) # D penalty matrix

    # Create a vector of penalties

    λ = [10, 10]

    # Build penalty design matrix

    X_p = Matrix(
        vcat(
            # Column bind the Basis Matricies
            hcat(X...), 
            # Create a block diagonal matrix of penalized differences
            blockdiag((sqrt.(λ).*sparse.(D))...)
        )
    )

    # Create augmented penalty response

    y_p = vcat(y, repeat([0],sum(first.(size.(D)))))

    # Fit GLM

    #------------------------
    # Handle non-smooth terms
    #------------------------

    #

    #---------------- Compute final GAM ---------------

    # Compute actual additive process

    #

    # Return final object

    outs = GAMModel(ModelFormula, y_var, data, covariateFits)
    return(outs)
end