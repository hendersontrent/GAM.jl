"""
    gam(ModelFormula, data; family, link, optimizer)
Computes a generalized additive model (GAM). An intercept is added by default.
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

    #------------ Penalty optimisation procedure -----------

    # Track which columns correspond to which predictor

    ###

    #--------------------
    # Handle smooth terms
    #--------------------

    smoothCovs = GAMForm.covariates[GAMForm.covariates.smooth .== true, :]
    BasisMatrices = []
    Differences = []

    for i in 1:nrow(smoothCovs)
        col_data = data[!, smoothCovs[i, 1]]
        knots = quantile(col_data, range(0, 1; length = smoothCovs[i, 2]))
        basis = BSplineBasis(smoothCovs[i, 3], knots)
        X = BasisMatrix(basis, col_data)
        D = DifferenceMatrix(basis)
        push!(BasisMatrices, X)
        push!(Differences, D)
    end

    #------------------------
    # Handle non-smooth terms
    #------------------------

    nonSmoothCovs = GAMForm.covariates[GAMForm.covariates.smooth .== false, :]

    # Create identity matrices for Differences

    DiffsNoSmooth = []

    for i in 1:nrow(nonSmoothCovs)
        Identity = I(size(data)[1])
        push!(DiffsNoSmooth, Identity)
    end

    #---------------------
    # Matrix concatenation
    #---------------------

    ###

    #------------------------------
    # Actual optimisation procedure
    #------------------------------

    λs = OptimizeGCVLambda(BasisMatrices, Differences, data[:, y_var], optimizer)

    #------------------------------
    # Build penalised design matrix
    #------------------------------
    
    X_p = Matrix(
        vcat(
            # cbind the Basis Matricies
            hcat(BasisMatrices...), 
            # create a block diagonal matrix of penalized differences
            blockdiag((sqrt.(λs).*sparse.(Differences))...)
            )
        )

    #------------------------------------------
    # Build augmented penalty response variable
    #------------------------------------------

    y_p = vcat(y, repeat([0], sum(first.(size.(Differences)))))

    #------------ Model fit -----------

    # Prepare single dataset in a DataFrame format for GLM.jl

    trial_df = hcat(y_p, X_p)
    trial_df = DataFrame(trial_df, :auto)
    rename!(trial_df, :x1 => :y)

    # Generate dynamic formula with intercept removed since we manually added it earlier

    f = @eval(@formula($(Meta.parse("y ~ 0 + " * join(names(trial_df[:, Not(:y)]), " + ")))))

    # Fit model

    mod = glm(f, trial_df, family, link)

    #------------ Generate GAM object -----------

    ModelInformation = ModelFit()

    # Return final object

    outs = GAMModel(ModelFormula, data, y_var, ModelInformation)
    return(outs)
end
