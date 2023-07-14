"""
    FitGAM(formula, data; family, link, optimizer)
Computes a basic generalized additive model (GAM) on input data. An intercept is added by default.
Usage:
```julia-repl
FitGAM(formula, data; family, link, optimizer)
```
Arguments:
- `formula` : `String` containing the expression of the model. Continuous covariates are wrapped in s() like `mgcv` in R, where `s()` has 3 parts: name of column, `k`` (integer denoting number of knots), and `degree` (polynomial degree of the spline). An example expression is `"Y ~ s(MPG, k=5, degree=3) + WHT + s(TRL, k=5, degree=2)"`
- `data` : `DataFrame` containing the covariates and response variable to use.
- `family` : `Distribution` denoting the likelihood to use. Must be one of the options in `GLM.jl`. Defaults to `Normal()`.
- `link` : denoting the link function to use for `family`. Defaults to the canonical link of `family`.
- `optimizer` : `Optim.jl` optimizer to use. Defaults to `GradientDescent()`. Other common choices might be `BFGS()` or `LBFGS()`.
"""

function FitGAM(formula::String, data::DataFrame; family=Normal(), link=canonicallink(family), optimizer=GradientDescent())

    # Add a column of ones to the dataframe for the intercept term and add to formula

    intercept = DataFrame(Intercept = ones(size(data, 1)))
    data = hcat(intercept, data)
    formula = split(formula, " ~ ")[1] * " ~ :Intercept + " * split(formula, " ~ ")[2]
    GAMForm = ParseFormula(formula)
    y = data[!, GAMForm.y]

    #---------------- Do spline operations for each covariate ---------------

    covariateFits = Union{SmoothData, NoSmoothData}[]

    for i in 1:nrow(GAMForm.covariates)

        if GAMFom.covariates[i, 4] == true

            # Extract spline components

            variable = GAMForm.covariates[i, 1]
            x = data[!, variable]
            k = GAMForm.covariates[i, 2]
            degree = GAMForm.covariates[i, 3] + 1 # Add 1 to degree to get the degree needed to fit what was specified

            # Compute basis

            Basis = QuantileBasis(x, k, degree)

            # Compute optimised λ

            λ_opt = OptimizeGCVLambda(Basis, x, y, optimizer)

            # Build penalised design matrix

            Xp_opt, yp_opt = PenaltyMatrix(Basis, λ_opt, x, y)

            # Fit GLM for coefficients

            X_tmp = DataFrame(Xp_opt, :auto)
            y_tmp = DataFrame(yp_opt = yp_opt)
            GAMModelMatrix = hcat(y_tmp, X_tmp)
            cov_names = names(GAMModelMatrix)[2:end].|> Symbol
            f = Term(:yp_opt)~sum(Term.(cov_names))
            β_opt = coef(glm(f, GAMModelMatrix, family, link))[2:end] # Need to find a way to remove the intercept...
        
            # Define optimised spline object
        
            Spline_opt = Spline(Basis, β_opt)

            # Add to struct

            predictorFit = SmoothData(variable, β_opt, λ_opt, Spline_opt)

            # Store in covariate array

            push!(covariateFits, predictorFit)
        else
            
            # Extract components and fit GLM for coefficient

            variable = GAMForm.covariates[i, 1]
            x = data[!, variable]
            GAMModelMatrix = DataFrame(x = x, y = y)
            β_opt = coef(glm(@formula(y ~ x), GAMModelMatrix, family, link))[2:end] # Need to find a way to remove the intercept...

            # Add to struct

            predictorFit = NoSmoothData(variable, β_opt)

            # Store in covariate array
            
            push!(covariateFits, predictorFit)
        end
    end

    #---------------- Compute final GAM ---------------

    # NEED TO CREATE THE `model` OBJECT HERE FOR BELOW DOING THE ACTUAL ADDITIVE PROCESS

    # Return final object

    outs = GAMModel(formula, data, model, covariateFits)
    return(outs)
end