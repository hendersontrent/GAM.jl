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

    #---------------- Do spline operations for each covariate ---------------

    covariateFits = Union{SmoothData, NoSmoothData}[]

    for i in 1:nrow(GAMForm.covariates)

        if GAMForm.covariates[i, 4] == true

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
            mod = glm(f, GAMModelMatrix, family, link)
            β_opt = coef(mod)[2:end] # Need to find a way to remove the intercept...
            β_opt_confint = confint(mod)[2:end, :]

            # Define optimised spline object
        
            Spline_opt = Spline(Basis, β_opt)
            Spline_opt_lower = Spline(Basis, β_opt_confint[:, 1])
            Spline_opt_upper = Spline(Basis, β_opt_confint[:, 2])

            # Add to struct

            predictorFit = SmoothData(variable, β_opt, β_opt_confint, λ_opt, Spline_opt, Spline_opt_lower, Spline_opt_upper)

            # Store in covariate array

            push!(covariateFits, predictorFit)
        else
            
            # Extract components and fit GLM for coefficient

            variable = GAMForm.covariates[i, 1]
            x = data[!, variable]
            GAMModelMatrix = DataFrame(x = x, y = y)
            mod = glm(@formula(y ~ x), GAMModelMatrix, family, link) # Need to find a way to remove the intercept...
            β_opt = coef(mod)[2:end] # Need to find a way to remove the intercept...
            β_opt_confint = confint(mod)[2:end, :]

            # Add to struct

            predictorFit = NoSmoothData(variable, β_opt[1], β_opt_confint)

            # Store in covariate array
            
            push!(covariateFits, predictorFit)
        end
    end

    #---------------- Compute final GAM ---------------

    # Compute actual additive process



    # Return final object

    outs = GAMModel(ModelFormula, y_var, data, covariateFits)
    return(outs)
end