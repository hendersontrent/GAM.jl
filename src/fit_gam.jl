"""
    GAMModel(formula, data, β, knots, degree, n_knots, family, λ, log_lik, aic, bic)
Holds information relevant to the GAM model including its components and fit statistics.

Arguments:
- `formula` : The model formula.
- `data` : DataFrame of input data.
- `β` : Model coefficients.
- `knots` : Spline knot positions.
- `degree` : Polynomial degree of the spline.
- `n_knots` : Number of knots in the spline.
- `family` : Family of the likelihood function.
- `λ` : Coefficient of the penalty term.
- `log_lik` : The log-likelihood of the model.
- `aic` : The Akaike information criterion.
- `bic` : The Bayesian information criterion.
"""
struct GAMModel
    formula::Expr
    df::DataFrame
    β::Vector{Float64}
    knots::Vector{Vector{Float64}}
    degree::Int
    n_knots::Vector{Int}
    family::Function
    λ::Float64
    log_lik::Float64
    aic::Float64
    bic::Float64
end

"""
    fit_gam(formula, data, family, knots, degree, n_knots, λ, n_folds)
Fits a generalised additive model (GAM) for a range of different likelihood distributions and computes model fit statistics.

Usage:
```julia-repl
fit_gam(formula, data, family)
```
Arguments:
- `formula` : The model formula.
- `data` : DataFrame of input data.
- `family` : Family of the likelihood function.
- `λ` : Coefficient of the penalty term.
- `n_folds` : Number of folds to use in generalised cross-validation of parameters.
"""
function fit_gam(formula::Union{Symbol, Expr}, data::DataFrame, family::Union{Symbol, Function}, λ::Union{Nothing, Float64}=nothing, n_folds::Int=5)
    # Check if the formula includes an intercept term
    if typeof(formula) == Expr
        # Check if the formula includes the intercept term
        if !(:Intercept in formula.rhs)
            # Add the intercept term to the formula
            formula = :(:($(formula.lhs)) ~ 1 + $(formula.rhs))
        end
    else
        # Add the intercept term to the formula
        formula = :(:($formula) ~ 1)
    end

    # Parse the formula and data
    y, smooth_terms, cat_vars, knots, degree, polynomial_degree = parse_formula(data, formula)

    # Extract the predictor variables and their spline basis functions
    X = Array{Float64}(undef, size(data, 1), 0)
    spline_basis = Vector{Matrix{Float64}}(undef, length(smooth_terms))

    # Iterate over the smooth terms
    for (i, term) in enumerate(smooth_terms)
        # Check if the term is the intercept
        if term == :Intercept
            X = hcat(X, ones(size(data, 1)))
        else
            # Extract the predictor and its spline basis functions
            predictor = data[term]
            knots_i = knots[i]
            degree_i = degree[i]
            polynomial_degree_i = polynomial_degree[i]
            spline_basis[i] = hcat(BSplineBasis(knots_i, degree_i, predictor, polynomial_degree_i))
            spline_basis[i] = spline_basis[i][:, 2:end]
            X = hcat(X, spline_basis[i])
        end
    end

    # Add the categorical variables to the predictor matrix
    for term in cat_vars
        # Convert the categorical variable to dummy variables
        X = hcat(X, dummy_encoder(data[term]))
    end

    # Initialize the model coefficients to zeros
    n_features = size(X, 2)
    β = zeros(n_features)

    # Set the distribution based on the likelihood type

    if family === :gaussian
        dist = Normal()
    elseif family === :binomial
        dist = Binomial()
    elseif family === :poisson
        dist = Poisson()
    elseif family === :gamma
        dist = Gamma()
    else
        error("Likelihood distribution family not recognised.")
    end

    # If λ is not specified, optimize it using cross-validation

    if λ === nothing
        lambda_vals = logspace(-3, 3, 7)
        cv_scores = Vector{Float64}(undef, length(lambda_vals))

        for (i, lambda_val) in enumerate(lambda_vals)
            # Compute the cross-validated log likelihood for λ
            fold_log_lik = Vector{Float64}(undef, n_folds)

            for j in 1:n_folds
                # Split the data into training and test sets

                idx = foldind(data[y], n_folds, j)
                X_train = X[.!idx, :]
                y_train = data[y][.!idx]
                X_test = X[idx, :]
                y_test = data[y][idx]

                # Compute the penalized log likelihood for the training data

                res = optimize(β -> penalized_log_lik(X_train, y_train, β, dist, lambda_val, penalty_matrix), β, BFGS(), Optim.Options(show_trace=false))
                β = res.minimizer
                fold_log_lik[j] = log_lik(X_test, y_test, β, dist)
            end

            # Compute the average cross-validated log likelihood
            cv_scores[i] = mean(fold_log_lik)
        end

            # Select the λ value with the highest cross-validated log likelihood
            λ = lambda_vals[argmax(cv_scores)]
    end

    # Compute the penalized log likelihood for the full dataset
    res = optimize(β -> penalized_log_lik(X, data[y], β, dist, λ, penalty_matrix), β, BFGS(), Optim.Options(show_trace=false))
    β = res.minimizer

    # Compute the log likelihood and information criteria for the model
    log_lik_val = log_lik(X, data[y], β, dist)
    n_obs = size(X, 1)
    n_params = size(X, 2)
    aic_val = -2 * log_lik_val + 2 * n_params
    bic_val = -2 * log_lik_val + n_params * log(n_obs)

    # Return the fitted model
    return GAMModel(formula, data, β, knots, degree, n_knots, family, λ, log_lik_val, aic_val, bic_val)
end
