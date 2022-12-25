"""
    GAMModel(β, knots, degree, n_knots, likelihood, λ, log_lik, aic, bic)
Holds information relevant to the GAM model including its components and fit statistics.

Usage:
```julia-repl
fit_gam(X, y)
```
Arguments:
- `β` : Model coefficients.
- `knots` : Spline knot positions.
- `degree` : Polynomial degree of the spline.
- `n_knots` : Number of knots in the spline.
- `likelihood` : Family of the likelihood function.
- `λ` : Regularization coefficient.
- `log_lik` : The log-likelihood of the model.
- `aic` : The Akaike information criterion.
- `bic` : The Bayesian information criterion.
"""
struct GAMModel
    β::Vector{Float64}
    knots::Vector{Vector{Float64}}
    degree::Int
    n_knots::Vector{Int}
    likelihood::Function
    λ::Float64
    log_lik::Float64
    aic::Float64
    bic::Float64
end

"""
    fit_gam(X, y, likelihood, knots, degree, n_knots, λ, n_folds)
Fits a generalised additive model (GAM) for a range of different likelihood distributions and computes model fit statistics.

Usage:
```julia-repl
fit_gam(X, y)
```
Arguments:
- `X` : Data matrix of predictor variables.
- `y` : Response variable vector.
- `likelihood` : Family of the likelihood function.
- `knots` : Spline knot positions.
- `degree` : Polynomial degree of the spline.
- `n_knots` : Number of knots in the spline.
- `λ` : Coefficient of the penalty term.
- `n_folds` : Number of folds to use in generalised cross-validation of parameters.
"""
function fit_gam(X::Array{Float64, 2}, y::Array{Float64, 1}, likelihood::Union{Symbol, Function}, knots::Union{Nothing, Vector{Vector{Float64}}}=nothing, degree::Int=3, n_knots::Union{Nothing, Vector{Int}}=nothing, λ::Union{Nothing, Float64}=nothing, n_folds::Int=5)

    # Add a column of ones to X for the intercept term

    X = hcat(ones(size(X, 1)), X)

    # Create spline basis functions for each predictor

    n_samples, n_features = size(X)
    spline_basis = [zeros(size(X, 1), 1) for i in 1:n_features]

    if knots === nothing
        knots = Vector{Vector{Float64}}(undef, n_features)
        n_knots = Vector{Int}(undef, n_features)
        for i in 1:n_features
            # Set default value for n_knots to ensure at least 2 knots are generated
            knots[i], _ = optimal_knots(X[:, i+1], degree, max(2, get(n_knots, i, 50)))
            n_knots[i] = length(knots[i]) - degree - 1
        end
    end

    for i in 1:n_features
        spline_basis[i] = hcat(spline_basis[i], BSplineBasis(knots[i], degree, X[:, i]))
        spline_basis[i] = spline_basis[i][:, 2:end]
    end

    # Concatenate the spline basis functions with the intercept term

    X = hcat(spline_basis..., X)

    # Initialize the model coefficients to zeros

    n_features = size(X, 2)
    β = zeros(n_features)

    # Set the distribution based on the likelihood type

    if likelihood === :gaussian
        dist = Normal()
    elseif likelihood === :binomial
        dist = Binomial()
    elseif likelihood === :poisson
        dist = Poisson()
    else
        error("Likelihood distribution not recognised.")
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

                idx = foldind(y, n_folds, j)
                X_train = X[.!idx, :]
                y_train = y[.!idx]
                X_test = X[idx, :]
                y_test = y[idx]

                # Compute the penalized log likelihood for the training data

                res = optimize(β -> penalized_log_lik(X_train, y_train, β, dist, lambda_val, penalty_matrix), β, BFGS(), Optim.Options(show_trace=false))
                β = res.minimizer
                fold_log_lik[j] = log_lik(X_test, y_test, β, dist)
            end

            # Compute the average cross-validated log likelihood
            cv_scores[i] = mean(fold_log_lik)
        end

        # Select the λ value with the lowest cross-validation score

        lambda_idx = argmin(cv_scores)
        λ = lambda_vals[lambda_idx]

        # Re-initialize the model coefficients to zeros

        β = zeros(n_features)
    end

    # Compute the penalty matrix for the spline basis functions

    penalty = zeros(size(X, 2))
    penalty[2:end] = 1
    penalty_matrix = BSplinePenalty(penalty, knots, degree)

    # Optimize the model coefficients using the penalized likelihood approach

    res = optimize(β -> penalized_log_lik(X, y, β, dist, λ, penalty_matrix), β, BFGS(), Optim.Options(show_trace=false))
    β = res.minimizer
    log_lik = -res.minimum

    # Compute the AIC and BIC

    n_params = n_features
    aic = -2 * log_lik + 2 * n_params
    bic = -2 * log_lik + n_params * log(length(y))

    # Create the model object

    model = GAMModel(β, knots, degree, n_knots, likelihood, λ, log_lik, aic, bic)

    return model
end
