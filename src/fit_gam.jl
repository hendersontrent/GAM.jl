struct GAMModel
    beta::Vector{Float64}
    knots::Vector{Vector{Float64}}
    degree::Int
    n_knots::Vector{Int}
    likelihood::Function
    alpha::Float64
    log_lik::Float64
    aic::Float64
    bic::Float64
end

function fit_gam(X::Array{Float64, 2}, y::Array{Float64, 1}, likelihood::Union{Symbol, Function}, knots::Union{Nothing, Vector{Vector{Float64}}}=nothing, degree::Int=3, n_knots::Union{Nothing, Vector{Int}}=nothing, alpha::Union{Nothing, Float64}=nothing, n_folds::Int=5)

    # Add a column of ones to X for the intercept term

    X = hcat(ones(size(X, 1)), X)

    # Create spline basis functions for each predictor

    n_samples, n_features = size(X)
    spline_basis = [zeros(size(X, 1), 1) for i in 1:n_features]

    if knots === nothing
        knots = Vector{Float64}(undef, n_features)
        n_knots = Vector{Int}(undef, n_features)
        for i in 1:n_features
            knots[i], _ = optimal_knots(X[:, i], degree, min(n_samples, 50))
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
    beta = zeros(n_features)

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

# If alpha is not specified, optimize it using cross-validation

    if alpha === nothing
        alpha_vals = logspace(-3, 3, 7)
        cv_scores = Vector{Float64}(undef, length(alpha_vals))

        for (i, alpha_val) in enumerate(alpha_vals)
            # Compute the cross-validated log likelihood for the alpha
            fold_log_lik = Vector{Float64}(undef, n_folds)

            for j in 1:n_folds

                # Split the data into training and test sets

                idx = foldind(y, n_folds, j)
                X_train = X[.!idx, :]
                y_train = y[.!idx]
                X_test = X[idx, :]
                y_test = y[idx]

                # Compute the penalized log likelihood for the training data

                res = optimize(beta -> penalized_log_lik(X_train, y_train, beta, dist, alpha_val, penalty_matrix), beta, BFGS(), Optim.Options(show_trace=false))
                beta = res.minimizer
                fold_log_lik[j] = log_lik(X_test, y_test, beta, dist)
            end

            # Compute the average cross-validated log likelihood
            cv_scores[i] = mean(fold_log_lik)
        end

        # Select the alpha value with the lowest cross-validation score

        alpha_idx = argmin(cv_scores)
        alpha = alpha_vals[alpha_idx]

        # Re-initialize the model coefficients to zeros

        beta = zeros(n_features)
    end

    # Compute the penalty matrix for the spline basis functions

    penalty = zeros(size(X, 2))
    penalty[2:end] = 1
    penalty_matrix = BSplinePenalty(penalty, knots, degree)

    # Optimize the model coefficients using the penalized likelihood approach

    res = optimize(beta -> penalized_log_lik(X, y, beta, dist, alpha, penalty_matrix), beta, BFGS(), Optim.Options(show_trace=false))
    beta = res.minimizer
    log_lik = -res.minimum

    # Compute the AIC and BIC

    n_params = n_features
    aic = -2 * log_lik + 2 * n_params
    bic = -2 * log_lik + n_params * log(length(y))

    # Create the model object

    model = GAMModel(beta, knots, degree, n_knots, likelihood, alpha, log_lik, aic, bic)

    return model
end
