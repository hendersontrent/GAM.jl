struct GAMModel
    beta::Vector{Float64}
    knots::Vector{Vector{Float64}}
    degree::Int
    n_knots::Vector{Int}
    likelihood::Function
    alpha::Float64
    lam::Float64
    log_lik::Float64
    aic::Float64
    bic::Float64
end

function fitGAM(X::Array{Float64, 2}, y::Array{Float64, 1}, knots::Union{Nothing, Vector{Vector{Float64}}}, degree::Int, n_knots::Union{Nothing, Vector{Int}}, likelihood::Union{Symbol, Function}, alpha::Union{Nothing, Float64}, lam::Float64, n_folds::Int)
    # Add a column of ones to X for the intercept term
    X = hcat(ones(size(X, 1)), X)
    # Create spline basis functions for each predictor
    n_samples, n_features = size(X)
    spline_basis = [zeros(size(X, 1), 1) for i in 1:n_features]
    if knots === nothing
        knots = Vector{Vector{Float64}}(undef, n_features)
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
    # Set the convergence tolerance and maximum number of iterations
    tol = 1e-4
    max_iter = 1000
    # Set the distribution based on the likelihood type
    if likelihood === :gaussian
        dist = Normal()
    elseif likelihood === :binomial
        dist = Binomial()
    elseif likelihood === :poisson
        dist = Poisson()
    else
        dist = likelihood
    end
    # If alpha is not specified, optimize it using cross-validation
    if alpha === nothing
        alpha_vals = logspace(-3, 3, 7)
        cv_scores = Vector{Float64}(undef, length(alpha_vals))
        for (i, alpha_val) in enumerate(alpha_vals)
            # Compute the cross-validated log likelihood for the current alpha value
            fold
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
        model = GAMModel(beta, knots, degree, n_knots, likelihood, alpha, lam, log_lik, aic, bic)
        return model
    end
end
# Compute the penalized log likelihood
function penalized_log_lik(X, y, beta, dist, alpha, penalty_matrix)
    y_pred = X * beta
    y_pred = dist.linkinv.(y_pred)
    log_lik = sum(logpdf.(dist, y, y_pred))
    penalty = beta' * penalty_matrix * beta
    return -(log_lik - alpha * penalty)
end
# Compute the cross-validated log likelihood
function cv_log_lik(X, y, beta, dist, penalty_matrix, train_idx, test_idx)
    # Extract the training and test data
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]
    # Compute the predicted values on the training data
    y_pred_train = X_train * beta
    y_pred_train = dist.linkinv.(y_pred_train)
    # Compute the log likelihood on the training data
    log_lik_train = sum(logpdf.(dist, y_train, y_pred_train))
    # Compute the penalty on the training data
    penalty_train = beta' * penalty_matrix * beta
    # Compute the predicted values on the test data
    y_pred_test = X_test * beta
    y_pred_test = dist.linkinv.(y_pred_test)
    # Compute the log likelihood on the test data
    log_lik_test = sum(logpdf.(dist, y_test, y_pred_test))
    # Return the negative of the sum of the training and test log likelihoods
    return -(log_lik_train + log_lik_test - alpha * penalty_train)
end
# Compute optimal knots for B-splines
function optimal_knots(x::Array{Float64, 1}, degree::Int, n_knots::Int)
    knots = range(minimum(x), stop=maximum(x), length=n_knots)
    knots = knots[2:end-1]
    knots = sort(unique(knots))
    knots = vcat(knots[1] .- (knots[2] - knots[1]), knots, knots[end] .+ (knots[end] - knots[end-1]))
    return knots, length(knots) - degree - 1
end