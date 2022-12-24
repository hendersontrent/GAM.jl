function plot_smooths(model::GAMModel, X::Array{Float64, 2})

    # Create spline basis functions for each predictor

    n_samples, n_features = size(X)
    spline_basis = [zeros(size(X, 1), 1) for i in 1:n_features]

    for i in 1:n_features
        spline_basis[i] = hcat(spline_basis[i], BSplineBasis(model.knots[i], model.degree, X[:, i]))
        spline_basis[i] = spline_basis[i][:, 2:end]
    end

    # Concatenate the spline basis functions with the intercept term

    X_spline = hcat(spline_basis..., X)

    # Compute the fitted values and standard errors for each predictor

    y_hat = X_spline * model.beta
    se = sqrt.((X_spline .* (model.alpha .* model.beta))' * (X_spline .* (model.alpha .* model.beta)))

    # Create a plot for each predictor
    
    p = []
    for i in 1:n_features
        p = push!(p, plot(X[:, i], y_hat[:, i], ribbon=2*se[i, i]))
    end
    return p
end
