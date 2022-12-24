function plot_gam(model::GAMModel, X::Array{Float64, 2}, y::Array{Float64, 1}, x_var::Union{Int, String})

    # Extract the model coefficients and knots

    beta = model.beta
    knots = model.knots
    degree = model.degree
    n_knots = model.n_knots

    # Get the indices of the predictor variables

    if x_var isa Int
        x_idx = x_var
    else
        x_idx = findfirst(names(X), x_var)
    end
    
    # Create the spline basis functions for the predictor variable

    x_pred = X[:, x_idx]
    spline_basis = zeros(size(X, 1), 1)
    spline_basis = hcat(spline_basis, BSplineBasis(knots[x_idx], degree, x_pred))
    spline_basis = spline_basis[:, 2:end]
    # Compute the predicted values for the predictor variable

    y_pred = spline_basis * beta[2:(n_knots[x_idx] + 1)] + beta[1]

    # Compute confidence intervals

    n_samples, n_features = size(X)
    var_pred = zeros(n_samples)

    if model.likelihood === :gaussian
        for i in 1:n_samples
            var_pred[i] = 1 / model.alpha * sum(spline_basis[i, :] .^ 2)
        end
        se_pred = sqrt.(var_pred)
        ci_lower = y_pred .- 1.96 * se_pred
        ci_upper = y_pred .+ 1.96 * se_pred
    else
        for i in 1:n_samples
            var_pred[i] = y_pred[i] * (1 - y_pred[i]) / model.alpha
        end
        se_pred = sqrt.(var_pred)
        ci_lower = y_pred .- 1.96 * se_pred
        ci_upper = y_pred .+ 1.96 * se_pred
    end

    # Draw plot

    p = scatter(x_pred, y, yerror=[ci_lower, ci_upper], alpha=0.5)
    plot!(x_pred, y_pred, linewidth=2)
    return p
end