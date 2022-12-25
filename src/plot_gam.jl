"""
    plot_gam(model, X, y, x_var, prob)
Generate a plot of the smooth for one of the predictor variables in the GAM with a confidence interval.

Usage:
```julia-repl
plot_gam(model, X, y, x_var, prob)
```
Arguments:
- `model` : The `GAMModel`.
- `X` : Data matrix of predictor variables.
- `y` : Response variable vector.
- `x_var` : Index of the variable of interest in the data matrix `X`.
- `prob` : The probability for the confidence intervals.
"""
function plot_gam(model::GAMModel, X::Array{Float64, 2}, y::Array{Float64, 1}, x_var::Union{Int, String}, prob::Float64=0.95)

    # Extract the model coefficients and knots

    β = model.β
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

    y_pred = spline_basis * β[2:(n_knots[x_idx] + 1)] + β[1]

    # Compute confidence intervals

    n_samples, n_features = size(X)
    var_pred = zeros(n_samples)

    if model.likelihood === :gaussian
        for i in 1:n_samples
            var_pred[i] = 1 / model.λ * sum(spline_basis[i, :] .^ 2)
        end
        se_pred = sqrt.(var_pred)
        ci = inv(Normal(), prob)
        lower = y_pred .- ci * se_pred
        upper = y_pred .+ ci * se_pred
    else
        for i in 1:n_samples
            var_pred[i] = y_pred[i] * (1 - y_pred[i]) / model.λ
        end
        se_pred = sqrt.(var_pred)
        ci = inv(Normal(), prob)
        lower = y_pred .- ci * se_pred
        upper = y_pred .+ ci * se_pred
    end

    # Draw plot

    p = scatter(x_pred, y, alpha=0.5)
    ribbon(x_pred, lower, upper, alpha=0.2)
    plot!(x_pred, y_pred, linewidth=2)
    return p
end