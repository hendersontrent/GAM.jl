"""
    plot_gam(model, x_var, prob)
Generate a plot of the smooth for one of the predictor variables in the GAM with a confidence interval.

Usage:
```julia-repl
plot_gam(model, x_var, prob)
```
Arguments:
- `model` : The `GAMModel`.
- `x_var` : Index of the variable of interest in the data matrix `X`.
- `prob` : The probability for the confidence intervals.
"""
function plot_gam(model::GAMModel, x_var::String, prob::Float64=0.95)

    # Extract the model coefficients, knots, degrees, and polynomial degrees

    β = model.β
    knots = model.knots
    degree = model.degree
    n_knots = model.n_knots
    polynomial_degree = model.polynomial_degree

    # Get the indices of the predictor variables

    x_idx = findfirst(names(model.df), x_var)

    # Create the spline basis functions for the predictor variable

    x_pred = model.df[:, x_idx]
    spline_basis = zeros(size(model.df, 1), 1)
    spline_basis = hcat(spline_basis, BSplineBasis(knots[x_idx], degree[x_idx], x_pred, polynomial_degree[x_idx]))
    spline_basis = spline_basis[:, 2:end]
    # Compute the predicted values for the predictor variable

    y_pred = spline_basis * β[2:(n_knots[x_idx] + 1)] + β[1]

    # Compute confidence intervals

    n_samples, n_features = size(model.df)
    var_pred = zeros(n_samples)

    if model.family === :gaussian
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