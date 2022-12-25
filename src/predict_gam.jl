"""
    predict_gam(model, X, type)
Generates a vector of predictions for new data using the fitted GAM.

Usage:
```julia-repl
predict_gam(model, X, type)
```
Arguments:
- `model` : The `GAMModel` object.
- `X` : Matrix of new input data.
- `type` : The type of prediction to make.
"""
function predict_gam(model::GAMModel, X::Array{Float64, 2}, type::Symbol)

    # Extract the model coefficients and knots

    β = model.β
    knots = model.knots
    degree = model.degree
    n_knots = model.n_knots
    dist = model.likelihood

    # Create the spline basis functions for each predictor variable

    spline_basis = zeros(size(X, 1), 1)
    n_features = size(X, 2)
    for i in 1:n_features
        x_pred = X[:, i]
        spline_basis = hcat(spline_basis, BSplineBasis(knots[i], degree, x_pred))
    end

    spline_basis = spline_basis[:, 2:end]

    # Compute the predicted values for the predictor variables

    y_pred = spline_basis * β[2:end] + β[1]

    # Compute the mean or probability predictions

    if type === :mean
        return dist.linkinv.(y_pred)
    elseif type === :prob
        return dist.link.(y_pred)
    else
        error("Invalid prediction type.")
    end
end