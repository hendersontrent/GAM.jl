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
function predict_gam(model::GAMModel, newdata::DataFrame)
    # Extract the model coefficients and knots
    β = model.β
    knots = model.knots
    degree = model.degree
    n_knots = model.n_knots

    # Parse the formula and data
    y, smooth_terms, cat_vars, knots, degree, polynomial_degree = parse_formula(newdata, model.formula)

    # Extract the predictor variables and their spline basis functions
    X = Array{Float64}(undef, size(newdata, 1), 0)
    spline_basis = Vector{Matrix{Float64}}(undef, length(smooth_terms))

    # Iterate over the smooth terms
    for (i, term) in enumerate(smooth_terms)
        # Check if the term is the intercept
        if term == :Intercept
            X = hcat(X, ones(size(newdata, 1)))
        else
            # Extract the predictor and its spline basis functions
            predictor = newdata[term]
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
        X_new = hcat(X_new, dummy_encoder(data[term]))
    end

    # Compute the predictions
    predictions = X_new * β
    return predictions
end