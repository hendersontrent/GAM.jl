"""
    predict_gam(model, newdata)
Generates a vector of predictions for new data using the fitted GAM.

Usage:
```julia-repl
predict_gam(model, newdata)
```
Arguments:
- `model` : The `GAMModel` object.
- `newdata` : DataFrame of new input data.
"""
function predict_gam(model::GAMModel, newdata::DataFrame)
    # Extract the model coefficients and knots
    β = model.β
    knots = model.knots
    degree = model.degree
    n_knots = model.n_knots

    # Parse the formula and data
    y, smooth_terms, cat_vars, knots, degree, polynomial_degree = parse_formula(newdata, model.formula)

    # Initialize the predictor matrix
    X = []

    # Extract the predictor variables and their spline basis functions
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
            spline_basis = hcat(BSplineBasis(knots_i, degree_i, predictor, polynomial_degree_i))
            spline_basis = spline_basis[:, 2:end]
            X = hcat(X, spline_basis)
        end
    end

    # Add the categorical variables to the predictor matrix
    for term in cat_vars
        # Convert the categorical variable to dummy variables
        X = hcat(X, dummy_encoder(data[term]))
    end

    # Compute the predictions
    predictions = X * β
    return predictions
end
