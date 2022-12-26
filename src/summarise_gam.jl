"""
    summarise_gam(model, prob)
Generate a summary table of model information, including coefficients, their standard errors, p-values, and confidence intervals.

Usage:
```julia-repl
summarise_gam(model, prob)
```
Arguments:
- `model` : The `GAMModel`.
- `prob` : The probability for the confidence intervals.
"""
function summarise_gam(model::GAMModel, prob=0.95)

    # Extract the model coefficients and knots

    β = model.β
    knots = model.knots
    degree = model.degree
    n_knots = model.n_knots

    # Initialize the summary data frame

    df = DataFrame(Term = String[], Coefficient = Float64[], Test_Statistic = Float64[], P_Value = Float64[],
                   Std_Error = Float64[], Confidence_Interval = Tuple{Float64, Float64}[])

    # Compute the summary statistics for each term

    for i in 1:length(knots)

        # Extract the spline basis functions for the predictor variable

        x_pred = X[:, i]
        spline_basis = zeros(size(X, 1), 1)
        spline_basis = hcat(spline_basis, BSplineBasis(knots[i], degree[i], x_pred))
        spline_basis = spline_basis[:, 2:end]

        # Compute the predicted values for the predictor variable
        y_pred = spline_basis * β[2:(n_knots[i] + 1)] + β[1]

        # Compute the variance of the predicted values
        var_pred = zeros(n_samples)
        if model.likelihood === :gaussian
            for j in 1:n_samples
                var_pred[j] = 1 / model.λ * sum(spline_basis[j, :] .^ 2)
            end
        else
            for j in 1:n_samples
                var_pred[j] = y_pred[j] * (1 - y_pred[j]) / model.λ
            end
        end
        se_pred = sqrt.(var_pred)

        # Compute the test statistic, p-value, and confidence interval
        test_stat = β[1] / se_pred[1]
        p_value = 2 * (1 - cdf(TDist(n_samples - length(β)), abs(test_stat)))
        lower = β[1] - quantile(TDist(n_samples - length(β)), 1 - prob/2) * se_pred[1]
        upper = β[1] + quantile(TDist(n_samples - length(β)), 1 - ci_prob/2) * se_pred[1]

        # Add the summary statistics to the data frame

        term_name = names(X)[i]
        push!(df, (term_name, β[1], test_stat, p_value, se_pred[1], (lower, upper)))
    end
    return df
end


"""
    summarize_gam(model, prob)
Generate a summary table of model information, including coefficients, their standard errors, p-values, and confidence intervals.

Usage:
```julia-repl
summarize_gam(model, prob)
```
Arguments:
- `model` : The `GAMModel`.
- `prob` : The probability for the confidence intervals.
"""
function summarize_gam(model::GAMModel, prob=0.95)

    # Extract the model coefficients and knots

    β = model.β
    knots = model.knots
    degree = model.degree
    n_knots = model.n_knots

    # Initialize the summary data frame

    df = DataFrame(Term = String[], Coefficient = Float64[], Test_Statistic = Float64[], P_Value = Float64[],
                   Std_Error = Float64[], Confidence_Interval = Tuple{Float64, Float64}[])

    # Compute the summary statistics for each term

    for i in 1:length(knots)

        # Extract the spline basis functions for the predictor variable

        x_pred = X[:, i]
        spline_basis = zeros(size(X, 1), 1)
        spline_basis = hcat(spline_basis, BSplineBasis(knots[i], degree[i], x_pred))
        spline_basis = spline_basis[:, 2:end]

        # Compute the predicted values for the predictor variable
        y_pred = spline_basis * β[2:(n_knots[i] + 1)] + β[1]

        # Compute the variance of the predicted values
        var_pred = zeros(n_samples)
        if model.likelihood === :gaussian
            for j in 1:n_samples
                var_pred[j] = 1 / model.λ * sum(spline_basis[j, :] .^ 2)
            end
        else
            for j in 1:n_samples
                var_pred[j] = y_pred[j] * (1 - y_pred[j]) / model.λ
            end
        end
        se_pred = sqrt.(var_pred)

        # Compute the test statistic, p-value, and confidence interval
        test_stat = β[1] / se_pred[1]
        p_value = 2 * (1 - cdf(TDist(n_samples - length(β)), abs(test_stat)))
        lower = β[1] - quantile(TDist(n_samples - length(β)), 1 - prob/2) * se_pred[1]
        upper = β[1] + quantile(TDist(n_samples - length(β)), 1 - ci_prob/2) * se_pred[1]

        # Add the summary statistics to the data frame

        term_name = names(X)[i]
        push!(df, (term_name, β[1], test_stat, p_value, se_pred[1], (lower, upper)))
    end
    return df
end