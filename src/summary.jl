function summary(model::GAMModel)
    # Extract the model coefficients and knots
    beta = model.beta
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
        spline_basis = hcat(spline_basis, BSplineBasis(knots[i], degree, x_pred))
        spline_basis = spline_basis[:, 2:end]
        # Compute the predicted values for the predictor variable
        y_pred = spline_basis * beta[2:(n_knots[i] + 1)] + beta[1]
        # Compute the variance of the predicted values
        var_pred = zeros(n_samples)
        if model.likelihood === :gaussian
            for j in 1:n_samples
                var_pred[j] = 1 / model.alpha * sum(spline_basis[j, :] .^ 2)
            end
        else
            for j in 1:n_samples
                var_pred[j] = y_pred[j] * (1 - y_pred[j]) / model.alpha
            end
        end
        se_pred = sqrt.(var_pred)
        # Compute the test statistic, p-value, and confidence interval
        test_stat = beta[1] / se_pred[1]
        p_value = 2 * (1 - cdf(TDist(n_samples - length(beta)), abs(test_stat)))
        ci_lower = beta[1] - 1.96 * se_pred[1]
        ci_upper = beta[1] + 1.96 * se_pred[1]
        # Add the summary statistics to the data frame
        term_name = names(X)[i]
        push!(df, (term_name, beta[1], test_stat, p_value, se_pred[1], (ci_lower, ci_upper)))
    end
    return df
end