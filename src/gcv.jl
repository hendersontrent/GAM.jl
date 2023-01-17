function gcv(x::Vector{Float64}, y::Vector{Float64}, degree::Int, polynomial_degree::Int)
    # Sort the predictor and response
    idx = sortperm(x)
    x = x[idx]
    y = y[idx]

    # Compute the number of samples
    n = length(x)

    # Initialize the knots to the minimum and maximum values of the predictor
    knots = [minimum(x), maximum(x)]

    # Initialize the GCV score
    gcv_score = Inf

    # Initialize the smoothness parameter
    smoothness_parameter = 0

    # Iterate over the smoothness parameters
    for i in 1:n
        # Compute the thin plate regression spline basis functions
        tprs_basis = tprs_basis_matrix(x, knots, degree, polynomial_degree)

        # Compute the model coefficients
        β = (tprs_basis'tprs_basis) \ tprs_basis'y

        # Compute the residuals
        residuals = y - tprs_basis * β

        # Compute the sum of squared residuals
        ssr = sum(residuals .^ 2)

        # Compute the smoothness penalty
        smoothness_penalty = tprs_smoothness_penalty(tprs_basis, i)

        # Compute the GCV score
        gcv_score_new = (ssr / n) / (1 - (2 * smoothness_penalty / n) / (n - smoothness_penalty))

        # Check if the GCV score has improved
        if gcv_score_new < gcv_score
            # Update the GCV score and smoothness parameter
            gcv_score = gcv_score_new
            smoothness_parameter = i
        end
    end

    # Return the knots and GCV score
    return knots, gcv_score
end