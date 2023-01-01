function tprs_knots(data::DataFrame, y::Symbol, x::Symbol, k::Int, degree::Int, polynomial_degree::Int)
    # Extract the predictor and response
    predictor = data[x]
    response = data[y]

    # Compute the knots using automatic smoothness estimation
    knots, _ = gcv(predictor, response, k, degree, polynomial_degree)

    return knots
end
