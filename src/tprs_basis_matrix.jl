function tprs_basis_matrix(x::Vector{Float64}, knots::Vector{Float64}, degree::Int, polynomial_degree::Int)
    # Create the thin plate regression spline basis
    tprs_basis = Array{Float64}(undef, length(x), 1 + length(knots) + polynomial_degree)
    tprs_basis[:, 1] .= 1.0

    # Iterate over the knots
    for (i, knot) in enumerate(knots)
        # Compute the thin plate regression spline basis function
        tprs_basis[:, i + 1] = thin_plate_spline(x, knot, degree)
    end

    # Add the polynomial terms
    tprs_basis[:, end - polynomial_degree + 1:end] = x .^ (1:polynomial_degree)

    return tprs_basis
end
