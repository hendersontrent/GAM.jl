"""
    optimal_knots(x, degree, n_knots)
Computes the optimal number of knots for a given spline.

Usage:
```julia-repl
optimal_knots(x, degree, n_knots)
```
Arguments:
- `x` : Vector of values for a predictor variable.
- `degree` : Polynomial degree of the spline.
- `n_knots` : Number of knots in the spline.
"""
function optimal_knots(x::Array{Float64, 1}, degree::Int, n_knots::Int)

    # Check if the input array is empty or has only one unique element

    if isempty(x) || length(unique(x)) == 1
        # Return default knots
        knots = [minimum(x), maximum(x)]
    else
        if n_knots < 2
            n_knots = 2
        end

        knots = range(minimum(x), stop=maximum(x), length=n_knots)
        knots = knots[2:end-1]
        knots = sort(unique(knots))
        knots = vcat(knots[1] .- (knots[2] - knots[1]), knots, knots[end] .+ (knots[end] - knots[end-1]))

        if length(knots) < 2
            error("Error: Number of knots must be at least 2.")
        end
    end
    return knots, length(knots) - degree - 1
end
