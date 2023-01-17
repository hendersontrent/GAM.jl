"""
    parse_formula(formula, data)
Extract components of the formula for use in constructing the model.

Usage:
```julia-repl
parse_formula(formula, data)
```
Arguments:
- `formula` : The model formula.
- `data` : DataFrame of input data.
"""
function parse_formula(formula::String, data::DataFrame)
    # Split the formula string into the left-hand side (response variable) and right-hand side (predictor formula)
    lhs, rhs = split(formula, "~")
    lhs = Symbol(rstrip(lhs)) # Remove trailing whitespace
    rhs = lstrip(rhs) # Remove leading whitespace
    rhs = filter(e->e∉"+",rhs) # Remove plus signs to only keep terms
    rhs = filter(!=(""), split(rhs, " ")) # Remove empty strings

    # Initialize the lists of smooth terms, categorical variables, knots, and degrees
    smooth_terms = Vector{Symbol}(undef, 0)
    cat_vars = Vector{Symbol}(undef, 0)
    knots = Vector{Union{Nothing, AbstractArray{T, 1}} where T<:Real}(undef, 0)
    degree = Vector{Int}(undef, 0)
    polynomial_degree = Vector{Int}(undef, 0)

    # Iterate over the terms in the predictor formula
    for term in rhs
        # Check if the term is a smooth term
        if starts(term, "s(")
            # Extract the variable name and optional arguments
            var_name = Symbol(split(split(term, ",")[1], "(")[2])
            parser = parse_smooth_term(term)
            k = parser[1]
            degree_i = parser[2]
            knots_i = nothing
            polynomial_degree_i = 0

            # Check if knots were specified

            if knots_i === nothing
                # Estimate the knots using automatic smoothness estimation
                knots_i = tprs_knots(data, lhs, var_name, k, degree_i, polynomial_degree_i)
            end

            # Add the smooth term to the list of smooth terms
            push!(smooth_terms, var_name)
            push!(knots, knots_i)
            push!(degree, degree_i)
            push!(polynomial_degree, polynomial_degree_i)
        else
            # Add the term to the list of categorical variables
            term = Symbol(strip(term))
            push!(cat_vars, term)
        end
    end
    return lhs, smooth_terms, cat_vars, knots, degree, polynomial_degree
end
