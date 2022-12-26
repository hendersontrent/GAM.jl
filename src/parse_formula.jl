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
function parse_formula(data::DataFrame, formula::Union{Symbol, Expr})
    if typeof(formula) == Symbol
        return formula, [:Intercept], nothing, nothing, nothing
    end

    y = formula.lhs
    terms = formula.rhs

    # Initialize empty arrays for the smooth terms and their knots, degrees, and polynomial degrees
    smooth_terms = []
    knots = []
    degree = []
    n_knots = []
    polynomial_degree = []

    # Iterate over the terms in the formula
    for term in terms
        if term.head == :call
            # Check if the term is a smooth term
            if term.args[1] == :s
                # Extract the predictor variable and knots
                predictor = term.args[2]
                knots_arg = term.args[3]
                # Set the default values for degree and polynomial degree
                degree_val = 3
                polynomial_degree_val = 1
                # Check if the degree and polynomial degree are specified
                for arg in term.args[4:end]
                    if arg.head == :kw
                        if arg.args[1] == :degree
                            degree_val = arg.args[2]
                        elseif arg.args[1] == :k
                            polynomial_degree_val = arg.args[2]
                        end
                    end
                end
                # Append the predictor, knots, degree, and polynomial degree to the arrays
                push!(smooth_terms, predictor)
                push!(knots, knots_arg)
                push!(degree, degree_val)
                push!(polynomial_degree, polynomial_degree_val)
            else
                # Append the term to the smooth terms array
                push!(smooth_terms, term)
            end
        end
    end

    # Extract the categorical variables from the smooth terms
    cat_vars = []
    for term in smooth_terms
        if !ismissing(data, term) && eltype(data[term]) <: Union{CategoricalValue, Missing}
            push!(cat_vars, term)
        end
    end

    # Return the response variable, smooth terms, categorical variables, knots, degrees, and polynomial degrees
    return y, smooth_terms, cat_vars, knots, degree, polynomial_degree
end
    
