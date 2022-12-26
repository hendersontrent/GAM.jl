"""
    dummy_encoder(x)
Takes a vector x of `CategoricalValue` or `Missing `elements, and returns a matrix X of dummy variables for the categories in x. The matrix X has one column for each category in x, except for the first category which is left out to avoid multicollinearity. Each row in X has a 1 in the column corresponding to the category of the element in x, and 0s in all other columns. If an element in x is Missing, the corresponding row in X has all 0s.

Arguments:
- `x` : The vector of values to encode.
"""
function dummy_encoder(x::Vector{Union{CategoricalValue, Missing}})
    categories = levels(x)
    n_samples = length(x)
    n_categories = length(categories)
    X = zeros(n_samples, n_categories - 1)
    for i in 1:n_samples
        if !ismissing(x[i])
            X[i, categories .== x[i]] = 1.0
        end
    end
    return X
end
