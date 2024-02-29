"""
    columns_to_vector()
Converts columns of a DataFrame to an n-dimensional vector.

Usage:
```julia-repl
columns_to_vector()
```
Arguments:
- `data` : `DataFrame` to be reshaped.
"""

function columns_to_vector(data::DataFrame)
    num_columns = ncol(data)
    vectors = Vector{Vector{Float64}}(undef, num_columns)
    
    for i in 1:num_columns
        vectors[i] = convert(Vector{Float64}, data[!, i])
    end
    
    return vectors
end