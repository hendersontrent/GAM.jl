"""
    diffm(A, dims, differences)
General helper function to take the differences of matrices.
Usage:
```julia-repl
diffm(A, dims, differences)
```
Arguments:
- `A` : `AbstractVecOrMat` containing the matrix to be differenced.
- `dims` : `Int64` denoting the matrix dimension.
- `differences` : `Int64` denoting the number of differences to take.
"""

function diffm(A::AbstractVecOrMat, dims::Int64, differences::Int64)
    out = A
    for i in 1:differences
        out = Base.diff(out, dims=dims)
    end
    return out
end