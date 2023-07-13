function diffm(A::AbstractVecOrMat, dims::Int64, differences::Int64)
    out = A
    for i in 1:differences
        out = Base.diff(out, dims=dims)
    end
    return out
end