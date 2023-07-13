function DifferenceMatrix(Basis::BSplineBasis{Vector{Float64}})
    D = diffm(
        diagm(0 => ones(length(Basis))),
        1, # matrix dimension
        2  # number of differences
    )
    return D
end