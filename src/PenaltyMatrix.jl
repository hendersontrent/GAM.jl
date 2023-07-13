function PenaltyMatrix(Basis::BSplineBasis{Vector{Float64}}, λ::Float64, x::AbstractVector, y::AbstractVector)

    X = BasisMatrix(Basis, x) # Basis Matrix
    D = DifferenceMatrix(Basis) # D penalty matrix
    Xp = vcat(X, sqrt(λ)*D) # augment model matrix with penalty
    yp = vcat(y, repeat([0],size(D)[1])) # augment data with penalty

    return Xp, yp
end