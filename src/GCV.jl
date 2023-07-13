function GCV(param::AbstractVector, Basis::BSplineBasis{Vector{Float64}}, x::AbstractVector, y::AbstractVector)
    n = length(Basis.breakpoints)
    Xp, yp = PenaltyMatrix(Basis, param[1], x, y)
    β = coef(lm(Xp,yp))
    H = Xp*inv(Xp'Xp)Xp' # hat matrix
    trF = sum(diag(H)[1:n])
    y_hat = Xp*β
    rss = sum((yp-y_hat)[1:n].^2) ## residual SS
    gcv = n*rss/(n-trF)^2
    return gcv
end