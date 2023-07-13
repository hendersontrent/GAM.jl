function OptimizeGCVLambda(Basis::BSplineBasis{Vector{Float64}}, x::AbstractVector, y::AbstractVector)

    # optimization bounds     
    lower = [0]
    upper = [Inf]
    initial_lambda = [1.0]

    # Run Optimization
    inner_optimizer = GradientDescent()
    res = Optim.optimize(
        lambda -> GCV(lambda, Basis, x, y), 
        lower, upper, initial_lambda, 
        Fminbox(inner_optimizer)
    )
    return Optim.minimizer(res)[1]
end