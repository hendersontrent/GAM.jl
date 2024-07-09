using Distributions, RDatasets, GLM, Optim, BSplines, LinearAlgebra
using DataFrames, CSV, Plots, Optim

identity(x) = x
logit(p) = log(p/(1-p))
expit(x) = 1/(1+exp(-x))

# df = CSV.read("dev/engine.csv", DataFrame)
# y = df.wear; x = [df.size]
# BasisArgs = [(15,4)]

df = dataset("datasets", "trees");
x = [df.Girth, df.Height]
y = df.Volume
sp = [2,2]
BasisArgs = [(10, 2), (10,2)]

Links = Dict(
    :Identity => Dict(
        :Name => "Identity",
        :Function => identity,
        :Inverse => identity,
        :Derivative => (x -> 1),
        :Second_Derivative => (x -> 0)
    ),
    :Log => Dict(
        :Name => "Log",
        :Function => log,
        :Inverse => exp,
        :Derivative => (x -> 1/x),
        :Second_Derivative => (x -> -1/(x^2))
    )
)

Dists = Dict(
    :Normal => Dict(
        :Name => "Normal",
        :Distribution => Normal,
        :V => (mu -> 1),
        :V_Derivative => (mu -> 0),
        :Link => Links[:Identity]
    ),
    :Gamma => Dict(
        :Name => "Gamma",
        :Distribution => Gamma,
        :V => (mu -> mu^2),
        :V_Derivative => (mu -> 2*mu),
        :Link => Links[:Log]
    ),
    :Poisson => Dict(
        :Name => "Poisson",
        :Distribution => Poisson,
        :V => (mu -> mu),
        :V_Derivative => (mu -> 1),
        :Link => Links[:Log]
    )
)

mutable struct GAMData
    y::AbstractArray
    x::AbstractArray
    Basis::AbstractArray{BSplineBasis}
    Dist::Dict
    Link::Dict
    Coef::AbstractArray
    ColMeans::AbstractArray
    CoefIndex::AbstractArray
    Fitted::AbstractArray
    Diagnostics::Dict

    function GAMData(
        y::AbstractArray,
        x::AbstractArray,
        Basis::AbstractArray,
        Dist::Dict,
        Link::Dict,
        Coef::AbstractArray,
        ColMeans::AbstractArray,
        CoefIndex::AbstractArray,
        Fitted::AbstractArray,
        Diagnostics::Dict
    )
        new(y, x, Basis, Dist, Link, Coef, ColMeans, CoefIndex, Fitted, Diagnostics)
    end
end

function BuildUniformBasis(x::Vector, n_knots::Int64, order::Int64)
    KnotsList = range(minimum(x), maximum(x), length = n_knots);
    Basis = BSplineBasis(order, KnotsList);
    return Basis
end

function BuildQuantileBasis(x::Vector, n_knots::Int64, order::Int64)
    KnotsList = quantile(x, range(0, 1; length=n_knots));
    Basis = BSplineBasis(order, KnotsList);
    return Basis
end

Basis = map((xi, argi) -> BuildUniformBasis(xi, argi[1], argi[2]), x, BasisArgs)

plot(Basis[1])

function BuildBasisMatrix(Basis::BSplineBasis, x::AbstractVector)
    splines = vec(
        mapslices(
            x -> Spline(Basis,x), 
            diagm(ones(length(Basis))),
            dims=1
        )
    );
    X = hcat([s.(x) for s in splines]...)
    return X 
end

# General Function to take differences of Matrices
function diffm(A::AbstractVecOrMat, dims::Int64, differences::Int64)
    out = A
    for i in 1:differences
        out = Base.diff(out, dims=dims)
    end
    return out
end

# Difference Matrix
# Note we take the difference twice.
function BuildDifferenceMatrix(Basis::BSplineBasis)
    #nk = length(Basis.breakpoints)
    D = diffm(
        diagm(0 => ones(length(Basis))),
        1, # matrix dimension
        2  # number of differences
    )
    return D
end

function dcat(matrices)

    # Initialize the final matrix with zeros
    n_rows = sum(size.(matrices, 1))
    n_cols = sum(size.(matrices, 2))
    final_matrix = zeros(n_rows, n_cols)

    # Row start index for each matrix
    row_start = 1
    col_start = 1

    for (i, mat) in enumerate(matrices)
        # Get the current matrix dimensions
        rows, cols = size(mat)

        # Calculate the column end index for the current block
        col_end = col_start + cols - 1

        # Place the matrix in the corresponding block
        final_matrix[row_start:(row_start + rows - 1), col_start:col_end] = mat

        # Update the row and column start indices for the next matrix
        row_start += rows
        col_start = col_end + 1
    end

    return final_matrix
end

function BuildBasisMatrixColMeans(BasisMatrix::AbstractArray)
    # Calculate means of each column
    return mean(BasisMatrix, dims=1)
end

function CenterBasisMatrix(BasisMatrix::AbstractMatrix, BasisMatrixColMeans::AbstractArray)
    return BasisMatrix .- BasisMatrixColMeans
    #return map((c, m) -> (c .- m), gamDataBasisMatrix, gamDataBasisMatrixColMeans)
end

function DropCol(X::AbstractMatrix, ix::Int)
    cols = [1:size(X,2);]
    deleteat!(cols, ix)
    return X[:, cols]
end

function BuildCoefIndex(BasisMatrix::AbstractArray)
    ix = [1:1]
    ix_end = cumsum(vcat([1], size.(BasisMatrix,2)))
    append!(ix, [ix_end[i-1]+1:ix_end[i] for i in 2:length(ix_end)])
    return ix[2:end]
end

function BuildPenaltyMatrix(y, x, sp, Basis)

    n = length(y)
    n_knots = map(x -> length(x.breakpoints), Basis)
    X = map(BuildBasisMatrix, Basis, x)
    D = map(BuildDifferenceMatrix, Basis)
    # Drop one column from X and D for identifiability
    X = map(DropCol, X, n_knots)
    D = map(DropCol, D, n_knots)

    # Center
    ColMeans = map(BuildBasisMatrixColMeans, X)
    X = map(CenterBasisMatrix, X, ColMeans)

    # Store Coef index
    CoefIndex = BuildCoefIndex(X)

    # Build Design Matrix
    X = hcat(repeat([1],n), hcat(X...)) # add intercept

    # Build Penalty Matrix
    D = dcat(map((p, d) -> (sqrt(p) * d), sp, D))
    D = hcat(repeat([0], size(D,1)), D) # add 0 for intercept

    return X, y, D, ColMeans, CoefIndex
end

function HatMatrix(X, D, W)
    return W * (X * inv(X' * W * X + D' * D) ) * X'
end

function ModelDiagnostics(y, X, D, W, B)
    n = length(y)
    H = HatMatrix(X, D, W) # hat matrix
    trF = sum(diag(H)[1:n]) # EDF
    rsd = y - (X * B) # residuals
    rss = sum(rsd.^2) # residual SS
    sig_hat = rss/(n-trF) # residual variance
    gcv = sig_hat*n/(n-trF) # GCV score

    return Dict(
        :RSS => rss,
        :EDF => trF,
        :GCV => gcv
    )
end

function FitOLS(y, x, sp, Basis)

    n = length(y)
    X, Y, D, ColMeans, CoefIndex = BuildPenaltyMatrix(y, x, sp, Basis)

    Xp = vcat(X, D)
    Yp = vcat(y, repeat([0], size(D,1)))
    B = Xp \ Yp
    fitted = (Xp * B)[1:n]
    diagnostics = ModelDiagnostics(y, X, D, Diagonal(ones(n)), B)

    return GAMData(
        y,
        x,
        Basis,
        Dists[:Normal],
        Links[:Identity],
        B,
        ColMeans,
        CoefIndex,
        fitted,
        diagnostics
    )
end

FitOLS(y, x, [2, 2], Basis).Diagnostics[:GCV]

function FitWPS(y, x, sp, Basis, w = ones(length(y)))

    X, Y, D, ColMeans, CoefIndex = BuildPenaltyMatrix(y, x, sp, Basis)
    W = Diagonal(w)
    # Left-hand side (LHS) matrix
    LHS = X' * W * X + D' * D
    # Right-hand side (RHS) vector
    RHS = X' * W * Y
    # Solve for beta
    B = LHS \ RHS
    # Fitted values
    fitted = (X * B)
    # Run Diagnostics
    diagnostics = ModelDiagnostics(y, X, D, W, B)

    return GAMData(
        y,
        x,
        Basis,
        Dists[:Normal],
        Links[:Identity],
        B,
        ColMeans,
        CoefIndex,
        fitted,
        diagnostics
    )
end

# Coef the same
plot(
    FitOLS(y, x, [2,2], Basis).Coef,
    FitWPS(y, x, [2, 2], Basis).Coef
)

# GCV the same
FitOLS(y, x, [2,2], Basis).Diagnostics[:GCV]
FitWPS(y, x, [2, 2], Basis).Diagnostics[:GCV] # no weights should return same as OLS


# # Deprecated -> use PIRLS now
# function GAMFit(y, x, sp, Basis, Dist, Link)
#     eta = Link[:Function].(y)
#     old_gcv = -100
#     for i in 1:25
#         mu = Link[:Inverse].(eta)
#         z = (y .- mu) ./ mu .+ eta
#         global mod = FitOLS(z, x, sp, Basis)

#         if abs(mod.Diagnostics[:GCV] - old_gcv) < 1e-5*mod.Diagnostics[:GCV]
#             break
#         end
#         old_gcv = mod.Diagnostics[:GCV]
#         eta = mod.Fitted
#     end
#     mod.Dist = Dist
#     mod.Fitted = Link[:Inverse].(mod.Fitted)
#     return mod
# end


function alpha(y, mu, Dist, Link)
    "see page 250 in Wood, 2nd ed"
    return @. 1 + (y - mu) * (
        Dist[:V_Derivative](mu) / Dist[:V](mu) + 
        Link[:Second_Derivative](mu) / Link[:Derivative](mu)
    )
end

function PIRLS(y, x, sp, Basis, Dist, Link, maxIter = 25, tol = 1e-6)
    
    # Initial Predictions
    n = length(y)
    mu = y
    eta = Link[:Function].(mu)
    
    # Deviance
    logLik = sum(map(x -> logpdf(Dist[:Distribution](x), x), y))
    dev = logLik
    for i in 1:maxIter
        # Compute weights
        a = alpha(y, mu, Dist, Link)
        z = @. Link[:Derivative](mu) * (y - mu) / a + eta
        w = @. a / (Link[:Derivative](mu)^2 * Dist[:V](mu))

        global mod = FitWPS(z, x, sp, Basis, w)
        eta = mod.Fitted
        mu = Link[:Inverse].(eta)
        oldDev = dev
        dev = 2 * (logLik - sum(map((x,y) -> logpdf(Dist[:Distribution](x), y), mu, y)))
        if abs(dev - oldDev) < 1e-6 * dev
            break
        end
    end
    mod.Dist = Dist
    mod.Fitted = Link[:Inverse].(mod.Fitted)
    return mod
end


# 0.01014639 from R code;
sp = [1,1]
mod = PIRLS(y, x, sp, Basis, Dists[:Gamma], Links[:Log])
mod.Diagnostics[:GCV] # Diff fitting method -> close enough for comfort


rho = range(-20, 11)
out = []
for i in eachindex(rho)
    mod = PIRLS(y, x, repeat([exp(rho[i])],length(x)), Basis, Dists[:Gamma], Links[:Log])
    push!(out, mod.Diagnostics[:GCV])
end
plot(rho, out)


# looking for  8.528943 246888083.011660
function OptimPIRLS(y, x, Basis, Dist, Link)
    # Find Optimal Smoothing Params
    res = optimize(
        sp -> PIRLS(y, x, exp.(sp), Basis, Dist, Link).Diagnostics[:GCV], 
        zeros(length(x)), 
        NelderMead()
    )
    sp = exp.(Optim.minimizer(res))
    # Fit Optimal Model
    return PIRLS(y, x, sp, Basis, Dist, Link)
end

mod = OptimPIRLS(y, x, Basis, Dists[:Gamma], Links[:Log])

function BuildPredictionMatrix(x::AbstractArray, Basis::BSplineBasis, ColMeans::AbstractArray)
    basisMatrix = DropCol(BuildBasisMatrix(Basis, x), length(Basis.breakpoints))
    return CenterBasisMatrix(basisMatrix, ColMeans)
end

test = BuildPredictionMatrix(x[1], Basis[1], mod.ColMeans[1])

function PredictPartial(mod, ix)
    predMatrix = BuildPredictionMatrix(mod.x[ix], mod.Basis[ix], mod.ColMeans[ix])
    predBeta = mod.Coef[mod.CoefIndex[ix]]
    return predMatrix * predBeta
end

PredictPartial(mod, 1)


function PartialDependencePlot(mod, ix)
    x = mod.x[ix]
    pred = PredictPartial(mod, ix)
    ord = sortperm(x)
    return plot(x[ord], pred[ord])
end

PartialDependencePlot(mod, 2)

function plotGAM(mod)
    partialPlot = map(x -> PartialDependencePlot(mod, x), eachindex(mod.x))
    plot(partialPlot..., layout=(1, length(partialPlot)), link = :y)
end

plotGAM(mod)