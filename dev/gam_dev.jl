using Distributions, RDatasets, GLM, Optim, BSplines, LinearAlgebra
using DataFrames, CSV, Plots, Optim

identity(x) = x
logit(p) = log(p/(1-p))

# df = CSV.read("dev/engine.csv", DataFrame)
# y = df.wear; x = [df.size]
# BasisArgs = [(15,4)]

df = dataset("datasets", "trees");
x = [df.Girth, df.Height]
y = df.Volume
sp = [2,2]
BasisArgs = [(10, 2), (10,2)]

Families = Dict(
    "Normal" => Dict(
        "Name" => "Normal",
        "Distribution" => Normal,
        "Link" => identity,
        "InverseLink" => identity
    ),
    "Gamma" => Dict(
        "Name" => "Gamma",
        "Distribution" => Gamma,
        "Link" => log,
        "InverseLink" => exp
    ),
    "Poisson" => Dict(
        "Name" => "Poisson",
        "Distribution" => Poisson,
        "Link" => log,
        "InverseLink" => exp
    )
)

mutable struct GAMData
    y::AbstractArray
    x::AbstractArray
    Basis::AbstractArray{BSplineBasis}
    Family::Dict
    Coef::AbstractArray
    ColMeans::AbstractArray
    CoefIndex::AbstractArray
    Fitted::AbstractArray
    Diagnostics::Dict

    function GAMData(
        y::AbstractArray,
        x::AbstractArray,
        Basis::AbstractArray,
        Family::Dict,
        Coef::AbstractArray,
        ColMeans::AbstractArray,
        CoefIndex::AbstractArray,
        Fitted::AbstractArray,
        Diagnostics::Dict
    )
        new(y, x, Basis, Family, Coef, ColMeans, CoefIndex, Fitted, Diagnostics)
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

    Dp = dcat(map((p, d) -> (sqrt(p) * d), sp, D))
    Xi = vcat(
        repeat([1], n),
        repeat([0], size(Dp,1))
    )
    Xp = hcat(Xi, vcat(hcat(X...), Dp))
    Yp = vcat(y, repeat([0], size(Dp, 1)))

    return Xp, Yp, ColMeans, CoefIndex
end

function ModelDiagnostics(y, y_hat, X)
    n = length(y)
    H = X * pinv(X' * X) * X' # hat matrix
    trF = sum(diag(H)[1:n]) # EDF
    rsd = y - y_hat # residuals
    rss = sum(rsd.^2) # residual SS
    sig_hat = rss/(n-trF) # residual variance
    gcv = sig_hat*n/(n-trF) # GCV score

    return Dict(
        "RSS" => rss,
        "EDF" => trF,
        "GCV" => gcv
    )
end

# function FitOLS(y, x, sp, Basis)

#     n = length(y)
#     Xp, Yp, ColMeans, CoefIndex = BuildPenaltyMatrix(y, x, sp, Basis)
#     B = Xp \ Yp
#     y_hat = (Xp * B)[1:n]
#     diagnostics = ModelDiagnostics(y, y_hat, Xp)

#     return GAMData(
#         y,
#         x,
#         Basis,
#         Families["Normal"],
#         B,
#         ColMeans,
#         CoefIndex,
#         y_hat,
#         diagnostics
#     )
# end
#
# FitOLS(y, x, [2, 2], Basis).Diagnostics["GCV"]

function FitWPS(y, x, sp, Basis, w = ones(length(y)))

    n = length(y)
    w = sqrt.(w)
    yw = w .* y
    Xp, Yp, ColMeans, CoefIndex = BuildPenaltyMatrix(yw, x, sp, Basis)
    B = Xp \ Yp
    y_hat = (Xp * B)[1:n]
    diagnostics = ModelDiagnostics(y, y_hat, Xp)

    return GAMData(
        y,
        x,
        Basis,
        Families["Normal"],
        B,
        ColMeans,
        CoefIndex,
        y_hat,
        diagnostics
    )
end

FitWPS(y, x, [2, 2], Basis).Diagnostics["GCV"]

rho = range(-20, 11)
out = []
for i in eachindex(rho)
    mod = FitWPS(y, x,repeat([exp(rho[i])], length(x)), Basis)
    push!(out, mod.Diagnostics["GCV"])
end
plot(rho, out)


function GAMFit(y, x, sp, Basis, Family)
    eta = Family["Link"].(y)
    old_gcv = -100
    for i in 1:25
        mu = Family["InverseLink"].(eta)
        z = (y .- mu) ./ mu .+ eta
        global mod = FitWPS(z, x, sp, Basis)

        if abs(mod.Diagnostics["GCV"] - old_gcv) < 1e-5*mod.Diagnostics["GCV"]
            break
        end
        old_gcv = mod.Diagnostics["GCV"]
        eta = mod.Fitted
    end
    mod.Family = Family
    mod.Fitted = Family["InverseLink"].(mod.Fitted)
    return mod
end


function PIRLS(y, x, sp, Basis, Family)
    
    mu = y
    eta = Family["Link"].(mu)
    
    logLik = sum(map(x -> logpdf(Family["Distribution"](x), x), y))
    dev = logLik
    for i in 1:25
        z = @. (y - mu) / mu + eta
        #w = mu
        w = ones(length(mu))
        global mod = FitWPS(z, x, sp, Basis, w)
        eta = mod.Fitted
        mu = Family["InverseLink"].(eta)
        oldDev = dev
        dev = 2 * (logLik - sum(map((x,y) -> logpdf(Family["Distribution"](x), y), mu, y)))
        if abs(dev - oldDev) < 1e-6 * dev
            break
        end
    end
    mod.Family = Family
    mod.Fitted = Family["InverseLink"].(mod.Fitted)
    return mod
end

# 0.01014639;
sp = [1,1]
GAMFit(y, x, sp, Basis, Families["Gamma"]).Diagnostics["GCV"]

PIRLS(y, x, sp, Basis, Families["Gamma"]).Diagnostics["GCV"]

rho = range(-20, 11)
out = []
for i in eachindex(rho)
    mod = PIRLS(y, x, repeat([exp(rho[i])], length(x)), Basis, Families["Gamma"])
    push!(out, mod.Diagnostics["GCV"])
end
plot(rho, out)


# looking for  8.528943 246888083.011660
function GAMFitGCV(y, x, Basis, Family)
    res = optimize(sp -> PIRLS(y, x, exp.(sp), Basis, Family).Diagnostics["GCV"], zeros(length(x)), NelderMead())
    exp.(Optim.minimizer(res))
end

modsp = GAMFitGCV(y, x, Basis, Families["Gamma"])
mod = GAMFit(y, x, modsp, Basis, Families["Gamma"])


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