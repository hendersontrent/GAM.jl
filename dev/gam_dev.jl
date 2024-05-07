using Distributions, RDatasets, GLM, Optim, BSplines, LinearAlgebra
using DataFrames, CSV, Plots, Optim

df = CSV.read("julia/2023-06-21-gams-julia/engine.csv", DataFrame)
y = df.wear; x = [df.size]
BasisArgs = [(15,4)]

df = dataset("datasets", "trees");
x = [df.Girth, df.Height]
y = df.Volume
BasisArgs = [(5, 4), (5,4)]

mutable struct GAMData
    y::AbstractArray
    x::AbstractArray
    BasisArgs::Array{Tuple{Int,Int}}
    Basis::AbstractArray{BSplineBasis}
    BasisMatrix::AbstractArray
    BasisMatrixColMeans::AbstractArray
    CoefIndex::AbstractArray
    Penalty::AbstractArray
    Coef::AbstractArray
    Fit::Bool

    function GAMData(y::AbstractArray, x::AbstractArray, BasisArgs::Array{Tuple{Int,Int}})
        Basis = map((xi, argi) -> BuildQuantileBasis(xi, argi[1], argi[2]), x, BasisArgs)
        BasisMatrix = map(BuildBasisMatrix, Basis, x)
        BasisMatrixColMeans = BuildBasisMatrixColMeans(BasisMatrix)
        CoefIndex = BuildCoefIndex(BasisMatrix)
        Penalty = fill(nothing, length(x))
        Coef = fill(nothing, 1 + sum(size.(BasisMatrix,1)))
        new(
            y, x, BasisArgs, Basis, BasisMatrix, 
            BasisMatrixColMeans, CoefIndex, Penalty, Coef, false
        )
    end
end

function BuildQuantileBasis(x::Vector, n_knots::Int64, order::Int64)
    KnotsList = quantile(x, range(0, 1; length=n_knots));
    Basis = BSplineBasis(order, KnotsList);
    return Basis
end

Basis = map((xi, argi) -> BuildQuantileBasis(xi, argi[1], argi[2]), x, BasisArgs)

# Build a Matrix representation of the Basis
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

X = map(BuildBasisMatrix, Basis, x)

function BuildBasisMatrixColMeans(BasisMatrix::AbstractArray)
    # Calculate means of each column
    return mean.(X, dims=1)
end

colMeans = BuildBasisMatrixColMeans(X)

function BuildCoefIndex(BasisMatrix::AbstractArray)
    ix = [1:1]
    ix_end = cumsum(vcat([1], size.(BasisMatrix,2)))
    append!(ix, [ix_end[i-1]+1:ix_end[i] for i in 2:length(ix_end)])
    return ix[2:end]
end

CoefIndex = BuildCoefIndex(X)

gamData = GAMData(y, x, BasisArgs)


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
function BuildDifferenceMatrix(Basis::BSplineBasis{Vector{Float64}})
    D = diffm(
        diagm(0 => ones(length(Basis))),
        1, # matrix dimension
        2  # number of differences
    )
    return D
end

D = map(BuildDifferenceMatrix, gamData.Basis)

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

function CenterBasisMatrix(BasisMatrix::AbstractMatrix, BasisMatrixColMeans::AbstractArray)
    return BasisMatrix .- BasisMatrixColMeans
    #return map((c, m) -> (c .- m), gamDataBasisMatrix, gamDataBasisMatrixColMeans)
end

X = map(CenterBasisMatrix, gamData.BasisMatrix, gamData.BasisMatrixColMeans)

penalties = repeat([exp(1.0)], length(gamData.x))
function BuildPenaltyMatrix(gamData::GAMData, penalties::AbstractVector)

    n = length(gamData.y)
    X = map(CenterBasisMatrix, gamData.BasisMatrix, gamData.BasisMatrixColMeans)
    D = map(BuildDifferenceMatrix, gamData.Basis)
    Dp = dcat(map((p, d) -> (sqrt(p) * d), penalties, D))
    Xi = vcat(
        repeat([1], n),
        repeat([0], size(Dp,1))
    )
    Xp = hcat(Xi, vcat(hcat(X...), Dp))
    Yp = vcat(gamData.y, repeat([0], size(Dp, 1)))

    return Xp, Yp
end

Xp, Yp = BuildPenaltyMatrix(gamData, penalties)

# Function to estimate GAM Coefs
function EstimatePenaltyCoef(gamData::GAMData, penalties::AbstractVector)
    Xp, Yp = BuildPenaltyMatrix(gamData, penalties)
    B = Xp \ Yp
    return B
end

B = EstimatePenaltyCoef(gamData, penalties)

# Function to calculate GCV for given penalty
function GCV(gamData::GAMData, penalties::AbstractVector)
    n = length(gamData.y) # number of observations
    Xp, Yp = BuildPenaltyMatrix(gamData, penalties)
    B = Xp \ Yp
    H = Xp * pinv(Xp' * Xp) * Xp' # hat matrix
    trF = sum(diag(H)[1:n]) # EDF
    y_hat = Xp * B # predictions
    rsd = gamData.y - y_hat[1:n] # residuals
    rss = sum(rsd.^2) # residual SS
    sig_hat = rss/(n-trF) # residual variance
    gcv = sig_hat*n/(n-trF) # GCV score
    return gcv
end

GCV(gamData, penalties)

rho = range(-9, 11)
out = []
for i in eachindex(rho)
    push!(out, GCV(gamData, repeat([exp(rho[i])], length(gamData.x))))
end
plot(rho, out)

function OptimizePenaltyGCV(gamData::GAMData)
    initial = zeros(length(gamData.x))
    res = optimize(x -> GCV(gamData, exp.(x)), initial, NelderMead())
    return exp.(Optim.minimizer(res))
end

optim_penalty = OptimizePenaltyGCV(gamData)

function setPenalty!(gamData::GAMData, Penalty::AbstractArray)
    setfield!(gamData, :Penalty, Penalty)
end

setPenalty!(gamData, optim_penalty)

function setCoef!(gamData::GAMData, Coef::AbstractVector)
    setfield!(gamData, :Coef, Coef)
    setfield!(gamData, :Fit, true)
end

optim_coef = EstimatePenaltyCoef(gamData, optim_penalty)

setCoef!(gamData, optim_coef)

function FitGAM(y::AbstractArray, x::AbstractArray, BasisArgs::Array{Tuple{Int,Int}})
    gamData = GAMData(y::AbstractArray, x::AbstractArray, BasisArgs::Array{Tuple{Int,Int}})
    penalties = OptimizePenaltyGCV(gamData)
    gamCoef = EstimatePenaltyCoef(gamData, penalties)
    setPenalty!(gamData, penalties)
    setCoef!(gamData, gamCoef)
    return gamData
end

gamData = FitGAM(y, x, BasisArgs)

function BuildPredictionMatrix(gamData::GAMData, idx::Int, values::AbstractArray)
    # need to assert model is Fit
    @assert gamData.Fit
    basis = gamData.Basis[idx]
    colMeans = gamData.BasisMatrixColMeans[idx]
    basisMatrix = BuildBasisMatrix(basis, values)
    return CenterBasisMatrix(basisMatrix, colMeans)
end

function PredictPartial(gamData::GAMData, idx::Int, values::AbstractArray)
    predMatrix = BuildPredictionMatrix(gamData, idx, values)
    predBeta = gamData.Coef[gamData.CoefIndex[idx]]
    return predMatrix * predBeta
end

function PartialDependencePlot(gamData::GAMData, idx::Int)
    values = gamData.x[idx]
    sort!(values)
    pred = PredictPartial(gamData, idx, values)
    return plot(values, pred)
end

function plotGAM(gamData::GAMData)
    partialPlot = map(x -> PartialDependencePlot(gamData, x), eachindex(gamData.x))
    plot(partialPlot..., layout=(1, length(partialPlot)), link = :y)
end

plotGAM(gamData)