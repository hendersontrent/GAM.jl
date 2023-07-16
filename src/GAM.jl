module GAM

using Random, FillArrays, SparseArrays, DataFrames, Distributions, StatsPlots, GLM, Optim, BSplines, LinearAlgebra

include("QuantileBasis.jl")
include("BasisMatrix.jl")
include("diffm.jl")
include("DifferenceMatrix.jl")
include("PenaltyMatrix.jl")
include("GCV.jl")
include("OptimizeGCVLambda.jl")
include("ParseFormula.jl")
include("SmoothData.jl")
include("GAMModel.jl")
include("gam_fit.jl")
include("PlotSmooth.jl")

export QuantileBasis
export BasisMatrix
export diffm
export DifferenceMatrix
export PenaltyMatrix
export GCV
export OptimizeGCVLambda
export GAMModel
export PlotSmooth
export GAMFormula
export ParseFormula
export SmoothData
export NoSmoothData
export gam

end
