module GAM

using Random, FillArrays, DataFrames, Distributions, GLM, Optim, BSplines, LinearAlgebra, SparseArrays

include("BasisMatrix.jl")
include("diffm.jl")
include("DifferenceMatrix.jl")
include("PenaltyMatrix.jl")
include("GCV.jl")
include("OptimizeGCVLambda.jl")
include("ParseFormula.jl")
include("GAMModel.jl")
include("gam_fit.jl")

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
export gam
export ModelFit

end
