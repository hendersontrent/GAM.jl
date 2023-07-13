module GAM

using Random, FillArrays, Distributions, StatsPlots, GLM, Optim, BSplines, LinearAlgebra

include("QuantileBasis.jl")
include("BasisMatrix.jl")
include("diffm.jl")
include("DifferenceMatrix.jl")
include("PenaltyMatrix.jl")
include("GCV.jl")
include("OptimizeGCVLambda.jl")
include("FitGAM.jl")
include("GAMModel.jl")
include("PlotGAM.jl")

export QuantileBasis
export BasisMatrix
export diffm
export DifferenceMatrix
export PenaltyMatrix
export GCV
export OptimizeGCVLambda
export FitGAM
export GAMModel
export PlotGAM

end
