module GAM

using LinearAlgebra, Distributions, BSplines, Optim, StatsBase, Plots, DataFrames

include("FitGAM.jl")
include("plot.jl")
include("predict.jl")
include("summary.jl")

export GAMModel
export fitGAM
export plot
export predict
export summary

end # module
