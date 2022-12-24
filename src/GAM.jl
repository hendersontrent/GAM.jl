module GAM

using LinearAlgebra, Distributions, BSplines, Optim, StatsBase, Plots, DataFrames

include("pen_log_lik.jl")
include("cv_log_lik.jl")
include("optimal_knots.jl")
include("FitGAM.jl")
include("plot_gam.jl")
include("plot_smooths.jl")
include("predict.jl")
include("summary.jl")

export pen_log_lik
export cv_log_lik
export optimal_knots
export GAM
export fitGAM
export plot_gam
export plot_smooths
export predict
export summary

end
