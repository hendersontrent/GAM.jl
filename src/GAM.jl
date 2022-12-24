module GAM

using LinearAlgebra, Distributions, BSplines, Optim, StatsBase, Plots, DataFrames

include("pen_log_lik.jl")
include("cv_log_lik.jl")
include("optimal_knots.jl")
include("fit_gam.jl")
include("plot_gam.jl")
include("predict_gam.jl")
include("summary.jl")

export pen_log_lik
export cv_log_lik
export optimal_knots
export GAMModel
export fit_gam
export plot_gam
export predict_gam
export summary

end
