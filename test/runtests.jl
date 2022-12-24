using GAM
using Test
using Random, Distributions, RDatasets, Plots

# Get mtcars data

mtcars = dataset("datasets", "mtcars")
X = Matrix(mtcars[:, [:AM, :Cyl, :WT, :HP]])
y = mtcars[:, :MPG]

@testset "GAM.jl" begin

    # Fit GAM

    model = fit_gam(X, y, :gaussian)
    @test model isa GAM

    # Return summary of GAM

    summary = summary(model)
    @test summary isa DataFrames.DataFrame

    # Plot GAM for first predictor
    
    p = plot_gam(model, X, y, 1)
    @test p isa Plots.Plot

    # Predict with GAM
    
    preds = predict_gam(model, X, :mean)
    @test p isa Array
end