using GAM
using Test
using RDatasets, DataFrames, Plots

# Get mtcars data

mtcars = dataset("datasets", "mtcars")

@testset "GAM.jl" begin

    # Fit GAM

    mod = fit_gam(@formula(MPG ~ s(WT, 3) + s(HP, 3) + AM + Cyl), mtcars, :gaussian)
    @test mod isa GAMModel

    # Return summary of GAM

    summary = summarise_gam(mod)
    @test summary isa DataFrames.DataFrame

    # Plot GAM for a continuous predictor
    
    p = plot_gam(mod, "WT", 0.95)
    @test p isa Plots.Plot

    # Predict with GAM
    
    preds = predict_gam(mod, mtcars)
    @test preds isa Array
end