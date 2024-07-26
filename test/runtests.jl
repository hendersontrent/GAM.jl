using RDatasets, Test

#-------------------- Set up data -----------------

# `trees` dataset

df = dataset("datasets", "trees");
x = [df.Girth, df.Height]
y = df.Volume
sp = [2,2]
BasisArgs = [(10, 2), (10,2)]

#-------------------- Run tests -----------------

@testset "GAM.jl" begin

    # Main function and plot

    mod = FitGAM(y, x, Basis, Dists[:Gamma], Links[:Log])
    p = plotGAM(mod)
    @test p isa Plots.Plot
end