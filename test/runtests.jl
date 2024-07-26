using GAM
using Test
using RDatasets, Plots

#-------------------- Set up data -----------------

df = dataset("datasets", "trees");
x = [df.Girth, df.Height]
y = df.Volume

#-------------------- Run tests -----------------

@testset "GAM.jl" begin
    mod = FitGAM(y, x, Dists[:Gamma], Links[:Log], [(10, 2), (10, 2)])
    p = plotGAM(mod)
    @test p isa Plots.Plot
end