using GAM
using Test
using RDatasets, Plots

#-------------------- Set up data -----------------

df = dataset("datasets", "trees");
#x = [df.Girth, df.Height]
#y = df.Volume

#-------------------- Run tests -----------------

@testset "GAM.jl" begin

    mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)

    p = plotGAM(mod)
    @test p isa Plots.Plot
end