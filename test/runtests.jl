using GAM
using Test
using RDatasets, Plots

#-------------------- Set up data -----------------

df = dataset("datasets", "trees");

#-------------------- Run tests -----------------

@testset "GAM.jl" begin

    mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)

    p = plotGAM(mod)
    @test p isa Plots.Plot

    # Gamma version

    mod2 = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df; Family = "Gamma", Link = "Log")

    p1 = plotGAM(mod2)
    @test p1 isa Plots.Plot
end