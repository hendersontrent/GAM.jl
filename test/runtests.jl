using DataFrames, Plots, Distributions, Optim, GLM, GAM, Test

#-------------------- Set up data -----------------

# Hard-coded version of the `trees` dataset

df = DataFrame(Girth = [8.3,8.6,8.8,10.5,10.7,10.8,11,11,11.1,11.2,11.3,11.4,11.4,11.7,12,12.9,12.9,13.3,13.7,13.8,14,14.2,14.5,16,16.3,17.3,17.5,17.9,18,18,20.6], 
               Height = [70,65,63,72,81,83,66,75,80,75,79,76,76,69,75,74,85,86,71,64,78,80,74,72,77,81,82,80,80,80,87],
               Volume = [10.3,10.3,10.2,16.4,18.8,19.7,15.6,18.2,22.6,19.9,24.2,21,21.4,21.3,19.1,22.2,33.8,27.4,25.7,24.9,34.5,31.7,36.3,38.3,42.6,55.4,55.7,58.3,51.5,51,77],
               )

#-------------------- Run tests -----------------

@testset "GAM.jl" begin
    
    # Core function

    tree_gam = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df; family=Normal(), link=IdentityLink(), optimizer=GradientDescent())
    @test tree_gam isa GAMModel

    # Plotting function

    p = PlotSmooth(tree_gam; smooth_index=2)
    @test p isa Plots.Plot
end