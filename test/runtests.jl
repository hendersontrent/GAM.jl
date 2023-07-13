using StatisticalRethinking, Plots, GAM, Test

# Reproduce analysis at https://yahrmason.github.io/bayes/gams-julia/

#-------------------- Set up data -----------------

# Import the cherry_blossoms dataset

data = CSV.read(sr_datadir("cherry_blossoms.csv"), DataFrame);

# Drop records that have missing day of year values

data = data[(data.doy .!= "NA"),:];

# Convert day of year to numeric column

data[!,:doy] = parse.(Float64,data[!,:doy]);

# Create x and y variables

x = data.year;
y_mean = mean(data.doy);
y = data.doy .- y_mean;

#-------------------- Run tests -----------------

@testset "GAM.jl" begin
    
    # Core function

    MyGAM = FitGAM(x, y; n_knots=15, degree=3)
    @test MyGAM isa GAMModel

    # Plotting function

    p = PlotGAM(MyGAM; alpha=0.2)
    @test p isa Plots.Plot
end