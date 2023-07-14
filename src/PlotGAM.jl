"""
    PlotGAM(model)
Plots smooths for a fitted GAMModel object.
Usage:
```julia-repl
PlotGAM(model)
```
Arguments:
- `model` : `GAMModel` containing the fitted GAM.
"""

function PlotGAM(model::GAMModel; kwargs...)
    y_mean = mean(model.y);
    p = scatter(model.x, model.y .+ y_mean, label = "Data"; kwargs...)
    xlabel!("x")
    ylabel!("y")
    plot!(model.x, model.Spline_opt.(model.x) .+ y_mean, color = :blue, linewidth = 3, label = "Spline: λ = $(round(model.λ_opt,digits=3))")
    return p
end