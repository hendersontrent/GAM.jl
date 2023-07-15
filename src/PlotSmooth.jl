"""
    PlotSmooth(mymodel; smooth_index)
Plot a smooth.
Usage:
```julia-repl
PlotSmooth(mymodel; smooth_index)
```
Arguments:
- `mymodel` : `GAMModel` containing the fitted GAM.
- `smooth_index` : `Int64` denoting the index of the smooth in `mymodel.covariateFits` to plot. Defaults to `2` as `1` is the Intercept.
"""

function PlotSmooth(mymodel::GAMModel; smooth_index::Int64=2, kwargs...)

    # Check that the smooth_index is a smooth and not a discrete covariate

    if typeof(mymodel.covariateFits[smooth_index]) == SmoothData
        p = scatter(mymodel.data[!, mymodel.covariateFits[smooth_index].variable], mymodel.data[!, mymodel.y_var], label = "Data", color = :black; kwargs...)
        xlabel!(String(mymodel.covariateFits[smooth_index].variable))
        ylabel!(String(y_var))
        plot!(mymodel.data[!, mymodel.covariateFits[smooth_index].variable], mymodel.covariateFits[smooth_index].Spline_opt_lower.(data[!, mymodel.covariateFits[smooth_index].variable]) .+ mymodel.covariateFits[1].β_opt, fillrange = mymodel.covariateFits[smooth_index].Spline_opt_upper.(mymodel.data[!, mymodel.covariateFits[smooth_index].variable]) .+ mymodel.covariateFits[1].β_opt, fillalpha = 0.2, pointalpha = 0.0, color = :grey, label = "CI")
        plot!(mymodel.data[!, mymodel.covariateFits[smooth_index].variable], mymodel.covariateFits[smooth_index].Spline_opt.(mymodel.data[!, mymodel.covariateFits[smooth_index].variable]) .+ mymodel.covariateFits[1].β_opt, color = :grey, linewidth = 2, label = "Optimal smooth λ = $(round(mymodel.covariateFits[smooth_index].λ_opt,digits=3))")
        return p 
    else
        error("Specified smooth_index is not a smooth.")
    end
end