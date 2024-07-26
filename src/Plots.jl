"""
    PartialDependencePlot(mod, ix)
Draw partial dependence plot.

Usage:
```julia-repl
PartialDependencePlot(mod, ix)
```
Arguments:
- `mod` : `GAMData` containing the model.
- `ix` : `Int` denoting the variable to plot.
"""
function PartialDependencePlot(mod, ix)
    x = mod.x[ix]
    pred = PredictPartial(mod, ix)
    ord = sortperm(x)
    return plot(x[ord], pred[ord])
end

"""
    plotGAM(mod)
Plot GAM.

Usage:
```julia-repl
plotGAM(mod)
```
Arguments:
- `mod` : `GAMData` containing the model.
"""
function plotGAM(mod)
    partialPlot = map(x -> PartialDependencePlot(mod, x), eachindex(mod.x))
    plot(partialPlot..., layout=(1, length(partialPlot)), link = :y)
end
