"""
    BuildPredictionMatrix(x, Basis, ColMeans)
Build prediction matrix.

Usage:
```julia-repl
BuildPredictionMatrix(x, Basis, ColMeans)
```
Arguments:
- `x` : `Vector` of data for a variable.
- `Basis` : `AbstractMatrix` containing basis matrix.
- `ColMeans` : `Matrix` containing column means.
"""
function BuildPredictionMatrix(x::AbstractArray, Basis::BSplineBasis, ColMeans::AbstractArray)
    basisMatrix = DropCol(BuildBasisMatrix(Basis, x), length(Basis.breakpoints))
    return CenterBasisMatrix(basisMatrix, ColMeans)
end

"""
    PredictPartial(mod, ix)
Predict partial values.

Usage:
```julia-repl
PredictPartial(mod, ix)
```
Arguments:
- `mod` : `GAMData` containing the model.
- `ix` : `Int` denoting the variable to plot.
"""
function PredictPartial(mod, ix)
    predMatrix = BuildPredictionMatrix(mod.x[ix], mod.Basis[ix], mod.ColMeans[ix])
    predBeta = mod.Coef[mod.CoefIndex[ix]]
    return predMatrix * predBeta
end
