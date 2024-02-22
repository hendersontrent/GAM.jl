"""
    ModelFit()
Holds model fit data for the model.

Usage:
```julia-repl
ModelFit()
```
Arguments:
- `ModelFormula` : `XX` XX.
"""
struct ModelFit
    #
end

"""
    GAMModel(ModelFormula, Data, Response, ModelFit)
Holds information relevant to the final fitted GAM model.

Usage:
```julia-repl
GAMModel(ModelFormula, Data, Response, ModelFit)
```
Arguments:
- `ModelFormula` : `String` containing the expression of the model.
- `Data` : `DataFrame` containing the covariates and response variable to use.
- `Response` : `Symbol` denoting the response variable column name.
- `ModelFit` : `ModelFit` containing the model fit information.
"""
struct GAMModel
    ModelFormula::String
    Data::DataFrame
    Response::Symbol
    ModelFit::Any
end