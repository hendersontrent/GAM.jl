"""
    FitGLM(df, family, link)
Internal worker function to fit a GLM from an augmented penalty matrix.

Usage:
```julia-repl
FitGLM(df, family, link)
```
Arguments:
- `data` : `DataFrame` containing the covariates and response variable to use.
- `family` : `Distribution` denoting the likelihood to use. Must be one of the options in `GLM.jl`.
- `link` : denotes the link function to use for `family`.
"""
function FitGLM(data::DataFrame, family, link)
    cov_names = filter(col -> col != "y", names(data)) .|> Symbol # Filter out the target variable (y_opt) from the list of column names
    f = Term(:y)~sum(Term.(cov_names)) # Construct the formula dynamically
    model = glm(f, data, family, link) # Fit GLM
    return model
end