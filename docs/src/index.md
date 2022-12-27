# GAM.jl manual

Fit, evaluate, and visualise generalised additive models (GAMs) in native Julia.

## Installation

*Coming soon!*

## Fitting GAMs

Currently, only a formula-based way of fitting GAMs is possible: `fit_gam(formula, data, family, knots, degree, n_knots, λ, n_folds)`. This design decision was made to ensure consistency with the popular R package for fitting GAMs [`mgcv`](https://cran.r-project.org/web/packages/mgcv/mgcv.pdf). However, only three arguments are essential, and a GAM can be fit as simply as with `fit_gam(formula, data, family)`. The arguments include:

* `formula` --- The model formula.
* `data` --- DataFrame of input data.
* `family` --- Family of the likelihood function.
* `λ` --- Coefficient of the penalty term.
* `n_folds` --- Number of folds to use in generalised cross-validation of parameters.

Currently, the following family types are supported:

* `:gaussian` --- `Normal()` (continuous response variable).
* `:gamma` --- `Gamma()` (continuous, positive response variable).
* `:binomial` --- `Binomial()` (discrete, two-category response variable).
* `:poisson` --- `Poisson()` (discrete, integer count response variable).

As an example, here is a simple GAM fit on the classic [`mtcars`](https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/mtcars) dataset.

```jldoctest
using RDatasets, GAM

mtcars = dataset("datasets", "mtcars")
mod = fit_gam(@formula(MPG ~ s(WT, 3) + s(HP, 3) + AM + Cyl), mtcars, :gaussian)
```

Note the same syntactical structure as `mgcv` where `s()` in `GAM.jl` denotes a smooth term to be constructed over a continuous predictor. Categorical predictors are left as-is---such as `AM` in the above example. For smooth terms, only one argument is required, which is the variable (column) name in the DataFrame. For more nuanced modelling, users can ignore the defaults to the second two arguments (degree---`degree` and knots---`k`). `degree` takes an integer value between 1 and 4. A value of 1 corresponds to a linear spline, a value of 2 corresponds to a quadratic spline, and so on. The default value for degree is 3, which corresponds to a cubic spline. `k` takes an integer value between 1 and 3. A value of 1 corresponds to a linear polynomial, a value of 2 corresponds to a quadratic polynomial, and a value of 3 corresponds to a cubic polynomial. The default value for k is 1, which corresponds to a linear polynomial.

Note that if a prepended `1` is omitted from the formula, `GAM.jl` will automatically include a column of ones into the data matrix for the intercept. Alternatively, if a prepended `1` is detected (e.g., `@formula(MPG ~ 1 + s(WT, 3) + s(HP, 3) + AM + Cyl)`), `GAM.jl` will also take that to include an intercept term. Either way, one is included.

## Evaluating GAMs

`GAM.jl` provides a host of functionality to understand your GAM. These functions include:

* `summarise_gam` (`summarize_gam`) --- Generate a summary table of model information, including coefficients, their standard errors, p-values, and confidence intervals.
* `plot_gam` --- Generate a plot of the smooth for one of the predictor variables in the GAM with a confidence interval.
* `predict_gam` --- Generates a vector of predictions for new data using the fitted GAM.

### Summarising

A useful tool in regression modelling is the coefficients table. `GAM.jl` produces this using `summarise_gam` (or `summarize_gam`) which only takes two arguments:

* `model` --- The `GAMModel` object created by `fit_gam`.
* `prob` --- The probability for the confidence intervals. Defaults to `0.95` for 95% confidence intervals.

This function produces a familiar-looking table to those who use tools such as [`GLM.jl`](https://github.com/JuliaStats/GLM.jl) or any regression function in R such as `lm`, `glm`, or `gam` from `mgcv`:

```jldoctest
println(summarise_gam(mod, 0.95))
```

### Plotting

Currently, `GAM.jl` only supports one type of plot which is a ribbon plot of the smooth of one of the predictor variables, complete with confidence intervals. `plot_gam` only takes three arguments:

* `model` --- The `GAMModel`.
* `x_var` --- String name of the variable of interest in the DataFrame.
* `prob` --- The probability for the confidence intervals.

This function produces similar graphics to `mgcv::plot.gam` in R:

```jldoctest
plot_gam(mod, 0.95)
```

### Predicting

Generating predictions for new data using a fitted GAM is very simple in `GAM.jl`. The function `predict_gam` only takes two arguments:

- `model` --- The `GAMModel` object.
- `newdata` --- DataFrame of new input data.

```jldoctest
predict_gam(mod, 0.95)
```