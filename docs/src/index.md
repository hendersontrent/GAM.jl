# GAM.jl manual

Fit and evaluate generalised additive models (GAMs) in Julia

## Installation

*Coming soon!*

## Fitting GAMs

Currently, only a single non-formula based way of fitting GAMs is possible: `fit_gam(X, y, family, knots, degree, n_knots, λ, n_folds)`, however, only three arguments are essential, and a GAM can be fit as simply as with `fit_gam(X, y, family)`. The arguments include:

* `X` --- Data matrix of predictor variables.
* `y` --- Response variable vector.
* `family` --- Family of the likelihood function.
* `knots` --- Spline knot positions.
* `degree` --- Polynomial degree of the spline.
* `n_knots` --- Number of knots in the spline.
* `λ` --- Coefficient of the penalty term.
* `n_folds` --- Number of folds to use in generalised cross-validation of parameters.

Currently, the following family types are supported:

* `:gaussian` --- `Normal()` (continuous response variable).
* `:binomial` --- `Binomial()` (discrete, two-category response variable).
* `:poisson` --- `Poisson()` (discrete, integer count response variable).

As an exmaple, here is a simple GAM fit on the classic [`mtcars`](https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/mtcars) dataset.

```jldoctest
using Random, RDatasets, GAM

mtcars = dataset("datasets", "mtcars")
X = Matrix(mtcars[:, [:AM, :Cyl, :WT, :HP]])
y = mtcars[:, :MPG]
model = fit_gam(X, y, :gaussian)
```

## Evaluating GAMs

`GAM.jl` provides a host of functionality to understand your GAM. These functions include:

* `summarise_gam` (`summarize_gam`) --- Generate a summary table of model information, including coefficients, their standard errors, p-values, and confidence intervals.
* `plot_gam` --- Generate a plot of the smooth for one of the predictor variables in the GAM with a confidence interval.
* `predict_gam` --- Generates a vector of predictions for new data using the fitted GAM.

### Summarising

A useful tool in regression modelling is the coefficients table. `GAM.jl` produces this using `summarise_gam` (or `summarize_gam`) which only takes two arguments:

* `model` --- The `GAMModel` object created by `fit_gam`.
* `prob` --- The probability for the confidence intervals. Defaults to `0.95` for 95% confidence intervals.

This function produces a familiar-looking table to those who use tools such as [`GLM.jl`](https://github.com/JuliaStats/GLM.jl) or any regression function in R such as `lm`, `glm`, or `gam` from [`mgcv`](https://cran.r-project.org/web/packages/mgcv/mgcv.pdf):



### Plotting

Currently, `GAM.jl` only supports one type of plot which is a ribbon plot of the smooth of one of the predictor variables, complete with confidence intervals. `plot_gam` only takes five arguments:

* `model` --- The `GAMModel` object created by `fit_gam`.
* `X` --- Data matrix of predictor variables.
* `y` --- Response variable vector.
* `x_var` --- The column index of the variable in `X` to plot the smooth for.
* `prob` --- The probability for the confidence intervals. Defaults to `0.95` for 95% confidence intervals.

### Predicting

Generating predictions for a GAM is very simple in `GAM.jl`. The function `predict_gam` only takes three arguments:

* `model` --- The `GAMModel` object created by `fit_gam`.
* `X` --- Matrix of new input data.
* `type` --- Whether to generate mean or probability predictions.