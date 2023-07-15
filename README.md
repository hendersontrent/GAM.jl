# GAM.jl
Fit, evaluate, and visualise generalised additive models (GAMs) in native Julia

## Motivation

[Generalised additive models](https://en.wikipedia.org/wiki/Generalized_additive_model) (GAMs) are an incredibly powerful modelling tool for regression practitioners. However, the functionality of the excellent R package [`mgcv`](https://cran.r-project.org/web/packages/mgcv/mgcv.pdf) is yet to be built in native Julia. This package aims to do just that, albeit at much less complexity given how sophisticated `mgcv` is.

Credit where credit is due, the original structure of `GAM.jl` was very different to the version it is now. The change occurred after I read [this excellent post](https://yahrmason.github.io/bayes/gams-julia/) by [Mason Yahr](https://twitter.com/yahrMason)---of which the current state of this package is more or less a port of.

### What is a GAM?

A generalised additive model (GAM) is a generalised linear model (GLM) where the linear predictor is given by a user specified sum of smooth functions of the covariates plus a parametric component of the linear predictor. For example:

$$
\text{log}(E(y_{i})) = \alpha + f_{1}(x_{1i}) + f_{2}(x_{2i})
$$

where the response variable is $y_{i} \sim \text{Poi}$ and $f_{1}$ and $f_{2}$ are smooth functions of covariates $x_{1}$ and $x_{2}$. The log is an example of a link function which makes a GLM "generalised". Each family (i.e., Poisson) has a set of appropriate link functions that can be expressed, depending on the context.

Similar to `mgcv`, the `GAM.jl` implementation of GAMs represents the smooth functions using penalised regression splines, and by default uses basis functions for these splines that are designed to be optimal, given the number basis functions used.

## Development notes

`GAM.jl` is very much a work in progress. Please check back for updates and new features or feel free to contribute yourself!