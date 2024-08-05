# GAM.jl
Fit, evaluate, and visualise generalised additive models (GAMs) in native Julia

## Motivation

[Generalised additive models](https://en.wikipedia.org/wiki/Generalized_additive_model) (GAMs) are an incredibly powerful modelling tool for regression practitioners. However, the functionality of the excellent R package [`mgcv`](https://cran.r-project.org/web/packages/mgcv/mgcv.pdf) is yet to be built in native Julia. This package aims to do just that, albeit at much less complexity given how sophisticated `mgcv` is.

## Usage

The basic interface to `GAM.jl` is the `gam` function, which is as easy as:

```{julia}
    mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)
```

by default, `gam` uses a Normal likliehood and the identity link function. In this example, `df` is the `trees` dataset included in R and in Julia through the [`RDatasets`](https://github.com/JuliaStats/RDatasets.jl) package. We are using two covariates in `df` -- `Girth` and `Height` -- and we are specifying 10 knots and a polynomial order of 3 for the splines for both. Note the similarity between the formula specification and that used by `mgcv` in R. Unfortunately, we have not yet solved a symbolic way to represent the formula using a macro instead of a string, but hopefully this will be come in the near future!

Users can also control the penalised iteratively reweighted least squares algorithm directly, as well as change the likelihood family and link function:

```{julia}
mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)",        df; Family = "Gamma", Link = "Log", 
          Optimizer = NelderMead(), maxIter = 1e5,
          tol = 1e-6)
```

Note that currently, the following families are supported:

* `Normal`
* `Poisson`
* `Gamma`

And the following link functions:

* `Identity`
* `Log`

## Development notes

`GAM.jl` is very much in active development. Please check back for updates and new features or feel free to contribute yourself! The project to-date has been a collaboration between [Trent Henderson](https://github.com/hendersontrent) and [Mason Yahr](https://github.com/yahrMason).
