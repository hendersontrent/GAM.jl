# GAM.jl
Fit, evaluate, and visualise generalised additive models (GAMs) in native Julia

## Motivation

[Generalised additive models](https://en.wikipedia.org/wiki/Generalized_additive_model) (GAMs) are an incredibly powerful modelling tool for regression practitioners. However, the functionality of the excellent R package [`mgcv`](https://cran.r-project.org/web/packages/mgcv/mgcv.pdf) is yet to be built in native Julia. This package aims to do just that, albeit at much less complexity given how sophisticated `mgcv` is.

## Usage

The basic interface to `GAM.jl` is the `FitGAM` function, which is as easy as:

```{julia}
mod = FitGAM(y, x, Dists[:Gamma], Links[:Log], [(10, 2), (10, 2)])
```

where `Dists` is a Dictionary of available likelihood families and `Links` is a Dictionary of link functions that can be used for each likelihood. In this example, `X` is comprised of two covariates and we are specifying 10 knots and a polynomial order of 2 for the splines for both.

Users can also control the penalised iteratively reweighted least squares algorithm directly:

```{julia}
mod = FitGAM(mod = FitGAM(y, x, Dists[:Gamma], 
             Links[:Log], [(10, 2), (10, 2)]);
             Optimizer = NelderMead(), maxIter = 1e5,
             tol = 1e-6)
```

Note that this current interface is not as elegant as the formula-driven `gam` function in R's `mgcv` package -- this aspect of `GAM.jl` is a work-in-progress (it's tricky to get user-specified smooths down!).

## Development notes

`GAM.jl` is very much in active development. Please check back for updates and new features or feel free to contribute yourself! The project to-date has been a collaboration between [Trent Henderson](https://github.com/hendersontrent) and [Mason Yahr](https://github.com/yahrMason).
