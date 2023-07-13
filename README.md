# GAM.jl
Fit, evaluate, and visualise generalised additive models (GAMs) in native Julia

## Motivation

[Generalised additive models](https://en.wikipedia.org/wiki/Generalized_additive_model) (GAMs) are an incredibly powerful modelling tool for regression practitioners. However, the functionality of the excellent R package [`mgcv`](https://cran.r-project.org/web/packages/mgcv/mgcv.pdf) is yet to be built in native Julia. This package aims to do just that, albeit at much less complexity given how sophisticated `mgcv` is.

Credit where credit is due, the original structure of `GAM.jl` was very different to the version it is now. The change occurred after I read [this excellent post](https://yahrmason.github.io/bayes/gams-julia/) by [Mason Yahr](https://twitter.com/yahrMason)---of which the current state of this package is more or less a port of.

## Development notes

`GAM.jl` is very much a work in progress. Currently, functionality is oriented towards a single predictor variable, but we hope to extend this to a full design matrix soon!

Please check back for updates and new features!