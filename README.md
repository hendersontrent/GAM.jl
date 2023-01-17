# GAM.jl
Fit, evaluate, and visualise generalised additive models (GAMs) in native Julia

## Installation

*Coming soon!*

## Motivation

[Generalised additive models](https://en.wikipedia.org/wiki/Generalized_additive_model) (GAMs) are an incredibly powerful modelling tool for regression practitioners. However, the functionality of the excellent R package [`mgcv`](https://cran.r-project.org/web/packages/mgcv/mgcv.pdf) is yet to be built in native Julia. This package aims to do just that, albeit at much less complexity given how sophisticated `mgcv` is.

## Development notes

`GAM.jl` is very much a work in progress. Please check back for updates and new features!

## Model structure

Currently, `GAM.jl` fits the following model:

$$
y = g^{-1}(\beta_0 + \sum_{j=1}^{p} f_j(x_j)) + \epsilon \,
$$

where $y$ is the response variable, $g^{-1}$ is the link function, $\beta_0$ is the intercept, $f_j$ is the smooth term for predictor $x_j$, and $\epsilon$ is the error term.

The smooth term $f_j$ is modeled using a spline basis expansion:

$$
f_j(x_{ij}) = \sum_{k=1}^{K_j} \beta_{jk} B_{jk}(x_{ij}) \,
$$

where $K_j$ is the number of spline basis functions for predictor $j$, $\beta_{jk}$ is the coefficient for spline basis function $k$ for predictor $j$, and $B_{jk}(x_{ij})$ is the $k$ -th spline basis function for predictor $j$ evaluated at $x_{ij}$. The model is fit by minimizing the penalized negative log likelihood of the model, which is given by:

$$
\mathcal{L}(\beta) = -\sum_{i=1}^n \log p(y_i \mid \beta) + \lambda \sum_{j=1}^p \sum_{k=1}^{K_j} \beta_{jk}^2 \,
$$

where $\mathcal{L}(\beta)$ is the negative log likelihood, $n$ is the number of samples, $p$ is the number of smooth terms, $K_j$ is the number of knots for the $j$-th smooth term, $\beta_{jk}$ is the coefficient for the $k$-th knot in the $j$-th smooth term, $y_i$ is the $i$-th response variable, $\lambda$ is the penalty term, and $p(y_i \mid \beta)$ is the probability density function (PDF) or probability mass function (PMF) of the response variable given the model parameters.. The first term in the sum is the negative log likelihood of the response variable given the model coefficients, and the second term is the penalty term that is added to the negative log likelihood to enforce smoothness in the model.