# GAM.jl
Julia package for fitting, evaluating, and visualising generalised additive models (GAMs)

## Motivation

[Generalised additive models](https://en.wikipedia.org/wiki/Generalized_additive_model) (GAMs) are an incredibly powerful modelling tool for regression practitioners. However, the functionality of the excellent R package [`mgcv`](https://cran.r-project.org/web/packages/mgcv/mgcv.pdf) is yet to be built in native Julia. This package aims to do just that, albeit at much less complexity given how sophisticated `mgcv` is.

## Development notes

`GAM.jl` is very much a work in progress. Please check back for updates and new features!

## Model structure

Currently, `GAM.jl` fits the following model:

$$
y = \beta_0 + \sum_{j=1}^{p} f_j(x_j) + \epsilon \,
$$

where $y$ is the response variable, $\beta_0$ is the intercept, $f_j$ is the smooth term for predictor $x_j$, and $\epsilon$ is the error term.

The smooth term $f_j$ is modeled using a spline basis expansion:

$$
f_j(x_j) = \sum_{k=1}^{K_j} \beta_{j,k} B_{k,j}(x_j) \,
$$

where $K_j$ is the number of spline basis functions for predictor $x_j$, $\beta_{j,k}$ is the coefficient for the $k$th spline basis function, and $B_{k,j}(x_j)$ is the $k$th spline basis function for predictor $x_j$.

The spline basis functions are defined as:

$$
B_{k,j}(x_j) = \prod_{m=1}^{d} (x_j - \text{knots}_{k,m})^{[x_j \geq \text{knots}_{k,m}]} \,
$$

where $d$ is the degree of the spline, $\text{knots}_{k,m}$ is the $m$th knot for the $k$th spline basis function, and $[\cdot]$ is the Iverson bracket.

The spline basis functions are defined such that they are zero outside of the range of the knots. The coefficients $\beta$ are estimated by minimizing the negative log-likelihood:

$$
\mathcal{L} = -\sum_{i=1}^{n} \log p(y_i | f(x_i)) \,
$$

where $p(y_i | f(x_i))$ is the probability density function of the likelihood distribution. The negative log-likelihood is regularised by adding a penalty term:

$$
\mathcal{L} = \frac{1}{2} \sum_{i=1}^{n} \left( y_i - \left( \beta_0 + \sum_{j=1}^{p} f_j(x_{i,j}) \right) \right)^2 + \lambda \beta^T P \beta \,
$$

where $\lambda$ is the regularization strength and $P$ is the penalty matrix.