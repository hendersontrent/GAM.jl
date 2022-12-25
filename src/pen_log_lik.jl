"""
    pen_log_lik(X, y, β, dist, λ, penalty_matrix)
XX

Usage:
```julia-repl
pen_log_lik(X, y, β, dist, λ, penalty_matrix)
```
Arguments:
- `X` : Data matrix of predictor variables.
- `y` : Response variable vector.
- `β` : Vector of model coefficients.
- `dist` : Llikelihood distribution function.
- `λ` : Coefficient of the penalty term.
- `penalty_matrix` : Matrix of penalised values.
"""
function pen_log_lik(X, y, β, dist, λ, penalty_matrix)
    y_pred = X * β
    y_pred = dist.linkinv.(y_pred)
    log_lik = sum(logpdf.(dist, y, y_pred))
    penalty = β' * penalty_matrix * β
    return -(log_lik - λ * penalty)
end