function pen_log_lik(X, y, beta, dist, lambda, penalty_matrix)
    y_pred = X * beta
    y_pred = dist.linkinv.(y_pred)
    log_lik = sum(logpdf.(dist, y, y_pred))
    penalty = beta' * penalty_matrix * beta
    return -(log_lik - lambda * penalty)
end