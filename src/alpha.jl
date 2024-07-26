"""
    alpha(y, mu, Dist, Link)
Calculate alpha.

Usage:
```julia-repl
alpha((y, mu, Dist, Link)
```
Arguments:
- `y` : `Vector` containing the response variable.
- `Dist` : Likelihood distribution.
- `Link` : Link function.
"""
function alpha(y, mu, Dist, Link)
    "see page 250 in Wood, 2nd ed"
    return @. 1 + (y - mu) * (
        Dist[:V_Derivative](mu) / Dist[:V](mu) + 
        Link[:Second_Derivative](mu) / Link[:Derivative](mu)
    )
end