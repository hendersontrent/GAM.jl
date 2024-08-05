identity(x) = x
logit(p) = log(p/(1-p))
expit(x) = 1/(1+exp(-x))

Dist_Map = Dict(
    "Normal" => :Normal,
    "Poisson" => :Poisson,
    "Gamma" => :Gamma
)

Link_Map = Dict(
    "Identity" => :Identity,
    "Log" => :Log,
)

Links = Dict(
    :Identity => Dict(
        :Name => "Identity",
        :Function => identity,
        :Inverse => identity,
        :Derivative => (x -> 1),
        :Second_Derivative => (x -> 0)
    ),
    :Log => Dict(
        :Name => "Log",
        :Function => log,
        :Inverse => exp,
        :Derivative => (x -> 1/x),
        :Second_Derivative => (x -> -1/(x^2))
    )
)

Dists = Dict(
    :Normal => Dict(
        :Name => "Normal",
        :Distribution => Normal,
        :V => (mu -> 1),
        :V_Derivative => (mu -> 0),
        :Link => Links[:Identity]
    ),
    :Gamma => Dict(
        :Name => "Gamma",
        :Distribution => Gamma,
        :V => (mu -> mu^2),
        :V_Derivative => (mu -> 2*mu),
        :Link => Links[:Log]
    ),
    :Poisson => Dict(
        :Name => "Poisson",
        :Distribution => Poisson,
        :V => (mu -> mu),
        :V_Derivative => (mu -> 1),
        :Link => Links[:Log]
    )
)
