using RDatasets, Documenter, GAM

makedocs(
    format = Documenter.HTML(),
    sitename = "GAM",
    modules = [GAM],
    pages = [
        "Home" => "index.md",
        "examples.md"
    ],
    debug = false,
    doctest = true,
    strict = :doctest,
)

deploydocs(
    repo   = "github.com/hendersontrent/GAM.jl.git",
)