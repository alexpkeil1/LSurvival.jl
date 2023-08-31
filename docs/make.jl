using Documenter, LSurvival, Random, Distributions, LinearAlgebra

#DocMeta.setdocmeta!(LSurvival, :DocTestSetup, :(using LSurvival); recursive = true)

makedocs(;
    format = Documenter.HTML(),
    modules = [LSurvival],
    sitename = "LSurvival",
    pages = ["Home" => "index.md"],
    debug = false,
    doctest = true,
    strict = :doctest,
)

deploydocs(;
    repo = "github.com/alexpkeil1/LSurvival.jl.git",
)
