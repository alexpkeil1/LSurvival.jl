using Documenter, LSurvival, Random, Distributions, LinearAlgebra

#DocMeta.setdocmeta!(LSurvival, :DocTestSetup, :(using LSurvival); recursive = true)

push!(LOAD_PATH, "../src/")

makedocs(;
    format = Documenter.HTML(),
    modules = [LSurvival],
    sitename = "LSurvival: survival analysis for left-truncated, right-censored outcomes",
    pages = [
        "Help" => "index.md", 
        "Parametric likelihood functions" => "Likelihood.md"
        ],
    debug = true,
    doctest = true,
    source = "src",
    build = "build",
    highlightsig = true,
)

deploydocs(;
    repo = "github.com/alexpkeil1/LSurvival.jl.git",
    branch = "gh-pages",
    target = "build",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
)
