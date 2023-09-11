using Documenter, LSurv, Random, Distributions, LinearAlgebra

#DocMeta.setdocmeta!(LSurv, :DocTestSetup, :(using LSurv); recursive = true)

push!(LOAD_PATH,"../src/")

makedocs(;
    format = Documenter.HTML(),
    modules = [LSurv],
    sitename = "LSurv",
    pages = ["Home" => "index.md"],
    debug = true,
    doctest = true,
    strict = :doctest,
    source = "src",
    build = "build"
)

deploydocs(;
    repo = "github.com/alexpkeil1/LSurvival.jl.git",
    branch = "gh-pages",
    target = "build",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
)
