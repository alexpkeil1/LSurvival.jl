using LSurvival
using Documenter

DocMeta.setdocmeta!(LSurvival, :DocTestSetup, :(using LSurvival); recursive=true)

makedocs(;
    modules=[LSurvival],
    authors="Alex Keil <alex.keil@nih.gov>",
    repo="https://github.com/alexpkeil1/LSurvival.jl/blob/{commit}{path}#{line}",
    sitename="LSurvival.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://alexpkeil1.github.io/LSurvival.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/alexpkeil1/LSurvival.jl",
    devbranch="main",
)
