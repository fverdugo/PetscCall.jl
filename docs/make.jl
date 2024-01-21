using PETSC
using Documenter

DocMeta.setdocmeta!(PETSC, :DocTestSetup, :(using PETSC); recursive=true)

makedocs(;
    modules=[PETSC],
    sitename="PETSC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fverdugo.github.io/PETSC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Usage" => "basic.md",
        "Advanced API" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/fverdugo/PETSC.jl",
    devbranch="main",
)
