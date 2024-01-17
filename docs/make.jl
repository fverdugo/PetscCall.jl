using PETSC
using Documenter

DocMeta.setdocmeta!(PETSC, :DocTestSetup, :(using PETSC); recursive=true)

makedocs(;
    modules=[PETSC],
    authors="Francesc Verdugo <f.verdugo.rojano@vu.nl> and contributors",
    repo="https://github.com/fverdugo/PETSC.jl/blob/{commit}{path}#{line}",
    sitename="PETSC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fverdugo.github.io/PETSC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fverdugo/PETSC.jl",
    devbranch="main",
)
