using Documenter
using MemoryAccess

makedocs(
    sitename="MemoryAccess.jl",
    modules=[MemoryAccess],
    checkdocs=:exports,  # Only check exported symbols
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true"
    ),
)

# For GitHub Pages deployment
deploydocs(
    repo="github.com/epilliat/MemoryAccess.jl.git",
    devbranch="main",
)