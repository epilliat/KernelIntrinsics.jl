using Documenter
using KernelIntrinsics

makedocs(
    sitename="KernelIntrinsics.jl",
    modules=[KernelIntrinsics],
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
    repo="github.com/epilliat/KernelIntrinsics.jl.git",
    devbranch="main",
)