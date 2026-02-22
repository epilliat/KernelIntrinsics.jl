using Pkg
Pkg.activate("docs")
using Documenter
using KernelIntrinsics
using TOML

project = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
version = project["version"]
makedocs(
    sitename="KernelIntrinsics.jl",
    version=version,
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

deploydocs(
    repo="github.com/epilliat/KernelIntrinsics.jl",
    devbranch="main",
)