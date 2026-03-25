using Documenter
using Immersa

makedocs(
    sitename = "Immersa",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    modules = [Immersa],
    checkdocs = :none,
    pages = [
        "Home"          => "index.md",
        "Examples"      => "examples/examples.md",
        "API Reference" => [
            "ArrayPools"         => "API/array_pools.md",
            "Problem Definition" => "API/problems.md",
            "CNAB"               => "API/cnab.md",
            "Prescribed Bodies"  => "API/prescribed_bodies.md",
            "Structural Bodies"  => "API/structural_bodies.md",
            "Operators"          => "API/operators.md",
            "FFT Transforms"     => "API/fft_r2r.md",
            "Utilities"          => "API/utilities.md",
            
        ],
    ],
)

deploydocs(
    repo      = "github.com/NUFgroup/Immersa.jl",
    devbranch = "main",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
