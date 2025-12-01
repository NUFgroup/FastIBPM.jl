using Documenter
using FastIBPM

makedocs(
    sitename = "FastIBPM",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    modules = [FastIBPM],
    checkdocs = :none,
    pages = [
        "Home"          => "index.md",
        "Examples"      => "examples.md",
        "API Reference" => [
            "Problem Definition" => "problem_definition.md",
            "CNAB"    => "cnab.md",
            "Prescribed Bodies"  => "prescribed_bodies.md",
            "Operators"          => "operators.md",
            "FFT Transforms"     => "fft_r2r.md",
            "Utilities"          => "utils.md",
            
        ],
    ],
)

deploydocs(
    repo      = "github.com/NUFgroup/FastIBPM.jl",
    devbranch = "main",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
