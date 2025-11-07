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
        "API Reference" => "api.md",     # <- add this
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
