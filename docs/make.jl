using Documenter
using FastIBPM

makedocs(
    sitename = "FastIBPM",
    format = Documenter.HTML(),
    modules = [FastIBPM],
    checkdocs = :none
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
