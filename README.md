# Immersa

*Immersa.jl* is a Julia package that implements immersed boundary methods for the numerical solution of strongly coupled fluid–structure interaction (FSI) problems governed by the incompressible Navier–Stokes equations.

The package includes implementations of:
- The immersed boundary method of [Colonius & Taira (2008)](https://doi.org/10.1016/j.cma.2007.08.014)
- The strongly coupled FSI formulation of [Goza & Colonius (2017)](https://doi.org/10.1016/j.jcp.2017.02.027)

An implementation of the primitive-variable formulation proposed by [Kunihiko & Colonius (2007)](https://doi.org/10.1016/j.jcp.2007.03.005) is currently in progress.

Prescribed structural deformations are supported in both 2D and 3D flows. Strongly coupled FSI is currently limited to 2D configurations. The solver uses far-field boundary conditions and represents immersed structures as point clouds.

The code uses [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/) for parallel execution on both CPUs and GPUs. It also supports automatic differentiation via [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/) to compute sensitivities of solver outputs.

## Installation
Install the latest version of Julia, then start it and and add the *Immersa* package:

```julia
using Pkg
Pkg.add(url="https://github.com/NUFgroup/Immersa.jl")
using Immersa
```

To run bundled examples from a fresh temporary environment without cloning:

```bash
julia --project=@. -e 'using Pkg; Pkg.add(url="https://github.com/NUFgroup/Immersa.jl")'
julia --project=@. examples/<example_name>.jl
```

To run examples from the REPL after adding the Immersa package:

```julia
using Immersa
include(joinpath(pkgdir(Immersa), "examples", "<example_name>.jl"))
```

## Running Examples
- **Cylinder:** non-deforming cylinder in a steady-state freestream flow.
- **Flag:** flag immersed in a steady-state freestream flow (FSI).
- **Heaving cylinder:** 
- **Naca0012:**
- **Oscillating plate:**
- **Plate:**
- **Plate actuation:**

## Quick Start
Run the cylinder simulation example included in the repository:

```bash
julia --project=@. examples/cylinder.jl
```
Or from the Julia REPL:
```julia
using Immersa
include(joinpath(pkgdir(Immersa), "examples", "cylinder.jl"))
```
This will run the simulation and save results (vorticity, lift, and drag) in HDF5 and CGNS format files. You can then visualize the outputs using the plotting scripts provided, or importing the CGNS files into [ParaView](https://www.paraview.org/).

## How to Contribute


## References
- Colonius, T., & Taira, K. (2008). A fast immersed boundary method using a nullspace approach and multi-domain far-field boundary conditions.
- Goza, A., & Colonius, T. (2017). A strongly-coupled immersed-boundary method for thin elastic structures.
- Kunihiko, T., & Colonius, T. (2007). An immersed boundary method for incompressible flows using primitive variables.
