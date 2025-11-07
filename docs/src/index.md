# FastIBPM.jl

Julia CFD package based on the Immersed Boundary Projection Method (IBPM) from [Taira & Colonius (2007)](https://doi.org/10.1016/j.jcp.2007.03.005), [Colonius & Taira (2008)](https://doi.org/10.1016/j.cma.2007.08.014) and [Goza & Colonius (2008)](https://doi.org/10.1016/j.jcp.2017.02.027)

---
## Overview

**FastIBPM.jl** is a high-performance Julia package for simulating incompressible fluid flows with immersed boundaries. It provides a modular and extensible framework for **fluid–structure interaction (FSI)** problems using FFT-based Poisson solvers and strongly-coupled time integration schemes.

Key features include:
- Multilevel grid hierarchy for efficiency.
- Support for static, prescribed, and deformable bodies.
- GPU and multi-threaded support through [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl).


## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/NUFgroup/FastIBPM.jl")
```