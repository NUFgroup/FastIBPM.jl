"""
    Immersa

Main module of the **Immersa** package for high-performance immersed boundary simulations.

This module defines the core namespace of the package and integrates all components required for fluid–structure interaction problems. It loads external dependencies, includes internal source files, and exposes the primary API for setting up and solving immersed boundary formulations. The implementation emphasizes numerical efficiency through FFT-based solvers, multilevel grid operations, and GPU or parallel execution via `KernelAbstractions`.

# Arguments
None.

# Returns
The `Immersa` module namespace, providing access to the main data structures, solvers, and utility functions for immersed boundary simulations.
"""
module Immersa

# ------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------
# Linear algebra, static arrays, and offset arrays for numerical efficiency.
# KernelAbstractions enables GPU and multi-threaded parallelism.
# LinearMaps and IterativeSolvers provide efficient iterative methods.
# FFTW is used for spectral solvers and transforms.
# FunctionWrappers and Adapt support dynamic and GPU-compatible function calls.
# EllipsisNotation simplifies array slicing and broadcasting syntax.

using LinearAlgebra
using StaticArrays
using StaticArrays: SOneTo
using OffsetArrays
using OffsetArrays: no_offset_view
using KernelAbstractions
using EllipsisNotation
using LinearMaps
using IterativeSolvers
using FunctionWrappers: FunctionWrapper
import Adapt
import FFTW

# ------------------------------------------------------------------------
# Public API Exports
# ------------------------------------------------------------------------
# Core grid types and locations
export GridKind, Primal, Dual
export GridLocation, Node, Edge, Loc_u, Loc_ω, Loc_p
export Grid, gridcorner, gridstep, coord, cell_axes, boundary_axes, grid_zeros
export IncludeBoundary, ExcludeBoundary

# Flow field models
export IrrotationalFlow, UniformFlow

# Body dynamics and structure types
export AbstractBody, AbstractPrescribedBody, StaticBody, MovingBody, GeometricNonlinearBody
export StructureBC

# Immersed boundary problem setup and solvers
export AbstractFormulation, FastIBPM
export IBProblem
export set_time!,
    step!, initialize_fields!, zero_vorticity!, zero_velocity!, zero_pressure!,
    apply_vorticity!, surface_force!, surface_force_sum

# Time integration and diagnostics
export CNAB
export log_timestep

# ------------------------------------------------------------------------
# Internal Source Files
# ------------------------------------------------------------------------
# Each file defines a subsystem of the Immersa package. They are included
# here to assemble the full immersed boundary solver framework.

# FFT-based real-to-real transforms and Poisson solvers
include("utils/fft_r2r.jl")
using .fft_r2r

include("utils/offset_tuples.jl")
using .offset_tuples

# General-purpose numerical and array utilities
include("utils/utilities.jl")
using .utilities

include("utils/array_pools.jl")
using .array_pools

# Eulerian (fluid) grid: staggered grid types, coordinates, index helpers, allocators
include("fluid_domain/eulerian_grid.jl")

# Lagrangian (body) grid: AbstractBody root type and BodyPoints marker container
include("body_domain/lagrangian_grid.jl")

# Models for prescribed (kinematically constrained) bodies
include("body_domain/body_ops/prescribed_bodies.jl")

# Models for deformable bodies
include("body_domain/body_ops/structural_bodies.jl")

# Fluid-domain operators: kinematic (rot/curl/nonlinear), spectral Laplacian, and
# multidomain multigrid Poisson solver.
include("fluid_domain/fluid_ops/kinematic_ops.jl")
include("fluid_domain/fluid_ops/laplacian_solver.jl")
include("fluid_domain/fluid_ops/multi_domain.jl")

# Interface coupling (regularization machinery): Reg struct, delta functions, E/Eᵀ operators.
# Included BEFORE init/problems.jl because CNAB{..., R<:Reg, ...} requires Reg defined first.
include("interface-coupling/interface_coupling.jl")

# Problem definition + CNAB integrator + initialization routines.
include("init/problems.jl")

# Formulation-specific solver state (FastIBPMState / IBPMState), its construction
# (formulation_state) and reset (initialize_fields!). AFTER problems.jl, since the
# reset routines dispatch on CNAB.
include("init/state.jl")

# Interface coupling (force redistribution): _f_tilde_factor, f_to_f_tilde!, redist!,
# update_redist_weights!. These dispatch on CNAB, so they must come AFTER init/problems.jl.
include("interface-coupling/force_redistribution.jl")

# Assembly operators (formulation-dispatched): viscous inverse (Ainv / Ainv!) and
# body-coupling operators (B_*). Included before cnab.jl, which calls them.
include("time_stepping/assembly_ops.jl")

# CNAB time-stepping: per-iteration routines (step!, prediction, coupling, projection, velocity recovery)
include("time_stepping/cnab.jl")

# Post-processing: checkpoint I/O and surface force extraction
include("post_proc/post_proc.jl")

"""
    load!(filename::AbstractString, x)

Load the state of object `x` from a file on disk.

This function opens the file specified by `filename` and delegates the actual loading to a user-defined `load!` method for the object `x`. It acts as a convenient wrapper to handle file I/O while preserving Julia's multiple dispatch semantics.

# Arguments
- `filename::AbstractString` : Path to the file to load from.
- `x`                       : Object to populate with the loaded data.

# Returns
- The updated object `x` with its state loaded from the file.
"""
function load!(filename::AbstractString, x)
    open(filename) do file
        load!(file, x)
    end
end

"""
    save(filename::AbstractString, x)

Save the state of object `x` to a file on disk.

This function opens the file specified by `filename` for writing and delegates the actual saving to a user-defined `save` method for the object `x`. It serves as a wrapper to manage file I/O while preserving multiple dispatch.

# Arguments
- `filename::AbstractString` : Path to the file to write to.
- `x`                       : Object whose state will be saved.

# Returns
- Nothing. The function writes the object's state to disk.
"""
function save(filename::AbstractString, x)
    open(filename, "w") do file
        save(file, x)
    end
end

end

