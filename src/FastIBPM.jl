"""
    FastIBPM

Main module of the **FastIBPM** package for high-performance immersed boundary simulations.

This module defines the core namespace of the package and integrates all components required for fluid–structure interaction problems. It loads external dependencies, includes internal source files, and exposes the primary API for setting up and solving immersed boundary formulations. The implementation emphasizes numerical efficiency through FFT-based solvers, multilevel grid operations, and GPU or parallel execution via `KernelAbstractions`.

# Arguments
None.

# Returns
The `FastIBPM` module namespace, providing access to the main data structures, solvers, and utility functions for immersed boundary simulations.
"""
module FastIBPM

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
export GridLocation, Node, Edge, Loc_u, Loc_ω
export Grid, gridcorner, gridstep, coord, cell_axes, boundary_axes, grid_zeros
export IncludeBoundary, ExcludeBoundary

# Flow field models
export IrrotationalFlow, UniformFlow

# Body dynamics and structure types
export AbstractBody, AbstractPrescribedBody, StaticBody, GeometricNonlinearBody
export StructureBC

# Immersed boundary problem setup and solvers
export IBProblem
export set_time!,
    step!, zero_vorticity!, apply_vorticity!, surface_force!, surface_force_sum

# Time integration and diagnostics
export CNAB
export log_timestep


# ------------------------------------------------------------------------
# Internal Source Files
# ------------------------------------------------------------------------
# Each file defines a subsystem of the FastIBPM package. They are included
# here to assemble the full immersed boundary solver framework.

# FFT-based real-to-real transforms and Poisson solvers
include("FFT_R2R.jl")

# General-purpose numerical and array utilities
include("utils.jl")

# Problem setup and initialization routines
include("problems.jl")

# Models for prescribed (kinematically constrained) bodies
include("prescribed-bodies.jl")

# Models for deformable bodies
include("structural-bodies.jl")

# Discrete differential operators and grid mappings
include("operators.jl")

#CNAB time integration scheme implementation
include("cnab.jl")


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
