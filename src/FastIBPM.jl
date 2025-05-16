module FastIBPM

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

export GridKind, Primal, Dual
export GridLocation, Node, Edge, Loc_u, Loc_ω
export Grid, gridcorner, gridstep, coord, cell_axes, boundary_axes, grid_zeros
export IncludeBoundary, ExcludeBoundary
export IrrotationalFlow, UniformFlow
export AbstractBody, AbstractPrescribedBody, StaticBody, GeometricNonlinearBody
export StructureBC
export IBProblem
export set_time!,
    step!, zero_vorticity!, apply_vorticity!, surface_force!, surface_force_sum
export CNAB

include("FFT_R2R.jl")
include("utils.jl")
include("problems.jl")
include("prescribed-bodies.jl")
include("structural-bodies.jl")
include("operators.jl")
include("cnab.jl")

function load!(filename::AbstractString, x)
    open(filename) do file
        load!(file, x)
    end
end

function save(filename::AbstractString, x)
    open(filename, "w") do file
        save(file, x)
    end
end

end
