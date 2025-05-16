abstract type GridKind end
struct Primal <: GridKind end
struct Dual <: GridKind end

abstract type GridLocation{K<:GridKind} end
struct Node{K} <: GridLocation{K} end
struct Edge{K} <: GridLocation{K}
    i::Int
end

const Loc_u = Edge{Primal}
const Loc_ω = Edge{Dual}

"""
    Grid(; h, n, x0, levels)

A Cartesian grid with uniform spacing `h`, `n` cells along each dimension, and lower corner
position `x0`.  `n` is rounded up to the next multiple of 4.
"""
@kwdef struct Grid{N,T<:AbstractFloat}
    h::T
    n::SVector{N,Int}
    x0::SVector{N,T}
    levels::Int
    function Grid(h::T, n, x0, levels) where {T}
        let n = @. 4 * cld(n, 4)
            new{length(n),T}(h, n, x0, levels)
        end
    end
end

gridcorner(grid::Grid) = grid.x0
gridcorner((; x0, h, n)::Grid, level::Integer) = x0 + h * n * (1 - 2^(level - 1)) / 2

gridstep(grid::Grid) = grid.h
gridstep(grid::Grid, level::Integer) = grid.h * 2^(level - 1)

function coord(grid::Grid, loc, I::SVector{N,<:Integer}, args...) where {N}
    x0 = gridcorner(grid, args...)
    h = gridstep(grid, args...)
    x0 + h * (I + _cellcoord(loc, Val(N)))
end

coord(grid, loc, I::Tuple, args...) = coord(grid, loc, SVector(I), args...)
coord(grid, loc, I::CartesianIndex, args...) = coord(grid, loc, SVector(Tuple(I)), args...)

_cellcoord((; i)::Edge{Primal}, ::Val{N}) where {N} = SVector(ntuple(≠(i), N)) / 2
_cellcoord((; i)::Edge{Dual}, ::Val{N}) where {N} = SVector(ntuple(==(i), N)) / 2

function coord(grid, loc, r::Tuple{Vararg{AbstractRange}}, args...)
    x1 = coord(grid, loc, first.(r), args...)
    x2 = coord(grid, loc, last.(r), args...)
    ntuple(length(r)) do i
        range(x1[i], x2[i], length(r[i]))
    end
end

struct IncludeBoundary end
struct ExcludeBoundary end

function cell_axes(n::SVector{N}, loc::Edge, ::IncludeBoundary) where {N}
    ntuple(j -> _on_bndry(loc, j) ? (0:n[j]) : (0:n[j]-1), Val(N))
end

function cell_axes(n::SVector{N}, loc::Edge, ::ExcludeBoundary) where {N}
    ntuple(j -> _on_bndry(loc, j) ? (1:n[j]-1) : (0:n[j]-1), Val(N))
end

function cell_axes(n::SVector{N}, loc::Type{<:Edge}, args...) where {N}
    axs = edge_axes(Val(N), loc)
    map(i -> cell_axes(n, loc(i), args...), axs)
end
cell_axes(grid::Grid, args...) = cell_axes(grid.n, args...)

function grid_length(grid::Grid, loc::Edge, args...)
    prod(length, cell_axes(grid, loc, args...))
end

function grid_length(grid::Grid{N}, loc::Type{<:Edge}, args...) where {N}
    axs = edge_axes(Val(N), loc)
    sum(i -> grid_length(grid, loc(i), args...), axs)
end

_on_bndry((; i)::Edge{Primal}, j) = i == j
_on_bndry((; i)::Edge{Dual}, j) = i ≠ j

function boundary_axes(n::SVector{N}, loc::Edge) where {N}
    a = cell_axes(n, loc, IncludeBoundary())
    (SArray ∘ map)(CartesianIndices(SOneTo.((2, N)))) do index
        dir, j = Tuple(index)
        if _on_bndry(loc, j)
            let Iⱼ = (a[j][begin], a[j][end])[dir]
                setindex(a, Iⱼ:Iⱼ, j)
            end
        else
            ntuple(_ -> 1:0, N)
        end
    end
end

function boundary_axes(
    n::SVector{N}, loc::Type{<:Edge}; dims=edge_axes(Val(N), loc)
) where {N}
    map(i -> boundary_axes(n, loc(i)), dims)
end

boundary_axes(grid::Grid, args...; kw...) = boundary_axes(grid.n, args...; kw...)

function boundary_length(grid::Grid, loc::Edge)
    sum(dims -> prod(length, dims), boundary_axes(grid, loc))
end

function boundary_length(grid::Grid{N}, loc::Type{<:Edge}) where {N}
    axs = edge_axes(Val(N), loc)
    sum(i -> boundary_length(grid, loc(i)), axs)
end

function _exclude_boundary(a, grid, loc)
    map(tupleindices(a)) do i
        R = CartesianIndices(
            Base.IdentityUnitRange.(cell_axes(grid, loc(i), ExcludeBoundary()))
        )
        @view a[i][R]
    end
end

edge_axes(::Val{N}, loc::Type{<:Edge}) where {N} = ntuple(identity, N)
edge_axes(::Val{2}, loc::Type{Edge{Dual}}) = OffsetTuple{3}((3,))

function grid_zeros(
    backend, grid::Grid{N,T}, loc::GridLocation, bndry=IncludeBoundary()
) where {N,T}
    R = cell_axes(grid, loc, bndry)
    OffsetArray(KernelAbstractions.zeros(backend, T, length.(R)), R)
end

function grid_zeros(backend, grid::Grid{N}, loc::Type{<:Edge}, args...; levels=1) where {N}
    map(levels) do _
        map(edge_axes(Val(N), loc)) do i
            grid_zeros(backend, grid, loc(i), args...)
        end
    end
end

function boundary_zeros(backend, grid::Grid{N,T}, loc) where {N,T}
    dims = edge_axes(Val(N), loc)
    Rb = boundary_axes(grid, loc; dims)
    map(dims) do i
        (SArray ∘ map)(CartesianIndices(Rb[i])) do index
            dir, j = Tuple(index)
            r = Rb[i][dir, j]
            OffsetArray(KernelAbstractions.zeros(backend, T, length.(r)), r)
        end
    end
end

function grid_view(a, grid, loc, bndry)
    R = cell_axes(grid, loc, bndry)
    map(tupleindices(a)) do i
        r = CartesianIndices(Base.IdentityUnitRange.(R[i]))
        @view a[i][r]
    end
end

"""
    IrrotationalFlow

Specifies a flow velocity with zero discrete curl.
"""
abstract type IrrotationalFlow end

"""
    UniformFlow(u)

A flow with uniform freestream velocity `u(t)`.
"""
struct UniformFlow{U} <: IrrotationalFlow
    u::U
end

function add_flow!(u, flow::UniformFlow, _, _, _, t)
    backend = get_backend(u[1])
    u0 = flow.u(t)
    for i in eachindex(u)
        let u = u[i], u0 = u0[i]
            @loop backend (I in CartesianIndices(u)) u[I] += u0
        end
    end
    u
end

struct BodyPoints{N,T,V<:AbstractVector{SVector{N,T}},S<:AbstractVector{T}}
    x::V
    u::V
    ds::S
end

function BodyPoints{N,T}(backend, n_max) where {N,T}
    x, u = ntuple(2) do _
        KernelAbstractions.zeros(backend, SVector{N,T}, n_max)
    end
    ds = KernelAbstractions.zeros(backend, T, n_max)
    BodyPoints(x, u, ds)
end

function Base.view(points::BodyPoints, r)
    x = view(points.x, r)
    u = view(points.u, r)
    ds = view(points.ds, r)
    BodyPoints(x, u, ds)
end

"""
    AbstractBody

A body that interacts with the fluid.  Bodies specify a set of points, and prescribe the
flow velocity in a small region near each point.
"""
abstract type AbstractBody end

struct IBProblem{N,T,B<:AbstractBody,U<:IrrotationalFlow}
    grid::Grid{N,T}
    body::B
    Re::T
    u0::U
end

function surface_force! end
function surface_force_sum end
