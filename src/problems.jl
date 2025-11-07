"""
    abstract type GridKind end

Abstract type used to distinguish between different kinds of staggered grids.
See `Primal` and `Dual`.
"""
abstract type GridKind end

"""
    struct Primal <: GridKind end

A `GridKind` tag for the main grid, typically used for primary variables
(e.g., velocity).
"""
struct Primal <: GridKind end

"""
    struct Dual <: GridKind end

A `GridKind` tag for the offset (dual) grid, often used for derived
quantities (e.g., vorticity, pressure).
"""
struct Dual <: GridKind end

"""
    abstract type GridLocation{K<:GridKind} end

An abstract type representing where a variable is stored on a grid (e.g., node, cell, edge).
It is parameterized by the `GridKind` `K` (either `Primal` or `Dual`).
"""
abstract type GridLocation{K<:GridKind} end

"""
    struct Node{K} <: GridLocation{K} end

A `GridLocation` type indicating that a variable lives at a node of a grid of kind `K`.
"""
struct Node{K} <: GridLocation{K} end

"""
    struct Edge{K} <: GridLocation{K}
        i::Int
    end

A `GridLocation` type indicating that a variable lives on an edge of a grid of kind `K`.

# Fields
- `i::Int`: Specifies the direction of the edge (e.g., 1 for x, 2 for y).
"""
struct Edge{K} <: GridLocation{K}
    i::Int
end

"""
    const Loc_u = Edge{Primal}

Type alias for velocity, stored on the edges of the `Primal` grid.
"""
const Loc_u = Edge{Primal}

"""
    const Loc_ω = Edge{Dual}

Type alias for vorticity, stored on the edges of the `Dual` grid.
"""
const Loc_ω = Edge{Dual}

"""
    struct Grid{N,T<:AbstractFloat}
        h::T
        n::SVector{N,Int}
        x0::SVector{N,T}
        levels::Int
    end

Defines a multi-resolution, uniform Cartesian grid.

# Fields
- `h::T`: Grid spacing (a scalar float, e.g., 0.01).
- `n::SVector{N,Int}`: Number of grid cells in each spatial dimension `N`.
- `x0::SVector{N,T}`: Position of the bottom-left (origin) corner of the grid.
- `levels::Int`: Number of grid levels for multigrid.

# Constructor
    Grid(h::T, n, x0, levels)

Creates a `Grid`. The number of cells `n` in each dimension is automatically
rounded up to the nearest multiple of 4 to ensure compatibility with
certain solvers (e.g., FFTs or multigrid coarsening).
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

"""
    gridcorner(grid::Grid)

Returns the corner position (origin) of the base grid (`grid.x0`).

# Arguments
- `grid::Grid`: The grid object.

# Returns
- `SVector`: The physical coordinates of the grid's corner.
"""
gridcorner(grid::Grid) = grid.x0

"""
    gridcorner(grid::Grid, level::Integer)

Computes the corner position (origin) of a coarser multigrid level,
which is centered relative to the fine grid.

# Arguments
- `grid::Grid`: The grid object.
- `level::Integer`: The multigrid level.

# Returns
- `SVector`: The physical coordinates of the coarse grid's corner.
"""
gridcorner((; x0, h, n)::Grid, level::Integer) = x0 + h * n * (1 - 2^(level - 1)) / 2

"""
    gridstep(grid::Grid)

Returns the grid spacing (e.g., Δx) for the base grid (`grid.h`).

# Arguments
- `grid::Grid`: The grid object.

# Returns
- `T`: The base grid spacing.
"""
gridstep(grid::Grid) = grid.h

"""
    gridstep(grid::Grid, level::Integer)

Computes the grid spacing for a coarser multigrid level.
Each level doubles the spacing (`grid.h * 2^(level - 1)`).

# Arguments
- `grid::Grid`: The grid object.
- `level::Integer`: The multigrid level.

# Returns
- `T`: The coarse grid spacing.
"""
gridstep(grid::Grid, level::Integer) = grid.h * 2^(level - 1)

"""
    coord(grid::Grid, loc, I::SVector{N,<:Integer}, args...)

Computes the physical coordinates for a given grid index `I` and `GridLocation` `loc`.
This is the core logic that accounts for staggered grid offsets.

# Arguments
- `grid::Grid`: The grid object.
- `loc::GridLocation`: The location type (e.g., `Node()`, `Edge{Primal}(1)`).
- `I::SVector`: The integer grid index.
- `args...`: Optional multigrid level.

# Returns
- `SVector`: The physical coordinates `x(I) = x0 + h * (I + offset)`.
"""
function coord(grid::Grid, loc, I::SVector{N,<:Integer}, args...) where {N}
    x0 = gridcorner(grid, args...)
    h = gridstep(grid, args...)
    x0 + h * (I + _cellcoord(loc, Val(N)))
end

"""
    coord(grid, loc, I::Tuple, args...)

Convenience method for `coord` that accepts indices as a `Tuple`.
"""
coord(grid, loc, I::Tuple, args...) = coord(grid, loc, SVector(I), args...)

"""
    coord(grid, loc, I::CartesianIndex, args...)

Convenience method for `coord` that accepts indices as a `CartesianIndex`.
"""
coord(grid, loc, I::CartesianIndex, args...) = coord(grid, loc, SVector(Tuple(I)), args...)

"""
    coord(grid, loc, r::Tuple{Vararg{AbstractRange}}, args...)

Computes the physical coordinate ranges corresponding to a block of grid indices.

# Arguments
- `grid`: The grid object.
- `loc`: The grid location.
- `r::Tuple{Vararg{AbstractRange}}`: A tuple of index ranges.
- `args...`: Optional multigrid level.

# Returns
- `Tuple`: A tuple of physical coordinate ranges (e.g., `(xrange, yrange)`).
"""
function coord(grid, loc, r::Tuple{Vararg{AbstractRange}}, args...)
    x1 = coord(grid, loc, first.(r), args...)
    x2 = coord(grid, loc, last.(r), args...)
    ntuple(length(r)) do i
        range(x1[i], x2[i], length(r[i]))
    end
end

"""
    _cellcoord(loc::Edge{Primal}, ::Val{N})

Computes the fractional cell offset for a `Primal` edge (e.g., velocity).
This is a half-cell offset in directions *other* than the edge's direction `i`.

# Arguments
- `loc::Edge{Primal}`: The primal edge location.
- `::Val{N}`: The grid dimension.

# Returns
- `SVector`: A vector of fractional offsets (e.g., `[0.0, 0.5]`).
"""
_cellcoord((; i)::Edge{Primal}, ::Val{N}) where {N} = SVector(ntuple(≠(i), N)) / 2

"""
    _cellcoord(loc::Edge{Dual}, ::Val{N})

Computes the fractional cell offset for a `Dual` edge (e.g., vorticity).
This is a half-cell offset *along* the edge's direction `i`.

# Arguments
- `loc::Edge{Dual}`: The dual edge location.
- `::Val{N}`: The grid dimension.

# Returns
- `SVector`: A vector of fractional offsets (e.g., `[0.5, 0.0]`).
"""
_cellcoord((; i)::Edge{Dual}, ::Val{N}) where {N} = SVector(ntuple(==(i), N)) / 2


"""
    struct IncludeBoundary end

Tag type used as a flag to request index ranges that *include* boundary points.
"""
struct IncludeBoundary end

"""
    struct ExcludeBoundary end

Tag type used as a flag to request index ranges that *exclude* boundary points,
returning only the grid interior.
"""
struct ExcludeBoundary end

"""
    cell_axes(n::SVector{N}, loc::Edge, ::IncludeBoundary)

Determines the iterable index ranges for a grid location, *including* boundaries.

# Arguments
- `n::SVector{N}`: Number of grid cells per dimension.
- `loc::Edge`: The grid location.
- `::IncludeBoundary`: Tag specifying boundary inclusion.

# Returns
- `Tuple` of `UnitRange`: Index ranges (e.g., `(0:n[1], 0:n[2]-1)`).
"""
function cell_axes(n::SVector{N}, loc::Edge, ::IncludeBoundary) where {N}
    ntuple(j -> _on_bndry(loc, j) ? (0:n[j]) : (0:n[j]-1), Val(N))
end

"""
    cell_axes(n::SVector{N}, loc::Edge, ::ExcludeBoundary)

Determines the iterable index ranges for a grid location, *excluding* boundaries.

# Arguments
- `n::SVector{N}`: Number of grid cells per dimension.
- `loc::Edge`: The grid location.
- `::ExcludeBoundary`: Tag specifying boundary exclusion.

# Returns
- `Tuple` of `UnitRange`: Index ranges for the interior (e.g., `(1:n[1]-1, 0:n[2]-1)`).
"""
function cell_axes(n::SVector{N}, loc::Edge, ::ExcludeBoundary) where {N}
    ntuple(j -> _on_bndry(loc, j) ? (1:n[j]-1) : (0:n[j]-1), Val(N))
end

"""
    cell_axes(n::SVector{N}, loc::Type{<:Edge}, args...)

Vectorized method for `cell_axes`. Returns a `map` of axes for all
possible edge directions of a given `Edge` type.

# Arguments
- `n::SVector{N}`: Number of grid cells per dimension.
- `loc::Type{<:Edge}`: The edge type (e.g., `Loc_u`).
- `args...`: Boundary flags (`IncludeBoundary` or `ExcludeBoundary`).

# Returns
- A `map` or `OffsetTuple` containing the axes for each component.
"""
function cell_axes(n::SVector{N}, loc::Type{<:Edge}, args...) where {N}
    axs = edge_axes(Val(N), loc)
    map(i -> cell_axes(n, loc(i), args...), axs)
end

"""
    cell_axes(grid::Grid, args...)

Convenience method for `cell_axes` that extracts the cell count `n` from the `Grid` object.
"""
cell_axes(grid::Grid, args...) = cell_axes(grid.n, args...)

"""
    grid_length(grid::Grid, loc::Edge, args...)

Computes the total number of grid points for a *single* edge-centered component.

# Arguments
- `grid::Grid`: The grid object.
- `loc::Edge`: The specific edge component (e.g., `Loc_u(1)`).
- `args...`: Boundary flags (`IncludeBoundary` or `ExcludeBoundary`).

# Returns
- `Int`: The total number of points (`prod(length, cell_axes(...)`).
"""
function grid_length(grid::Grid, loc::Edge, args...)
    prod(length, cell_axes(grid, loc, args...))
end

"""
    grid_length(grid::Grid{N}, loc::Type{<:Edge}, args...)

Computes the total number of grid points summed over *all* components
of a given `Edge` type.

# Arguments
- `grid::Grid`: The grid object.
- `loc::Type{<:Edge}`: The edge type (e.g., `Loc_u`).
- `args...`: Boundary flags.

# Returns
- `Int`: The *sum* of points over all components.
"""
function grid_length(grid::Grid{N}, loc::Type{<:Edge}, args...) where {N}
    axs = edge_axes(Val(N), loc)
    sum(i -> grid_length(grid, loc(i), args...), axs)
end

"""
    _on_bndry(loc::Edge{Primal}, j)

Utility function. Returns `true` if a `Primal` edge (like velocity)
is defined on the boundary in direction `j`.

# Arguments
- `loc::Edge{Primal}`: The primal edge location.
- `j::Int`: The coordinate direction to check.

# Returns
- `Bool`: `true` if `loc.i == j`, `false` otherwise.
"""
_on_bndry((; i)::Edge{Primal}, j) = i == j

"""
    _on_bndry(loc::Edge{Dual}, j)

Utility function. Returns `true` if a `Dual` edge (like vorticity)
is defined on the boundary in direction `j`.

# Arguments
- `loc::Edge{Dual}`: The dual edge location.
- `j::Int`: The coordinate direction to check.

# Returns
- `Bool`: `true` if `loc.i != j`, `false` otherwise.
"""
_on_bndry((; i)::Edge{Dual}, j) = i ≠ j

"""
    boundary_axes(n::SVector{N}, loc::Edge)

Returns ranges of grid indices that lie exactly on the boundaries of a
*single* edge-defined field component.

# Arguments
- `n::SVector{N}`: Number of grid cells per dimension.
- `loc::Edge`: A specific edge component (e.g., `Loc_u(1)`).

# Returns
- `SArray`: An array of index ranges for each boundary face (e.g., left, right, top, bottom).
"""
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

"""
    boundary_axes(n::SVector{N}, loc::Type{<:Edge}; dims=...)

Applies `boundary_axes` to all edge directions for a vector-valued field,
returning a list of boundary index ranges for each component.

# Arguments
- `n::SVector{N}`: Number of grid cells per dimension.
- `loc::Type{<:Edge}`: An edge type (e.g., `Loc_u`).
- `dims`: The dimensions to iterate over.

# Returns
- A `map` of results from the single-component `boundary_axes` method.
"""
function boundary_axes(
    n::SVector{N}, loc::Type{<:Edge}; dims=edge_axes(Val(N), loc)
) where {N}
    map(i -> boundary_axes(n, loc(i)), dims)
end

"""
    boundary_axes(grid::Grid, args...; kw...)

Convenience method for `boundary_axes` that extracts the cell count `n`
from the `Grid` object.
"""
boundary_axes(grid::Grid, args...; kw...) = boundary_axes(grid.n, args...; kw...)

"""
    boundary_length(grid::Grid, loc::Edge)

Computes the total number of degrees of freedom (grid points) located
exactly on the boundaries for a *single* staggered field component.

# Arguments
- `grid::Grid`: The grid object.
- `loc::Edge`: A specific edge component.

# Returns
- `Int`: Total number of boundary points for this component.
"""
function boundary_length(grid::Grid, loc::Edge)
    sum(dims -> prod(length, dims), boundary_axes(grid, loc))
end

"""
    boundary_length(grid::Grid{N}, loc::Type{<:Edge})

Computes the total number of boundary degrees of freedom summed over
*all* components of a given `Edge` type.

# Arguments
- `grid::Grid`: The grid object.
- `loc::Type{<:Edge}`: An edge type (e.g., `Loc_u`).

# Returns
- `Int`: *Sum* of boundary points over all components.
"""
function boundary_length(grid::Grid{N}, loc::Type{<:Edge}) where {N}
    axs = edge_axes(Val(N), loc)
    sum(i -> boundary_length(grid, loc(i)), axs)
end

"""
    _exclude_boundary(a, grid, loc)

Returns a collection of "interior" views of an array `a`, excluding boundary points.

# Arguments
- `a`: An array or tuple of arrays (one for each field component).
- `grid::Grid`: The grid object.
- `loc`: The grid location type (e.g., `Loc_u`).

# Returns
- A `Tuple` or `Array` of non-allocating views (`@view`) into the
  interior region of each component array in `a`.
"""
function _exclude_boundary(a, grid, loc)
    map(tupleindices(a)) do i
        R = CartesianIndices(
            Base.IdentityUnitRange.(cell_axes(grid, loc(i), ExcludeBoundary()))
        )
        @view a[i][R]
    end
end

"""
    edge_axes(::Val{N}, loc::Type{<:Edge})

Provides the list of possible directions (axes) associated with edges on a
grid of dimension `N`.

# Arguments
- `::Val{N}`: The grid dimension.
- `loc::Type{<:Edge}`: The edge type.

# Returns
- A tuple of directions, e.g., `(1, 2)` for `N=2`.
"""
edge_axes(::Val{N}, loc::Type{<:Edge}) where {N} = ntuple(identity, N)

"""
    edge_axes(::Val{2}, loc::Type{Edge{Dual}})

Provides the list of axes for a 2D `Dual` edge.
This is a special case, likely for 2D vorticity, which returns `OffsetTuple{3}((3,))`
to represent the single z-component.

# Arguments
- `::Val{2}`: The grid dimension (specifically 2).
- `loc::Type{Edge{Dual}}`: The Dual edge type.

# Returns
- `OffsetTuple{3}((3,))`
"""
edge_axes(::Val{2}, loc::Type{Edge{Dual}}) = OffsetTuple{3}((3,))

"""
    grid_zeros(backend, grid, loc::GridLocation, bndry=IncludeBoundary())

Creates a single array of zeros, correctly sized and indexed for a
given grid location, allocated on the specified backend.

# Arguments
- `backend`: The `KernelAbstractions` backend (e.g., `CPU()`).
- `grid::Grid`: The grid object.
- `loc::GridLocation`: The specific grid location (e.g., `Loc_u(1)`).
- `bndry`: Boundary flag (default: `IncludeBoundary()`).

# Returns
- A single `OffsetArray` of zeros, with indices matching `cell_axes`.
"""
function grid_zeros(
    backend, grid::Grid{N,T}, loc::GridLocation, bndry=IncludeBoundary()
) where {N,T}
    R = cell_axes(grid, loc, bndry)
    OffsetArray(KernelAbstractions.zeros(backend, T, length.(R)), R)
end

"""
    grid_zeros(backend, grid, loc::Type{<:Edge}, args...; levels=1)

Creates a structure of zero-filled arrays for all components of an `Edge` type,
potentially replicated for multiple multigrid `levels`.

# Arguments
- `backend`: The `KernelAbstractions` backend.
- `grid::Grid`: The grid object.
- `loc::Type{<:Edge}`: The edge type (e.g., `Loc_u`).
- `args...`: Boundary flags.
- `levels::Int`: Number of multigrid levels (default: 1).

# Returns
- A `map` (or `OffsetTuple`) of arrays for all components, potentially nested
  inside another `map` if `levels > 1`.
"""
function grid_zeros(backend, grid::Grid{N}, loc::Type{<:Edge}, args...; levels=1) where {N}
    map(levels) do _
        map(edge_axes(Val(N), loc)) do i
            grid_zeros(backend, grid, loc(i), args...)
        end
    end
end

"""
    boundary_zeros(backend, grid::Grid{N,T}, loc)

Returns a nested structure of zero-filled arrays to store values on the
grid boundaries. These arrays are correctly indexed `OffsetArray`s
allocated on the specified backend.

# Arguments
- `backend`: The `KernelAbstractions` backend.
- `grid::Grid`: The grid object.
- `loc`: The grid location type (e.g., `Loc_u`).

# Returns
- A nested structure: `map` over components -> `SArray` over faces -> `OffsetArray`
  of zeros for that specific boundary face.
"""
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

"""
    grid_view(a, grid, loc, bndry)

Creates views of grid data arrays, sliced according to the active region
defined by `cell_axes`.

# Arguments
- `a`: An array or tuple of arrays (one for each field component).
- `grid::Grid`: The grid object.
- `loc`: The grid location type (e.g., `Loc_u`).
- `bndry`: Boundary flag (`IncludeBoundary()` or `ExcludeBoundary()`).

# Returns
- A `map` of non-allocating views (`@view`) into the active region of each
  component array in `a`.
"""
function grid_view(a, grid, loc, bndry)
    R = cell_axes(grid, loc, bndry)
    map(tupleindices(a)) do i
        r = CartesianIndices(Base.IdentityUnitRange.(R[i]))
        @view a[i][r]
    end
end

"""
    IrrotationalFlow

Abstract type specifying a flow where the curl of the velocity is zero
(∇ × u = 0). This is a "tag" type used for dispatch.
"""
abstract type IrrotationalFlow end

"""
    UniformFlow(u)

A type of `IrrotationalFlow` representing a flow that is uniform in space
but may vary in time.

# Fields
- `u`: The freestream velocity. Can be a constant `SVector` or a
  function `u(t)` that returns an `SVector`.
"""
struct UniformFlow{U} <: IrrotationalFlow
    u::U
end

"""
    add_flow!(u, flow::UniformFlow, _, _, _, t)

Adds the background `UniformFlow` velocity to the velocity field `u` in-place.
Computes `u_new = u_old + u_flow(t)`.

# Arguments
- `u`: The velocity field (e.g., a tuple of arrays). Modified in-place.
- `flow::UniformFlow`: The background flow object.
- `_`: Placeholders for unused arguments.
- `t`: The current simulation time.

# Returns
- The modified velocity field `u`.
"""
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

"""
    struct BodyPoints{N,T,V<:AbstractVector{SVector{N,T}},S<:AbstractVector{T}}
        x::V
        u::V
        ds::S
    end

An "immersed object container" that holds the Lagrangian body points and
their attributes.

# Fields
- `x`: Vector of body point positions (each an `SVector{N,T}`).
- `u`: Vector of body point velocities (each an `SVector{N,T}`).
- `ds`: Vector of quadrature weights (e.g., line segment lengths) for each point.
"""
struct BodyPoints{N,T,V<:AbstractVector{SVector{N,T}},S<:AbstractVector{T}}
    x::V
    u::V
    ds::S
end

"""
    BodyPoints{N,T}(backend, n_max)

Constructs a `BodyPoints` instance by pre-allocating memory for `n_max` points
on the specified `backend`.

# Arguments
- `backend`: The `KernelAbstractions` backend.
- `n_max::Int`: The maximum number of points to allocate for.

# Returns
- `BodyPoints`: A new `BodyPoints` object with `x`, `u`, and `ds` fields
  initialized as zero-filled arrays on the backend.
"""
function BodyPoints{N,T}(backend, n_max) where {N,T}
    x, u = ntuple(2) do _
        KernelAbstractions.zeros(backend, SVector{N,T}, n_max)
    end
    ds = KernelAbstractions.zeros(backend, T, n_max)
    BodyPoints(x, u, ds)
end

"""
    Base.view(points::BodyPoints, r)

Overloads `Base.view` to create a lightweight, non-allocating "slice"
or "window" of a `BodyPoints` object.

# Arguments
- `points::BodyPoints`: The original `BodyPoints` object.
- `r`: An index range (e.g., `201:400`).

# Returns
- `BodyPoints`: A new `BodyPoints` object whose fields (`x`, `u`, `ds`) are
  `view`s into the fields of the original object.
"""
function Base.view(points::BodyPoints, r)
    x = view(points.x, r)
    u = view(points.u, r)
    ds = view(points.ds, r)
    BodyPoints(x, u, ds)
end

"""
    AbstractBody

An abstract type defining the interface for a body that interacts with the
fluid. A body specifies a set of points and prescribes the flow velocity
in a small region near each point.
"""
abstract type AbstractBody end

"""
    struct IBProblem{N,T,B<:AbstractBody,U<:IrrotationalFlow}
        grid::Grid{N,T}
        body::B
        Re::T
        u0::U
    end

Defines the entire immersed boundary problem to be solved. An `IBProblem`
instance contains all necessary components: grid, body, Reynolds number,
and background flow.

# Parameters
- `N`: Dimension of the problem (2D or 3D).
- `T`: Scalar type (e.g., `Float64`).
- `B<:AbstractBody`: The concrete body type.
- `U<:IrrotationalFlow`: The concrete background flow type.

# Fields
- `grid::Grid{N,T}`: The fluid grid.
- `body::B`: The immersed body (must be a subtype of `AbstractBody`).
- `Re::T`: The Reynolds number.
- `u0::U`: The background flow (must be a subtype of `IrrotationalFlow`).
"""
struct IBProblem{N,T,B<:AbstractBody,U<:IrrotationalFlow}
    grid::Grid{N,T}
    body::B
    Re::T
    u0::U
end

# These are stubs. The docstrings are removed to avoid "duplicate docs" errors.
# The real docstrings should be in the file where these are implemented.
function surface_force! end
function surface_force_sum end