"""
Eulerian (fluid) grid definitions for the immersed boundary projection method.

Defines the staggered Cartesian grid hierarchy used throughout Immersa:
  - Grid kind and location types (Primal/Dual, Node/Edge).
  - The `Grid` struct and its coordinate/spacing utilities.
  - Index-range helpers for interior, boundary, and edge-aligned cells.
  - Allocation helpers (`grid_zeros`, `boundary_zeros`, `grid_view`).
"""

# ---------------------------------------------------------------------------
# Grid kind and location tags
# ---------------------------------------------------------------------------

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
    const Loc_p = Node{Dual}

Type alias for pressure (and any other cell-centered scalar), stored at the
centers of the `Dual` grid cells. This is the natural location for the
primitive-variable formulation, where `p` sits at cell centers and the discrete
divergence `∇ · u` maps velocity edges to this location.
"""
const Loc_p = Node{Dual}

# ---------------------------------------------------------------------------
# Grid struct
# ---------------------------------------------------------------------------

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
"""
gridcorner(grid::Grid) = grid.x0

"""
    gridcorner(grid::Grid, level::Integer)

Computes the corner position (origin) of a coarser multigrid level,
which is centered relative to the fine grid.
"""
gridcorner((; x0, h, n)::Grid, level::Integer) = x0 + h * n * (1 - 2^(level - 1)) / 2

"""
    gridstep(grid::Grid)

Returns the grid spacing for the base grid (`grid.h`).
"""
gridstep(grid::Grid) = grid.h

"""
    gridstep(grid::Grid, level::Integer)

Computes the grid spacing for a coarser multigrid level.
Each level doubles the spacing: `grid.h * 2^(level - 1)`.
"""
gridstep(grid::Grid, level::Integer) = grid.h * 2^(level - 1)

"""
    coord(grid::Grid, loc, I::SVector{N,<:Integer}, args...)

Computes the physical coordinates for a given grid index `I` and `GridLocation` `loc`.
This is the core logic that accounts for staggered grid offsets.
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

Fractional cell offset for a `Primal` edge (velocity): half-cell offset in
directions *other* than the edge's direction `i`.
"""
_cellcoord((; i)::Edge{Primal}, ::Val{N}) where {N} = SVector(ntuple(≠(i), N)) / 2

"""
    _cellcoord(loc::Edge{Dual}, ::Val{N})

Fractional cell offset for a `Dual` edge (vorticity): half-cell offset *along*
the edge's direction `i`.
"""
_cellcoord((; i)::Edge{Dual}, ::Val{N}) where {N} = SVector(ntuple(==(i), N)) / 2

"""
    _cellcoord(loc::Node{Primal}, ::Val{N})

Fractional cell offset for a `Primal` node (grid vertex): zero offset in every
direction, i.e. the variable sits on the cell corners.
"""
_cellcoord(::Node{Primal}, ::Val{N}) where {N} = SVector(ntuple(_ -> 0, N)) / 2

"""
    _cellcoord(loc::Node{Dual}, ::Val{N})

Fractional cell offset for a `Dual` node (cell center): half-cell offset in
*every* direction, i.e. the variable sits at the center of each cell (where the
pressure lives).
"""
_cellcoord(::Node{Dual}, ::Val{N}) where {N} = SVector(ntuple(_ -> 1, N)) / 2

# ---------------------------------------------------------------------------
# Boundary flags
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Index / axes helpers
# ---------------------------------------------------------------------------

"""
    edge_axes(::Val{N}, loc::Type{<:Edge})

Provides the list of possible directions (axes) associated with edges on a
grid of dimension `N`.
"""
# TODO: ::Val{N} is a dispatch-only argument (N is used via `where`, not the value).
# Consider whether this is the right pattern or if N should be passed differently.
edge_axes(::Val{N}, loc::Type{<:Edge}) where {N} = ntuple(identity, N)

"""
    edge_axes(::Val{2}, loc::Type{Edge{Dual}})

Special case for 2D `Dual` edges: returns `OffsetTuple{3}((3,))` to represent
the single out-of-plane vorticity component.
"""
# TODO: same dispatch-only ::Val{2} pattern as above.
edge_axes(::Val{2}, loc::Type{Edge{Dual}}) = OffsetTuple{3}((3,))

"""
    cell_axes(n::SVector{N}, loc::Edge, ::IncludeBoundary)

Index ranges for a grid location, *including* boundaries.
"""
function cell_axes(n::SVector{N}, loc::Edge, ::IncludeBoundary) where {N}
    ntuple(j -> _on_bndry(loc, j) ? (0:n[j]) : (0:(n[j]-1)), Val(N))
end

"""
    cell_axes(n::SVector{N}, loc::Edge, ::ExcludeBoundary)

Index ranges for a grid location, *excluding* boundaries (interior only).
"""
function cell_axes(n::SVector{N}, loc::Edge, ::ExcludeBoundary) where {N}
    ntuple(j -> _on_bndry(loc, j) ? (1:(n[j]-1)) : (0:(n[j]-1)), Val(N))
end

"""
    cell_axes(n::SVector{N}, loc::Node{Dual}, bndry)

Index ranges for a cell-centered (`Dual` node) scalar field, e.g. pressure. The
`n` cell centers span `0:(n[j]-1)` in each direction. Unlike edges, no cell
center lies on the domain boundary, so the `IncludeBoundary`/`ExcludeBoundary`
flag does not change the result.
"""
function cell_axes(
    n::SVector{N}, ::Node{Dual}, ::Union{IncludeBoundary,ExcludeBoundary}
) where {N}
    ntuple(j -> 0:(n[j]-1), Val(N))
end

"""
    cell_axes(n::SVector{N}, loc::Type{<:Edge}, args...)

Vectorised `cell_axes` over all edge directions of a given `Edge` type.
"""
function cell_axes(n::SVector{N}, loc::Type{<:Edge}, args...) where {N}
    axs = edge_axes(Val(N), loc)
    map(i -> cell_axes(n, loc(i), args...), axs)
end

"""
    cell_axes(grid::Grid, args...)

Convenience method that extracts `n` from the `Grid` object.
"""
cell_axes(grid::Grid, args...) = cell_axes(grid.n, args...)

"""
    _on_bndry(loc::Edge{Primal}, j)

Returns `true` if a `Primal` edge is defined on the boundary in direction `j`
(i.e., `loc.i == j`).
"""
_on_bndry((; i)::Edge{Primal}, j) = i == j

"""
    _on_bndry(loc::Edge{Dual}, j)

Returns `true` if a `Dual` edge is defined on the boundary in direction `j`
(i.e., `loc.i != j`).
"""
_on_bndry((; i)::Edge{Dual}, j) = i ≠ j

"""
    grid_length(grid::Grid, loc::Edge, args...)

Total number of grid points for a *single* edge-centered component.
"""
function grid_length(grid::Grid, loc::Edge, args...)
    prod(length, cell_axes(grid, loc, args...))
end

"""
    grid_length(grid::Grid{N}, loc::Type{<:Edge}, args...)

Total number of grid points summed over *all* components of a given `Edge` type.
"""
function grid_length(grid::Grid{N}, loc::Type{<:Edge}, args...) where {N}
    axs = edge_axes(Val(N), loc)
    sum(i -> grid_length(grid, loc(i), args...), axs)
end

"""
    boundary_axes(n::SVector{N}, loc::Edge)

Index ranges that lie exactly on the boundaries for a *single* edge component.
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

Applies `boundary_axes` to all edge directions for a vector-valued field.
"""
function boundary_axes(
    n::SVector{N}, loc::Type{<:Edge}; dims=edge_axes(Val(N), loc)
) where {N}
    map(i -> boundary_axes(n, loc(i)), dims)
end

"""
    boundary_axes(grid::Grid, args...; kw...)

Convenience method that extracts `n` from the `Grid` object.
"""
boundary_axes(grid::Grid, args...; kw...) = boundary_axes(grid.n, args...; kw...)

"""
    boundary_length(grid::Grid, loc::Edge)

Total number of DOFs located exactly on the boundaries for a *single* component.
"""
function boundary_length(grid::Grid, loc::Edge)
    sum(dims -> prod(length, dims), boundary_axes(grid, loc))
end

"""
    boundary_length(grid::Grid{N}, loc::Type{<:Edge})

Total boundary DOFs summed over *all* components of a given `Edge` type.
"""
function boundary_length(grid::Grid{N}, loc::Type{<:Edge}) where {N}
    axs = edge_axes(Val(N), loc)
    sum(i -> boundary_length(grid, loc(i)), axs)
end

"""
    _exclude_boundary(a, grid, loc)

Returns non-allocating interior views of each component array in `a`,
excluding boundary points according to `cell_axes(..., ExcludeBoundary())`.
"""
function _exclude_boundary(a, grid, loc)
    map(tupleindices(a)) do i
        R = CartesianIndices(
            Base.IdentityUnitRange.(cell_axes(grid, loc(i), ExcludeBoundary()))
        )
        @view a[i][R]
    end
end

# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

"""
    grid_zeros(backend, grid, loc::GridLocation, bndry=IncludeBoundary())

Allocate a single zero-filled `OffsetArray` for a given grid location on `backend`.
"""
function grid_zeros(
    backend, grid::Grid{N,T}, loc::GridLocation, bndry=IncludeBoundary()
) where {N,T}
    R = cell_axes(grid, loc, bndry)
    OffsetArray(KernelAbstractions.zeros(backend, T, length.(R)), R)
end

"""
    grid_zeros(backend, grid, loc::Type{<:Edge}, args...; levels=1)

Allocate zero-filled arrays for all components of an `Edge` type, optionally
across multiple multigrid `levels`.
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

Allocate a nested structure of zero-filled `OffsetArray`s for the grid boundaries,
one per boundary face per field component.
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

Create non-allocating views into the active region of each component array in `a`,
sliced by `cell_axes(grid, loc, bndry)`.
"""
function grid_view(a, grid, loc, bndry)
    R = cell_axes(grid, loc, bndry)
    map(tupleindices(a)) do i
        r = CartesianIndices(Base.IdentityUnitRange.(R[i]))
        @view a[i][r]
    end
end
