"""
Lagrangian (body) grid definitions for the immersed boundary projection method.

Defines the data structures representing the Lagrangian marker points that
discretise immersed bodies on the fluid grid:
  - `AbstractBody`  — root abstract type for all immersed-body models.
  - `BodyPoints`    — container for Lagrangian marker positions, velocities,
                      and quadrature weights; the fundamental Lagrangian grid object.
"""

# ---------------------------------------------------------------------------
# Root body abstract type
# ---------------------------------------------------------------------------

"""
    AbstractBody

An abstract type defining the interface for a body that interacts with the
fluid. A body specifies a set of points and prescribes the flow velocity
in a small region near each point.
"""
abstract type AbstractBody end

# ---------------------------------------------------------------------------
# Lagrangian marker data structure
# ---------------------------------------------------------------------------

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
