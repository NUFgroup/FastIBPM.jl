"""
This file defines types and methods for handling bodies whose motion is defined ahead of time.

A prescribed body can be static or moving, but cannot be fully coupled or deformable.

- `AbstractPrescribedBody` (a subtype of `AbstractBody`) is the superclass for all predefined-motion bodies.
- `AbstractStaticBody` is a special case where the motion is zero everywhere (i.e., static).

Hierarchy sketch:

  AbstractBody
    +- AbstractPrescribedBody
      +- AbstractStaticBody
        +- NothingBody         (dummy)
        +- StaticBody          (real, immobile body)

"""


abstract type AbstractPrescribedBody <: AbstractBody end

abstract type AbstractStaticBody <: AbstractPrescribedBody end


"""
    NothingBody{}

The NothingBody struct defines a trivial, empty static body used as a placeholder when no physical body is present.

# Related functions
Functions associated with NothingBody perform no operations:
- `point_count(::NothingBody)`: returns 0 (no points).
- `init_body_points!(::BodyPoints, ::NothingBody)`: does nothing.
- `update_body_points!(::BodyPoints, ::NothingBody, i, t)`: does nothing.
"""
struct NothingBody <: AbstractStaticBody end

point_count(::NothingBody) = 0
init_body_points!(::BodyPoints, ::NothingBody) = nothing
update_body_points!(::BodyPoints, ::NothingBody, i, t) = nothing



"""
    StaticBody{N,T,S<:AbstractVector{T},A<:AbstractVector{SVector{N,T}}}

The StaticBody struct represents a real, immobile body.

# Fields
- `x::A` : Vector of positions (`SVector{N,T}`) in N-dimensional space.
- `ds::S` : Vector of weights or spacings associated with points.

# Type parameters
- `N` : Spatial dimension.
- `T` : Scalar type (e.g., `Float64`).
- `S` : Type of vector for `ds`.
- `A` : Type of vector for `x`.

# Related functions
- `point_count(body)` : Returns the number of points on the body.
- `init_body_points!(points, body)` : Initializes `BodyPoints` with the static body's positions, zero velocities, and spacings.
- `update_body_points!(points, body, i, t)` : No-op; static bodies do not move.
"""
struct StaticBody{N,T,S<:AbstractVector{T},A<:AbstractVector{SVector{N,T}}, U} <:
       AbstractStaticBody
    x::A
    ds::S
    u::U
end

StaticBody(x, ds) = StaticBody(x, ds, (u, i, t) -> nothing)

point_count(body::StaticBody) = length(body.x)

function init_body_points!(points::BodyPoints, body::StaticBody{N,T}) where {N,T}
    points.x .= body.x
    points.u .= (zero(SVector{N,T}),)
    points.ds[:] = body.ds
end

function update_body_points!(points::BodyPoints, body::StaticBody, i, t)
    body.u(points.u, i, t)
end

# plate.jl
# tangent = SA[tx, ty]
# w(s) = 0.1 < s < 0.3 ? 1 : 0
# StaticBody(
#     x,
#     ds,
#     function (u, i, t)
#         s = range(0, 1, length(u))
#         for j in eachindex(u)
#             u[j] = ua * sin(2π * f * t + ϕ) * tangent * w(s)
#         end
#     end
# )

update_body_points!(::BodyPoints, ::StaticBody, i, t) = nothing

struct GroupedPrescribedBody{T<:Tuple{Vararg{AbstractPrescribedBody}}} <: AbstractPrescribedBody
    bodies::T
end

point_count(body::GroupedPrescribedBody) = sum(point_count, body.bodies)

function init_body_points!(points::BodyPoints, body::GroupedPrescribedBody)
    foreach(b -> init_body_points!(points, b), body.bodies)
end

function update_body_points!(points::BodyPoints, body::GroupedPrescribedBody, i, t)
    foreach(b -> update_body_points!(points, b, i, t), body.bodies)
end
