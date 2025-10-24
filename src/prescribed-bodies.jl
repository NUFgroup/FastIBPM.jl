abstract type AbstractPrescribedBody <: AbstractBody end

abstract type AbstractStaticBody <: AbstractPrescribedBody end

struct NothingBody <: AbstractStaticBody end

point_count(::NothingBody) = 0
init_body_points!(::BodyPoints, ::NothingBody) = nothing
update_body_points!(::BodyPoints, ::NothingBody, i, t) = nothing

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
