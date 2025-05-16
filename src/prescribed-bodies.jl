abstract type AbstractPrescribedBody <: AbstractBody end

abstract type AbstractStaticBody <: AbstractPrescribedBody end

struct NothingBody <: AbstractStaticBody end

point_count(::NothingBody) = 0
init_body_points!(::BodyPoints, ::NothingBody) = nothing
update_body_points!(::BodyPoints, ::NothingBody, i, t) = nothing

struct StaticBody{N,T,S<:AbstractVector{T},A<:AbstractVector{SVector{N,T}}} <:
       AbstractStaticBody
    x::A
    ds::S
end

point_count(body::StaticBody) = length(body.x)

function init_body_points!(points::BodyPoints, body::StaticBody{N,T}) where {N,T}
    points.x .= body.x
    points.u .= (zero(SVector{N,T}),)
    points.ds[:] = body.ds
end

update_body_points!(::BodyPoints, ::StaticBody, i, t) = nothing
