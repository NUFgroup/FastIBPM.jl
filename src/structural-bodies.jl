struct StructuralState{T,A<:AbstractVector{T}}
    χ::A  # Structural displacements
    ζ::A  # Structural velocities
    ζdot::A  # Structural accelerations
end

function StructuralState{T}(backend, n) where {T}
    (χ, ζ, ζdot) = ntuple(3) do _
        KernelAbstractions.zeros(backend, T, n)
    end
    StructuralState(χ, ζ, ζdot)
end

struct StructureBC{T}
    index::Int
    value::FunctionWrapper{T,Tuple{Int,Int,T}}

    function StructureBC{T}(index, value) where {T}
        new{T}(index, FunctionWrapper{T,Tuple{Int,Int,T}}(value))
    end
end

@kwdef struct GeometricNonlinearBody{
    N,T,S<:AbstractVector{T},V<:AbstractVector{SVector{N,T}},B<:AbstractPrescribedBody
} <: AbstractBody
    xref::V
    ds0::S
    m::S
    kb::S
    ke::S
    bcs::Vector{StructureBC{T}}
    prescribed::B = NothingBody()
end

deforming_point_count(body::GeometricNonlinearBody) = length(body.xref)
deforming_point_range(body::GeometricNonlinearBody) = 1:deforming_point_count(body)

structure_var_count(::AbstractPrescribedBody) = 0
structure_var_count(body::GeometricNonlinearBody{2}) = 3 * deforming_point_count(body)

function prescribed_point_range(body::GeometricNonlinearBody)
    deforming_point_count(body) .+ (1:point_count(body.prescribed))
end

function point_count(body::GeometricNonlinearBody)
    deforming_point_count(body) + point_count(body.prescribed)
end

@kwdef mutable struct GeometricNonlinearBodyOperators{TM,TK,TQ,S,TKh1,TKh2}
    M::TM
    K::TK
    Q::TQ
    β0::S
    Khat_temp::TKh1
    Khat::TKh2
    Fint::S
    f_temp::S
end

function structural_operators(backend::CPU, body::GeometricNonlinearBody{N,T}) where {N,T}
    n = structure_var_count(body)
    nel = deforming_point_count(body) - 1

    M = zeros(T, n, n)
    K = zeros(T, n, n)
    Q = Q_matrix(backend, body)

    β0 = Vector{T}(undef, nel)
    for i in 1:nel
        dx = body.xref[i+1] - body.xref[i]
        β0[i] = atan(dx[2], dx[1])
    end

    Fint = zeros(T, n)
    f_temp = zeros(T, n)
    Khat_temp = Matrix{T}(I, n, n)
    Khat = lu(Khat_temp)

    GeometricNonlinearBodyOperators(; M, K, Q, β0, Khat_temp, Khat, Fint, f_temp)
end

function Q_matrix(::CPU, body::GeometricNonlinearBody{N,T}) where {N,T}
    n = structure_var_count(body)
    nel = deforming_point_count(body) - 1
    LinearMap{T}(n; ismutating=true) do y, x
        y .= 0
        for i in 1:nel
            ds0 = body.ds0[i]
            el_ind = @. (i - 1) * 3 + (1:6)

            fx1, fy1, _, fx2, fy2, _ = view(x, el_ind)

            F_e = SA[
                ds0 / 3 * fx1 + ds0 / 6 * fx2
                26 * ds0 / 70 * fy1 + 9 * ds0 / 70 * fy2
                -11 * ds0^2 / 210 * fy1 - ds0^2 * 13 / 420 * fy2
                ds0 / 6 * fx1 + ds0 / 3 * fx2
                9 * ds0 / 70 * fy1 + 26 * ds0 / 70 * fy2
                13 * ds0^2 / 420 * fy1 + ds0^2 * 11 / 210 * fy2
            ]

            @views y[el_ind] .+= F_e
        end

        for bc in body.bcs
            y[bc.index] = 0
        end

        return y
    end
end

function init_structure_operators!(
    ops::GeometricNonlinearBodyOperators,
    body::GeometricNonlinearBody,
    points::BodyPoints,
    state::StructuralState,
    dt,
)
    update_M!(ops, body, points, state)
    update_K!(ops, body, points, state)
    update_Khat!(ops, dt)
    update_Fint!(ops, body, points, state)

    nothing
end

function update_structure_operators!(
    ops::GeometricNonlinearBodyOperators,
    body::GeometricNonlinearBody,
    points::BodyPoints,
    state::StructuralState,
    dt,
)
    update_K!(ops, body, points, state)
    update_Khat!(ops, dt)
    update_Fint!(ops, body, points, state)

    nothing
end

function update_Khat!(ops::GeometricNonlinearBodyOperators, dt)
    @. ops.Khat_temp = ops.K + (4 / dt^2) * ops.M
    ops.Khat = lu!(ops.Khat_temp)

    nothing
end

function update_M!(
    ops::GeometricNonlinearBodyOperators,
    body::GeometricNonlinearBody,
    points::BodyPoints,
    ::StructuralState,
)
    nb = deforming_point_count(body)
    nel = nb - 1

    ops.M .= 0

    # We will build these matrices by element and assemble in a loop
    for i_el in 1:nel
        Δs = points.ds[i_el]
        m = body.m[i_el]

        # Indices corresponding with the 6 unknowns associated w/ each element
        el_ind = @. (i_el - 1) * 3 + (1:6)

        M_e =
            m * Δs / 420 * @SMatrix [
                140 0 0 70 0 0
                0 156 22*Δs 0 54 -13*Δs
                0 22*Δs 4*Δs^2 0 13*Δs -3*Δs^2
                70 0 0 140 0 0
                0 54 13*Δs 0 156 -22*Δs
                0 -13*Δs -3*Δs^2 0 -22*Δs 4*Δs^2
            ]

        # Assemble into global matrices
        # Add contributions for each DOF in the element
        @views @. ops.M[el_ind, el_ind] .+= M_e
    end

    # Account for BCs
    for bc in body.bcs
        ops.M[bc.index, :] .= 0
        ops.M[:, bc.index] .= 0
    end

    nothing
end

function update_K!(
    ops::GeometricNonlinearBodyOperators,
    body::GeometricNonlinearBody,
    points::BodyPoints,
    state::StructuralState,
)
    nb = deforming_point_count(body) # Number of body points
    nel = nb - 1 # Number of finite elements

    ops.K .= 0

    # We will build these matrices by element and assemble in a loop
    for i_el in 1:nel
        Δs = points.ds[i_el]
        Δs0 = body.ds0[i_el]
        kb = body.kb[i_el]
        ke = body.ke[i_el]

        # Indices corresponding with the 6 unknowns associated w/ each element
        el_ind = @. (i_el - 1) * 3 + (1:6)

        # ke is equivalent to K_s in the Fortran version
        r_c = kb / ke
        CL = ke / Δs0 * @SMatrix [
            1 0 0
            0 4*r_c 2*r_c
            0 2*r_c 4*r_c
        ]

        (dx, dy) = points.x[i_el+1] - points.x[i_el]
        cβ = dx / Δs
        sβ = dy / Δs

        B = @SMatrix [
            -cβ -sβ 0 cβ sβ 0
            -sβ/Δs cβ/Δs 1 sβ/Δs -cβ/Δs 0
            -sβ/Δs cβ/Δs 0 sβ/Δs -cβ/Δs 1
        ]

        K1 = B' * CL * B

        z = SVector(sβ, -cβ, 0, -sβ, cβ, 0)
        r = -SVector(cβ, sβ, 0, -cβ, sβ, 0)

        # Better conditioned formula for Δs-Δs0 when the difference is small
        uL = (Δs^2 - Δs0^2) / (Δs + Δs0)

        Nf = ke * uL / Δs0

        θ1 = state.χ[el_ind[3]]
        θ2 = state.χ[el_ind[6]]

        β0 = ops.β0[i_el]
        β1 = θ1 + β0
        β2 = θ2 + β0

        θ1L = atan(cβ * sin(β1) - sβ * cos(β1), cβ * cos(β1) + sβ * sin(β1))
        θ2L = atan(cβ * sin(β2) - sβ * cos(β2), cβ * cos(β2) + sβ * sin(β2))

        Mf1 = 2 * kb / Δs0 * (2 * θ1L + θ2L)
        Mf2 = 2 * kb / Δs0 * (θ1L + 2 * θ2L)

        Kσ = Nf / Δs * z * z' + (Mf1 + Mf2) / Δs^2 * (r * z' + z * r')

        K_e = K1 + Kσ

        # Assemble into global matrices
        # Add contributions for each DOF in the element
        @views @. ops.K[el_ind, el_ind] .+= K_e
    end

    # Account for BCs
    for bc in body.bcs
        i = bc.index
        ops.K[i, :] .= 0.0
        ops.K[:, i] .= 0.0
        ops.K[i, i] = 1.0
    end

    nothing
end

function update_Fint!(
    ops::GeometricNonlinearBodyOperators,
    body::GeometricNonlinearBody,
    points::BodyPoints,
    state::StructuralState,
)
    nb = deforming_point_count(body) # Number of body points
    nel = nb - 1 # Number of finite elements

    ops.Fint .= 0

    # We will build these matrices by element and assemble in a loop
    for i_el in 1:nel
        Δs = points.ds[i_el]
        Δs0 = body.ds0[i_el]
        kb = body.kb[i_el]
        ke = body.ke[i_el]

        # Indices corresponding with the 6 unknowns associated w/ each element
        el_ind = @. (i_el - 1) * 3 + (1:6)

        (dx, dy) = points.x[i_el+1] - points.x[i_el]
        cβ = dx / Δs
        sβ = dy / Δs

        B = @SMatrix [
            -cβ -sβ 0 cβ sβ 0
            -sβ/Δs cβ/Δs 1 sβ/Δs -cβ/Δs 0
            -sβ/Δs cβ/Δs 0 sβ/Δs -cβ/Δs 1
        ]

        # Better conditioned formula for Δs-Δs0 when the difference is small
        uL = (Δs^2 - Δs0^2) / (Δs + Δs0)

        Nf = ke * uL / Δs0

        θ1 = state.χ[el_ind[3]]
        θ2 = state.χ[el_ind[6]]

        β0 = ops.β0[i_el]
        β1 = θ1 + β0
        β2 = θ2 + β0

        θ1L = atan(cβ * sin(β1) - sβ * cos(β1), cβ * cos(β1) + sβ * sin(β1))
        θ2L = atan(cβ * sin(β2) - sβ * cos(β2), cβ * cos(β2) + sβ * sin(β2))

        Mf1 = 2 * kb / Δs0 * (2 * θ1L + θ2L)
        Mf2 = 2 * kb / Δs0 * (θ1L + 2 * θ2L)

        qL = @SVector [Nf, Mf1, Mf2]

        # Internal forces in global frame
        qint = B' * qL

        @views @. ops.Fint[el_ind] .+= qint
    end

    for bc in body.bcs
        ops.Fint[bc.index] = 0
    end

    nothing
end

function structure_to_fluid_displacement!(
    x_fluid,
    x_structure,
    body::GeometricNonlinearBody{N,T},
    ::GeometricNonlinearBodyOperators,
) where {N,T}
    a_structure = reinterpret(
        eltype(x_fluid), vec(@view reshape(x_structure, 3, :)[1:2, :])
    )
    x_fluid[deforming_point_range(body)] = a_structure
    x_fluid
end

function fluid_to_structure_force!(
    f_structure,
    f_fluid,
    ::GeometricNonlinearBody{N,T},
    ops::GeometricNonlinearBodyOperators,
) where {N,T}
    nb = length(f_fluid)
    f = ops.f_temp
    f .= 0
    @views vec(reshape(f, 3, nb)[1:2, :]) .= reinterpret(T, f_fluid)

    mul!(f_structure, ops.Q, f)
end

function update_structure!(
    points::BodyPoints,
    def::StructuralState,
    body::GeometricNonlinearBody,
    ops::GeometricNonlinearBodyOperators,
    i,
    t,
)
    nb = deforming_point_count(body)
    ib_deform = view(points, deforming_point_range(body))

    structure_to_fluid_displacement!(ib_deform.x, def.χ, body, ops)
    @. ib_deform.x += body.xref

    structure_to_fluid_displacement!(ib_deform.u, def.ζ, body, ops)

    for i in 1:nb-1
        ib_deform.ds[i] = norm(ib_deform.x[i+1] - ib_deform.x[i])
    end
    if nb > 1
        ib_deform.ds[end] = ib_deform.ds[end-1]
    end

    nothing
end

function update_structure_bc!(χ, body::GeometricNonlinearBody, i, t)
    for bc in body.bcs
        χ[bc.index] = bc.value(bc.index, i, t)
    end
end
