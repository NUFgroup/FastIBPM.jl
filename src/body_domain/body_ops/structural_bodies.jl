"""
    StructuralState{T,A<:AbstractVector{T}}

Container for the kinematic state of a deformable structure, storing displacements,
velocities, and accelerations in generalised (structural) coordinates.

# Fields
- `χ::A`    : Structural displacements.
- `ζ::A`    : Structural velocities.
- `ζdot::A` : Structural accelerations.

# Constructor
    StructuralState{T}(backend, n)

Allocates a zero-initialised state with `n` structural degrees of freedom on the
given computation `backend` (e.g. `CPU()`).

# Arguments
- `backend` : Computation backend (`CPU()` or GPU device).
- `n::Int`  : Number of structural degrees of freedom.

# Returns
A `StructuralState{T}` with all fields initialised to zero.
"""
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

"""
    StructureBC{T}

Represents a single structural boundary condition that prescribes the value of a
specific degree of freedom as a function of `(index, step, time)`.

# Fields
- `index::Int`                              : Global DOF index to constrain.
- `value::FunctionWrapper{T,Tuple{Int,Int,T}}` : Callable `(index, i, t) -> T` returning the
  prescribed value at step `i` and time `t`.

# Constructor
    StructureBC{T}(index, value)

Wraps `value` in a `FunctionWrapper` for type-stable dispatch.
"""
struct StructureBC{T}
    index::Int
    value::FunctionWrapper{T,Tuple{Int,Int,T}}

    function StructureBC{T}(index, value) where {T}
        new{T}(index, FunctionWrapper{T,Tuple{Int,Int,T}}(value))
    end
end

"""
    GeometricNonlinearBody{N,T,S,V,B} <: AbstractBody

A deformable body described by a geometrically nonlinear (co-rotational) beam model.

The body is discretised as a chain of 2-node Euler–Bernoulli beam elements in `N`
dimensions, each carrying axial, transverse, and rotational degrees of freedom.
An optional `prescribed` sub-body handles any rigidly attached or prescribed-motion
points (e.g. a clamped root or a moving base).

# Type parameters
- `N` : Spatial dimension (typically 2).
- `T` : Scalar type (e.g. `Float64`).
- `S` : Vector type for per-element scalar properties.
- `V` : Vector type for per-node position vectors (`SVector{N,T}`).
- `B` : Type of the prescribed sub-body.

# Fields
- `xref::V`                    : Reference (undeformed) positions of the deforming nodes.
- `ds0::S`                     : Reference element arc-lengths.
- `m::S`                       : Per-element mass per unit length.
- `kb::S`                      : Per-element bending stiffness.
- `ke::S`                      : Per-element extensional stiffness.
- `bcs::Vector{StructureBC{T}}`: Boundary conditions on structural DOFs.
- `prescribed::B`              : Sub-body for any prescribed-motion points (default `NothingBody()`).
"""
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

"""
    deforming_point_count(body::GeometricNonlinearBody) -> Int

Return the number of deforming (structural) nodes on `body`.
"""
deforming_point_count(body::GeometricNonlinearBody) = length(body.xref)

"""
    deforming_point_range(body::GeometricNonlinearBody) -> UnitRange{Int}

Return the index range `1:n` covering the deforming nodes in the combined point array.
"""
deforming_point_range(body::GeometricNonlinearBody) = 1:deforming_point_count(body)

"""
    structure_var_count(body) -> Int

Return the number of structural degrees of freedom for `body`.

- For any `AbstractPrescribedBody`, returns `0` (no structural DOFs).
- For a 2-D `GeometricNonlinearBody`, returns `3n` where `n` is the number of
  deforming nodes (two translations + one rotation per node).
"""
structure_var_count(::AbstractPrescribedBody) = 0
structure_var_count(body::GeometricNonlinearBody{2}) = 3 * deforming_point_count(body)

"""
    prescribed_point_range(body::GeometricNonlinearBody) -> UnitRange{Int}

Return the index range covering the prescribed (non-deforming) points that follow
the deforming nodes in the combined point array.
"""
function prescribed_point_range(body::GeometricNonlinearBody)
    deforming_point_count(body) .+ (1:point_count(body.prescribed))
end

"""
    point_count(body::GeometricNonlinearBody) -> Int

Return the total number of immersed-boundary points (deforming + prescribed).
"""
function point_count(body::GeometricNonlinearBody)
    deforming_point_count(body) + point_count(body.prescribed)
end

"""
    GeometricNonlinearBodyOperators{TM,TK,TQ,S,TKh1,TKh2}

Mutable container holding the finite-element operators for a
`GeometricNonlinearBody`. These are assembled and updated during
the simulation as the body deforms.

# Fields
- `M::TM`            : Global consistent mass matrix.
- `K::TK`            : Global tangent stiffness matrix (material + geometric).
- `Q::TQ`            : Force-projection operator (fluid → structure), implemented as a `LinearMap`.
- `β0::S`            : Reference element orientations (angles of undeformed elements).
- `Khat_temp::TKh1`  : Workspace for the effective stiffness matrix `K + (4/dt²) M`.
- `Khat::TKh2`       : LU factorisation of `Khat_temp`, used in the Newmark solve.
- `Fint::S`          : Internal (elastic) force vector in global coordinates.
- `f_temp::S`        : Temporary vector for force transformations.
"""
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

"""
    structural_operators(backend::CPU, body::GeometricNonlinearBody{N,T}) -> GeometricNonlinearBodyOperators

Allocate and initialise all finite-element operators for `body` on the given `backend`.

Computes the reference element orientations `β0`, allocates the global mass, stiffness,
and internal-force arrays, builds the force-projection operator `Q`, and returns
a fully constructed `GeometricNonlinearBodyOperators`.

# Arguments
- `backend::CPU`                         : Computation backend.
- `body::GeometricNonlinearBody{N,T}`    : The nonlinear structural body.

# Returns
A `GeometricNonlinearBodyOperators` ready for use in FSI time-stepping.
"""
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

"""
    Q_matrix(::CPU, body::GeometricNonlinearBody{N,T}) -> LinearMap{T}

Build the force-projection operator `Q` that maps fluid-frame nodal forces to
generalised structural forces using consistent finite-element shape functions.

The operator is returned as a `LinearMap` (matrix-free, in-place). Rows
corresponding to constrained DOFs (boundary conditions) are zeroed.

# Arguments
- `::CPU`                             : Computation backend.
- `body::GeometricNonlinearBody{N,T}` : The nonlinear structural body.

# Returns
A `LinearMap{T}` of size `n × n` where `n = structure_var_count(body)`.
"""
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

"""
    init_structure_operators!(ops, body, points, state, dt)

Perform the initial assembly of all structural operators at the start of a simulation.

Assembles the mass matrix `M`, tangent stiffness `K`, effective stiffness `K̂`,
and internal force vector `Fint` from the current geometry.

# Arguments
- `ops::GeometricNonlinearBodyOperators` : Operator container to fill.
- `body::GeometricNonlinearBody`         : The structural body.
- `points::BodyPoints`                   : Current body-point positions and spacings.
- `state::StructuralState`               : Current structural state.
- `dt`                                   : Time step size.
"""
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

"""
    update_structure_operators!(ops, body, points, state, dt)

Re-assemble geometry-dependent structural operators after the body has deformed.

Unlike `init_structure_operators!`, this skips the mass matrix (which depends only
on the reference configuration) and updates only the tangent stiffness `K`,
effective stiffness `K̂`, and internal forces `Fint`.

# Arguments
- `ops::GeometricNonlinearBodyOperators` : Operator container to update.
- `body::GeometricNonlinearBody`         : The structural body.
- `points::BodyPoints`                   : Current body-point positions and spacings.
- `state::StructuralState`               : Current structural state.
- `dt`                                   : Time step size.
"""
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

"""
    update_Khat!(ops, dt)

Form and factorise the effective stiffness matrix used in the Newmark time integrator:

    K̂ = K + (4 / dt²) M

The result is stored as an in-place LU factorisation in `ops.Khat`.

# Arguments
- `ops::GeometricNonlinearBodyOperators` : Operator container (modified in place).
- `dt`                                   : Time step size.
"""
function update_Khat!(ops::GeometricNonlinearBodyOperators, dt)
    @. ops.Khat_temp = ops.K + (4 / dt^2) * ops.M
    ops.Khat = lu!(ops.Khat_temp)

    nothing
end

"""
    update_M!(ops, body, points, state)

Assemble the global consistent mass matrix from element contributions.

Each element uses the standard Euler–Bernoulli beam mass matrix (axial +
transverse inertia) with mass per unit length `m` and current element spacing `Δs`.
Rows and columns corresponding to boundary conditions are zeroed.

# Arguments
- `ops::GeometricNonlinearBodyOperators` : Operator container (modified in place).
- `body::GeometricNonlinearBody`         : The structural body.
- `points::BodyPoints`                   : Current body-point positions and spacings.
- `state::StructuralState`               : Current structural state (unused; kept for interface consistency).
"""
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

"""
    update_K!(ops, body, points, state)

Assemble the global tangent stiffness matrix (material + geometric contributions).

For each element, the material stiffness is computed from the co-rotational
strain–displacement matrix `B` and the constitutive matrix `CL` (bending + extensional).
The geometric stiffness `Kσ` accounts for the normal force `Nf` and bending moments
`Mf1`, `Mf2` in the deformed configuration. Rows and columns corresponding to
boundary conditions are zeroed with a unit diagonal.

# Arguments
- `ops::GeometricNonlinearBodyOperators` : Operator container (modified in place).
- `body::GeometricNonlinearBody`         : The structural body.
- `points::BodyPoints`                   : Current body-point positions and spacings.
- `state::StructuralState`               : Current structural state (provides rotations `θ`).
"""
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

"""
    update_Fint!(ops, body, points, state)

Assemble the global internal (elastic) force vector from element contributions.

For each element, the local generalised forces — normal force `Nf` and bending
moments `Mf1`, `Mf2` — are computed in the co-rotational frame and mapped to
global DOFs via the strain–displacement matrix `B`. Constrained DOFs are zeroed.

# Arguments
- `ops::GeometricNonlinearBodyOperators` : Operator container (modified in place).
- `body::GeometricNonlinearBody`         : The structural body.
- `points::BodyPoints`                   : Current body-point positions and spacings.
- `state::StructuralState`               : Current structural state (provides rotations `θ`).
"""
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

"""
    structure_to_fluid_displacement!(x_fluid, x_structure, body, ops)

Map structural DOFs to fluid-frame (immersed-boundary) displacements.

Extracts the translational components from the interleaved structural vector
`x_structure` (which stores `[u₁, v₁, θ₁, u₂, v₂, θ₂, …]`) and writes them
as `SVector{N,T}` entries into the deforming-point range of `x_fluid`.

# Arguments
- `x_fluid`      : Fluid-frame displacement or velocity array (modified in place).
- `x_structure`   : Structural DOF vector.
- `body::GeometricNonlinearBody{N,T}`    : The structural body.
- `ops::GeometricNonlinearBodyOperators` : Structural operators (unused; kept for dispatch).

# Returns
The modified `x_fluid`.
"""
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

"""
    fluid_to_structure_force!(f_structure, f_fluid, body, ops)

Map fluid-frame body forces to generalised structural forces via the
projection operator `Q`.

The `SVector{N,T}` entries in `f_fluid` are unpacked into an interleaved
temporary vector, then multiplied by `ops.Q` to produce the structural
force vector `f_structure`.

# Arguments
- `f_structure` : Output structural force vector (modified in place).
- `f_fluid`     : Fluid-frame body force array (`Vector{SVector{N,T}}`).
- `body::GeometricNonlinearBody{N,T}`    : The structural body (unused; kept for dispatch).
- `ops::GeometricNonlinearBodyOperators` : Structural operators (provides `Q` and workspace).

# Returns
The modified `f_structure`.
"""
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

"""
    update_structure!(points, def, body, ops, i, t)

Synchronise the immersed-boundary point data with the current structural state.

Converts structural displacements and velocities to fluid-frame quantities, updates
the deforming node positions (reference + displacement), velocities, and recomputes
the element arc-lengths `ds` from the deformed geometry.

# Arguments
- `points::BodyPoints`                   : Body-point data (modified in place).
- `def::StructuralState`                 : Current structural state.
- `body::GeometricNonlinearBody`         : The structural body.
- `ops::GeometricNonlinearBodyOperators` : Structural operators.
- `i`                                    : Current time step index.
- `t`                                    : Current simulation time.
"""
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

    for i in 1:(nb-1)
        ib_deform.ds[i] = norm(ib_deform.x[i+1] - ib_deform.x[i])
    end
    if nb > 1
        ib_deform.ds[end] = ib_deform.ds[end-1]
    end

    nothing
end

"""
    update_structure_bc!(χ, body, i, t)

Apply structural boundary conditions by overwriting constrained entries in the
displacement vector `χ` with their prescribed values at step `i` and time `t`.

# Arguments
- `χ`                            : Structural displacement vector (modified in place).
- `body::GeometricNonlinearBody` : The structural body (provides `bcs`).
- `i`                            : Current time step index.
- `t`                            : Current simulation time.
"""
function update_structure_bc!(χ, body::GeometricNonlinearBody, i, t)
    for bc in body.bcs
        χ[bc.index] = bc.value(bc.index, i, t)
    end
end

