"""
Time-stepping routines for the CNAB (Crank-Nicolson Adams-Bashforth) scheme.

Contains the per-iteration stepping logic and the Crank-Nicolson / Adams-Bashforth
integration only:
  - set_time!, step!, _step!
  - prediction_step!, coupling_step!, projection_step!, apply_vorticity!
  - update_reg!, ab_coeffs

The operator-level pieces (`Ainv`/`_A_factor`, the coupling operators
`B_rigid_mul!` / `B_deform_mul!` / `B_inverse_rigid`, and the primitive `Ainv!`)
live in `assembly_ops.jl`, dispatched on the formulation.
"""

"""
    set_time!(sol::CNAB, i::Integer)

Advance the CNAB integrator to a specific time step.

This function updates the internal step index (`i`) and computes the corresponding 
physical time (`t`) of the simulation using:

    t = t0 + dt * (i - 1)

where `t0` is the initial simulation time and `dt` is the time step size. This ensures 
that all time-dependent operations in the simulation remain consistent.

# Arguments
- `sol::CNAB` : The CNAB simulation object to update.
- `i::Integer` : The target time step index.

# Returns
The updated `CNAB` object with the new time step and physical time.
"""
function set_time!(sol::CNAB, i::Integer)
    sol.i = i
    sol.t = sol.t0 + sol.dt * (i - 1)
    sol
end

"""
    step!(sol::CNAB)

Advance the CNAB simulation by one time step.

This is the public entry point for time integration. It increments the time
counter and then delegates to `_step!`, which dispatches on
`sol.prob.formulation` to select the correct pipeline for the chosen
numerical formulation.

# Arguments
- `sol::CNAB` : The CNAB simulation object representing the current state.

# Returns
The updated `CNAB` object after one complete time step.
"""
function step!(sol::CNAB)
    set_time!(sol, sol.i + 1)
    _step!(sol, sol.prob.formulation)
    sol
end

"""
    _step!(sol::CNAB, ::FastIBPM)

Streamfunction-vorticity (nullspace) time-stepping pipeline.

Executes the four-stage CNAB sequence for the ψ-ω formulation:

1. `prediction_step!`  — advances vorticity with semi-implicit diffusion (CN)
                         and explicit advection (AB2).
2. `coupling_step!`    — solves for the body force that enforces no-slip at
                         the immersed boundary.
3. `projection_step!`  — corrects vorticity by spreading the body force curl
                         back onto the grid.
4. `apply_vorticity!`  — recovers the velocity field via the multi-domain
                         Poisson solve (∇²ψ = −ω, then u = ∇×ψ).
"""
function _step!(sol::CNAB, f::FastIBPM)
    prediction_step!(sol, f)
    coupling_step!(sol, f)
    projection_step!(sol, f)
    apply_vorticity!(sol)
end

"""
    update_reg!(sol::CNAB, body, i)

Update the regularization weights for the simulation based on the type of body.

There are two methods:

1. **Static bodies (`AbstractStaticBody`)**  
   - No update is necessary because the body does not move or deform.  
   - The function returns `nothing`.

2. **Prescribed-motion bodies (`AbstractPrescribedBody`)**  
   - Updates the regularization weights by calling `update_weights!` with the 
     current body point positions.  
   - Ensures that the mapping from body points to the grid reflects the current motion.

# Arguments
- `sol::CNAB` : CNAB simulation object containing the grid and body fields.  
- `body`      : The body object (`AbstractStaticBody` or `AbstractPrescribedBody`).  
- `i`         : Index or set of points for which the regularization is updated (used for prescribed bodies).

# Returns
- Nothing for static bodies.  
- Updates `sol.reg` in-place for prescribed-motion bodies.
"""
update_reg!(::CNAB, ::AbstractStaticBody, _) = nothing
function update_reg!(sol::CNAB, ::AbstractPrescribedBody, i)
    update_weights!(sol.reg, sol.prob.grid, sol.points.x, i)
end


"""
    prediction_step!(sol::CNAB)
    prediction_step!(sol::CNAB, level)
    prediction_step!(sol::CNAB, level, u_work)

Perform the CNAB prediction of the vorticity field.

This function advances the fluid state by computing the predicted vorticity
using a semi-implicit Crank–Nicolson treatment for diffusion and an
Adams–Bashforth treatment for nonlinear convection. It supports multigrid
levels and avoids unnecessary allocations with array pools.

# Arguments
- `sol::CNAB`: CNAB simulation object.
- `level` (optional): Grid level for single-level update.
- `u_work` (optional): Preallocated velocity array for in-place computation.

# Returns
- Updated vorticity field in-place within `sol`.
"""
prediction_step!(sol::CNAB) = prediction_step!(sol, sol.prob.formulation)

function prediction_step!(sol::CNAB, ::FastIBPM)
    _cycle!(sol.state.nonlin)

    for level in sol.prob.grid.levels:-1:1
        prediction_step!(sol, level)
    end

    sol.state.nonlin_count = min(sol.state.nonlin_count + 1, length(sol.state.nonlin))
end

function prediction_step!(sol::CNAB{N,T}, level) where {N,T}
    u_axes = cell_axes(sol.prob.grid, Loc_u, ExcludeBoundary())
    with_arrays(sol.fluid_pool, (T, u_axes)) do u_work
        prediction_step!(sol::CNAB, level, u_work)
    end
end

function prediction_step!(sol::CNAB{N,T}, level, u_work) where {N,T}
    backend = get_backend(sol.state.u[1][1])

    grid = sol.prob.grid
    h = gridstep(grid, level)
    ωˢ = grid_view(sol.state.ψ[level], grid, Loc_ω, ExcludeBoundary())
    a = _A_factor(sol)

    curl!(u_work, sol.state.ω[level]; h)
    rot!(ωˢ, u_work; h)

    for i in eachindex(ωˢ)
        let ωˢ = ωˢ[i], ω = sol.state.ω[level][i]
            @loop backend (I in CartesianIndices(ωˢ)) begin
                ωˢ[I] = ω[I] - a * ωˢ[I]
            end
        end
    end

    if level < grid.levels
        with_arrays(sol.bndry_pool, (T, sol.state.ω_bndry)) do ψb
            multidomain_interpolate!(ψb, sol.state.ψ[level+1]; n=grid.n)
            add_laplacian_bc!(ωˢ, Loc_ω, a / h^2, ψb)
        end
    end

    nonlin_full = sol.state.nonlin_count == length(sol.state.nonlin)

    if nonlin_full
        for i_step in eachindex(sol.state.nonlin), i in eachindex(ωˢ)
            let ωˢ = ωˢ[i], N = sol.state.nonlin[i_step][level][i], k = sol.dt * sol.β[end-i_step]
                @loop backend (I in CartesianIndices(ωˢ)) begin
                    ωˢ[I] = ωˢ[I] + k * N[I]
                end
            end
        end
    end

    nonlinear!(u_work, sol.state.u[level], sol.state.ω[level])
    rot!(sol.state.nonlin[end][level], u_work; h)

    for i in eachindex(ωˢ)
        let ωˢ = ωˢ[i],
            N = sol.state.nonlin[end][level][i],
            k = nonlin_full ? sol.dt * sol.β[end] : sol.dt

            @loop backend (I in CartesianIndices(ωˢ)) begin
                ωˢ[I] = ωˢ[I] + k * N[I]
            end
        end
    end

    Ainv(sol, level)(ωˢ, ωˢ)
end

"""
    coupling_step!(sol::CNAB)

Perform the fluid–structure coupling step for the current CNAB time step.

This function dispatches to the appropriate coupling routine based on the
solver's `coupler`. For a prescribed body, it computes the fluid velocity
at body points, evaluates the coupling residual, and solves for the
correcting body force to enforce velocity constraints.

# Arguments
- `sol::CNAB`: CNAB simulation object with the current fluid and body state.

# Returns
- Updates `sol.f_tilde` and body-related fields in-place.
"""
coupling_step!(sol::CNAB) = coupling_step!(sol, sol.prob.formulation)
coupling_step!(sol::CNAB, ::FastIBPM) = _coupling_step!(sol, sol.coupler)

function _coupling_step!(sol::CNAB{N,T}, coupler::PrescribedBodyCoupler) where {N,T}
    with_arrays_like(sol.body_pool, sol.f_tilde) do rhs
        with_arrays(sol.bndry_pool, (T, sol.state.ω_bndry)) do ψ_b_work
            _coupling_step!(sol, coupler, rhs, ψ_b_work)
        end
    end
end

function _coupling_step!(sol::CNAB, coupler::PrescribedBodyCoupler, rhs, ψ_b_work)
    grid = sol.prob.grid
    body = sol.prob.body
    ωˢ = sol.state.ψ
    ψ = (sol.state.ω[1],)
    u¹ = sol.state.u[1]

    multidomain_poisson!(ωˢ, ψ, (u¹,), ψ_b_work, grid, sol.state.plan)
    add_flow!(u¹, sol.prob.u0, grid, 1, sol.i, sol.t)

    update_body_points!(sol.points, body, sol.i, sol.t)
    update_reg!(sol, body, eachindex(sol.points.x))
    # Arturo: for moving bodies, we need to update reg here(?)
    update_redist_weights!(sol)
    interpolate_body!(rhs, sol.reg, u¹)

    rhs .-= sol.points.u

    coupler.Binv(sol.f_tilde, rhs, sol)
end

# NOTE: B_inverse_rigid is only called during initialization (initial_sol for static bodies)
# but lives here for consistency with B_rigid_mul! and B_deform_mul!.

"""
    _coupling_step!(sol::CNAB, coupler::FsiCoupler)
    _coupling_step!(sol::CNAB, coupler::FsiCoupler, fs, χs, ψ_b_work)

Advance the fluid–structure interaction (FSI) system for deformable bodies.

This function couples the fluid and structure dynamics within the immersed
boundary framework, ensuring that the motion of a deformable body and the
surrounding fluid remain consistent.

The first method prepares temporary arrays and calls the main solver.
The second performs the actual coupling iterations until convergence.

# Purpose
Used in simulations where the body can deform under fluid forces.
It enforces the mutual interaction between fluid and structure during each time step.

# Notes
Works with deformable immersed bodies through an implicit iterative scheme.

# Returns
Nothing. Updates the solver state in place.
"""
function _coupling_step!(sol::CNAB{N,T}, coupler::FsiCoupler) where {N,T}
    with_arrays_like(sol.body_pool, ntuple(_ -> sol.f_tilde, 3)...) do fs...
        with_arrays_like(sol.structure_pool, ntuple(_ -> coupler.state.χ, 8)...) do χs...
            with_arrays(sol.bndry_pool, (T, sol.state.ω_bndry)) do ψ_b_work
                _coupling_step!(sol, coupler, fs, χs, ψ_b_work)
            end
        end
    end
end

function _coupling_step!(sol::CNAB{N,T}, coupler::FsiCoupler, fs, χs, ψ_b_work) where {N,T}
    (rhsf, F_kp1, F_sm) = fs
    (χ_k, ζ_k, ζdot_k, r_c, r_ζ, F_bg, Δχ, χ_temp) = χs
    (; χ, ζ, ζdot) = coupler.state

    grid = sol.prob.grid
    body = sol.prob.body
    ops = coupler.ops
    dt = sol.dt
    h = gridstep(grid)

    nf = N * point_count(body)
    B = LinearMap(nf; ismutating=true) do y, x
        B_deform_mul!(y, x, sol)
    end

    ωˢ = sol.state.ψ
    ψ = (sol.state.ω[1],)
    u¹ = sol.state.u[1]

    multidomain_poisson!(ωˢ, ψ, (u¹,), ψ_b_work, grid, sol.state.plan)
    add_flow!(u¹, sol.prob.u0, grid, 1, sol.i, sol.t)

    i_deform = deforming_point_range(body)
    i_prescribed = prescribed_point_range(body)
    update_body_points!(view(sol.points, i_prescribed), body.prescribed, sol.i, sol.t)

    update_reg!(sol, body.prescribed, i_prescribed)

    it = 0
    χ_k .= χ
    ζ_k .= ζ
    ζdot_k .= ζdot

    update_structure!(sol.points, coupler.state, body, coupler.ops, sol.i, sol.t)
    update_structure_operators!(ops, body, sol.points, coupler.state, sol.dt)
    update_weights!(sol.reg, grid, sol.points.x, i_deform)
    update_redist_weights!(sol)

    while true
        if it + 1 > coupler.maxiter
            error("exceeded maximum iteration count")
        end

        interpolate_body!(F_kp1, sol.reg, u¹)

        @views @. F_kp1[i_prescribed] -= sol.points.u[i_prescribed]

        @. r_c = 2 / dt * (χ_k - χ) - ζ

        @. χ_temp = ζdot + 4 / dt * ζ + 4 / dt^2 * (χ - χ_k)
        mul!(r_ζ, ops.M, χ_temp)
        @. χ_temp = r_ζ - ops.Fint

        ldiv!(r_ζ, ops.Khat, χ_temp)

        @. F_bg = -(2 / dt * r_ζ + r_c)

        fill!(F_sm, zero(SVector{N,T}))
        structure_to_fluid_displacement!(view(F_sm, i_deform), F_bg, body, ops)
        @. rhsf = F_sm + F_kp1

        bicgstabl!(
            reinterpret(T, sol.f_tilde), B, reinterpret(T, rhsf); coupler.bicgstabl_args...
        )

        # Redistribute
        sol.f .= sol.f_tilde
        f_to_f_tilde!(sol.f, sol; inverse=true)
        redist!(sol.f, sol)

        fluid_to_structure_force!(χ_temp, view(sol.f, i_deform), body, ops)
        ldiv!(Δχ, ops.Khat, χ_temp)
        @. Δχ += r_ζ

        χ_norm = norm(χ_k, Inf)
        Δχ_norm = norm(Δχ, Inf)
        err = χ_norm > 1e-13 ? Δχ_norm / χ_norm : Δχ_norm

        @. χ_k = χ_k + Δχ
        update_structure_bc!(χ_k, body, sol.i, sol.t)

        @. ζ_k = -ζ + 2 / dt * (χ_k - χ)
        @. ζdot_k = 4 / dt^2 * (χ_k - χ) - 4 / dt * ζ - ζdot

        state_k = StructuralState(χ_k, ζ_k, ζdot_k)
        update_structure!(sol.points, state_k, body, coupler.ops, sol.i, sol.t)
        update_structure_operators!(ops, body, sol.points, state_k, sol.dt)
        update_weights!(sol.reg, grid, sol.points.x, i_deform)
        update_redist_weights!(sol)

        if err < coupler.tol
            break
        end

        it += 1
    end

    χ .= χ_k
    ζ .= ζ_k
    ζdot .= ζdot_k

    nothing
end


"""
    projection_step!(sol::CNAB)

Project the fluid vorticity field to remove the effect of the applied body forces (`f_tilde`),
ensuring that the flow field satisfies the updated constraints after force spreading and redistribution.

# Arguments
- `sol::CNAB`: The CNAB solver object containing the fluid and body state.

# Effects
- Modifies `sol.state.ω` in-place to account for the applied body forces.
- Swaps `sol.state.ω` and `sol.state.ψ` internally to reuse memory.

# Returns
- `nothing`: The projection modifies the solver state in-place.
"""
projection_step!(sol::CNAB) = projection_step!(sol, sol.prob.formulation)

function projection_step!(sol::CNAB{N,T}, ::FastIBPM) where {N,T}
    grid = sol.prob.grid
    backend = get_backend(sol.state.u[1][1])

    u_axes = cell_axes(grid, Loc_u, IncludeBoundary())
    ω_axes = cell_axes(grid, Loc_ω, ExcludeBoundary())

    with_arrays(sol.fluid_pool, (T, u_axes), (T, ω_axes)) do u_work, ω_work
        regularize!(u_work, sol.reg, sol.f_tilde)
        rot!(ω_work, u_work; h=grid.h)
        Ainv(sol, 1)(ω_work, ω_work)

        (sol.state.ω, sol.state.ψ) = (sol.state.ψ, sol.state.ω)

        for i in eachindex(ω_work)
            let ω = sol.state.ω[1][i], ω_work = ω_work[i]
                @loop backend (I in CartesianIndices(ω_work)) begin
                    ω[I] -= ω_work[I]
                end
            end
        end
    end
end

"""
    apply_vorticity!(sol::CNAB)

Compute the fluid velocity field from the current vorticity (`ω`) and update the solution 
to satisfy boundary conditions and base flow.

# Arguments
- `sol::CNAB`: The CNAB solver object containing the fluid and body state.

# Effects
- Updates `sol.state.u` (velocity field) and ensures `sol.state.ω` satisfies boundary conditions.
- Handles all multigrid levels, applying necessary interpolations between levels.

# Returns
- `nothing`: Modifies the solver state in-place.
"""
function apply_vorticity!(sol::CNAB{N,T}) where {N,T}
    with_arrays(sol.bndry_pool, (T, sol.state.ω_bndry)) do ψ_b_work
        apply_vorticity!(sol, ψ_b_work)
    end
end

function apply_vorticity!(sol::CNAB, ψ_b_work)
    grid = sol.prob.grid
    multidomain_poisson!(sol.state.ω, sol.state.ψ, sol.state.u, ψ_b_work, grid, sol.state.plan)

    for level in 1:grid.levels
        if level == grid.levels
            for i in eachindex(ψ_b_work)
                foreach(b -> fill!(b, 0), ψ_b_work[i])
            end
        else
            multidomain_interpolate!(ψ_b_work, sol.state.ω[level+1]; n=grid.n)
        end

        set_boundary!(sol.state.ω[level], ψ_b_work)

        add_flow!(sol.state.u[level], sol.prob.u0, grid, level, sol.i, sol.t)
    end
end

# ===========================================================================
# Primitive-variable (IBPM) time step — Taira & Colonius (2007), Eqs. 25-27
# ===========================================================================
#
# One CNAB step of the fractional-step / projection method:
#
#   (1) prediction  A q*      = r1              q* = Bᴺ r1
#   (2) coupling    QᵀBᴺQ λ   = Qᵀ q* - r2      λ = (φ, f_tilde)
#   (3) projection  q^{n+1}   = q* - Bᴺ Q λ
#
# with A = (1/Δt)I - (1/2Re)L. The convective term is taken in *rotational form*,
# `u×ω` (= `nonlinear!`), so the "pressure" φ is the total pressure p + ½|u|².
# There is no dt on the nonlinear/mass terms of r1: the dt lives in Bᴺ = `Ainv!`.

function _step!(sol::CNAB, f::IBPM)
    prediction_step!(sol, f)
    coupling_step!(sol, f)
    projection_step!(sol, f)
    recover_velocity!(sol, f)
end

function prediction_step!(sol::CNAB{N,T}, ::IBPM) where {N,T}
    st = sol.state
    grid = sol.prob.grid
    h = grid.h
    dt = sol.dt
    a = _A_factor(sol)
    ν = one(T) / sol.prob.Re
    β = sol.β
    backend = get_backend(st.q[1])
    interior(i) = CartesianIndices(cell_axes(grid, Loc_u(i), ExcludeBoundary()))

    _cycle!(st.nonlin)

    # Vorticity of the current physical velocity. The far-field flow is
    # irrotational, so the boundary vorticity is zero; only the interior is
    # filled from `rot!` (which needs the neighbours `u_full` provides).
    for i in eachindex(st.ω)
        fill!(st.ω[i], 0)
    end
    rot!(grid_view(st.ω, grid, Loc_ω, ExcludeBoundary()), st.u_full; h)

    # r1 = (ν/2) L q^n   (homogeneous Laplacian; boundary part added via bc1 below)
    r1 = st.r1
    q = st.q
    _apply_aL!(r1, q, ν / 2, h)

    nonlin_full = st.nonlin_count == length(st.nonlin)

    for i in eachindex(r1)
        R = interior(i)
        r1i, qi = r1[i], q[i]
        @loop backend (I in R) r1i[I] += qi[I] / dt          # + (1/Δt) q^n
        if nonlin_full                                        # + Σ β[end-k] N^{n-k}
            for k in eachindex(st.nonlin)                     #   (old history, pre-overwrite)
                Nk, c = st.nonlin[k][i], β[end-k]
                @loop backend (I in R) r1i[I] += c * Nk[I]
            end
        end
    end

    # New nonlinear term N^n = u×ω, overwriting the newest history slot.
    nonlinear!(st.nonlin[end], st.u_full, st.ω)
    cnew = nonlin_full ? β[end] : one(T)                      # AB2 weight, or 1 (Euler) on step 1
    for i in eachindex(r1)
        r1i, Ni = r1[i], st.nonlin[end][i]
        @loop backend (I in interior(i)) r1i[I] += cnew * Ni[I]
    end

    # bc1: viscous boundary contribution of the prescribed ∂D velocity. Both
    # Crank-Nicolson half-terms (levels n and n+1) coincide for a steady free
    # stream, giving coefficient ν, i.e. factor ν/h².
    #
    # `add_laplacian_bc!` locates the boundary from `axes`, so it must be handed
    # the *interior-unknown* view (ExcludeBoundary), not the haloed `r1`: with the
    # haloed axes it would write into the ∂D/halo face itself instead of the
    # interior node adjacent to it.
    r1_interior = map(a -> @view(a[CartesianIndices(Base.IdentityUnitRange.(_interior_range(a)))]), r1)
    add_laplacian_bc!(r1_interior, Loc_u, ν / h^2, background_velocity(sol.prob.u0, sol.t), grid)

    # Intermediate velocity q* = A⁻¹ r1 ≈ Bᴺ r1.
    Ainv!(st.q_star, r1, st.work.term, st.work.tmp; a, dt, n_taylor=st.n_taylor, h)

    st.nonlin_count = min(st.nonlin_count + 1, length(st.nonlin))
    sol
end

coupling_step!(sol::CNAB, f::IBPM) = _coupling_step!(sol, sol.coupler, f)

function _coupling_step!(sol::CNAB{N,T}, coupler::PrescribedBodyCoupler, ::IBPM) where {N,T}
    st = sol.state
    grid = sol.prob.grid
    h = grid.h
    body = sol.prob.body

    # Refresh body geometry (moving bodies) and the prescribed ∂D velocity.
    update_body_points!(sol.points, body, sol.i, sol.t)
    update_reg!(sol, body, eachindex(sol.points.x))
    update_redist_weights!(sol)
    # Inflow + lateral boundaries: Dirichlet free stream. Outflow: convective
    # (overwrites the outlet face of `ub` computed above).
    set_velocity_boundary!(st.ub, grid, background_velocity(sol.prob.u0, sol.t))
    update_outflow!(sol, IBPM())

    # RHS of the modified Poisson: Qᵀ q* - r2.
    #   pressure block:  -D q*  -  D∂ u_BC   =  -D_full q*   (removes q*'s divergence)
    #   force block:      E q*  -  u_B                        (no-slip residual)
    QT_mul!(st.rhs_φ, st.rhs_f, st.q_star, sol.reg; h)   # (-D q*, E q*)
    divergence_bc!(st.rhs_φ, -1 / h, st.ub)              # add -D∂ u_BC
    st.rhs_f .-= sol.points.u

    # Solve QᵀBᴺQ (φ, f_tilde) = (rhs_φ, rhs_f), warm-started from the previous λ.
    coupler.Binv(
        st.φ, sol.f_tilde, st.rhs_φ, st.rhs_f, sol.reg, st.work, IBPM();
        h, a=_A_factor(sol), dt=sol.dt, n_taylor=st.n_taylor,
    )
    sol
end

function projection_step!(sol::CNAB{N,T}, ::IBPM) where {N,T}
    st = sol.state
    grid = sol.prob.grid
    h = grid.h
    dt = sol.dt
    a = _A_factor(sol)
    backend = get_backend(st.q[1])

    # Bᴺ Q λ, then q^{n+1} = q* - Bᴺ Q λ (interior unknowns).
    Q_mul!(st.work.q, st.φ, sol.f_tilde, sol.reg; h)
    Ainv!(st.work.y, st.work.q, st.work.term, st.work.tmp; a, dt, n_taylor=st.n_taylor, h)
    for i in eachindex(st.q)
        R = CartesianIndices(cell_axes(grid, Loc_u(i), ExcludeBoundary()))
        qi, qsi, yi = st.q[i], st.q_star[i], st.work.y[i]
        @loop backend (I in R) qi[I] = qsi[I] - yi[I]
    end
    sol
end

"""
    recover_velocity!(sol::CNAB, ::IBPM)

Rebuild the physical velocity `u_full` from the interior unknowns `q^{n+1}` and
the prescribed boundary values, so the next step's `rot!`/`nonlinear!` (whose
stencils reach ∂D) have a complete field. The far-field flow fills the whole
field; the interior is then overwritten with `q`.
"""
function recover_velocity!(sol::CNAB{N,T}, ::IBPM) where {N,T}
    st = sol.state
    grid = sol.prob.grid
    backend = get_backend(st.u_full[1])

    for i in eachindex(st.u_full)
        fill!(st.u_full[i], 0)
    end
    add_flow!(st.u_full, sol.prob.u0, grid, 1, sol.i, sol.t)
    for i in eachindex(st.u_full)
        R = CartesianIndices(cell_axes(grid, Loc_u(i), ExcludeBoundary()))
        ui, qi = st.u_full[i], st.q[i]
        @loop backend (I in R) ui[I] = qi[I]
    end
    # Carry the convected outflow velocity into the physical field (add_flow!
    # above overwrote it with the free stream). This is `u_out^{n+1}`, which the
    # next step reads as `u_out^n` for the convective update.
    let of = st.ub[1][2, 1], uo = st.u_full[1]
        @loop backend (I in CartesianIndices(of)) uo[I] = of[I]
    end
    sol
end

"""
    update_outflow!(sol::CNAB, ::IBPM)

Advance the outflow boundary velocity by the convective (Orlanski-type) condition
`∂u/∂t + u∞ ∂u/∂x = 0` of Taira & Colonius (2007) §5.3, so vorticity leaves the
domain freely instead of being pinned to the free stream.

Specialized to a free stream in `+x` (hence the high-x boundary is the outlet).
The outlet normal velocity is advanced explicitly from the previous step,

    u_out^{n+1} = u_out^n - u∞ (Δt/h) (u_out^n - u_interior^n),

where `u_out^n` is carried in `u_full`'s outlet face (set by
[`recover_velocity!`](@ref)) and `u_interior^n` is the velocity one cell inside.
The result overwrites the outlet face of the boundary buffer `ub`, so the
divergence BC (`bc2`) sees the convected outflow.

!!! note
    `bc1` still uses the free-stream value at the outlet (it reads a function, not
    `ub`). Since the viscous boundary term is small and the projection removes
    divergence regardless, this only mildly affects near-outlet accuracy, not the
    divergence-free property. Extending `bc1` to the convected value needs the
    buffer-based `bc1` noted in `set_velocity_boundary!`.
"""
function update_outflow!(sol::CNAB{N,T}, ::IBPM) where {N,T}
    st = sol.state
    grid = sol.prob.grid
    h = grid.h
    dt = sol.dt
    U = background_velocity(sol.prob.u0, sol.t)(zero(SVector{N,T}))[1]  # free-stream speed
    face = st.ub[1][2, 1]        # high-x outlet: u[1] normal face at x = x_max
    uf = st.u_full[1]
    backend = get_backend(face)
    @loop backend (I in CartesianIndices(face)) begin
        δ = axisunit(I)
        u_prev = uf[I]           # u_out^n (carried in the physical field's outlet face)
        u_int = uf[I-δ(1)]       # interior velocity one cell inside the outlet
        face[I] = u_prev - U * dt / h * (u_prev - u_int)
    end
    sol
end

"""
    ab_coeffs(T, n)

Return the Adams-Bashforth coefficients for a given order `n`.

# Arguments
- `T`: Container type to hold the coefficients (e.g., Tuple or Vector).
- `n`: Order of the Adams-Bashforth scheme (currently only 1 or 2).

# Returns
- A container of type `T` with the AB coefficients:
  - `n = 1`: `[1]` (forward Euler, AB1)
  - `n = 2`: `[-1//2, 3//2]` (AB2)

# Notes
- AB1 is first-order explicit Euler.
- AB2 is second-order, using current and previous derivative values for better accuracy.
- Only `n=1` and `n=2` are supported; other values throw a `DomainError`.

# Example
```julia
ab_coeffs(Tuple, 1)  # returns (1,)
ab_coeffs(Tuple, 2)  # returns (-1//2, 3//2)
```
"""
function ab_coeffs(T, n)
    if n == 1
        T[1]
    elseif n == 2
        T[-1//2, 3//2]
    else
        throw(DomainError(n, "only n=1 and n=2 are supported"))
    end
end
