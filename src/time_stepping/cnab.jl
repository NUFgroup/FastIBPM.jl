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
