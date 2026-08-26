"""
Primitive-variable (`IBPM`) time-stepping pipeline — Taira & Colonius (2007).

Contains both grid paths: the uniform `Grid` pipeline and the parallel
mass-weighted `StretchedGrid` one (the `_*_stretched!` routines).

Shared machinery is in `cnab_common.jl`; the operators are in `ops_ibpm.jl`.
"""

"""
    _step!(sol::CNAB, ::IBPM)

Primitive-variable time-stepping pipeline — Taira & Colonius (2007), Eqs. 25-27.

One CNAB step of the fractional-step / projection method:

  1. prediction  `A q* = r1`,            i.e. `q* = Bᴺ r1`
  2. coupling    `QᵀBᴺQ λ = Qᵀ q* - r2`, with `λ = (φ, f_tilde)`
  3. projection  `q^{n+1} = q* - Bᴺ Q λ`

with `A = (1/Δt)I - (1/2Re)L`. The convective term is taken in *rotational form*,
`u×ω` (= [`nonlinear!`](@ref)), so the "pressure" `φ` is the total pressure
`p + ½|u|²`. There is no `Δt` on the nonlinear/mass terms of `r1`: the `Δt` lives
in `Bᴺ` = [`Ainv!`](@ref).

The step dispatches on the grid type: the uniform `Grid` pipeline, and a
parallel mass-weighted one for a `StretchedGrid` (the `_*_stretched!` routines in
the second half of this file, with the operators in `stretched_domain.jl`).
"""
_step!(sol::CNAB, f::IBPM) = _step!(sol, f, sol.prob.grid)

function _step!(sol::CNAB, f::IBPM, ::Grid)
    prediction_step!(sol, f)
    coupling_step!(sol, f)
    projection_step!(sol, f)
    recover_velocity!(sol, f)
end

function _step!(sol::CNAB, f::IBPM, grid::StretchedGrid)
    _prediction_stretched!(sol, grid)
    _coupling_stretched!(sol, grid)
    _projection_stretched!(sol, grid)
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

# ===========================================================================
# Primitive-variable (IBPM) time step — StretchedGrid pipeline
# ===========================================================================
#
# Parallel to the uniform IBPM stepping above, but on a stretched grid: the RHS
# `r1` is mass-weighted (`(1/Δt)M qⁿ + M·Nⁿ`), the operators are the symmetric FV
# ones, and the boundary folds / outflow use the local spacing. The uniform `Grid`
# stepping is untouched; these run only for a `StretchedGrid`.

function _prediction_stretched!(sol::CNAB{N,T}, grid::StretchedGrid) where {N,T}
    st = sol.state
    dt = sol.dt
    a = _A_factor(sol)
    ν = one(T) / sol.prob.Re
    β = sol.β
    backend = get_backend(st.q[1])
    interior(i) = CartesianIndices(cell_axes(grid, Loc_u(i), ExcludeBoundary()))

    _cycle!(st.nonlin)

    # Vorticity ω = ∇×uⁿ for the rotational-form convection (stretched curl).
    for i in eachindex(st.ω)
        fill!(st.ω[i], 0)
    end
    rot!(grid_view(st.ω, grid, Loc_ω, ExcludeBoundary()), st.u_full, grid)

    # r1 = (ν/2) L_sym qⁿ   (homogeneous; boundary part added via bc1 below)
    r1 = st.r1
    q = st.q
    _apply_aL!(r1, q, ν / 2, grid)

    nonlin_full = st.nonlin_count == length(st.nonlin)

    # + (1/Δt) M qⁿ  and  + Σ β[end-k] M Nⁿ⁻ᵏ   (mass-weighted RHS terms)
    for i in eachindex(r1)
        R = interior(i)
        r1i, qi = r1[i], q[i]
        @loop backend (I in R) r1i[I] += mass(grid, i, I) * qi[I] / dt
        if nonlin_full
            for k in eachindex(st.nonlin)
                Nk, c = st.nonlin[k][i], β[end-k]
                @loop backend (I in R) r1i[I] += mass(grid, i, I) * c * Nk[I]
            end
        end
    end

    nonlinear!(st.nonlin[end], st.u_full, st.ω)
    cnew = nonlin_full ? β[end] : one(T)
    for i in eachindex(r1)
        r1i, Ni = r1[i], st.nonlin[end][i]
        @loop backend (I in interior(i)) r1i[I] += mass(grid, i, I) * cnew * Ni[I]
    end

    # bc1: viscous boundary contribution (stretched L_sym coefficients).
    r1_interior = map(a -> @view(a[CartesianIndices(Base.IdentityUnitRange.(_interior_range(a)))]), r1)
    viscous_bc!(r1_interior, ν, background_velocity(sol.prob.u0, sol.t), grid)

    # Intermediate velocity q* = A⁻¹ r1 ≈ Bᴺ r1.
    Ainv!(st.q_star, r1, st.work.term, st.work.tmp, grid; a, dt, n_taylor=st.n_taylor)

    st.nonlin_count = min(st.nonlin_count + 1, length(st.nonlin))
    sol
end

function _coupling_stretched!(sol::CNAB{N,T}, grid::StretchedGrid) where {N,T}
    st = sol.state
    body = sol.prob.body

    update_body_points!(sol.points, body, sol.i, sol.t)
    update_reg!(sol, body, eachindex(sol.points.x))
    update_redist_weights!(sol)
    set_velocity_boundary!(st.ub, grid, background_velocity(sol.prob.u0, sol.t))
    _update_outflow_stretched!(sol, grid)

    # RHS of the modified Poisson: Qᵀ q* - r2.
    QT_mul!(st.rhs_φ, st.rhs_f, st.q_star, sol.reg, grid)   # (-D q*, E q*)
    continuity_bc!(st.rhs_φ, st.ub, grid)                   # add -D∂ u_BC
    st.rhs_f .-= sol.points.u

    sol.coupler.Binv(
        st.φ, sol.f_tilde, st.rhs_φ, st.rhs_f, sol.reg, st.work, IBPM(), grid;
        a=_A_factor(sol), dt=sol.dt, n_taylor=st.n_taylor,
    )
    sol
end

function _projection_stretched!(sol::CNAB{N,T}, grid::StretchedGrid) where {N,T}
    st = sol.state
    dt = sol.dt
    a = _A_factor(sol)
    backend = get_backend(st.q[1])

    Q_mul!(st.work.q, st.φ, sol.f_tilde, sol.reg, grid)
    Ainv!(st.work.y, st.work.q, st.work.term, st.work.tmp, grid; a, dt, n_taylor=st.n_taylor)
    for i in eachindex(st.q)
        R = CartesianIndices(cell_axes(grid, Loc_u(i), ExcludeBoundary()))
        qi, qsi, yi = st.q[i], st.q_star[i], st.work.y[i]
        @loop backend (I in R) qi[I] = qsi[I] - yi[I]
    end
    sol
end

# Convective (Orlanski) outflow, stretched: uses the local outlet cell width.
function _update_outflow_stretched!(sol::CNAB{N,T}, grid::StretchedGrid) where {N,T}
    st = sol.state
    dt = sol.dt
    U = background_velocity(sol.prob.u0, sol.t)(zero(SVector{N,T}))[1]
    face = st.ub[1][2, 1]        # high-x outlet
    uf = st.u_full[1]
    backend = get_backend(face)
    @loop backend (I in CartesianIndices(face)) begin
        δ = axisunit(I)
        u_prev = uf[I]
        u_int = uf[I-δ(1)]
        Δx = cell_width(grid, 1, I[1] - 1)   # last interior cell width at the outlet
        face[I] = u_prev - U * dt / Δx * (u_prev - u_int)
    end
    sol
end
