"""
Manifold-projection (`IMAP`) time-stepping pipeline.

Shared machinery is in `cnab_common.jl`; the operators are in `ops_imap.jl`.

Note this file depends on `cnab_ibpm.jl` at *run time*: `recover_velocity!` and
`update_outflow!` are delegated to their `IBPM` methods rather than duplicated,
since neither depends on how no-slip is enforced. Method resolution is dynamic,
so the include order of the two files does not matter.
"""

"""
    _step!(sol::CNAB, ::IMAP)

Manifold-projection time-stepping pipeline: one CNAB step of the fractional-step
method with no-slip enforced by projection instead of by a boundary force.

  1. prediction  `A q* = r1`, i.e. `q* = Bᴺ r1`   — [`prediction_step!`](@ref)
  2. pressure    `GᵀBᴺG φ = Gᵀ q* - r2`           — [`pressure_step!`](@ref)
  3. correction  `q^{n+1} = q* - Bᴺ G φ`          — [`correction_step!`](@ref)
  4. force       `f̃ = (RᵀR)⁻¹Rᵀ[…] - (RᵀR)⁻¹Rᵀ[G φ]` — [`recover_force!`](@ref),
     accumulated across stages 1, 3 and 4, each of which already forms the
     multiplier it needs.

with `A = (1/Δt)I - (1/2Re) P L` and `Bᴺ` the projected viscous inverse
([`Ainv_IMAP!`](@ref)). Compared with the `IBPM` pipeline in `cnab_ibpm.jl`, the
multiplier has lost its force block (`λ = φ`), so stage 2 is grid-sized and
stage 3 spreads no force — the body enters only through `P`. Stage 4 is
diagnostic: `f̃` is an output, not an unknown, so it never feeds back into the
solve.

!!! note "Naming"
    Stage 3 is what `IBPM`/`FastIBPM` call `projection_step!` — the
    fractional-step pressure projection. It is named [`correction_step!`](@ref)
    here, as in the IMAP document, because in this formulation "projection"
    already means the *constraint* projection `P`, and the two must not be
    confused: `P` enforces no-slip and lives inside stages 1 and 2, while this
    one enforces incompressibility.

IMAP is uniform-`Grid` only — `formulation_state` refuses anything else at
construction — so unlike `IBPM` there is no grid dispatch here.
"""

function _step!(sol::CNAB, f::IMAP)
    prediction_step!(sol, f)
    pressure_step!(sol, f)
    correction_step!(sol, f)
    recover_force!(sol, f)
    # `u_full` = interior unknowns + prescribed ∂D values. Not a change of
    # variables — `q` deliberately carries *zero* boundary/halo values because the
    # operators above are the homogeneous ones (the physical BCs live on the
    # right-hand side as bc1/bc2), while the next step's `rot!`/`nonlinear!`
    # stencils reach ∂D and need the real values there.
    recover_velocity!(sol, f)
end

"""
    prediction_step!(sol::CNAB, ::IMAP)

Momentum prediction for `IMAP`: form

    r1 = (1/Δt) qⁿ + P[ (ν/2) L qⁿ + N^AB2 + bc1 ]

and solve `A q* = r1` as `q* = Bᴺ r1` ([`Ainv_IMAP!`](@ref)).

Structurally this is the `IBPM` [`prediction_step!`](@ref) with one projection
inserted. Note **`(1/Δt) qⁿ` is outside the projection** (the document's Eq. 4),
which is what lets the whole right-hand side be built exactly as `IBPM` builds it,
projected once, and then have the mass term added — one `P` per step rather than
one per term.

The Adams-Bashforth history is stored *unprojected*: `P` is linear, so projecting
the combination `3/2 Nⁿ - 1/2 Nⁿ⁻¹` is identical to projecting each term, and the
raw `N` then means the same thing in both formulations.

The multiplier discarded by that projection is stashed in `sol.f_tilde`; see
[`recover_force!`](@ref).
"""
function prediction_step!(sol::CNAB{N,T}, ::IMAP) where {N,T}
    st = sol.state
    grid = sol.prob.grid
    h = grid.h
    dt = sol.dt
    a = _A_factor(sol)
    ν = one(T) / sol.prob.Re
    β = sol.β
    backend = get_backend(st.q[1])
    cpl = sol.coupler.Binv
    interior(i) = CartesianIndices(cell_axes(grid, Loc_u(i), ExcludeBoundary()))

    _cycle!(st.nonlin)

    # Vorticity of the current physical velocity (the far field is irrotational,
    # so only the interior is filled) — for the rotational-form convection.
    for i in eachindex(st.ω)
        fill!(st.ω[i], 0)
    end
    rot!(grid_view(st.ω, grid, Loc_ω, ExcludeBoundary()), st.u_full; h)

    # --- the terms that get projected ---------------------------------------
    # r1 = (ν/2) L qⁿ   (homogeneous Laplacian; boundary part added via bc1 below)
    r1 = st.r1
    q = st.q
    _apply_aL!(r1, q, ν / 2, h)

    nonlin_full = st.nonlin_count == length(st.nonlin)

    if nonlin_full                                             # + Σ β[end-k] N^{n-k}
        for i in eachindex(r1)
            R = interior(i)
            r1i = r1[i]
            for k in eachindex(st.nonlin)                       #   (old history, pre-overwrite)
                Nk, c = st.nonlin[k][i], β[end-k]
                @loop backend (I in R) r1i[I] += c * Nk[I]
            end
        end
    end

    # New nonlinear term N^n = u×ω, overwriting the newest history slot.
    nonlinear!(st.nonlin[end], st.u_full, st.ω)
    cnew = nonlin_full ? β[end] : one(T)                       # AB2 weight, or 1 (Euler) on step 1
    for i in eachindex(r1)
        r1i, Ni = r1[i], st.nonlin[end][i]
        @loop backend (I in interior(i)) r1i[I] += cnew * Ni[I]
    end

    # bc1: viscous boundary contribution of the prescribed ∂D velocity, added
    # *before* the projection (the document's `P bc1`). As in the IBPM method,
    # `add_laplacian_bc!` needs the interior-unknown view, not the haloed field.
    r1_interior = map(a -> @view(a[CartesianIndices(Base.IdentityUnitRange.(_interior_range(a)))]), r1)
    add_laplacian_bc!(r1_interior, Loc_u, ν / h^2, background_velocity(sol.prob.u0, sol.t), grid)

    # --- the projection, applied once to the whole bundle --------------------
    P_mul!(r1, cpl.proj)
    # Stash the multiplier before the projections inside `Ainv_IMAP!` overwrite
    # it: this is the qⁿ half of the boundary force (see `recover_force!`).
    sol.f_tilde .= cpl.proj.λ

    # --- the mass term, which Eq. 4 leaves unprojected ------------------------
    for i in eachindex(r1)
        r1i, qi = r1[i], q[i]
        @loop backend (I in interior(i)) r1i[I] += qi[I] / dt
    end

    # Intermediate velocity q* = A⁻¹ r1 ≈ Bᴺ r1, with the projected series.
    Ainv_IMAP!(
        st.q_star, r1, st.work.term, st.work.tmp, cpl.proj;
        a, dt, n_taylor=st.n_taylor, h, symmetric=cpl.symmetric,
    )

    st.nonlin_count = min(st.nonlin_count + 1, length(st.nonlin))
    sol
end

"""
    pressure_step!(sol::CNAB, ::IMAP)

Solve the `IMAP` pressure system

    Gᵀ Bᴺ G φ = Gᵀ q* - r2,     r2 = bc2 = D∂ u_BC

This occupies the same slot as [`coupling_step!`](@ref) in the `IBPM` pipeline but
is deliberately *not* a coupling step: with no boundary force among the unknowns
there is nothing to couple here, and the body reaches this solve only through the
`P` inside `Bᴺ`. Hence the different name.

`r2` carries **no projection**: `bc2` is the divergence of the prescribed boundary
velocity and lives in pressure space, where the velocity-space projector `P` does
not act.
"""
function pressure_step!(sol::CNAB{N,T}, ::IMAP) where {N,T}
    st = sol.state
    grid = sol.prob.grid
    h = grid.h
    body = sol.prob.body
    cpl = sol.coupler.Binv

    # Refresh body geometry (moving bodies) and the prescribed ∂D velocity.
    update_body_points!(sol.points, body, sol.i, sol.t)
    update_reg!(sol, body, eachindex(sol.points.x))
    # Inflow + lateral boundaries: Dirichlet free stream. Outflow: convective
    # (overwrites the outlet face of `ub` set just above).
    set_velocity_boundary!(st.ub, grid, background_velocity(sol.prob.u0, sol.t))
    update_outflow!(sol, IMAP())

    # RHS: Gᵀ q* - r2 = -D q* - D∂ u_BC = -D_full q*  (removes q*'s divergence).
    GT_mul!(st.rhs_φ, st.q_star; h)
    divergence_bc!(st.rhs_φ, -1 / h, st.ub)

    # Solve GᵀBᴺG φ = rhs_φ, warm-started from the previous pressure.
    cpl.solve(
        st.φ, st.rhs_φ, cpl.proj, st.work, IMAP();
        h, a=_A_factor(sol), dt=sol.dt, n_taylor=st.n_taylor, symmetric=cpl.symmetric,
    )
    sol
end

"""
    correction_step!(sol::CNAB, ::IMAP)

Correct the intermediate velocity onto the divergence-free space:

    q^{n+1} = q* - Bᴺ G φ

The IMAP counterpart of [`projection_step!`](@ref) — renamed because in this
formulation "projection" already means the constraint projector `P` (see the
naming note at the top of this section). The `IBPM` version subtracts
`Bᴺ Q λ = Bᴺ(G φ + Eᵀ f̃)`; with the force block gone nothing is spread onto the
grid, because no-slip was already enforced by the projections in the prediction
and inside `Bᴺ`.
"""
function correction_step!(sol::CNAB{N,T}, ::IMAP) where {N,T}
    st = sol.state
    grid = sol.prob.grid
    h = grid.h
    dt = sol.dt
    a = _A_factor(sol)
    backend = get_backend(st.q[1])
    cpl = sol.coupler.Binv

    G_mul!(st.work.q, st.φ; h)
    P_mul!(st.work.q, cpl.proj)      # the projected gradient: see `B_mul!(…, ::IMAP)`
    # The multiplier just discarded is the *pressure* part of the boundary force —
    # the body's reaction to the surface pressure, and the dominant term in bluff-
    # body drag. It is free here and expensive to recompute, so stash it now; see
    # `recover_force!` for where it belongs in the force balance.
    sol.f_tilde .-= cpl.proj.λ
    Ainv_IMAP!(
        st.work.y, st.work.q, st.work.term, st.work.tmp, cpl.proj;
        a, dt, n_taylor=st.n_taylor, h, symmetric=cpl.symmetric,
    )
    for i in eachindex(st.q)
        R = CartesianIndices(cell_axes(grid, Loc_u(i), ExcludeBoundary()))
        qi, qsi, yi = st.q[i], st.q_star[i], st.work.y[i]
        @loop backend (I in R) qi[I] = qsi[I] - yi[I]
    end
    sol
end

"""
    recover_force!(sol::CNAB, ::IMAP)

Recover the immersed-boundary force `f̃` that IMAP never solves for, and store it
in `sol.f_tilde` so the ordinary [`surface_force!`](@ref) / [`surface_force_sum`](@ref)
post-processing works unchanged.

# Where it comes from

Ask what force would make the IMAP solution satisfy the *`IBPM`* momentum equation
`A_IBPM q^{n+1} + G φ + Eᵀ f̃ = r1_IBPM`. Subtracting the equation actually solved
(the projected one) makes every shared term cancel and leaves

    Eᵀ f̃ = (I - P)[ (ν/2) L(qⁿ + q^{n+1}) + N^AB2 + bc1 ]  -  (I - P) G φ,

i.e. exactly the momentum the projections threw away. Since
`(I - P)w = R(RᵀR)⁻¹Rᵀw`, each piece is the constraint multiplier of its bracket,
and the resulting `f̃` is the *same* quantity `IBPM` solves for — same
normalization, hence the shared `_f_tilde_factor`.

!!! warning "The pressure term is not optional"
    That second term exists because the pressure gradient is projected too (see
    `B_mul!(…, ::IMAP)`): `P` discards the off-manifold part of `G φ`, so the
    constraint force must supply it. It is the **pressure force on the body**, and
    for a bluff body it dominates — dropping it underpredicts the drag on a
    cylinder at Re = 40 by a factor of ~3.6 while leaving the velocity field (and
    so the wake length) visibly perfect, which makes it a silent error.

# Why it is nearly free

Every bracket is linear in the multiplier, and two of the three are computed
anyway by projections that already happen:

  - the `qⁿ` viscous/convective half is the multiplier `P` forms during
    [`prediction_step!`](@ref) — stashed there, no work at all;
  - the `G φ` term is the multiplier `P` forms when it projects the gradient in
    [`correction_step!`](@ref) — likewise stashed, no work at all;
  - only the `q^{n+1}` half is computed here: one Laplacian apply plus one
    [`constraint_multiplier!`](@ref) (an interpolation and an `N·n_b`
    back-substitution against the factorization the projector already holds).

No spreading, no Krylov solve, nothing grid-sized beyond the single Laplacian.
This is a diagnostic: `f̃` never re-enters the solve.
"""
function recover_force!(sol::CNAB{N,T}, ::IMAP) where {N,T}
    st = sol.state
    ν = one(T) / sol.prob.Re
    h = sol.prob.grid.h
    proj = sol.coupler.Binv.proj

    # (ν/2) L q^{n+1} into scratch (free now that `Ainv_IMAP!` is done), then its
    # multiplier, added to the qⁿ half stashed by the prediction step.
    _apply_aL!(st.work.tmp, st.q, ν / 2, h)
    with_arrays_like(sol.body_pool, sol.f_tilde) do λ
        constraint_multiplier!(λ, proj, st.work.tmp)
        sol.f_tilde .+= λ
    end
    sol
end

# Reassembling the physical velocity and advancing the convective outflow do not
# depend on how no-slip is enforced: both touch only `q`, `u_full` and `ub`, which
# `IMAPState` shares with `IBPMState`. Delegated rather than duplicated so the two
# formulations cannot drift apart.
recover_velocity!(sol::CNAB, ::IMAP) = recover_velocity!(sol, IBPM())
update_outflow!(sol::CNAB, ::IMAP) = update_outflow!(sol, IBPM())
