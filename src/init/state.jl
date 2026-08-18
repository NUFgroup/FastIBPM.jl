"""
Solver state for the immersed-boundary formulations.

The `CNAB` integrator (see `problems.jl`) carries only what is common to every
formulation — the clock, the Adams-Bashforth weights, the body coupling and the
memory pools. Everything *formulation-specific* — the unknowns themselves — lives
here, in `CNAB.state`:

  - [`FastIBPMState`](@ref) — streamfunction-vorticity. Unknowns are the vorticity
    `ω` and streamfunction `ψ` on the multi-level (multidomain) grids. Continuity
    is automatic (`u = ∇×ψ`), so there is no pressure.
  - [`IBPMState`](@ref) — primitive variables. Unknowns are the velocity `q` and a
    pressure `φ`; the Lagrange multiplier is `λ = (φ, f_tilde)`.
  - [`IMAPState`](@ref) — primitive variables with a manifold projection. Same
    unknowns as `IBPMState` **minus the boundary force**: no-slip is enforced by
    projecting onto the constraint manifold, so `λ = φ` alone.

[`formulation_state`](@ref) builds the right one, and [`initialize_fields!`](@ref)
resets it — each dispatched on `prob.formulation`, so no formulation ever
carries (or has to reason about) another's fields.
"""

# ===========================================================================
# Formulation-specific state
# ===========================================================================

"""
    FastIBPMState

Streamfunction-vorticity (`FastIBPM`) solver state, held in `CNAB.state`.

Its unknowns live in *vorticity* space on the multi-level (multidomain) grids:
`ω` is the vorticity, `ψ` the streamfunction, and `nonlin` the Adams-Bashforth
history of the **vorticity-transport** term `∇×(u×ω)`. Continuity is automatic
here (`u = ∇×ψ`), so there is no pressure and no divergence constraint. `plan`
holds the FFT plans for the spectral Laplacian inverse.

`ω`/`ψ` are mutable because `projection_step!` swaps them to reuse memory.

See also [`IBPMState`](@ref), the primitive-variable counterpart.
"""
mutable struct FastIBPMState{P,Aω,Au,W}
    const plan::P
    ω::Vector{Aω}
    ψ::Vector{Aω}
    const u::Vector{Au}
    const nonlin::Vector{Vector{Aω}}
    nonlin_count::Int
    ω_bndry::W
end

"""
    IBPMState

Primitive-variable (`IBPM`) solver state, held in `CNAB.state`.

Unlike [`FastIBPMState`](@ref), the unknowns live in *velocity* space and the
Lagrange multiplier `λ = (φ, f_tilde)` carries a **pressure** as well as the body
force. Two velocity representations are kept, because the projection operators and
the advection terms need different layouts:

  - `q`, `q_star`, `r1` — **haloed** fields (see `Ainv_zeros`) holding the interior
    unknowns with a **zero** halo. The projection operators (`Ainv!`, `Q_mul!`,
    `B_mul!`) are the *homogeneous* ones — the physical boundary values are carried
    separately on the right-hand side as `bc1`/`bc2` — so these must have a zero halo.
  - `u_full` — the ordinary `Loc_u` (`IncludeBoundary`) field holding the *physical*
    velocity: interior unknowns **plus** the prescribed boundary values. `rot!` and
    `nonlinear!` need this, since the advection stencils reach the domain boundary.

# Fields
- `q`, `q_star`, `r1` : velocity unknowns, intermediate velocity `q*`, momentum RHS.
- `u_full`, `ω`       : physical velocity and its vorticity (advection scratch).
- `φ`, `rhs_φ`        : pressure block of `λ`, and of the modified-Poisson RHS.
- `rhs_f`             : force block of the modified-Poisson RHS (`E q* - u_B`).
- `ub`                : prescribed velocity on ∂D (boundary buffer).
- `nonlin`            : Adams-Bashforth history of the **momentum** nonlinear term
                        (velocity space, `ExcludeBoundary`) — note this is `u×ω`
                        itself, not its curl as in `FastIBPMState`.
- `work`              : scratch for `B_mul!` / `Ainv!` (see `B_work`).
- `n_taylor`          : truncation order of `Bᴺ` (3 recommended).
"""
mutable struct IBPMState{Qh,Au,Aw,Ap,Vb,Ub,An,Wk}
    const q::Qh
    const q_star::Qh
    const r1::Qh
    const u_full::Au
    const ω::Aw
    const φ::Ap
    const rhs_φ::Ap
    const rhs_f::Vb
    const ub::Ub
    const nonlin::Vector{An}
    nonlin_count::Int
    const work::Wk
    const n_taylor::Int
end

function IBPMState(backend, grid::AbstractGrid{N,T}, n_ib, n_step; n_taylor=3) where {N,T}
    haloed() = Ainv_zeros(backend, grid)
    IBPMState(
        haloed(),
        haloed(),
        haloed(),
        grid_zeros(backend, grid, Loc_u),
        grid_zeros(backend, grid, Loc_ω),
        grid_zeros(backend, grid, Loc_p()),
        grid_zeros(backend, grid, Loc_p()),
        KernelAbstractions.zeros(backend, SVector{N,T}, n_ib),
        boundary_zeros(backend, grid, Loc_u),
        [grid_zeros(backend, grid, Loc_u, ExcludeBoundary()) for _ in 1:max(n_step - 1, 1)],
        0,
        B_work(backend, grid),
        n_taylor,
    )
end

"""
    IMAPState

Manifold-projection (`IMAP`) solver state, held in `CNAB.state`.

IMAP keeps the primitive-variable structure of [`IBPMState`](@ref) but **drops the
immersed-boundary force from the unknowns**: no-slip is enforced by projecting
onto the constraint manifold `Rᵀu = 0` (see `ManifoldProjection`) instead of
solving for an `f` that produces it. So where `IBPM` carries the Lagrange
multiplier `λ = (φ, f_tilde)`, here `λ = φ` alone, and the field `rhs_f` — the
force block of the modified-Poisson right-hand side — has no counterpart.

Everything else is deliberately identical to `IBPMState`, **including the field
names**: the two formulations share the same velocity layouts (haloed unknowns
`q`/`q_star`/`r1` with a zero halo, physical `u_full` with the boundary values)
and therefore share the reset routines [`zero_velocity!`](@ref) unchanged.

Note the projector itself is *not* stored here. It is an operator, not an
unknown, and — like the `FastIBPM` precomputed coupling inverse — it can only be
built once the regularization weights exist, so it lives in the coupler
(`sol.coupler.Binv.proj`, see `IMAPCoupling`).

# Fields
- `q`, `q_star`, `r1` : velocity unknowns, intermediate velocity `q*`, momentum RHS.
- `u_full`, `ω`       : physical velocity and its vorticity (advection scratch).
- `φ`, `rhs_φ`        : pressure, and the right-hand side of the pressure solve.
- `ub`                : prescribed velocity on ∂D (boundary buffer).
- `nonlin`            : Adams-Bashforth history of the momentum nonlinear term.
- `work`              : scratch for the viscous inverse and the pressure operator.
- `n_taylor`          : truncation order of `Bᴺ` (3 recommended).
"""
mutable struct IMAPState{Qh,Au,Aw,Ap,Ub,An,Wk}
    const q::Qh
    const q_star::Qh
    const r1::Qh
    const u_full::Au
    const ω::Aw
    const φ::Ap
    const rhs_φ::Ap
    const ub::Ub
    const nonlin::Vector{An}
    nonlin_count::Int
    const work::Wk
    const n_taylor::Int
end

# `n_ib` is accepted for interface symmetry with `IBPMState` (both are called
# positionally by `formulation_state`) but is unused *here*: the only body-sized
# object in IMAP is the projector `P = I - R(RᵀR)⁻¹Rᵀ`, and that is built in the
# coupler — `coupling_Binv` sizes `R` from `point_count(body)`. The state itself
# allocates no body-sized unknown, which is the whole point of the formulation.
function IMAPState(backend, grid::AbstractGrid{N,T}, n_ib, n_step; n_taylor=3) where {N,T}
    haloed() = Ainv_zeros(backend, grid)
    IMAPState(
        haloed(),
        haloed(),
        haloed(),
        grid_zeros(backend, grid, Loc_u),
        grid_zeros(backend, grid, Loc_ω),
        grid_zeros(backend, grid, Loc_p()),
        grid_zeros(backend, grid, Loc_p()),
        boundary_zeros(backend, grid, Loc_u),
        [grid_zeros(backend, grid, Loc_u, ExcludeBoundary()) for _ in 1:max(n_step - 1, 1)],
        0,
        B_work(backend, grid),
        n_taylor,
    )
end

# ===========================================================================
# Construction
# ===========================================================================

"""
    formulation_state(backend, grid, formulation, n_ib, n_step)

Build the formulation-specific unknowns stored in `CNAB.state`: a
[`FastIBPMState`](@ref) for `FastIBPM`, an [`IBPMState`](@ref) for `IBPM`.
"""
function formulation_state(backend, grid::Grid, ::FastIBPM, n_ib, n_step)
    ω = grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels)
    plan = let ωe = grid_view(ω[1], grid, Loc_ω, ExcludeBoundary())
        laplacian_plans(ωe, grid.n)
    end
    FastIBPMState(
        plan,
        ω,
        grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels),
        grid_zeros(backend, grid, Loc_u; levels=1:grid.levels),
        map(1:(n_step-1)) do _
            grid_zeros(backend, grid, Loc_ω, ExcludeBoundary(); levels=1:grid.levels)
        end,
        0,
        boundary_axes(grid, Loc_ω),
    )
end

formulation_state(backend, grid::AbstractGrid, ::IBPM, n_ib, n_step) =
    IBPMState(backend, grid, n_ib, n_step)

formulation_state(backend, grid::Grid, ::IMAP, n_ib, n_step) =
    IMAPState(backend, grid, n_ib, n_step)

# Guard: IMAP is currently uniform-grid only. The stretched operators are
# mass-weighted (`Bᴺ = Δt Σ (a M⁻¹ L_sym)ᵏ M⁻¹`), and the constraint projector
# would have to be made orthogonal in the *mass* inner product to match — that is
# a separate derivation, not a port, so it is deliberately not silently allowed.
formulation_state(backend, grid::AbstractGrid, ::IMAP, n_ib, n_step) = throw(
    ArgumentError(
        "IMAP currently supports only a uniform Grid (got $(typeof(grid))). The " *
        "mass-weighted manifold projection needed for a StretchedGrid is not " *
        "implemented yet; use the IBPM formulation for stretched grids.",
    ),
)

# Guard: the FastIBPM (streamfunction-vorticity) solver inverts the Laplacian
# spectrally via FFTs on the multidomain hierarchy, which requires uniform spacing.
# A StretchedGrid is only meaningful for the IBPM projection path.
formulation_state(backend, grid::StretchedGrid, ::FastIBPM, n_ib, n_step) = throw(
    ArgumentError(
        "StretchedGrid is only supported by the IBPM formulation; FastIBPM requires " *
        "a uniform Grid (its FFT/multidomain Poisson solver assumes uniform spacing).",
    ),
)

# ===========================================================================
# Initialization / reset
# ===========================================================================

"""
    zero_vorticity!(sol::CNAB)

Reset all fluid-related fields in a CNAB simulation object.

This function sets the vorticity (`ω`), streamfunction (`ψ`), and velocity (`u`) 
fields to zero across all grid levels. It also resets the nonlinear history counter 
and re-applies the initial prescribed flow field (`u0`).

# Arguments
- `sol::CNAB` : The CNAB simulation object whose fluid fields are being reset.

# Returns
The updated `CNAB` object with zeroed fluid fields and initial flow re-applied.
"""
function zero_vorticity!(sol::CNAB)
    grid = sol.prob.grid

    for level in 1:grid.levels
        for i in eachindex(sol.state.ω[level])
            sol.state.ω[level][i] .= 0
            sol.state.ψ[level][i] .= 0
        end
        for i in eachindex(sol.state.u[level])
            sol.state.u[level][i] .= 0
        end
    end

    sol.state.nonlin_count = 0

    for level in eachindex(sol.state.u)
        add_flow!(sol.state.u[level], sol.prob.u0, grid, level, sol.i, sol.t)
    end

    sol
end

"""
    background_velocity(flow, t)

Return the background (irrotational) flow as a function `x -> u` of position at
time `t`. This is what the primitive formulation prescribes on ∂D — for external
flow the far-field boundary simply carries the free stream.
"""
background_velocity(flow::UniformFlow, t) = let u0 = flow.u(t)
    x -> u0
end

"""
    zero_velocity!(sol::CNAB)

Reset the primitive-variable velocity state: clear the velocity unknowns, the
advection scratch and the Adams-Bashforth history, then seed the field with the
background flow.

Shared by **both** primitive-variable formulations (`IBPM` and `IMAP`): they use
the same velocity layouts and field names, and neither's velocity reset depends
on how no-slip is enforced. Only the multiplier reset differs — see
[`zero_pressure!`](@ref).

Like [`zero_vorticity!`](@ref), this does *not* leave the velocity at zero — it
leaves it at the background (free-stream) flow, which is irrotational and
therefore a consistent initial condition. It also writes that same background
flow into the boundary buffer `ub`, since it is the velocity prescribed on ∂D.

The interior unknowns `q` are seeded from the physical field `u_full`; the halo of
`q` is left at zero, as the projection operators require.
"""
function zero_velocity!(sol::CNAB)
    grid = sol.prob.grid
    st = sol.state
    backend = get_backend(st.φ)

    for fld in (st.q, st.q_star, st.r1, st.u_full, st.ω)
        for i in eachindex(fld)
            fill!(fld[i], 0)
        end
    end
    for k in eachindex(st.nonlin), i in eachindex(st.nonlin[k])
        fill!(st.nonlin[k][i], 0)
    end
    st.nonlin_count = 0

    # The background flow is both the initial condition and the velocity
    # prescribed on the domain boundary.
    add_flow!(st.u_full, sol.prob.u0, grid, 1, sol.i, sol.t)
    set_velocity_boundary!(st.ub, grid, background_velocity(sol.prob.u0, sol.t))

    # Seed the interior unknowns from the physical field (halo stays zero).
    for i in eachindex(st.q)
        R = CartesianIndices(cell_axes(grid, Loc_u(i), ExcludeBoundary()))
        let q = st.q[i], u = st.u_full[i]
            @loop backend (I in R) q[I] = u[I]
        end
    end

    sol
end

"""
    zero_pressure!(sol::CNAB)

Zero the primitive-variable (`IBPM`) pressure and the modified-Poisson
right-hand-side blocks. The pressure is a Lagrange multiplier with no history, so
zero is the natural start (and it is only ever determined up to a constant, which
the solver pins).
"""
function zero_pressure!(sol::CNAB)
    st = sol.state
    fill!(st.φ, 0)
    fill!(st.rhs_φ, 0)
    fill!(st.rhs_f, zero(eltype(st.rhs_f)))
    sol
end

"""
    initialize_fields!(sol::CNAB)
    initialize_fields!(sol::CNAB, formulation)

Reset the solver's unknowns to their initial state. Master entry point:
dispatches on `sol.prob.formulation`, because the two formulations have entirely
different unknowns.

  - `FastIBPM` → [`zero_vorticity!`](@ref) (vorticity/streamfunction).
  - `IBPM`     → [`zero_velocity!`](@ref) and [`zero_pressure!`](@ref)
    (velocity/pressure/boundary force).
  - `IMAP`     → the same, minus the boundary-force block (no such unknown).
"""
initialize_fields!(sol::CNAB) = initialize_fields!(sol, sol.prob.formulation)

initialize_fields!(sol::CNAB, ::FastIBPM) = zero_vorticity!(sol)

function initialize_fields!(sol::CNAB, ::IBPM)
    zero_velocity!(sol)
    zero_pressure!(sol)
    sol
end

"""
    initialize_fields!(sol::CNAB, ::IMAP)

`IMAP` reset: the primitive-variable velocity/pressure reset, **followed by a
projection of the initial velocity onto the constraint manifold**.

That last step is not optional. IMAP *preserves* the constraint rather than
imposing it — with the pressure gradient projected, applying `Rᵀ` to the momentum
equation gives exactly `Rᵀu^{n+1} = Rᵀuⁿ`, so whatever violation the initial
condition carries is carried forever. The background flow that
[`zero_velocity!`](@ref) seeds is a free stream, which violates no-slip by the
full free-stream magnitude (`‖Rᵀu⁰‖ = U∞`), so it must be projected before the
first step. This is the standard consistent-initial-condition requirement for a
constrained system, and it is what `IBPM` never needs: there the boundary force
is re-solved every step, so it repairs any violation instead of propagating it.

Projecting makes `u⁰` no longer divergence-free; the first pressure solve removes
that, exactly as it would remove the divergence of any other predictor.
"""
function initialize_fields!(sol::CNAB, ::IMAP)
    zero_velocity!(sol)
    zero_pressure!(sol, IMAP())

    # `sol.coupler.Binv` carries the projector; `initial_sol` builds it before
    # calling this, so it is available here.
    P_mul!(sol.state.q, sol.coupler.Binv.proj)
    recover_velocity!(sol, IMAP())      # re-sync `u_full` with the projected `q`

    sol
end

"""
    zero_pressure!(sol::CNAB, ::IMAP)

Manifold-projection (`IMAP`) method of the pressure reset: identical to
[`zero_pressure!(sol)`](@ref) but without the `rhs_f` block, which does not exist
here — IMAP's Lagrange multiplier is `λ = φ` alone, with no boundary force.
"""
function zero_pressure!(sol::CNAB, ::IMAP)
    st = sol.state
    fill!(st.φ, 0)
    fill!(st.rhs_φ, 0)
    sol
end
