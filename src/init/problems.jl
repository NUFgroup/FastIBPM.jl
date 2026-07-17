"""
    IrrotationalFlow

Abstract type specifying a flow where the curl of the velocity is zero
(∇ × u = 0). This is a "tag" type used for dispatch.
"""
abstract type IrrotationalFlow end

"""
    UniformFlow(u)

A type of `IrrotationalFlow` representing a flow that is uniform in space
but may vary in time.

# Fields
- `u`: The freestream velocity. Can be a constant `SVector` or a
  function `u(t)` that returns an `SVector`.
"""
struct UniformFlow{U} <: IrrotationalFlow
    u::U
end

"""
    add_flow!(u, flow::UniformFlow, _, _, _, t)

Adds the background `UniformFlow` velocity to the velocity field `u` in-place.
Computes `u_new = u_old + u_flow(t)`.

# Arguments
- `u`: The velocity field (e.g., a tuple of arrays). Modified in-place.
- `flow::UniformFlow`: The background flow object.
- `_`: Placeholders for unused arguments.
- `t`: The current simulation time.

# Returns
- The modified velocity field `u`.
"""
function add_flow!(u, flow::UniformFlow, _, _, _, t)
    backend = get_backend(u[1])
    u0 = flow.u(t)
    for i in eachindex(u)
        let u = u[i], u0 = u0[i]
            @loop backend (I in CartesianIndices(u)) u[I] += u0
        end
    end
    u
end

# BodyPoints and AbstractBody have moved to body_domain/lagrangian_grid.jl

"""
    abstract type AbstractFormulation end

Abstract type for the numerical formulation used to solve the immersed
boundary problem. Subtypes act as dispatch tags so that formulation-specific
methods (time stepping, Poisson solves, field allocations) can be selected
at compile time without runtime branches.

Current subtypes:
- `FastIBPM` — streamfunction-vorticity (nullspace) formulation. Colonius & Taira (2008)
- `IBPM` — primitive variables formulation. Taira & Colonius (2007)
- `IMAP` — primitive variables using the Interface Manifold Aware Projection Method. (in development)
"""
abstract type AbstractFormulation end

"""
    struct FastIBPM <: AbstractFormulation end

Formulation tag for the streamfunction-vorticity (ψ-ω) IBPM.

Incompressibility is satisfied everywhere by construction: the velocity is
always computed as `u = ∇×ψ`, so `∇·u = 0` is exact to machine precision.
This is the "nullspace" or "fast" approach from Colonius & Taira (2008).

Primary state variables: vorticity `ω` and streamfunction `ψ`.
"""
struct FastIBPM <: AbstractFormulation end

"""
    struct IBPM <: AbstractFormulation end

Formulation tag for the primitive variables IBPM.

TODO: Add description of the IBPM formulation, including its primary state variables and how it differs from FastIBPM.
"""
struct IBPM <: AbstractFormulation end

"""
    struct IMAP <: AbstractFormulation end

Formulation tag for the primitive variables IMAP.

IMAP uses principles from differential geometry to cast the no-slip condition
as a constraint manifold. The flow is restricted to evolve along this manifold
by designing computationally cheap surface-local projection operations.
"""
struct IMAP <: AbstractFormulation end

# TODO: add flags for the CNAB using IBPM and IMAP in src/time_stepping/cnab.jl, and add the corresponding CNAB methods for these formulations.

"""
    struct IBProblem{N,T,B<:AbstractBody,U<:IrrotationalFlow,F<:AbstractFormulation}
        grid::AbstractGrid{N,T}
        body::B
        Re::T
        u0::U
        formulation::F
    end

Defines the entire immersed boundary problem to be solved. An `IBProblem`
instance contains all necessary components: grid, body, Reynolds number,
background flow, and the choice of numerical formulation.

# Parameters
- `N`: Dimension of the problem (2D or 3D).
- `T`: Scalar type (e.g., `Float64`).
- `B<:AbstractBody`: The concrete body type.
- `U<:IrrotationalFlow`: The concrete background flow type.
- `F<:AbstractFormulation`: The numerical formulation (default: `FastIBPM`).

# Fields
- `grid::AbstractGrid{N,T}`: The fluid grid (`Grid` for uniform, `StretchedGrid` for
  the stretched IBPM path).
- `body::B`: The immersed body (must be a subtype of `AbstractBody`).
- `Re::T`: The Reynolds number.
- `u0::U`: The background flow (must be a subtype of `IrrotationalFlow`).
- `formulation::F`: Selects the solver formulation (e.g., `FastIBPM()`).
"""
struct IBProblem{N,T,B<:AbstractBody,U<:IrrotationalFlow,F<:AbstractFormulation}
    grid::AbstractGrid{N,T}
    body::B
    Re::T
    u0::U
    formulation::F
end

# This is to use the default formulation if the user doesn't specify one.
IBProblem(grid, body, Re, u0) = IBProblem(grid, body, Re, u0, FastIBPM())

# These are stubs. The docstrings are removed to avoid "duplicate docs" errors.
# The real docstrings should be in the file where these are implemented.
function surface_force! end
function surface_force_sum end


# ---------------------------------------------------------------------------
# Coupler types, CNAB struct, constructor, and initialization routines
# (moved from cnab.jl)
# ---------------------------------------------------------------------------

"""
    AbstractCoupler

Abstract interface for coupling strategies in the CNAB time integration framework.

This type defines a common interface for different coupler implementations that manage interactions between PDE components or immersed bodies during time integration. Specific couplers include:

- `NothingCoupler`          : No coupling required.
- `PrescribedBodyCoupler`   : For bodies with prescribed motion or known behavior.
- `FsiCoupler`              : For fluid-structure interaction problems.

# Arguments
None.

# Returns
An abstract type that serves as a base for all coupling strategies used in CNAB-based simulations.
"""
abstract type AbstractCoupler end

"""
    NothingCoupler

A simple subtype of `AbstractCoupler` representing the absence of coupling.

Use this type when no body-fluid interaction or other coupling is required in the CNAB time integration framework. It serves as a placeholder that satisfies the coupler interface without modifying the solution.

# Arguments
None.

# Returns
An instance of `NothingCoupler`, indicating that no coupling is applied during the simulation.
"""
struct NothingCoupler <: AbstractCoupler end

"""
    PrescribedBodyCoupler{M}

A subtype of `AbstractCoupler` for prescribed body motion or constraints.

This coupler stores a field `Binv`, representing a precomputed operator used to enforce prescribed motion (and forces) on the body. The type `M` is parametric, allowing flexibility in the data structure used for `Binv` (e.g., arrays or linear operators).

# Arguments
- `Binv::M` : Precomputed operator or matrix used to apply constraints for the prescribed body.

# Returns
An instance of `PrescribedBodyCoupler` suitable for simulations where body motion or behavior is explicitly prescribed.
"""
struct PrescribedBodyCoupler{M} <: AbstractCoupler
    Binv::M
end

"""
    FsiCoupler{T,O<:GeometricNonlinearBodyOperators,B}

A subtype of `AbstractCoupler` for fluid–structure interaction (FSI) problems 
with nonlinear structural dynamics.

This type encapsulates the structural state, geometric nonlinear operators, solver 
tolerance, BiCGStab solver arguments, and iteration limits required for solving 
coupled FSI problems using the CNAB time integration framework. 

The keyword constructor allows easy setup of the coupler, including automatic 
initialization of the structural state and operators for a given `GeometricNonlinearBody`.

# Fields
- `state::StructuralState{T}`          : Current state of the structural body (displacements, velocities, deformations), parameterized by numeric type `T`.
- `ops::O`                             : Operator object representing the geometric nonlinear structural model.
- `tol::T`                             : Numerical tolerance for the iterative solve.
- `bicgstabl_args::B`                  : Arguments for the BiCGStab(ℓ) iterative solver (tuple or named container).
- `maxiter::Int`                       : Maximum iterations during the solver phase.

# Arguments (via keyword constructor)
- `backend::CPU`                       : Computation backend (CPU-based).
- `body::GeometricNonlinearBody{N,T}`  : Nonlinear structural body with `N` spatial dimensions and numeric type `T`.
- `tol`                                : Solver tolerance (default `1e-5`).
- `bicgstabl_args`                     : Solver arguments (default `(; abstol=T(1e-5), reltol=T(0.0))`).
- `maxiter::Int`                       : Maximum iterations (default `100`).

# Returns
A fully initialized `FsiCoupler` instance ready to couple the fluid and structural solvers in FSI simulations.
"""
@kwdef struct FsiCoupler{T,O<:GeometricNonlinearBodyOperators,B} <: AbstractCoupler
    state::StructuralState{T}
    ops::O
    tol::T
    bicgstabl_args::B
    maxiter::Int
end

function FsiCoupler(
    backend::CPU,
    body::GeometricNonlinearBody{N,T};
    tol=1e-5,
    bicgstabl_args=(; abstol=T(1e-5), reltol=T(0.0)),
    maxiter=100,
) where {N,T}
    n = deforming_point_count(body)
    nel = n - 1
    nf = N * point_count(body)

    state = StructuralState{T}(backend, structure_var_count(body))
    ops = structural_operators(backend, body)

    FsiCoupler(; state, ops, tol, bicgstabl_args, maxiter)
end


"""
    CNAB{N,T,B,U,F,R,C,Au,Vb,BP,A,S}

State and configuration of the Crank-Nicolson / Adams-Bashforth time integrator.

`CNAB` holds only what is **common to every formulation** — the problem, clock
(`i`, `t`, `dt`), the Adams-Bashforth weights `β`, the body coupling (`reg`,
`coupler`, `f`, `f_tilde`, `points`, `redist_weights`) and the memory pools.

Everything formulation-specific — the unknowns themselves — lives in `state`,
which is dispatched on `prob.formulation`:

  - [`FastIBPMState`](@ref) for `FastIBPM`: vorticity `ω`, streamfunction `ψ`, the
    multi-level velocity `u`, the FFT `plan`, and a vorticity-space nonlinear
    history. No pressure — continuity is automatic (`u = ∇×ψ`).
  - [`IBPMState`](@ref) for `IBPM`: velocity unknowns `q`, pressure `φ`, the
    prescribed-velocity boundary buffer, and a velocity-space nonlinear history.

Keeping them apart means neither formulation carries the other's (large) fields,
and the time-stepping methods in `cnab.jl` dispatch on `prob.formulation` to
select the matching pipeline.

# Fields
- `prob`, `t0`, `i`, `t`, `dt`, `β` : problem and time-integration state.
- `reg`, `coupler`, `f`, `f_tilde`, `points`, `redist_weights` : body coupling.
- `body_pool`, `fluid_pool`, `bndry_pool`, `structure_pool` : scratch pools.
- `state` : formulation-specific unknowns (see above).
"""
@kwdef mutable struct CNAB{
    N,T,B,U,F<:AbstractFormulation,R<:Reg,C<:AbstractCoupler,Au,Vb,BP<:BodyPoints,A<:ArrayPool,S
}
    const prob::IBProblem{N,T,B,U,F}
    const t0::T
    i::Int
    t::T
    const dt::T
    const β::Vector{T}
    const reg::R
    const coupler::C
    const redist_weights::Au
    const f_tilde::Vb
    const f::Vb
    const points::BP
    body_pool::A
    fluid_pool::A
    bndry_pool::A
    structure_pool::A
    # Formulation-specific unknowns: `FastIBPMState` or `IBPMState`.
    const state::S
end


function CNAB(
    prob::IBProblem{N,T};
    dt,
    t0=zero(T),
    n_step=2,
    delta=DeltaYang3S(),
    backend=CPU(),
    coupler_args=(;),
) where {N,T}
    grid = prob.grid
    body = prob.body

    n_ib = point_count(body)
    n_structure = structure_var_count(body)

    max_fluid_vars = maximum(
        loc -> grid_length(grid, loc, IncludeBoundary()), (Loc_u, Loc_ω)
    )
    max_bndry_vars = boundary_length(grid, Loc_ω)

    args = (;
        prob,
        t0,
        i=0,
        t=zero(T),
        dt,
        β=ab_coeffs(T, n_step),
        reg=Reg(backend, T, delta, n_ib, Val(N)),
        redist_weights=grid_zeros(backend, grid, Loc_u),
        f_tilde=KernelAbstractions.zeros(backend, SVector{N,T}, n_ib),
        f=KernelAbstractions.zeros(backend, SVector{N,T}, n_ib),
        points=BodyPoints{N,T}(backend, n_ib),
        body_pool=ArrayPool(backend, n_ib * sizeof(SVector{N,T})),
        fluid_pool=ArrayPool(backend, max_fluid_vars * sizeof(T)),
        bndry_pool=ArrayPool(backend, max_bndry_vars * sizeof(T)),
        structure_pool=ArrayPool(backend, n_structure * sizeof(T)),
        state=formulation_state(backend, grid, prob.formulation, n_ib, n_step),
    )

    sol = initial_sol(backend, body, args, coupler_args)

    sol
end

"""
    coupling_Binv(sol0, body, formulation)

Select the `B⁻¹` strategy for the body coupling, dispatching on **both** the body
type and the formulation.

  - **`FastIBPM` + static body** → [`B_inverse_rigid`](@ref). Here `B` is constant
    (the body never moves, so `E`/`Eᵀ` never change) *and small*: it acts on the
    body force alone, so it is only `N·n_b` square. Assemble it once and
    Cholesky-factor it.
  - **`FastIBPM` + moving body** → `CNAB_Binv_Iterative`. `B` changes every step, so
    there is nothing to precompute.
  - **`IBPM` + *any* body (static included)** → `CNAB_Binv_Iterative` (CG).

The last case is the important constraint: the primitive `B = QᵀBᴺQ` acts on
`λ = (φ, f_tilde)`, so it is `(#cells + N·n_b)` square — not `N·n_b`. Assembling
and factoring that densely is hopeless (a 300×300 grid alone would need ~65 GB,
and 3D is far worse), so **even a static body must be solved iteratively**. That
is exactly what Taira & Colonius do, and it is why a static `IBPM` problem cannot
reuse the `FastIBPM` precompute shortcut.
"""
coupling_Binv(sol0::CNAB, ::AbstractStaticBody, ::FastIBPM) = B_inverse_rigid(sol0)

coupling_Binv(sol0::CNAB{N,T}, ::AbstractBody, ::FastIBPM) where {N,T} =
    CNAB_Binv_Iterative{T}()

coupling_Binv(sol0::CNAB{N,T}, ::AbstractBody, ::IBPM) where {N,T} =
    CNAB_Binv_Iterative{T}()
"""
    initial_sol(backend, body, sol_args, coupler_args)

Initialize a CNAB simulation object based on the type of body in the problem.

This function has two methods depending on whether `body` is a static or geometrically 
nonlinear (deforming) body:

1. **Static Body Initialization (`AbstractStaticBody`)**  
   - Constructs a temporary CNAB with a `NothingCoupler`.  
   - Initializes body point positions.  
   - Computes regularization weights.  
   - Computes the inverse of the body–fluid coupling matrix.  
   - Creates a `PrescribedBodyCoupler` with the precomputed operator.  
   - Builds the final CNAB object with the coupler.  
   - Sets simulation time and initializes fluid fields.

2. **Geometric Nonlinear Body Initialization (`GeometricNonlinearBody`)**  
   - Constructs an `FsiCoupler` for nonlinear structural dynamics.  
   - Builds the CNAB object with this coupler.  
   - Sets simulation time and zeros the fluid fields.  
   - Splits prescribed and deforming points.  
   - Initializes prescribed body points.  
   - Updates the structural state and initializes structure operators.  
   - Computes regularization and redistribution weights.

# Arguments
- `backend`                 : Computation backend (`CPU()` or GPU device).  
- `body`                    : The body in the problem (`AbstractStaticBody` or `GeometricNonlinearBody`).  
- `sol_args`                : Named tuple with CNAB fields and buffers.  
- `coupler_args`            : Keyword arguments passed to the coupler constructor.

# Returns
A fully initialized `CNAB` object ready for time integration, configured according 
to the type of body and the specified coupling strategy.
"""

function initial_sol(backend, body::AbstractStaticBody, sol_args, coupler_args)
    sol0 = CNAB(; sol_args..., coupler=NothingCoupler())

    init_body_points!(sol0.points, body)
    update_weights!(sol0.reg, sol0.prob.grid, sol0.points.x, eachindex(sol0.points.x))
    Binv = coupling_Binv(sol0, body, sol0.prob.formulation)

    coupler = PrescribedBodyCoupler(Binv; coupler_args...)
    sol = CNAB(; sol_args..., coupler)
    set_time!(sol, 0)
    initialize_fields!(sol)
    update_redist_weights!(sol)

    sol
end

# Arturo: Add initial sol for moving bodies

function initial_sol(backend, body::AbstractPrescribedBody, sol_args, coupler_args)
    # Build a temporary CNAB to get geometry-dependent weights
    sol0 = CNAB(; sol_args..., coupler=NothingCoupler())

    # Initialize body points/velocities at t=0
    init_body_points!(sol0.points, body)

    # Build regularization/redistribution for current geometry
    update_weights!(sol0.reg, sol0.prob.grid, sol0.points.x, eachindex(sol0.points.x))
    update_redist_weights!(sol0)

    # A moving body changes `B` every step, so there is nothing to precompute.
    Binv = coupling_Binv(sol0, body, sol0.prob.formulation)

    coupler = PrescribedBodyCoupler(Binv; coupler_args...)
    sol = CNAB(; sol_args..., coupler)
    set_time!(sol, 0)
    initialize_fields!(sol)
    update_redist_weights!(sol)

    return sol
end

function initial_sol(
    backend, body::GeometricNonlinearBody{N,T}, sol_args, coupler_args
) where {N,T}
    coupler = FsiCoupler(backend, body; coupler_args...)
    sol = CNAB(; sol_args..., coupler)
    set_time!(sol, 0)
    initialize_fields!(sol)

    i_deform = deforming_point_range(body)
    i_prescribed = prescribed_point_range(body)

    init_body_points!(view(sol.points, i_prescribed), body.prescribed)
    update_structure!(sol.points, coupler.state, body, coupler.ops, sol.i, sol.t)
    init_structure_operators!(coupler.ops, body, sol.points, coupler.state, sol.dt)
    update_weights!(sol.reg, sol.prob.grid, sol.points.x, eachindex(sol.points.x))
    update_redist_weights!(sol)

    sol
end


# ---------------------------------------------------------------------------
# Body-force inverse operators (chosen at init time, called during coupling)
# ---------------------------------------------------------------------------
"""
CNAB_Binv_Precomputed(B)

A precomputed coupling operator for the CNAB solver.

Holds a precomputed matrix `B` used to solve the body–fluid coupling system 
directly. Efficient when `B` is constant (e.g., for rigid or prescribed-motion bodies).

The object is callable like a function to compute the body force `f` given the 
desired body velocity `u_ib` and the CNAB solver `sol`.

Arguments:
- `B`: Precomputed coupling matrix.
- `f`: Body force vector (updated in-place).
- `u_ib`: Coupling right-hand side (desired body velocity minus interpolated fluid velocity).
- `sol::CNAB`: CNAB solver object (included for signature consistency, not used here).

Returns:
- Updates `f` in-place.
"""
struct CNAB_Binv_Precomputed{M}
    B::M
end

function (x::CNAB_Binv_Precomputed)(f, u_ib, ::CNAB{N,T}) where {N,T}
    let f = reinterpret(T, f), u_ib = reinterpret(T, u_ib)
        ldiv!(f, x.B, u_ib)
    end
end

"""
    CNAB_Binv_Iterative{T}

Iterative `B⁻¹` for the CNAB solver: rather than precomputing an inverse, solve
`B λ = rhs` matrix-free at each coupling step, warm-started from the current
contents of `λ`. Used when `B` changes every step (e.g. moving prescribed bodies),
and it is the only option for `IBPM`.

This is **dispatched on the formulation**: the two schemes have a different `B`, a
different Lagrange multiplier, and — deliberately — a different Krylov method.

  - **`FastIBPM`** (streamfunction-vorticity). `B` is applied by `B_rigid_mul!`,
    and the multiplier is the boundary force *alone* (continuity is automatic,
    since `u = ∇×ψ`). That `B` routes through the *approximate*
    `multidomain_poisson!`, so it is only **near**-symmetric → **BiCGStab(ℓ)**.
  - **`IBPM`** (primitive variables). `B = Qᵀ Bᴺ Q` is applied by `B_mul!`, and the
    multiplier `λ = (φ, f_tilde)` carries the pressure *and* the force. That `B` is
    *exactly* symmetric positive definite once pinned → **CG** (measured at 2-4x
    fewer matvecs and ~10x lower solution error than BiCGStab here).

# Fields
- `abstol::T`    : absolute solver tolerance (default `1e-5`).
- `reltol::T`    : relative solver tolerance (default `0.0`).
- `pin::Int`     : **`IBPM` only** — pressure DOF pinned to zero (default `1`).
- `maxiter::Int` : **`IBPM` only** — CG iteration cap (default `5000`).

# Call signatures
    (op::CNAB_Binv_Iterative)(f, rhs, sol::CNAB)                    # FastIBPM
    (op::CNAB_Binv_Iterative)(φ, f_tilde, rhs_φ, rhs_f, reg, work,
                              ::IBPM; h, a, dt, n_taylor)           # IBPM
"""
Base.@kwdef struct CNAB_Binv_Iterative{T}
    abstol::T = T(1e-5)
    reltol::T = T(0.0)
    # `IBPM` only: linear index of the pressure cell pinned to zero. `B = QᵀBᴺQ`
    # is singular along the constant-pressure mode, so one DOF must be fixed.
    pin::Int = 1
    # `IBPM` only: iteration cap for the CG solve.
    maxiter::Int = 5000
end

"""
    (op::CNAB_Binv_Iterative)(f, rhs, sol::CNAB)

Streamfunction-vorticity (`FastIBPM`) method of the iterative `B⁻¹`: solve
`B f = rhs` in place for the boundary force `f`, where `B` is applied matrix-free
from the current geometry via [`B_rigid_mul!`](@ref).

Solved with **BiCGStab(ℓ)**: this `B` goes through the *approximate*
`multidomain_poisson!`, which leaves it only near-symmetric, so CG's symmetry
assumption does not hold. (The `IBPM` method below uses CG instead — see the
formulation comparison in the type docstring.)
"""
function (op::CNAB_Binv_Iterative{T})(f, rhs, sol::CNAB{N,T}) where {N,T}
    n_ib = point_count(sol.prob.body)
    n = N * n_ib

    # y := B*x with current geometry (uses your existing B_rigid_mul!)
    Bmap = LinearMap(n; ismutating=true) do y, x
        B_rigid_mul!(reinterpret(SVector{N,T}, y), reinterpret(SVector{N,T}, x), sol)
    end

    # Solve B f = rhs, warm-started by current contents of f
    bicgstabl!(
        reinterpret(T, f), Bmap, reinterpret(T, rhs); abstol=op.abstol, reltol=op.reltol
    )
    nothing
end


"""
    (op::CNAB_Binv_Iterative)(φ, f_tilde, rhs_φ, rhs_f, reg, work, ::IBPM; h, a, dt, n_taylor)

Primitive-variable (`IBPM`) method of the iterative `B⁻¹`: solve the *modified
Poisson* system of Taira & Colonius (2007), Eq. 26,

    B λ = (φ_rhs, f_rhs),    B = Qᵀ Bᴺ Q,    λ = (φ, f_tilde),

in place. This mirrors the streamfunction-vorticity method above — a matrix-free
`LinearMap` around the `B` matvec, warm-started from the current contents of `λ` —
but with two differences: `B` is applied by [`B_mul!`](@ref), the Lagrange
multiplier carries the pressure *and* the boundary force (in `FastIBPM` continuity
is automatic, so its `B` sees only the force), and the solve uses **CG** rather
than BiCGStab(ℓ) because this `B` is exactly symmetric positive definite once
pinned (see the comment at the `cg!` call).

`B` is singular along the constant-pressure mode (`Q(1, 0) = 0`, since the
gradient annihilates a constant), so the pressure DOF `op.pin` is pinned to zero:
the operator actually solved is `P B P + eₚ eₚᵀ`, which is nonsingular.

Note the `Bᴺ` Taylor series only converges when `a/h² ≲ 1` (the paper's
`νΔt/Δx² ≲ 1`); beyond that `B` becomes severely ill-conditioned.
"""
function (op::CNAB_Binv_Iterative{T})(
    φ, f_tilde, rhs_φ, rhs_f, reg, work, form::IBPM; h, a, dt, n_taylor
) where {T}
    N = length(eltype(f_tilde))
    np = length(no_offset_view(φ))
    nf = N * length(f_tilde)
    n = np + nf
    pin = op.pin

    φi, φo = similar(φ), similar(φ)
    fi, fo = similar(f_tilde), similar(f_tilde)

    # y := (P B P + eₚ eₚᵀ) x — pinned so the system is nonsingular.
    Bmap = LinearMap(n; ismutating=true) do y, x
        vec(no_offset_view(φi)) .= @view x[1:np]
        no_offset_view(φi)[pin] = 0                  # project the pinned DOF out
        reinterpret(T, fi) .= @view x[(np+1):n]

        B_mul!(φo, fo, φi, fi, reg, work, form; h, a, dt, n_taylor)

        let yφ = @view y[1:np]
            yφ .= vec(no_offset_view(φo))
            yφ[pin] = x[pin]                         # identity row/col on the pinned DOF
        end
        @view(y[(np+1):n]) .= reinterpret(T, fo)
        y
    end

    b = zeros(T, n)
    b[1:np] .= vec(no_offset_view(rhs_φ))
    b[pin] = 0                                       # enforces φ[pin] = 0
    b[(np+1):n] .= reinterpret(T, rhs_f)

    # Warm start from the current contents of λ, as the FastIBPM method does.
    x = zeros(T, n)
    x[1:np] .= vec(no_offset_view(φ))
    x[pin] = 0
    x[(np+1):n] .= reinterpret(T, f_tilde)

    # CG, not BiCGStab(ℓ) as in the FastIBPM method above. That is deliberate:
    # this `B = QᵀBᴺQ` is *exactly* symmetric positive definite once pinned (it is
    # a congruence of the symmetric `Bᴺ`, a polynomial in the symmetric `L`), so CG
    # is optimal — measured at 2-4x fewer matvecs and ~10x lower solution error
    # than BiCGStab at matched tolerances. The FastIBPM `B` instead routes through
    # the *approximate* `multidomain_poisson!`, leaving it only near-symmetric,
    # which is why BiCGStab is the right choice there.
    cg!(x, Bmap, b; abstol=op.abstol, reltol=op.reltol, maxiter=op.maxiter)

    vec(no_offset_view(φ)) .= @view x[1:np]
    reinterpret(T, f_tilde) .= @view x[(np+1):n]
    (φ, f_tilde)
end

"""
    (op::CNAB_Binv_Iterative)(φ, f_tilde, rhs_φ, rhs_f, reg, work, ::IBPM, grid::StretchedGrid; a, dt, n_taylor)

`StretchedGrid` method of the `IBPM` iterative `B⁻¹`. Identical to the uniform
method above (pinned CG on `B = QᵀBᴺQ`), except it calls the mass-weighted
stretched [`B_mul!`](@ref) (which is still exactly SPD once pinned — see the
symmetry gate), so no scalar `h` is needed.
"""
function (op::CNAB_Binv_Iterative{T})(
    φ, f_tilde, rhs_φ, rhs_f, reg, work, form::IBPM, grid::StretchedGrid; a, dt, n_taylor
) where {T}
    N = length(eltype(f_tilde))
    np = length(no_offset_view(φ))
    nf = N * length(f_tilde)
    n = np + nf
    pin = op.pin

    φi, φo = similar(φ), similar(φ)
    fi, fo = similar(f_tilde), similar(f_tilde)

    Bmap = LinearMap(n; ismutating=true) do y, x
        vec(no_offset_view(φi)) .= @view x[1:np]
        no_offset_view(φi)[pin] = 0
        reinterpret(T, fi) .= @view x[(np+1):n]

        B_mul!(φo, fo, φi, fi, reg, work, form, grid; a, dt, n_taylor)

        let yφ = @view y[1:np]
            yφ .= vec(no_offset_view(φo))
            yφ[pin] = x[pin]
        end
        @view(y[(np+1):n]) .= reinterpret(T, fo)
        y
    end

    b = zeros(T, n)
    b[1:np] .= vec(no_offset_view(rhs_φ))
    b[pin] = 0
    b[(np+1):n] .= reinterpret(T, rhs_f)

    x = zeros(T, n)
    x[1:np] .= vec(no_offset_view(φ))
    x[pin] = 0
    x[(np+1):n] .= reinterpret(T, f_tilde)

    cg!(x, Bmap, b; abstol=op.abstol, reltol=op.reltol, maxiter=op.maxiter)

    vec(no_offset_view(φ)) .= @view x[1:np]
    reinterpret(T, f_tilde) .= @view x[(np+1):n]
    (φ, f_tilde)
end
