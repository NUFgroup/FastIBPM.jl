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
        grid::Grid{N,T}
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
- `grid::Grid{N,T}`: The fluid grid.
- `body::B`: The immersed body (must be a subtype of `AbstractBody`).
- `Re::T`: The Reynolds number.
- `u0::U`: The background flow (must be a subtype of `IrrotationalFlow`).
- `formulation::F`: Selects the solver formulation (e.g., `FastIBPM()`).
"""
struct IBProblem{N,T,B<:AbstractBody,U<:IrrotationalFlow,F<:AbstractFormulation}
    grid::Grid{N,T}
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
    CNAB{N,T,B,U,P,R<:Reg,C<:AbstractCoupler,Au,Aω,Vb,BP<:BodyPoints,A<:ArrayPool,W}

Central mutable type representing the state and configuration of a Crank–Nicolson 
Adams–Bashforth (CNAB) time integration scheme for coupled fluid–structure simulations.

This struct holds all data required for time-stepping the simulation, including 
fluid and body fields, transform plans, regularization operators, memory pools, 
and solver buffers. It is designed for high-performance computing with support 
for GPU/CPU backends and flexible handling of complex bodies and couplers.

# Fields
- `prob::IBProblem{N,T,B,U}`             : The immersed boundary problem defining the grid and bodies.
- `t0::T`                                : Initial simulation time.
- `i::Int`                               : Current time step index.
- `t::T`                                 : Current simulation time.
- `dt::T`                                : Time step size.
- `β::Vector{T}`                         : CNAB scheme coefficients.
- `plan::P`                              : FFT or spectral transform plan.
- `reg::R`                               : Regularizer or interpolation operator.
- `coupler::C`                           : Coupling strategy (`FsiCoupler`, `PrescribedBodyCoupler`, `NothingCoupler`).
- `redist_weights::Au`                   : Redistribution weights for fluid variables.
- `ω::Vector{Aω}`                        : Vorticity field(s).
- `ψ::Vector{Aω}`                        : Streamfunction or auxiliary field(s).
- `u::Vector{Au}`                        : Velocity field(s).
- `f_tilde::Vb`, `f::Vb`                 : Body force arrays.
- `points::BP`                           : Body point data structure.
- `nonlin::Vector{Vector{Aω}}`           : Buffers for nonlinear term history.
- `nonlin_count::Int`                    : Counter for nonlinear buffers.
- `ω_bndry::W`                           : Boundary vorticity data.
- `body_pool::A, fluid_pool::A, bndry_pool::A, structure_pool::A` : Memory pools to reduce allocations.

# Arguments (via constructor)
- `prob::IBProblem{N,T}`                 : Immersed boundary problem containing grid and body setup.
- `dt`                                   : Time step size.
- `t0`                                   : Initial simulation time (default `0`).
- `n_step`                               : Number of previous time steps to retain for CNAB (default `2`).
- `delta`                                : Regularization kernel (default `DeltaYang3S()`).
- `backend`                              : Computation backend (`CPU()` or GPU device).
- `coupler_args`                         : Keyword arguments for the coupling constructor (e.g., `FsiCoupler`).

# Description
The constructor automatically allocates all buffers, precomputes FFT plans, 
regularization operators, and memory pools, and bundles them into a CNAB object 
ready for time integration. It performs the following main steps:

1. **Setup grid and body**: retrieves `grid` and `body` from `prob`.  
2. **Pre-allocate main fluid field**: creates vorticity arrays.  
3. **Create FFT plan**: precomputes spectral transforms for efficient solves.  
4. **Determine problem sizes**: computes number of body points and structure variables.  
5. **Allocate memory pools**: sizes pools for fluid, body, boundary, and structure arrays.  
6. **Bundle arguments**: stores all fields and buffers in a named tuple.  
7. **Build the solution object**: calls `initial_sol` to wrap arguments into a fully initialized CNAB instance.

# Returns
A `CNAB` object fully initialized for coupled time-stepping with the CNAB scheme.
"""
@kwdef mutable struct CNAB{
    N,T,B,U,F<:AbstractFormulation,P,R<:Reg,C<:AbstractCoupler,Au,Aω,Vb,BP<:BodyPoints,A<:ArrayPool,W
}
    const prob::IBProblem{N,T,B,U,F}
    const t0::T
    i::Int
    t::T
    const dt::T
    const β::Vector{T}
    const plan::P
    const reg::R
    const coupler::C
    const redist_weights::Au
    ω::Vector{Aω}
    ψ::Vector{Aω}
    const u::Vector{Au}
    const f_tilde::Vb
    const f::Vb
    const points::BP
    const nonlin::Vector{Vector{Aω}}
    nonlin_count::Int
    ω_bndry::W
    body_pool::A
    fluid_pool::A
    bndry_pool::A
    structure_pool::A
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
    ω = grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels)

    plan = let ωe = grid_view(ω[1], grid, Loc_ω, ExcludeBoundary())
        laplacian_plans(ωe, grid.n)
    end

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
        plan,
        reg=Reg(backend, T, delta, n_ib, Val(N)),
        redist_weights=grid_zeros(backend, grid, Loc_u),
        ω,
        ψ=grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels),
        u=grid_zeros(backend, grid, Loc_u; levels=1:grid.levels),
        f_tilde=KernelAbstractions.zeros(backend, SVector{N,T}, n_ib),
        f=KernelAbstractions.zeros(backend, SVector{N,T}, n_ib),
        points=BodyPoints{N,T}(backend, n_ib),
        nonlin=map(1:(n_step-1)) do _
            grid_zeros(backend, grid, Loc_ω, ExcludeBoundary(); levels=1:grid.levels)
        end,
        nonlin_count=0,
        ω_bndry=boundary_axes(grid, Loc_ω),
        body_pool=ArrayPool(backend, n_ib * sizeof(SVector{N,T})),
        fluid_pool=ArrayPool(backend, max_fluid_vars * sizeof(T)),
        bndry_pool=ArrayPool(backend, max_bndry_vars * sizeof(T)),
        structure_pool=ArrayPool(backend, n_structure * sizeof(T)),
    )

    sol = initial_sol(backend, body, args, coupler_args)

    sol
end

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
    Binv = B_inverse_rigid(sol0)

    coupler = PrescribedBodyCoupler(Binv; coupler_args...)
    sol = CNAB(; sol_args..., coupler)
    set_time!(sol, 0)
    zero_vorticity!(sol)
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

    # Install iterative inverse (no precompute)
    T = typeof(sol_args.dt)
    Binv = CNAB_Binv_Iterative{T}()

    coupler = PrescribedBodyCoupler(Binv; coupler_args...)
    sol = CNAB(; sol_args..., coupler)
    set_time!(sol, 0)
    zero_vorticity!(sol)
    update_redist_weights!(sol)

    return sol
end

function initial_sol(
    backend, body::GeometricNonlinearBody{N,T}, sol_args, coupler_args
) where {N,T}
    coupler = FsiCoupler(backend, body; coupler_args...)
    sol = CNAB(; sol_args..., coupler)
    set_time!(sol, 0)
    zero_vorticity!(sol)

    i_deform = deforming_point_range(body)
    i_prescribed = prescribed_point_range(body)

    init_body_points!(view(sol.points, i_prescribed), body.prescribed)
    update_structure!(sol.points, coupler.state, body, coupler.ops, sol.i, sol.t)
    init_structure_operators!(coupler.ops, body, sol.points, coupler.state, sol.dt)
    update_weights!(sol.reg, sol.prob.grid, sol.points.x, eachindex(sol.points.x))
    update_redist_weights!(sol)

    sol
end

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
        for i in eachindex(sol.ω[level])
            sol.ω[level][i] .= 0
            sol.ψ[level][i] .= 0
        end
        for i in eachindex(sol.u[level])
            sol.u[level][i] .= 0
        end
    end

    sol.nonlin_count = 0

    for level in eachindex(sol.u)
        add_flow!(sol.u[level], sol.prob.u0, grid, level, sol.i, sol.t)
    end

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

An iterative coupling operator for the CNAB solver, used when the body–fluid
coupling matrix `B` changes every step (e.g. for moving prescribed bodies).

Instead of precomputing `B⁻¹`, solves the linear system `B f = rhs` with
BiCGStab(ℓ) at each coupling step. The current contents of `f` serve as
a warm start.

# Fields
- `abstol::T` : Absolute solver tolerance (default `1e-5`).
- `reltol::T` : Relative solver tolerance (default `0.0`).

# Call signature
    (op::CNAB_Binv_Iterative)(f, rhs, sol::CNAB)

Solves `B f = rhs` in place, where `B` is assembled from the current geometry
via `B_rigid_mul!`.
"""
Base.@kwdef struct CNAB_Binv_Iterative{T}
    abstol::T = T(1e-5)
    reltol::T = T(0.0)
end

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
