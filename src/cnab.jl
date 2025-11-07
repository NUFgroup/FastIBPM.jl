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
    N,T,B,U,P,R<:Reg,C<:AbstractCoupler,Au,Aω,Vb,BP<:BodyPoints,A<:ArrayPool,W
}
    const prob::IBProblem{N,T,B,U}
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
        nonlin=map(1:n_step-1) do _
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

This is the main time integration routine that updates both the fluid and structure 
fields according to the CNAB scheme. A single call to `step!` performs the following sequence:

1. Advance the simulation time step (`set_time!`).  
2. Predict the new fluid and body state (`prediction_step!`).  
3. Apply fluid–structure coupling (`coupling_step!`).  
4. Project the velocity field to enforce incompressibility (`projection_step!`).  
5. Update the vorticity field (`apply_vorticity!`).  

# Arguments
- `sol::CNAB` : The CNAB simulation object representing the current state.

# Returns
The updated `CNAB` object after one complete time step.
"""
function step!(sol::CNAB)
    set_time!(sol, sol.i + 1)

    prediction_step!(sol)
    coupling_step!(sol)
    projection_step!(sol)
    apply_vorticity!(sol)

    sol
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
    _A_factor(sol::CNAB)

Compute the diffusion coefficient used in the CNAB time-stepping scheme.

This coefficient arises in the semi-implicit (Crank–Nicolson) treatment of the
viscous term and is given by:

    A = Δt / (2 * Re)

# Arguments
- `sol::CNAB` : CNAB simulation object containing the current state and parameters.

# Returns
- Diffusion coefficient (same numeric type as `sol.dt`).
"""
_A_factor(sol::CNAB) = sol.dt / (2sol.prob.Re)



"""
    Ainv(sol::CNAB, level)

Construct the inverse viscous operator used in the CNAB time-stepping scheme.

# Arguments
- `sol::CNAB`: CNAB simulation object containing the problem definition and FFT plans.
- `level`: Grid refinement level at which to build the operator.

# Returns
- `EigenbasisTransform`: An operator that applies the inverse of
  `(I - aΔ)` in spectral space, typically used for implicit diffusion
  updates within the CNAB integrator.
"""
function Ainv(sol::CNAB, level)
    h = gridstep(sol.prob.grid, level)
    a = _A_factor(sol)
    EigenbasisTransform(λ -> 1 / (1 - a * λ / h^2), sol.plan)
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
function prediction_step!(sol::CNAB)
    _cycle!(sol.nonlin)

    for level in sol.prob.grid.levels:-1:1
        prediction_step!(sol, level)
    end

    sol.nonlin_count = min(sol.nonlin_count + 1, length(sol.nonlin))
end

function prediction_step!(sol::CNAB{N,T}, level) where {N,T}
    u_axes = cell_axes(sol.prob.grid, Loc_u, ExcludeBoundary())
    with_arrays(sol.fluid_pool, (T, u_axes)) do u_work
        prediction_step!(sol::CNAB, level, u_work)
    end
end

function prediction_step!(sol::CNAB{N,T}, level, u_work) where {N,T}
    backend = get_backend(sol.u[1][1])

    grid = sol.prob.grid
    h = gridstep(grid, level)
    ωˢ = grid_view(sol.ψ[level], grid, Loc_ω, ExcludeBoundary())
    a = _A_factor(sol)

    curl!(u_work, sol.ω[level]; h)
    rot!(ωˢ, u_work; h)

    for i in eachindex(ωˢ)
        let ωˢ = ωˢ[i], ω = sol.ω[level][i]
            @loop backend (I in CartesianIndices(ωˢ)) begin
                ωˢ[I] = ω[I] - a * ωˢ[I]
            end
        end
    end

    if level < grid.levels
        with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψb
            multidomain_interpolate!(ψb, sol.ψ[level+1]; n=grid.n)
            add_laplacian_bc!(ωˢ, a / h^2, ψb)
        end
    end

    nonlin_full = sol.nonlin_count == length(sol.nonlin)

    if nonlin_full
        for i_step in eachindex(sol.nonlin), i in eachindex(ωˢ)
            let ωˢ = ωˢ[i], N = sol.nonlin[i_step][level][i], k = sol.dt * sol.β[end-i_step]
                @loop backend (I in CartesianIndices(ωˢ)) begin
                    ωˢ[I] = ωˢ[I] + k * N[I]
                end
            end
        end
    end

    nonlinear!(u_work, sol.u[level], sol.ω[level])
    rot!(sol.nonlin[end][level], u_work; h)

    for i in eachindex(ωˢ)
        let ωˢ = ωˢ[i],
            N = sol.nonlin[end][level][i],
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
coupling_step!(sol::CNAB) = _coupling_step!(sol, sol.coupler)

function _coupling_step!(sol::CNAB{N,T}, coupler::PrescribedBodyCoupler) where {N,T}
    with_arrays_like(sol.body_pool, sol.f_tilde) do rhs
        with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψ_b_work
            _coupling_step!(sol, coupler, rhs, ψ_b_work)
        end
    end
end

function _coupling_step!(sol::CNAB, coupler::PrescribedBodyCoupler, rhs, ψ_b_work)
    grid = sol.prob.grid
    body = sol.prob.body
    ωˢ = sol.ψ
    ψ = (sol.ω[1],)
    u¹ = sol.u[1]

    multidomain_poisson!(ωˢ, ψ, (u¹,), ψ_b_work, grid, sol.plan)
    add_flow!(u¹, sol.prob.u0, grid, 1, sol.i, sol.t)

    update_body_points!(sol.points, body, sol.i, sol.t)
    update_reg!(sol, body, eachindex(sol.points.x))
    interpolate_body!(rhs, sol.reg, u¹)

    rhs .-= sol.points.u

    coupler.Binv(sol.f_tilde, rhs, sol)
end



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
    B_inverse_rigid(sol::CNAB{N,T,<:AbstractStaticBody})

Construct a precomputed coupling operator for a rigid (static) body.

This function builds the body–fluid coupling matrix `B` and precomputes
its inverse via Cholesky factorization. The returned object is a 
`CNAB_Binv_Precomputed`, which can be applied during the CNAB 
coupling step as `coupler.Binv(sol.f_tilde, rhs, sol)`.

- Only for static/non-deforming bodies.
- Precomputing `B` ensures fast solves at each timestep.
- The matrix is assumed symmetric positive definite (SPD).

# Inputs
- `sol::CNAB`: CNAB simulation object containing the body and grid.

# Returns
- `CNAB_Binv_Precomputed`: Callable object that efficiently applies `B⁻¹`.

"""
function B_inverse_rigid(sol::CNAB{N,T,<:AbstractStaticBody}) where {N,T}
    backend = get_backend(sol.f_tilde)
    n_ib = point_count(sol.prob.body)

    n = N * n_ib
    B_map = LinearMap(n; ismutating=true) do u_ib, f
        B_rigid_mul!(u_ib, f, sol)
    end
    B_mat = KernelAbstractions.zeros(backend, T, n, n)

    with_arrays(sol.body_pool, (T, (n,))) do f
        for i in 1:n
            @. f = ifelse((1:n) == i, 1, 0)
            mul!(@view(B_mat[:, i]), B_map, f)
        end
    end

    CNAB_Binv_Precomputed(cholesky!(Hermitian(B_mat)))
end

function B_rigid_mul!(u_ib::AbstractVector{<:Number}, f, sol::CNAB{N,T}) where {N,T}
    let u_ib = reinterpret(SVector{N,T}, u_ib), f = reinterpret(SVector{N,T}, f)
        B_rigid_mul!(u_ib, f, sol)
    end

    u_ib
end



"""
    B_rigid_mul!(u_ib, f, sol::CNAB{N,T})

Apply the rigid-body coupling operator to a force vector.

This function defines the action of the rigid-body coupling matrix `B` such that
`u_ib = B * f`, where `f` is a body force distribution and `u_ib` is the resulting
velocity at the immersed boundary points.

Two methods are provided:
1. A wrapper that reinterprets flat arrays as vectors of `SVector{N,T}` and calls
   the core implementation.
2. The main routine, which:
   - Regularizes the body forces to the fluid grid.
   - Solves for the induced velocity field via the vorticity–streamfunction formulation.
   - Interpolates the resulting fluid velocity back to the body points.

This operation is used when assembling the coupling matrix in `B_inverse_rigid`
and represents how the fluid mediates the response of the rigid body to applied forces.

# Inputs
- `u_ib`: Output array for body velocities.
- `f`: Body force vector.
- `sol::CNAB`: CNAB solver containing grid, operators, and regularization data.

# Returns
- `u_ib`: The updated body velocity vector after applying the operator.
"""
function B_rigid_mul!(u_ib, f, sol::CNAB{N,T}) where {N,T}
    grid = sol.prob.grid
    h = grid.h
    ω = sol.ω
    ω¹ = grid_view(ω[1], grid, Loc_ω, ExcludeBoundary())

    with_arrays_like(sol.fluid_pool, sol.u[1], sol.ψ[1]) do u¹, ψ¹
        regularize!(u¹, sol.reg, f)
        rot!(ω¹, u¹; h)
        Ainv(sol, 1)(ω¹, ω¹)

        for level in 2:grid.levels, i in eachindex(ω[level])
            fill!(ω[level][i], 0)
        end

        with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψb
            multidomain_poisson!(ω, (ψ¹,), (u¹,), ψb, grid, sol.plan)
        end

        interpolate_body!(u_ib, sol.reg, u¹)
    end
end



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
            with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψ_b_work
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

    ωˢ = sol.ψ
    ψ = (sol.ω[1],)
    u¹ = sol.u[1]

    multidomain_poisson!(ωˢ, ψ, (u¹,), ψ_b_work, grid, sol.plan)
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
    B_deform_mul!(u_ib, f, sol::CNAB)

Apply the fluid–structure coupling operator `B` for a deformable body.

This function maps body forces `f` to immersed-boundary velocities `u_ib`,
accounting for force spreading, structural response, and velocity interpolation
back to the Lagrangian points.

Three methods are provided:
1. A converter that reinterprets flat scalar vectors as structured SVector arrays.
2. A wrapper that allocates workspace arrays.
3. The core routine that computes the coupled fluid–structure response.

# Arguments
- `u_ib` : Output array for body velocities (updated in-place).
- `f`    : Input body force vector.
- `sol::CNAB` : CNAB solver object containing grid, operators, and regularization.

# Returns
- `u_ib` updated in-place with the velocity induced by `f`.
"""
function B_deform_mul!(u_ib::AbstractVector{<:Number}, f, sol::CNAB{N,T}) where {N,T}
    S = SVector{N,T}
    B_deform_mul!(reinterpret(S, u_ib), reinterpret(S, f), sol)
end

function B_deform_mul!(u_ib, f, sol::CNAB)
    χ = sol.coupler.state.χ
    with_arrays_like(sol.body_pool, sol.f_tilde) do f_work
        with_arrays_like(sol.structure_pool, χ, χ) do f1, f2
            B_deform_mul!(u_ib, f, sol::CNAB, f_work, f1, f2)
        end
    end
end

function B_deform_mul!(u_ib, f, sol::CNAB, f_work, f1, f2)
    grid = sol.prob.grid
    body = sol.prob.body::GeometricNonlinearBody
    h = gridstep(grid)
    dt = sol.dt
    i_deform = deforming_point_range(body)
    u_ib_deform = view(u_ib, i_deform)
    f_work_deform = view(f_work, i_deform)

    u_ib .= f
    f_to_f_tilde!(u_ib, sol; inverse=true)
    redist!(u_ib, sol)
    fluid_to_structure_force!(f1, u_ib_deform, body, sol.coupler.ops)
    ldiv!(f2, sol.coupler.ops.Khat, f1)
    structure_to_fluid_displacement!(f_work_deform, f2, body, sol.coupler.ops)
    f_work_deform .*= 2 / dt

    B_rigid_mul!(u_ib, f, sol)
    u_ib_deform .+= f_work_deform

    u_ib
end



"""
    f_to_f_tilde!(f, sol::CNAB; inverse=false)

Convert between the physical body force `f` and its regularized (spread) form
`f_tilde` used in the fluid solver.

This function rescales the immersed boundary force depending on the direction
of conversion:
- When `inverse=false` (default), it converts `f_tilde → f`, applying the proper
  scaling for the boundary point spacing and coupling factor.
- When `inverse=true`, it converts `f → f_tilde`, restoring the fluid solver’s
  representation of the force.

This transformation ensures consistent units and coupling strength between the
structure and the fluid solvers.

# Notes
The conversion uses:
- `dt` : Time step size.
- `ds` : Arc length of immersed boundary points.
- `h`  : Grid spacing.
- `k = _f_tilde_factor(sol)` : Coupling-dependent scaling factor.

# Arguments
- `f` : Force vector, modified in-place.
- `sol::CNAB` : CNAB solver object containing time step, grid, and coupling parameters.
- `inverse` : Whether to apply the inverse scaling (`f → f_tilde`).

# Returns
This function returns `nothing`; the input `f` is modified in place.
"""
function f_to_f_tilde!(f, sol::CNAB; inverse=false)
    dt = sol.dt
    ds = @view sol.points.ds[eachindex(f)]
    h = sol.prob.grid.h
    k = _f_tilde_factor(sol)

    if inverse
        @. f *= -k / ds
    else
        @. f *= ds / -k
    end
end



"""
    redist!(f, sol::CNAB)

Redistribute forces on the immersed boundary to ensure consistency with the fluid.

This function corrects the body forces `f` after numerical operations by:
- Spreading the force to the fluid grid.
- Applying precomputed redistribution weights.
- Interpolating the corrected forces back to the body points.

# Arguments
- `f`: The body force vector (modified in-place).
- `sol::CNAB`: The CNAB solver state containing fluid and body information.

# Returns
- `nothing`: The input `f` is updated in-place.
"""
function redist!(f, sol::CNAB{N,T}) where {N,T}
    with_arrays_like(sol.fluid_pool, sol.u[1]) do u_work
        regularize!(u_work, sol.reg, f)

        for i in eachindex(u_work)
            u_work[i] .*= sol.redist_weights[i]
        end

        interpolate_body!(f, sol.reg, u_work)
    end
end



"""
    update_redist_weights!(sol::CNAB; tol=1e-10)

Compute the redistribution weights used in `redist!` to ensure consistent
transfer of forces between the immersed boundary and the fluid grid.

The weights correct for imbalances caused by spreading forces from
body points to the grid, so that later redistribution preserves the
physical accuracy of the simulation.

# Arguments
- `sol::CNAB`: The CNAB solver object containing the body and fluid state.
- `tol`: Minimum threshold for weight inversion to avoid division by zero (default `1e-10`).

# Returns
- `nothing`: The redistribution weights are stored in `sol.redist_weights` and updated in-place.
"""
function update_redist_weights!(sol::CNAB{N,T}; tol=T(1e-10)) where {N,T}
    w = sol.redist_weights
    backend = get_backend(w[1])

    with_arrays_like(sol.fluid_pool, sol.f_tilde) do f
        reinterpret(T, f) .= 1
        regularize!(w, sol.reg, f)
    end

    for wi in w
        @loop backend (I in CartesianIndices(wi)) begin
            wi[I] = wi[I] < tol ? zero(T) : 1 / wi[I]
        end
    end
end



"""
    projection_step!(sol::CNAB)

Project the fluid vorticity field to remove the effect of the applied body forces (`f_tilde`),
ensuring that the flow field satisfies the updated constraints after force spreading and redistribution.

# Arguments
- `sol::CNAB`: The CNAB solver object containing the fluid and body state.

# Effects
- Modifies `sol.ω` in-place to account for the applied body forces.
- Swaps `sol.ω` and `sol.ψ` internally to reuse memory.

# Returns
- `nothing`: The projection modifies the solver state in-place.
"""
function projection_step!(sol::CNAB{N,T}) where {N,T}
    grid = sol.prob.grid
    backend = get_backend(sol.u[1][1])

    u_axes = cell_axes(grid, Loc_u, IncludeBoundary())
    ω_axes = cell_axes(grid, Loc_ω, ExcludeBoundary())

    with_arrays(sol.fluid_pool, (T, u_axes), (T, ω_axes)) do u_work, ω_work
        regularize!(u_work, sol.reg, sol.f_tilde)
        rot!(ω_work, u_work; h=grid.h)
        Ainv(sol, 1)(ω_work, ω_work)

        (sol.ω, sol.ψ) = (sol.ψ, sol.ω)

        for i in eachindex(ω_work)
            let ω = sol.ω[1][i], ω_work = ω_work[i]
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
- Updates `sol.u` (velocity field) and ensures `sol.ω` satisfies boundary conditions.
- Handles all multigrid levels, applying necessary interpolations between levels.

# Returns
- `nothing`: Modifies the solver state in-place.
"""
function apply_vorticity!(sol::CNAB{N,T}) where {N,T}
    with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψ_b_work
        apply_vorticity!(sol, ψ_b_work)
    end
end

function apply_vorticity!(sol::CNAB, ψ_b_work)
    grid = sol.prob.grid
    multidomain_poisson!(sol.ω, sol.ψ, sol.u, ψ_b_work, grid, sol.plan)

    for level in 1:grid.levels
        if level == grid.levels
            for i in eachindex(ψ_b_work)
                foreach(b -> fill!(b, 0), ψ_b_work[i])
            end
        else
            multidomain_interpolate!(ψ_b_work, sol.ω[level+1]; n=grid.n)
        end

        set_boundary!(sol.ω[level], ψ_b_work)

        add_flow!(sol.u[level], sol.prob.u0, grid, level, sol.i, sol.t)
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



"""
    _f_tilde_factor(sol)

Compute the scaling factor used to convert between the physical body force `f` 
and the regularized force `f_tilde` used in the fluid solver.

# Arguments
- `sol`: CNAB solver object containing grid and time step information.

# Returns
- A scalar factor `k = - h^N / Δt` where:
  - `h` is the uniform grid spacing
  - `N` is the spatial dimension
  - `Δt` is the time step size
  - The negative sign follows the solver convention for force transformation

# Notes
- This factor is used in `f_to_f_tilde!` to scale forces correctly between 
  the immersed boundary and the fluid grid.
"""
function _f_tilde_factor(sol::CNAB{N}) where {N}
    grid = sol.prob.grid
    -grid.h^N / sol.dt
end



"""
    surface_force!(f, sol)

Convert the redistributed force `f_tilde` back to the physical surface force `f`.

# Arguments
- `f`: Output array that will store the physical surface force.
- `sol`: CNAB solver object containing `f_tilde` and grid/time info.

# Behavior
- Computes a scaling factor `k = -h^N / Δt` using `_f_tilde_factor(sol)`.
- Applies the formula `f .= -k * sol.f_tilde` to recover the actual force on the body surface.

# Notes
This reverses the scaling applied in `f_to_f_tilde!`.

# Returns
- `nothing`: The physical surface force is written in-place to `f`.
"""
function surface_force!(f, sol::CNAB)
    k = _f_tilde_factor(sol)
    @. f = -k * sol.f_tilde
end



"""
    surface_force_sum(sol)

Compute the total hydrodynamic force exerted by the fluid on the immersed body.

# Arguments
- `sol`: CNAB solver object containing `f_tilde` and grid/time info.

# Behavior
- Computes the scaling factor `k = -h^N / Δt` using `_f_tilde_factor(sol)`.
- Sums all entries of `sol.f_tilde` (the redistributed force at Lagrangian points).
- Scales and flips the sign to recover the physical total force:
  `total_force = -k * sum(sol.f_tilde)`.

# Returns
- `total_force`: The net physical force vector acting on the body.
"""
function surface_force_sum(sol::CNAB)
    k = _f_tilde_factor(sol)
    -k * sum(sol.f_tilde)
end



"""
    const CNAB_signature

A compile-time constant used as a unique identifier for the `CNAB` structure.
It stores the string `FastIBPM.jl:CNAB` as a vector of bytes (`Vector{UInt8}`).

This signature can be used, for example, in type-checking, serialization, or validation routines.
"""
const CNAB_signature = Vector{UInt8}("FastIBPM.jl:CNAB")



"""
    save(io::IO, sol::CNAB)

Serialize the current state of a `CNAB` simulation and write it to the given I/O stream
(e.g., a file). This allows the simulation to be saved and later restored.

# Arguments
- `io`: An I/O stream to write the binary data to (e.g., a file handle).
- `sol`: The `CNAB` solver object containing the current simulation state.

# Output
- Nothing; writes directly to the provided I/O stream.
"""
function save(io::IO, sol::CNAB{N,T}) where {N,T}
    grid = sol.prob.grid

    write(io, CNAB_signature)
    write(io, htol(UInt32(sizeof(T))))
    write(io, htol(UInt32(N)))
    write(io, htol.(SVector{N,UInt32}(grid.n)))
    write(io, htol(UInt32(grid.levels)))
    write(io, htol(Int32(sol.i)))

    # Messy because copy!(a, b) where b is a CUDA array only seems to work when a is an
    # Array (not a view or OffsetArray).

    let ω_tmp = map(ax -> zeros(T, length.(ax)), cell_axes(grid, Loc_ω, IncludeBoundary()))
        for ω_lev in sol.ω, (i, ω_i) in pairs(ω_lev)
            copy!(ω_tmp[i], no_offset_view(ω_i))
            a = view(
                OffsetArray(ω_tmp[i], axes(ω_i)),
                cell_axes(grid, Loc_ω(i), ExcludeBoundary())...,
            )
            @. a = htol(a)
            write(io, a)
        end
    end

    write(io, htol(UInt32(sol.nonlin_count)))

    let ω_tmp = map(ax -> zeros(T, length.(ax)), cell_axes(grid, Loc_ω, ExcludeBoundary()))
        for k in 1:sol.nonlin_count,
            nonlin_lev in sol.nonlin[k],
            (i, nonlin_i) in pairs(nonlin_lev)

            copy!(ω_tmp[i], no_offset_view(nonlin_i))
            @. ω_tmp[i] = htol(ω_tmp[i])
            write(io, ω_tmp[i])
        end
    end

    nothing
end



"""
    load!(io::IO, sol::CNAB)

Restore a previously saved `CNAB` simulation state from an I/O stream (e.g., a binary file)
into an existing solver object. This reconstructs the simulation exactly as it was when saved.

# Arguments
- `io`: An I/O stream to read the binary data from (e.g., a file handle).
- `sol`: The `CNAB` solver object to populate with the loaded state.

# Output
- Nothing; the `sol` object is updated in-place with the loaded simulation state.
"""
function load!(io::IO, sol::CNAB{N,T}) where {N,T}
    grid = sol.prob.grid

    @assert read(io, length(CNAB_signature)) == CNAB_signature
    @assert ltoh(read(io, UInt32)) == sizeof(T)
    @assert ltoh(read(io, UInt32)) == N
    @assert ltoh.(read(io, SVector{N,UInt32})) == grid.n
    @assert ltoh(read(io, UInt32)) == grid.levels

    i = Int(ltoh(read(io, Int32)))
    set_time!(sol, i)

    let ω_tmp = map(ax -> zeros(T, length.(ax)), cell_axes(grid, Loc_ω, IncludeBoundary()))
        for ω_lev in sol.ω, (i, ω_i) in pairs(ω_lev)
            a = view(
                OffsetArray(ω_tmp[i], axes(ω_i)),
                cell_axes(grid, Loc_ω(i), ExcludeBoundary())...,
            )
            read!(io, a)
            @. a = ltoh(a)
            copy!(no_offset_view(ω_i), ω_tmp[i])
        end
    end

    sol.nonlin_count = ltoh(read(io, UInt32))

    let ω_tmp = map(ax -> zeros(T, length.(ax)), cell_axes(grid, Loc_ω, ExcludeBoundary()))
        for k in 1:sol.nonlin_count,
            nonlin_lev in sol.nonlin[k],
            (i, nonlin_i) in pairs(nonlin_lev)

            a = ω_tmp[i]
            read!(io, a)
            @. a = ltoh(a)
            copy!(no_offset_view(nonlin_i), a)
        end
    end

    apply_vorticity!(sol)

    nothing
end