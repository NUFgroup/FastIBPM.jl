"""
CNAB time integration: the parts shared by every formulation.

Holds the clock (`set_time!`), the public entry point (`step!`), the body-geometry
refresh (`update_reg!`), the Adams-Bashforth coefficients (`ab_coeffs`), and the
*generic* stage entry points that dispatch on `sol.prob.formulation`.

Each formulation's own pipeline lives beside this file:

  - `cnab_fastibpm.jl` — streamfunction-vorticity (Colonius & Taira 2008)
  - `cnab_ibpm.jl`     — primitive variables (Taira & Colonius 2007), uniform and stretched
  - `cnab_imap.jl`     — manifold projection

The operator-level building blocks are in `ops_common.jl` and the matching
`ops_*.jl` files.
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
