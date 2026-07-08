"""
CNAB force-redistribution helpers for the immersed-boundary interface coupling.

These operators convert between the physical body force `f` and the regularized
force `f_tilde`, and redistribute forces so that spreading (Eᵀ) followed by
interpolation (E) is consistent. They all dispatch on `CNAB`, so this file MUST
be included AFTER the CNAB struct is defined (init/problems.jl).

Split out from interface_coupling.jl to resolve the Reg → CNAB → coupling-method
circular include-order dependency: Reg (in interface_coupling.jl) is included
before CNAB; these CNAB-dependent helpers are included after.
"""

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
    f_to_f_tilde!(f, sol::CNAB; inverse=false)

Convert between the physical body force `f` and its regularized (spread) form
`f_tilde` used in the fluid solver.

This function rescales the immersed boundary force depending on the direction
of conversion:
- When `inverse=false` (default), it converts `f_tilde → f`, applying the proper
  scaling for the boundary point spacing and coupling factor.
- When `inverse=true`, it converts `f → f_tilde`, restoring the fluid solver's
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
    dt = sol.dt  # TODO: check if dt is needed here
    ds = @view sol.points.ds[eachindex(f)]
    h = sol.prob.grid.h  # TODO: check if h is needed here
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
