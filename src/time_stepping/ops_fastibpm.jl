"""
Streamfunction-vorticity (`FastIBPM`) assembly operators.

The spectral viscous inverse `Ainv` (an `EigenbasisTransform` applying
`(I - aΔ)⁻¹` through the Laplacian eigenbasis) and the body-coupling operators
`B_inverse_rigid` / `B_rigid_mul!` / `B_deform_mul!`.

See `ops_common.jl` for the master `Ainv` entry point this method serves.
"""

"""
    Ainv(sol::CNAB, level, ::FastIBPM)

Streamfunction-vorticity (`FastIBPM`) method of the viscous inverse: applies
`(I - aΔ)⁻¹` spectrally via the Laplacian eigenbasis, on the given grid `level`.

Returns an `EigenbasisTransform` — an *operator object*, applied as
`Ainv(sol, level)(y, x)`. This is possible because the spectral inverse is exact
and carries its state in the precomputed FFT plans (`sol.state.plan`). The `IBPM`
counterpart [`Ainv!`](@ref) is instead an in-place applier; see the master
[`Ainv`](@ref) docstring for the comparison.
"""
function Ainv(sol::CNAB, level, ::FastIBPM)
    h = gridstep(sol.prob.grid, level)
    a = _A_factor(sol)
    EigenbasisTransform(λ -> 1 / (1 - a * λ / h^2), sol.state.plan)
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
    ω = sol.state.ω
    ω¹ = grid_view(ω[1], grid, Loc_ω, ExcludeBoundary())

    with_arrays_like(sol.fluid_pool, sol.state.u[1], sol.state.ψ[1]) do u¹, ψ¹
        regularize!(u¹, sol.reg, f)
        rot!(ω¹, u¹; h)
        Ainv(sol, 1)(ω¹, ω¹)

        for level in 2:grid.levels, i in eachindex(ω[level])
            fill!(ω[level][i], 0)
        end

        with_arrays(sol.bndry_pool, (T, sol.state.ω_bndry)) do ψb
            multidomain_poisson!(ω, (ψ¹,), (u¹,), ψb, grid, sol.state.plan)
        end

        interpolate_body!(u_ib, sol.reg, u¹)
    end
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
