"""
Assembly operators for the immersed-boundary time integration.

Collects the *operator-level* building blocks used by the time steppers in
`cnab.jl` (as opposed to the per-iteration stepping logic itself). Each operator
is dispatched on the formulation so the two schemes can share entry points:

  - `FastIBPM` — streamfunction-vorticity (the fast multidomain method):
    `Ainv` (spectral inverse of the implicit viscous operator) and the coupling
    operators `B_inverse_rigid` / `B_rigid_mul!` / `B_deform_mul!`.
  - `IBPM` — primitive-variable projection (Taira & Colonius): `Ainv!`
    (truncated-Taylor `Bᴺ ≈ A⁻¹`, built from repeated Laplacian applications).

The shared coefficient `_A_factor = Δt/(2Re)` is formulation-independent.
"""

# ===========================================================================
# Streamfunction-vorticity (FastIBPM) assembly operators
# ===========================================================================

"""
    _A_factor(sol::CNAB)

Diffusion coefficient `a = Δt / (2 Re)` of the semi-implicit (Crank-Nicolson)
viscous term. Formulation-independent.
"""
_A_factor(sol::CNAB) = sol.dt / (2sol.prob.Re)

"""
    Ainv(sol::CNAB, level)

Inverse of the implicit viscous operator `A = I - aΔ`. Master entry point: it
dispatches on the problem formulation (`sol.prob.formulation`) to the
formulation-specific implementation.
"""
Ainv(sol::CNAB, level) = Ainv(sol, level, sol.prob.formulation)

"""
    Ainv(sol::CNAB, level, ::FastIBPM)

Streamfunction-vorticity implementation: applies `(I - aΔ)⁻¹` spectrally via the
Laplacian eigenbasis (`EigenbasisTransform`) on the given grid `level`.
"""
function Ainv(sol::CNAB, level, ::FastIBPM)
    h = gridstep(sol.prob.grid, level)
    a = _A_factor(sol)
    EigenbasisTransform(λ -> 1 / (1 - a * λ / h^2), sol.plan)
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

# ===========================================================================
# Primitive-variable (IBPM) assembly operators
# ===========================================================================

"""
    Ainv_zeros(backend, grid)

Allocate zero-filled "haloed" velocity work fields for the `Ainv!` (`Bᴺ`) apply.

Each velocity component `u[i]` is stored over its interior-unknown box
(`Loc_u(i)`, `ExcludeBoundary`) expanded by one cell on every side. That extra
ring is a homogeneous ghost halo: inside `Bᴺ` the boundary conditions live on the
right-hand side (`bc1`), so the Laplacian acts with zero boundary/ghost values,
and the (zero) halo lets the stencil read its neighbors without going out of
bounds — including the transverse edges the plain `Loc_u` layout omits.
"""
function Ainv_zeros(backend, grid::Grid{N,T}) where {N,T}
    map(edge_axes(Val(N), Loc_u)) do i
        Re = cell_axes(grid, Loc_u(i), ExcludeBoundary())
        Rh = map(r -> (first(r)-1):(last(r)+1), Re)
        OffsetArray(KernelAbstractions.zeros(backend, T, length.(Rh)), Rh)
    end
end

# Interior index ranges of a haloed field: its axes shrunk by one cell on every
# side (inverse of the expansion in `Ainv_zeros`) — the box the Laplacian is
# evaluated over.
_interior_range(a) = map(r -> (first(r)+1):(last(r)-1), UnitRange.(axes(a)))

"""
    _apply_aL!(dest, src, a, h)

Compute `dest = a · L src` over the interior of each component, in place, where
`L` is the discrete Laplacian (`laplacian`). `src` must be a haloed field (from
`Ainv_zeros`) so the stencil can read its neighbors; only `dest`'s interior is
written.
"""
function _apply_aL!(dest, src, a, h)
    for i in eachindex(dest)
        d = dest[i]
        s = src[i]
        backend = get_backend(d)
        R = CartesianIndices(_interior_range(d))
        @loop backend (I in R) d[I] = a * laplacian(s, I; h)
    end
    dest
end

"""
    Ainv!(y, x, term, tmp; a, dt, n_taylor, h)

Apply the truncated viscous inverse `y = Bᴺ x` in place — the primitive-variable
(`IBPM`) viscous inverse, the in-place counterpart of the streamfunction-vorticity
`Ainv(sol, level, ::FastIBPM)` (which instead returns a spectral operator).

Implements the Taira & Colonius (2007) Taylor approximation of `A⁻¹` (their
Eq. 5) on a uniform grid (mass matrix `M = I`):

    Bᴺ = Δt · Σ_{k=0}^{n_taylor-1} (a L)^k  ≈  A⁻¹,
    A  = (1/Δt) I - (1/2Re) L,   a = Δt/(2Re),

by Horner accumulation — each term is the previous one with one more application
of `a L`. `L` is the discrete Laplacian; no spectral solve is used.

All of `y`, `x`, `term`, `tmp` are haloed velocity fields (see `Ainv_zeros`) with
zero halo; `x` holds the (homogeneous-BC) input on its interior and `y` receives
the result. `term` and `tmp` are scratch.

# Arguments
- `y`: output field (modified in-place).
- `x`: input field.
- `term`, `tmp`: scratch fields, same shape as `y`.
- `a`, `dt`, `n_taylor`, `h`: viscous coefficient `Δt/(2Re)`, time step, number of
  Taylor terms (3 recommended), and grid spacing (keywords).

# Returns
The updated field `y`.
"""
function Ainv!(y, x, term, tmp; a, dt, n_taylor, h)
    for i in eachindex(y)
        _set!(y[i], x[i])       # k = 0 term
        _set!(term[i], x[i])
    end
    for _ in 2:n_taylor
        _apply_aL!(tmp, term, a, h)     # tmp = a L term
        for i in eachindex(y)
            _set!(term[i], tmp[i])      # advance the running term
            backend = get_backend(y[i])
            let yi = y[i], ti = term[i]
                @loop backend (I in CartesianIndices(yi)) yi[I] += ti[I]
            end
        end
    end
    for i in eachindex(y)
        backend = get_backend(y[i])
        let yi = y[i]
            @loop backend (I in CartesianIndices(yi)) yi[I] *= dt
        end
    end
    y
end
