"""
Assembly operators for the immersed-boundary time integration.

Collects the *operator-level* building blocks used by the time steppers in
`cnab.jl` (as opposed to the per-iteration stepping logic itself), organized by
formulation:

  - **`FastIBPM`** — streamfunction-vorticity (the fast multidomain method):
    `Ainv` (spectral inverse of the implicit viscous operator) and the coupling
    operators `B_inverse_rigid` / `B_rigid_mul!` / `B_deform_mul!`.
  - **`IBPM`** — primitive-variable projection (Taira & Colonius): `Ainv!`
    (truncated-Taylor `Bᴺ ≈ A⁻¹` from repeated Laplacian applications), the
    coupling operators `Q_mul!` / `QT_mul!` (`Q = [G  Eᵀ]`), and the
    modified-Poisson left-hand side `B_mul!` (`B = QᵀBᴺQ`).

Where the two formulations expose the *same* interface they share a
formulation-dispatched entry point (as `CNAB_Binv_Iterative` does for `B⁻¹`).
Where they do not, they deliberately stay separate: the viscous inverse is the
main case — `FastIBPM`'s `Ainv` *returns an operator object*, while `IBPM`'s
`Ainv!` *applies in place* and needs haloed work fields, so there is no
`Ainv(sol, level, ::IBPM)` method. See the `Ainv` docstring.

The coefficient `_A_factor = Δt/(2Re)` is formulation-independent.
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

Inverse of the implicit viscous operator `A = I - aΔ`, with `a = Δt/(2Re)`
([`_A_factor`](@ref)). Master entry point: dispatches on the problem formulation
(`sol.prob.formulation`) to the formulation-specific method.

The two formulations invert `A` by genuinely different means, and — unlike
`CNAB_Binv_Iterative` — they do **not** yet share this entry point:

  - **`FastIBPM`** — [`Ainv(sol, level, ::FastIBPM)`](@ref) below. Returns an
    *operator object* (an `EigenbasisTransform`) that applies `(I - aΔ)⁻¹`
    spectrally; it is used as `Ainv(sol, level)(y, x)`.
  - **`IBPM`** — the viscous inverse is [`Ainv!`](@ref), the truncated-Taylor
    `Bᴺ ≈ A⁻¹` built from repeated Laplacian applications (no spectral solve).
    It has a *different interface*: it applies `A⁻¹` **in place** and needs haloed
    velocity fields plus scratch ([`Ainv_zeros`](@ref)), so it cannot be returned
    from this entry point as an operator object.

!!! note
    Consequently there is **no `Ainv(sol, level, ::IBPM)` method**: calling this
    entry point on an `IBPM` problem is a `MethodError` by design. Use [`Ainv!`](@ref)
    directly. A wrapper making `IBPM` reachable here would have to carry the
    haloed work fields, so it belongs with the primitive time-stepper.
"""
Ainv(sol::CNAB, level) = Ainv(sol, level, sol.prob.formulation)

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
function Ainv_zeros(backend, grid::AbstractGrid{N,T}) where {N,T}
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

!!! warning "Convergence constraint on Δt"
    This is a truncated **Neumann series**, so it only approximates `A⁻¹` when the
    spectral radius of `a L` is below one — i.e. when `a/h² ≲ 1`, the paper's
    `νΔt/Δx² ≲ 1` condition. Beyond that the series *diverges*: the eigenvalues of
    `Bᴺ` blow up instead of decaying, and any operator built on it (notably
    `B = QᵀBᴺQ`) becomes severely ill-conditioned. Unlike the spectral `FastIBPM`
    `Ainv`, which is exact for any `Δt`, this places a real upper bound on `Δt`
    for the primitive scheme.

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

"""
    Q_mul!(q, φ, f_tilde, reg; h)

Apply the coupling operator `Q = [G  Eᵀ]` to the Lagrange multiplier
`λ = (φ, f_tilde)`, in place — matrix-free (`Q` is never assembled):

    q = Q λ = G φ + Eᵀ f_tilde

Following Taira & Colonius (2007) Eq. 22, the discrete pressure `φ` and the
transformed boundary force `f_tilde` are grouped into a single Lagrange
multiplier, so `Q` maps that pair into velocity space. `G` is the discrete
gradient ([`gradient`](@ref)) and `Eᵀ` is the regularization (spreading) operator
([`regularize!`](@ref)).

`regularize!` zeroes its whole output field before accumulating, so calling it
first both seeds `q` with `Eᵀ f_tilde` *and* leaves the halo at zero (as the
homogeneous-BC operators require); `G φ` is then added over the interior-unknown
box only. No scratch field is needed.

# Arguments
- `q`: haloed velocity field (see [`Ainv_zeros`](@ref)); overwritten with `Q λ`.
- `φ`: cell-centered pressure (`Loc_p`).
- `f_tilde`: transformed boundary force at the Lagrangian points.
- `reg`: regularization/interpolation structure (`Reg`).
- `h`: grid spacing (keyword).

# Returns
The updated field `q`.
"""
function Q_mul!(q, φ, f_tilde, reg; h)
    regularize!(q, reg, f_tilde)              # q = Eᵀ f_tilde  (also zeroes the halo)
    for i in eachindex(q)
        qᵢ = q[i]
        backend = get_backend(qᵢ)
        R = CartesianIndices(_interior_range(qᵢ))
        @loop backend (I in R) qᵢ[I] += gradient(i, φ, I; h)
    end
    q
end

"""
    QT_mul!(φ, f_tilde, q, reg; h)

Apply the transpose coupling operator `Qᵀ = [Gᵀ; E]` to a velocity field `q`,
in place — matrix-free:

    Qᵀ q = (Gᵀ q, E q) = (-D q, E q)

On the staggered grid `G = -Dᵀ`, hence `Gᵀ = -D` (Taira & Colonius Eq. 50), so the
first block is just the negated discrete divergence — no separate operator is
needed. The second block is the interpolation `E` ([`interpolate_body!`](@ref)).

Because the velocity unknowns carry homogeneous boundary values inside the
projection (the physical BCs live on the right-hand side as `bc1`/`bc2`), `q`'s
boundary faces and halo are zero, so `divergence!` over all cells returns exactly
the interior part of `D q`.

# Arguments
- `φ`: cell-centered output (`Loc_p`); overwritten with `Gᵀ q = -D q`.
- `f_tilde`: body-point output; overwritten with `E q`.
- `q`: haloed velocity field.
- `reg`: regularization/interpolation structure (`Reg`).
- `h`: grid spacing (keyword).

# Returns
The tuple `(φ, f_tilde)`.
"""
function QT_mul!(φ, f_tilde, q, reg; h)
    divergence!(φ, q; h)                      # φ = D q
    backend = get_backend(φ)
    @loop backend (I in CartesianIndices(φ)) φ[I] = -φ[I]   # Gᵀ = -D
    interpolate_body!(f_tilde, reg, q)        # f_tilde = E q
    (φ, f_tilde)
end

"""
    B_work(backend, grid)

Allocate the haloed velocity scratch fields needed by [`B_mul!`](@ref) (the
`IBPM` modified-Poisson operator): the intermediate `Q λ`, the result `Bᴺ Q λ`,
and the two scratch fields consumed by [`Ainv!`](@ref).
"""
function B_work(backend, grid::AbstractGrid)
    (;
        q=Ainv_zeros(backend, grid),
        y=Ainv_zeros(backend, grid),
        term=Ainv_zeros(backend, grid),
        tmp=Ainv_zeros(backend, grid),
    )
end

"""
    B_mul!(φ_out, f_out, φ, f_tilde, reg, work, ::IBPM; h, a, dt, n_taylor)

Apply the primitive-variable projection-step left-hand side `B`, in place —
matrix-free:

    B λ = Qᵀ Bᴺ Q λ,      λ = (φ, f_tilde)

This is the *modified Poisson* operator of Taira & Colonius (2007) Eq. 26. It is
the `IBPM` counterpart of the streamfunction-vorticity `B` (whose action is
[`B_rigid_mul!`](@ref) / [`B_deform_mul!`](@ref)) — note the two act on different
spaces: in `FastIBPM` continuity is automatic (`u = ∇×ψ`), so `B` sees only the
body force, whereas here the Lagrange multiplier `λ` carries the pressure *and*
the force.

It is applied as the composition [`Q_mul!`](@ref) → [`Ainv!`](@ref) →
[`QT_mul!`](@ref); nothing is assembled. Because `Bᴺ` is symmetric and
`Qᵀ(·)Q` is a congruence, `B` is symmetric positive semi-definite — the property
that lets the modified Poisson system be solved by conjugate gradients (its only
null direction is the constant-pressure mode, which is pinned by the solver).

# Arguments
- `φ_out`, `f_out`: output blocks of `B λ` (cell-centered / body-point).
- `φ`, `f_tilde`: input blocks of `λ`.
- `reg`: regularization/interpolation structure (`Reg`).
- `work`: scratch from [`B_work`](@ref).
- `h`, `a`, `dt`, `n_taylor`: grid spacing, viscous coefficient `Δt/(2Re)`, time
  step, and number of Taylor terms (keywords).

# Returns
The tuple `(φ_out, f_out)`.
"""
function B_mul!(φ_out, f_out, φ, f_tilde, reg, work, ::IBPM; h, a, dt, n_taylor)
    Q_mul!(work.q, φ, f_tilde, reg; h)                        # q  = Q λ
    Ainv!(work.y, work.q, work.term, work.tmp; a, dt, n_taylor, h)  # y = Bᴺ Q λ
    QT_mul!(φ_out, f_out, work.y, reg; h)                     # out = Qᵀ Bᴺ Q λ
    (φ_out, f_out)
end

# ===========================================================================
# Primitive-variable (IBPM) assembly — StretchedGrid methods
# ===========================================================================
#
# Parallel to the uniform methods above, but mass-weighted so the operators stay
# symmetric on a non-uniform grid (see the FV operators in `stretched_domain.jl`).
# The uniform `Grid` methods are untouched; these are reached only for a
# `StretchedGrid` (via the stretched IBPM stepping in `cnab.jl`). The key
# difference is the mass matrix `M`: `Bᴺ = Δt Σₖ (a M⁻¹ L_sym)ᵏ M⁻¹`.

# y = M⁻¹ x  (elementwise divide by the diagonal mass over the interior).
function _mass_inv!(y, x, grid::StretchedGrid)
    for i in eachindex(y)
        yi, xi = y[i], x[i]
        backend = get_backend(yi)
        R = CartesianIndices(_interior_range(yi))
        @loop backend (I in R) yi[I] = xi[I] / mass(grid, i, I)
    end
    y
end

# dest = a · L_sym · src  (symmetric FV Laplacian; no M⁻¹ — that is applied
# separately inside `Ainv!`). Used both for `r1` and for the `Bᴺ` iteration.
function _apply_aL!(dest, src, a, grid::StretchedGrid)
    for i in eachindex(dest)
        d, s = dest[i], src[i]
        backend = get_backend(d)
        R = CartesianIndices(_interior_range(d))
        @loop backend (I in R) d[I] = a * laplacian(s, I, i, grid)
    end
    dest
end

# y = Bᴺ x = Δt Σ_{k=0}^{n_taylor-1} (a M⁻¹ L_sym)ᵏ M⁻¹ x, by Horner accumulation.
function Ainv!(y, x, term, tmp, grid::StretchedGrid; a, dt, n_taylor)
    _mass_inv!(y, x, grid)                       # y = z = M⁻¹ x   (k = 0 term)
    for i in eachindex(y)
        _set!(term[i], y[i])
    end
    for _ in 2:n_taylor
        _apply_aL!(tmp, term, a, grid)           # tmp = a L_sym term
        _mass_inv!(tmp, tmp, grid)               # tmp = a M⁻¹ L_sym term = P·term
        for i in eachindex(y)
            _set!(term[i], tmp[i])
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

# q = Q λ = G φ + Eᵀ f_tilde  (weighted gradient).
function Q_mul!(q, φ, f_tilde, reg, grid::StretchedGrid)
    regularize!(q, reg, f_tilde)                 # q = Eᵀ f_tilde (also zeroes the halo)
    for i in eachindex(q)
        qᵢ = q[i]
        backend = get_backend(qᵢ)
        R = CartesianIndices(_interior_range(qᵢ))
        @loop backend (I in R) qᵢ[I] += gradient(i, φ, I, grid)
    end
    q
end

# Qᵀ q = (Gᵀ q, E q) = (-D q, E q)  (weighted divergence; D = -Gᵀ holds exactly).
function QT_mul!(φ, f_tilde, q, reg, grid::StretchedGrid)
    divergence!(φ, q, grid)
    backend = get_backend(φ)
    @loop backend (I in CartesianIndices(φ)) φ[I] = -φ[I]
    interpolate_body!(f_tilde, reg, q)
    (φ, f_tilde)
end

# B λ = Qᵀ Bᴺ Q λ.
function B_mul!(φ_out, f_out, φ, f_tilde, reg, work, ::IBPM, grid::StretchedGrid; a, dt, n_taylor)
    Q_mul!(work.q, φ, f_tilde, reg, grid)
    Ainv!(work.y, work.q, work.term, work.tmp, grid; a, dt, n_taylor)
    QT_mul!(φ_out, f_out, work.y, reg, grid)
    (φ_out, f_out)
end
