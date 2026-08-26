"""
Primitive-variable (`IBPM`) assembly operators — Taira & Colonius (2007).

The truncated-Taylor viscous inverse `Ainv!` (`Bᴺ ≈ A⁻¹`), the coupling operators
`Q_mul!` / `QT_mul!` (`Q = [G  Eᵀ]`), and the modified-Poisson left-hand side
`B_mul!` (`B = QᵀBᴺQ`) — each in a uniform-`Grid` and a mass-weighted
`StretchedGrid` version.

Shared haloed-field machinery is in `ops_common.jl`.
"""

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
# `StretchedGrid` (via the stretched IBPM stepping in `cnab_ibpm.jl`). The key
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
