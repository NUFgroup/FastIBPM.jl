"""
Manifold-projection (`IMAP`) assembly operators.

The constraint projector `ManifoldProjection` / `P_mul!` and its multiplier
`constraint_multiplier!`, the projected viscous inverse `Ainv_IMAP!`, and the
pressure-solve operators `G_mul!` / `GT_mul!` / `B_mul!` (`B = GᵀBᴺPG`). IMAP
solves for no boundary force, so there is no `Eᵀ` block in its coupling operator.

Shared haloed-field machinery is in `ops_common.jl`.
"""

# ===========================================================================
# Manifold-projection (IMAP) assembly operators
# ===========================================================================
#
# IMAP keeps the primitive-variable structure of `IBPM` but eliminates the
# immersed-boundary force: instead of solving for an `f` that enforces no-slip,
# every term that would have produced one is projected onto the constraint
# manifold `Rᵀu = 0`. This file holds that projector; the `IBPM` operators in
# `ops_ibpm.jl` are untouched.
#
# Notation. The document writes the constraint as `Rᵀu = 0`, with `Rᵀ`
# interpolating the Eulerian velocity to the Lagrangian points. In this code
# that interpolation is `E` (`interpolate_body!`) and its adjoint, the spreading
# operator, is `Eᵀ` (`regularize!`). So
#
#     R = Eᵀ,   Rᵀ = E,   RᵀR = E Eᵀ,
#
# and the projector is `P = I - Eᵀ(EEᵀ)⁻¹E`. `E` and `Eᵀ` are exact ℓ² adjoints
# (the property `test_Q` checks), so `P` is symmetric as well as idempotent.

"""
    ManifoldProjection(backend, grid, reg, n_ib)

Precomputed constraint projector `P = I - R(RᵀR)⁻¹Rᵀ` for the `IMAP`
formulation, with `R = Eᵀ` the spreading operator and `Rᵀ = E` the
interpolation to body points (see the notation note above).

`P` is never assembled. What *is* assembled is the small dense Gram matrix
`RᵀR = E Eᵀ`, which is only `(N·n_b)` square — the same size regime as the
`FastIBPM` coupling matrix, and built the same way: apply `E Eᵀ` to each unit
vector in turn ([`B_inverse_rigid`](@ref) does this for `B`). For a **static
body** `R` never changes, so the Cholesky factorization is computed once here
and reused by every [`P_mul!`](@ref) for the rest of the run.

Note this is the whole of IMAP's body coupling: because no boundary force is an
unknown, there is no `(N·n_b)`-sized Krylov solve per step — only this one
back-substitution per projection.

!!! note "Body placement"
    The projector is applied to *haloed* velocity fields (see [`Ainv_zeros`](@ref))
    and updates their interior only, leaving the halo at zero as the homogeneous
    operators require. That is exact as long as the body's delta-function support
    lies inside the interior box, i.e. the body is not pressed against ∂D — the
    same assumption the `IBPM` regularization already makes.

# Fields
- `chol` : Cholesky factorization of the Gram matrix `RᵀR = E Eᵀ`.
- `reg`  : the regularization structure defining `E`/`Eᵀ` (held by reference, so
  a future moving-body variant can refactorize in place against updated weights).
- `λ`    : body-space scratch holding the constraint multiplier `(RᵀR)⁻¹Rᵀv`.
- `q`    : haloed velocity scratch holding the correction `R λ`.
"""
struct ManifoldProjection{C,R<:Reg,Vb,Q}
    chol::C
    reg::R
    λ::Vb
    q::Q
end

function ManifoldProjection(backend, grid::AbstractGrid{N,T}, reg::Reg, n_ib) where {N,T}
    n = N * n_ib
    gram = KernelAbstractions.zeros(backend, T, n, n)
    λ = KernelAbstractions.zeros(backend, SVector{N,T}, n_ib)
    q = Ainv_zeros(backend, grid)

    # Column j of RᵀR is Rᵀ R eⱼ: spread the unit body vector, interpolate back.
    e = KernelAbstractions.zeros(backend, T, n)
    let eb = reinterpret(SVector{N,T}, e)
        for j in 1:n
            @. e = ifelse((1:n) == j, 1, 0)
            regularize!(q, reg, eb)          # q = R eⱼ
            interpolate_body!(λ, reg, q)     # λ = Rᵀ R eⱼ
            @view(gram[:, j]) .= reinterpret(T, λ)
        end
    end

    chol = try
        cholesky!(Hermitian(gram))
    catch err
        err isa PosDefException || rethrow()
        throw(
            ArgumentError(
                "the IMAP Gram matrix RᵀR = E Eᵀ is not positive definite, so the " *
                "constraint Rᵀu = 0 is degenerate. This means the body points are not " *
                "independently resolvable by the grid — usually because Δs ≪ Δx, so " *
                "neighbouring points share a delta-function stencil. Coarsen the body " *
                "discretization towards Δs ≈ Δx, or refine the grid.",
            ),
        )
    end

    ManifoldProjection(chol, reg, λ, q)
end

"""
    constraint_multiplier!(λ, P::ManifoldProjection, v)

Compute the constraint multiplier

    λ = (RᵀR)⁻¹ Rᵀ v

in place — the body-space half of the projection [`P_mul!`](@ref), which removes
`R λ` from `v`.

Exposed separately because `λ` is not just an implementation detail of `P`: it is
the **boundary force** IMAP never solves for. Every projection discards exactly
the part of the momentum an explicit boundary force would have supplied, so these
multipliers recover `f̃` for free — see `recover_force!`.

# Returns
The updated body-space vector `λ`.
"""
function constraint_multiplier!(λ, P::ManifoldProjection, v)
    T = eltype(eltype(λ))
    interpolate_body!(λ, P.reg, v)                  # λ = Rᵀ v
    ldiv!(P.chol, reinterpret(T, λ))                # λ = (RᵀR)⁻¹ Rᵀ v
    λ
end

"""
    P_mul!(v, P::ManifoldProjection)

Apply the constraint projection `v ← P v` in place, matrix-free:

    P v = v - R (RᵀR)⁻¹ Rᵀ v

(the document's Eq. 3). Evaluated right-to-left as
[`interpolate_body!`](@ref) → Cholesky back-substitution →
[`regularize!`](@ref) → subtract, so the only grid-sized work is one
interpolation, one spread and one subtraction; the solve itself is `(N·n_b)`
square.

`v` is a haloed velocity field ([`Ainv_zeros`](@ref)); only its interior is
modified, so a zero halo stays zero.

On return `P.λ` holds the multiplier `(RᵀR)⁻¹Rᵀv` that was removed — the caller
may read it (it is overwritten by the next projection), which is how
`recover_force!` gets the boundary force without extra work.

# Returns
The projected field `v`.
"""
function P_mul!(v, P::ManifoldProjection)
    constraint_multiplier!(P.λ, P, v)               # λ = (RᵀR)⁻¹ Rᵀ v
    regularize!(P.q, P.reg, P.λ)                    # q = R λ   (also zeroes the halo)

    for i in eachindex(v)
        vi, qi = v[i], P.q[i]
        backend = get_backend(vi)
        R = CartesianIndices(_interior_range(vi))
        @loop backend (I in R) vi[I] -= qi[I]
    end
    v
end

"""
    Ainv_IMAP!(y, x, term, tmp, proj; a, dt, n_taylor, h, symmetric=true)

Apply the projected viscous inverse `y = Bᴺ x` in place — the `IMAP` counterpart
of [`Ainv!`](@ref).

IMAP's implicit viscous operator carries the constraint projection,

    A = (1/Δt) I - (1/2Re) P L,     a = Δt/(2Re),

so the same truncated Neumann series as `IBPM` applies with `L` replaced by `P L`:

    Bᴺ = Δt Σ_{k=0}^{n_taylor-1} (a P L)ᵏ  ≈  A⁻¹.

Everything the `IBPM` [`Ainv!`](@ref) docstring says about the series — Horner
accumulation, and the convergence bound on `Δt` — applies unchanged.

The bound is worth stating sharply, since `Ainv!`'s docstring gives it loosely.
The series converges iff `ρ(a·PLP) < 1`, and with the standard second-order
Laplacian (eigenvalues spanning `[-4N/h², 0]`) that is

    a/h² < 1/(4N)   ⟺   νΔt/Δx² < 1/(2N)   ⟺   Δt < Re·h²/(2N),

i.e. `a/h² < 0.125` in 2D and `< 0.083` in 3D — *not* the `a/h² ≲ 1` quoted there,
which is loose by a factor of `4N`. (Verified numerically: `ρ(aL)` measures
`4N·a/h²` to three digits.) Exceeding it does not blow up: `Bᴺ` is a polynomial
and, for odd `n_taylor`, still SPD, so the solve runs — but it stops approximating
`A⁻¹`, and past the bound *more* Taylor terms make it worse, not better.

# The `symmetric` keyword

`P L` is not symmetric (`P` and `L` each are, but they do not commute), so the
literal series `Σ(a P L)ᵏ` is not a symmetric operator on its own. Setting
`symmetric=true` (the default) uses the **symmetrized** series

    Bᴺ = Δt Σ_{k=0}^{n_taylor-1} (a P L P)ᵏ,

whose every term is symmetric.

The flag is *not*, however, what makes CG applicable. The pressure operator
`B = Gᵀ Bᴺ P G` is symmetric either way, because the `P G` on the right and the
`Gᵀ` on the left already supply the projections — see the `IMAP` method of
[`CNAB_Binv_Iterative`](@ref). Nor is the symmetrization an approximation on the
constraint manifold: since `P² = P`, expanding the product and collapsing each
doubled projector gives

    (a P L P)ᵏ = (a P L)ᵏ P     for k ≥ 1,

so every interior projector is redundant and the two variants differ *only* in
whether the series input is projected — coinciding exactly whenever `P x = x`,
which holds for both call sites (the prediction's `r1` and the pressure solve's
`P G φ`). That identity is also why the flag costs one extra `P` per apply rather
than one per Taylor term: it is implemented by projecting the running term once,
up front.

What the flag buys is *numerical*, and it is worth about 25% of the run time:
re-projecting makes every matvec land in exactly the same subspace, so the
warm-started CG sees a consistent operator instead of one perturbed at roundoff
level from iterate to iterate. See the `IMAPCoupling` docstring for the
measurements, and for the moving-body caveat.

# Arguments
- `y`: output field (modified in-place).
- `x`: input field.
- `term`, `tmp`: scratch fields, same shape as `y`.
- `proj`: the [`ManifoldProjection`](@ref).
- `a`, `dt`, `n_taylor`, `h`: as in [`Ainv!`](@ref) (keywords).
- `symmetric`: use the `P L P` series (default `true`).

# Returns
The updated field `y`.
"""
function Ainv_IMAP!(y, x, term, tmp, proj; a, dt, n_taylor, h, symmetric=true)
    for i in eachindex(y)
        _set!(y[i], x[i])       # k = 0 term — the leading I of A⁻¹, never projected
        _set!(term[i], x[i])
    end

    # (a P L P)ᵏ x = (a P L)ᵏ (P x) for k ≥ 1: one projection here symmetrizes the
    # whole series, since every later term already ends with a `P` and P² = P.
    if symmetric && n_taylor > 1
        P_mul!(term, proj)
    end

    for _ in 2:n_taylor
        _apply_aL!(tmp, term, a, h)     # tmp = a L term
        P_mul!(tmp, proj)               # tmp = a P L term
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
    G_mul!(q, φ; h)

Apply the `IMAP` coupling operator to the Lagrange multiplier `λ = φ`, in place:

    q = G φ

This is the `IBPM` [`Q_mul!`](@ref) with the `Eᵀ f_tilde` block deleted — IMAP
solves for no boundary force, so the multiplier is the pressure alone and the
coupling operator is just the discrete gradient.

!!! note
    Unlike `Q_mul!`, this must zero `q` itself. `Q_mul!` gets that for free from
    [`regularize!`](@ref), which clears its whole output (halo included) before
    accumulating; with the force block gone there is nothing to do it, and the
    homogeneous operators downstream require a zero halo.

# Returns
The updated field `q`.
"""
function G_mul!(q, φ; h)
    for i in eachindex(q)
        qᵢ = q[i]
        backend = get_backend(qᵢ)
        fill!(qᵢ, 0)                                    # incl. the halo
        R = CartesianIndices(_interior_range(qᵢ))
        @loop backend (I in R) qᵢ[I] = gradient(i, φ, I; h)
    end
    q
end

"""
    GT_mul!(φ, q; h)

Apply the transpose `IMAP` coupling operator to a velocity field, in place:

    Gᵀ q = -D q

the `IBPM` [`QT_mul!`](@ref) without the `E q` block. As there, `G = -Dᵀ` holds
exactly on the staggered grid, so the negated divergence *is* the transpose, and
`q`'s zero boundary/halo values make `divergence!` over all cells return exactly
the interior part of `D q`.

# Returns
The updated field `φ`.
"""
function GT_mul!(φ, q; h)
    divergence!(φ, q; h)                                # φ = D q
    backend = get_backend(φ)
    @loop backend (I in CartesianIndices(φ)) φ[I] = -φ[I]   # Gᵀ = -D
    φ
end

"""
    B_mul!(φ_out, φ, proj, work, ::IMAP; h, a, dt, n_taylor, symmetric=true)

Apply the `IMAP` pressure-solve left-hand side, in place — matrix-free:

    B φ = Gᵀ Bᴺ G φ

the counterpart of the `IBPM` modified-Poisson operator [`B_mul!`](@ref) with the
boundary-force block removed. Two consequences follow from that removal:

  - **It is grid-sized.** `IBPM`'s `B` acts on `λ = (φ, f_tilde)` and is
    `(#cells + N·n_b)` square; this one acts on the pressure alone. IMAP's
    body-sized work is instead the one-off Cholesky inside
    [`ManifoldProjection`](@ref).
  - **It is a discrete Poisson operator**, not a coupled system — but with the
    *projected* viscous inverse [`Ainv_IMAP!`](@ref) in the middle, which is where
    the immersed body enters.

Applied as [`G_mul!`](@ref) → [`Ainv_IMAP!`](@ref) → [`GT_mul!`](@ref); nothing is
assembled.

# Why the gradient is projected

The pressure gradient is projected like every other term of the momentum
equation. This matters: leaving `G φ` unprojected does not preserve no-slip,
because applying `Rᵀ` to the momentum equation then leaves

    Rᵀu^{n+1} = Rᵀuⁿ - Δt RᵀG p^{n+1}

and at an impulsive start `p ~ O(1/Δt)`, so that drift is `O(1)` *independent of
Δt* — measured at `‖Rᵀu‖ ≈ 0.9` after a single step from an exactly-projected
initial condition, i.e. refining `Δt` cannot fix it. Projecting kills the term
outright (`RᵀP = 0`), giving `Rᵀu^{n+1} = Rᵀuⁿ` to roundoff.

It costs nothing in structure: `Bᴺ` and `P` **commute** (every term of
`Σ(aPLP)ᵏ` both starts and ends with `P`), so

    Gᵀ Bᴺ P G = (P G)ᵀ Bᴺ (P G),

a congruence of the SPD `Bᴺ` — still symmetric positive semi-definite, still CG.

# Returns
The updated field `φ_out`.
"""
function B_mul!(φ_out, φ, proj, work, ::IMAP; h, a, dt, n_taylor, symmetric=true)
    G_mul!(work.q, φ; h)                                        # q = G φ
    P_mul!(work.q, proj)                                        # q = P G φ
    Ainv_IMAP!(work.y, work.q, work.term, work.tmp, proj; a, dt, n_taylor, h, symmetric)
    GT_mul!(φ_out, work.y; h)                                   # out = Gᵀ Bᴺ P G φ
    φ_out
end
