"""
Preconditioners for the iterative (CG) solves in the IBPM coupling step.

Isolated here so preconditioning strategies can be added without touching the
solver. A preconditioner only changes how fast CG converges, never the solution,
so it stays fully consistent with the Taira & Colonius (2007) method.

Currently:
  - [`NoPreconditioner`](@ref) — identity (default; the uniform-grid IBPM and all
    FastIBPM paths, whose CG/BiCGStab calls are left untouched).
  - [`JacobiPreconditioner`](@ref) — the diagonal of the modified-Poisson operator
    `B = QᵀBᴺQ`, for the **stretched-grid** IBPM CG. On a stretched grid the mass
    matrix `M` spans several orders of magnitude, which spreads the eigenvalues on
    top of the Poisson conditioning; the Jacobi diagonal cancels that penalty.

To add a strategy: define a new `AbstractPreconditioner` subtype, a builder that
returns it, and a [`preconditioner_Pl`](@ref) method giving the object to hand to
`cg!`'s `Pl` keyword.
"""

"""
    AbstractPreconditioner

Supertype for CG preconditioners used in the IBPM coupling solve. See
[`NoPreconditioner`](@ref) and [`JacobiPreconditioner`](@ref).
"""
abstract type AbstractPreconditioner end

"""
    NoPreconditioner()

Identity preconditioner — CG runs unpreconditioned. The default, so any path that
does not opt in behaves exactly as before.
"""
struct NoPreconditioner <: AbstractPreconditioner end

"""
    JacobiPreconditioner(D::Diagonal)

Diagonal (Jacobi) preconditioner holding `D = diag(B)` of the modified-Poisson
operator. Applied by CG as `ldiv!(D, r) = r ./ diag`. Build it with
[`jacobi_preconditioner`](@ref).
"""
struct JacobiPreconditioner{T} <: AbstractPreconditioner
    D::Diagonal{T,Vector{T}}
end

"""
    preconditioner_Pl(p::AbstractPreconditioner)

The object handed to `cg!(...; Pl=...)`: `I` (identity) for [`NoPreconditioner`](@ref),
the stored `Diagonal` for [`JacobiPreconditioner`](@ref).
"""
preconditioner_Pl(::NoPreconditioner) = I
preconditioner_Pl(p::JacobiPreconditioner) = p.D

"""
    _build_precond(spec, sol0)

Resolve the `precond` option passed to `CNAB` into an [`AbstractPreconditioner`] for
the IBPM coupling. `spec` may be:

  - `:auto` (default) — Jacobi on a `StretchedGrid`, identity otherwise.
  - `:none`   — force [`NoPreconditioner`](@ref) (e.g. to benchmark against Jacobi).
  - `:jacobi` — force [`JacobiPreconditioner`](@ref) (also allowed on a uniform grid).
  - an `AbstractPreconditioner` instance — used as-is (extension hook for new strategies).
"""
_build_precond(spec::AbstractPreconditioner, sol0) = spec
function _build_precond(spec::Symbol, sol0)
    if spec === :auto
        sol0.prob.grid isa StretchedGrid ? jacobi_preconditioner(sol0) : NoPreconditioner()
    elseif spec === :none
        NoPreconditioner()
    elseif spec === :jacobi
        jacobi_preconditioner(sol0)
    else
        throw(ArgumentError(
            "unknown preconditioner spec :$spec (use :auto, :none, :jacobi, or an " *
            "AbstractPreconditioner instance)",
        ))
    end
end

"""
    jacobi_preconditioner(sol; pin=1)

Build the diagonal (Jacobi) preconditioner for the primitive-variable modified
Poisson operator `B = QᵀBᴺQ = [GᵀBᴺG  GᵀBᴺEᵀ; EBᴺG  EBᴺEᵀ]` on the current
(static) geometry, ordered to match the CG unknown `[vec(φ); reinterpret(f̃)]`.

`sol` is a `CNAB` (untyped here only because this file is included before `CNAB` is
defined). Uses the leading `Bᴺ ≈ Δt·M⁻¹` term — cheap (`O(cells)`) and the term
that carries the mass-matrix conditioning we are cancelling:

  - pressure cell `I`:  `d_φ[I] = Δt · Σᵢ wᵢ(I)² (1/Mᵢ(I) + 1/Mᵢ(I+δᵢ))`
    (`wᵢ = _gweight`, `Mᵢ = mass`; uniform-core check `wᵢ=1/h, M=1 → 2N·Δt/h²`),
  - body DOF `(k,i)`:  `d_f[k,i] = Δt · Σ_stencil E² / M_g ≈ Δt · Σ_stencil E²`
    (`M_g ≈ 1` since the body lives in the uniform core).

The pinned pressure DOF gets diagonal `1` (it is an identity row in the solved
operator). Built once at setup and reused every step.
"""
function jacobi_preconditioner(sol; pin=1)
    grid = sol.prob.grid
    dt = sol.dt
    reg = sol.reg
    N = length(eltype(sol.f_tilde))
    T = eltype(eltype(sol.f_tilde))

    # --- pressure block, ordered like vec(no_offset_view(φ)) ---
    φtmp = grid_zeros(get_backend(sol.state.φ), grid, Loc_p())
    for I in CartesianIndices(φtmp)
        δ = axisunit(I)
        s = zero(T)
        for i in 1:N
            w = _gweight(grid, i, I)
            s += w^2 * (1 / mass(grid, i, I) + 1 / mass(grid, i, I + δ(i)))
        end
        φtmp[I] = dt * s
    end
    dφ = collect(vec(no_offset_view(φtmp)))

    # --- force block, ordered like reinterpret(T, f̃) = [f₁ₓ,f₁ᵧ,f₂ₓ,f₂ᵧ,…] ---
    n_ib = length(sol.f_tilde)
    stencil = CartesianIndices(axes(reg.weights)[1:N])
    df = Vector{T}(undef, N * n_ib)
    for ib in 1:n_ib, i in 1:N
        s = zero(T)
        for k in stencil
            s += reg.weights[k, ib, i]^2
        end
        df[(ib-1)*N+i] = dt * s
    end

    d = vcat(dφ, df)
    d[pin] = one(T)                       # pinned pressure DOF: identity row
    @. d = ifelse(iszero(d), one(T), d)   # guard against any zero diagonal
    JacobiPreconditioner(Diagonal(d))
end
