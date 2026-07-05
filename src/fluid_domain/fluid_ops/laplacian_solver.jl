"""
Spectral Laplacian solver for the streamfunction-vorticity formulation.

Uses FFT-based real-to-real transforms (via `fft_r2r`) to solve the Poisson
equation ∇²ψ = -ω spectrally on each grid level.

Key types and functions:
  - `LaplacianPlan`       — precomputed FFT plans + eigenvalues for one field component.
  - `laplacian_plans`     — builds a plan for every component of a vector field.
  - `EigenbasisTransform` — applies f(λ) in the Laplacian eigenbasis (e.g. inverse Laplacian).
"""

"""
    struct LaplacianPlan

Holds the data required to compute the Laplacian efficiently in spectral space
using FFTs.

# Fields
- `λ`: eigenvalues of the Laplacian for spectral multiplication.
- `work`: temporary workspace array.
- `fwd`: forward FFT plan.
- `inv`: inverse FFT plan.
- `n_logical`: logical size of the FFT domain.

# Constructor

    LaplacianPlan(ω_i, i, n)

Creates a `LaplacianPlan` for a given field component `ω_i` on a grid of size
`n` (e.g., `SVector(nx, ny)`).

# Arguments
- `ω_i`: array representing one component of the field (e.g., vorticity).
- `i`: component index (1, 2, or 3 for x, y, z).
- `n`: grid resolution as `SVector{N}`.
"""
struct LaplacianPlan{P1,P2,L<:AbstractArray,W<:AbstractArray}
    λ::L
    work::W
    fwd::P1
    inv::P2
    n_logical::Int
end

function LaplacianPlan(ωᵢ, i, n::SVector{N}) where {N}
    R = cell_axes(n, Loc_ω(i), ExcludeBoundary())
    nω = length.(R)
    λ = OffsetArray(similar(ωᵢ, nω), R)
    laplacian_eigvals!(λ, i)

    kind = laplacian_fft_kind(i, N)
    flags = FFTW.EXHAUSTIVE
    fwd = fft_r2r.plan_r2r!(ωᵢ, kind; flags)
    inv = fft_r2r.plan_r2r!(ωᵢ, map(k -> FFTW.inv_kind[k], kind); flags)
    n_logical = prod(map(FFTW.logical_size, nω, kind))

    LaplacianPlan(λ, similar(ωᵢ), fwd, inv, n_logical)
end

"""
    laplacian_fft_kind(i, nd)

Return the tuple of FFT kinds to use along each dimension for the Laplacian.

- Dimension `i` uses a cosine transform (`FFTW.REDFT01`).
- All other dimensions use a sine transform (`FFTW.RODFT00`).
"""
laplacian_fft_kind(i, nd) = ntuple(j -> i == j ? FFTW.REDFT01 : FFTW.RODFT00, nd)

"""
    laplacian_eigvals!(λ, i)

Compute the eigenvalues of the discrete Laplacian in-place.

`i` selects which dimension uses the cosine transform (DCT); the rest use the
sine transform (DST). The computation runs in parallel on the backend of `λ`.
"""
function laplacian_eigvals!(λ, i)
    backend = get_backend(λ)
    nd = ndims(λ)
    R = CartesianIndices(λ)
    n = size(λ)
    @loop backend (I in R) begin
        I₁ = Tuple(I - first(R)) .+ 1
        s = zero(eltype(λ))
        for j in 1:nd
            s += if (i == j)
                -4 * sin(π * (I₁[j] - 1) / (2n[j]))^2
            else
                -4 * sin(π * I₁[j] / (2(n[j] + 1)))^2
            end
        end
        λ[I] = s
    end
    λ
end

"""
    laplacian_plans(ω, n)

Build a `LaplacianPlan` for each component of a vector field `ω` on a grid of
size `n`. Returns a tuple of plans, one per component.
"""
laplacian_plans(ω, n) = map(i -> LaplacianPlan(ω[i], i, n), tupleindices(ω))

"""
    EigenbasisTransform

Applies a spectral function `f(λ)` in the Laplacian eigenbasis.

Wraps a set of `LaplacianPlan`s and, for each field component, executes:
  1. Inverse FFT (to eigenbasis).
  2. Pointwise multiplication by `f(λ) / n_logical`.
  3. Forward FFT (back to physical space).

# Fields
- `f`: scalar function of the eigenvalue (e.g. `λ -> -1 / λ` for the inverse Laplacian).
- `plan`: `OffsetTuple` of `LaplacianPlan`s, one per field component.

# Constructors
    EigenbasisTransform(f, plans::Tuple)   # wraps into OffsetTuple automatically

# Call signatures
    (X::EigenbasisTransform)(y, ω)         # applies to all components
    (X::EigenbasisTransform)(yᵢ, ωᵢ, i)   # applies to component i
"""
struct EigenbasisTransform{F,O,P<:Tuple{Vararg{LaplacianPlan}}}
    f::F
    plan::OffsetTuple{O,P}
end

EigenbasisTransform(f, plan::Tuple) = EigenbasisTransform(f, OffsetTuple(plan))

function (X::EigenbasisTransform)(y, ω)
    for i in eachindex(ω)
        X(y[i], ω[i], i)
    end
    y
end

function (X::EigenbasisTransform)(yᵢ, ωᵢ, i)
    plan = X.plan[i]
    _set!(plan.work, ωᵢ)
    let λ = no_offset_view(plan.λ), a = no_offset_view(plan.work)
        mul!(a, plan.inv, a)
        @. a *= X.f(λ) / plan.n_logical
        mul!(a, plan.fwd, a)
    end
    _set!(yᵢ, plan.work)
end
