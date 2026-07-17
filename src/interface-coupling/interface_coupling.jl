# Regularization machinery for the Eulerian–Lagrangian interface coupling:
#   - Discrete delta functions (AbstractDeltaFunc, DeltaYang3S, DeltaYang3S2).
#   - The Reg operator and its weight computation (update_weights!).
#   - The interpolation (E) and spreading/regularization (Eᵀ) operators.
#
# This half depends only on the grid (Grid, Loc_u, coord) and is included BEFORE
# init/problems.jl, because the CNAB struct declares a type parameter R<:Reg.
# The CNAB-dependent redistribution helpers live in force_redistribution.jl, which
# is included after CNAB is defined. See that file's header for the full rationale.

"""
    AbstractDeltaFunc

Abstract type for delta-function-like objects. Subtypes define specific delta kernels.

# Usage
A delta function can be called on a vector `r` to evaluate the multidimensional delta:
```julia
delta(r)  # evaluates as the product of 1D delta values along each component
```
"""
abstract type AbstractDeltaFunc end

(delta::AbstractDeltaFunc)(r::AbstractVector) = prod(delta, r)

"""
    DeltaYang3S <: AbstractDeltaFunc

Smooth delta function approximation with compact support [-2, 2], following Yang et al. (2009).
- `support(::DeltaYang3S) = 2` gives its support radius.
- Calling `delta(r::Real)` evaluates the function at a real point `r` using a piecewise formula:
  - |r| < 1 → first formula
  - 1 ≤ |r| < 2 → second formula
  - |r| ≥ 2 → returns 0
The function is smooth and satisfies partition-of-unity and moment conditions.
"""
struct DeltaYang3S <: AbstractDeltaFunc end
support(::DeltaYang3S) = 2

function (::DeltaYang3S)(r::AbstractFloat)
    u = one(r)
    a = abs(r)
    if a < 1
        17u / 48 + sqrt(3u) * π / 108 + a / 4 - r^2 / 4 +
        (1 - 2 * a) / 16 * sqrt(-12 * r^2 + 12 * a + 1) -
        sqrt(3u) / 12 * asin(sqrt(3u) / 2 * (2 * a - 1))
    elseif a < 2
        55u / 48 - sqrt(3u) * π / 108 - 13 * a / 12 +
        r^2 / 4 +
        (2 * a - 3) / 48 * sqrt(-12 * r^2 + 36 * a - 23) +
        sqrt(3u) / 36 * asin(sqrt(3u) / 2 * (2 * a - 3))
    else
        zero(r)
    end
end

"""
    DeltaYang3S2 <: AbstractDeltaFunc

Smoother and wider delta function than DeltaYang3S, with compact support [-3, 3].
- `support(::DeltaYang3S2) = 3` gives its support radius.
- Calling `delta(x::Real)` evaluates the function at `x` using piecewise formulas:
  - r ≤ 1
  - 1 < r ≤ 2
  - 2 < r ≤ 3
  - r > 3 → 0
Each segment uses polynomials, square roots, and arcsine terms to ensure smoothness and correct moment conditions.
"""
struct DeltaYang3S2 <: AbstractDeltaFunc end
support(::DeltaYang3S2) = 3

function (::DeltaYang3S2)(x::Float64)
    r = abs(x)
    r2 = r * r
    r3 = r2 * r
    r4 = r3 * r

    if r <= 1.0
        a5 = asin((1.0 / 2.0) * sqrt(3.0) * (2.0 * r - 1.0))
        a8 = sqrt(1.0 - 12.0 * r2 + 12.0 * r)

        4.166666667e-2 * r4 +
        (-0.1388888889 + 3.472222222e-2 * a8) * r3 +
        (-7.121664902e-2 - 5.208333333e-2 * a8 + 0.2405626122 * a5) * r2 +
        (-0.2405626122 * a5 - 0.3792313933 + 0.1012731481 * a8) * r +
        8.0187537413e-2 * a5 - 4.195601852e-2 * a8 + 0.6485698427

    elseif r <= 2.0
        a6 = asin((1.0 / 2.0) * sqrt(3.0) * (-3.0 + 2.0 * r))
        a9 = sqrt(-23.0 + 36.0 * r - 12.0 * r2)

        -6.250000000e-2 * r4 +
        (0.4861111111 - 1.736111111e-2 * a9) .* r3 +
        (-1.143175026 + 7.812500000e-2 * a9 - 0.1202813061 * a6) * r2 +
        (0.8751991178 + 0.3608439183 * a6 - 0.1548032407 * a9) * r - 0.2806563809 * a6 +
        8.22848104e-3 +
        0.1150173611 * a9

    elseif r <= 3.0
        a1 = asin((1.0 / 2.0 * (2.0 * r - 5.0)) * sqrt(3.0))
        a7 = sqrt(-71.0 - 12.0 * r2 + 60.0 * r)

        2.083333333e-2 * r4 +
        (3.472222222e-3 * a7 - 0.2638888889) * r3 +
        (1.214391675 - 2.604166667e-2 * a7 + 2.405626122e-2 * a1) * r2 +
        (-0.1202813061 * a1 - 2.449273192 + 7.262731481e-2 * a7) * r +
        0.1523563211 * a1 +
        1.843201677 - 7.306134259e-2 * a7
    else
        0.0
    end
end

"""
    Reg{D,T,N,A,M,W}
    Reg(backend, T, delta, nb, Val{N})

Represents a regularization operator used for interpolation and spreading
based on a discrete delta function.

# Fields
- `delta` — the regularized delta function (a subtype of `AbstractDeltaFunc`).
- `I` — a matrix of index offsets defining the discrete stencil.
- `weights` — preallocated delta weights for each stencil point.

The struct is adapted for GPU execution via `Adapt.@adapt_structure`, allowing
`Reg` objects to be transferred automatically between CPU and GPU memory.

# Constructor
`Reg(backend, T, delta, nb, Val{N})` creates a regularization operator in `N`
dimensions. It allocates:
- the stencil index matrix `I`, and
- the multidimensional `weights` array whose size is determined by the support
  of the delta function.

`backend` controls where arrays are allocated (CPU or GPU), and `nb` is the
number of bodies or markers for which weights are stored.

This type is typically used in immersed-boundary methods for evaluating and
applying discrete delta functions.
"""
struct Reg{
    D<:AbstractDeltaFunc,T,N,A<:AbstractArray{SVector{N,Int},2},M,W<:AbstractArray{T,M}
}
    delta::D
    I::A
    weights::W
end

Adapt.@adapt_structure Reg

function Reg(backend, T, delta, nb, ::Val{N}) where {N}
    I = KernelAbstractions.zeros(backend, SVector{N,Int}, nb, N)

    s = support(delta)
    r = ntuple(_ -> length((-s):s), N)
    weights = KernelAbstractions.zeros(backend, T, r..., nb, N)

    Reg(delta, I, weights)
end

"""
    update_weights!(reg, grid, xbs, ibs)

Update interpolation/spreading weights for immersed boundary markers.

This function computes the stencil indices and delta-function weights used to
transfer data between Lagrangian marker positions (`xbs`) and the Eulerian grid
(`grid`). Only markers listed in `ibs` are updated. The result is stored
in-place inside the `Reg` object `reg`.

# Arguments
- `reg::Reg`: Regularization structure containing delta kernel, stencil offsets,
  and a weight array to be filled.
- `grid::Grid{N}`: Eulerian grid used for mapping marker positions to grid
  coordinates.
- `xbs`: Array of marker positions (typically `SVector{N,Float}`).
- `ibs`: Indices of the markers to update.

# Notes
- If `ibs` is empty, the function returns `reg` unchanged.
- For each marker and each velocity/force component, the function:
  1. Computes the integer grid offset `I` nearest to the marker.
  2. Iterates over all stencil points within the delta kernel's support.
  3. Evaluates the delta function at normalized offsets `(xb - xu) / h`.
  4. Stores the resulting weights in `reg.weights`.

# Returns
Returns the updated `reg`.
"""
function update_weights!(reg::Reg, grid::Grid{N}, xbs, ibs) where {N}
    isempty(ibs) && return reg

    backend = get_backend(reg.weights)
    for i in 1:N
        @loop backend (J in CartesianIndices(ibs)) begin
            ib = ibs[J[1]]
            xb = xbs[ib]

            xu0 = coord(grid, Loc_u(i), zeros(SVector{N,Int}))
            reg.I[ib, i] = I = @. round(Int, (xb - xu0) / grid.h)

            for k in CartesianIndices(axes(reg.weights)[1:N])
                ΔI = (-support(reg.delta) - 1) .+ SVector(Tuple(k))
                xu = coord(grid, Loc_u(i), I + ΔI)
                reg.weights[k, ib, i] = reg.delta((xb - xu) / grid.h)
            end
        end
    end
    reg
end

"""
    interpolate_body!(ub, reg, u)

Interpolate the Eulerian velocity field `u` onto the Lagrangian marker velocities
`ub` using precomputed regularization stencils stored in `reg`.

This function gathers velocity values from the Eulerian grid for each marker and
each velocity component, applies the corresponding delta–function weights, and
stores the resulting interpolated velocities in-place in `ub`.

# Arguments
- `ub`: Output array of marker velocities (e.g., `Vector{SVector{N,T}}`).
- `reg::Reg`: Regularization structure containing interpolation indices `I` and
  delta weights `weights`.
- `u`: Eulerian velocity field, given as an array of `N` grid arrays
  (`u[1], u[2], …`).

# Notes
- For each marker, the function loops over velocity components and computes a
  weighted sum of nearby grid values using the delta kernel's support.
- Uses the precomputed stencil offsets `reg.I` and weight tensors
  `reg.weights`, which must be updated before calling this function.
- Updates `ub` in-place and also returns it.

# Returns
Updates `ub` in-place.
"""
function interpolate_body!(ub, reg::Reg{<:Any,T,N}, u) where {T,N}
    s = support(reg.delta)
    backend = get_backend(ub)
    @loop backend (J in CartesianIndices(ub)) begin
        ib = J[1]
        ubJ = zero(MVector{N,T})
        for i in 1:N
            w = @view reg.weights[.., ib, i]
            Ib = reg.I[ib, i]
            I = CartesianIndices(map(i -> i .+ ((-s):s), Tuple(Ib)))
            uᵢ = @view u[i][I]
            ubJ[i] = sum_map(*, w, uᵢ)
        end
        ub[J] = ubJ
    end
end

"""
    regularize!(fu, reg, fb)

Spread Lagrangian forces `fb` onto the Eulerian force field `fu` using the
regularization stencils stored in `reg`.

This function distributes each marker force to nearby Eulerian grid points using
the delta–function weights in `reg.weights` and the corresponding index offsets
in `reg.I`. The resulting Eulerian force field is written in-place in `fu`.

# Arguments
- `fu`: Output Eulerian force field, given as an array of `N` grids
  (`fu[1], fu[2], …`). All entries are reset to zero before accumulation.
- `reg::Reg`: Regularization structure containing interpolation/spreading
  indices `I` and delta weights `weights`.
- `fb`: Lagrangian forces, typically stored as `Vector{SVector{N}}`, one
  force vector per marker.

# Notes
- For each marker, the force components are distributed over the delta kernel's
  support region.
- This operation is the adjoint (transpose) of `interpolate_body!` in the
  immersed boundary method.
- Updates `fu` in-place and also returns it.

# Returns
Updates `fu` in-place.
"""
function regularize!(fu, reg::Reg{<:Any,<:Any,N}, fb) where {N}
    R = CartesianIndices(axes(reg.weights)[1:N])
    backend = get_backend(fu[1])

    for fuᵢ in fu
        @loop backend (I in CartesianIndices(fuᵢ)) fuᵢ[I] = 0
    end

    for ib in eachindex(fb)
        @loop backend (K in R) begin
            for i in 1:N
                I0 = CartesianIndex(Tuple(reg.I[ib, i] .- (support(reg.delta) + 1)))
                I = I0 + K
                fu[i][I] += fb[ib][i] * reg.weights[K, ib, i]
            end
        end
    end

    fu
end
