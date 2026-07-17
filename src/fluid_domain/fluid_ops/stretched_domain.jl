"""
Stretched Cartesian grid for the primitive-variable (`IBPM`) path.

A [`StretchedGrid`](@ref) keeps a **uniform core** of spacing `Δx_min` around the
immersed body (where the regularized delta function requires uniform cells) and
lets the spacing grow **geometrically** outward, so the far-field boundaries can be
placed many diameters away at a fraction of the cell count a uniform mesh would
need. This is the mesh strategy of Taira & Colonius (2007, §5, Table 1) — the
uniform-`Grid` far field would otherwise cost ~10–100× more cells for the same
extent.

The grid stores, per axis, the 1D **face positions** and **cell widths** (the
per-axis 1D spacing arrays chosen over a fully general per-cell metric — simpler
and faster for a tensor-product stretch; AMR would swap this backing later). All
geometry/metric queries the operators make — [`coord`](@ref), [`gridstep`](@ref),
[`cell_width`](@ref), [`center_distance`](@ref) — are answered from these arrays,
so the operators stay metric-agnostic and the uniform path is unaffected.

Scope: `IBPM` only. The FastIBPM FFT/multidomain solver assumes uniform spacing, so
pairing a `StretchedGrid` with `FastIBPM` is rejected at construction of the state.

See also [`Grid`](@ref) (uniform) and the metric-accessor seam in `eulerian_grid.jl`.
"""

# One per-axis coordinate array, indexed by the (0-based) grid index with one
# ghost cell of padding on each end (`-1 : n`), so `coord` and the boundary folds
# can evaluate one cell outside the domain without going out of bounds.
const _StretchAxis{T} = OffsetArrays.OffsetVector{T,Vector{T}}

"""
    StretchedGrid{N,T} <: AbstractGrid{N,T}

Per-axis stretched Cartesian grid (uniform core + geometric growth) for the `IBPM`
formulation. Build it with the keyword constructor
[`StretchedGrid(; dx_min, core, growth, extent, max_dx)`](@ref).

# Fields
- `n::SVector{N,Int}`     : number of cells per axis (computed from the stretching).
- `x0::SVector{N,T}`      : low corner (the low face of cell 0; may extend slightly
                            past the requested `extent` since `n` is rounded up).
- `dx_min::T`             : the uniform-core spacing (`Δx_min`); this is what
                            [`gridstep`](@ref) returns and what `Reg`/the force
                            factor use (the body lives in the uniform core).
- `levels::Int`           : always `1` (single level; multi-level is FastIBPM-only).
- `xf::NTuple{N,_StretchAxis{T}}` : face positions, indexed `-1 : n[d]`.
- `dx::NTuple{N,_StretchAxis{T}}` : cell widths, indexed `-1 : n[d]` (with ghosts).
- `core::SMatrix{N,2,T}`  : the realized uniform-core bounds `[lo hi]` per axis.
"""
struct StretchedGrid{N,T} <: AbstractGrid{N,T}
    n::SVector{N,Int}
    x0::SVector{N,T}
    dx_min::T
    levels::Int
    xf::NTuple{N,_StretchAxis{T}}
    dx::NTuple{N,_StretchAxis{T}}
    core::SMatrix{N,2,T}
end

# Widths of a geometrically growing tail spanning at least `span`, starting from a
# first cell of `dx_min*growth` (i.e. ratio `growth` relative to the core cell) and
# multiplying by `growth` each step, clamped to `max_dx`. Empty if `span <= 0`.
function _geom_tail(dx_min::T, growth::T, span::T, max_dx::T) where {T}
    widths = T[]
    w = dx_min * growth
    total = zero(T)
    while total < span
        wu = min(w, max_dx)
        push!(widths, wu)
        total += wu
        w *= growth
    end
    widths
end

# Build one axis: returns (x0, cell_widths::Vector{T}, core_lo, core_hi_snapped).
function _stretch_axis(dx_min::T, growth::T, core_lo::T, core_hi::T,
                       ext_lo::T, ext_hi::T, max_dx::T) where {T}
    n_core = max(round(Int, (core_hi - core_lo) / dx_min), 1)
    core_hi_s = core_lo + n_core * dx_min
    right = _geom_tail(dx_min, growth, ext_hi - core_hi_s, max_dx)
    left = _geom_tail(dx_min, growth, core_lo - ext_lo, max_dx)
    widths = vcat(reverse(left), fill(dx_min, n_core), right)
    x0 = core_lo - sum(left; init=zero(T))
    (x0, widths, core_lo, core_hi_s)
end

# Turn a 1-based cell-width vector into ghost-padded face/width OffsetVectors
# indexed -1 : n (n = number of cells).
function _axis_arrays(x0::T, widths::Vector{T}) where {T}
    n = length(widths)
    xf = OffsetArray(Vector{T}(undef, n + 2), -1:n)   # faces at -1 : n
    xf[0] = x0
    for k in 1:n
        xf[k] = xf[k-1] + widths[k]
    end
    xf[-1] = xf[0] - widths[1]                          # ghost low face
    dx = OffsetArray(Vector{T}(undef, n + 2), -1:n)     # cell widths at -1 : n
    for k in -1:(n-1)
        dx[k] = xf[k+1] - xf[k]
    end
    dx[n] = widths[n]                                    # ghost high cell width
    (xf, dx)
end

"""
    StretchedGrid(; dx_min, core, growth, extent, max_dx=Inf)

Construct a [`StretchedGrid`](@ref) (uniform core + geometric growth), physics-driven:
you give the spacing and geometry, the cell count `n` is computed.

# Keywords
- `dx_min`  : uniform-core cell size `Δx_min` (must enclose the body — set `core`
              large enough to also cover the ~3-cell delta-function stencil margin).
- `core`    : `N×2` matrix `[lo hi]` per axis — the uniform-core bounds (same shape
              as the `gridlims` used with `Grid`, e.g. `SA[-1.0 2.0; -1.0 1.0]`).
- `growth`  : geometric ratio `r > 1` applied per cell outside the core (e.g. `1.03`).
- `extent`  : `N×2` matrix `[lo hi]` per axis — the *minimum* outer domain bounds;
              the realized domain reaches at least this far (rounds outward).
- `max_dx`  : optional cap on the stretched cell size (default `Inf`); beyond it the
              far field is uniform-coarse.

The body must sit inside `core`; that region stays exactly uniform at `dx_min`.
"""
function StretchedGrid(; dx_min, core, growth, extent, max_dx=Inf)
    corem = SMatrix{size(core, 1),2}(core)
    extm = SMatrix{size(extent, 1),2}(extent)
    N = size(corem, 1)
    T = promote_type(typeof(float(dx_min)), eltype(corem), eltype(extm), typeof(float(growth)))
    dxm = T(dx_min)
    g = T(growth)
    g > 1 || throw(ArgumentError("growth ratio must be > 1 (got $growth)"))
    mdx = T(max_dx)

    axes_data = ntuple(N) do d
        _stretch_axis(dxm, g, T(corem[d, 1]), T(corem[d, 2]), T(extm[d, 1]), T(extm[d, 2]), mdx)
    end
    x0 = SVector{N,T}(ntuple(d -> axes_data[d][1], N))
    n = SVector{N,Int}(ntuple(d -> length(axes_data[d][2]), N))
    core_out = SMatrix{N,2,T}(hcat(
        SVector{N,T}(ntuple(d -> axes_data[d][3], N)),
        SVector{N,T}(ntuple(d -> axes_data[d][4], N)),
    ))
    arrs = ntuple(d -> _axis_arrays(axes_data[d][1], axes_data[d][2]), N)
    xf = ntuple(d -> arrs[d][1], N)
    dx = ntuple(d -> arrs[d][2], N)

    StretchedGrid{N,T}(n, x0, dxm, 1, xf, dx, core_out)
end

# ---------------------------------------------------------------------------
# Geometry / metric accessors (dispatched on StretchedGrid)
# ---------------------------------------------------------------------------

gridcorner(grid::StretchedGrid) = grid.x0

# The characteristic (finest) spacing — the uniform-core Δx_min. Used by the
# regularized delta function (`Reg`) and the force factor, both of which act only
# in the uniform core.
gridstep(grid::StretchedGrid) = grid.dx_min

cell_width(grid::StretchedGrid, d, k) = grid.dx[d][k]

center_distance(grid::StretchedGrid, d, k) = (grid.dx[d][k-1] + grid.dx[d][k]) / 2

# Position of index `I` (with the location's fractional cell offset) on the
# stretched grid: within cell `I[d]`, offset `frac[d] ∈ {0, ½, 1}` of the cell
# width. frac 0 → low face, ½ → center, 1 → high face — matching the uniform
# `coord` convention, and exact in the uniform core.
function coord(grid::StretchedGrid{N,T}, loc, I::SVector{N,<:Integer}) where {N,T}
    frac = _cellcoord(loc, Val(N))
    SVector{N,T}(ntuple(N) do d
        k = I[d]
        grid.xf[d][k] + frac[d] * grid.dx[d][k]
    end)
end

# Actual (non-uniform) per-axis coordinate vectors for a block of indices — used
# for output/plotting. Overrides the uniform `range`-based method, which would
# wrongly linspace across the stretched region.
function coord(grid::StretchedGrid{N,T}, loc, r::Tuple{Vararg{AbstractRange}}) where {N,T}
    frac = _cellcoord(loc, Val(N))
    ntuple(N) do d
        [grid.xf[d][k] + frac[d] * grid.dx[d][k] for k in r[d]]
    end
end

# ===========================================================================
# Symmetric finite-volume operators (StretchedGrid, IBPM)
# ===========================================================================
#
# On a non-uniform grid the naive difference operators are *not* symmetric, which
# would break the CG solves. We instead use a mass-weighted finite-volume
# discretization (Taira & Colonius 2007, Appendix): face-area-weighted gradient
# and divergence so that `D = -Gᵀ` exactly, and a symmetric FV Laplacian `L_sym`
# with a diagonal mass matrix `M`. So `A = (1/Δt)M - (1/2Re)L_sym` and
# `QᵀBᴺQ` stay symmetric positive-definite and CG applies unchanged.
#
# Everything is normalized by `Δx_minᴺ`, so in the uniform core (`Δ = Δx_min`)
# these reduce *exactly* to the uniform operators (`M = I`, `L_sym = L_uniform`,
# `G_i = (p[I]-p[I-δi])/Δx_min`). Only the stretched far field carries `M ≠ I`.
# The uniform `Grid` keeps its own (scalar-`h`) operators; these methods are
# reached only for a `StretchedGrid`.

# Face-area weight of the velocity edge `u_i(I)`: ∏_{d≠i} Δ_d[I[d]] / Δx_minᴺ.
# This is `w_i(I)` such that `G_i = w_i·(p[I]-p[I-δi])` and `D = -Gᵀ`. Constant
# along axis `i`. Uniform: `1/Δx_min`.
@inline function _gweight(grid::StretchedGrid{N,T}, i, I) where {N,T}
    w = one(T)
    for d in 1:N
        d == i && continue
        w *= cell_width(grid, d, I[d])
    end
    w / grid.dx_min^N
end

"""
    mass(grid::StretchedGrid, i, I)

Diagonal mass-matrix entry `M` for the velocity component `i` at index `I`: the
control volume `V_i(I)` normalized by `Δx_minᴺ`. Equals
`center_distance_i(I[i]) · _gweight(i,I)`; it is `1` throughout the uniform core
and grows in the stretched far field. Used by `Ainv!`/`A` (`Bᴺ` carries `M⁻¹`).
"""
@inline mass(grid::StretchedGrid, i, I) =
    center_distance(grid, i, I[i]) * _gweight(grid, i, I)

"""
    rot(i, u, I, grid::StretchedGrid)
    rot!(ω, u, grid::StretchedGrid)

Stretched-grid vorticity `ω = ∇×u` (used for the rotational-form convective term).
Same stencil as the uniform [`rot`](@ref) but each difference is divided by the
local center-to-center distance instead of a scalar `h`. Reduces to the uniform
curl in the core; in the smooth far field the differences (hence `ω`) are ~0
regardless of spacing, so the near-body/wake vorticity is what this resolves.
"""
@inline function rot(i, u, I, grid::StretchedGrid)
    δ = axisunit(I)
    sumcross(i) do j, k
        (u[k][I] - u[k][I-δ(j)]) / center_distance(grid, j, I[j])
    end
end

function rot!(ω, u, grid::StretchedGrid)
    for (i, ωᵢ) in pairs(ω)
        backend = get_backend(ωᵢ)
        @loop backend (I in CartesianIndices(ωᵢ)) ωᵢ[I] = rot(i, u, I, grid)
    end
    ω
end

"""
    divergence(u, I, grid::StretchedGrid)

Weighted discrete divergence at cell `I`: `∑_i w_i(I)·(u_i[I+δi] - u_i[I])`, the
negative transpose of [`gradient`](@ref) (`D = -Gᵀ`). Reduces to the uniform
`∑_i (u_i[I+δi]-u_i[I])/Δx_min` in the core.
"""
@inline function divergence(u, I, grid::StretchedGrid)
    δ = axisunit(I)
    sum(tupleindices(u)) do i
        _gweight(grid, i, I) * (u[i][I+δ(i)] - u[i][I])
    end
end

"""
    gradient(i, p, I, grid::StretchedGrid)

Weighted discrete gradient component `i` at edge `I`: `w_i(I)·(p[I] - p[I-δi])`.
Transpose-consistent with [`divergence`](@ref). Uniform: `(p[I]-p[I-δi])/Δx_min`.
"""
@inline function gradient(i, p, I, grid::StretchedGrid)
    δ = axisunit(I)
    _gweight(grid, i, I) * (p[I] - p[I-δ(i)])
end

# Neighbor distance from edge `u_i(I)` to `u_i(I ± δk)` along axis `k`:
#  - k == i (face axis):   the cell width between the two faces.
#  - k ≠ i (center axis):  the center-to-center distance.
@inline _nbr_dist_hi(grid::StretchedGrid, i, k, I) =
    k == i ? cell_width(grid, i, I[i]) : center_distance(grid, k, I[k] + 1)
@inline _nbr_dist_lo(grid::StretchedGrid, i, k, I) =
    k == i ? cell_width(grid, i, I[i] - 1) : center_distance(grid, k, I[k])

# Area of the `u_i(I)` control-volume face perpendicular to axis `k`:
# ∏_{m≠k} width_m, where width_i = center_distance_i, width_{d≠i} = Δ_d[I[d]].
@inline function _face_area(grid::StretchedGrid{N,T}, i, k, I) where {N,T}
    A = one(T)
    for m in 1:N
        m == k && continue
        A *= (m == i ? center_distance(grid, i, I[i]) : cell_width(grid, m, I[m]))
    end
    A
end

"""
    laplacian(a, I, i, grid::StretchedGrid)

Symmetric finite-volume Laplacian of velocity component `i` (field `a = u[i]`) at
index `I`, normalized by `Δx_minᴺ`:

    (L_sym u)_I = (1/Δx_minᴺ) ∑ₖ [ A_k/d_k⁺ (u[I+δk]-u[I]) + A_k/d_k⁻ (u[I-δk]-u[I]) ]

with `A_k` the control-volume face area ⟂ axis `k` and `d_k±` the neighbor
distances. The shared face coefficient `A_k/d_k` makes the assembled operator
symmetric. Reduces exactly to the uniform 5-point `∑ₖ(u[I+δk]-2u[I]+u[I-δk])/Δx_min²`
in the core. Note the extra `i` argument (the uniform `laplacian` is
component-independent; the FV control volume is not).
"""
@inline function laplacian(a, I::CartesianIndex{N}, i, grid::StretchedGrid) where {N}
    δ = axisunit(I)
    inv_dxN = 1 / grid.dx_min^N
    aI = a[I]
    s = zero(eltype(a))
    for k in 1:N
        A = _face_area(grid, i, k, I)
        c_hi = A / _nbr_dist_hi(grid, i, k, I)
        c_lo = A / _nbr_dist_lo(grid, i, k, I)
        s += c_hi * (a[I+δ(k)] - aI) + c_lo * (a[I-δ(k)] - aI)
    end
    s * inv_dxN
end

# ---------------------------------------------------------------------------
# Stretched operator drivers and boundary folds (StretchedGrid only)
# ---------------------------------------------------------------------------
# These parallel the uniform kernels in `kinematic_ops.jl` and are used only by
# the stretched IBPM pipeline; the uniform `Grid` path keeps its own methods.

# --- divergence driver (Qᵀ) ---
function divergence!(d, u, grid::StretchedGrid)
    backend = get_backend(d)
    @loop backend (I in CartesianIndices(d)) d[I] = divergence(u, I, grid)
    d
end

# --- bc1: viscous (Laplacian) boundary fold ---
# `ν` is the kinematic viscosity; the L_sym boundary-face coefficient is applied
# internally. Both CN half-levels coincide for a steady free stream, giving the
# full `ν` weight (as in the uniform `add_laplacian_bc!` with factor `ν/h²`).
function viscous_bc!(rhs, ν, f, grid::StretchedGrid)
    for i in tupleindices(rhs)
        a = rhs[i]
        backend = get_backend(a)
        ax = UnitRange.(axes(a))
        for j in 1:ndims(a), dir in 1:2
            Iⱼ = (ax[j][begin], ax[j][end])[dir]
            R = CartesianIndices(setindex(ax, Iⱼ:Iⱼ, j))
            s = outward(dir)
            @loop backend (I in R) begin
                δ = axisunit(I)
                A = _face_area(grid, i, j, I)
                d = dir == 2 ? _nbr_dist_hi(grid, i, j, I) : _nbr_dist_lo(grid, i, j, I)
                c = A / d / grid.dx_min^ndims(a)               # L_sym boundary-face coefficient
                a[I] += ν * c * f(coord(grid, Loc_u(i), I + s * δ(j)))[i]
            end
        end
    end
    rhs
end

# --- bc2: divergence (continuity) boundary fold ---
function continuity_bc!(rhs, ub, grid::StretchedGrid)
    backend = get_backend(rhs)
    ax = UnitRange.(axes(rhs))
    for j in 1:ndims(rhs), dir in 1:2
        face = ub[j][dir, j]
        isempty(face) && continue
        Iⱼ = (ax[j][begin], ax[j][end])[dir]
        R = CartesianIndices(setindex(ax, Iⱼ:Iⱼ, j))
        s = outward(dir)
        @loop backend (I in R) begin
            δ = axisunit(I)
            rhs[I] += -_gweight(grid, j, I) * s * face[I+(dir-1)*δ(j)]
        end
    end
    rhs
end

# ---------------------------------------------------------------------------
# Regularization support: nearest-node lookup for a body point
# ---------------------------------------------------------------------------
# The uniform `Reg` finds the stencil base index by the linear map
# `round((xb - x₀)/h)`; that is wrong on a stretched grid (index 0 is out in the
# coarse tail), so `StretchedGrid` searches the per-axis node coordinates. The
# body lives in the uniform core, so the delta function itself (normalized by
# `gridstep = Δx_min`) is unchanged.

# Per-axis physical coordinate of `loc` at index `k` along axis `d`.
@inline function _coord_axis(grid::StretchedGrid{N}, loc, d, k) where {N}
    frac = _cellcoord(loc, Val(N))[d]
    grid.xf[d][k] + frac * grid.dx[d][k]
end

function _reg_base_index(grid::StretchedGrid{N}, loc, xb) where {N}
    SVector{N,Int}(ntuple(N) do d
        best_k = 0
        best = typemax(float(eltype(xb)))
        for k in 0:(grid.n[d] - 1)
            e = abs(_coord_axis(grid, loc, d, k) - xb[d])
            if e < best
                best = e
                best_k = k
            end
        end
        best_k
    end)
end
