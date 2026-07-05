"""
Multidomain multigrid operators for the multi-level Poisson solver.

Implements the restriction (coarsening) and prolongation (interpolation)
operators that move vorticity and streamfunction data between grid levels,
as well as the boundary condition correction and the top-level Poisson driver.

Key functions:
  - `multidomain_coarsen!`     — restrict ω from fine → coarse level.
  - `multidomain_interpolate!` — prolongate ψ boundary values from coarse → fine.
  - `set_boundary!`            — apply boundary buffer into the main field array.
  - `add_laplacian_bc!`        — correct the Laplacian RHS for inter-level BCs.
  - `multidomain_poisson!`     — full multi-level Poisson solve driver.
"""

"""
    _coarse_indices(n::NTuple{N}, loc::Edge{Dual}) where {N}

Return the index ranges of the coarse-grid interior within the fine-grid array
for a dual-edge field component `loc`. Used by `multidomain_coarsen!`.
"""
function _coarse_indices(n::NTuple{N}, loc::Edge{Dual}) where {N}
    ntuple(N) do i
        n4 = n[i] .÷ 4
        i == loc.i ? (n4:(3n4-1)) : ((n4+1):(3n4-1))
    end
end

"""
    multidomain_coarsen!(ω², ω¹; n)

Restrict the fine-grid field `ω¹` into the coarser field `ω²` using a
weighted 3×3 averaging stencil. Operates in-place on `ω²`.

# Arguments
- `ω²`: Output field on the coarser grid (modified in-place).
- `ω¹`: Input field on the finer grid.
- `n`: Grid resolution tuple.
"""
function multidomain_coarsen!(ω², ω¹; n)
    backend = get_backend(ω²[3])
    for i in eachindex(ω²)
        R = CartesianIndices(_coarse_indices(Tuple(n), Loc_ω(i)))
        @loop backend (I in R) ω²[i][I] = multidomain_coarsen(i, ω¹[i], I; n)
    end
    ω²
end

"""
    multidomain_coarsen(i, ωᵢ, I²; n)

Compute the coarse-grid value at index `I²` for component `i` by applying the
3×3 weighted stencil to the fine-grid data `ωᵢ`. Called by `multidomain_coarsen!`.
"""
function multidomain_coarsen(i, ωᵢ, I²; n)
    T = eltype(ωᵢ)
    stencil = _coarsen_stencil(T)
    s = zero(T)
    indices = _fine_indices(i, Tuple(n), Tuple(I²))
    for I¹ in indices
        s += sum_map(*, SMatrix{3,3}(@view ωᵢ[I¹]), stencil)
    end
    s / length(indices)
end

"""
    _coarsen_stencil(T)

Return the normalized 3×3 bilinear restriction stencil:

    [1 2 1; 2 4 2; 1 2 1] / 16

Entries sum to 1, preserving the integral of the coarsened field.
"""
function _coarsen_stencil(T)
    (@SMatrix [
        1 2 1
        2 4 2
        1 2 1
    ]) / T(16)
end

"""
    _fine_indices(i, n, I)

Map a coarse-grid index `I` to the corresponding fine-grid region(s).

- 2D: returns one `CartesianIndices` covering the 3-point stencil.
- 3D: returns two `CartesianIndices` planes (one per fine-grid layer along
  dimension `i`) that map to the coarse index.
"""
_fine_indices(_, n::NTuple{2}, I::NTuple{2}) = (CartesianIndices(_fine_range.(n, I)),)

function _fine_indices(i, n::NTuple{3}, I::NTuple{3})
    plane1 = 2(I[i] - (n[i] ÷ 4))
    r = _fine_range.(n, I)
    ntuple(2) do plane
        j = plane1 + plane - 1
        CartesianIndices(setindex(r, j:j, i))
    end
end

"""
    _fine_range(n, I)

Return the 3-point fine-grid range that corresponds to coarse-grid index `I`
along one dimension (assuming a 4:1 refinement ratio).
"""
function _fine_range(n::Int, I::Int)
    2(I - (n ÷ 4)) .+ (-1:1)
end

"""
    multidomain_interpolate!(ωb, ω; n)

Prolongate boundary values from a coarse-level field `ω` into the boundary
buffer `ωb` of the next finer level, using linear (2D) or bilinear (3D)
interpolation.

# Arguments
- `ωb`: Output boundary buffer (modified in-place).
- `ω`: Input coarse-grid field (e.g., streamfunction ψ).
- `n`: Grid resolution tuple.
"""
function multidomain_interpolate!(ωb, ω; n)
    backend = get_backend(ω[3])
    for i in eachindex(ω), (j, k) in axes_permutations(i), dir in 1:2
        b = ωb[i][dir, k]
        @loop backend (I in CartesianIndices(b)) begin
            b[I] = multidomain_interpolate(ω[i], (i, j, k), dir, I; n)
        end
    end
    ωb
end

"""
    multidomain_interpolate(ωᵢ, (i, j, k), dir, I¹; n)

Interpolation kernel used by `multidomain_interpolate!`.

- **2D** (`CartesianIndex{2}`): linear interpolation along the non-component direction.
- **3D** (`CartesianIndex{3}`): bilinear interpolation on the face plane,
  accounting for the staggered offset of component `i`.
"""
# TODO: `dir` is unused in both 2D and 3D methods — present for interface consistency with the
# caller in multidomain_interpolate!. Decide whether to keep it or remove it from the signature.
function multidomain_interpolate(ωᵢ, (i, j, k), dir, I¹::CartesianIndex{2}; n)
    δ = axisunit(Val(2))
    I² = CartesianIndex(ntuple(dim -> n[dim] ÷ 4 + fld(I¹[dim], 2), 2))
    if iseven(I¹[j])
        ωᵢ[I²]
    else
        (ωᵢ[I²] + ωᵢ[I²+δ(j)]) / 2
    end
end

function multidomain_interpolate(ωᵢ, (i, j, k), dir, I¹::CartesianIndex{3}; n)
    δ = axisunit(Val(3))
    n4 = Tuple(n) .÷ 4
    I² = CartesianIndex(
        ntuple(3) do dim
            if dim == i
                n4[dim] + fld(I¹[dim] - 1, 2)
            else
                n4[dim] + fld(I¹[dim], 2)
            end
        end,
    )
    a = (1 + 2mod(I¹[i] + 1, 2)) / 4
    if iseven(I¹[j])
        (1 - a) * ωᵢ[I²] + a * ωᵢ[I²+δ(i)]
    else
        ((1 - a) * (ωᵢ[I²] + ωᵢ[I²+δ(j)]) + a * (ωᵢ[I²+δ(i)] + ωᵢ[I²+δ(i)+δ(j)])) / 2
    end
end

"""
    set_boundary!(ω, ωb)

Copy boundary values from buffer `ωb` into the main field `ω`. Typically
called after `multidomain_interpolate!` to enforce inter-level boundary
conditions on the vorticity or streamfunction field.
"""
function set_boundary!(ω, ωb)
    backend = get_backend(ω[3])
    for i in eachindex(ω), b in ωb[i]
        if length(b) > 0
            @loop backend (I in CartesianIndices(b)) begin
                ω[i][I] = b[I]
            end
        end
    end
    ω
end

"""
    add_laplacian_bc!(Lψ, factor, ψb)

Add inter-level boundary condition corrections to the Laplacian RHS `Lψ`.

When solving on a non-coarsest level the streamfunction from the coarser
level provides Dirichlet-like boundary values. This function folds those
values into `Lψ` so the interior solve sees the correct right-hand side.

# Arguments
- `Lψ`: Vector of interior arrays for the Laplacian RHS (modified in-place).
- `factor`: Scaling factor (typically `1 / h²`).
- `ψb`: Boundary buffer populated by `multidomain_interpolate!`.
"""
function add_laplacian_bc!(Lψ, factor, ψb)
    backend = get_backend(Lψ[3])

    for i in eachindex(Lψ), j in 1:ndims(Lψ[i]), dir in 1:2
        ax = UnitRange.(axes(Lψ[i]))
        if i == j
            let Iᵢ = (ax[i][begin], ax[i][end])[dir],
                R = CartesianIndices(setindex(ax, Iᵢ:Iᵢ, i)),
                # StaticArrays doesn't adapt data for GPU, so use a tuple of tuples.
                ψb = map(_nd_tuple, ψb)

                @loop backend (I in R) begin
                    Lψ[i][I] += factor * laplacian_bc_ii(ψb, i, dir, I)
                end
            end
        else
            let b = ψb[i][dir, j],
                rb = axes(b, j),
                Iⱼ = (rb[begin], rb[end])[dir],
                R = CartesianIndices(setindex(ax, Iⱼ:Iⱼ, j))

                @loop backend (I in R) begin
                    δ = axisunit(I)
                    Lψ[i][I-outward(dir)*δ(j)] += factor * b[I]
                end
            end
        end
    end
end

"""
    laplacian_bc_ii(ψb, i, dir, I)

Compute the diagonal Laplacian boundary correction for the component `i` face
in direction `dir` at interior index `I`. Helper for `add_laplacian_bc!`.
"""
function laplacian_bc_ii(ψb, i, dir, I)
    δ = axisunit(I)
    T = eltype(ψb[3][1][1])
    Iₒ = I + (dir - 1) * δ(i)
    s = zero(T)
    for (j, _) in axes_permutations(i)
        b = ψb[j][i][dir]
        s += b[Iₒ] - b[Iₒ-δ(j)]
    end
    -outward(dir) * s
end

"""
    multidomain_poisson!(ω, ψ, u, ψb, grid, fft_plan)

Multi-level Poisson solver driver (∇²ψ = -ω across all grid levels).

# Algorithm
1. Restrict `ω` from fine to coarse through all levels.
2. For each level from coarsest to finest:
   a. Coarsest level: direct spectral solve with zero boundary conditions.
   b. Finer levels: prolongate `ψ` from the coarser level into `ψb`,
      correct the RHS with `add_laplacian_bc!`, then solve spectrally.
3. Set boundary values of `ψ` and recover `u = ∇ × ψ` via `curl!`.

# Arguments
- `ω`: vorticity fields, one per grid level.
- `ψ`: streamfunction fields, one per grid level (updated in-place).
- `u`: velocity fields, one per grid level (updated in-place).
- `ψb`: boundary buffer (reused across levels).
- `grid`: `Grid` object with level count and resolution.
- `fft_plan`: precomputed `EigenbasisTransform` plans.
"""
function multidomain_poisson!(ω, ψ, u, ψb, grid::Grid, fft_plan)
    Base.require_one_based_indexing(ψ)

    for level in 2:grid.levels
        multidomain_coarsen!(ω[level], ω[level-1]; n=grid.n)
    end

    for level in grid.levels:-1:1
        h = gridstep(grid, level)
        ψi = ψ[min(lastindex(ψ), level)]
        ψe = _exclude_boundary(ψi, grid, Loc_ω)

        if level == grid.levels
            for i in eachindex(ψe)
                _set!(ψe[i], ω[level][i])
                foreach(b -> fill!(b, 0), ψb[i])
            end
        else
            let ψci = ψ[min(lastindex(ψ), level + 1)],
                ψce = _exclude_boundary(ψci, grid, Loc_ω)

                multidomain_interpolate!(ψb, ψce; n=grid.n)
            end
            for i in eachindex(ψe)
                _set!(ψe[i], ω[level][i])
            end
            add_laplacian_bc!(ψe, 1 / h^2, ψb)
        end

        EigenbasisTransform(λ -> -1 / (λ / h^2), fft_plan)(ψe, ψe)

        set_boundary!(ψi, ψb)

        if level in eachindex(u)
            curl!(u[level], ψi; h)
        end
    end
end
