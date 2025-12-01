"""
This module provides numerical operators and transforms for the immersed boundary projection method (IBPM).

This file defines the main computational operators used in FastIBPM, including
nonlinear advection terms, FFT-based Laplacian solvers, multilevel grid
coarsening and interpolation, and boundary condition utilities. It also includes
delta-function kernels and regularization structures used for coupling between
the fluid grid and immersed bodies.
"""







"""
    nonlinear!(nonlin, u, ω)

Compute the nonlinear advection term in-place.

This function updates the nonlinear term array `nonlin` by evaluating the
nonlinear contribution at every grid point, based on the velocity field `u`
and the vorticity field `ω`. It represents the convective term
the convective term (u · ∇)u = u × ω for incompressible flow.

The computation is parallelized over the grid using the appropriate backend
(e.g. CPU or GPU).

# Arguments
- `nonlin`: array (or array of arrays) storing the nonlinear term; modified in place.
- `u`: velocity field.
- `ω`: vorticity field.

# Returns
The updated `nonlin` field.
"""
function nonlinear!(nonlin, u, ω)
    backend = get_backend(nonlin[1])
    for (i, nonlinᵢ) in pairs(nonlin)
        @loop backend (I in CartesianIndices(nonlinᵢ)) begin
            nonlinᵢ[I] = nonlinear(i, u, ω, I)
        end
    end
    nonlin
end



"""
    nonlinear(i, u, ω, I)

Compute the nonlinear advection term for component `i` at grid point `I`.

This function evaluates the local contribution of the nonlinear term
the i-th component of the cross product u × ω computed using bilinear interpolation of the velocity and vorticity fields.
velocity and vorticity fields. It is called internally by [`nonlinear!`](@ref).

# Arguments
- `i`: index of the velocity component being computed.
- `u`: velocity field.
- `ω`: vorticity field.
- `I`: Cartesian grid index.

# Returns
The scalar nonlinear term at the specified component and grid location.
"""
function nonlinear(i, u, ω, I)
    δ = axisunit(I)
    sumcross(i, vec_kind(u), vec_kind(ω)) do j, k
        uI = (u[j][I] + u[j][I-δ(i)] + u[j][I+δ(j)] + u[j][I-δ(i)+δ(j)]) / 4
        ωI = (ω[k][I] + ω[k][I+δ(j)]) / 2
        uI * ωI
    end
end



"""
    rot!(ω, u; h)

Compute the vorticity field `ω` from the velocity field `u` in-place.

# Arguments
- `ω`: Array of arrays where the computed vorticity components will be stored (mutated in-place).
- `u`: Array of arrays representing the velocity field.
- `h`: Grid spacing used for finite-difference approximation of the curl (keyword argument).

# Description
This function updates `ω` directly by looping over each component and grid index, calling `rot(i, u, I; h)` for each point.  
The computation is parallelized using the available backend (CPU or GPU) for efficiency.

# Returns
- `ω`: The updated vorticity field (same array as input, modified in-place).

"""
function rot!(ω, u; h)
    backend = get_backend(ω[3])
    for (i, ωᵢ) in pairs(ω)
        @loop backend (I in CartesianIndices(ωᵢ)) begin
            ωᵢ[I] = rot(i, u, I; h)
        end
    end
    ω
end



"""
    rot(i, u, I; h)

Compute the i-th component of the vorticity (curl) at a single grid point `I` from a velocity field `u`.

# Arguments
- `i`: Index of the vorticity component to compute (e.g., 1 for x, 2 for y).
- `u`: Array of arrays representing the velocity field.
- `I`: Cartesian index of the grid point where the curl is computed.
- `h`: Grid spacing used for finite-difference approximation.

# Returns
- Scalar value representing the i-th component of the vorticity at point `I`.

# Notes
- This function computes the curl at a single point. To compute the full vorticity field over the grid, use `rot!(ω, u; h)`, which applies `rot` in-place across all grid points.
- The function uses finite differences and cross-product indexing (`sumcross`) to compute the curl:
(∇ × u)_i = sum_{(j,k)} (u_k[I] - u_k[I-δ(j)]) / h.
"""
function rot(i, u, I; h)
    δ = axisunit(I)
    sumcross(i) do j, k
        (u[k][I] - u[k][I-δ(j)]) / h
    end
end



"""
    curl!(u, ψ; h)

Compute the velocity field `u` as the curl of a potential field `ψ` over the entire grid.

# Arguments
- `u`: Array of arrays representing the velocity field (updated in-place).
- `ψ`: Array of arrays representing the potential field.
- `h`: Grid spacing used for finite-difference approximation.

# Returns
- The updated velocity field `u`.

# Notes
- This function applies the `curl` computation at every grid point in-place.
- It uses finite differences and the helper function `curl(i, ψ, I; h)` to compute each component.
- The `backend` is automatically chosen for parallel or GPU execution.
"""
function curl!(u, ψ; h)
    backend = get_backend(u[1])
    for (i, uᵢ) in pairs(u)
        @loop backend (I in CartesianIndices(uᵢ)) begin
            uᵢ[I] = curl(i, ψ, I; h)
        end
    end
    u
end



"""
    curl(i, ψ, I; h)

Compute the `i`-th component of a velocity field as the curl of a scalar potential `ψ` at a specific grid point.

# Arguments
- `i`: Index of the velocity component to compute (e.g., 1 for x, 2 for y).
- `ψ`: Array of arrays representing the scalar potential field.
- `I`: Cartesian index of the grid point where the curl is evaluated.
- `h`: Grid spacing used for finite-difference approximation.

# Returns
- A scalar value representing the `i`-th component of the curl at grid point `I`.

# Notes
- The result is divergence-free by construction.
- Uses centered finite differences and `sumcross` to handle component indices.
"""
function curl(i, ψ, I; h)
    δ = axisunit(I)
    sumcross(i, Vec(), vec_kind(ψ)) do j, k
        (ψ[k][I+δ(j)] - ψ[k][I]) / h
    end
end



"""
    struct LaplacianPlan

Holds the data required to compute the Laplacian efficiently in spectral space using FFTs.

# Fields
- `λ`: eigenvalues of the Laplacian for spectral multiplication.
- `work`: temporary workspace array.
- `fwd`: forward FFT plan.
- `inv`: inverse FFT plan.
- `n_logical`: logical size of the FFT domain.

# Constructor

    LaplacianPlan(ω_i, i, n)

Creates a `LaplacianPlan` for a given field component `ω_i` on a grid of size `n` (e.g., `SVector(nx, ny)`).

# Arguments
- `ω_i`: array representing one component of the field (e.g., vorticity).
- `i`: component index (1, 2, or 3 for x, y, z).
- `n`: grid resolution as `SVector{N}`.

# Returns
- A `LaplacianPlan` struct containing all data needed to:
  1. Transform the field to spectral space.
  2. Multiply by Laplacian eigenvalues.
  3. Transform back to physical space.

# Notes
- The constructor sets up FFT plans (`fwd` and `inv`) and eigenvalues (`λ`) automatically.
- The Laplacian is then applied as `Δu ≈ λ .* fwd(u)`.
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
    fwd = FFT_R2R.plan_r2r!(ωᵢ, kind; flags)
    inv = FFT_R2R.plan_r2r!(ωᵢ, map(k -> FFTW.inv_kind[k], kind); flags)
    n_logical = prod(map(FFTW.logical_size, nω, kind))

    LaplacianPlan(λ, similar(ωᵢ), fwd, inv, n_logical)
end



"""
    laplacian_fft_kind(i, nd)

Return a tuple specifying the FFT type to use along each dimension of a multidimensional array for the Laplacian operator.

# Arguments
- `i`: The dimension index that should use a cosine transform (`FFTW.REDFT01`).
- `nd`: Total number of dimensions of the array.

# Returns
- A tuple of length `nd` with `FFTW.REDFT01` for dimension `i` and `FFTW.RODFT00` for all other dimensions.

# Notes
- `REDFT01` (DCT-I) is typically used for Neumann or periodic boundary conditions.
- `RODFT00` (DST-I) is typically used for Dirichlet (zero-value) boundary conditions.	
"""
laplacian_fft_kind(i, nd) = ntuple(j -> i == j ? FFTW.REDFT01 : FFTW.RODFT00, nd)



"""
    laplacian_eigvals!(λ, i)

Compute the eigenvalues of the discrete Laplacian on a multidimensional grid and store them in λ.

# Arguments
- λ: Array to hold the eigenvalues (modified in-place)
- i: The dimension index along which the cosine transform (DCT) is applied; other dimensions use sine transforms (DST)

# Returns
- The array λ filled with Laplacian eigenvalues

# Notes
- The computation runs in parallel on the backend associated with λ
- Eigenvalues correspond to the discrete Laplacian under DCT (dimension i) and DST (other dimensions)
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

Create a Laplacian plan for each component of a vector field ω.

# Arguments
- ω: Vector field whose components require Laplacian plans
- n: Grid resolution (e.g., tuple of (nx, ny) or (nx, ny, nz))

# Returns
- A tuple of LaplacianPlan objects, one for each component of ω

# Notes
- Each LaplacianPlan contains the eigenvalues and FFT plans needed for spectral Poisson solvers
- Useful when different components require different FFT types due to mixed boundary conditions
"""
laplacian_plans(ω, n) = map(i -> LaplacianPlan(ω[i], i, n), tupleindices(ω))



"""
    EigenbasisTransform

Represents a spectral transform that applies a function `f` (e.g., inverse Laplacian) in the eigenbasis of the Laplacian.
It uses precomputed Laplacian plans to efficiently apply forward and inverse FFTs, multiply by `f(λ)` in spectral space, and transform back.

# Fields
- `f`: Function or scalar applied to the eigenvalues.
- `plan`: Tuple of LaplacianPlan objects wrapped in an OffsetTuple for staggered grid alignment.

# Constructor
- `EigenbasisTransform(f, plans::Tuple)` wraps a regular tuple of `LaplacianPlan` objects into an `OffsetTuple` automatically. 
  This allows you to pass a standard tuple without needing to manually construct an `OffsetTuple`.
  
# Call Overloads
- `(X::EigenbasisTransform)(y, ω)` applies the transform to all components of `ω`, storing the result in `y`.
- `(X::EigenbasisTransform)(yᵢ, ωᵢ, i)` applies the transform to the `i`-th component using its corresponding Laplacian plan.  

# Notes
- Useful for spectral Poisson solvers and other operations in the Laplacian eigenbasis.
- The OffsetTuple ensures proper alignment for staggered grid variables.
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



"""
    _coarse_indices(n::NTuple{N}, loc::Edge{Dual}) where {N}

Compute the index ranges for a coarser grid that is a subset of the original grid.

# Description
This function determines which indices to retain when coarsening a field defined on edges of the dual grid.  
It is used in multigrid methods to identify the portion of the grid that remains after reducing resolution.

# Arguments
- `n`: Tuple giving the number of grid points in each dimension.
- `loc`: Specifies the edge location in the dual grid.

# Returns
A tuple of index ranges corresponding to the coarse grid coordinates.
"""
function _coarse_indices(n::NTuple{N}, loc::Edge{Dual}) where {N}
    ntuple(N) do i
        n4 = n[i] .÷ 4
        i == loc.i ? (n4:3n4-1) : (n4+1:3n4-1)
    end
end



"""
    multidomain_coarsen!(ω², ω¹; n)

Coarsen a fine-grid field ω¹ into a coarser field ω² for use in multigrid solvers.

# Description
This function reduces the resolution of each component of a vector field by averaging fine-grid values into the coarser grid using a restriction stencil.  
It runs in parallel across the computational backend and relies on `_coarse_indices` and `multidomain_coarsen` for index mapping and averaging.

# Arguments
- `ω²`: Output field on the coarser grid.
- `ω¹`: Input field on the finer grid.
- `n`: Tuple giving the grid resolution.

# Returns
The coarsened field `ω²`.
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

Compute the coarse-grid value of a field component from its fine-grid representation.

# Description
For component `i`, this function calculates the value at coarse-grid index `I²` by averaging corresponding fine-grid points using a predefined stencil.  
It is used internally by `multidomain_coarsen!` to define the restriction operation.

# Arguments
- `i`: Component index.
- `ωᵢ`: Fine-grid array for the given component.
- `I²`: Cartesian index on the coarse grid.
- `n`: Tuple giving the grid resolution.

# Returns
The coarse-grid value at index `I²`.
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

Return a normalized 3×3 smoothing stencil used for grid coarsening.

# Description
This function defines a weighted averaging kernel for use in multigrid coarsening or smoothing operations.  
It produces a static 3×3 matrix from `StaticArrays.jl` with higher weight at the center and symmetric lower weights around it.  
The entries sum to 1, ensuring mass conservation during coarsening.

# Arguments
- `T`: Element type of the output matrix.

# Returns
A 3×3 static matrix of type `T` used as a normalized coarsening stencil.
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

Return the fine-grid indices corresponding to a coarse-grid index `I`.

# Description
This function identifies which fine-grid cells map to a given coarse-grid location, used for coarsening or restriction in multigrid solvers.  
It supports both 2D and 3D cases:
- In 2D, it returns a single `CartesianIndices` range.
- In 3D, it returns two `CartesianIndices` planes associated with the selected component `i`.

# Arguments
- `i`: Component index (used only in 3D).
- `n`: Tuple representing the grid size.
- `I`: Coarse-grid index tuple.

# Returns
A tuple of one or more `CartesianIndices` objects representing fine-grid regions corresponding to the coarse-grid index `I`.
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

Return the range of fine-grid indices corresponding to a coarse-grid index `I` along one dimension.

# Description
Maps a 1D coarse-grid index to the corresponding three-point region on the fine grid, assuming a 4:1 refinement ratio.  
The range is centered around `2 * (I - n/4)` and includes the neighboring points at offsets `-1`, `0`, and `+1`.

# Arguments
- `n`: Grid size along the considered dimension.
- `I`: Coarse-grid index.

# Returns
A `UnitRange` of three fine-grid indices corresponding to the coarse-grid index `I`.
"""
function _fine_range(n::Int, I::Int)
    2(I - (n ÷ 4)) .+ (-1:1)
end



"""
    multidomain_interpolate!(ωb, ω; n)

Interpolate a fine-grid field `ω` onto boundary or ghost regions, storing results in `ωb`.

# Description
This function computes interpolated boundary values from a fine-grid field `ω` and writes them into the corresponding boundary arrays `ωb`.  
It supports both 2D and 3D cases and calls the helper function [`multidomain_interpolate`] for the actual interpolation logic.

# Arguments
- `ωb`: Output array for interpolated boundary values.
- `ω`: Input fine-grid field (e.g., vorticity).
- `n`: Grid size tuple, used for mapping fine-to-coarse indices.

# Returns
The updated `ωb` containing interpolated boundary values.
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

Interpolate values from a fine grid onto a coarser grid or boundary face.

# Description
Defines the interpolation rule used by [`multidomain_interpolate!`].  
Two methods are available:
- **2D version** (`CartesianIndex{2}`): performs simple linear interpolation along one direction.
- **3D version** (`CartesianIndex{3}`): performs bilinear interpolation on a 2D plane, adjusting for component offsets.

# Arguments
- `ωᵢ`: Component `i` of the fine-grid field.
- `(i, j, k)`: Index tuple defining component and orientation.
- `dir`: Direction index for boundary interpolation.
- `I¹`: Fine-grid index at which interpolation is performed.
- `n`: Grid size tuple.

# Returns
Interpolated scalar value for the target coarse-grid or boundary location.
"""
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

Copy boundary values from a boundary buffer `ωb` into the main field `ω`.

# Description
This function updates the field `ω` by setting its boundary regions to the corresponding values stored in `ωb`.  
It is typically used after interpolation or restriction steps in multigrid or multidomain solvers to enforce consistent boundary conditions.

The operation is performed in parallel using backend-agnostic loops.

# Arguments
- `ω`: Main field to be updated (e.g., vorticity or velocity component arrays).
- `ωb`: Boundary buffer containing precomputed boundary values for each component of `ω`.

# Returns
The modified field `ω` with updated boundary values.
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

Apply boundary condition corrections to the Laplacian of a vector field.

# Description
This function modifies the Laplacian `Lψ` of a field by adding contributions from boundary values in `ψb`.  
It ensures that Dirichlet or Neumann boundary conditions are correctly enforced in the discrete Laplacian.

# Arguments
- `Lψ`: Vector of arrays representing the Laplacian of the field ψ.
- `factor`: Scalar multiplier for the boundary corrections.
- `ψb`: Boundary buffer holding Dirichlet or Neumann values at the domain boundaries.

# Returns
The modified `Lψ` with boundary corrections applied (in-place).
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

Compute the diagonal Laplacian boundary correction for component `i`.

# Description
This function calculates the correction term for the Laplacian at a boundary along direction `dir` for the `i`-th component.  
It uses differences of precomputed boundary values in `ψb` to enforce proper boundary conditions in the discrete Laplacian operator.

# Arguments
- `ψb`: Boundary buffer arrays for each component and direction.
- `i`: Index of the component being corrected.
- `dir`: Direction of the boundary (e.g., 1 = x, 2 = y, 3 = z).
- `I`: Cartesian index in the boundary array where the correction is computed.

# Returns
A scalar value representing the diagonal Laplacian boundary correction at the given index.
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

Solve Poisson equations across multiple grid levels using a multigrid-like approach.

# Description
This function computes the solution of Poisson problems on a hierarchical set of grids.  
It performs the following steps for each level:
  1. Coarsen the source term `ω` from finer to coarser grids.
  2. Solve from coarse to fine.
  3. Optionally compute the velocity field `u` at selected levels using `curl!`.

# Arguments
- `ω`: Vector of vector fields representing source terms at each grid level.
- `ψ`: Vector of potential fields to store the solution at each level.
- `u`: Vector of velocity fields to update via curl of `ψ`.
- `ψb`: Boundary buffers for the potential fields.
- `grid`: Grid object containing the domain size and level information.
- `fft_plan`: Precomputed `EigenbasisTransform` plans for spectral solves.

# Returns
The function updates `ψ` and `u` in-place with the Poisson solution and velocity field.
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
    r = ntuple(_ -> length(-s:s), N)
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
  2. Iterates over all stencil points within the delta kernel’s support.
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
  weighted sum of nearby grid values using the delta kernel’s support.
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
            I = CartesianIndices(map(i -> i .+ (-s:s), Tuple(Ib)))
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
- For each marker, the force components are distributed over the delta kernel’s
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
