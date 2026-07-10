"""
Kinematic operators acting on the staggered Eulerian grid.

Provides the discrete curl and cross-product operators needed for the
streamfunction-vorticity formulation:
  - `nonlinear` / `nonlinear!` — advection term  u × ω
  - `rot`       / `rot!`       — vorticity from velocity  ω = ∇ × u
  - `curl`      / `curl!`      — velocity from streamfunction  u = ∇ × ψ

and the operators needed for the primitive-variable (velocity–pressure)
formulation:
  - `divergence` / `divergence!` — mass flux out of each cell  d = ∇ · u
  - `gradient`   / `gradient!`   — pressure gradient on edges   g = ∇ p
  - `laplacian`  / `laplacian!`  — viscous (vector) Laplacian    L u = ∇² u

The discrete gradient is the negative transpose of the divergence, `G = -Dᵀ`,
which (with uniform spacing) holds exactly since both operators carry `1/h`.

The Laplacian `L` is the *forward* finite-difference stencil, block-diagonal
across velocity components (`L = diag(Lᵘ, Lᵛ, …)`). Unlike the
streamfunction-vorticity solver in `laplacian_solver.jl`, which inverts the
Laplacian spectrally via FFTs, this operator is applied directly — the
primitive-variable projection uses it inside the Taylor approximation of `A⁻¹`
and in the CG solve of the modified Poisson equation, both of which need only
forward operator products.
"""

"""
    nonlinear!(nonlin, u, ω)

Compute the nonlinear advection term in-place.

This function updates the nonlinear term array `nonlin` by evaluating the
nonlinear contribution at every grid point, based on the velocity field `u`
and the vorticity field `ω`. It represents the convective term
(u · ∇)u = u × ω for incompressible flow.

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

This function evaluates the local contribution of the nonlinear term —
the i-th component of the cross product u × ω computed using bilinear
interpolation of the velocity and vorticity fields. It is called internally
by [`nonlinear!`](@ref).

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

Compute the i-th component of the vorticity (curl) at a single grid point `I`
from a velocity field `u`.

# Arguments
- `i`: Index of the vorticity component to compute (e.g., 1 for x, 2 for y).
- `u`: Array of arrays representing the velocity field.
- `I`: Cartesian index of the grid point where the curl is computed.
- `h`: Grid spacing used for finite-difference approximation.

# Returns
Scalar value representing the i-th component of the vorticity at point `I`.
Uses finite differences: (∇ × u)_i = Σ_{(j,k)} (u_k[I] - u_k[I-δ(j)]) / h.
"""
function rot(i, u, I; h)
    δ = axisunit(I)
    sumcross(i) do j, k
        (u[k][I] - u[k][I-δ(j)]) / h
    end
end

"""
    curl!(u, ψ; h)

Compute the velocity field `u` as the curl of a potential field `ψ` over the
entire grid, in-place.

# Arguments
- `u`: Array of arrays representing the velocity field (updated in-place).
- `ψ`: Array of arrays representing the potential field.
- `h`: Grid spacing used for finite-difference approximation.

# Returns
The updated velocity field `u`.
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

Compute the `i`-th component of a velocity field as the curl of a scalar
potential `ψ` at a specific grid point.

# Arguments
- `i`: Index of the velocity component to compute (e.g., 1 for x, 2 for y).
- `ψ`: Array of arrays representing the scalar potential field.
- `I`: Cartesian index of the grid point where the curl is evaluated.
- `h`: Grid spacing used for finite-difference approximation.

# Returns
Scalar value of the `i`-th curl component at grid point `I`.
The result is divergence-free by construction.
"""
function curl(i, ψ, I; h)
    δ = axisunit(I)
    sumcross(i, Vec(), vec_kind(ψ)) do j, k
        (ψ[k][I+δ(j)] - ψ[k][I]) / h
    end
end

"""
    divergence!(d, u; h)

Compute the discrete divergence `d = ∇ · u` of the velocity field `u` in-place.

`u` is stored on `Primal` edges (`Loc_u`), i.e. `u[j]` is the flux perpendicular
to the cell face ⟂ to axis `j`. The result `d` is a single cell-centered scalar
field (`Node{Dual}`, where the pressure lives). The same expression applies in
2D and 3D, since the divergence carries no cross-product structure.

The computation is parallelized over the grid using the appropriate backend
(e.g. CPU or GPU).

# Arguments
- `d`: cell-centered scalar array holding the divergence; modified in place.
- `u`: velocity field (tuple of edge-centered component arrays).
- `h`: grid spacing used for the finite-difference approximation (keyword argument).

# Returns
The updated field `d`.
"""
function divergence!(d, u; h)
    backend = get_backend(d)
    @loop backend (I in CartesianIndices(d)) begin
        d[I] = divergence(u, I; h)
    end
    d
end

"""
    divergence(u, I; h)

Compute the discrete divergence `∇ · u` at a single cell-centered grid point `I`.

For a cell centered at index `I`, component `u[j]` on the low face lives at index
`I` and on the high face at `I + δ(j)`, so the divergence is the sum of forward
differences over all axes:

    (∇ · u)[I] = Σⱼ (u[j][I+δ(j)] - u[j][I]) / h.

# Arguments
- `u`: velocity field (tuple of edge-centered component arrays).
- `I`: Cartesian index of the cell where the divergence is evaluated.
- `h`: grid spacing used for the finite-difference approximation.

# Returns
Scalar value of the divergence at cell `I`.
"""
function divergence(u, I; h)
    δ = axisunit(I)
    sum(tupleindices(u)) do j
        (u[j][I+δ(j)] - u[j][I]) / h
    end
end

"""
    gradient!(g, p; h)

Compute the discrete gradient `g = ∇ p` of the cell-centered scalar field `p`
in-place.

`p` is a single cell-centered array (`Node{Dual}`, `Loc_p`, where the pressure
lives). The result `g` is an edge-centered vector field (`Loc_u`), i.e. `g[i]`
lives on the same primal edges as the velocity component `u[i]`. This operator is
the negative transpose of the divergence, `G = -Dᵀ`, so it applies unchanged in
2D and 3D.

The computation is parallelized over the grid using the appropriate backend
(e.g. CPU or GPU).

# Arguments
- `g`: edge-centered vector field holding the gradient; modified in place.
- `p`: cell-centered scalar field.
- `h`: grid spacing used for the finite-difference approximation (keyword argument).

# Returns
The updated field `g`.
"""
function gradient!(g, p; h)
    backend = get_backend(g[1])
    for (i, gᵢ) in pairs(g)
        @loop backend (I in CartesianIndices(gᵢ)) begin
            gᵢ[I] = gradient(i, p, I; h)
        end
    end
    g
end

"""
    gradient(i, p, I; h)

Compute the `i`-th component of the gradient `∇ p` at a single edge-centered grid
point `I`.

The edge `g[i][I]` sits between cell centers `I - δ(i)` (low) and `I` (high), so
the gradient component is the backward difference

    (∇ p)_i[I] = (p[I] - p[I-δ(i)]) / h.

# Arguments
- `i`: index of the gradient component to compute (e.g., 1 for x, 2 for y).
- `p`: cell-centered scalar field.
- `I`: Cartesian index of the edge where the gradient is evaluated.
- `h`: grid spacing used for the finite-difference approximation.

# Returns
Scalar value of the `i`-th gradient component at edge `I`.
"""
function gradient(i, p, I; h)
    δ = axisunit(I)
    (p[I] - p[I-δ(i)]) / h
end

"""
    laplacian!(lap, u; h)

Compute the discrete (vector) Laplacian `lap = ∇² u` in-place.

The Laplacian is block-diagonal across components (`L = diag(Lᵘ, Lᵛ, …)`), so
each velocity component `u[i]` is transformed independently by the scalar
Laplacian [`laplacian`](@ref). Both `u` and `lap` are edge-centered vector
fields (`Loc_u`). This is the forward stencil used to build the implicit viscous
operator `A = (1/Δt)I - (1/2Re)L`; it is *not* inverted here.

The computation is parallelized over the grid using the appropriate backend
(e.g. CPU or GPU).

# Arguments
- `lap`: edge-centered vector field holding the Laplacian; modified in place.
- `u`: edge-centered velocity field.
- `h`: grid spacing used for the finite-difference approximation (keyword argument).

# Returns
The updated field `lap`.
"""
function laplacian!(lap, u; h)
    backend = get_backend(lap[1])
    for (i, lapᵢ) in pairs(lap)
        @loop backend (I in CartesianIndices(lapᵢ)) begin
            lapᵢ[I] = laplacian(u[i], I; h)
        end
    end
    lap
end

"""
    laplacian(a, I; h)

Compute the scalar Laplacian `∇² a` of a single field `a` at grid point `I`,
using the standard second-difference stencil summed over every axis:

    (∇² a)[I] = Σⱼ (a[I+δ(j)] - 2 a[I] + a[I-δ(j)]) / h².

Because the vector Laplacian is block-diagonal, this same scalar operator is
applied to each velocity component (and works equally for any cell- or
edge-centered scalar field). It reads the two neighbors along every axis, so it
is defined only on interior points; boundary contributions are supplied by the
caller's index range / boundary data.

# Arguments
- `a`: a single scalar field array.
- `I`: Cartesian index where the Laplacian is evaluated.
- `h`: grid spacing used for the finite-difference approximation.

# Returns
Scalar value of `∇² a` at `I`.
"""
function laplacian(a, I::CartesianIndex{N}; h) where {N}
    δ = axisunit(I)
    aI = a[I]
    s = sum(ntuple(Val(N)) do j
        a[I+δ(j)] - 2 * aI + a[I-δ(j)]
    end)
    s / h^2
end

"""
    set_velocity_boundary!(ub, grid, f)

Fill the velocity boundary buffer `ub` with the prescribed boundary velocity `f`,
in place. `ub` is the buffer returned by `boundary_zeros(backend, grid, Loc_u)`:
`ub[i]` holds component `u[i]` on the two faces perpendicular to axis `i`, i.e.
the **normal-velocity** faces that lie exactly on the domain boundary ∂D. Only
those faces are populated (the other, degenerate, entries are left empty).

`f(x)` returns the prescribed velocity vector at physical point `x`; component
`i` is stored on `u[i]`'s boundary faces. This buffer is the complete boundary
data for the divergence BC (`bc2`) and the normal part of the Laplacian BC
(`bc1`); the tangential boundary values needed by `bc1` in the transverse
directions are supplied separately.

The fill is parallelized over the backend of each boundary face.

# Arguments
- `ub`: velocity boundary buffer (modified in-place).
- `grid`: `Grid` supplying face coordinates via `coord`.
- `f`: prescribed velocity `f(x) -> SVector`, evaluated at each face coordinate.

# Returns
The updated buffer `ub`.
"""
function set_velocity_boundary!(ub, grid, f)
    for i in tupleindices(ub)
        loc = Loc_u(i)
        faces = ub[i]
        for idx in CartesianIndices(faces)
            face = faces[idx]
            isempty(face) && continue
            backend = get_backend(face)
            @loop backend (I in CartesianIndices(face)) begin
                face[I] = f(coord(grid, loc, I))[i]
            end
        end
    end
    ub
end

"""
    add_laplacian_bc!(rhs, Loc_u, factor, f, grid)

Primitive-variable (`Loc_u`, primal-edge) implementation of the Laplacian
boundary fold — the `bc1` term. Dispatches from the master
[`add_laplacian_bc!`](@ref); see there for the shared contract.

Uses the **far-field / ghost-node** boundary treatment appropriate for external
flow: the prescribed velocity `f` is imposed at the neighbor node just outside
the interior-unknown region, and the interior Laplacian stencil stays uniform.
For every boundary-adjacent interior node the boundary neighbor's contribution
is folded into `rhs` (explicit-RHS, additive):

    rhs[i][I] += factor · f(x_neighbor)[i]

summed over each boundary direction. Normal (`j==i`, neighbor on ∂D) and
transverse (`j≠i`, ghost a half-cell outside) contributions are handled
uniformly, since `coord` returns the correct physical position for either.

# Arguments
- `rhs`: interior velocity RHS arrays (`ExcludeBoundary` layout; modified in-place).
- `factor`: scaling factor (e.g. `1/h²`, or the viscous `a/h²`).
- `f`: prescribed boundary velocity `f(x) -> SVector`.
- `grid`: `Grid` supplying neighbor coordinates via `coord`.

# Returns
The updated `rhs`.
"""
add_laplacian_bc!(rhs, ::Type{<:Edge{Primal}}, factor, f, grid) =
    _add_laplacian_bc_primitive!(rhs, factor, grid, f)

function _add_laplacian_bc_primitive!(rhs, factor, grid, f)
    for i in tupleindices(rhs)
        loc = Loc_u(i)
        a = rhs[i]
        backend = get_backend(a)
        ax = UnitRange.(axes(a))
        for j in 1:ndims(a), dir in 1:2
            Iⱼ = (ax[j][begin], ax[j][end])[dir]
            R = CartesianIndices(setindex(ax, Iⱼ:Iⱼ, j))
            s = outward(dir)   # -1 at the low end, +1 at the high end
            @loop backend (I in R) begin
                δ = axisunit(I)
                a[I] += factor * f(coord(grid, loc, I + s * δ(j)))[i]
            end
        end
    end
    rhs
end
