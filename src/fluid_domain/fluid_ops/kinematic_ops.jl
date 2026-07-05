"""
Kinematic operators acting on the staggered Eulerian grid.

Provides the discrete curl and cross-product operators needed for the
streamfunction-vorticity formulation:
  - `nonlinear` / `nonlinear!` — advection term  u × ω
  - `rot`       / `rot!`       — vorticity from velocity  ω = ∇ × u
  - `curl`      / `curl!`      — velocity from streamfunction  u = ∇ × ψ
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
