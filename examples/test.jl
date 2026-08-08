# -*- coding: utf-8 -*-
# cylinder_mgn_dataset.jl
#
# Generate an Immersa cylinder-flow HDF5 dataset for a MeshGraphNet-style surrogate.
#
# Main idea:
#   - The training target is still omega^n -> omega^{n+1}.
#   - The HDF5 stores omega snapshots on Loc_ω(3).
#   - The HDF5 also stores the true Immersa-style delta stencils evaluated on
#     Loc_u(1) and Loc_u(2), because interpolation/regularization in the solver
#     acts on staggered velocity locations.
#   - For convenience, the file also stores body_to_omega edges evaluated on
#     Loc_ω(3). These are useful for a vorticity-only MVP, but they are labeled
#     as a convenience graph, not the exact Immersa interpolation stencil.

using Immersa
using StaticArrays
using ProgressMeter
using OffsetArrays
using LinearAlgebra
using HDF5
using Statistics
using Printf
using Dates
using Plots

# ------------------------------------------------------------
# Output paths
# ------------------------------------------------------------
const _FILEPATH = let f = @__FILE__
    isempty(f) ? PROGRAM_FILE : f
end

const CASE   = isempty(_FILEPATH) ? "session" : first(splitext(basename(_FILEPATH)))
const SRCDIR = isempty(_FILEPATH) ? pwd() : dirname(_FILEPATH)
const OUTDIR = joinpath(SRCDIR, "figures", CASE)
mkpath(OUTDIR)

# ------------------------------------------------------------
# User parameters
# ------------------------------------------------------------
h = 0.1
gridlims = SA[-1.0 6.0; -2.0 2.0]
grid_levels = 2

dt = 0.005
Re = 100.0
tf = 100.0
snapshot_freq = 20

cylinder_radius = 0.5
D = 2.0 * cylinder_radius

# Set true when you want to regenerate the HDF5 with new metadata/graph fields.
overwrite_h5 = true

# Delta kernel used by the solver and by the saved graph stencils.
delta_kernel = Immersa.DeltaYang3S2()

# ------------------------------------------------------------
# Grid and body
# ------------------------------------------------------------
grid = Grid(;
    h,
    n = @.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)),
    x0 = gridlims[:, 1],
    levels = grid_levels
)

r = cylinder_radius
S = 2π * r

# Marker spacing approximately 2h.
n_ib = round(Int, S / (2h))
ds = S / n_ib

body = let
    θs = range(0, 2π; length = n_ib + 1)[1:end-1]
    x = map(θs) do θ
        r * SA[cos(θ), sin(θ)]
    end
    StaticBody(x, fill(ds, n_ib))
end

u0 = UniformFlow(t -> SA[1.0, 0.0])
prob = IBProblem(grid, body, Re, u0)

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

function safe_create_group(parent, name)
    haskey(parent, name) && return parent[name]
    return create_group(parent, name)
end

function write_dataset!(group, name::AbstractString, data)
    ds = create_dataset(group, name, eltype(data), size(data))
    write(ds, data)
    return ds
end

function write_vector_dataset!(group, name::AbstractString, data)
    ds = create_dataset(group, name, eltype(data), size(data))
    write(ds, data)
    return ds
end

"""
Zero-based node id with x-index fastest.

If A has dimensions Nx × Ny, then:
    node_id(i,j) = (j-1)*Nx + (i-1)
where i,j are one-based Julia array indices.
"""
fluid_node_id(i, j, Nx) = (j - 1) * Nx + (i - 1)

"""
Get x/y coordinate vectors for a staggered-grid array location.

A is the array whose axes define the valid index range.
loc is Loc_ω(3), Loc_u(1), or Loc_u(2).
"""
function coord_vectors_for_array(grid, loc, A, lev)
    iidx = first(axes(A, 1)):last(axes(A, 1))
    jidx = first(axes(A, 2)):last(axes(A, 2))

    X, Y = coord(grid, loc, (iidx, jidx), lev)

    xvec = collect(X[:, 1])
    yvec = collect(Y[1, :])

    return xvec, yvec, collect(iidx), collect(jidx)
end

"""
Save x/y coordinate vectors and original Julia firstindex for one staggered location.
"""
function write_location_coords!(snapshot_group, locname, grid, loc, A)
    g = safe_create_group(snapshot_group, locname)

    x0, y0, iidx, jidx = coord_vectors_for_array(grid, loc, A, 1)
    Nx = length(x0)
    Ny = length(y0)

    xcoords = Matrix{Float64}(undef, Nx, grid.levels)
    ycoords = Matrix{Float64}(undef, Ny, grid.levels)

    for lev in 1:grid.levels
        xvec, yvec, _, _ = coord_vectors_for_array(grid, loc, A, lev)
        xcoords[:, lev] = xvec
        ycoords[:, lev] = yvec
    end

    write_dataset!(g, "x_coords", xcoords)
    write_dataset!(g, "y_coords", ycoords)
    write_vector_dataset!(g, "i_indices", Int64.(iidx))
    write_vector_dataset!(g, "j_indices", Int64.(jidx))

    write_attribute(g, "firstindex_i", first(axes(A, 1)))
    write_attribute(g, "firstindex_j", first(axes(A, 2)))
    write_attribute(g, "shape_Nx", Nx)
    write_attribute(g, "shape_Ny", Ny)

    return xcoords, ycoords, iidx, jidx
end

"""
Boundary mask for an Eulerian node grid.

0 = interior
1 = left boundary / inlet
2 = right boundary / outlet
3 = bottom boundary
4 = top boundary
"""
function make_boundary_mask(Nx, Ny)
    mask = zeros(Int32, Nx, Ny)

    mask[1, :]   .= 1
    mask[end, :] .= 2
    mask[:, 1]   .= 3
    mask[:, end] .= 4

    return mask
end

"""
Return body points, normals, tangents, and directed closed-loop body edges.

All edge indices are zero-based for PyTorch.
"""
function make_body_geometry(body)
    Nb = length(body.x)

    points = reduce(hcat, body.x)
    normals = similar(points)
    tangents = similar(points)

    for k in 1:Nb
        n = body.x[k] / norm(body.x[k])
        t = SA[-n[2], n[1]]

        normals[:, k] .= n
        tangents[:, k] .= t
    end

    # Directed closed-loop edges: k -> k+1 and k -> k-1.
    edge_index = Matrix{Int64}(undef, 2, 2Nb)

    e = 1
    for k in 1:Nb
        kp = k == Nb ? 1 : k + 1
        km = k == 1  ? Nb : k - 1

        edge_index[:, e] .= (k - 1, kp - 1)
        e += 1

        edge_index[:, e] .= (k - 1, km - 1)
        e += 1
    end

    return points, normals, tangents, edge_index
end

"""
Make points from tensor-product coordinate vectors.

Output shape is 2 × Nnodes with x-index fastest.
"""
function make_grid_points(xvec, yvec)
    Nx = length(xvec)
    Ny = length(yvec)
    pts = Matrix{Float64}(undef, 2, Nx * Ny)

    p = 1
    for j in 1:Ny
        for i in 1:Nx
            pts[1, p] = xvec[i]
            pts[2, p] = yvec[j]
            p += 1
        end
    end

    return pts
end

"""
Four-neighbor directed graph on a tensor-product grid.

edge_attr = [dx/h_level, dy/h_level, distance/h_level]
"""
function make_four_neighbor_edges(xvec, yvec)
    Nx = length(xvec)
    Ny = length(yvec)

    hlevel_x = abs(xvec[2] - xvec[1])
    hlevel_y = abs(yvec[2] - yvec[1])
    hlevel = 0.5 * (hlevel_x + hlevel_y)

    n_edges = 2 * (Nx - 1) * Ny + 2 * Nx * (Ny - 1)

    edge_index = Matrix{Int64}(undef, 2, n_edges)
    edge_attr = Matrix{Float64}(undef, 3, n_edges)

    e = 1

    function add_edge!(is, js, it, jt)
        src = fluid_node_id(is, js, Nx)
        dst = fluid_node_id(it, jt, Nx)

        dx = xvec[it] - xvec[is]
        dy = yvec[jt] - yvec[js]

        edge_index[:, e] .= (src, dst)
        edge_attr[:, e] .= (dx / hlevel, dy / hlevel, sqrt(dx^2 + dy^2) / hlevel)

        return nothing
    end

    for j in 1:Ny
        for i in 1:(Nx - 1)
            add_edge!(i, j, i + 1, j)
            e += 1
            add_edge!(i + 1, j, i, j)
            e += 1
        end
    end

    for j in 1:(Ny - 1)
        for i in 1:Nx
            add_edge!(i, j, i, j + 1)
            e += 1
            add_edge!(i, j + 1, i, j)
            e += 1
        end
    end

    return edge_index, edge_attr
end

"""
Body-to-grid delta-kernel edges.

This is the generic routine used for:
    body -> u1     exact Immersa-style stencil on Loc_u(1)
    body -> u2     exact Immersa-style stencil on Loc_u(2)
    body -> omega  convenience graph on Loc_ω(3)

edge_index[:, e] = [body_marker_id, grid_node_id], zero-based.

edge_attr[:, e] = [rx, ry, r, wki]

where here we use the Immersa sign convention:
    rx = (X_body - x_grid)/h_level
    ry = (Y_body - y_grid)/h_level
    r  = sqrt(rx^2 + ry^2)
    wki = delta_kernel(SA[rx, ry])

The weight is the tensor-product multidimensional delta because Immersa defines
delta(r::AbstractVector) = prod(delta, r).
"""
function make_body_to_grid_delta_edges(
    xvec,
    yvec,
    body_points;
    delta = Immersa.DeltaYang3S2(),
    tol = 0.0,
)
    Nx = length(xvec)
    Ny = length(yvec)
    Nb = size(body_points, 2)

    hlevel_x = abs(xvec[2] - xvec[1])
    hlevel_y = abs(yvec[2] - yvec[1])
    hlevel = 0.5 * (hlevel_x + hlevel_y)

    s = Immersa.support(delta)

    edge_src = Int64[]
    edge_dst = Int64[]
    attrs = NTuple{4, Float64}[]

    for k in 1:Nb
        Xk = body_points[1, k]
        Yk = body_points[2, k]

        # Search only the compact support box.
        ix1 = searchsortedfirst(xvec, Xk - s * hlevel)
        ix2 = searchsortedlast(xvec,  Xk + s * hlevel)
        iy1 = searchsortedfirst(yvec, Yk - s * hlevel)
        iy2 = searchsortedlast(yvec,  Yk + s * hlevel)

        ix1 = clamp(ix1, 1, Nx)
        ix2 = clamp(ix2, 1, Nx)
        iy1 = clamp(iy1, 1, Ny)
        iy2 = clamp(iy2, 1, Ny)

        for j in iy1:iy2
            for i in ix1:ix2
                rx = (Xk - xvec[i]) / hlevel
                ry = (Yk - yvec[j]) / hlevel

                wki = delta(SA[rx, ry])

                if abs(wki) > tol
                    push!(edge_src, k - 1)
                    push!(edge_dst, fluid_node_id(i, j, Nx))
                    push!(attrs, (rx, ry, sqrt(rx^2 + ry^2), wki))
                end
            end
        end
    end

    n_edges = length(edge_src)

    edge_index = Matrix{Int64}(undef, 2, n_edges)
    edge_attr = Matrix{Float64}(undef, 4, n_edges)

    for e in 1:n_edges
        edge_index[1, e] = edge_src[e]
        edge_index[2, e] = edge_dst[e]

        edge_attr[1, e] = attrs[e][1]
        edge_attr[2, e] = attrs[e][2]
        edge_attr[3, e] = attrs[e][3]
        edge_attr[4, e] = attrs[e][4]
    end

    return edge_index, edge_attr
end

function write_root_metadata!(file; tf, snapshot_freq)
    write_attribute(file, "case_name", CASE)
    write_attribute(file, "created_on", string(now()))
    write_attribute(file, "description", "2D Immersa static-cylinder flow dataset with true body-to-velocity delta stencils for GNN preprocessing")

    write_attribute(file, "h", h)
    write_attribute(file, "dt", dt)
    write_attribute(file, "Re", Re)
    write_attribute(file, "tf", tf)
    write_attribute(file, "snapshot_freq", snapshot_freq)

    write_attribute(file, "grid_levels", grid.levels)
    write_attribute(file, "grid_xmin", gridlims[1, 1])
    write_attribute(file, "grid_xmax", gridlims[1, 2])
    write_attribute(file, "grid_ymin", gridlims[2, 1])
    write_attribute(file, "grid_ymax", gridlims[2, 2])

    write_attribute(file, "cylinder_radius", cylinder_radius)
    write_attribute(file, "cylinder_diameter", D)
    write_attribute(file, "n_ib", n_ib)
    write_attribute(file, "ds", ds)

    write_attribute(file, "inflow_u", 1.0)
    write_attribute(file, "inflow_v", 0.0)

    write_attribute(file, "delta_kernel_solver", "Immersa.DeltaYang3S2")
    write_attribute(file, "delta_support", Immersa.support(delta_kernel))

    write_attribute(file, "edge_index_base", 0)
    write_attribute(file, "node_flattening", "node_id = (j-1)*Nx + (i-1), zero-based, x-index fastest")
end

"""
Write graph groups for every level.

Requires coordinate arrays:
    omega_xcoords, omega_ycoords
    u1_xcoords,    u1_ycoords
    u2_xcoords,    u2_ycoords
"""
function write_graph_data!(
    file,
    body_points,
    omega_xcoords,
    omega_ycoords,
    u1_xcoords,
    u1_ycoords,
    u2_xcoords,
    u2_ycoords,
)
    graph_group = safe_create_group(file, "graph")

    write_attribute(graph_group, "edge_index_base", 0)
    write_attribute(graph_group, "edge_attr_delta", "[rx, ry, r, wki]")
    write_attribute(graph_group, "delta_sign_convention", "rx=(X_body-x_grid)/h_level, ry=(Y_body-y_grid)/h_level")
    write_attribute(graph_group, "delta_kernel", "Immersa.DeltaYang3S2")
    write_attribute(graph_group, "delta_support", Immersa.support(delta_kernel))
    write_attribute(graph_group, "body_to_u1_note", "Exact Immersa-style stencil evaluated at Loc_u(1), the x-velocity staggered grid.")
    write_attribute(graph_group, "body_to_u2_note", "Exact Immersa-style stencil evaluated at Loc_u(2), the y-velocity staggered grid.")
    write_attribute(graph_group, "body_to_omega_note", "Convenience GNN graph evaluated at Loc_omega(3); not the exact solver interpolation/regularization stencil.")

    for lev in 1:grid.levels
        level_group = safe_create_group(graph_group, "level_$lev")

        # -------------------------
        # Omega graph nodes and 4-neighbor edges
        # -------------------------
        xω = omega_xcoords[:, lev]
        yω = omega_ycoords[:, lev]

        omega_points = make_grid_points(xω, yω)
        omega_edge_index, omega_edge_attr = make_four_neighbor_edges(xω, yω)

        write_dataset!(level_group, "omega_points", omega_points)
        write_dataset!(level_group, "omega_edge_index", omega_edge_index)
        write_dataset!(level_group, "omega_edge_attr", omega_edge_attr)

        # Convenience body -> omega edges.
        bω_edge_index, bω_edge_attr = make_body_to_grid_delta_edges(
            xω,
            yω,
            body_points;
            delta = delta_kernel,
        )

        write_dataset!(level_group, "body_to_omega_edge_index", bω_edge_index)
        write_dataset!(level_group, "body_to_omega_edge_attr", bω_edge_attr)

        # -------------------------
        # True Immersa-style body -> velocity stencils
        # -------------------------
        xu1 = u1_xcoords[:, lev]
        yu1 = u1_ycoords[:, lev]

        xu2 = u2_xcoords[:, lev]
        yu2 = u2_ycoords[:, lev]

        u1_points = make_grid_points(xu1, yu1)
        u2_points = make_grid_points(xu2, yu2)

        bu1_edge_index, bu1_edge_attr = make_body_to_grid_delta_edges(
            xu1,
            yu1,
            body_points;
            delta = delta_kernel,
        )

        bu2_edge_index, bu2_edge_attr = make_body_to_grid_delta_edges(
            xu2,
            yu2,
            body_points;
            delta = delta_kernel,
        )

        write_dataset!(level_group, "u1_points", u1_points)
        write_dataset!(level_group, "u2_points", u2_points)

        write_dataset!(level_group, "body_to_u1_edge_index", bu1_edge_index)
        write_dataset!(level_group, "body_to_u1_edge_attr", bu1_edge_attr)

        write_dataset!(level_group, "body_to_u2_edge_index", bu2_edge_index)
        write_dataset!(level_group, "body_to_u2_edge_attr", bu2_edge_attr)

        write_attribute(level_group, "level", lev)
        write_attribute(level_group, "omega_num_nodes", size(omega_points, 2))
        write_attribute(level_group, "u1_num_nodes", size(u1_points, 2))
        write_attribute(level_group, "u2_num_nodes", size(u2_points, 2))
        write_attribute(level_group, "num_body_to_omega_edges", size(bω_edge_index, 2))
        write_attribute(level_group, "num_body_to_u1_edges", size(bu1_edge_index, 2))
        write_attribute(level_group, "num_body_to_u2_edges", size(bu2_edge_index, 2))
    end
end

# ------------------------------------------------------------
# Solver and HDF5 writer
# ------------------------------------------------------------
function solution(file; tf, snapshot_freq)
    T = Float64

    write_root_metadata!(file; tf, snapshot_freq)

    sol = CNAB(prob; dt, delta = delta_kernel)

    # Initial perturbation to induce vortex shedding.
    map!(sol.ω[1][3], CartesianIndices(sol.ω[1][3])) do I
        x = coord(grid, Loc_ω(3), I)
        p = x - SA[-0.75, 0.0]
        rp = 0.25
        0.5 * (1 - clamp(norm(p) / rp, 0, 1))
    end
    apply_vorticity!(sol)

    i_all = 1:(1 + round(Int, tf / dt))
    n_all = length(i_all)

    i_snapshot = i_all[1:snapshot_freq:end]
    n_snapshot = length(i_snapshot)

    # ---------------------------
    # Body group
    # ---------------------------
    body_group = create_group(file, "body")

    body_points, body_normals, body_tangents, body_edge_index = make_body_geometry(body)

    write_dataset!(body_group, "points", body_points)
    write_vector_dataset!(body_group, "lengths", collect(body.ds))
    write_dataset!(body_group, "normals", body_normals)
    write_dataset!(body_group, "tangents", body_tangents)
    write_dataset!(body_group, "edge_index", body_edge_index)

    write_attribute(body_group, "edge_index_base", 0)
    write_attribute(body_group, "edge_direction", "directed closed-loop nearest-neighbor marker edges")
    write_attribute(body_group, "node_features_recommended_mvp", "[X/D, Y/D]")
    write_attribute(body_group, "node_features_possible_ablation", "[X/D, Y/D, nx, ny, tx, ty]")

    # ---------------------------
    # All-time group
    # ---------------------------
    all_group = create_group(file, "all")

    t_all = create_dataset(all_group, "t", T, (n_all,))
    Cl = create_dataset(all_group, "Cl", T, (n_all,))
    Cd = create_dataset(all_group, "Cd", T, (n_all,))

    # ---------------------------
    # Snapshot group
    # ---------------------------
    snapshot_group = create_group(file, "snapshots")

    t_snapshot = create_dataset(snapshot_group, "t", T, (n_snapshot,))

    # Main target field: vorticity/circulation location.
    omega = create_dataset(
        snapshot_group,
        "omega",
        T,
        (size(sol.ω[1][3])..., grid.levels, n_snapshot)
    )

    write_attribute(omega, "location", "Loc_omega(3)")
    write_attribute(omega, "description", "Scalar vorticity/circulation field used as ML state and target.")
    write_attribute(omega, "firstindex", collect(first.(axes(sol.ω[1][3]))))

    # Velocity fields are saved too, because the real delta stencils are defined on Loc_u.
    ux = create_dataset(
        snapshot_group,
        "ux",
        T,
        (size(sol.u[1][1])..., grid.levels, n_snapshot)
    )

    uy = create_dataset(
        snapshot_group,
        "uy",
        T,
        (size(sol.u[1][2])..., grid.levels, n_snapshot)
    )

    write_attribute(ux, "location", "Loc_u(1)")
    write_attribute(uy, "location", "Loc_u(2)")
    write_attribute(ux, "firstindex", collect(first.(axes(sol.u[1][1]))))
    write_attribute(uy, "firstindex", collect(first.(axes(sol.u[1][2]))))

    # Coordinates for each staggered location.
    coords_group = create_group(snapshot_group, "coords")
    write_attribute(coords_group, "note", "Coordinate vectors are saved separately for omega, u1, and u2 because they live on staggered locations.")

    omega_xcoords, omega_ycoords, _, _ = write_location_coords!(
        coords_group,
        "omega",
        grid,
        Loc_ω(3),
        sol.ω[1][3],
    )

    u1_xcoords, u1_ycoords, _, _ = write_location_coords!(
        coords_group,
        "u1",
        grid,
        Loc_u(1),
        sol.u[1][1],
    )

    u2_xcoords, u2_ycoords, _, _ = write_location_coords!(
        coords_group,
        "u2",
        grid,
        Loc_u(2),
        sol.u[1][2],
    )

    # Boundary mask for omega nodes, used as optional fluid-node feature.
    Nxω, Nyω = size(sol.ω[1][3])
    boundary_mask = create_dataset(snapshot_group, "omega_boundary_mask", Int32, (Nxω, Nyω, grid.levels))

    for lev in 1:grid.levels
        boundary_mask[:, :, lev] = make_boundary_mask(Nxω, Nyω)
    end

    write_attribute(snapshot_group, "omega_boundary_mask_labels", "0 interior, 1 left/inlet, 2 right/outlet, 3 bottom, 4 top")

    # ---------------------------
    # Graph group
    # ---------------------------
    write_graph_data!(
        file,
        body_points,
        omega_xcoords,
        omega_ycoords,
        u1_xcoords,
        u1_ycoords,
        u2_xcoords,
        u2_ycoords,
    )

    # ---------------------------
    # Time integration
    # ---------------------------
    @showprogress desc = "solving" for _ in 0:round(Int, tf / dt)
        step!(sol)

        f = surface_force_sum(sol)

        t_all[sol.i] = sol.t
        Cd[sol.i] = 2 * f[1]
        Cl[sol.i] = 2 * f[2]

        if sol.i in i_snapshot
            isnap = 1 + (sol.i - first(i_snapshot)) ÷ step(i_snapshot)

            t_snapshot[isnap] = sol.t

            for lev in eachindex(sol.ω)
                omega[:, :, lev, isnap] = OffsetArrays.no_offset_view(sol.ω[lev][3])
                ux[:, :, lev, isnap] = OffsetArrays.no_offset_view(sol.u[lev][1])
                uy[:, :, lev, isnap] = OffsetArrays.no_offset_view(sol.u[lev][2])
            end
        end
    end
end

# ------------------------------------------------------------
# Run or reuse simulation
# ------------------------------------------------------------
soln_path = joinpath(SRCDIR, "$(CASE).h5")

if isfile(soln_path) && overwrite_h5
    @info "Removing old file" soln_path
    rm(soln_path)
end

if isfile(soln_path)
    @info "File already exists" soln_path
else
    h5open(soln_path, "cw") do file
        solution(file; tf = tf, snapshot_freq = snapshot_freq)
    end
end

# ------------------------------------------------------------
# Inspect HDF5 structure
# ------------------------------------------------------------
h5open(soln_path, "r") do file
    println("\nHDF5 groups:")
    println(keys(file))

    println("\n/body:")
    println(keys(file["body"]))

    println("\n/snapshots:")
    println(keys(file["snapshots"]))

    println("\n/snapshots/coords:")
    println(keys(file["snapshots/coords"]))

    println("\n/graph:")
    println(keys(file["graph"]))

    for lev in 1:grid.levels
        g = file["graph/level_$lev"]
        println("\n/graph/level_$lev:")
        println(keys(g))
        println("  num body_to_u1 edges     = ", read_attribute(g, "num_body_to_u1_edges"))
        println("  num body_to_u2 edges     = ", read_attribute(g, "num_body_to_u2_edges"))
        println("  num body_to_omega edges  = ", read_attribute(g, "num_body_to_omega_edges"))
    end
end

# ------------------------------------------------------------
# Quick contour animation with actual IB marker scatter points
# ------------------------------------------------------------
h5open(soln_path, "r") do file
    t = read(file["snapshots/t"])
    omega = file["snapshots/omega"]

    body_pts = read(file["body/points"])
    xb = body_pts[1, :]
    yb = body_pts[2, :]

    xω = read(file["snapshots/coords/omega/x_coords"])
    yω = read(file["snapshots/coords/omega/y_coords"])

    omega_lim = 12.0
    anim = Animation()

    levels_to_plot = [1]

    @showprogress desc = "plotting" for isnap in eachindex(t)
        p = plot(
            legend = false,
            aspect_ratio = :equal,
            xlim = (-1, 6),
            ylim = (-2, 2),
            framestyle = :box
        )

        for lev in levels_to_plot
            xvec = xω[:, lev]
            yvec = yω[:, lev]

            z = omega[:, :, lev, isnap]
            zplot = copy(z)

            finite_mask = isfinite.(zplot)
            zplot[finite_mask] .= clamp.(zplot[finite_mask], -omega_lim, omega_lim)
            zplot[.!finite_mask] .= NaN

            contourf!(
                p,
                xvec,
                yvec,
                zplot';
                aspect_ratio = :equal,
                colormap = :seaborn_icefire_gradient,
                levels = 18,
                lw = 0,
                legend = false,
                clim = (-omega_lim, omega_lim),
                xlim = (-1, 6),
                ylim = (-2, 2),
            )
        end

        scatter!(
            p,
            xb,
            yb;
            color = :black,
            markerstrokecolor = :black,
            markersize = 4,
            label = false
        )

        title!(p, @sprintf("t = %.2f", t[isnap]))
        frame(anim, p)
    end

    mkpath(OUTDIR)
    gif(anim, joinpath(OUTDIR, "$(CASE)_vorticity_contour_markers.gif"); fps = 24)
end