# -*- coding: utf-8 -*-
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

# ── Path setup ────────────────────────────────────────────────────────────────
const _FILEPATH = let f = @__FILE__
    isempty(f) ? PROGRAM_FILE : f
end
const CASE   = isempty(_FILEPATH) ? "session" : first(splitext(basename(_FILEPATH)))
const SRCDIR = isempty(_FILEPATH) ? pwd() : dirname(_FILEPATH)
const OUTDIR = joinpath(SRCDIR, "figures", CASE)
mkpath(OUTDIR)

# ── Simulation parameters ─────────────────────────────────────────────────────
h               = 0.1
gridlims        = SA[-1.0 6.0; -2.0 2.0]
grid_levels     = 2
dt              = 0.005
Re              = 100.0
tf              = 100.0
snapshot_freq   = 20
cylinder_radius = 0.5
cylinder_center = SA[0.0, 0.0]   # explicit center — currently origin, preserved for generality
D               = 2.0 * cylinder_radius
overwrite_h5    = true
delta_kernel    = Immersa.DeltaYang3S2()   # used by BOTH the solver and graph builder

# ── Grid ──────────────────────────────────────────────────────────────────────
grid = Grid(;
    h,
    n      = @.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)),
    x0     = gridlims[:, 1],
    levels = grid_levels,
)

# ── Static cylinder body ──────────────────────────────────────────────────────
r    = cylinder_radius
S    = 2π * r
n_ib = round(Int, S / (2h))
ds   = S / n_ib

body = let
    θs = range(0, 2π; length = n_ib + 1)[1:end-1]
    x  = map(θs) do θ
        cylinder_center + r * SA[cos(θ), sin(θ)]   # parameterized around explicit center
    end
    StaticBody(x, fill(ds, n_ib))
end

# ── Problem and solver (unchanged) ────────────────────────────────────────────
u0   = UniformFlow(t -> SA[1.0, 0.0])
prob = IBProblem(grid, body, Re, u0)
sol  = CNAB(prob; dt, delta = delta_kernel)

# Initial vorticity perturbation to trigger shedding
map!(sol.ω[1][3], CartesianIndices(sol.ω[1][3])) do I
    x  = coord(grid, Loc_ω(3), I)
    p  = x - SA[-0.75, 0.0]
    rp = 0.25
    0.5 * (1 - clamp(norm(p) / rp, 0, 1))
end
apply_vorticity!(sol)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Body normals and tangents (CCW parameterisation) ──────────────────────────
function body_normals_tangents(xb)
    nb       = length(xb)
    normals  = zeros(2, nb)
    tangents = zeros(2, nb)
    for k in 1:nb
        xnext          = xb[mod1(k + 1, nb)]
        xprev          = xb[mod1(k - 1, nb)]
        t              = xnext - xprev
        t              = t / norm(t)
        tangents[1, k] = t[1]
        tangents[2, k] = t[2]
        # outward normal for CCW curve: rotate tangent 90° clockwise
        normals[1, k]  =  t[2]
        normals[2, k]  = -t[1]
    end
    return normals, tangents
end

# ── Directed closed-loop body graph with edge attributes ──────────────────────
# Returns:
#   edge_index : 2 × (2*nb) Int32   — zero-based [src; dst]
#   edge_attr  : 3 × (2*nb) Float64 — [(Xq-Xk)/D, (Yq-Yk)/D, distance/D]
# Edge ordering matches the original closed-loop convention:
#   columns 1..nb    — forward edges  k → k+1
#   columns nb+1..2nb — backward edges k → k-1
# (all modulo nb, 0-based node IDs)
function body_closed_loop_graph(body_pts, D)
    nb = size(body_pts, 2)
    ei = zeros(Int32,   2, 2 * nb)
    ea = zeros(Float64, 3, 2 * nb)
    for k in 0:(nb - 1)
        k1 = k + 1                      # 1-based index of this marker

        # forward: k -> k+1 (mod nb)
        q_fwd  = mod(k + 1, nb)
        q1_fwd = q_fwd + 1              # 1-based
        dx_f   = (body_pts[1, q1_fwd] - body_pts[1, k1]) / D
        dy_f   = (body_pts[2, q1_fwd] - body_pts[2, k1]) / D
        ei[1, k + 1]      = Int32(k)
        ei[2, k + 1]      = Int32(q_fwd)
        ea[1, k + 1]      = dx_f
        ea[2, k + 1]      = dy_f
        ea[3, k + 1]      = sqrt(dx_f^2 + dy_f^2)

        # backward: k -> k-1 (mod nb)
        q_bwd  = mod(k - 1 + nb, nb)
        q1_bwd = q_bwd + 1              # 1-based
        dx_b   = (body_pts[1, q1_bwd] - body_pts[1, k1]) / D
        dy_b   = (body_pts[2, q1_bwd] - body_pts[2, k1]) / D
        ei[1, k + 1 + nb] = Int32(k)
        ei[2, k + 1 + nb] = Int32(q_bwd)
        ea[1, k + 1 + nb] = dx_b
        ea[2, k + 1 + nb] = dy_b
        ea[3, k + 1 + nb] = sqrt(dx_b^2 + dy_b^2)
    end
    return ei, ea
end

# ── Physical coordinate vectors for a staggered grid location at a level ──────
# Returns (xvec, yvec) — plain 1-D Float64 arrays.
function grid_coord_vecs(grid, loc, arr, lev)
    ax1  = axes(arr, 1)
    ax2  = axes(arr, 2)
    X, Y = coord(grid, loc, (ax1, ax2), lev)
    return collect(X), collect(Y)
end

# ── Flat (2 × N) point array; column = zero-based node_id ────────────────────
# Flattening convention: node_id = (j-1)*Nx + (i-1), x-index fastest.
function grid_points_flat(xvec, yvec)
    Nx  = length(xvec)
    Ny  = length(yvec)
    pts = zeros(Float64, 2, Nx * Ny)
    for j in 1:Ny, i in 1:Nx
        col         = (j - 1) * Nx + (i - 1) + 1   # 1-based Julia column
        pts[1, col] = xvec[i]
        pts[2, col] = yvec[j]
    end
    return pts
end

# ── Omega 4-neighbor directed graph ───────────────────────────────────────────
# Edge attributes: [dx/h_level, dy/h_level, distance/h_level]
# On a unit-normalised grid these are ±1 or 0, distance = 1.
function omega_4neighbor_graph(xvec, yvec)
    Nx = length(xvec)
    Ny = length(yvec)

    dirs = (
        ( 1,  0,  1.0,  0.0),
        (-1,  0, -1.0,  0.0),
        ( 0,  1,  0.0,  1.0),
        ( 0, -1,  0.0, -1.0),
    )

    src_buf = Int32[]
    dst_buf = Int32[]
    dx_buf  = Float64[]
    dy_buf  = Float64[]
    d_buf   = Float64[]

    for j in 1:Ny, i in 1:Nx
        src = Int32((j - 1) * Nx + (i - 1))
        for (di, dj, dx, dy) in dirs
            ni, nj = i + di, j + dj
            (ni < 1 || ni > Nx || nj < 1 || nj > Ny) && continue
            push!(src_buf, src)
            push!(dst_buf, Int32((nj - 1) * Nx + (ni - 1)))
            push!(dx_buf, dx)
            push!(dy_buf, dy)
            push!(d_buf,  sqrt(dx^2 + dy^2))
        end
    end

    n_e = length(src_buf)
    ei  = zeros(Int32,   2, n_e)
    ea  = zeros(Float64, 3, n_e)
    for e in 1:n_e
        ei[1, e] = src_buf[e]
        ei[2, e] = dst_buf[e]
        ea[1, e] = dx_buf[e]
        ea[2, e] = dy_buf[e]
        ea[3, e] = d_buf[e]
    end
    return ei, ea
end

# ── Body-to-grid delta-kernel edges ───────────────────────────────────────────
#
# body_pts : 2 × Nb physical coordinates of body markers.
# xvec/yvec: 1-D coordinate arrays for the target staggered grid at this level.
#
# Returns (edge_index, edge_attr):
#   edge_index : 2 × E Int32   — [body_marker_id (0-based); grid_node_id (0-based)]
#   edge_attr  : 4 × E Float64 — [rx, ry, r, wki]
#
# Fixed geometry convention (body position minus grid position):
#   rx = (X_body - x_grid) / h_level
#   ry = (Y_body - y_grid) / h_level
#
# The kernel is evaluated directly via the Immersa AbstractDeltaFunc product rule:
#   delta(SA[rx, ry]) = δ(rx) · δ(ry)
# No manual reimplementation of the Yang kernel.
function make_body_to_grid_delta_edges(
    xvec, yvec, body_pts;
    delta = Immersa.DeltaYang3S2(),
    tol   = 0.0,
)
    Nx = length(xvec)
    Ny = length(yvec)
    Nb = size(body_pts, 2)

    # Infer h_level from the uniformly-spaced coordinate vector.
    hlevel = if Nx > 1
        (xvec[end] - xvec[1]) / (Nx - 1)
    elseif Ny > 1
        (yvec[end] - yvec[1]) / (Ny - 1)
    else
        1.0
    end

    s = Immersa.support(delta)   # compact-support radius in normalised units

    src_buf = Int32[]
    dst_buf = Int32[]
    rx_buf  = Float64[]
    ry_buf  = Float64[]
    r_buf   = Float64[]
    w_buf   = Float64[]

    for k in 1:Nb
        Xb = body_pts[1, k]
        Yb = body_pts[2, k]
        for j in 1:Ny
            yg = yvec[j]
            ry = (Yb - yg) / hlevel
            abs(ry) > s && continue
            for i in 1:Nx
                xg  = xvec[i]
                rx  = (Xb - xg) / hlevel
                abs(rx) > s && continue
                wki = delta(SA[rx, ry])   # Immersa kernel: δ(rx)·δ(ry)
                abs(wki) <= tol && continue
                push!(src_buf, Int32(k - 1))
                push!(dst_buf, Int32((j - 1) * Nx + (i - 1)))
                push!(rx_buf,  rx)
                push!(ry_buf,  ry)
                push!(r_buf,   sqrt(rx^2 + ry^2))
                push!(w_buf,   wki)
            end
        end
    end

    n_e        = length(src_buf)
    edge_index = zeros(Int32,   2, n_e)
    edge_attr  = zeros(Float64, 4, n_e)
    for e in 1:n_e
        edge_index[1, e] = src_buf[e]
        edge_index[2, e] = dst_buf[e]
        edge_attr[1, e]  = rx_buf[e]
        edge_attr[2, e]  = ry_buf[e]
        edge_attr[3, e]  = r_buf[e]
        edge_attr[4, e]  = w_buf[e]
    end
    return edge_index, edge_attr
end

# ── Reverse an edge_index (swap src and dst rows) ────────────────────────────
# Used to produce interpolation-like grid-to-body edges from the stored
# body-to-grid connectivity.  Attributes are REUSED without negation because
# the fixed convention (body position minus grid position) is independent of
# the direction of message passing.
function reverse_edge_index(edge_index)
    reversed        = similar(edge_index)
    reversed[1, :] .= edge_index[2, :]
    reversed[2, :] .= edge_index[1, :]
    return reversed
end

# ── Omega boundary mask ────────────────────────────────────────────────────────
# 0 = interior  1 = left/inlet  2 = right/outlet  3 = bottom  4 = top
function omega_boundary_mask(Nx, Ny)
    mask = zeros(Int32, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        mask[i, j] = if i == 1
            Int32(1)
        elseif i == Nx
            Int32(2)
        elseif j == 1
            Int32(3)
        elseif j == Ny
            Int32(4)
        else
            Int32(0)
        end
    end
    return mask
end

# ══════════════════════════════════════════════════════════════════════════════
# TIME-STEP BOOKKEEPING
# ══════════════════════════════════════════════════════════════════════════════
n_steps    = round(Int, tf / dt)
i_all      = 1:(1 + n_steps)
n_all      = length(i_all)
i_snapshot = i_all[1:snapshot_freq:end]
n_snapshot = length(i_snapshot)

# ── HDF5 output path ──────────────────────────────────────────────────────────
soln_path = joinpath(SRCDIR, "$(CASE).h5")

if overwrite_h5 && isfile(soln_path)
    rm(soln_path)
    @info "Deleted existing file" soln_path
end

# ══════════════════════════════════════════════════════════════════════════════
# STATIC GEOMETRY (computed once, before opening HDF5)
# ══════════════════════════════════════════════════════════════════════════════
const T_F = Float64

Nx_ω,  Ny_ω  = size(sol.ω[1][3])
Nx_u1, Ny_u1 = size(sol.u[1][1])
Nx_u2, Ny_u2 = size(sol.u[1][2])

body_pts_mat               = reduce(hcat, body.x)             # 2 × n_ib
body_nrm_mat, body_tan_mat = body_normals_tangents(body.x)
bnd_mask                   = omega_boundary_mask(Nx_ω, Ny_ω)

# Body-centered, nondimensional node features: [(X-Xc)/D, (Y-Yc)/D]
body_center_col        = reshape(collect(cylinder_center), 2, 1)
body_node_features_mat = (body_pts_mat .- body_center_col) ./ D   # 2 × n_ib

# Body closed-loop graph with edge attributes [(Xq-Xk)/D, (Yq-Yk)/D, distance/D]
body_ei, body_ea = body_closed_loop_graph(body_pts_mat, D)

# ── Body-graph consistency assertions ─────────────────────────────────────────
@assert size(body_node_features_mat) == (2, n_ib)
@assert size(body_ei, 1) == 2
@assert size(body_ea, 1) == 3
@assert size(body_ei, 2) == size(body_ea, 2)
@assert size(body_ei, 2) == 2 * n_ib

# ══════════════════════════════════════════════════════════════════════════════
# BUILD HDF5 FILE, RUN SIMULATION, WRITE GRAPH
# ══════════════════════════════════════════════════════════════════════════════
h5open(soln_path, "cw") do file

    # ── Root attributes ────────────────────────────────────────────────────────
    write_attribute(file, "case_name",           CASE)
    write_attribute(file, "created_on",          string(now()))
    write_attribute(file, "description",
        "MeshGraphNet-style dataset: flow past static cylinder Re=$(Re). " *
        "ML task: omega^n -> omega^(n+1).")
    write_attribute(file, "h",                   h)
    write_attribute(file, "dt",                  dt)
    write_attribute(file, "Re",                  Re)
    write_attribute(file, "tf",                  tf)
    write_attribute(file, "snapshot_freq",       snapshot_freq)
    write_attribute(file, "grid_levels",         grid_levels)
    write_attribute(file, "grid_xmin",           gridlims[1, 1])
    write_attribute(file, "grid_xmax",           gridlims[1, 2])
    write_attribute(file, "grid_ymin",           gridlims[2, 1])
    write_attribute(file, "grid_ymax",           gridlims[2, 2])
    write_attribute(file, "cylinder_radius",     cylinder_radius)
    write_attribute(file, "cylinder_diameter",   D)
    write_attribute(file, "cylinder_center_x",   cylinder_center[1])
    write_attribute(file, "cylinder_center_y",   cylinder_center[2])
    write_attribute(file, "n_ib",                n_ib)
    write_attribute(file, "ds",                  ds)
    write_attribute(file, "inflow_u",            1.0)
    write_attribute(file, "inflow_v",            0.0)
    write_attribute(file, "delta_kernel_solver", "Immersa.DeltaYang3S2")
    write_attribute(file, "delta_support",       Immersa.support(delta_kernel))
    write_attribute(file, "edge_index_base",     0)
    write_attribute(file, "node_flattening",
        "node_id = (j-1)*Nx + (i-1), zero-based, x-index fastest")

    # ── /body ─────────────────────────────────────────────────────────────────
    grp_body = create_group(file, "body")
    write_attribute(grp_body, "edge_index_base",  0)
    write_attribute(grp_body, "edge_direction",
        "directed closed-loop nearest-neighbor marker edges")
    write_attribute(grp_body, "node_features_recommended_mvp",
        "[(X-Xc)/D, (Y-Yc)/D]")
    write_attribute(grp_body, "node_features_possible_ablation",
        "[(X-Xc)/D, (Y-Yc)/D, nx, ny, tx, ty]")
    write_attribute(grp_body, "body_edge_attr_description",
        "[(Xq-Xk)/D, (Yq-Yk)/D, distance/D] for directed body edge k->q.")

    # Raw Lagrangian marker coordinates (physical units, unchanged)
    dset_bpts = create_dataset(grp_body, "points",     T_F,   (2, n_ib))
    dset_blen = create_dataset(grp_body, "lengths",    T_F,   (n_ib,))
    dset_bnrm = create_dataset(grp_body, "normals",    T_F,   (2, n_ib))
    dset_btan = create_dataset(grp_body, "tangents",   T_F,   (2, n_ib))
    dset_bpts[:, :] = body_pts_mat
    dset_blen[:]    = body.ds
    dset_bnrm[:, :] = body_nrm_mat
    dset_btan[:, :] = body_tan_mat

    # Body-centered nondimensional node features for MVP 2
    dset_bnf = create_dataset(grp_body, "node_features_mvp2", T_F, (2, n_ib))
    dset_bnf[:, :] = body_node_features_mat
    write_attribute(dset_bnf, "feature_1", "(X - Xc)/D")
    write_attribute(dset_bnf, "feature_2", "(Y - Yc)/D")
    write_attribute(dset_bnf, "description",
        "Body-centered nondimensional marker positions. " *
        "Xc = cylinder_center_x, D = cylinder_diameter.")

    # Body graph connectivity and edge attributes
    dset_bei = create_dataset(grp_body, "edge_index", Int32, (2, 2 * n_ib))
    dset_bea = create_dataset(grp_body, "edge_attr",  T_F,   (3, 2 * n_ib))
    dset_bei[:, :] = body_ei
    dset_bea[:, :] = body_ea
    write_attribute(dset_bea, "description",
        "[(Xq-Xk)/D, (Yq-Yk)/D, distance/D]. " *
        "Forward edges (k->k+1) in columns 1..Nb, backward edges (k->k-1) in columns Nb+1..2Nb.")

    # ── /all ──────────────────────────────────────────────────────────────────
    grp_all  = create_group(file, "all")
    ds_t_all = create_dataset(grp_all, "t",  T_F, (n_all,))
    ds_Cl    = create_dataset(grp_all, "Cl", T_F, (n_all,))
    ds_Cd    = create_dataset(grp_all, "Cd", T_F, (n_all,))

    # ── /snapshots ────────────────────────────────────────────────────────────
    grp_snap  = create_group(file, "snapshots")
    ds_t_snap = create_dataset(grp_snap, "t", T_F, (n_snapshot,))

    ds_omega = create_dataset(
        grp_snap, "omega", T_F, (Nx_ω, Ny_ω, grid_levels, n_snapshot),
    )
    write_attribute(ds_omega, "location",    "Loc_omega(3)")
    write_attribute(ds_omega, "description",
        "Scalar vorticity/circulation field used as ML state and target.")
    write_attribute(ds_omega, "firstindex",  collect(first.(axes(sol.ω[1][3]))))

    ds_ux = create_dataset(
        grp_snap, "ux", T_F, (Nx_u1, Ny_u1, grid_levels, n_snapshot),
    )
    write_attribute(ds_ux, "location",   "Loc_u(1)")
    write_attribute(ds_ux, "firstindex", collect(first.(axes(sol.u[1][1]))))

    ds_uy = create_dataset(
        grp_snap, "uy", T_F, (Nx_u2, Ny_u2, grid_levels, n_snapshot),
    )
    write_attribute(ds_uy, "location",   "Loc_u(2)")
    write_attribute(ds_uy, "firstindex", collect(first.(axes(sol.u[1][2]))))

    # Categorical boundary mask — recommended to one-hot encode in Python loader
    ds_mask = create_dataset(grp_snap, "omega_boundary_mask", Int32, (Nx_ω, Ny_ω))
    ds_mask[:, :] = bnd_mask
    write_attribute(ds_mask, "encoding",
        "categorical integer: 0=interior, 1=left/inlet, 2=right/outlet, 3=bottom, 4=top")
    write_attribute(ds_mask, "recommended_python_usage",
        "Convert to one-hot in the Python data loader. " *
        "Do not treat category IDs as a continuous scalar. " *
        "Example: torch.nn.functional.one_hot(mask, num_classes=5).")
    write_attribute(ds_mask, "mvp1_fluid_node_features",
        "[omega, one_hot(boundary_type)] — construct in Python from these two datasets.")

    # ── /snapshots/coords ─────────────────────────────────────────────────────
    grp_coords = create_group(grp_snap, "coords")

    # omega
    grp_co_omega = create_group(grp_coords, "omega")
    let
        ax1 = axes(sol.ω[1][3], 1)
        ax2 = axes(sol.ω[1][3], 2)
        dxc = create_dataset(grp_co_omega, "x_coords",  T_F,   (Nx_ω, grid_levels))
        dyc = create_dataset(grp_co_omega, "y_coords",  T_F,   (Ny_ω, grid_levels))
        dxi = create_dataset(grp_co_omega, "i_indices", Int32, (Nx_ω,))
        dxj = create_dataset(grp_co_omega, "j_indices", Int32, (Ny_ω,))
        dxi[:] = Int32.(collect(ax1))
        dxj[:] = Int32.(collect(ax2))
        write_attribute(grp_co_omega, "firstindex_i", first(ax1))
        write_attribute(grp_co_omega, "firstindex_j", first(ax2))
        write_attribute(grp_co_omega, "shape_Nx",     Nx_ω)
        write_attribute(grp_co_omega, "shape_Ny",     Ny_ω)
        for lev in 1:grid_levels
            Xv, Yv      = grid_coord_vecs(grid, Loc_ω(3), sol.ω[lev][3], lev)
            dxc[:, lev] = Xv
            dyc[:, lev] = Yv
        end
    end

    # u1
    grp_co_u1 = create_group(grp_coords, "u1")
    let
        ax1 = axes(sol.u[1][1], 1)
        ax2 = axes(sol.u[1][1], 2)
        dxc = create_dataset(grp_co_u1, "x_coords",  T_F,   (Nx_u1, grid_levels))
        dyc = create_dataset(grp_co_u1, "y_coords",  T_F,   (Ny_u1, grid_levels))
        dxi = create_dataset(grp_co_u1, "i_indices", Int32, (Nx_u1,))
        dxj = create_dataset(grp_co_u1, "j_indices", Int32, (Ny_u1,))
        dxi[:] = Int32.(collect(ax1))
        dxj[:] = Int32.(collect(ax2))
        write_attribute(grp_co_u1, "firstindex_i", first(ax1))
        write_attribute(grp_co_u1, "firstindex_j", first(ax2))
        write_attribute(grp_co_u1, "shape_Nx",     Nx_u1)
        write_attribute(grp_co_u1, "shape_Ny",     Ny_u1)
        for lev in 1:grid_levels
            Xv, Yv      = grid_coord_vecs(grid, Loc_u(1), sol.u[lev][1], lev)
            dxc[:, lev] = Xv
            dyc[:, lev] = Yv
        end
    end

    # u2
    grp_co_u2 = create_group(grp_coords, "u2")
    let
        ax1 = axes(sol.u[1][2], 1)
        ax2 = axes(sol.u[1][2], 2)
        dxc = create_dataset(grp_co_u2, "x_coords",  T_F,   (Nx_u2, grid_levels))
        dyc = create_dataset(grp_co_u2, "y_coords",  T_F,   (Ny_u2, grid_levels))
        dxi = create_dataset(grp_co_u2, "i_indices", Int32, (Nx_u2,))
        dxj = create_dataset(grp_co_u2, "j_indices", Int32, (Ny_u2,))
        dxi[:] = Int32.(collect(ax1))
        dxj[:] = Int32.(collect(ax2))
        write_attribute(grp_co_u2, "firstindex_i", first(ax1))
        write_attribute(grp_co_u2, "firstindex_j", first(ax2))
        write_attribute(grp_co_u2, "shape_Nx",     Nx_u2)
        write_attribute(grp_co_u2, "shape_Ny",     Ny_u2)
        for lev in 1:grid_levels
            Xv, Yv      = grid_coord_vecs(grid, Loc_u(2), sol.u[lev][2], lev)
            dxc[:, lev] = Xv
            dyc[:, lev] = Yv
        end
    end

    # ── Time loop ──────────────────────────────────────────────────────────────
    @showprogress desc = "solving" for _ in 0:n_steps
        step!(sol)

        # TODO: Save local per-marker force — shape (2, n_ib, n_snapshot) as
        # /snapshots/body_force.  surface_force_sum(sol) integrates the marker
        # quantities in sol.f_tilde (or the equivalent internal field) to
        # produce the global force vector returned here.  The exact field name,
        # scaling factor, and physical units must be verified against the
        # Immersa source (cnab.jl, surface_force_sum / surface_force!) before
        # saving.  Do NOT reconstruct local forces from the global Cl / Cd.
        f = surface_force_sum(sol)

        ds_t_all[sol.i] = sol.t
        ds_Cd[sol.i]    = 2 * f[1]
        ds_Cl[sol.i]    = 2 * f[2]

        if sol.i in i_snapshot
            isnap            = 1 + (sol.i - first(i_snapshot)) ÷ step(i_snapshot)
            ds_t_snap[isnap] = sol.t
            for lev in 1:grid_levels
                ds_omega[:, :, lev, isnap] = OffsetArrays.no_offset_view(sol.ω[lev][3])
                ds_ux[:, :,    lev, isnap] = OffsetArrays.no_offset_view(sol.u[lev][1])
                ds_uy[:, :,    lev, isnap] = OffsetArrays.no_offset_view(sol.u[lev][2])
            end
        end
    end

    # ── /graph ────────────────────────────────────────────────────────────────
    grp_graph = create_group(file, "graph")
    write_attribute(grp_graph, "edge_index_base",  0)
    write_attribute(grp_graph, "edge_attr_delta",  "[rx, ry, r, wki]")
    write_attribute(grp_graph, "delta_kernel",     "Immersa.DeltaYang3S2")
    write_attribute(grp_graph, "delta_support",    Immersa.support(delta_kernel))
    # Fixed geometry convention — independent of message-passing direction
    write_attribute(grp_graph, "cross_edge_geometry_convention",
        "rx=(X_body-x_grid)/h_level and ry=(Y_body-y_grid)/h_level, " *
        "independent of edge direction. " *
        "body position minus grid position in all cross-edge attr columns.")
    write_attribute(grp_graph, "body_to_u1_note",
        "Delta-kernel support and kernel weights evaluated at Loc_u(1), " *
        "the x-velocity staggered grid. " *
        "Not the complete interpolation or spreading matrix.")
    write_attribute(grp_graph, "body_to_u2_note",
        "Delta-kernel support and kernel weights evaluated at Loc_u(2), " *
        "the y-velocity staggered grid. " *
        "Not the complete interpolation or spreading matrix.")
    write_attribute(grp_graph, "u1_to_body_note",
        "Interpolation-like GNN edges: reversed body_to_u1 connectivity. " *
        "Edge attributes reuse the fixed body-minus-grid geometry (not negated). " *
        "Not the exact application of the Immersa interpolation operator E.")
    write_attribute(grp_graph, "u2_to_body_note",
        "Interpolation-like GNN edges: reversed body_to_u2 connectivity. " *
        "Edge attributes reuse the fixed body-minus-grid geometry (not negated). " *
        "Not the exact application of the Immersa interpolation operator E.")
    write_attribute(grp_graph, "body_to_omega_note",
        "GNN edges connecting Lagrangian markers to vorticity-grid nodes. " *
        "Uses DeltaYang3S2 compact support and weights at Loc_omega(3). " *
        "NOT the exact Immersa interpolation or spreading operator: " *
        "those operators act on the staggered velocity grids, not omega nodes.")

    for lev in 1:grid_levels
        grp_lev = create_group(grp_graph, "level_$(lev)")

        # ── Compute all coordinates and graphs for this level ──────────────────
        Xv_ω,  Yv_ω  = grid_coord_vecs(grid, Loc_ω(3), sol.ω[lev][3], lev)
        Xv_u1, Yv_u1 = grid_coord_vecs(grid, Loc_u(1), sol.u[lev][1], lev)
        Xv_u2, Yv_u2 = grid_coord_vecs(grid, Loc_u(2), sol.u[lev][2], lev)

        omega_pts          = grid_points_flat(Xv_ω,  Yv_ω)
        u1_pts             = grid_points_flat(Xv_u1, Yv_u1)
        u2_pts             = grid_points_flat(Xv_u2, Yv_u2)
        omega_ei, omega_ea = omega_4neighbor_graph(Xv_ω, Yv_ω)

        # body-to-grid delta-kernel edges (uses delta_kernel = DeltaYang3S2())
        b2ω_ei,  b2ω_ea  = make_body_to_grid_delta_edges(
            Xv_ω,  Yv_ω,  body_pts_mat; delta = delta_kernel,
        )
        b2u1_ei, b2u1_ea = make_body_to_grid_delta_edges(
            Xv_u1, Yv_u1, body_pts_mat; delta = delta_kernel,
        )
        b2u2_ei, b2u2_ea = make_body_to_grid_delta_edges(
            Xv_u2, Yv_u2, body_pts_mat; delta = delta_kernel,
        )

        # reverse connectivity for interpolation-like branches
        u1_to_body_ei = reverse_edge_index(b2u1_ei)
        u2_to_body_ei = reverse_edge_index(b2u2_ei)

        # ── Per-level consistency assertions ───────────────────────────────────
        n_omega_nodes = size(omega_pts, 2)
        n_u1_nodes    = size(u1_pts,   2)
        n_u2_nodes    = size(u2_pts,   2)

        @assert size(b2ω_ei,  2) == size(b2ω_ea,  2)
        @assert size(b2u1_ei, 2) == size(b2u1_ea, 2)
        @assert size(b2u2_ei, 2) == size(b2u2_ea, 2)

        if size(b2ω_ei, 2) > 0
            @assert minimum(b2ω_ei[1, :]) >= 0
            @assert maximum(b2ω_ei[1, :]) <  n_ib
            @assert minimum(b2ω_ei[2, :]) >= 0
            @assert maximum(b2ω_ei[2, :]) <  n_omega_nodes
        end
        if size(b2u1_ei, 2) > 0
            @assert minimum(b2u1_ei[1, :]) >= 0
            @assert maximum(b2u1_ei[1, :]) <  n_ib
            @assert minimum(b2u1_ei[2, :]) >= 0
            @assert maximum(b2u1_ei[2, :]) <  n_u1_nodes
        end
        if size(b2u2_ei, 2) > 0
            @assert minimum(b2u2_ei[1, :]) >= 0
            @assert maximum(b2u2_ei[1, :]) <  n_ib
            @assert minimum(b2u2_ei[2, :]) >= 0
            @assert maximum(b2u2_ei[2, :]) <  n_u2_nodes
        end
        if size(b2u1_ei, 2) > 0
            @assert all(u1_to_body_ei[1, :] .== b2u1_ei[2, :])
            @assert all(u1_to_body_ei[2, :] .== b2u1_ei[1, :])
        end
        if size(b2u2_ei, 2) > 0
            @assert all(u2_to_body_ei[1, :] .== b2u2_ei[2, :])
            @assert all(u2_to_body_ei[2, :] .== b2u2_ei[1, :])
        end

        # ── Write omega graph (MVP 1 — unchanged) ─────────────────────────────
        dset_ωpts = create_dataset(grp_lev, "omega_points",     T_F,   size(omega_pts))
        dset_ωei  = create_dataset(grp_lev, "omega_edge_index", Int32, size(omega_ei))
        dset_ωea  = create_dataset(grp_lev, "omega_edge_attr",  T_F,   size(omega_ea))
        dset_ωpts[:, :] = omega_pts
        dset_ωei[:, :]  = omega_ei
        dset_ωea[:, :]  = omega_ea

        # ── Write velocity grid point arrays ───────────────────────────────────
        dset_u1pts = create_dataset(grp_lev, "u1_points", T_F, size(u1_pts))
        dset_u1pts[:, :] = u1_pts
        dset_u2pts = create_dataset(grp_lev, "u2_points", T_F, size(u2_pts))
        dset_u2pts[:, :] = u2_pts

        # ── Write body→omega edges (GNN boundary-influence edges for MVP 2) ───
        if size(b2ω_ei, 2) > 0
            d = create_dataset(grp_lev, "body_to_omega_edge_index", Int32, size(b2ω_ei))
            d[:, :] = b2ω_ei
            d = create_dataset(grp_lev, "body_to_omega_edge_attr",  T_F,   size(b2ω_ea))
            d[:, :] = b2ω_ea
        end

        # ── Write body→u1 and u1→body edges ──────────────────────────────────
        # body_to_u1: delta-kernel support at Loc_u(1)
        if size(b2u1_ei, 2) > 0
            d = create_dataset(grp_lev, "body_to_u1_edge_index", Int32, size(b2u1_ei))
            d[:, :] = b2u1_ei
            d = create_dataset(grp_lev, "body_to_u1_edge_attr",  T_F,   size(b2u1_ea))
            d[:, :] = b2u1_ea
            # u1_to_body: reversed connectivity, same attributes (body-minus-grid, not negated)
            d = create_dataset(grp_lev, "u1_to_body_edge_index", Int32, size(u1_to_body_ei))
            d[:, :] = u1_to_body_ei
            d = create_dataset(grp_lev, "u1_to_body_edge_attr",  T_F,   size(b2u1_ea))
            d[:, :] = b2u1_ea
        end

        # ── Write body→u2 and u2→body edges ──────────────────────────────────
        # body_to_u2: delta-kernel support at Loc_u(2)
        if size(b2u2_ei, 2) > 0
            d = create_dataset(grp_lev, "body_to_u2_edge_index", Int32, size(b2u2_ei))
            d[:, :] = b2u2_ei
            d = create_dataset(grp_lev, "body_to_u2_edge_attr",  T_F,   size(b2u2_ea))
            d[:, :] = b2u2_ea
            # u2_to_body: reversed connectivity, same attributes (body-minus-grid, not negated)
            d = create_dataset(grp_lev, "u2_to_body_edge_index", Int32, size(u2_to_body_ei))
            d[:, :] = u2_to_body_ei
            d = create_dataset(grp_lev, "u2_to_body_edge_attr",  T_F,   size(b2u2_ea))
            d[:, :] = b2u2_ea
        end

        @info "Graph edges — level $lev" omega_edges=size(omega_ei,2) body_to_omega=size(b2ω_ei,2) body_to_u1=size(b2u1_ei,2) u1_to_body=size(u1_to_body_ei,2) body_to_u2=size(b2u2_ei,2) u2_to_body=size(u2_to_body_ei,2)
    end

    # ── Summary ────────────────────────────────────────────────────────────────
    println("\n=== HDF5 structure ===")
    println("keys(file):                     ", keys(file))
    println("keys(file[\"body\"]):             ", keys(file["body"]))
    println("keys(file[\"snapshots\"]):        ", keys(file["snapshots"]))
    println("keys(file[\"snapshots/coords\"]): ", keys(file["snapshots/coords"]))
    println("keys(file[\"graph\"]):            ", keys(file["graph"]))

    println("\n=== Edge dataset dimensions ===")
    println("  body/edge_index : ", size(file["body/edge_index"]))
    println("  body/edge_attr  : ", size(file["body/edge_attr"]))
    for lev in 1:grid_levels
        println("  --- level $lev ---")
        for name in [
            "omega_edge_index",     "omega_edge_attr",
            "body_to_omega_edge_index", "body_to_omega_edge_attr",
            "body_to_u1_edge_index", "body_to_u1_edge_attr",
            "u1_to_body_edge_index", "u1_to_body_edge_attr",
            "body_to_u2_edge_index", "body_to_u2_edge_attr",
            "u2_to_body_edge_index", "u2_to_body_edge_attr",
        ]
            path = "graph/level_$(lev)/$(name)"
            if haskey(file, path)
                println("  $(path): ", size(file[path]))
            end
        end
    end

end  # h5open

# ── Contour animation ─────────────────────────────────────────────────────────
mkpath(OUTDIR)
h5open(soln_path, "r") do file
    t_snap = read(file["snapshots/t"])
    ω_ds   = file["snapshots/omega"]

    xvec_ω = read(file["snapshots/coords/omega/x_coords"])[:, 1]
    yvec_ω = read(file["snapshots/coords/omega/y_coords"])[:, 1]

    bpts = read(file["body/points"])
    xb   = bpts[1, :]
    yb   = bpts[2, :]

    ωlim = 12.0
    anim = Animation()

    @showprogress desc = "animating" for i in eachindex(t_snap)
        z      = ω_ds[:, :, 1, i]
        zplot  = copy(z)
        fm     = isfinite.(zplot)
        zplot[fm]  .= clamp.(zplot[fm], -ωlim, ωlim)
        zplot[.!fm] .= 0.0

        p = contourf(
            xvec_ω, yvec_ω, zplot';
            aspect_ratio = :equal,
            colormap     = :seaborn_icefire_gradient,
            levels       = 18,
            lw           = 0,
            legend       = false,
            clim         = (-ωlim, ωlim),
            xlim         = (-1.0, 6.0),
            ylim         = (-2.0, 2.0),
            framestyle   = :box,
        )

        scatter!(
            p, xb, yb;
            color             = :white,
            markerstrokecolor = :black,
            markersize        = 3,
            label             = false,
        )

        title!(p, @sprintf("t = %.2f", t_snap[i]))
        frame(anim, p)
    end

    gif_path = joinpath(OUTDIR, "$(CASE)_vorticity_contour_markers.gif")
    gif(anim, gif_path; fps = 24)
    @info "Saved animation" path = gif_path
end
