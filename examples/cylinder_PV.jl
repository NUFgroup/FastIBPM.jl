# -*- coding: utf-8 -*-
# %%
using Immersa
using Immersa: IBPM, rot!, grid_view
using KernelAbstractions: get_backend
using StaticArrays
using ProgressMeter
using OffsetArrays
using LinearAlgebra
using HDF5
using Printf
using Plots

# Flow around a stationary cylinder using the PRIMITIVE-VARIABLE (IBPM) formulation
# of Taira & Colonius (2007) — the projection-approach counterpart of the
# streamfunction-vorticity `cylinder.jl`. Parameters follow the paper's Table 1
# Case A (Δx_min = 0.04, Δt = 0.005, n_b ≈ 78) at Re = 40 (steady-state validation;
# Table 2 reference: l/d = 2.30, C_D = 1.54).
#
# GRID: set `use_stretched` below.
#   * STRETCHED (default): a `StretchedGrid` with a uniform Δx_min core around the
#     cylinder that grows geometrically out to ±30 — the paper's Case A mesh, ~150×150
#     cells (a uniform ±30 mesh at Δx_min would need ~1500²). This places the far-field
#     boundaries far enough away that the Dirichlet free stream no longer causes the
#     blockage that biases C_D upward. Tune `core`/`growth`/`extent` freely below.
#   * UNIFORM: a compact uniform `Grid`. Cheaper but the nearby lateral boundaries
#     cause blockage that biases C_D upward — for a quick qualitative demo only.
#
# Other notes:
#   * The outflow uses the paper's convective condition ∂u/∂t + u∞ ∂u/∂x = 0.
#   * The modified-Poisson solve is an unpreconditioned CG over the whole grid, so
#     each step is costly and reaching steady state is a long run. `tf` below is set
#     modestly for a demonstration — increase it (e.g. tf = 30) for validation.

# %%
const _FILEPATH = let f = @__FILE__
    isempty(f) ? PROGRAM_FILE : f
end
const CASE = isempty(_FILEPATH) ? "session" : first(splitext(basename(_FILEPATH)))
const SRCDIR = isempty(_FILEPATH) ? pwd() : dirname(_FILEPATH)
const OUTDIR = joinpath(SRCDIR, "figures", CASE)
mkpath(OUTDIR)

# %%
use_stretched = true   # ← switch: paper Case-A stretched grid, or compact uniform

h = 0.04  # Δx_min: the uniform-core (stretched) / uniform (compact) cell size

grid = if use_stretched
    # Paper Case A: a uniform Δx_min core around the cylinder that grows geometrically
    # out to ±30 (~150×150 cells; a uniform ±30 mesh would need ~1500²). The body
    # (r = 0.5) must lie inside `core`; tune these freely.
    StretchedGrid(;
        dx_min = h,
        core   = SA[-1.0 1.0; -1.0 1.0],    # uniform region around the body
        growth = 1.085,                      # geometric cell-growth ratio (→ n≈150)
        extent = SA[-30.0 30.0; -30.0 30.0], # far-field extent
    )
else
    # Compact uniform domain (blockage-biased; qualitative demo).
    gridlims = SA[-3.0 6.0; -3.0 3.0]
    Grid(; h, n=@.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)), x0=gridlims[:, 1], levels=1)
end

# Zoom window for the vorticity animation (the full ±30 domain would dwarf the body).
plotlims = SA[-2.0 8.0; -3.0 3.0]

# %%
r = 0.5              # cylinder radius (diameter d = 1)
S = 2π * r           # circumference
n_ib = round(Int, S / h)   # Δs ≈ Δx  (Taira & Colonius §3.3);  ≈ 78 for Case A
ds = S / n_ib

body = let
    x = map(range(0, 2π, n_ib + 1)[1:(end-1)]) do θ
        r * SA[cos(θ), sin(θ)]
    end
    StaticBody(x, fill(ds, n_ib))
end;

# %%
dt = 0.005
Re = 40.0
u0 = UniformFlow(t -> SA[1.0, 0.0])
prob = IBProblem(grid, body, Re, u0, IBPM());   # <-- primitive-variable formulation

# %%
# Fresh vorticity ω = ∇×u from the physical velocity, for output/visualization.
function vorticity!(ω, sol)
    for i in eachindex(ω)
        fill!(ω[i], 0)
    end
    g = sol.prob.grid
    v = grid_view(ω, g, Loc_ω, ExcludeBoundary())
    g isa StretchedGrid ? rot!(v, sol.state.u_full, g) : rot!(v, sol.state.u_full; h=g.h)
    ω
end

# %%
function solution(file; tf, snapshot_freq)
    T = Float64
    sol = CNAB(prob; dt, delta=Immersa.DeltaYang3S2())
    st = sol.state

    # Re = 40 is steady (no vortex shedding below Re ≈ 47), so NO perturbation is
    # injected — the flow settles to a steady wake from the free-stream start.

    n_all = 1 + round(Int, tf / dt)
    i_snapshot = 1:snapshot_freq:n_all
    n_snapshot = length(i_snapshot)

    body_group = create_group(file, "body")
    create_dataset(body_group, "points", T, (2, length(body.x)))[:, :] = reduce(hcat, body.x)
    create_dataset(body_group, "lengths", T, (length(body.ds),))[:] = body.ds

    all_group = create_group(file, "all")
    t_all = create_dataset(all_group, "t", T, (n_all,))
    Cd = create_dataset(all_group, "Cd", T, (n_all,))
    Cl = create_dataset(all_group, "Cl", T, (n_all,))

    ωbuf = grid_zeros(get_backend(st.φ), grid, Loc_ω)

    snap = create_group(file, "snapshots")
    t_snapshot = create_dataset(snap, "t", T, (n_snapshot,))
    ux = create_dataset(snap, "ux", T, (size(st.u_full[1])..., n_snapshot))
    uy = create_dataset(snap, "uy", T, (size(st.u_full[2])..., n_snapshot))
    pr = create_dataset(snap, "p", T, (size(st.φ)..., n_snapshot))
    ωd = create_dataset(snap, "omega", T, (size(ωbuf[3])..., n_snapshot))
    write_attribute(ωd, "firstindex", collect(first.(axes(ωbuf[3]))))
    let (X, Y) = coord(grid, Loc_ω(3), (UnitRange.(axes(ωbuf[3]))...,))
        create_dataset(snap, "x_coords", T, (length(X[:, 1]),))[:] = X[:, 1]
        create_dataset(snap, "y_coords", T, (length(Y[:, 1]),))[:] = Y[:, 1]
    end

    @showprogress desc = "solving (IBPM)" for _ in 1:n_all
        step!(sol)

        # surface_force_sum uses the IBPM-specific f̃ factor (-hᴺ); C = 2·F for
        # ½ρU²d = 1 (d = 1, ρ = 1, U∞ = 1).
        f = surface_force_sum(sol)
        t_all[sol.i] = sol.t
        Cd[sol.i] = 2 * f[1]
        Cl[sol.i] = 2 * f[2]

        if sol.i in i_snapshot
            i = 1 + (sol.i - first(i_snapshot)) ÷ step(i_snapshot)
            t_snapshot[i] = sol.t
            ux[:, :, i] = OffsetArrays.no_offset_view(st.u_full[1])
            uy[:, :, i] = OffsetArrays.no_offset_view(st.u_full[2])
            pr[:, :, i] = OffsetArrays.no_offset_view(st.φ)
            vorticity!(ωbuf, sol)
            ωd[:, :, i] = OffsetArrays.no_offset_view(ωbuf[3])
        end
    end
end

# %%
soln_path = joinpath(SRCDIR, "$(CASE).h5")

if isfile(soln_path)
    @info "File already exists" soln_path
else
    h5open(soln_path, "cw") do file
        # Modest tf for a demo run; increase (e.g. tf = 30) to approach steady state.
        solution(file; tf=0.2, snapshot_freq=20)
    end
end

# %%
# Vorticity animation
h5open(soln_path, "r") do file
    t = read(file["snapshots/t"])
    ω = file["snapshots/omega"]
    x = read(file["snapshots/x_coords"])
    y = read(file["snapshots/y_coords"])
    nt = length(t)

    ωlim = 5.0
    θ = range(0, 2π; length=200)
    cx, cy = r .* cos.(θ), r .* sin.(θ)

    anim = Animation()
    @showprogress for i in 1:nt
        p = plot(; legend=false, aspect_ratio=:equal, framestyle=:box)
        heatmap!(
            x, y, ω[:, :, i]';
            colormap=:bwr, clim=(-ωlim, ωlim),
            xlim=(plotlims[1, 1], plotlims[1, 2]), ylim=(plotlims[2, 1], plotlims[2, 2]),
        )
        plot!(Shape(cx, cy); color=:gray, lw=0)
        title!(@sprintf("Re=%d   t = %.2f", Re, t[i]))
        frame(anim, p)
    end
    gif(anim, joinpath(OUTDIR, "$(CASE)_vorticity.gif"); fps=15)
end

# %%
# Drag / lift history
results = h5open(soln_path, "r") do soln
    (; t=read(soln["all/t"]), Cd=read(soln["all/Cd"]), Cl=read(soln["all/Cl"]))
end;

p = plot(; xlabel="t", ylabel="", framestyle=:box, legend=:right)
plot!(p, results.t, results.Cd; color=:red, label="Cd")
plot!(p, results.t, results.Cl; color=:blue, label="Cl")
hline!(p, [1.54]; color=:black, linestyle=:dash, label="paper Cd (Re=40)")
savefig(p, joinpath(OUTDIR, "$(CASE)_Cd_Cl.png"))

@info "final Cd" Cd_final = results.Cd[end] paper_ref = 1.54
