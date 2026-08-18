# -*- coding: utf-8 -*-
# %%
using Immersa
using Immersa: IMAP, rot!, grid_view
using KernelAbstractions: get_backend
using StaticArrays
using ProgressMeter
using OffsetArrays
using LinearAlgebra
using HDF5
using Printf
using Plots

# Flow around a stationary cylinder using IMAP — the manifold-projection
# formulation. IMAP keeps the primitive-variable structure of `cylinder_PV.jl`
# (Taira & Colonius 2007) but eliminates the immersed-boundary force: no-slip is
# enforced by projecting onto the constraint manifold Rᵀu = 0 rather than by
# solving for an f that produces it.
#
# WHAT DIFFERS FROM `cylinder_PV.jl`
#
#   * The unknown is (u, p) — not (u, p, f). The per-step linear solve is the
#     pressure system alone, `#cells` unknowns instead of `#cells + N·n_b`. The
#     only body-sized work is one Cholesky factorization of RᵀR = E Eᵀ, done once
#     at construction and reused for the whole run (the body is static).
#
#   * UNIFORM GRID ONLY. The paper's stretched Case-A mesh is not available here:
#     the stretched operators are mass-weighted, and the projector would have to
#     be made orthogonal in the mass inner product to match. `CNAB` refuses a
#     `StretchedGrid` with an explanatory error rather than running something
#     subtly wrong. The consequence is blockage: the lateral boundaries sit at
#     ±3 instead of ±30, which biases C_D upward. See the validation note below.
#
#   * The initial condition is projected onto the manifold at construction.
#     This is required, not cosmetic — see the no-slip diagnostic below.
#
#   * `noslip_residual(sol)` is logged every step. For IMAP this is the headline
#     diagnostic: no-slip is *conserved* (Rᵀu^{n+1} = Rᵀuⁿ holds exactly), so it
#     should sit at machine precision — around 1e-15 — for the entire run, not
#     merely be small. Anything that grows means the constraint is leaking.
#
# VALIDATION. Re = 40 is steady (no shedding below Re ≈ 47). The reference values
# of Taira & Colonius Table 2 are C_D = 1.54 and l/d = 2.30, but those are for the
# far-field Case-A mesh; on this compact uniform domain both solvers overpredict
# C_D through blockage. The meaningful check is therefore IMAP against IBPM-PV on
# an *identical* grid — run `cylinder_PV.jl` with `use_stretched = false` and the
# same h/dt/extent to compare directly.

# %%
const _FILEPATH = let f = @__FILE__
    isempty(f) ? PROGRAM_FILE : f
end
const CASE = isempty(_FILEPATH) ? "session" : first(splitext(basename(_FILEPATH)))
const SRCDIR = isempty(_FILEPATH) ? pwd() : dirname(_FILEPATH)
const OUTDIR = joinpath(SRCDIR, "figures", CASE)
mkpath(OUTDIR)

# %%
h = 0.05                      # uniform cell size
gridlims = SA[-3.0 6.0; -3.0 3.0]
grid = Grid(;
    h,
    n=@.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)),
    x0=gridlims[:, 1],
    levels=1,                 # IMAP is single-level: there is no multidomain here
)

# Zoom window for the vorticity animation.
plotlims = SA[-2.0 5.0; -2.5 2.5]

# %%
r = 0.5              # cylinder radius (diameter d = 1)
S = 2π * r           # circumference
n_ib = round(Int, S / h)   # Δs ≈ Δx  (Taira & Colonius §3.3)
ds = S / n_ib

body = let
    x = map(range(0, 2π, n_ib + 1)[1:(end-1)]) do θ
        r * SA[cos(θ), sin(θ)]
    end
    StaticBody(x, fill(ds, n_ib))
end;

# %%
# Δt is bounded by the Bᴺ Taylor series, which needs a/h² = Δt/(2Re·h²) ≲ 1 — the
# paper's νΔt/Δx² ≲ 1 — and by the explicit AB2 convection (CFL = u∞Δt/h). Both
# are comfortable here: a/h² = 0.05, CFL = 0.2.
dt = 0.01
Re = 40.0
u0 = UniformFlow(t -> SA[1.0, 0.0])
prob = IBProblem(grid, body, Re, u0, IMAP());   # <-- manifold-projection formulation

@info "IMAP cylinder" grid=Tuple(grid.n) cells=prod(grid.n) n_ib dt taylor_ratio=dt/(2Re*h^2) CFL=dt/h

# %%
# Fresh vorticity ω = ∇×u from the physical velocity, for output/visualization.
function vorticity!(ω, sol)
    for i in eachindex(ω)
        fill!(ω[i], 0)
    end
    g = sol.prob.grid
    v = grid_view(ω, g, Loc_ω, ExcludeBoundary())
    rot!(v, sol.state.u_full; h=g.h)
    ω
end

# Recirculation length l/d: the first sign change of u_x along the centreline
# behind the body, measured from the rear of the cylinder and scaled by d = 2r.
# (Taira & Colonius Table 2 report l/d = 2.30 at Re = 40 on the far-field mesh.)
function wake_length(sol)
    g = sol.prob.grid
    u = sol.state.u_full[1]
    R = CartesianIndices(cell_axes(g, Loc_u(1), ExcludeBoundary()))
    i0, i1 = first(R)[1], last(R)[1]
    # the two rows of u_x straddling y = 0
    js = sort(unique(I[2] for I in R);
              by=j -> abs(coord(g, Loc_u(1), CartesianIndex(i0, j))[2]))[1:2]
    xs, us = Float64[], Float64[]
    for i in i0:i1
        x = coord(g, Loc_u(1), CartesianIndex(i, js[1]))[1]
        x > r || continue
        push!(xs, x)
        push!(us, (u[i, js[1]] + u[i, js[2]]) / 2)
    end
    k = findfirst(m -> us[m] < 0 && us[m+1] >= 0, 1:(length(us)-1))
    k === nothing && return NaN
    xr = xs[k] + (xs[k+1] - xs[k]) * (-us[k]) / (us[k+1] - us[k])
    (xr - r) / (2r)
end

# %%
function solution(file; tf, snapshot_freq)
    T = Float64
    sol = CNAB(prob; dt, delta=Immersa.DeltaYang3S2())
    st = sol.state

    # Re = 40 is steady (no vortex shedding below Re ≈ 47), so NO perturbation is
    # injected — the flow settles to a steady wake.
    #
    # The first step still carries the usual impulsive-start pressure impulse
    # (p ~ O(1/Δt), so C_D is O(100) for one step) even though the initial
    # condition is projected onto the manifold: projecting fixes no-slip, not the
    # sudden onset of the free stream. IBPM-PV shows the same spike. Ignore the
    # first few samples when reading the C_D history.
    @info "initial no-slip residual" resid = noslip_residual(sol)

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
    # The IMAP-specific diagnostic: no-slip is conserved, so this must stay at
    # roundoff for the whole run rather than merely being small.
    resid = create_dataset(all_group, "noslip", T, (n_all,))

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

    @showprogress desc = "solving (IMAP)" for _ in 1:n_all
        step!(sol)

        # `surface_force_sum` works unchanged: `recover_force!` reconstructs the
        # boundary force IMAP never solves for, in the same normalization IBPM
        # uses. C = 2·F for ½ρU²d = 1 (d = 1, ρ = 1, U∞ = 1).
        f = surface_force_sum(sol)
        t_all[sol.i] = sol.t
        Cd[sol.i] = 2 * f[1]
        Cl[sol.i] = 2 * f[2]
        resid[sol.i] = noslip_residual(sol)

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

    @info "final state" Cd = Cd[end] l_over_d = wake_length(sol) noslip = resid[end]
end

# %%
soln_path = joinpath(SRCDIR, "$(CASE).h5")

if isfile(soln_path)
    @info "File already exists" soln_path
else
    h5open(soln_path, "cw") do file
        # The pressure solve is an unpreconditioned CG over the whole grid, so a
        # run to steady state is long (of order an hour here). Reduce `tf` for a
        # quick qualitative look; t ≈ 25 is needed for C_D to settle at Re = 40.
        solution(file; tf=25, snapshot_freq=50)
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
        title!(@sprintf("IMAP  Re=%d   t = %.2f", Re, t[i]))
        frame(anim, p)
    end
    gif(anim, joinpath(OUTDIR, "$(CASE)_vorticity.gif"); fps=15)
end

# %%
results = h5open(soln_path, "r") do soln
    (;
        t=read(soln["all/t"]),
        Cd=read(soln["all/Cd"]),
        Cl=read(soln["all/Cl"]),
        noslip=read(soln["all/noslip"]),
    )
end;

# Drag / lift history
p = plot(; xlabel="t", ylabel="", framestyle=:box, legend=:right)
plot!(p, results.t, results.Cd; color=:red, label="Cd")
plot!(p, results.t, results.Cl; color=:blue, label="Cl")
hline!(p, [1.54]; color=:black, linestyle=:dash, label="paper Cd (Re=40, far field)")
savefig(p, joinpath(OUTDIR, "$(CASE)_Cd_Cl.png"))

# The constraint diagnostic. A flat line at ~1e-15 is the point of IMAP: the
# no-slip condition is conserved by construction, not re-established each step.
q = plot(;
    xlabel="t", ylabel="‖Rᵀu − u_B‖∞", framestyle=:box, legend=false, yscale=:log10,
)
plot!(q, results.t, max.(results.noslip, 1e-18); color=:purple)
savefig(q, joinpath(OUTDIR, "$(CASE)_noslip.png"))

@info "final Cd" Cd_final = results.Cd[end] paper_ref_far_field = 1.54
@info "no-slip residual" max = maximum(results.noslip)
