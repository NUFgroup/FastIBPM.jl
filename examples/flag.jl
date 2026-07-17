# -*- coding: utf-8 -*-
# %%
using Immersa
using StaticArrays
using ProgressMeter
#using CairoMakie
using Plots
using DelimitedFiles
using OffsetArrays
using LinearAlgebra
using HDF5
using Peaks
using Statistics
using Printf

CairoMakie.activate!(; type="svg")

# %%
Re = 1000.0  # Reynolds number
h = 0.003  # grid cell size
dt = 6e-4  # time step


function u(t)
    s = clamp(t / 0.1, 0, 1)
    s * (2 - s)
end
u0 = UniformFlow(t -> SA[u(t), 0.0])

gridlims = SA[-0.2 1.8; -0.3 0.3]
grid = Grid(;
    h,
    n=@.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)),
    x0=gridlims[:, 1],  # first grid cell position
    levels=5,  # number of grid levels
)

# %%
L = 1.0  # length of the flag
n_ib = 1 + round(Int, L / (2h))
m = 0.075
kb = 0.0001
ke = 1e3

body = let xs = range(0, L, n_ib)
    GeometricNonlinearBody(;
        xref=[SA[x, 0] for x in xs],
        ds0=fill(step(xs), n_ib),
        m=fill(m, n_ib),
        kb=fill(kb, n_ib),
        ke=fill(ke, n_ib),
        bcs=[StructureBC{Float64}(i, (_, _, _) -> 0.0) for i in 1:3],
    )
end

prob = IBProblem(grid, body, Re, u0);

# %%
function solution(file; tf, snapshot_freq)
    sol = CNAB(prob; dt, coupler_args=(; bicgstabl_args=(;)), delta=Immersa.DeltaYang3S2())

    θ = deg2rad(1)
    let x = range(0, L, n_ib)
        χ = sol.coupler.state.χ
        @. χ[2:3:end] = sin(θ) * x
        @. χ[3:3:end] = θ
    end

    i_all = 1:(1+round(Int, tf/dt))
    n_all = length(i_all)

    i_snapshot = i_all[1:snapshot_freq:end]
    n_snapshot = length(i_snapshot)

    g_all = create_group(file, "all")
    t_all = create_dataset(g_all, "t", Float64, (n_all,))
    CL = create_dataset(g_all, "CL", Float64, (n_all,))
    CD = create_dataset(g_all, "CD", Float64, (n_all,))
    x_tip = create_dataset(g_all, "x_tip", Float64, (2, n_all))

    g_snapshot = create_group(file, "snapshots")
    t_snapshot = create_dataset(g_snapshot, "t", Float64, (n_snapshot,))

    omega = create_dataset(
        g_snapshot, "omega", Float64, (size(sol.state.ω[1][3])..., grid.levels, n_snapshot)
    )
    write_attribute(omega, "firstindex", collect(first.(axes(sol.state.ω[1][3]))))

    x_ib = create_dataset(g_snapshot, "x_ib", Float64, (2, n_ib, n_snapshot))

    @showprogress desc = "solving" for _ in i_all
        step!(sol)

        f = surface_force_sum(sol)

        t_all[sol.i] = sol.t
        CD[sol.i] = 2 * f[1]
        CL[sol.i] = 2 * f[2]
        x_tip[:, sol.i] = sol.points.x[end]

        if sol.i in i_snapshot
            i = 1 + (sol.i - first(i_snapshot)) ÷ step(i_snapshot)

            t_snapshot[i] = sol.t

            for level in eachindex(sol.state.ω)
                omega[:, :, level, i] = OffsetArrays.no_offset_view(sol.state.ω[level][3])
            end

            x_ib[:, :, i] = reinterpret(reshape, Float64, sol.points.x)
        end
    end
end

# %%
soln_path = "flag.h5"

if isfile(soln_path)
    @info "File already exists" soln_path
else
    h5open(soln_path, "cw") do soln
        # solution(soln; tf=0.5, snapshot_freq=100)
        solution(soln; tf=5.0, snapshot_freq=100)
    end
end

# %%
results = h5open(soln_path, "r") do soln
    (; t=read(soln["all/t"]), CL=read(soln["all/CL"]), x_tip=read(soln["all/x_tip"]))
end;

# %%
i_end = length(results.t)
i_start = i_end - round(Int, 4 / dt)
y_extrema = let y = @view results.x_tip[2, Base.IdentityUnitRange(i_start:end)]
    w = 10
    (minima=argminima(y, w), maxima=argmaxima(y, w))
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="t")
    colors = Makie.wong_colors()

    r = 1:10:length(results.t)
    @views lines!(
        results.t[r], results.x_tip[2, r], color=colors[1], label="tip displacement"
    )
    @views lines!(results.t[r], results.CL[r], color=colors[2], linewidth=1, label="CL")

    for (i, inds) in enumerate(y_extrema)
        scatter!(results.t[inds], results.x_tip[2, inds]; color=colors[1])
    end

    axislegend(ax; position=:lt)

    fig
end


# Comparison with Goza et al. data (2017)
let
    goza = readdlm("examples/figures/flag/Goza2017_data.csv", ',')
    t_goza = goza[:, 1]
    y_goza = goza[:, 2]

    r = 1:10:length(results.t)
    p = @views plot(results.t[r], results.x_tip[2, r];
                    xlabel="t", label="tip displacement", legend=:topleft)
    #@views plot!(p, results.t[r], results.CL[r]; linewidth=1, label="CL")

    for inds in y_extrema
        @views scatter!(p, results.t[inds], results.x_tip[2, inds]; label="", color=1)
    end

    scatter!(p, t_goza, y_goza;
             label="Goza 2017", color=:black, markersize=3, markerstrokewidth=0)

    p
end

# %%
y_limits = map(i -> mean(results.x_tip[2, i]), y_extrema)
y_mean = (y_limits[1] + y_limits[2]) / 2
y_ampl = (y_limits[2] - y_limits[1]) / 2
@printf "y_ampl = %#.3f\n" y_ampl

# %%
y_period = results.t[y_extrema.maxima] |> diff |> mean
y_freq = 1 / y_period
@printf "y_freq = %#.2f\n" y_freq

# %%
h5open(soln_path, "r") do soln
    t = read(soln["snapshots/t"])
    ω = soln["snapshots/omega"]
    ω_start = read_attribute(ω, "firstindex")
    ω_axes = map((i0, s) -> @.(i0 + (0:(s-1))), Tuple(ω_start), size(ω))
    x_ib = read(soln["snapshots/x_ib"])

    fig = Figure(; size=(1000, 300))

    ax = Axis(fig[1, 1]; limits=((-0.2, 2.4), (-0.35, 0.35)), aspect=DataAspect())

    ωi = Observable(ω[:, :, :, 1])
    ωlim = 15.0
    hm = map(grid.levels:-1:1) do level
        (x, y) = coord(grid, Loc_ω(3), ω_axes, level)
        ωl = @lift @view $ωi[:, :, level]
        heatmap!(ax, x, y, ωl; colormap=:coolwarm, colorrange=(-ωlim, ωlim))
    end

    ti = Observable(0.0)
    label = @lift @sprintf "t=%05.2f" $ti
    font = Makie.to_font("DejaVu Sans Mono")
    text!(
        ax,
        0,
        1;
        text=label,
        align=(:left, :top),
        offset=(4, -2),
        space=:relative,
        font=font,
    )

    x = Observable(x_ib[:, :, 1])
    lines!(ax, x; color=:black, linewidth=2)

    Colorbar(fig[1, 2], hm[1]; label="vorticity")

    Record(fig, eachindex(t); fps=30) do i
        ti[] = t[i]
        ωi[] = ω[:, :, :, i]
        x[] = @view x_ib[:, :, i]
    end
end


# For saving the gif
using Plots
h5open(soln_path, "r") do soln
    t = read(soln["snapshots/t"])
    ω = soln["snapshots/omega"]
    ω_start = read_attribute(ω, "firstindex")
    nx, ny, nlev, nt = size(ω)
    xidx = ω_start[1]:(ω_start[1] + nx - 1)
    yidx = ω_start[2]:(ω_start[2] + ny - 1)
    x_ib = read(soln["snapshots/x_ib"])

    ωlim = 15.0

    anim = Animation()
    @showprogress for i in eachindex(t)
        p = plot(legend=false, colorbar=true, colorbar_title="ω",
                 aspect_ratio=:equal,
                 framestyle=:box,
                 foreground_color_axis=:black,
                 top_margin=1Plots.mm,
                 xlim=(gridlims[1, 1], gridlims[1, 2]),
                 ylim=(gridlims[2, 1], gridlims[2, 2]),
                 )

        for lev in nlev:-1:1
            X, Y = coord(grid, Loc_ω(3), (xidx, yidx), lev)
            xvec, yvec = X[:, 1], Y[:, 1]
            z = ω[:, :, lev, i]

            zplot = copy(z)
            finite_mask = isfinite.(zplot)
            zplot[finite_mask] .= clamp.(zplot[finite_mask], -ωlim, ωlim)

            contourf!(xvec, yvec, zplot';
                aspect_ratio=:equal,
                colormap=:bwr,
                levels=30,
                lw=0,
                legend=false,
                clim=(-ωlim, ωlim),
                framestyle=:box,
                foreground_color_axis=:black,
            )
        end

        plot!(p, x_ib[1, :, i], x_ib[2, :, i]; color=:black, lw=2, label="", framestyle=:box)

        title!(p, @sprintf("t = %.3f", t[i]))
        frame(anim, p)
    end

    gif(anim, replace(soln_path, ".h5" => "_vorticity.gif"); fps=20)
end
