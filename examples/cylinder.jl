# -*- coding: utf-8 -*-
# %%
using FastIBPM
using StaticArrays
using ProgressMeter
using CairoMakie
using OffsetArrays
using LinearAlgebra
using HDF5
using Peaks
using Statistics
using Printf

CairoMakie.activate!(; type="svg")

# %%
h = 0.02  # grid cell size
gridlims = SA[-1.0 3.0; -2.0 2.0]
grid = Grid(;
    h, n=@.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)), x0=gridlims[:, 1], levels=5
)

# %%
r = 0.5  # cylinder radius
S = 2π * r  # cylinder circumference
n_ib = round(Int, S / (2 * h))  # number of immersed boundary points
ds = S / n_ib  # arclength delta at each point

body = let
    x = map(range(0, 2π, n_ib + 1)[1:end-1]) do θ
        r * SA[cos(θ), sin(θ)]
    end
    StaticBody(x, fill(ds, n_ib))
end;

# %%
dt = 0.002
Re = 200.0
u0 = UniformFlow(t -> SA[1.0, 0.0])
prob = IBProblem(grid, body, Re, u0);

# %%
function solution(file; tf, snapshot_freq)
    T = Float64
    sol = CNAB(prob; dt, delta=FastIBPM.DeltaYang3S2())

    # Perturbation to induce vortex shedding
    map!(sol.ω[1][3], CartesianIndices(sol.ω[1][3])) do I
        x = coord(grid, Loc_ω(3), I)
        p = x - SA[-0.75, 0]
        r = 0.25
        0.5 * (1 - clamp(norm(p) / r, 0, 1))
    end
    apply_vorticity!(sol)

    i_all = 1:1+round(Int, tf / dt)
    n_all = length(i_all)

    i_snapshot = i_all[1:snapshot_freq:end]
    n_snapshot = length(i_snapshot)

    all_group = create_group(file, "all")
    t_all = create_dataset(all_group, "t", T, (n_all,))
    CL = create_dataset(all_group, "CL", T, (n_all,))
    CD = create_dataset(all_group, "CD", T, (n_all,))

    snapshot_group = create_group(file, "snapshots")
    t_snapshot = create_dataset(snapshot_group, "t", T, (n_snapshot,))
    ω = create_dataset(
        snapshot_group, "omega", T, (size(sol.ω[1][3])..., grid.levels, n_snapshot)
    )
    write_attribute(ω, "firstindex", collect(first.(axes(sol.ω[1][3]))))

    @showprogress desc = "solving" for _ in 0:round(Int, tf / dt)
        step!(sol)

        f = surface_force_sum(sol)
        t_all[sol.i] = sol.t
        CD[sol.i] = 2 * f[1]
        CL[sol.i] = 2 * f[2]

        if sol.i in i_snapshot
            i = 1 + (sol.i - first(i_snapshot)) ÷ step(i_snapshot)
            t_snapshot[i] = sol.t
            for level in eachindex(sol.ω)
                ω[:, :, level, i] = OffsetArrays.no_offset_view(sol.ω[level][3])
            end
        end
    end
end

# %%
soln_path = "cylinder.h5"

if isfile(soln_path)
    @info "File already exists" soln_path
else
    h5open(soln_path, "cw") do file
        solution(file; tf=80.0, snapshot_freq=100)
    end
end

# %%
h5open(soln_path, "r") do file
    fig = Figure(; size=(800, 300))
    ax = Axis(fig[1, 1]; limits=((-2, 8), (-2, 2)), aspect=DataAspect())

    t = file["snapshots/t"][:]
    ω = file["snapshots/omega"]
    ω_start = read_attribute(ω, "firstindex")
    ω_axes = map((i0, s) -> @.(i0 + (0:s-1)), Tuple(ω_start), size(ω))

    ωi = Observable(ω[:, :, :, 1])
    ωlim = 5.0
    hm = map(grid.levels:-1:1) do level
        (x, y) = coord(grid, Loc_ω(3), ω_axes, level)
        ωl = @lift @view $ωi[:, :, level]
        heatmap!(ax, x, y, ωl; colormap=:coolwarm, colorrange=(-ωlim, ωlim))
    end

    ti = Observable(0.0)
    label = @lift @sprintf "t=%04.1f" $ti
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

    poly!(ax, body.x; color=:transparent, strokecolor=:black, strokewidth=3)

    Colorbar(fig[1, 2], hm[1]; label="vorticity")

    Record(fig, eachindex(t); fps=30) do i
        ti[] = t[i]
        ωi[] = ω[:, :, :, i]
    end
end

# %%
results = h5open(soln_path, "r") do soln
    (; t=read(soln["all/t"]), CL=read(soln["all/CL"]), CD=read(soln["all/CD"]))
end;

# %%
i_start = 1 + round(Int, 50 / dt)

peaks = map((; CL=results.CL, CD=results.CD)) do y
    z = @view y[Base.IdentityUnitRange(i_start:end)]
    (findminima(z), findmaxima(z))
end

periods = map(x -> mean(diff(results.t[x[2].indices])), peaks)

Pr = 1 / periods.CL

oscillations = map(peaks) do p
    (a, b) = map(x -> mean(x.heights), p)
    ((a + b) / 2, (b - a) / 2)
end

# %%
let
    fig = Figure()
    ax = Axis(fig[1, 1]; limits=(nothing, (-2, 2)))

    for f in (:CL, :CD)
        C = results[f]
        p = peaks[f]
        o = oscillations[f]

        lines!(ax, results.t, C)

        i = [(x.indices for x in p)...;]
        scatter!(ax, results.t[i], C[i])

        hlines!(ax, @.(o[1] + [-1, 1] * o[2]); linestyle=:dash)
    end

    fig
end
