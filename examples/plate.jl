# -*- coding: utf-8 -*-
# %%
using FastIBPM
using StaticArrays
using ProgressMeter
using OffsetArrays
using LinearAlgebra
using HDF5
using Peaks
using Statistics
using Printf
using Plots
#using CairoMakie

# This example simulates flow around a plate at Re=400 at θ = 12 deg. It uses a uniform inflow
# condition and perturbs the initial vorticity field to induce vortex shedding.
# The simulation data is saved to an HDF5 file, which is then used to create a
# visualization of the vorticity field and plots of the lift and drag coefficients. 
# You can use the Makie backend of your choice; here we use the Plots backend for static plots. 
# To use Makie, simply comment the ''' using Plots ''' and uncomment the ''' using CairoMakie ''' line.
# You can also change the output format by modifying the argument to
# '''CairoMakie.activate!''' (e.g., to "png" or "pdf").


#CairoMakie.activate!(; type="svg")

# %%
# Set up output directory
const _FILEPATH = let f = @__FILE__; isempty(f) ? PROGRAM_FILE : f end
const CASE     = isempty(_FILEPATH) ? "session" : first(splitext(basename(_FILEPATH)))
const SRCDIR   = isempty(_FILEPATH) ? pwd()      : dirname(_FILEPATH)
const OUTDIR   = joinpath(SRCDIR, "figures", CASE)
mkpath(OUTDIR)


# %%
h = 0.005  # grid cell size
gridlims = SA[-1.0 3.0; -2.0 2.0]
grid = Grid(;
    h, n=@.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)), x0=gridlims[:, 1], levels=5
)

# %%
L = 1.0  # plate length
x0 = SA[-0.25, 0.0] # plate starting position
theta = 12 # plate angle of attack
n_ib = round(Int, L / (2 * h))  # number of immersed boundary points
ds = L / n_ib  # arclength delta at each point

body = let
    θ = deg2rad(-theta)
    s = range(0, L, n_ib)
    x = map(s) do si 
        SA[x0[1] + si * cos(θ), x0[2] + si * sin(θ)]
    end
    StaticBody(x, fill(ds, n_ib))
end;

# %%
dt = 0.001
Re = 400.0
u0 = UniformFlow(t -> SA[1.0, 0.0])
prob = IBProblem(grid, body, Re, u0)

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
    Cl = create_dataset(all_group, "Cl", T, (n_all,))
    Cd = create_dataset(all_group, "Cd", T, (n_all,))

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
        Cd[sol.i] = 2 * f[1]
        Cl[sol.i] = 2 * f[2]

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
soln_path = joinpath(SRCDIR, "$(CASE).h5")

if isfile(soln_path)
    @info "File already exists" soln_path
else
    h5open(soln_path, "cw") do file
        solution(file; tf=40.0, snapshot_freq=100)
    end
end

# %%
# Using Plots to visualize the vorticity field and save as an animation
h5open(soln_path, "r") do file
    t = read(file["snapshots/t"])
    ω = file["snapshots/omega"]
    ω_start = read_attribute(ω, "firstindex")
    nx, ny, nlev, nt = size(ω)
    xidx = ω_start[1]:(ω_start[1] + nx - 1)
    yidx = ω_start[2]:(ω_start[2] + ny - 1)

    ωlim = 7.5
    # plate polyline (precompute once)
    plate_x = getindex.(body.x, 1)
    plate_y = getindex.(body.x, 2)
    
    anim = Animation()
    @showprogress for i in eachindex(t)
        # start a fresh frame
        p = plot(legend=false, aspect_ratio=:equal,
             xlim=(-2,8), ylim=(-2,2), framestyle=:box)

        # draw coarse→fine so the finest sits on top
        for lev in nlev:-1:1
            X, Y = coord(grid, Loc_ω(3), (xidx, yidx), lev)
            xvec, yvec = X[:,1], Y[:,1]
            z = ω[:, :, lev, i]
            # # xvec, yvec from coord; z is Nx×Ny
            heatmap!(xvec, yvec, z'; aspect_ratio=:equal, colormap=:bwr,
             xlim=(-2,15), ylim=(-3,3), legend=false, clim = (-ωlim,ωlim))
        end

        plot!(plate_x, plate_y, color=:gray, lw=3)
        title!(@sprintf("t = %.2f", t[i]))
        frame(anim, p)
    end

    gif(anim, joinpath(OUTDIR, "$(CASE)_vorticity.gif"), fps=30)
end

# %%
# Using Makie to visualize the vorticity field and save as an animation
#= h5open(soln_path, "r") do file
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
end =#

# %%
results = h5open(soln_path, "r") do soln
    (; t=read(soln["all/t"]), Cl=read(soln["all/Cl"]), Cd=read(soln["all/Cd"]))
end;

# %%
i_start = 1 + round(Int, 50 / dt)

peaks = map((; Cl=results.Cl, Cd=results.Cd)) do y
    z = @view y[Base.IdentityUnitRange(i_start:end)]
    (findminima(z), findmaxima(z))
end

periods = map(x -> mean(diff(results.t[x[2].indices])), peaks)

Pr = 1 / periods.Cl

oscillations = map(peaks) do p
    (a, b) = map(x -> mean(x.heights), p)
    ((a + b) / 2, (b - a) / 2)
end

# %%
# Using Plots to visualize lift and drag coefficients with peaks and oscillation bands
p = plot(legend = :topright, xlabel = "t", ylabel = "", ylims = (-2, 2),
         framestyle = :box)

for f in (:Cl, :Cd)
    t = results.t
    C = results[f]
    pks = peaks[f]              # (minima, maxima)
    μ, A = oscillations[f]      # (mean, amplitude)

    # pick a color per signal
    col = f == :Cl ? :blue : :red

    # line
    plot!(p, t, C; color=col, label=String(f))

    # peak markers (both minima and maxima)
    idx = vcat(pks[1].indices, pks[2].indices)
    scatter!(p, t[idx], C[idx]; color=col, ms=2.5, label="")

    # horizontal bands at mean ± amplitude
    hline!(p, [μ - A, μ + A]; color=col, linestyle=:dash, label="")
end

savefig(p, joinpath(OUTDIR, "$(CASE)_Cl_Cd.png"))

# %%    
# Using Makie to visualize lift and drag coefficients with peaks and oscillation bands
# let
#     fig = Figure()
#     ax = Axis(fig[1, 1]; limits=(nothing, (-2, 2)))

#     for f in (:Cl, :Cd)
#         C = results[f]
#         p = peaks[f]
#         o = oscillations[f]

#         lines!(ax, results.t, C)

#         i = [(x.indices for x in p)...;]
#         scatter!(ax, results.t[i], C[i])

#         hlines!(ax, @.(o[1] + [-1, 1] * o[2]); linestyle=:dash)
#     end

#     fig
# end
