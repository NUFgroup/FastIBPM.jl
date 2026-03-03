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

# %%
# Set up output directory
const _FILEPATH = let f = @__FILE__; isempty(f) ? PROGRAM_FILE : f end
const CASE     = isempty(_FILEPATH) ? "session" : first(splitext(basename(_FILEPATH)))
const SRCDIR   = isempty(_FILEPATH) ? pwd()      : dirname(_FILEPATH)
const OUTDIR   = joinpath(SRCDIR, "figures", CASE)
mkpath(OUTDIR)

# --- helper: make a horizontal-surge rigid-motion callback (consistent with your ibpm xc/uc idea) ---
function make_surge_motion(x_ref::Vector{SVector{2,Float64}}; vx = -1.0)
    # returns: (x, u, i, t) -> nothing  that overwrites positions & velocities
    return function (x::AbstractVector{SVector{2,Float64}},
                     u::AbstractVector{SVector{2,Float64}},
                     i::Int, t::Float64)
        xshift = vx * t
        xdot   = vx
        @inbounds for j in eachindex(x)
            # translate every reference point by (xshift, 0); rigid velocity is (xdot, 0)
            x[j] = x_ref[j] + SA[xshift, 0.0]
            u[j] = SA[xdot,  0.0]
        end
        return nothing
    end
end


# %%
h = 0.01  # grid cell size
gridlims = SA[-5.0 5.0; -3.0 3.0]
grid = Grid(;
    h, n=@.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)), x0=gridlims[:, 1], levels=5
)

# %%
# --- moving cylinder ---
r = 0.5                   # cylinder radius
xc, yc = 0.0, 0.0         # center at t = 0
S = 2π * r                # circumference
n_ib = round(Int, S / (2 * h))
ds = S / n_ib

θs = range(0, 2π, length=n_ib + 1)[1:end-1]

# reference points for cylinder centered at (xc, yc)
x_ref = [SA[xc, yc] + r * SA[cos(θ), sin(θ)] for θ in θs]

# impulsive horizontal start with constant velocity vx
vx_cyl = -1.0
motion! = make_surge_motion(x_ref; vx=vx_cyl)

body = MovingBody(x_ref, fill(ds, n_ib), motion!)


# %%
dt = 0.001
Re = 40.0
u0 = UniformFlow(t -> SA[0.0, 0.0])
prob = IBProblem(grid, body, Re, u0);

# %%
function solution(file; tf, snapshot_freq)
    T = Float64
    sol = CNAB(prob; dt, delta=FastIBPM.DeltaYang3S2())

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
        solution(file; tf=4, snapshot_freq=100)
    end
end



h5open(soln_path, "r") do file
    t = read(file["snapshots/t"])
    ω = file["snapshots/omega"]
    ω_start = read_attribute(ω, "firstindex")
    nx, ny, nlev, nt = size(ω)
    xidx = ω_start[1]:(ω_start[1] + nx - 1)
    yidx = ω_start[2]:(ω_start[2] + ny - 1)

    ωlim = 3

    tplot = [1.0, 2.5, 3.5]
    idxplot = [argmin(abs.(t .- tp)) for tp in tplot]

    x_plate0 = getindex.(x_ref, 1)
    y_plate0 = getindex.(x_ref, 2)

    p = plot(layout=(1,3), size=(1200,400), legend=false)

    for (j,i) in enumerate(idxplot)

        vx = -1.0
        xshift = vx * t[i]

        plot!(p[j];
            aspect_ratio=:equal,
            xlim=(-5, 1),
            ylim=(-2, 2),
            framestyle=:box)

        for lev in nlev:-1:1
            X, Y = coord(grid, Loc_ω(3), (xidx, yidx), lev)
            xvec, yvec = X[:,1], Y[:,1]
            z = ω[:, :, lev, i]

            zplot = copy(z)
            finite_mask = isfinite.(zplot)
            zplot[finite_mask] .= clamp.(zplot[finite_mask], -5, 5)

            contourf!(p[j], xvec, yvec, zplot';
                aspect_ratio=:equal,
                colormap=:bwr,
                levels=30,
                lw=0,
                legend=false,
                clim=(-5, 5))
        end

        plot!(p[j], x_plate0 .+ xshift, y_plate0;
            color=:gray, lw=5, label="")

        title!(p[j], @sprintf("t = %.3f", t[i]))
    end

    display(p)
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

    ωlim = 3

    # --- plate draw cache (from your x_ref used to build the body) ---
    x_plate0 = getindex.(x_ref, 1)
    y_plate0 = getindex.(x_ref, 2)

    anim = Animation()
    @showprogress for i in eachindex(t)

        # horizontal surge position (matches motion!)
        vx = -1.0
        xshift = vx * t[i]
        xdot   = vx
        # start a fresh frame
        p = plot(legend=false, aspect_ratio=:equal,
                 xlim=(gridlims[1,1], gridlims[1,2]),
                 ylim=(gridlims[2,1], gridlims[2,2]),
                 framestyle=:box)

        # draw coarse→fine so the finest sits on top
        for lev in nlev:-1:1
            X, Y = coord(grid, Loc_ω(3), (xidx, yidx), lev)
            xvec, yvec = X[:,1], Y[:,1]
            z = ω[:, :, lev, i]

            # heatmap!(xvec, yvec, z';
            #          aspect_ratio=:equal,
            #          colormap=:bwr,
            #          legend=false,
            #          clim=(-ωlim, ωlim))
            

           zplot = copy(z)

            finite_mask = isfinite.(zplot)
            zplot[finite_mask] .= clamp.(zplot[finite_mask], -5, 5)

            contourf!(xvec, yvec, zplot';
                aspect_ratio = :equal,
                colormap = :bwr,
                levels = 30,
                lw = 0,
                legend = false,
                clim = (-5, 5),
            )
        end

        # draw the surging plate
        plot!(p, x_plate0 .+ xshift, y_plate0; color=:gray, lw=5, label="")

        title!(p, @sprintf("t = %.3f", t[i]))
        frame(anim, p)
    end

    gif(anim, joinpath(OUTDIR, "$(CASE)_vorticity.gif"); fps=30)
end


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
p = plot(legend = :topright, xlabel = "t", ylabel = "", ylims = (-20, 20),
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


