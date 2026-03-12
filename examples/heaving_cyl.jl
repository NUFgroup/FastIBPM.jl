# -*- coding: utf-8 -*-
# %%
using Immersa
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

# This example simulates flow around a cylinder at Re=200. It uses a uniform inflow
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
const _FILEPATH = let f = @__FILE__;
    isempty(f) ? PROGRAM_FILE : f
end
const CASE = isempty(_FILEPATH) ? "session" : first(splitext(basename(_FILEPATH)))
const SRCDIR = isempty(_FILEPATH) ? pwd() : dirname(_FILEPATH)
const OUTDIR = joinpath(SRCDIR, "figures", CASE)
mkpath(OUTDIR)

# --- helper: make a vertical-heave rigid-motion callback ---
function make_heave_motion(x_ref::Vector{SVector{2,Float64}}; A=1.0, f=0.20, ϕ=0.0)
    # returns: (x, u, i, t) -> nothing  that overwrites positions & velocities
    return function (
        x::AbstractVector{SVector{2,Float64}},
        u::AbstractVector{SVector{2,Float64}},
        i::Int,
        t::Float64,
    )
        y = A * sin(2π*f*t + ϕ)
        ydot = 2π * f * A * cos(2π*f*t + ϕ)
        @inbounds for j in eachindex(x)
            # translate every reference point by (0, y); rigid velocity is (0, ydot)
            x[j] = x_ref[j] + SA[0.0, y]
            u[j] = SA[0.0, ydot]
        end
        return nothing
    end
end

# %%
h = 0.05  # grid cell size
gridlims = SA[-1.0 3.0; -2.0 2.0]
grid = Grid(;
    h, n=@.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)), x0=gridlims[:, 1], levels=5
)

# %%
# --- moving cylinder body (replaces your StaticBody block) ---
r = 0.5
S = 2π * r
n_ib = round(Int, S / (2h))          # IB spacing ≈ 2h
ds = S / n_ib

# reference circle (at rest, centered at origin)
x_ref = map(range(0, 2π, n_ib + 1)[1:(end-1)]) do θ
    r * SA[cos(θ), sin(θ)]
end

# build heave motion with amplitude = 1.0 (up/down), frequency f (Hz in your nondim time)
f_heave = 0.50
motion! = make_heave_motion(x_ref; A=1.0, f=f_heave)

# MovingBody must be your new type that <: AbstractPrescribedBody
# and whose update_body_points! calls `body.motion!(pts.x, pts.u, i, t)`
body = MovingBody(x_ref, fill(ds, n_ib), motion!)

# %%
dt = 0.002
Re = 200.0
u0 = UniformFlow(t -> SA[1.0, 0.0])
prob = IBProblem(grid, body, Re, u0);

# %%
function solution(file; tf, snapshot_freq)
    T = Float64
    sol = CNAB(prob; dt, delta=Immersa.DeltaYang3S2())

    # Perturbation to induce vortex shedding
    map!(sol.ω[1][3], CartesianIndices(sol.ω[1][3])) do I
        x = coord(grid, Loc_ω(3), I)
        p = x - SA[-0.75, 0]
        r = 0.25
        0.5 * (1 - clamp(norm(p) / r, 0, 1))
    end
    apply_vorticity!(sol)

    i_all = 1:(1+round(Int, tf/dt))
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

    @showprogress desc = "solving" for _ in 0:round(Int, tf/dt)
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
        solution(file; tf=4.0, snapshot_freq=100)
    end
end

# %%
# Using Plots to visualize the vorticity field and save as an animation
h5open(soln_path, "r") do file
    t = read(file["snapshots/t"])
    ω = file["snapshots/omega"]
    ω_start = read_attribute(ω, "firstindex")
    nx, ny, nlev, nt = size(ω)
    xidx = ω_start[1]:(ω_start[1]+nx-1)
    yidx = ω_start[2]:(ω_start[2]+ny-1)

    ωlim = 7.5
    r_draw = 0.485;
    θ = range(0, 2π; length=400)
    # cx, cy = r .* cos.(θ), r .* sin.(θ)
    A_heave = 1.0
    ϕ_heave = 0.0
    anim = Animation()
    @showprogress for i in eachindex(t)
        y0 = A_heave * sin(2π * f_heave * t[i] + ϕ_heave)
        # circle shifted by y0
        cx = r_draw .* cos.(θ)
        cy = y0 .+ r_draw .* sin.(θ)
        # start a fresh frame
        p = plot(
            legend=false, aspect_ratio=:equal, xlim=(-2, 8), ylim=(-2, 2), framestyle=:box
        )

        # draw coarse→fine so the finest sits on top
        for lev in nlev:-1:1
            X, Y = coord(grid, Loc_ω(3), (xidx, yidx), lev)
            xvec, yvec = X[:, 1], Y[:, 1]
            z = ω[:, :, lev, i]
            # # xvec, yvec from coord; z is Nx×Ny
            heatmap!(
                xvec,
                yvec,
                z';
                aspect_ratio=:equal,
                colormap=:bwr,
                xlim=(-2, 15),
                ylim=(-3, 3),
                legend=false,
                clim=(-ωlim, ωlim),
            )
        end

        plot!(Shape(cx, cy), color=:gray, lw=0)
        title!(@sprintf("t = %.2f", t[i]))
        frame(anim, p)
    end

    gif(anim, joinpath(OUTDIR, "$(CASE)_vorticity.gif"))
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
p = plot(; legend=:topright, xlabel="t", ylabel="", ylims=(-2, 2), framestyle=:box)

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
