# -*- coding: utf-8 -*-
# =========================================================
# FastIBPM example: Flow past a NACA 4-digit airfoil
# (airfoil points re-sampled so ds ≈ 2h and constant, like cylinder)
# =========================================================

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

# ---------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------
const _FILEPATH = let f = @__FILE__; isempty(f) ? PROGRAM_FILE : f end
const CASE     = isempty(_FILEPATH) ? "session" : first(splitext(basename(_FILEPATH)))
const SRCDIR   = isempty(_FILEPATH) ? pwd()      : dirname(_FILEPATH)
const OUTDIR   = joinpath(SRCDIR, "figures", CASE)
mkpath(OUTDIR)

# ---------------------------------------------------------
# NACA 4-digit geometry helper (dense curve generation)
# ---------------------------------------------------------
function naca4_body(code::AbstractString; chord=1.0, n=300, closed_te=false,
                    xshift=0.0, yshift=0.0, alpha=0.0)

    @assert length(code) == 4 "NACA 4-digit code must have 4 characters, e.g. \"0012\" or \"2412\"."

    d = collect(code)
    m = parse(Int, d[1]) / 100.0           # max camber
    p = parse(Int, d[2]) / 10.0            # location of max camber
    t = parse(Int, code[3:4]) / 100.0      # thickness

    # cosine-spaced x in [0,1]
    ξ = range(0.0, 1.0; length=n)
    x = @.(0.5 * (1 - cos(pi * ξ)))

    # thickness distribution
    c5 = closed_te ? 0.1036 : 0.1015
    yt = @.(5t * (0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - c5*x^4))

    # camber line and slope
    yc = similar(x)
    dyc_dx = similar(x)

    if m == 0.0 || p == 0.0
        yc .= 0.0
        dyc_dx .= 0.0
    else
        for i in eachindex(x)
            xi = x[i]
            if xi < p
                yc[i] = (m / p^2) * (2p*xi - xi^2)
                dyc_dx[i] = (2m / p^2) * (p - xi)
            else
                yc[i] = (m / (1 - p)^2) * ((1 - 2p) + 2p*xi - xi^2)
                dyc_dx[i] = (2m / (1 - p)^2) * (p - xi)
            end
        end
    end

    θ = atan.(dyc_dx)

    # upper / lower surfaces
    xu = @.(x - yt*sin(θ))
    yu = @.(yc + yt*cos(θ))
    xl = @.(x + yt*sin(θ))
    yl = @.(yc - yt*cos(θ))

    # Build CLOSED contour:
    # TE -> along upper to LE, then along lower back to TE (avoid duplicating TE/LE)
    upper = [SA[chord*xu[i], chord*yu[i]] for i in reverse(eachindex(xu))]
    lower = [SA[chord*xl[i], chord*yl[i]] for i in eachindex(xl)[2:end-1]]
    pts = vcat(upper, lower)

    # rotate + translate
    if alpha != 0.0
        ca, sa = cos(alpha), sin(alpha)
        R(p) = SA[ca*p[1] - sa*p[2], sa*p[1] + ca*p[2]]
        pts = [R(p) for p in pts]
    end
    if xshift != 0.0 || yshift != 0.0
        shift = SA[xshift, yshift]
        pts = [p + shift for p in pts]
    end

    # ds here is not used (we will re-sample & set constant ds like cylinder)
    return pts, nothing
end

# ---------------------------------------------------------
# Re-sample a closed curve to uniform arclength spacing
# ---------------------------------------------------------
function resample_closed_uniform_arclength(pts::Vector{SVector{2,T}}, n::Int) where {T}
    N = length(pts)
    @assert N >= 3

    s = zeros(T, N+1)
    for i in 1:N
        p1 = pts[i]
        p2 = pts[i == N ? 1 : i+1]
        s[i+1] = s[i] + norm(p2 - p1)
    end
    L = s[end]

    st = range(zero(T), L; length=n+1)[1:end-1]  # exclude endpoint to avoid duplicate point

    out = Vector{SVector{2,T}}(undef, n)
    seg = 1
    for (k, sk) in enumerate(st)
        while seg < N && sk > s[seg+1]
            seg += 1
        end
        p1 = pts[seg]
        p2 = pts[seg == N ? 1 : seg+1]
        denom = s[seg+1] - s[seg]
        α = denom == 0 ? zero(T) : (sk - s[seg]) / denom
        out[k] = (one(T) - α) * p1 + α * p2
    end

    return out, L
end

# ---------------------------------------------------------
# Grid / problem setup
# ---------------------------------------------------------
h = 0.0055
gridlims = SA[-1.0 1.0; -1.0 1.0]
grid = Grid(; h,
    n=@.(round(Int, (gridlims[:, 2] - gridlims[:, 1]) / h)),
    x0=gridlims[:, 1],
    levels=5
)

dt = 0.001
Re = 1000.0
u0 = UniformFlow(t -> SA[1.0, 0.0])

# ---------------------------------------------------------
# Build NACA airfoil body with constant ds ≈ 2h (like cylinder)
# ---------------------------------------------------------
naca_code = "0015"
alpha_deg = -15.0
alpha = alpha_deg * pi/180

# make a dense curve first
chord = 1.0

# rotate airfoil about its mid-chord, then translate so mid-chord is at (0,0)
mid_local = SA[0.5*chord, 0.0]
ca, sa = cos(alpha), sin(alpha)
mid_rot = SA[ca*mid_local[1] - sa*mid_local[2], sa*mid_local[1] + ca*mid_local[2]]

# set shifts so the rotated mid-chord ends up at (0,0)
xshift = -mid_rot[1]
yshift = -mid_rot[2]

pts_dense, _ = naca4_body(naca_code; chord=chord, n=800, closed_te=false,
                          alpha=alpha, xshift=xshift, yshift=yshift)
# perimeter S (like cylinder S = 2πr)
S = sum(norm(pts_dense[i == length(pts_dense) ? 1 : i+1] - pts_dense[i]) for i in eachindex(pts_dense))

# choose n_ib so ds ≈ 2h
n_ib = max(40, round(Int, S / (2h)))
ds = S / n_ib

# resample to uniform spacing
pts, Scheck = resample_closed_uniform_arclength(pts_dense, n_ib)
@info "Perimeter check" S Scheck n_ib ds ds_over_h = ds/h

# StaticBody with constant ds (exactly like cylinder)
body = StaticBody(pts, fill(ds, n_ib))

# ---------------------------------------------------------
# Preview geometry (airfoil + optional grid box)
# ---------------------------------------------------------
bx = [p[1] for p in body.x]
by = [p[2] for p in body.x]

pgeo = plot(bx, by;
    aspect_ratio=:equal,
    legend=false,
    framestyle=:box,
    xlabel="x", ylabel="y",
    title=@sprintf("NACA %s, α = %.1f°  (n_ib=%d, ds/h=%.2f)", naca_code, alpha_deg, n_ib, ds/h),
)

# close the curve visually
plot!(pgeo, [bx; bx[1]], [by; by[1]])

# show grid domain box
x0, x1 = gridlims[1,1], gridlims[1,2]
y0, y1 = gridlims[2,1], gridlims[2,2]
plot!(pgeo, [x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0]; lw=1, linestyle=:dash)

# mark a reference point (mid-chord in your shifted coords would be near (0,0) if you set it)
scatter!(pgeo, [0.0], [0.0]; ms=4)

display(pgeo)

savefig(pgeo, joinpath(OUTDIR, "$(CASE)_airfoil_preview.png"))

prob = IBProblem(grid, body, Re, u0)

# ---------------------------------------------------------
# Time integration + HDF5 output
# ---------------------------------------------------------
function solution(file; tf, snapshot_freq)
    
    T = Float64
    sol = CNAB(prob; dt, delta=FastIBPM.DeltaYang3S2())

    # Small perturbation to kick-start unsteady wake
    # map!(sol.ω[1][3], CartesianIndices(sol.ω[1][3])) do I
    #     x = coord(grid, Loc_ω(3), I)
    #     p = x - SA[-0.2, 0.0]
    #     r = 0.25
    #     0.15 * (1 - clamp(norm(p) / r, 0, 1))
    # end
    # apply_vorticity!(sol)

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
    ω = create_dataset(snapshot_group, "omega", T, (size(sol.ω[1][3])..., grid.levels, n_snapshot))
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

# ---------------------------------------------------------
# Run or reuse solution file
# ---------------------------------------------------------
soln_path = joinpath(SRCDIR, "$(CASE)_naca$(naca_code)_a$(round(Int,alpha_deg))deg.h5")

if isfile(soln_path)
    @info "File already exists" soln_path
else
    h5open(soln_path, "cw") do file
        solution(file; tf=20, snapshot_freq=10)
    end
end

# ---------------------------------------------------------
# Vorticity animation (Plots)
# ---------------------------------------------------------
h5open(soln_path, "r") do file
    t = read(file["snapshots/t"])
    ω = file["snapshots/omega"]
    ω_start = read_attribute(ω, "firstindex")
    nx, ny, nlev, nt = size(ω)
    xidx = ω_start[1]:(ω_start[1] + nx - 1)
    yidx = ω_start[2]:(ω_start[2] + ny - 1)

    ωlim = 4

    bx = [p[1] for p in body.x]
    by = [p[2] for p in body.x]

    anim = Animation()
    @showprogress for i in 1:5:2000 #eachindex(t)
        p = plot(legend=false, aspect_ratio=:equal,
                 xlim=(-0.6, 6.0), ylim=(-1.8, 1.8), framestyle=:box)

        for lev in 4:-1:1
            X, Y = coord(grid, Loc_ω(3), (xidx, yidx), lev)
            xvec, yvec = X[:, 1], Y[:, 1]
            z = ω[:, :, lev, i]
            heatmap!(xvec, yvec, z'; aspect_ratio=:equal,
                     colormap=:bwr, legend=false, clim=(-ωlim, ωlim), cbar = true)
        end

        plot!(Shape(bx, by), color=:gray, lw=0)
        # plot!(bx, by; color=:gray, lw=2)
        title!(@sprintf("t = %.2f", t[i]))
        frame(anim, p)
    end

    gif(anim, joinpath(OUTDIR, "$(CASE)_naca$(naca_code)_a$(round(Int,alpha_deg))deg_vorticity.gif"), fps=30)
end

# ---------------------------------------------------------
# Read forces and plot Cl/Cd with peaks + mean±amp bands
# ---------------------------------------------------------
results = h5open(soln_path, "r") do soln
    (; t=read(soln["all/t"]), Cl=read(soln["all/Cl"]), Cd=read(soln["all/Cd"]))
end

i_start = 1 + round(Int, 50 / dt)

peaks = map((; Cl=results.Cl, Cd=results.Cd)) do y
    z = @view y[Base.IdentityUnitRange(i_start:end)]
    (findminima(z), findmaxima(z))
end

periods = map(x -> mean(diff(results.t[x[2].indices])), peaks)
Pr = 1 / periods.Cl
@info "Estimated lift frequency" Pr

oscillations = map(peaks) do p
    (a, b) = map(x -> mean(x.heights), p)
    ((a + b) / 2, (b - a) / 2)
end

p = plot(legend=:topright, xlabel="t", ylabel="", ylims=(-2, 2), framestyle=:box)

for f in (:Cl, :Cd)
    tvec = results.t
    C = results[f]
    pks = peaks[f]
    μ, A = oscillations[f]
    col = f == :Cl ? :blue : :red

    plot!(p, tvec, C; color=col, label=String(f))
    idx = vcat(pks[1].indices, pks[2].indices)
    scatter!(p, tvec[idx], C[idx]; color=col, ms=2.5, label="")
    hline!(p, [μ - A, μ + A]; color=col, linestyle=:dash, label="")
end

savefig(p, joinpath(OUTDIR, "$(CASE)_naca$(naca_code)_a$(round(Int,alpha_deg))deg_Cl_Cd.png"))
