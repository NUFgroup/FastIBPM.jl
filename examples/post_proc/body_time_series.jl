using HDF5
using WriteVTK
using LinearAlgebra
using Printf


# create VTK polydata xml file for a closed polyline
# write file manually in XML, handlge points and forces as point data
function write_vtp_polyline(filename::AbstractString;
                            points::AbstractMatrix,         # 3×N (x,y,z by rows)
                            vectors::Union{AbstractMatrix,Nothing}=nothing,  # 3×N or nothing
                            vect_name::AbstractString="forces")

    @assert size(points,1) == 3 "points must be 3×N"
    N = size(points,2)
    has_vec = !(vectors === nothing)
    if has_vec
        @assert size(vectors) == size(points) "vectors must be 3×N to match points"
    end

    open(filename, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">")
        println(io, "  <PolyData>")
        println(io, "    <Piece NumberOfPoints=\"$N\" NumberOfLines=\"1\">")

        if has_vec
            println(io, "      <PointData>")
            println(io, "        <DataArray type=\"Float32\" Name=\"$vect_name\" NumberOfComponents=\"3\" format=\"ascii\">")
            for i in 1:N
                @printf(io, " %.7g %.7g %.7g", vectors[1,i], vectors[2,i], vectors[3,i])
            end
            println(io)  # newline
            println(io, "        </DataArray>")
            println(io, "      </PointData>")
        else
            println(io, "      <PointData/>")
        end

        println(io, "      <Points>")
        println(io, "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">")
        for i in 1:N
            @printf(io, " %.7g %.7g %.7g", points[1,i], points[2,i], points[3,i])
        end
        println(io)
        println(io, "        </DataArray>")
        println(io, "      </Points>")

        # One polyline using all points in order (loop is already closed if you repeated the first point)
        println(io, "      <Lines>")
        println(io, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">")
        for i in 0:N-1
            @printf(io, " %d", i)
        end
        println(io)
        println(io, "        </DataArray>")
        println(io, "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\"> $N </DataArray>")
        println(io, "      </Lines>")

        println(io, "    </Piece>")
        println(io, "  </PolyData>")
        println(io, "</VTKFile>")
    end
end

# create paraview data file (.pvd) that groups multiple vtk files
function write_pvd(filename::AbstractString, times::Vector{<:Real}, files::Vector{<:AbstractString})
    @assert length(times) == length(files)
    open(filename, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">")
        println(io, "  <Collection>")
        for (t, f) in zip(times, files)
            @printf(io, "    <DataSet timestep=\"%.9g\" group=\"\" part=\"0\" file=\"%s\"/>\n", float(t), f)
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
end

# turn point arrays into 3xN format
function as_3xN(A)
    ndims(A) == 2 || error("points must be 2D, got ndims=$(ndims(A))")
    r, c = size(A)
    if r == 3
        return A
    elseif r == 2
        return vcat(A, zeros(1, c))
    elseif c == 3
        return permutedims(A, (2,1))
    elseif c == 2
        B = permutedims(A, (2,1))
        return vcat(B, zeros(1, size(B,2)))
    else
        error("points must be (2 or 3)×N or N×(2 or 3); got $(size(A))")
    end
end

# extract force component (fx, fy) for a specific timestep
function forces_for_timestep(traction, t)
    nd = ndims(traction)
    if nd == 3
        a,b,tt = size(traction)
        t <= tt || error("timestep $t exceeds available time $tt")
        if a == 2
            return traction[1, :, t], traction[2, :, t]     # (2,N,T)
        elseif b == 2
            return traction[:, 1, t], traction[:, 2, t]     # (N,2,T)
        else
            error("traction must be (2,N,T) or (N,2,T); got $(size(traction))")
        end
    elseif nd == 2
        a,b = size(traction)
        if a == 2
            return traction[1, :], traction[2, :]           # (2,N)
        elseif b == 2
            return traction[:, 1], traction[:, 2]           # (N,2)
        else
            error("traction must be (2,N) or (N,2); got $(size(traction))")
        end
    else
        error("traction must be rank-2 or rank-3; got ndims=$nd")
    end
end

# ensure force vector arrays match the number of body points
function match_force_length!(fx, fy, n_points)
    if length(fx) == n_points - 1
        fx = vcat(fx, fx[1]);  fy = vcat(fy, fy[1])
    elseif length(fx) == n_points
        # ok
    elseif length(fx) == n_points + 1 && fx[end] ≈ fx[1] && fy[end] ≈ fy[1]
        fx = fx[1:n_points];   fy = fy[1:n_points]
    else
        error("traction length $(length(fx)) doesn’t match points count $n_points")
    end
    return fx, fy
end


# Include coordinate utilities if available
if isfile("coord_utils.jl")
    include("coord_utils.jl")
end

# =============================================================================
# Configuration
# =============================================================================

const FILENAME = get(ARGS, 1, "examples/moving_cylinder.h5")
const OUTPUT_DIR = "examples/vtk_output_body/moving_cylinder"
const N_TIMESTEPS = typemax(Int)  # export all available snapshots

# =============================================================================
# Main Processing Function
# =============================================================================

# check file existence, creates output directory, read HDF5 data, export both 
# single timestep and time series visualization
function process_body_data()
    # check if file exists
    if !isfile(FILENAME)
        error("File not found")
    end

    # create output directory
    mkpath(OUTPUT_DIR)

    println("Processing body data from: $FILENAME")
    println("Output directory: $OUTPUT_DIR")
    println("=" ^ 60)

    # read data from HDF5 file
    body_data = read_body_data(FILENAME)

    # export single time instant
    export_single_timestep(body_data)

    # export time series
    export_time_series(body_data)

    println("\n Processing complete!")
end

# =============================================================================
# Data Reading Functions
# =============================================================================

# read body simulation data from HDF5 file
# body point coords, lengths, time values, traction force data
# transpose traction data to match python's array layout
function read_body_data(filename)
    println("Reading HDF5 data...")

    local body_points, body_ds, t_traction, t_snapshot, traction, body_points_series

    h5open(filename, "r") do file
        for path in ["body/points", "body/lengths"]
            haskey(file, path) || error("Required dataset $path not found in HDF5 file!")
        end

        body_points = read(file["body/points"])
        body_ds     = read(file["body/lengths"])

        # time key varies between simulation outputs
        t_key = haskey(file, "all/t") ? "all/t" : "all/time"
        t_traction = haskey(file, t_key) ? read(file[t_key]) : Float64[]

        # snapshot time vector (used for moving bodies)
        t_snapshot = haskey(file, "snapshots/t") ? read(file["snapshots/t"]) : Float64[]

        # traction is optional — body shape can be exported without forces
        if haskey(file, "all/traction")
            traction_raw = read(file["all/traction"])
            traction = ndims(traction_raw) == 3 ?
                permutedims(traction_raw, (2, 1, 3)) :
                permutedims(traction_raw, (2, 1))
        else
            traction = nothing
        end

        # per-snapshot body positions for moving bodies (2×N×n_snap)
        body_points_series = haskey(file, "snapshots/body_points") ?
            read(file["snapshots/body_points"]) : nothing
    end

    println("  Body points shape: $(size(body_points))")
    println("  Moving body snapshots: $(body_points_series === nothing ? "no" : string(size(body_points_series)))")
    println("  Traction: $(traction === nothing ? "not available" : string(size(traction)))")
    println("  Time steps: $(length(t_traction))")

    return (
        points = body_points,
        lengths = body_ds,
        time = t_traction,
        t_snapshot = t_snapshot,
        traction = traction,
        body_points_series = body_points_series
    )
end

# =============================================================================
# Export Functions
# =============================================================================

# export one timestep as .vtp
function export_single_timestep(body_data)
    println("\nExporting single time instant...")

    time_instant = 1

    # use per-snapshot positions for moving bodies, static points otherwise
    raw_pts = body_data.body_points_series !== nothing ?
        body_data.body_points_series[:, :, time_instant] : body_data.points
    points = as_3xN(raw_pts)

    # close the loop by repeating the first point if needed
    if any(abs.(points[:,1] .- points[:,end]) .> 1e-12)
        points = hcat(points, points[:,1])
    end
    n_points = size(points, 2)

    force_vectors = if body_data.traction !== nothing
        fx, fy = forces_for_timestep(body_data.traction, time_instant)
        fx, fy = match_force_length!(fx, fy, n_points)
        fv = zeros(3, n_points)
        fv[1,:] = fx;  fv[2,:] = fy
        fv
    else
        nothing
    end

    fname = "airfoil_with_forces.vtp"
    write_vtp_polyline(fname; points=points, vectors=force_vectors, vect_name="forces")
    println("Saved: $(fname)")
end

# export multiple vtk files and create pvd collection for animation
function export_time_series(body_data)
    println("\n Exporting time series...")

    has_moving = body_data.body_points_series !== nothing
    t_max = if has_moving
        size(body_data.body_points_series, 3)
    elseif body_data.traction !== nothing
        size(body_data.traction, 3)
    else
        length(body_data.time)
    end
    if t_max == 0
        println("No time series data found, skipping...")
        return
    end

    n_steps   = min(N_TIMESTEPS, t_max)
    timesteps = unique(round.(Int, range(1, t_max, length=n_steps)))
    println("  Exporting $(length(timesteps)) timesteps from $t_max total")

    # for static bodies, pre-compute closed-loop points once
    static_points = if !has_moving
        pts = as_3xN(body_data.points)
        any(abs.(pts[:,1] .- pts[:,end]) .> 1e-12) ? hcat(pts, pts[:,1]) : pts
    else
        nothing
    end

    # output dir + filenames
    ispath(OUTPUT_DIR) || mkpath(OUTPUT_DIR)
    vtp_files = String[]
    times = Float64[]

    for (i, t) in enumerate(timesteps)
        # get body points for this timestep
        points = if has_moving
            pts = as_3xN(body_data.body_points_series[:, :, t])
            any(abs.(pts[:,1] .- pts[:,end]) .> 1e-12) ? hcat(pts, pts[:,1]) : pts
        else
            static_points
        end
        n_points = size(points, 2)

        force_vectors = if body_data.traction !== nothing
            fx, fy = forces_for_timestep(body_data.traction, t)
            fx, fy = match_force_length!(fx, fy, n_points)
            fv = zeros(3, n_points)
            fv[1,:] = fx;  fv[2,:] = fy
            fv
        else
            nothing
        end

        # file name (per step) and write .vtp
        base = @sprintf("body_%04d", i-1)
        vtp_path = joinpath(OUTPUT_DIR, base * ".vtp")
        write_vtp_polyline(vtp_path; points=points, vectors=force_vectors, vect_name="forces")

        push!(vtp_files, base * ".vtp")
        actual_time = if has_moving && length(body_data.t_snapshot) >= t
            float(body_data.t_snapshot[t])
        elseif length(body_data.time) >= t
            float(body_data.time[t])
        else
            float(t - 1)
        end
        push!(times, actual_time)

        if i % 10 == 0 || i == length(timesteps)
            @printf("\r  Progress: %d/%d", i, length(timesteps)); flush(stdout)
        end
    end
    println()

    # write the .pvd that points to .vtp files (relative paths)
    pvd_path = joinpath(OUTPUT_DIR, "body_timeseries.pvd")
    write_pvd(pvd_path, times, vtp_files)
    println("Saved time series: $(pvd_path)")
end


# =============================================================================
# Run Main Function
# =============================================================================

try
    process_body_data()
catch e
    println("Error: $e")
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    rethrow()
end
