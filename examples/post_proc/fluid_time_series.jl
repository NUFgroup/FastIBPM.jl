#!/usr/bin/env julia

using HDF5
using Printf
using WriteVTK

# =============================================================================
# Helper Functions
# =============================================================================

# flatten for VTK point data
to_vtk_point(A::AbstractMatrix) = vec(A)

# flatten for VTK cell data
to_vtk_cell(A::AbstractMatrix) = vec(A)


# extract physical coords from HDF5 file
function xy_from_HDF5(dset::HDF5.Dataset, lev::Int)
    # input: HDF5 dataset with coords, refinement level
    # output: (x_vector, y_vector or nothing)
    try
        attrs = attributes(dset)
        haskey(attrs, "coords") || return nothing

        # coords from HDF5 dataset
        coords = read(attrs["coords"])

        # parse coords structure to extract x, y records for level lev
        recx, recy = nothing, nothing

        # is coords an array and have 2 dimensions
        if isa(coords, AbstractArray) && ndims(coords) == 2
            # does column dimension have at least 'lev' elements and does row dimension = 2
            if size(coords, 2) >= lev && size(coords, 1) == 2
                # x-coord for level lev, y-coord for level lev
                recx, recy = coords[1, lev], coords[2, lev]
            # does row dimension have at least 'lev' elements, and does row dimension = 2
            elseif size(coords, 1) >= lev && size(coords, 2) == 2
                recx, recy = coords[lev, 1], coords[lev, 2]
            end
        # is coords a vector and have at least 'lev' elements
        elseif isa(coords, AbstractVector) && length(coords) >= lev
            pair = coords[lev]

            # is extracted pair a vector and have at least 2 elements
            if isa(pair, AbstractVector) && length(pair) >= 2
                recx, recy = pair[1], pair[2]
            end
        end

        recx === nothing && return nothing

        # helper to extract from record
        getkey(a, k) = isa(a, Dict) ? a[k] :
                        hasproperty(a, k) ? getproperty(a, k) :
                        isa(a, NamedTuple) && haskey(a, k) ? a[k] :
                        error("Unrecognized coords format")
        
        # build coord vector from min/step/length
        function coord_vec(a)
            len = Int(getkey(a, :length))
            step = float(getkey(a, :step))
            minv = float(getkey(a, :min))
            collect(minv:step:(minv + step * (len - 1)))
        end

        return (coord_vec(recx), coord_vec(recy))
    catch e
        println("Coord extraction failed: $e")
        return nothing
    end
end

# suppress WriteVTK output (fluid_0000.vts, etc.)
silence(f, quiet = true) = quiet ? (redirect_stdout(devnull) do; redirect_stderr(devnull) do; f() end end) : f()


# =============================================================================
# Configuration
# =============================================================================

const FILENAME = get(ARGS, 1, "examples/moving_cylinder.h5")
const N_TIMESTEPS = typemax(Int)  # export all available snapshots
const LEVEL = 1
const OUTPUT_DIR = "examples/vtk_output_fluid/moving_cylinder/level_$(LEVEL)"
const QUIET_VTK = true

# =============================================================================
# Main Processing
# =============================================================================

function main()
    #file validation
    isfile(FILENAME) || error("File not found: $FILENAME")
    mkpath(OUTPUT_DIR)

    # print header
    println("Processing body data from: $FILENAME")
    println("Output directory: $OUTPUT_DIR")
    println("=" ^ 60)

    # Read HDF5 data
    println("Reading HDF5 data ...")
    h5open(FILENAME, "r") do h5
        # check required datasets (accept both naming conventions)
        haskey(h5, "snapshots/ux") || error("Missing snapshots/ux")
        haskey(h5, "snapshots/uy") || error("Missing snapshots/uy")
        vort_key = haskey(h5, "snapshots/omega") ? "snapshots/omega" : "snapshots/vorticity"
        haskey(h5, vort_key) || error("Missing snapshots/omega or snapshots/vorticity")
        time_key = haskey(h5, "snapshots/t") ? "snapshots/t" : "snapshots/time"
        haskey(h5, time_key) || error("Missing snapshots/t or snapshots/time")

        # get dataset handles
        u_grp = h5["snapshots/ux"]
        v_grp = h5["snapshots/uy"]
        w_grp = h5[vort_key]
        # read data arrays
        u_raw = read(u_grp)
        v_raw = read(v_grp)
        vort = read(w_grp)
        time = read(h5[time_key])

        # read physical coordinate arrays (saved by cylinder.jl / similar scripts)
        x_coords = haskey(h5, "snapshots/x_coords") ? read(h5["snapshots/x_coords"]) : nothing
        y_coords = haskey(h5, "snapshots/y_coords") ? read(h5["snapshots/y_coords"]) : nothing

        # print data info
        println("  u field: $(size(u_raw))")
        println("  v field: $(size(v_raw))")
        println("  Vorticity: $(size(vort))")
        println("  Time steps: $(length(time))\n")
        # process data
        process_level(u_raw, v_raw, vort, w_grp, time, x_coords, y_coords)
    end

    println("Processing complete!")
end

function process_level(u_raw, v_raw, vort, w_grp, time, x_coords=nothing, y_coords=nothing)
    # Step 1: get grid dimensions and convert level to 1-based indexing
    Nx_v, Ny_v, Nlev_v, Nt_v = size(vort)
    lev = clamp(LEVEL + 1, 1, Nlev_v)

    # extract physical coordinates
    if x_coords !== nothing && y_coords !== nothing
        x = x_coords[:, lev]
        y = y_coords[:, lev]
        println("Using physical coords from HDF5 dataset")
    else
        coords = xy_from_HDF5(w_grp, lev)
        if coords === nothing
            println("Using uniform grid (no physical coords found)")
            x = collect(0.0:(Nx_v - 1))
            y = collect(0.0:(Ny_v - 1))
        else
            x, y = coords
            println("Using physical coords from HDF5 dataset")
        end
    end

    # step 2: sample timesteps
    t_max = Nt_v # get total timesteps
    n_steps = min(N_TIMESTEPS, t_max) # limit to N_TIMESTEPS
    timesteps = unique(round.(Int, range(1, t_max, length = n_steps))) # sample evenly

    println("Exporting time series for $LEVEL...")
    println("Exporting $(length(timesteps)) timesteps from $t_max total")

    # step 3: create output folder  
    outdir = replace(OUTPUT_DIR, r"_0$" => "_$LEVEL")
    mkpath(outdir)

    # step 4: create PVD collection file
    pvd = paraview_collection(joinpath(outdir, "fluid_level_$(LEVEL)_timeseries"))

    # step 6: export each timestep, call export timestep for each timestep
    for (i, t) in enumerate(timesteps)
        export_timestep(u_raw, v_raw, vort, x, y, lev, t, i, time, outdir, pvd)

        if i % 10 == 0 || i == length(timesteps)
            @printf("\r Progress: %d/%d", i, length(timesteps)) # update progress
            flush(stdout) # force display
        end
    end
    println()

    # step 7: save PVD collection
    silence(() -> vtk_save(pvd), QUIET_VTK)
    println(" Saved: $(outdir)/fluid_level_$(LEVEL)_timeseries.pvd")
end

function export_timestep(u_raw, v_raw, vort, x, y, lev, t, i, time, outdir, pvd)
    # input:
    # u_raw: 4D array (Nx, Ny, Nlev, Nt) (x-velocity with ghost cells)
    # v_raw: 4D array (y-velocity with ghost cells)
    # vort: 4D array (vorticity field)
    # x, y: coord vector
    # lev = refinement level
    # i: input file indexing
    # time: time array
    # outdir: output directory path
    # w_grp: vorticity from HDF5 dataset
    # time: 1D array of time values

    # output: None (create a VTS file)

    # step 1: extract vorticity (point data)
    @views vort_t = vort[:, :, lev, t]

    # step 2: average staggered velocities to cell centers
    # ux is (Nx+1, Ny) on x-faces → average along x → (Nx, Ny) cell-centered
    # uy is (Nx, Ny+1) on y-faces → average along y → (Nx, Ny) cell-centered
    @views u_lev = u_raw[:, :, lev, t]
    @views v_lev = v_raw[:, :, lev, t]
    ucell = 0.5 * (u_lev[1:end - 1, :] + u_lev[2:end, :])
    vcell = 0.5 * (v_lev[:, 1:end - 1] + v_lev[:, 2:end])

    # step 4: create structured grid file
    fname = @sprintf("fluid_%04d", i - 1)
    vtkpath = joinpath(outdir, fname)

    silence(QUIET_VTK) do 
        vtk_grid(vtkpath, x, y, [0.0]) do vtk
            # step 5: add vorticity as point data
            vtk["vorticity", VTKPointData()] = to_vtk_point(vort_t)

            # step 6: add u, v as cell data
            vtk["u", VTKCellData()] = to_vtk_cell(ucell)
            vtk["v", VTKCellData()] = to_vtk_cell(vcell)

            # step 7: add to PVD collection with timestamp
            actual_time = (length(time) >= t) ? Float64(time[t]) : Float(t - 1)
            pvd[actual_time] = vtk
        end
    end
end




# =============================================================================
# Run
# =============================================================================


try
    main()
catch e
    println("Error: $e")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    rethrow()
end



