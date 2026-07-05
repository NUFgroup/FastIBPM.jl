"""
Post-processing utilities: checkpoint I/O and surface force extraction.

  - surface_force!, surface_force_sum  — physical force recovery from f_tilde
  - save, load!                         — binary checkpoint serialisation
  - CNAB_signature                      — magic bytes for file format identification

# TODO: remove the surface_force! and surface_force_sum stub declarations in
# init/problems.jl once all includes are confirmed working — Julia does not need
# forward declarations; the first method definition here creates the function.
"""

"""
    surface_force!(f, sol)

Convert the redistributed force `f_tilde` back to the physical surface force `f`.

# Arguments
- `f`: Output array that will store the physical surface force.
- `sol`: CNAB solver object containing `f_tilde` and grid/time info.

# Behavior
- Computes a scaling factor `k = -h^N / Δt` using `_f_tilde_factor(sol)`.
- Applies the formula `f .= -k * sol.f_tilde` to recover the actual force on the body surface.

# Notes
This reverses the scaling applied in `f_to_f_tilde!`.

# Returns
- `nothing`: The physical surface force is written in-place to `f`.
"""
function surface_force!(f, sol::CNAB)
    k = _f_tilde_factor(sol)
    @. f = -k * sol.f_tilde
end

"""
    surface_force_sum(sol)

Compute the total hydrodynamic force exerted by the fluid on the immersed body.

# Arguments
- `sol`: CNAB solver object containing `f_tilde` and grid/time info.

# Behavior
- Computes the scaling factor `k = -h^N / Δt` using `_f_tilde_factor(sol)`.
- Sums all entries of `sol.f_tilde` (the redistributed force at Lagrangian points).
- Scales and flips the sign to recover the physical total force:
  `total_force = -k * sum(sol.f_tilde)`.

# Returns
- `total_force`: The net physical force vector acting on the body.
"""
function surface_force_sum(sol::CNAB)
    k = _f_tilde_factor(sol)
    -k * sum(sol.f_tilde)
end

"""
    const CNAB_signature

A compile-time constant used as a unique identifier for the `CNAB` structure.
It stores the string `Immersa.jl:CNAB` as a vector of bytes (`Vector{UInt8}`).

This signature can be used, for example, in type-checking, serialization, or validation routines.
"""
const CNAB_signature = Vector{UInt8}("Immersa.jl:CNAB")

"""
    save(io::IO, sol::CNAB)

Serialize the current state of a `CNAB` simulation to a binary I/O stream.

The binary layout (all integers little-endian) is:

1. `CNAB_signature` (magic bytes).
2. `UInt32` scalar size (`sizeof(T)`).
3. `UInt32` spatial dimension `N`.
4. `SVector{N,UInt32}` grid cell counts.
5. `UInt32` number of multigrid levels.
6. `Int32` current time step index.
7. Vorticity fields (interior values only, little-endian `T`), one block per
   component per level.
8. `UInt32` nonlinear history count.
9. Nonlinear history buffers (same layout as vorticity).

Use [`load!`](@ref) to restore the state into an existing `CNAB` object.

# Arguments
- `io::IO`      : Output stream (e.g. an open file).
- `sol::CNAB`   : Solver state to serialise.
"""
save(io::IO, sol::CNAB) = save(io, sol, sol.prob.formulation)

function save(io::IO, sol::CNAB{N,T}, ::FastIBPM) where {N,T}
    grid = sol.prob.grid

    write(io, CNAB_signature)
    write(io, htol(UInt32(sizeof(T))))
    write(io, htol(UInt32(N)))
    write(io, htol.(SVector{N,UInt32}(grid.n)))
    write(io, htol(UInt32(grid.levels)))
    write(io, htol(Int32(sol.i)))

    # Messy because copy!(a, b) where b is a CUDA array only seems to work when a is an
    # Array (not a view or OffsetArray).

    let ω_tmp = map(ax -> zeros(T, length.(ax)), cell_axes(grid, Loc_ω, IncludeBoundary()))
        for ω_lev in sol.ω, (i, ω_i) in pairs(ω_lev)
            copy!(ω_tmp[i], no_offset_view(ω_i))
            a = view(
                OffsetArray(ω_tmp[i], axes(ω_i)),
                cell_axes(grid, Loc_ω(i), ExcludeBoundary())...,
            )
            @. a = htol(a)
            write(io, a)
        end
    end

    write(io, htol(UInt32(sol.nonlin_count)))

    let ω_tmp = map(ax -> zeros(T, length.(ax)), cell_axes(grid, Loc_ω, ExcludeBoundary()))
        for k in 1:sol.nonlin_count,
            nonlin_lev in sol.nonlin[k],
            (i, nonlin_i) in pairs(nonlin_lev)

            copy!(ω_tmp[i], no_offset_view(nonlin_i))
            @. ω_tmp[i] = htol(ω_tmp[i])
            write(io, ω_tmp[i])
        end
    end

    nothing
end

"""
    load!(io::IO, sol::CNAB)

Restore a previously saved `CNAB` simulation state from a binary I/O stream into
an existing solver object.

Reads the binary format produced by [`save`](@ref), verifies the magic signature
and grid parameters, then overwrites the vorticity, time step, and nonlinear
history in `sol`. The solver is left in a consistent state ready for further
time-stepping.

# Arguments
- `io::IO`    : Input stream positioned at the start of a saved CNAB block.
- `sol::CNAB` : Solver object to populate (modified in place).
"""
load!(io::IO, sol::CNAB) = load!(io, sol, sol.prob.formulation)

function load!(io::IO, sol::CNAB{N,T}, ::FastIBPM) where {N,T}
    grid = sol.prob.grid

    @assert read(io, length(CNAB_signature)) == CNAB_signature
    @assert ltoh(read(io, UInt32)) == sizeof(T)
    @assert ltoh(read(io, UInt32)) == N
    @assert ltoh.(read(io, SVector{N,UInt32})) == grid.n
    @assert ltoh(read(io, UInt32)) == grid.levels

    i = Int(ltoh(read(io, Int32)))
    set_time!(sol, i)

    let ω_tmp = map(ax -> zeros(T, length.(ax)), cell_axes(grid, Loc_ω, IncludeBoundary()))
        for ω_lev in sol.ω, (i, ω_i) in pairs(ω_lev)
            a = view(
                OffsetArray(ω_tmp[i], axes(ω_i)),
                cell_axes(grid, Loc_ω(i), ExcludeBoundary())...,
            )
            read!(io, a)
            @. a = ltoh(a)
            copy!(no_offset_view(ω_i), ω_tmp[i])
        end
    end

    sol.nonlin_count = ltoh(read(io, UInt32))

    let ω_tmp = map(ax -> zeros(T, length.(ax)), cell_axes(grid, Loc_ω, ExcludeBoundary()))
        for k in 1:sol.nonlin_count,
            nonlin_lev in sol.nonlin[k],
            (i, nonlin_i) in pairs(nonlin_lev)

            a = ω_tmp[i]
            read!(io, a)
            @. a = ltoh(a)
            copy!(no_offset_view(nonlin_i), a)
        end
    end

    apply_vorticity!(sol)

    nothing
end

