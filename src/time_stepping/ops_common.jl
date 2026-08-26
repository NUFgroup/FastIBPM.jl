"""
Assembly operators shared by more than one formulation.

  - `_A_factor` — the diffusion coefficient `a = Δt/(2Re)`, formulation-independent.
  - `Ainv`      — the master entry point for the viscous inverse, dispatching on
                  `sol.prob.formulation`.
  - `Ainv_zeros`, `_interior_range`, `_apply_aL!`, `B_work` — the haloed-field
                  machinery of the primitive formulations, used by both `IBPM`
                  and `IMAP`.

The formulation-specific operators live in `ops_fastibpm.jl`, `ops_ibpm.jl` and
`ops_imap.jl`; the time steppers that call them are in the `cnab_*.jl` files.
"""

"""
    _A_factor(sol::CNAB)

Diffusion coefficient `a = Δt / (2 Re)` of the semi-implicit (Crank-Nicolson)
viscous term. Formulation-independent.
"""
_A_factor(sol::CNAB) = sol.dt / (2sol.prob.Re)

"""
    Ainv(sol::CNAB, level)

Inverse of the implicit viscous operator `A = I - aΔ`, with `a = Δt/(2Re)`
([`_A_factor`](@ref)). Master entry point: dispatches on the problem formulation
(`sol.prob.formulation`) to the formulation-specific method.

The two formulations invert `A` by genuinely different means, and — unlike
`CNAB_Binv_Iterative` — they do **not** yet share this entry point:

  - **`FastIBPM`** — [`Ainv(sol, level, ::FastIBPM)`](@ref), in `ops_fastibpm.jl`. Returns an
    *operator object* (an `EigenbasisTransform`) that applies `(I - aΔ)⁻¹`
    spectrally; it is used as `Ainv(sol, level)(y, x)`.
  - **`IBPM`** — the viscous inverse is [`Ainv!`](@ref), the truncated-Taylor
    `Bᴺ ≈ A⁻¹` built from repeated Laplacian applications (no spectral solve).
    It has a *different interface*: it applies `A⁻¹` **in place** and needs haloed
    velocity fields plus scratch ([`Ainv_zeros`](@ref)), so it cannot be returned
    from this entry point as an operator object.

!!! note
    Consequently there is **no `Ainv(sol, level, ::IBPM)` method**: calling this
    entry point on an `IBPM` problem is a `MethodError` by design. Use [`Ainv!`](@ref)
    directly. A wrapper making `IBPM` reachable here would have to carry the
    haloed work fields, so it belongs with the primitive time-stepper.
"""
Ainv(sol::CNAB, level) = Ainv(sol, level, sol.prob.formulation)

"""
    Ainv_zeros(backend, grid)

Allocate zero-filled "haloed" velocity work fields for the `Ainv!` (`Bᴺ`) apply.

Each velocity component `u[i]` is stored over its interior-unknown box
(`Loc_u(i)`, `ExcludeBoundary`) expanded by one cell on every side. That extra
ring is a homogeneous ghost halo: inside `Bᴺ` the boundary conditions live on the
right-hand side (`bc1`), so the Laplacian acts with zero boundary/ghost values,
and the (zero) halo lets the stencil read its neighbors without going out of
bounds — including the transverse edges the plain `Loc_u` layout omits.
"""
function Ainv_zeros(backend, grid::AbstractGrid{N,T}) where {N,T}
    map(edge_axes(Val(N), Loc_u)) do i
        Re = cell_axes(grid, Loc_u(i), ExcludeBoundary())
        Rh = map(r -> (first(r)-1):(last(r)+1), Re)
        OffsetArray(KernelAbstractions.zeros(backend, T, length.(Rh)), Rh)
    end
end

# Interior index ranges of a haloed field: its axes shrunk by one cell on every
# side (inverse of the expansion in `Ainv_zeros`) — the box the Laplacian is
# evaluated over.
_interior_range(a) = map(r -> (first(r)+1):(last(r)-1), UnitRange.(axes(a)))

"""
    _apply_aL!(dest, src, a, h)

Compute `dest = a · L src` over the interior of each component, in place, where
`L` is the discrete Laplacian (`laplacian`). `src` must be a haloed field (from
`Ainv_zeros`) so the stencil can read its neighbors; only `dest`'s interior is
written.
"""
function _apply_aL!(dest, src, a, h)
    for i in eachindex(dest)
        d = dest[i]
        s = src[i]
        backend = get_backend(d)
        R = CartesianIndices(_interior_range(d))
        @loop backend (I in R) d[I] = a * laplacian(s, I; h)
    end
    dest
end

"""
    B_work(backend, grid)

Allocate the haloed velocity scratch fields needed by the primitive-variable
projection-step operators: the intermediate `Q λ`, the result `Bᴺ Q λ`, and the
two scratch fields consumed by the viscous inverse.

Shared by both primitive formulations — `IBPM`'s `B_mul!`/`Ainv!` and `IMAP`'s
`B_mul!`/`Ainv_IMAP!` need the same four fields — and by their solver states.
"""
function B_work(backend, grid::AbstractGrid)
    (;
        q=Ainv_zeros(backend, grid),
        y=Ainv_zeros(backend, grid),
        term=Ainv_zeros(backend, grid),
        tmp=Ainv_zeros(backend, grid),
    )
end
