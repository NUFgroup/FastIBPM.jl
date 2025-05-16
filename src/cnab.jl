abstract type AbstractCoupler end

struct NothingCoupler <: AbstractCoupler end

struct PrescribedBodyCoupler{M} <: AbstractCoupler
    Binv::M
end

@kwdef struct FsiCoupler{T,O<:GeometricNonlinearBodyOperators,B} <: AbstractCoupler
    state::StructuralState{T}
    ops::O
    tol::T
    bicgstabl_args::B
    maxiter::Int
end

function FsiCoupler(
    backend::CPU,
    body::GeometricNonlinearBody{N,T};
    tol=1e-5,
    bicgstabl_args=(; abstol=T(1e-5), reltol=T(0.0)),
    maxiter=100,
) where {N,T}
    n = deforming_point_count(body)
    nel = n - 1
    nf = N * point_count(body)

    state = StructuralState{T}(backend, structure_var_count(body))
    ops = structural_operators(backend, body)

    FsiCoupler(; state, ops, tol, bicgstabl_args, maxiter)
end

@kwdef mutable struct CNAB{
    N,T,B,U,P,R<:Reg,C<:AbstractCoupler,Au,Aω,Vb,BP<:BodyPoints,A<:ArrayPool,W
}
    const prob::IBProblem{N,T,B,U}
    const t0::T
    i::Int
    t::T
    const dt::T
    const β::Vector{T}
    const plan::P
    const reg::R
    const coupler::C
    const redist_weights::Au
    ω::Vector{Aω}
    ψ::Vector{Aω}
    const u::Vector{Au}
    const f_tilde::Vb
    const f::Vb
    const points::BP
    const nonlin::Vector{Vector{Aω}}
    nonlin_count::Int
    ω_bndry::W
    body_pool::A
    fluid_pool::A
    bndry_pool::A
    structure_pool::A
end

function CNAB(
    prob::IBProblem{N,T};
    dt,
    t0=zero(T),
    n_step=2,
    delta=DeltaYang3S(),
    backend=CPU(),
    coupler_args=(;),
) where {N,T}
    grid = prob.grid
    body = prob.body
    ω = grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels)

    plan = let ωe = grid_view(ω[1], grid, Loc_ω, ExcludeBoundary())
        laplacian_plans(ωe, grid.n)
    end

    n_ib = point_count(body)
    n_structure = structure_var_count(body)

    max_fluid_vars = maximum(
        loc -> grid_length(grid, loc, IncludeBoundary()), (Loc_u, Loc_ω)
    )
    max_bndry_vars = boundary_length(grid, Loc_ω)

    args = (;
        prob,
        t0,
        i=0,
        t=zero(T),
        dt,
        β=ab_coeffs(T, n_step),
        plan,
        reg=Reg(backend, T, delta, n_ib, Val(N)),
        redist_weights=grid_zeros(backend, grid, Loc_u),
        ω,
        ψ=grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels),
        u=grid_zeros(backend, grid, Loc_u; levels=1:grid.levels),
        f_tilde=KernelAbstractions.zeros(backend, SVector{N,T}, n_ib),
        f=KernelAbstractions.zeros(backend, SVector{N,T}, n_ib),
        points=BodyPoints{N,T}(backend, n_ib),
        nonlin=map(1:n_step-1) do _
            grid_zeros(backend, grid, Loc_ω, ExcludeBoundary(); levels=1:grid.levels)
        end,
        nonlin_count=0,
        ω_bndry=boundary_axes(grid, Loc_ω),
        body_pool=ArrayPool(backend, n_ib * sizeof(SVector{N,T})),
        fluid_pool=ArrayPool(backend, max_fluid_vars * sizeof(T)),
        bndry_pool=ArrayPool(backend, max_bndry_vars * sizeof(T)),
        structure_pool=ArrayPool(backend, n_structure * sizeof(T)),
    )

    sol = initial_sol(backend, body, args, coupler_args)

    sol
end

function initial_sol(backend, body::AbstractStaticBody, sol_args, coupler_args)
    sol0 = CNAB(; sol_args..., coupler=NothingCoupler())

    init_body_points!(sol0.points, body)
    update_weights!(sol0.reg, sol0.prob.grid, sol0.points.x, eachindex(sol0.points.x))
    Binv = B_inverse_rigid(sol0)

    coupler = PrescribedBodyCoupler(Binv; coupler_args...)
    sol = CNAB(; sol_args..., coupler)
    set_time!(sol, 0)
    zero_vorticity!(sol)
    update_redist_weights!(sol)

    sol
end

function initial_sol(
    backend, body::GeometricNonlinearBody{N,T}, sol_args, coupler_args
) where {N,T}
    coupler = FsiCoupler(backend, body; coupler_args...)
    sol = CNAB(; sol_args..., coupler)
    set_time!(sol, 0)
    zero_vorticity!(sol)

    i_deform = deforming_point_range(body)
    i_prescribed = prescribed_point_range(body)

    init_body_points!(view(sol.points, i_prescribed), body.prescribed)
    update_structure!(sol.points, coupler.state, body, coupler.ops, sol.i, sol.t)
    init_structure_operators!(coupler.ops, body, sol.points, coupler.state, sol.dt)
    update_weights!(sol.reg, sol.prob.grid, sol.points.x, eachindex(sol.points.x))
    update_redist_weights!(sol)

    sol
end

function zero_vorticity!(sol::CNAB)
    grid = sol.prob.grid

    for level in 1:grid.levels
        for i in eachindex(sol.ω[level])
            sol.ω[level][i] .= 0
            sol.ψ[level][i] .= 0
        end
        for i in eachindex(sol.u[level])
            sol.u[level][i] .= 0
        end
    end

    sol.nonlin_count = 0

    for level in eachindex(sol.u)
        add_flow!(sol.u[level], sol.prob.u0, grid, level, sol.i, sol.t)
    end

    sol
end

function set_time!(sol::CNAB, i::Integer)
    sol.i = i
    sol.t = sol.t0 + sol.dt * (i - 1)
    sol
end

function step!(sol::CNAB)
    set_time!(sol, sol.i + 1)

    prediction_step!(sol)
    coupling_step!(sol)
    projection_step!(sol)
    apply_vorticity!(sol)

    sol
end

update_reg!(::CNAB, ::AbstractStaticBody, _) = nothing
function update_reg!(sol::CNAB, ::AbstractPrescribedBody, i)
    update_weights!(sol.reg, sol.prob.grid, sol.points.x, i)
end

_A_factor(sol::CNAB) = sol.dt / (2sol.prob.Re)

function Ainv(sol::CNAB, level)
    h = gridstep(sol.prob.grid, level)
    a = _A_factor(sol)
    EigenbasisTransform(λ -> 1 / (1 - a * λ / h^2), sol.plan)
end

function prediction_step!(sol::CNAB)
    _cycle!(sol.nonlin)

    for level in sol.prob.grid.levels:-1:1
        prediction_step!(sol, level)
    end

    sol.nonlin_count = min(sol.nonlin_count + 1, length(sol.nonlin))
end

function prediction_step!(sol::CNAB{N,T}, level) where {N,T}
    u_axes = cell_axes(sol.prob.grid, Loc_u, ExcludeBoundary())
    with_arrays(sol.fluid_pool, (T, u_axes)) do u_work
        prediction_step!(sol::CNAB, level, u_work)
    end
end

function prediction_step!(sol::CNAB{N,T}, level, u_work) where {N,T}
    backend = get_backend(sol.u[1][1])

    grid = sol.prob.grid
    h = gridstep(grid, level)
    ωˢ = grid_view(sol.ψ[level], grid, Loc_ω, ExcludeBoundary())
    a = _A_factor(sol)

    curl!(u_work, sol.ω[level]; h)
    rot!(ωˢ, u_work; h)

    for i in eachindex(ωˢ)
        let ωˢ = ωˢ[i], ω = sol.ω[level][i]
            @loop backend (I in CartesianIndices(ωˢ)) begin
                ωˢ[I] = ω[I] - a * ωˢ[I]
            end
        end
    end

    if level < grid.levels
        with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψb
            multidomain_interpolate!(ψb, sol.ψ[level+1]; n=grid.n)
            add_laplacian_bc!(ωˢ, a / h^2, ψb)
        end
    end

    nonlin_full = sol.nonlin_count == length(sol.nonlin)

    if nonlin_full
        for i_step in eachindex(sol.nonlin), i in eachindex(ωˢ)
            let ωˢ = ωˢ[i], N = sol.nonlin[i_step][level][i], k = sol.dt * sol.β[end-i_step]
                @loop backend (I in CartesianIndices(ωˢ)) begin
                    ωˢ[I] = ωˢ[I] + k * N[I]
                end
            end
        end
    end

    nonlinear!(u_work, sol.u[level], sol.ω[level])
    rot!(sol.nonlin[end][level], u_work; h)

    for i in eachindex(ωˢ)
        let ωˢ = ωˢ[i],
            N = sol.nonlin[end][level][i],
            k = nonlin_full ? sol.dt * sol.β[end] : sol.dt

            @loop backend (I in CartesianIndices(ωˢ)) begin
                ωˢ[I] = ωˢ[I] + k * N[I]
            end
        end
    end

    Ainv(sol, level)(ωˢ, ωˢ)
end

coupling_step!(sol::CNAB) = _coupling_step!(sol, sol.coupler)

function _coupling_step!(sol::CNAB{N,T}, coupler::PrescribedBodyCoupler) where {N,T}
    with_arrays_like(sol.body_pool, sol.f_tilde) do rhs
        with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψ_b_work
            _coupling_step!(sol, coupler, rhs, ψ_b_work)
        end
    end
end

function _coupling_step!(sol::CNAB, coupler::PrescribedBodyCoupler, rhs, ψ_b_work)
    grid = sol.prob.grid
    body = sol.prob.body
    ωˢ = sol.ψ
    ψ = (sol.ω[1],)
    u¹ = sol.u[1]

    multidomain_poisson!(ωˢ, ψ, (u¹,), ψ_b_work, grid, sol.plan)
    add_flow!(u¹, sol.prob.u0, grid, 1, sol.i, sol.t)

    update_body_points!(sol.points, body, sol.i, sol.t)
    update_reg!(sol, body, eachindex(sol.points.x))
    interpolate_body!(rhs, sol.reg, u¹)

    rhs .-= sol.points.u

    coupler.Binv(sol.f_tilde, rhs, sol)
end

struct CNAB_Binv_Precomputed{M}
    B::M
end

function (x::CNAB_Binv_Precomputed)(f, u_ib, ::CNAB{N,T}) where {N,T}
    let f = reinterpret(T, f), u_ib = reinterpret(T, u_ib)
        ldiv!(f, x.B, u_ib)
    end
end

function B_inverse_rigid(sol::CNAB{N,T,<:AbstractStaticBody}) where {N,T}
    backend = get_backend(sol.f_tilde)
    n_ib = point_count(sol.prob.body)

    n = N * n_ib
    B_map = LinearMap(n; ismutating=true) do u_ib, f
        B_rigid_mul!(u_ib, f, sol)
    end
    B_mat = KernelAbstractions.zeros(backend, T, n, n)

    with_arrays(sol.body_pool, (T, (n,))) do f
        for i in 1:n
            @. f = ifelse((1:n) == i, 1, 0)
            mul!(@view(B_mat[:, i]), B_map, f)
        end
    end

    CNAB_Binv_Precomputed(cholesky!(Hermitian(B_mat)))
end

function B_rigid_mul!(u_ib::AbstractVector{<:Number}, f, sol::CNAB{N,T}) where {N,T}
    let u_ib = reinterpret(SVector{N,T}, u_ib), f = reinterpret(SVector{N,T}, f)
        B_rigid_mul!(u_ib, f, sol)
    end

    u_ib
end

function B_rigid_mul!(u_ib, f, sol::CNAB{N,T}) where {N,T}
    grid = sol.prob.grid
    h = grid.h
    ω = sol.ω
    ω¹ = grid_view(ω[1], grid, Loc_ω, ExcludeBoundary())

    with_arrays_like(sol.fluid_pool, sol.u[1], sol.ψ[1]) do u¹, ψ¹
        regularize!(u¹, sol.reg, f)
        rot!(ω¹, u¹; h)
        Ainv(sol, 1)(ω¹, ω¹)

        for level in 2:grid.levels, i in eachindex(ω[level])
            fill!(ω[level][i], 0)
        end

        with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψb
            multidomain_poisson!(ω, (ψ¹,), (u¹,), ψb, grid, sol.plan)
        end

        interpolate_body!(u_ib, sol.reg, u¹)
    end
end

function _coupling_step!(sol::CNAB{N,T}, coupler::FsiCoupler) where {N,T}
    with_arrays_like(sol.body_pool, ntuple(_ -> sol.f_tilde, 3)...) do fs...
        with_arrays_like(sol.structure_pool, ntuple(_ -> coupler.state.χ, 8)...) do χs...
            with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψ_b_work
                _coupling_step!(sol, coupler, fs, χs, ψ_b_work)
            end
        end
    end
end

function _coupling_step!(sol::CNAB{N,T}, coupler::FsiCoupler, fs, χs, ψ_b_work) where {N,T}
    (rhsf, F_kp1, F_sm) = fs
    (χ_k, ζ_k, ζdot_k, r_c, r_ζ, F_bg, Δχ, χ_temp) = χs
    (; χ, ζ, ζdot) = coupler.state

    grid = sol.prob.grid
    body = sol.prob.body
    ops = coupler.ops
    dt = sol.dt
    h = gridstep(grid)

    nf = N * point_count(body)
    B = LinearMap(nf; ismutating=true) do y, x
        B_deform_mul!(y, x, sol)
    end

    ωˢ = sol.ψ
    ψ = (sol.ω[1],)
    u¹ = sol.u[1]

    multidomain_poisson!(ωˢ, ψ, (u¹,), ψ_b_work, grid, sol.plan)
    add_flow!(u¹, sol.prob.u0, grid, 1, sol.i, sol.t)

    i_deform = deforming_point_range(body)
    i_prescribed = prescribed_point_range(body)
    update_body_points!(view(sol.points, i_prescribed), body.prescribed, sol.i, sol.t)

    update_reg!(sol, body.prescribed, i_prescribed)

    it = 0
    χ_k .= χ
    ζ_k .= ζ
    ζdot_k .= ζdot

    update_structure!(sol.points, coupler.state, body, coupler.ops, sol.i, sol.t)
    update_structure_operators!(ops, body, sol.points, coupler.state, sol.dt)
    update_weights!(sol.reg, grid, sol.points.x, i_deform)
    update_redist_weights!(sol)

    while true
        if it + 1 > coupler.maxiter
            error("exceeded maximum iteration count")
        end

        interpolate_body!(F_kp1, sol.reg, u¹)

        @views @. F_kp1[i_prescribed] -= sol.points.u[i_prescribed]

        @. r_c = 2 / dt * (χ_k - χ) - ζ

        @. χ_temp = ζdot + 4 / dt * ζ + 4 / dt^2 * (χ - χ_k)
        mul!(r_ζ, ops.M, χ_temp)
        @. χ_temp = r_ζ - ops.Fint

        ldiv!(r_ζ, ops.Khat, χ_temp)

        @. F_bg = -(2 / dt * r_ζ + r_c)

        fill!(F_sm, zero(SVector{N,T}))
        structure_to_fluid_displacement!(view(F_sm, i_deform), F_bg, body, ops)
        @. rhsf = F_sm + F_kp1

        bicgstabl!(
            reinterpret(T, sol.f_tilde), B, reinterpret(T, rhsf); coupler.bicgstabl_args...
        )

        # Redistribute
        sol.f .= sol.f_tilde
        f_to_f_tilde!(sol.f, sol; inverse=true)
        redist!(sol.f, sol)

        fluid_to_structure_force!(χ_temp, view(sol.f, i_deform), body, ops)
        ldiv!(Δχ, ops.Khat, χ_temp)
        @. Δχ += r_ζ

        χ_norm = norm(χ_k, Inf)
        Δχ_norm = norm(Δχ, Inf)
        err = χ_norm > 1e-13 ? Δχ_norm / χ_norm : Δχ_norm

        @. χ_k = χ_k + Δχ
        update_structure_bc!(χ_k, body, sol.i, sol.t)

        @. ζ_k = -ζ + 2 / dt * (χ_k - χ)
        @. ζdot_k = 4 / dt^2 * (χ_k - χ) - 4 / dt * ζ - ζdot

        state_k = StructuralState(χ_k, ζ_k, ζdot_k)
        update_structure!(sol.points, state_k, body, coupler.ops, sol.i, sol.t)
        update_structure_operators!(ops, body, sol.points, state_k, sol.dt)
        update_weights!(sol.reg, grid, sol.points.x, i_deform)
        update_redist_weights!(sol)

        if err < coupler.tol
            break
        end

        it += 1
    end

    χ .= χ_k
    ζ .= ζ_k
    ζdot .= ζdot_k

    nothing
end

function B_deform_mul!(u_ib::AbstractVector{<:Number}, f, sol::CNAB{N,T}) where {N,T}
    S = SVector{N,T}
    B_deform_mul!(reinterpret(S, u_ib), reinterpret(S, f), sol)
end

function B_deform_mul!(u_ib, f, sol::CNAB)
    χ = sol.coupler.state.χ
    with_arrays_like(sol.body_pool, sol.f_tilde) do f_work
        with_arrays_like(sol.structure_pool, χ, χ) do f1, f2
            B_deform_mul!(u_ib, f, sol::CNAB, f_work, f1, f2)
        end
    end
end

function B_deform_mul!(u_ib, f, sol::CNAB, f_work, f1, f2)
    grid = sol.prob.grid
    body = sol.prob.body::GeometricNonlinearBody
    h = gridstep(grid)
    dt = sol.dt
    i_deform = deforming_point_range(body)
    u_ib_deform = view(u_ib, i_deform)
    f_work_deform = view(f_work, i_deform)

    u_ib .= f
    f_to_f_tilde!(u_ib, sol; inverse=true)
    redist!(u_ib, sol)
    fluid_to_structure_force!(f1, u_ib_deform, body, sol.coupler.ops)
    ldiv!(f2, sol.coupler.ops.Khat, f1)
    structure_to_fluid_displacement!(f_work_deform, f2, body, sol.coupler.ops)
    f_work_deform .*= 2 / dt

    B_rigid_mul!(u_ib, f, sol)
    u_ib_deform .+= f_work_deform

    u_ib
end

function f_to_f_tilde!(f, sol::CNAB; inverse=false)
    dt = sol.dt
    ds = @view sol.points.ds[eachindex(f)]
    h = sol.prob.grid.h
    k = _f_tilde_factor(sol)

    if inverse
        @. f *= -k / ds
    else
        @. f *= ds / -k
    end
end

function redist!(f, sol::CNAB{N,T}) where {N,T}
    with_arrays_like(sol.fluid_pool, sol.u[1]) do u_work
        regularize!(u_work, sol.reg, f)

        for i in eachindex(u_work)
            u_work[i] .*= sol.redist_weights[i]
        end

        interpolate_body!(f, sol.reg, u_work)
    end
end

function update_redist_weights!(sol::CNAB{N,T}; tol=T(1e-10)) where {N,T}
    w = sol.redist_weights
    backend = get_backend(w[1])

    with_arrays_like(sol.fluid_pool, sol.f_tilde) do f
        reinterpret(T, f) .= 1
        regularize!(w, sol.reg, f)
    end

    for wi in w
        @loop backend (I in CartesianIndices(wi)) begin
            wi[I] = wi[I] < tol ? zero(T) : 1 / wi[I]
        end
    end
end

function projection_step!(sol::CNAB{N,T}) where {N,T}
    grid = sol.prob.grid
    backend = get_backend(sol.u[1][1])

    u_axes = cell_axes(grid, Loc_u, IncludeBoundary())
    ω_axes = cell_axes(grid, Loc_ω, ExcludeBoundary())

    with_arrays(sol.fluid_pool, (T, u_axes), (T, ω_axes)) do u_work, ω_work
        regularize!(u_work, sol.reg, sol.f_tilde)
        rot!(ω_work, u_work; h=grid.h)
        Ainv(sol, 1)(ω_work, ω_work)

        (sol.ω, sol.ψ) = (sol.ψ, sol.ω)

        for i in eachindex(ω_work)
            let ω = sol.ω[1][i], ω_work = ω_work[i]
                @loop backend (I in CartesianIndices(ω_work)) begin
                    ω[I] -= ω_work[I]
                end
            end
        end
    end
end

function apply_vorticity!(sol::CNAB{N,T}) where {N,T}
    with_arrays(sol.bndry_pool, (T, sol.ω_bndry)) do ψ_b_work
        apply_vorticity!(sol, ψ_b_work)
    end
end

function apply_vorticity!(sol::CNAB, ψ_b_work)
    grid = sol.prob.grid
    multidomain_poisson!(sol.ω, sol.ψ, sol.u, ψ_b_work, grid, sol.plan)

    for level in 1:grid.levels
        if level == grid.levels
            for i in eachindex(ψ_b_work)
                foreach(b -> fill!(b, 0), ψ_b_work[i])
            end
        else
            multidomain_interpolate!(ψ_b_work, sol.ω[level+1]; n=grid.n)
        end

        set_boundary!(sol.ω[level], ψ_b_work)

        add_flow!(sol.u[level], sol.prob.u0, grid, level, sol.i, sol.t)
    end
end

function ab_coeffs(T, n)
    if n == 1
        T[1]
    elseif n == 2
        T[-1//2, 3//2]
    else
        throw(DomainError(n, "only n=1 and n=2 are supported"))
    end
end

function _f_tilde_factor(sol::CNAB{N}) where {N}
    grid = sol.prob.grid
    -grid.h^N / sol.dt
end

function surface_force!(f, sol::CNAB)
    k = _f_tilde_factor(sol)
    @. f = -k * sol.f_tilde
end

function surface_force_sum(sol::CNAB)
    k = _f_tilde_factor(sol)
    -k * sum(sol.f_tilde)
end

const CNAB_signature = Vector{UInt8}("FastIBPM.jl:CNAB")

function save(io::IO, sol::CNAB{N,T}) where {N,T}
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

function load!(io::IO, sol::CNAB{N,T}) where {N,T}
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
