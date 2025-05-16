function nonlinear!(nonlin, u, ω)
    backend = get_backend(nonlin[1])
    for (i, nonlinᵢ) in pairs(nonlin)
        @loop backend (I in CartesianIndices(nonlinᵢ)) begin
            nonlinᵢ[I] = nonlinear(i, u, ω, I)
        end
    end
    nonlin
end

function nonlinear(i, u, ω, I)
    δ = axisunit(I)
    sumcross(i, vec_kind(u), vec_kind(ω)) do j, k
        uI = (u[j][I] + u[j][I-δ(i)] + u[j][I+δ(j)] + u[j][I-δ(i)+δ(j)]) / 4
        ωI = (ω[k][I] + ω[k][I+δ(j)]) / 2
        uI * ωI
    end
end

function rot!(ω, u; h)
    backend = get_backend(ω[3])
    for (i, ωᵢ) in pairs(ω)
        @loop backend (I in CartesianIndices(ωᵢ)) begin
            ωᵢ[I] = rot(i, u, I; h)
        end
    end
    ω
end

function rot(i, u, I; h)
    δ = axisunit(I)
    sumcross(i) do j, k
        (u[k][I] - u[k][I-δ(j)]) / h
    end
end

function curl!(u, ψ; h)
    backend = get_backend(u[1])
    for (i, uᵢ) in pairs(u)
        @loop backend (I in CartesianIndices(uᵢ)) begin
            uᵢ[I] = curl(i, ψ, I; h)
        end
    end
    u
end

function curl(i, ψ, I; h)
    δ = axisunit(I)
    sumcross(i, Vec(), vec_kind(ψ)) do j, k
        (ψ[k][I+δ(j)] - ψ[k][I]) / h
    end
end

struct LaplacianPlan{P1,P2,L<:AbstractArray,W<:AbstractArray}
    λ::L
    work::W
    fwd::P1
    inv::P2
    n_logical::Int
end

function LaplacianPlan(ωᵢ, i, n::SVector{N}) where {N}
    R = cell_axes(n, Loc_ω(i), ExcludeBoundary())
    nω = length.(R)
    λ = OffsetArray(similar(ωᵢ, nω), R)
    laplacian_eigvals!(λ, i)

    kind = laplacian_fft_kind(i, N)
    flags = FFTW.EXHAUSTIVE
    fwd = FFT_R2R.plan_r2r!(ωᵢ, kind; flags)
    inv = FFT_R2R.plan_r2r!(ωᵢ, map(k -> FFTW.inv_kind[k], kind); flags)
    n_logical = prod(map(FFTW.logical_size, nω, kind))

    LaplacianPlan(λ, similar(ωᵢ), fwd, inv, n_logical)
end

laplacian_fft_kind(i, nd) = ntuple(j -> i == j ? FFTW.REDFT01 : FFTW.RODFT00, nd)

function laplacian_eigvals!(λ, i)
    backend = get_backend(λ)
    nd = ndims(λ)
    R = CartesianIndices(λ)
    n = size(λ)
    @loop backend (I in R) begin
        I₁ = Tuple(I - first(R)) .+ 1
        s = zero(eltype(λ))
        for j in 1:nd
            s += if (i == j)
                -4 * sin(π * (I₁[j] - 1) / (2n[j]))^2
            else
                -4 * sin(π * I₁[j] / (2(n[j] + 1)))^2
            end
        end
        λ[I] = s
    end
    λ
end

laplacian_plans(ω, n) = map(i -> LaplacianPlan(ω[i], i, n), tupleindices(ω))

struct EigenbasisTransform{F,O,P<:Tuple{Vararg{LaplacianPlan}}}
    f::F
    plan::OffsetTuple{O,P}
end

EigenbasisTransform(f, plan::Tuple) = EigenbasisTransform(f, OffsetTuple(plan))

function (X::EigenbasisTransform)(y, ω)
    for i in eachindex(ω)
        X(y[i], ω[i], i)
    end
    y
end

function (X::EigenbasisTransform)(yᵢ, ωᵢ, i)
    plan = X.plan[i]
    _set!(plan.work, ωᵢ)
    let λ = no_offset_view(plan.λ), a = no_offset_view(plan.work)
        mul!(a, plan.inv, a)
        @. a *= X.f(λ) / plan.n_logical
        mul!(a, plan.fwd, a)
    end
    _set!(yᵢ, plan.work)
end

function multidomain_coarsen!(ω², ω¹; n)
    backend = get_backend(ω²[3])
    for i in eachindex(ω²)
        R = CartesianIndices(_coarse_indices(Tuple(n), Loc_ω(i)))
        @loop backend (I in R) ω²[i][I] = multidomain_coarsen(i, ω¹[i], I; n)
    end
    ω²
end

function _coarse_indices(n::NTuple{N}, loc::Edge{Dual}) where {N}
    ntuple(N) do i
        n4 = n[i] .÷ 4
        i == loc.i ? (n4:3n4-1) : (n4+1:3n4-1)
    end
end

function multidomain_coarsen(i, ωᵢ, I²; n)
    T = eltype(ωᵢ)
    stencil = _coarsen_stencil(T)
    s = zero(T)
    indices = _fine_indices(i, Tuple(n), Tuple(I²))
    for I¹ in indices
        s += sum_map(*, SMatrix{3,3}(@view ωᵢ[I¹]), stencil)
    end
    s / length(indices)
end

function _coarsen_stencil(T)
    (@SMatrix [
        1 2 1
        2 4 2
        1 2 1
    ]) / T(16)
end

_fine_indices(_, n::NTuple{2}, I::NTuple{2}) = (CartesianIndices(_fine_range.(n, I)),)

function _fine_indices(i, n::NTuple{3}, I::NTuple{3})
    plane1 = 2(I[i] - (n[i] ÷ 4))
    r = _fine_range.(n, I)
    ntuple(2) do plane
        j = plane1 + plane - 1
        CartesianIndices(setindex(r, j:j, i))
    end
end

function _fine_range(n::Int, I::Int)
    2(I - (n ÷ 4)) .+ (-1:1)
end

function multidomain_interpolate!(ωb, ω; n)
    backend = get_backend(ω[3])
    for i in eachindex(ω), (j, k) in axes_permutations(i), dir in 1:2
        b = ωb[i][dir, k]
        @loop backend (I in CartesianIndices(b)) begin
            b[I] = multidomain_interpolate(ω[i], (i, j, k), dir, I; n)
        end
    end
    ωb
end

function multidomain_interpolate(ωᵢ, (i, j, k), dir, I¹::CartesianIndex{2}; n)
    δ = axisunit(Val(2))
    I² = CartesianIndex(ntuple(dim -> n[dim] ÷ 4 + fld(I¹[dim], 2), 2))
    if iseven(I¹[j])
        ωᵢ[I²]
    else
        (ωᵢ[I²] + ωᵢ[I²+δ(j)]) / 2
    end
end

function multidomain_interpolate(ωᵢ, (i, j, k), dir, I¹::CartesianIndex{3}; n)
    δ = axisunit(Val(3))
    n4 = Tuple(n) .÷ 4
    I² = CartesianIndex(
        ntuple(3) do dim
            if dim == i
                n4[dim] + fld(I¹[dim] - 1, 2)
            else
                n4[dim] + fld(I¹[dim], 2)
            end
        end,
    )
    a = (1 + 2mod(I¹[i] + 1, 2)) / 4
    if iseven(I¹[j])
        (1 - a) * ωᵢ[I²] + a * ωᵢ[I²+δ(i)]
    else
        ((1 - a) * (ωᵢ[I²] + ωᵢ[I²+δ(j)]) + a * (ωᵢ[I²+δ(i)] + ωᵢ[I²+δ(i)+δ(j)])) / 2
    end
end

function set_boundary!(ω, ωb)
    backend = get_backend(ω[3])
    for i in eachindex(ω), b in ωb[i]
        if length(b) > 0
            @loop backend (I in CartesianIndices(b)) begin
                ω[i][I] = b[I]
            end
        end
    end
    ω
end

function add_laplacian_bc!(Lψ, factor, ψb)
    backend = get_backend(Lψ[3])

    for i in eachindex(Lψ), j in 1:ndims(Lψ[i]), dir in 1:2
        ax = UnitRange.(axes(Lψ[i]))
        if i == j
            let Iᵢ = (ax[i][begin], ax[i][end])[dir],
                R = CartesianIndices(setindex(ax, Iᵢ:Iᵢ, i)),
                # StaticArrays doesn't adapt data for GPU, so use a tuple of tuples.
                ψb = map(_nd_tuple, ψb)

                @loop backend (I in R) begin
                    Lψ[i][I] += factor * laplacian_bc_ii(ψb, i, dir, I)
                end
            end
        else
            let b = ψb[i][dir, j],
                rb = axes(b, j),
                Iⱼ = (rb[begin], rb[end])[dir],
                R = CartesianIndices(setindex(ax, Iⱼ:Iⱼ, j))

                @loop backend (I in R) begin
                    δ = axisunit(I)
                    Lψ[i][I-outward(dir)*δ(j)] += factor * b[I]
                end
            end
        end
    end
end

function laplacian_bc_ii(ψb, i, dir, I)
    δ = axisunit(I)
    T = eltype(ψb[3][1][1])
    Iₒ = I + (dir - 1) * δ(i)
    s = zero(T)
    for (j, _) in axes_permutations(i)
        b = ψb[j][i][dir]
        s += b[Iₒ] - b[Iₒ-δ(j)]
    end
    -outward(dir) * s
end

function multidomain_poisson!(ω, ψ, u, ψb, grid::Grid, fft_plan)
    Base.require_one_based_indexing(ψ)

    for level in 2:grid.levels
        multidomain_coarsen!(ω[level], ω[level-1]; n=grid.n)
    end

    for level in grid.levels:-1:1
        h = gridstep(grid, level)
        ψi = ψ[min(lastindex(ψ), level)]
        ψe = _exclude_boundary(ψi, grid, Loc_ω)

        if level == grid.levels
            for i in eachindex(ψe)
                _set!(ψe[i], ω[level][i])
                foreach(b -> fill!(b, 0), ψb[i])
            end
        else
            let ψci = ψ[min(lastindex(ψ), level + 1)],
                ψce = _exclude_boundary(ψci, grid, Loc_ω)

                multidomain_interpolate!(ψb, ψce; n=grid.n)
            end
            for i in eachindex(ψe)
                _set!(ψe[i], ω[level][i])
            end
            add_laplacian_bc!(ψe, 1 / h^2, ψb)
        end

        EigenbasisTransform(λ -> -1 / (λ / h^2), fft_plan)(ψe, ψe)

        set_boundary!(ψi, ψb)

        if level in eachindex(u)
            curl!(u[level], ψi; h)
        end
    end
end

abstract type AbstractDeltaFunc end

(delta::AbstractDeltaFunc)(r::AbstractVector) = prod(delta, r)

struct DeltaYang3S <: AbstractDeltaFunc end
support(::DeltaYang3S) = 2

function (::DeltaYang3S)(r::AbstractFloat)
    u = one(r)
    a = abs(r)
    if a < 1
        17u / 48 + sqrt(3u) * π / 108 + a / 4 - r^2 / 4 +
        (1 - 2 * a) / 16 * sqrt(-12 * r^2 + 12 * a + 1) -
        sqrt(3u) / 12 * asin(sqrt(3u) / 2 * (2 * a - 1))
    elseif a < 2
        55u / 48 - sqrt(3u) * π / 108 - 13 * a / 12 +
        r^2 / 4 +
        (2 * a - 3) / 48 * sqrt(-12 * r^2 + 36 * a - 23) +
        sqrt(3u) / 36 * asin(sqrt(3u) / 2 * (2 * a - 3))
    else
        zero(r)
    end
end

struct DeltaYang3S2 <: AbstractDeltaFunc end
support(::DeltaYang3S2) = 3

function (::DeltaYang3S2)(x::Float64)
    r = abs(x)
    r2 = r * r
    r3 = r2 * r
    r4 = r3 * r

    if r <= 1.0
        a5 = asin((1.0 / 2.0) * sqrt(3.0) * (2.0 * r - 1.0))
        a8 = sqrt(1.0 - 12.0 * r2 + 12.0 * r)

        4.166666667e-2 * r4 +
        (-0.1388888889 + 3.472222222e-2 * a8) * r3 +
        (-7.121664902e-2 - 5.208333333e-2 * a8 + 0.2405626122 * a5) * r2 +
        (-0.2405626122 * a5 - 0.3792313933 + 0.1012731481 * a8) * r +
        8.0187537413e-2 * a5 - 4.195601852e-2 * a8 + 0.6485698427

    elseif r <= 2.0
        a6 = asin((1.0 / 2.0) * sqrt(3.0) * (-3.0 + 2.0 * r))
        a9 = sqrt(-23.0 + 36.0 * r - 12.0 * r2)

        -6.250000000e-2 * r4 +
        (0.4861111111 - 1.736111111e-2 * a9) .* r3 +
        (-1.143175026 + 7.812500000e-2 * a9 - 0.1202813061 * a6) * r2 +
        (0.8751991178 + 0.3608439183 * a6 - 0.1548032407 * a9) * r - 0.2806563809 * a6 +
        8.22848104e-3 +
        0.1150173611 * a9

    elseif r <= 3.0
        a1 = asin((1.0 / 2.0 * (2.0 * r - 5.0)) * sqrt(3.0))
        a7 = sqrt(-71.0 - 12.0 * r2 + 60.0 * r)

        2.083333333e-2 * r4 +
        (3.472222222e-3 * a7 - 0.2638888889) * r3 +
        (1.214391675 - 2.604166667e-2 * a7 + 2.405626122e-2 * a1) * r2 +
        (-0.1202813061 * a1 - 2.449273192 + 7.262731481e-2 * a7) * r +
        0.1523563211 * a1 +
        1.843201677 - 7.306134259e-2 * a7
    else
        0.0
    end
end

struct Reg{
    D<:AbstractDeltaFunc,T,N,A<:AbstractArray{SVector{N,Int},2},M,W<:AbstractArray{T,M}
}
    delta::D
    I::A
    weights::W
end

Adapt.@adapt_structure Reg

function Reg(backend, T, delta, nb, ::Val{N}) where {N}
    I = KernelAbstractions.zeros(backend, SVector{N,Int}, nb, N)

    s = support(delta)
    r = ntuple(_ -> length(-s:s), N)
    weights = KernelAbstractions.zeros(backend, T, r..., nb, N)

    Reg(delta, I, weights)
end

function update_weights!(reg::Reg, grid::Grid{N}, xbs, ibs) where {N}
    isempty(ibs) && return reg

    backend = get_backend(reg.weights)
    for i in 1:N
        @loop backend (J in CartesianIndices(ibs)) begin
            ib = ibs[J[1]]
            xb = xbs[ib]

            xu0 = coord(grid, Loc_u(i), zeros(SVector{N,Int}))
            reg.I[ib, i] = I = @. round(Int, (xb - xu0) / grid.h)

            for k in CartesianIndices(axes(reg.weights)[1:N])
                ΔI = (-support(reg.delta) - 1) .+ SVector(Tuple(k))
                xu = coord(grid, Loc_u(i), I + ΔI)
                reg.weights[k, ib, i] = reg.delta((xb - xu) / grid.h)
            end
        end
    end
    reg
end

function interpolate_body!(ub, reg::Reg{<:Any,T,N}, u) where {T,N}
    s = support(reg.delta)
    backend = get_backend(ub)
    @loop backend (J in CartesianIndices(ub)) begin
        ib = J[1]
        ubJ = zero(MVector{N,T})
        for i in 1:N
            w = @view reg.weights[.., ib, i]
            Ib = reg.I[ib, i]
            I = CartesianIndices(map(i -> i .+ (-s:s), Tuple(Ib)))
            uᵢ = @view u[i][I]
            ubJ[i] = sum_map(*, w, uᵢ)
        end
        ub[J] = ubJ
    end
end

function regularize!(fu, reg::Reg{<:Any,<:Any,N}, fb) where {N}
    R = CartesianIndices(axes(reg.weights)[1:N])
    backend = get_backend(fu[1])

    for fuᵢ in fu
        @loop backend (I in CartesianIndices(fuᵢ)) fuᵢ[I] = 0
    end

    for ib in eachindex(fb)
        @loop backend (K in R) begin
            for i in 1:N
                I0 = CartesianIndex(Tuple(reg.I[ib, i] .- (support(reg.delta) + 1)))
                I = I0 + K
                fu[i][I] += fb[ib][i] * reg.weights[K, ib, i]
            end
        end
    end

    fu
end
