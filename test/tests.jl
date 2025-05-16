module Tests

using FastIBPM
using FastIBPM: @loop, _set!, ArrayPool, with_arrays, with_arrays_like
using KernelAbstractions
using GPUArrays
using OffsetArrays: OffsetArray, no_offset_view
using StaticArrays
using LinearAlgebra
using Test
using Random

import FFTW
import FastIBPM: FFT_R2R

_backend(array) = get_backend(convert(array, [0]))

function _gridarray(f, array, grid, loc, R::Tuple{Vararg{AbstractRange}}; level=1)
    a = map(CartesianIndices(R)) do I
        x = coord(grid, loc, I, level)
        f(x)
    end
    OffsetArray(convert(array, a), R)
end

function _gridarray(f, array, grid::Grid{N}, loc::Type{<:Edge}, R; level=1) where {N}
    map(level) do lev
        map(FastIBPM.edge_axes(Val(N), loc)) do i
            _gridarray(x -> f(x)[i], array, grid, loc(i), R[i]; level=lev)
        end
    end
end

function _boundary_array(f, array, grid::Grid{N}, loc; kw...) where {N}
    Rb = boundary_axes(grid.n, loc; dims=ntuple(identity, 3))
    map(FastIBPM.edge_axes(Val(N), loc)) do i
        (SArray ∘ map)(CartesianIndices(Rb[i])) do index
            dir, j = Tuple(index)
            _gridarray(x -> f(x)[i], array, grid, loc(i), Rb[i][dir, j]; kw...)
        end
    end
end

struct LinearFunc{N,T,M}
    u0::SVector{N,T}
    du::SMatrix{N,3,T,M}
end
LinearFunc{N,T}(u0, du) where {N,T} = LinearFunc{N,T,3N}(u0, du)

(f::LinearFunc)(x::SVector{3}) = f.u0 + f.du * x
(f::LinearFunc)(x::SVector{2}) = f([x; 0])

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{LinearFunc{N,T}}) where {N,T}
    u0 = rand(rng, SVector{N,T})
    du = rand(rng, SMatrix{N,3,T})
    LinearFunc{N,T}(u0, du)
end

_rand_xy(T) = _rand_xy(Random.default_rng(), T)

function _rand_xy(rng::AbstractRNG, ::Type{LinearFunc{3,T}}) where {T}
    u0 = [@SArray(rand(rng, T, 2)); 0]
    du = [
        @SArray(rand(rng, T, 2, 2)) @SArray(zeros(T, 2, 1))
        @SArray(zeros(T, 1, 3))
    ]
    LinearFunc{3,T}(u0, du)
end

_rand_z(T) = _rand_z(Random.default_rng(), T)

function _rand_z(rng::AbstractRNG, ::Type{LinearFunc{3,T}}) where {T}
    ω0 = [@SArray(zeros(T, 2)); rand(rng, T)]
    dω = [
        @SArray(zeros(T, 2, 3))
        @SArray(rand(rng, T, 1, 2)) 0
    ]
    LinearFunc{3,T}(ω0, dω)
end

_is_xy(f::LinearFunc{3}) = iszero(f.u0[3]) && iszero(f.du[3, :]) && iszero(f.du[:, 3])
_is_z(f::LinearFunc{3}) = iszero(f.u0[1:2]) && iszero(f.du[1:2, :]) && iszero(f.du[3, 3])

function _with_divergence(f::LinearFunc{3,T}, d) where {T}
    i = diagind(f.du)
    du = setindex(f.du, d - sum(@view f.du[i[2:end]]), i[1])
    LinearFunc{3,T}(f.u0, du)
end

_div(f::LinearFunc{3}) = sum(diag(f.du))

function _curl(f::LinearFunc{3})
    A = f.du
    SVector(A[3, 2] - A[2, 3], A[1, 3] - A[3, 1], A[2, 1] - A[1, 2])
end

_kind_str(kind::Tuple) = string("(", join(FFTW.kind2string.(kind), ", "), ")")
_kind_str(kind) = FFTW.kind2string(kind)

function test_utils()
    @test FastIBPM.axisunit(Val(2), 1) == CartesianIndex((1, 0))
    @test FastIBPM.axisunit(Val(3), 1) == CartesianIndex((1, 0, 0))
    @test FastIBPM.axisunit(Val(3), 3) == CartesianIndex((0, 0, 1))
    @test FastIBPM.axisunit(Val(4))(2) == CartesianIndex((0, 1, 0, 0))

    @test_throws "I in R" @macroexpand1 @loop backend (2 in R) x[I] = y[I]
    @test_throws "I in R" @macroexpand1 @loop backend (in(I, R, S)) x[I] = y[I]
    @test_throws ArgumentError @macroexpand1 @loop backend I x[I] = y[I]
    @test_throws MethodError @macroexpand1 @loop backend (I in R) x[I] = y[I] extra

    let T = Int32, pool = ArrayPool(CPU(), 4 * sizeof(T))
        with_arrays(pool, (T, (2, 2)), (T, (4,)), (Int8, (2,))) do a, b, c
            @test eltype(a) == T
            @test eltype(b) == T
            @test eltype(c) == Int8

            vec(a) .= 1:4
            b .= 5:8
            c .= 9:10
            @test a == reshape(1:4, 2, 2)
            @test b == 5:8
            @test c == 9:10
        end

        with_arrays(pool, (Int8, ((2, 2), (3,)))) do (a, b)
            @test eltype(a) == eltype(b) == Int8

            vec(a) .= 1:4
            b .= 5:7
            @test a == reshape(1:4, 2, 2)
            @test b == 5:7
        end
    end
end

function test_loop(array)
    backend = _backend(array)

    let
        cmap(f, s...) = OffsetArray(map(f, CartesianIndices(s)), s...)
        asarray(T, a) = OffsetArray(T(no_offset_view(a)), axes(a)...)
        a1 = cmap(I -> 100 .+ float.(Tuple(I)), 2:5, 1:3, -4:-2)
        b1 = cmap(I -> float.(Tuple(I)), 2:4, 1:3, -4:-4)
        a2 = asarray(array, a1)
        b2 = asarray(array, b1)

        R = CartesianIndices((2:4, 1:2, -4:-4))

        @views a1[R] = b1[R]
        @loop backend (I in R) a2[I] = b2[I]

        # Drop the offset indexing and check equality on the CPU.
        @test no_offset_view(a1) == Array(no_offset_view(a2))
    end

    let
        a = array([1.0, 5.0, 2.5])
        b = array([3, 7, -4])
        c = array(zeros(3))
        @loop backend (I in CartesianIndices((2:3,))) begin
            c[I] = b[I] - 2 * a[I]
        end
        @test Array(c) ≈ [0, -3, -9]
    end

    let
        a = array([1.0, 2.0, 3.0])
        @test_throws TypeError @loop backend (I in +) a[I] = 0
    end
end

function test_problems()
    let grid = Grid(; h=0.05, n=(7, 12, 5), x0=(0, 1, 0.5), levels=3)
        @test grid.n == [8, 12, 8]
    end

    let grid = Grid(; h=0.05, n=(7, 12), x0=(0, 1), levels=3)
        @test grid.n == [8, 12]
    end

    let h = 0.25, n = SVector(8, 4), x0 = SVector(10, 19), grid = Grid(; h, n, x0, levels=5)
        @test gridcorner(grid) == gridcorner(grid, 1) == x0
        @test gridcorner(grid, 2) ≈ x0 - n * h / 2
        @test gridcorner(grid, 3) ≈ x0 - n * h * 3 / 2

        @test gridstep(grid) == gridstep(grid, 1) == h
        @test gridstep(grid, 2) ≈ 2 * h
        @test gridstep(grid, 3) ≈ 4 * h

        @test coord(grid, Edge{Dual}(3), (1, 3)) ≈ x0 + h * SVector(1, 3)
        @test coord(grid, Edge{Primal}(2), (1, 3)) ≈ x0 + h * SVector(1.5, 3)
        @test coord(grid, Edge{Dual}(2), (1, 3)) ≈ x0 + h * SVector(1, 3.5)
        @test coord(grid, Edge{Primal}(2), (1, 3), 2) ≈
            (x0 - n * h / 2) + 2h * SVector(1.5, 3)
        @test coord(grid, Edge{Dual}(2), (1, 3), 2) ≈
            (x0 - n * h / 2) + 2h * SVector(1, 3.5)

        @test cell_axes(grid, Edge{Dual}(3), IncludeBoundary()) == (0:8, 0:4)
        @test cell_axes(grid, Edge{Dual}(3), ExcludeBoundary()) == (1:7, 1:3)

        @test cell_axes(grid, Edge{Primal}(1), IncludeBoundary()) == (0:8, 0:3)
        @test cell_axes(grid, Edge{Primal}(1), ExcludeBoundary()) == (1:7, 0:3)
    end
    let h = 0.25,
        n = SVector(8, 4, 12),
        x0 = SVector(10, 19, 5),
        grid = Grid(; h, n, x0, levels=5)

        @test cell_axes(grid, Edge{Dual}(2), IncludeBoundary()) == (0:8, 0:3, 0:12)
        @test cell_axes(grid, Edge{Dual}(2), ExcludeBoundary()) == (1:7, 0:3, 1:11)

        @test cell_axes(grid, Edge{Primal}(2), IncludeBoundary()) == (0:7, 0:4, 0:11)
        @test cell_axes(grid, Edge{Primal}(2), ExcludeBoundary()) == (0:7, 1:3, 0:11)
    end
end

function test_fft_r2r(array)
    params = [
        (FFTW.RODFT00, (8, 7), 1:2),
        (FFTW.REDFT10, (9, 6), 1:2),
        (FFTW.REDFT01, (7, 8), 1:2),
        ((FFTW.RODFT00, FFTW.REDFT01), (5, 9), [(1, 2)]),
        ((FFTW.RODFT00, FFTW.REDFT10, FFTW.REDFT01), (3, 6, 4), [(1, 2, 3)]),
    ]
    @testset "$(_kind_str(kind)) size=$sz" for (kind, sz, dimss) in params
        test_fft_r2r(array, kind, sz, dimss)
    end
end

function test_fft_r2r(array, kind, sz, dimss)
    for dims in dimss
        x1 = rand(sz...)
        x2 = array(x1)

        p1 = FFTW.plan_r2r!(x1, kind, dims)
        p2 = FFT_R2R.bad_plan_r2r!(x2, Val.(kind), dims)

        mul!(x1, p1, x1)
        mul!(x2, p2, x2)
        @test x1 ≈ convert(Array, x2)
    end
end

function test_delta_func(δ::FastIBPM.AbstractDeltaFunc)
    s = FastIBPM.support(δ)
    let r = s .+ 0.5 .+ [0.0, 1e-3, 0.5, 1.0, 100.0]
        @test all(@. δ(r) ≈ 0)
        @test all(@. δ(-r) ≈ 0)
    end

    let n = 1000
        @test 2s / (n - 1) * sum(δ, range(-s, s, n)) ≈ 1
    end
end

function test_nonlinear(
    array, grid::Grid{N}, u_true::LinearFunc{3}, ω_true::LinearFunc{3}, R
) where {N}
    if N == 2
        @assert _is_xy(u_true)
        @assert _is_z(ω_true)
    end

    nonlin_true(x) = u_true(x) × ω_true(x)

    Ru = map(r -> first(r)-1:last(r)+1, R)
    Rω = map(r -> first(r):last(r)+1, R)

    u = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> Ru, 3))
    ω = _gridarray(ω_true, array, grid, Loc_ω, ntuple(_ -> Rω, 3))

    nonlin_expect = _gridarray(nonlin_true, array, grid, Loc_u, ntuple(_ -> R, 3))
    nonlin_got = FastIBPM.nonlinear!(map(zero, nonlin_expect), u, ω)

    @test all(@. no_offset_view(nonlin_got) ≈ no_offset_view(nonlin_expect))

    (; nonlin_true, Ru, Rω, u, ω, nonlin_expect, nonlin_got)
end

function test_nonlinear(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        u = _rand_xy(LinearFunc{3,Float64}),
        ω = _rand_z(LinearFunc{3,Float64}),
        R = (1:5, 3:8)

        test_nonlinear(array, grid, u, ω, R)
    end
    nothing
end

function test_nonlinear(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        u = rand(LinearFunc{3,Float64}),
        ω = rand(LinearFunc{3,Float64}),
        R = (2:4, 0:3, -1:1)

        test_nonlinear(array, grid, u, ω, R)
    end
    nothing
end

function test_rot(array, grid::Grid{N}, u_true::LinearFunc{3}, R) where {N}
    if N == 2
        @assert _is_xy(u_true)
    end

    ω_true(_) = _curl(u_true)

    Ru = map(r -> first(r)-1:last(r), R)

    u = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> Ru, 3))

    ω_expect = _gridarray(ω_true, array, grid, Loc_ω, ntuple(_ -> R, 3))
    ω_got = FastIBPM.rot!(map(zero, ω_expect), u; h=grid.h)

    @test all(i -> no_offset_view(ω_got[i]) ≈ no_offset_view(ω_expect[i]), eachindex(ω_got))

    (; ω_true, Ru, u, ω_expect, ω_got)
end

function test_rot(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        u = _rand_xy(LinearFunc{3,Float64}),
        R = (2:4, 0:3)

        test_rot(array, grid, u, R)
    end
    nothing
end

function test_rot(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        u = rand(LinearFunc{3,Float64}),
        R = (2:4, 0:3, -1:1)

        test_rot(array, grid, u, R)
    end
    nothing
end

function test_curl(array, grid::Grid{N}, ψ_true::LinearFunc{3}, R) where {N}
    if N == 2
        @assert _is_z(ψ_true)
    end

    u_true(_) = _curl(ψ_true)

    Rψ = map(r -> first(r):last(r)+1, R)

    ψ = _gridarray(ψ_true, array, grid, Loc_ω, ntuple(_ -> Rψ, 3))

    u_expect = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> R, 3))
    u_got = FastIBPM.curl!(map(zero, u_expect), ψ; h=grid.h)

    @test all(@. no_offset_view(u_got) ≈ no_offset_view(u_expect))

    (; u_true, Rψ, ψ, u_expect, u_got)
end

function test_curl(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ψ = _rand_z(LinearFunc{3,Float64}),
        R = (2:4, 0:3)

        test_curl(array, grid, ψ, R)
    end
    nothing
end

function test_curl(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ψ = rand(LinearFunc{3,Float64}),
        R = (2:4, 0:3, -1:1)

        test_curl(array, grid, ψ, R)
    end
    nothing
end

function test_laplacian_inv(array, grid::Grid{N}, ψ_true::LinearFunc{3,T}) where {N,T}
    @assert _div(ψ_true) < eps(T)

    if N == 2
        @assert _is_z(ψ_true)
    end

    backend = _backend(array)

    Rψ = ntuple(i -> cell_axes(grid, Loc_ω(i), ExcludeBoundary()), 3)
    Rψb = ntuple(i -> cell_axes(grid, Loc_ω(i), IncludeBoundary()), 3)
    Ru = ntuple(i -> cell_axes(grid, Loc_u(i), ExcludeBoundary()), 3)

    ψ = _gridarray(ψ_true, array, grid, Loc_ω, Rψb)
    for i in eachindex(ψ),
        (j, _) in FastIBPM.axes_permutations(i),
        Iⱼ in (Rψb[i][j][begin], Rψb[i][j][end])

        R = CartesianIndices(setindex(Rψb[i], Iⱼ:Iⱼ, j))
        @loop backend (I in R) ψ[i][I] = 0
    end

    ψ_expect = map(i -> OffsetArray(ψ[i][Rψ[i]...], Rψ[i]), FastIBPM.tupleindices(ψ))
    ψ_got = map(similar, ψ_expect)
    u = ntuple(N) do i
        dims = Ru[i]
        OffsetArray(
            KernelAbstractions.zeros(_backend(array), Float64, length.(dims)...), dims
        )
    end

    plan = FastIBPM.laplacian_plans(ψ_got, grid.n)

    FastIBPM.curl!(u, ψ; h=grid.h)
    FastIBPM.rot!(ψ_got, u; h=grid.h)
    FastIBPM.EigenbasisTransform(λ -> -1 / (λ / grid.h^2), plan)(ψ_got, ψ_got)

    @test all(i -> no_offset_view(ψ_got[i]) ≈ no_offset_view(ψ_expect[i]), eachindex(ψ_got))

    (; Rψ, Rψb, Ru, ψ, ψ_expect, ψ_got, u, plan)
end

function test_laplacian_inv(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ψ = _rand_z(LinearFunc{3,Float64})

        test_laplacian_inv(array, grid, ψ)
    end
    nothing
end

function test_laplacian_inv(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ψ = _with_divergence(rand(LinearFunc{3,Float64}), 0)

        test_laplacian_inv(array, grid, ψ)
    end
    nothing
end

function test_multidomain_coarsen(array, grid::Grid{N}, ω_true::LinearFunc{3}) where {N}
    backend = _backend(array)

    R = ntuple(i -> cell_axes(grid, Loc_ω(i), ExcludeBoundary()), 3)
    ω¹ = _gridarray(ω_true, array, grid, Loc_ω, R; level=1)
    ω²_expect = _gridarray(ω_true, array, grid, Loc_ω, R; level=2)
    ω²_got = map(copy, ω²_expect)

    for i in eachindex(ω²_got)
        R_inner = CartesianIndices(
            ntuple(N) do j
                n4 = grid.n[j] ÷ 4
                i == j ? (n4:3n4-1) : (n4+1:3n4-1)
            end,
        )
        @loop backend (I in R_inner) ω²_got[i][I] = 0
    end

    FastIBPM.multidomain_coarsen!(ω²_got, ω¹; n=grid.n)

    @test all(
        i -> no_offset_view(ω²_got[i]) ≈ no_offset_view(ω²_expect[i]), eachindex(ω²_got)
    )

    (; R, ω¹, ω²_expect, ω²_got)
end

function test_multidomain_coarsen(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ω = _rand_z(LinearFunc{3,Float64})

        test_multidomain_coarsen(array, grid, ω)
    end
    nothing
end

function test_multidomain_coarsen(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ω = rand(LinearFunc{3,Float64})

        test_multidomain_coarsen(array, grid, ω)
    end
    nothing
end

function test_multidomain_interpolate(array, grid::Grid{N}, ω_true::LinearFunc{3}) where {N}
    R = ntuple(i -> cell_axes(grid, Loc_ω(i), ExcludeBoundary()), 3)

    ω = _gridarray(ω_true, array, grid, Loc_ω, R; level=2)

    ω_b_expect = _boundary_array(ω_true, array, grid, Loc_ω; level=1)
    ω_b_got = map(a -> map(zero, a), ω_b_expect)

    FastIBPM.multidomain_interpolate!(ω_b_got, ω; n=grid.n)

    @test all(
        i -> all(@. no_offset_view(ω_b_got[i]) ≈ no_offset_view(ω_b_expect[i])),
        eachindex(ω_b_got),
    )

    (; R, ω, ω_b_expect, ω_b_got)
end

function test_multidomain_interpolate(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ω = _rand_z(LinearFunc{3,Float64})

        test_multidomain_interpolate(array, grid, ω)
    end
    nothing
end

function test_multidomain_interpolate(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ω = rand(LinearFunc{3,Float64})

        test_multidomain_interpolate(array, grid, ω)
    end
    nothing
end

function test_set_boundary(array, grid::Grid{N}, ω_true::LinearFunc{3}) where {N}
    backend = _backend(array)

    R = cell_axes(grid, Loc_ω, IncludeBoundary())
    Ri = cell_axes(grid, Loc_ω, ExcludeBoundary())

    ω_expect = _gridarray(ω_true, array, grid, Loc_ω, R)

    ω_got = _gridarray(x -> zero(SVector{3}), array, grid, Loc_ω, R)
    for i in eachindex(ω_got)
        a = ω_got[i]
        @loop backend (I in CartesianIndices(Ri[i])) begin
            a[I] = ω_true(coord(grid, Loc_ω(i), I))[i]
        end
    end

    ω_b = _boundary_array(ω_true, array, grid, Loc_ω; level=1)

    FastIBPM.set_boundary!(ω_got, ω_b)

    @test all(i -> no_offset_view(ω_got[i]) ≈ no_offset_view(ω_expect[i]), eachindex(ω_got))

    (; R, Ri, ω_expect, ω_got, ω_b)
end

function test_set_boundary(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ω = _rand_z(LinearFunc{3,Float64})

        test_set_boundary(array, grid, ω)
    end
    nothing
end

function test_set_boundary(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ω = rand(LinearFunc{3,Float64})

        test_set_boundary(array, grid, ω)
    end
    nothing
end

function test_multidomain_poisson(array, grid::Grid{N}, ψ_true::LinearFunc{3,T}) where {N,T}
    @assert _div(ψ_true) < eps(T)

    if N == 2
        @assert _is_z(ψ_true)
    end

    backend = _backend(array)

    Rωi = cell_axes(grid, Loc_ω, IncludeBoundary())
    Rωe = cell_axes(grid, Loc_ω, ExcludeBoundary())
    Rui = cell_axes(grid, Loc_u, IncludeBoundary())
    Rue = cell_axes(grid, Loc_u, ExcludeBoundary())

    ψ_got = _gridarray(_ -> zero(SVector{3}), array, grid, Loc_ω, Rωi; level=1:grid.levels)
    ψ_expect = _gridarray(ψ_true, array, grid, Loc_ω, Rωi; level=1:grid.levels)
    u = _gridarray(_ -> _curl(ψ_true), array, grid, Loc_u, Rui; level=1:grid.levels)
    ω = _gridarray(_ -> zero(SVector{3}), array, grid, Loc_ω, Rωe; level=1:grid.levels)

    let lev = grid.levels,
        h = gridstep(grid, lev),
        ui = map(FastIBPM.tupleindices(u[lev])) do i
            R = CartesianIndices(Base.IdentityUnitRange.(Rue[i]))
            @view u[lev][i][R]
        end

        for i in eachindex(ψ_expect[lev]), b in boundary_axes(grid, Loc_ω(i))
            R = CartesianIndices(b)
            a = ψ_expect[lev][i]
            if !isempty(R)
                @loop backend (I in R) a[I] = 0
            end
        end

        FastIBPM.curl!(ui, ψ_expect[lev]; h)
        FastIBPM.rot!(ω[lev], u[lev]; h)
    end

    for lev in 2:grid.levels, (i, ωᵢ) in pairs(ω[lev])
        R_inner = CartesianIndices(
            ntuple(N) do j
                n4 = grid.n[j] ÷ 4
                i == j ? (n4:3n4-1) : (n4+1:3n4-1)
            end,
        )
        @loop backend (I in R_inner) ωᵢ[I] = 999
    end

    ψ_b = _boundary_array(_ -> zero(SVector{3}), array, grid, Loc_ω)

    plan = FastIBPM.laplacian_plans(ω[1], grid.n)

    FastIBPM.multidomain_poisson!(ω, ψ_got, u, ψ_b, grid, plan)

    @test all(eachindex(ψ_got)) do level
        all(eachindex(ψ_got[level])) do i
            no_offset_view(ψ_got[level][i]) ≈ no_offset_view(ψ_expect[level][i])
        end
    end

    (; ψ_got, ψ_expect, u, ω, ψ_b, plan)
end

function test_multidomain_poisson(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ψ = _rand_z(LinearFunc{3,Float64})

        test_multidomain_poisson(array, grid, ψ)
    end
    nothing
end

function test_multidomain_poisson(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ψ = _with_divergence(rand(LinearFunc{3,Float64}), 0)

        test_multidomain_poisson(array, grid, ψ)
    end
    nothing
end

function test_regularization(
    array, grid::Grid{N}, u_true::LinearFunc{3}, xb::AbstractVector{<:SVector}
) where {N}
    backend = _backend(array)
    T = Float64
    nb = length(xb)

    reg = FastIBPM.Reg(backend, T, FastIBPM.DeltaYang3S(), nb, Val(N))
    FastIBPM.update_weights!(reg, grid, xb, eachindex(xb))

    R = ntuple(i -> cell_axes(grid, Loc_u(i), ExcludeBoundary()), N)

    u = _gridarray(u_true, array, grid, Loc_u, R)

    ub_expect = map(x -> u_true(x)[1:N], Array(xb))
    ub_got = KernelAbstractions.zeros(backend, SVector{N,T}, nb)
    FastIBPM.interpolate_body!(ub_got, reg, u)

    @test Array(ub_got) ≈ ub_expect

    fu = _gridarray(x -> zero(SVector{N}), array, grid, Loc_u, R)
    fb = KernelAbstractions.allocate(backend, SVector{N,T}, nb)
    fill!(fb, 1 .+ zero(SVector{N,T}))
    FastIBPM.regularize!(fu, reg, fb)

    @test all(@. sum(no_offset_view(fu)) ≈ nb)

    (; reg, R, u, ub_expect, ub_got, fu, fb)
end

function test_regularization(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(80, 80), x0=(-2.0, -1.95), levels=3),
        u = _rand_xy(LinearFunc{3,Float64}),
        nb = 20,
        xb = (array ∘ map)(range(0, 2π, nb)) do t
            SVector(cos(t), sin(t))
        end

        test_regularization(array, grid, u, xb)
    end
    nothing
end

function test_regularization(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(80, 80, 80), x0=(-2.0, -1.95, -2.05), levels=3),
        u = rand(LinearFunc{3,Float64}),
        nb = 20,
        xb = (array ∘ map)(range(0, 1, nb)) do t
            a = 2π * t
            SVector(cos(a), sin(a), 2t - 1)
        end

        test_regularization(array, grid, u, xb)
    end
    nothing
end

function test_cnab(array, prob::IBProblem{N,T}) where {N,T}
    backend = _backend(array)
    sol = CNAB(prob; dt=T(0.02), backend)
    grid = sol.prob.grid

    u_work = grid_zeros(backend, grid, Loc_u, ExcludeBoundary())

    ω_work_bounds = grid_zeros(backend, grid, Loc_ω, IncludeBoundary())
    ω_work = FastIBPM.grid_view(ω_work_bounds, grid, Loc_ω, ExcludeBoundary())

    f_work = similar(sol.f_tilde)

    # Make sure some vorticity is at the multi-domain boundary.
    for _ in 1:50
        step!(sol)
    end

    sol0 = map(1:length(sol.β)) do _
        s = deepcopy((; u=sol.u[1], ω=sol.ω[1]))
        step!(sol)
        s
    end

    FastIBPM.interpolate_body!(f_work, sol.reg, sol.u[1])
    unflatten(x) = reinterpret(reshape, T, x)
    @test unflatten(f_work) ≈ unflatten(sol.points.u) atol = sqrt(eps(T))

    ω = FastIBPM.grid_view(deepcopy(sol0[end].ω), grid, Loc_ω, ExcludeBoundary())

    for i_step in eachindex(sol0)
        FastIBPM.nonlinear!(u_work, sol0[i_step].u, sol0[i_step].ω)
        FastIBPM.rot!(ω_work, u_work; h=grid.h)
        for i in eachindex(ω)
            let ω = ω[i], ω_work = ω_work[i], k = sol.dt * sol.β[i_step]
                @loop backend (I in CartesianIndices(ω)) ω[I] += k * ω_work[I]
            end
        end
    end

    for i in eachindex(ω)
        let ω0 = sol0[end].ω[i], ω1 = sol.ω[1][i], ω_work_b = ω_work_bounds[i]
            @loop backend (I in CartesianIndices(ω0)) ω_work_b[I] = ω0[I] + ω1[I]
        end
    end

    FastIBPM.curl!(u_work, ω_work_bounds; h=grid.h)
    FastIBPM.rot!(ω_work, u_work; h=grid.h)

    for i in eachindex(ω)
        let ω = ω[i], ω_work = ω_work[i], k = sol.dt / (2sol.prob.Re)
            @loop backend (I in CartesianIndices(ω)) ω[I] -= k * ω_work[I]
        end
    end

    FastIBPM.regularize!(u_work, sol.reg, sol.f_tilde)
    FastIBPM.rot!(ω_work, u_work; h=grid.h)

    for i in eachindex(ω)
        let ω = ω[i], ω_work = ω_work[i]
            @loop backend (I in CartesianIndices(ω)) ω[I] -= ω_work[I]
        end
    end

    let ω_got = FastIBPM.grid_view(sol.ω[1], grid, Loc_ω, ExcludeBoundary()), ω_expect = ω
        @test all(eachindex(ω_got)) do i
            approx = OffsetArray(
                KernelAbstractions.zeros(backend, Bool, size(ω_got[i])...), axes(ω_got[i])
            )
            let ω_got = ω_got[i], ω_expect = ω_expect[i], atol = sqrt(eps(T))
                @loop backend (I in CartesianIndices(ω_got)) begin
                    approx[I] = isapprox(ω_got[I], ω_expect[I]; atol)
                end
            end
            all(no_offset_view(approx))
        end

        (; sol, sol0, ω_got, ω_expect)
    end
end

function test_cnab(array, ::Val{2})
    let grid = Grid(; h=0.1, n=(40, 40), x0=(-2.0, -1.95), levels=3),
        nb = 20,
        ds = fill(2π / nb, nb),
        xb = (array ∘ map)(range(0, 2π, nb + 1)[1:end-1]) do t
            SVector(cos(t), sin(t))
        end,
        body = StaticBody(xb, ds),
        Re = 50.0,
        u0 = UniformFlow(t -> SVector{2,Float64}(1, 0)),
        prob = IBProblem(grid, body, Re, u0)

        test_cnab(array, prob)
    end
end

function test_cnab(array, ::Val{3})
    let grid = Grid(; h=0.1, n=(40, 40, 40), x0=(-2.0, -1.95, -2.05), levels=3),
        nb = 30,
        ds = fill(1.0, nb),  # dummy value
        xb = (array ∘ map)(range(0, 1, nb)) do t
            a = 2π * t
            SVector(cos(a), sin(a), 2t - 1)
        end,
        body = StaticBody(xb, ds),
        Re = 50.0,
        u0 = UniformFlow(t -> SVector{3,Float64}(1, 0, 0)),
        prob = IBProblem(grid, body, Re, u0)

        test_cnab(array, prob)
    end
end

function test_cnab_io(sol::CNAB)
    grid = sol.prob.grid

    backend = get_backend(sol.f_tilde)
    ω = grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels)
    ψ = grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels)
    u = grid_zeros(backend, grid, Loc_u; levels=1:grid.levels)
    nonlin = map(eachindex(sol.nonlin)) do _
        grid_zeros(backend, grid, Loc_ω, ExcludeBoundary(); levels=1:grid.levels)
    end

    sol_i = sol.i
    sol_t = sol.t
    nonlin_count = sol.nonlin_count
    for level in 1:grid.levels
        for i in eachindex(ω[level])
            _set!(ω[level][i], sol.ω[level][i])
            _set!(ψ[level][i], sol.ψ[level][i])
            for k in eachindex(nonlin)
                _set!(nonlin[k][level][i], sol.nonlin[k][level][i])
            end
        end
        for i in eachindex(u[level])
            _set!(u[level][i], sol.u[level][i])
        end
    end

    io = IOBuffer()
    FastIBPM.save(io, sol)

    sol.i = -1
    sol.t = NaN
    sol.nonlin_count = -1
    for level in 1:grid.levels
        for i in eachindex(sol.ω[level])
            fill!(sol.ω[level][i], NaN)
            fill!(sol.ψ[level][i], NaN)
            for k in eachindex(sol.nonlin)
                fill!(sol.nonlin[k][level][i], NaN)
            end
        end
        for i in eachindex(sol.u[level])
            fill!(sol.u[level][i], NaN)
        end
    end

    seekstart(io)
    FastIBPM.load!(io, sol)

    @test sol.i == sol_i
    @test sol.t == sol_t
    @test sol.nonlin_count == nonlin_count
    @test all(
        no_offset_view(sol.ω[level][i]) == no_offset_view(ω[level][i]) for
        level in 1:grid.levels for i in eachindex(ω[level])
    )
    @test all(
        no_offset_view(sol.ψ[level][i]) == no_offset_view(ψ[level][i]) for
        level in 1:grid.levels for i in eachindex(ψ[level])
    )
    @test all(
        no_offset_view(sol.u[level][i]) == no_offset_view(u[level][i]) for
        level in 1:grid.levels for i in eachindex(u[level])
    )
    @test all(
        no_offset_view(sol.nonlin[k][level][i]) == no_offset_view(nonlin[k][level][i]) for
        k in 1:sol.nonlin_count for level in 1:grid.levels for
        i in eachindex(nonlin[k][level])
    )
end

function test_cnab_io(array, ::Val{2})
    let grid = Grid(; h=0.1, n=(40, 40), x0=(-2.0, -1.95), levels=3),
        nb = 20,
        xb = (array ∘ map)(range(0, 2π, nb)) do t
            SVector(cos(t), sin(t))
        end,
        ds = fill(1.0, nb),  # dummy value
        body = StaticBody(xb, ds),
        Re = 50.0,
        u0 = UniformFlow(t -> SVector{2,Float64}(1, 0)),
        prob = IBProblem(grid, body, Re, u0),
        backend = _backend(array),
        sol = CNAB(prob; dt=0.02, backend)

        for _ in 1:50
            step!(sol)
        end
        sol.nonlin_count = 0

        # 0 nonlinear terms stored
        test_cnab_io(sol)

        for _ in 1:50
            step!(sol)
        end

        # 1 nonlinear term stored
        test_cnab_io(sol)
    end
end

function test_cnab_io(array, ::Val{3})
    let grid = Grid(; h=0.4, n=(10, 10, 10), x0=(-2.0, -1.95, -2.05), levels=3),
        nb = 30,
        xb = (array ∘ map)(range(0, 1, nb)) do t
            a = 2π * t
            SVector(cos(a), sin(a), 2t - 1)
        end,
        ds = fill(1.0, nb),  # dummy value
        body = StaticBody(xb, ds),
        Re = 50.0,
        u0 = UniformFlow(t -> SVector{3,Float64}(1, 0, 0)),
        prob = IBProblem(grid, body, Re, u0),
        backend = _backend(array),
        sol = CNAB(prob; dt=0.02, backend)

        for _ in 1:50
            step!(sol)
        end
        sol.nonlin_count = 0

        # 0 nonlinear terms stored
        test_cnab_io(sol)

        for _ in 1:50
            step!(sol)
        end

        # 1 nonlinear term stored
        test_cnab_io(sol)
    end
end

end
