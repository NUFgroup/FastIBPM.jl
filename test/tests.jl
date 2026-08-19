module Tests

using Immersa
using Immersa: @loop, _set!
using Immersa.array_pools
using Immersa.offset_tuples
using Immersa.utilities
using KernelAbstractions
using GPUArrays
using OffsetArrays: OffsetArray, no_offset_view
using StaticArrays
using LinearAlgebra
using Test
using Random

import FFTW
import Immersa: fft_r2r

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
        map(Immersa.edge_axes(Val(N), loc)) do i
            _gridarray(x -> f(x)[i], array, grid, loc(i), R[i]; level=lev)
        end
    end
end

function _boundary_array(f, array, grid::Grid{N}, loc; kw...) where {N}
    Rb = boundary_axes(grid.n, loc; dims=ntuple(identity, 3))
    map(Immersa.edge_axes(Val(N), loc)) do i
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
    @test axisunit(Val(2), 1) == CartesianIndex((1, 0))
    @test axisunit(Val(3), 1) == CartesianIndex((1, 0, 0))
    @test axisunit(Val(3), 3) == CartesianIndex((0, 0, 1))
    @test axisunit(Val(4))(2) == CartesianIndex((0, 1, 0, 0))

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
        p2 = fft_r2r.bad_plan_r2r!(x2, Val.(kind), dims)

        mul!(x1, p1, x1)
        mul!(x2, p2, x2)
        @test x1 ≈ convert(Array, x2)
    end
end

function test_delta_func(δ::Immersa.AbstractDeltaFunc)
    s = Immersa.support(δ)
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

    Ru = map(r -> (first(r)-1):(last(r)+1), R)
    Rω = map(r -> first(r):(last(r)+1), R)

    u = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> Ru, 3))
    ω = _gridarray(ω_true, array, grid, Loc_ω, ntuple(_ -> Rω, 3))

    nonlin_expect = _gridarray(nonlin_true, array, grid, Loc_u, ntuple(_ -> R, 3))
    nonlin_got = Immersa.nonlinear!(map(zero, nonlin_expect), u, ω)

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

    Ru = map(r -> (first(r)-1):last(r), R)

    u = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> Ru, 3))

    ω_expect = _gridarray(ω_true, array, grid, Loc_ω, ntuple(_ -> R, 3))
    ω_got = Immersa.rot!(map(zero, ω_expect), u; h=grid.h)

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

    Rψ = map(r -> first(r):(last(r)+1), R)

    ψ = _gridarray(ψ_true, array, grid, Loc_ω, ntuple(_ -> Rψ, 3))

    u_expect = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> R, 3))
    u_got = Immersa.curl!(map(zero, u_expect), ψ; h=grid.h)

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

function test_divergence(array, grid::Grid{N}, u_true::LinearFunc{3}, R) where {N}
    if N == 2
        @assert _is_xy(u_true)
    end

    # For a linear velocity field, ∇·u is the constant sum of the diagonal.
    d_true(_) = _div(u_true)

    # Divergence at cell I reads u[j] at I and I+δ(j), so pad velocity one cell up.
    Ru = map(r -> first(r):(last(r)+1), R)

    u = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> Ru, 3))

    # Pressure/divergence lives at cell centers (Loc_p = Node{Dual}); scalar field.
    d_expect = _gridarray(d_true, array, grid, Loc_p(), R)
    d_got = Immersa.divergence!(zero(d_expect), u; h=grid.h)

    @test no_offset_view(d_got) ≈ no_offset_view(d_expect)

    (; d_true, Ru, u, d_expect, d_got)
end

function test_divergence(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        u = _rand_xy(LinearFunc{3,Float64}),
        R = (2:4, 0:3)

        test_divergence(array, grid, u, R)
    end
    nothing
end

function test_divergence(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        u = rand(LinearFunc{3,Float64}),
        R = (2:4, 0:3, -1:1)

        test_divergence(array, grid, u, R)
    end
    nothing
end

function test_gradient(array, grid::Grid{N}, c::SVector{N}, c0, R) where {N}
    # Scalar linear pressure p(x) = c·x + c0  ⇒  ∇p = c (constant per component).
    p_true(x) = c ⋅ x + c0
    g_true(_) = c

    # Gradient at edge I reads p at I and I-δ(i), so pad pressure one cell down.
    Rp = map(r -> (first(r)-1):last(r), R)

    p = _gridarray(p_true, array, grid, Loc_p(), Rp)

    g_expect = _gridarray(g_true, array, grid, Loc_u, ntuple(_ -> R, 3))
    g_got = Immersa.gradient!(map(zero, g_expect), p; h=grid.h)

    @test all(@. no_offset_view(g_got) ≈ no_offset_view(g_expect))

    (; p_true, g_true, Rp, p, g_expect, g_got)
end

function test_gradient(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        c = rand(SVector{2,Float64}),
        c0 = rand(),
        R = (2:4, 0:3)

        test_gradient(array, grid, c, c0, R)
    end
    nothing
end

function test_gradient(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        c = rand(SVector{3,Float64}),
        c0 = rand(),
        R = (2:4, 0:3, -1:1)

        test_gradient(array, grid, c, c0, R)
    end
    nothing
end

function test_velocity_boundary(array, grid::Grid{N}, f) where {N}
    backend = _backend(array)
    ub = Immersa.boundary_zeros(backend, grid, Loc_u)
    Immersa.set_velocity_boundary!(ub, grid, f)

    # Every normal boundary face must hold f(x)[i]; degenerate faces stay empty.
    ok = true
    for i in 1:N
        loc = Immersa.Edge{Immersa.Primal}(i)
        faces = ub[i]
        for idx in CartesianIndices(faces)
            face = faces[idx]
            isempty(face) && continue
            _, j = Tuple(idx)
            ok &= (j == i)   # only the normal (j==i) faces are populated
            for I in CartesianIndices(face)
                ok &= isapprox(face[I], f(coord(grid, loc, I))[i])
            end
        end
    end
    @test ok

    (; ub,)
end

function test_velocity_boundary(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        f = _rand_xy(LinearFunc{3,Float64})

        test_velocity_boundary(array, grid, f)
    end
    nothing
end

function test_velocity_boundary(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        f = rand(LinearFunc{3,Float64})

        test_velocity_boundary(array, grid, f)
    end
    nothing
end

function test_laplacian(array, grid::Grid{N}, C::SMatrix{N,N}, R) where {N}
    # Per-component quadratic velocity  u_i(x) = Σ_j C[i,j] x_j²  ⇒
    # (∇²u)_i = 2 Σ_j C[i,j]  (constant). The second-difference stencil is exact
    # for quadratics, so the discrete Laplacian matches to roundoff.
    u_true(x) = SVector(ntuple(i -> sum(ntuple(j -> C[i, j] * x[j]^2, N)), N))
    lap_true(_) = SVector(ntuple(i -> 2 * sum(ntuple(j -> C[i, j], N)), N))

    # Laplacian at edge I reads u[i] at I and I±δ(j): pad one cell on both sides.
    Ru = map(r -> (first(r)-1):(last(r)+1), R)

    u = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> Ru, 3))

    lap_expect = _gridarray(lap_true, array, grid, Loc_u, ntuple(_ -> R, 3))
    lap_got = Immersa.laplacian!(map(zero, lap_expect), u; h=grid.h)

    @test all(@. no_offset_view(lap_got) ≈ no_offset_view(lap_expect))

    (; u_true, lap_true, Ru, u, lap_expect, lap_got)
end

function test_laplacian(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        C = rand(SMatrix{2,2,Float64}),
        R = (2:4, 0:3)

        test_laplacian(array, grid, C, R)
    end
    nothing
end

function test_laplacian(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        C = rand(SMatrix{3,3,Float64}),
        R = (2:4, 0:3, -1:1)

        test_laplacian(array, grid, C, R)
    end
    nothing
end

function test_laplacian_bc(array, grid::Grid{N}, f) where {N}
    backend = _backend(array)
    T = Float64
    loc(i) = Loc_u(i)
    interior(i) = cell_axes(grid, loc(i), ExcludeBoundary())
    pad(r) = (first(r)-1):(last(r)+1)

    # `ufull`: prescribed field f on interior + one ghost ring; `uzero`: same but
    # with the ring zeroed. Then, exactly, laplacian(uzero) + bc1 == laplacian(ufull).
    ufull = ntuple(N) do i
        _gridarray(x -> f(x)[i], array, grid, loc(i), pad.(interior(i)))
    end
    uzero = ntuple(N) do i
        Rp = pad.(interior(i))
        z = OffsetArray(convert(array, zeros(T, length.(Rp))), Rp...)
        Re = CartesianIndices(Base.IdentityUnitRange.(interior(i)))
        _set!((@view z[Re]), (@view ufull[i][Re]))
        z
    end

    lapof(field) = ntuple(N) do i
        Re = interior(i)
        out = OffsetArray(convert(array, zeros(T, length.(Re))), Re...)
        a = field[i]
        @loop backend (I in CartesianIndices(out)) out[I] = Immersa.laplacian(a, I; h=grid.h)
        out
    end
    lap_full = lapof(ufull)
    lap_zero = lapof(uzero)

    bc1 = ntuple(N) do i
        Re = interior(i)
        OffsetArray(convert(array, zeros(T, length.(Re))), Re...)
    end
    Immersa.add_laplacian_bc!(bc1, Loc_u, 1 / grid.h^2, f, grid)

    @test all(1:N) do i
        no_offset_view(lap_zero[i]) .+ no_offset_view(bc1[i]) ≈ no_offset_view(lap_full[i])
    end

    (; ufull, uzero, lap_full, lap_zero, bc1)
end

# quadratic per-component field ⇒ ∇²f is a nonzero constant, so the split identity
# is exercised with both sides nonzero.
_quad_field(C::SMatrix{N,N}) where {N} =
    x -> SVector(ntuple(i -> sum(ntuple(j -> C[i, j] * x[j]^2, N)), N))

function test_laplacian_bc(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3)
        test_laplacian_bc(array, grid, _quad_field(rand(SMatrix{2,2,Float64})))
    end
    nothing
end

function test_laplacian_bc(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3)
        test_laplacian_bc(array, grid, _quad_field(rand(SMatrix{3,3,Float64})))
    end
    nothing
end

function test_divergence_bc(array, grid::Grid{N}, f) where {N}
    backend = _backend(array)
    T = Float64
    lu(j) = Loc_u(j)
    Rfull(j) = cell_axes(grid, lu(j), IncludeBoundary())
    Rint(j) = cell_axes(grid, lu(j), ExcludeBoundary())

    # `u_full`: f on all faces; `u_int`: f on interior faces, 0 on the normal ∂D
    # faces. Then, exactly, divergence(u_int) + bc2 == divergence(u_full).
    u_full = ntuple(j -> _gridarray(x -> f(x)[j], array, grid, lu(j), Rfull(j)), N)
    u_int = ntuple(N) do j
        z = OffsetArray(convert(array, zeros(T, length.(Rfull(j)))), Rfull(j)...)
        Re = CartesianIndices(Base.IdentityUnitRange.(Rint(j)))
        _set!((@view z[Re]), (@view u_full[j][Re]))
        z
    end

    d_full = Immersa.divergence!(grid_zeros(backend, grid, Loc_p()), u_full; h=grid.h)
    d_int = Immersa.divergence!(grid_zeros(backend, grid, Loc_p()), u_int; h=grid.h)

    ub = Immersa.boundary_zeros(backend, grid, Loc_u)
    Immersa.set_velocity_boundary!(ub, grid, f)
    bc2 = grid_zeros(backend, grid, Loc_p())
    Immersa.divergence_bc!(bc2, 1 / grid.h, ub)

    @test no_offset_view(d_int) .+ no_offset_view(bc2) ≈ no_offset_view(d_full)

    (; u_full, u_int, d_full, d_int, bc2)
end

function test_divergence_bc(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3)
        test_divergence_bc(array, grid, _quad_field(rand(SMatrix{2,2,Float64})))
    end
    nothing
end

function test_divergence_bc(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3)
        test_divergence_bc(array, grid, _quad_field(rand(SMatrix{3,3,Float64})))
    end
    nothing
end

function test_Ainv(array, grid::Grid{N}; a=0.037, dt=0.02, n_taylor=3) where {N}
    backend = _backend(array)
    h = grid.h
    interior(A) = CartesianIndices(map(r -> (first(r)+1):(last(r)-1), UnitRange.(axes(A))))

    # Neumann telescoping identity (exact, any input):
    #   (I - aL)(Bᴺ x / Δt) == x - (aL)ⁿ x .
    x = Immersa.Ainv_zeros(backend, grid)
    for i in eachindex(x)
        A = x[i]
        @loop backend (I in interior(A)) A[I] = sin(0.3 * sum(Tuple(I))) + 0.5 * i
    end

    y = Immersa.Ainv_zeros(backend, grid)
    t1 = Immersa.Ainv_zeros(backend, grid)
    t2 = Immersa.Ainv_zeros(backend, grid)
    Immersa.Ainv!(y, x, t1, t2; a, dt, n_taylor, h)

    # lhs = (y - aL y) / dt
    dy = Immersa.Ainv_zeros(backend, grid)
    Immersa._apply_aL!(dy, y, a, h)
    lhs = Immersa.Ainv_zeros(backend, grid)
    for i in eachindex(lhs)
        L, Y, D = lhs[i], y[i], dy[i]
        @loop backend (I in interior(L)) L[I] = (Y[I] - D[I]) / dt
    end

    # rhs = x - (aL)ⁿ x
    p = Immersa.Ainv_zeros(backend, grid)
    q = Immersa.Ainv_zeros(backend, grid)
    for i in eachindex(p)
        _set!(p[i], x[i])
    end
    for _ in 1:n_taylor
        Immersa._apply_aL!(q, p, a, h)
        for i in eachindex(p)
            _set!(p[i], q[i])
        end
    end
    rhs = Immersa.Ainv_zeros(backend, grid)
    for i in eachindex(rhs)
        R, X, P = rhs[i], x[i], p[i]
        @loop backend (I in interior(R)) R[I] = X[I] - P[I]
    end

    # halos are zero in both, so a norm-based ≈ over the full arrays is relative.
    @test all(i -> no_offset_view(lhs[i]) ≈ no_offset_view(rhs[i]), eachindex(lhs))

    (; x, y, lhs, rhs)
end

function test_Ainv(array, ::Val{2})
    test_Ainv(array, Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3))
    nothing
end

function test_Ainv(array, ::Val{3})
    test_Ainv(array, Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3))
    nothing
end

function test_Q(array, grid::Grid{N}, xb) where {N}
    backend = _backend(array)
    T = Float64
    h = grid.h
    nb = length(xb)
    reg = Immersa.Reg(backend, T, Immersa.DeltaYang3S(), nb, Val(N))
    Immersa.update_weights!(reg, grid, xb, eachindex(xb))

    interior(A) = CartesianIndices(Immersa._interior_range(A))

    # Adjoint identity  <Q λ, q> == <λ, Qᵀ q>,  with λ = (φ, f_tilde).
    # This checks Gᵀ = -D and that E/Eᵀ are exact transposes at once — the
    # symmetry that makes the modified Poisson operator QᵀBᴺQ solvable by CG.
    q = Immersa.Ainv_zeros(backend, grid)
    for i in eachindex(q)
        A = q[i]
        @loop backend (I in interior(A)) A[I] = sin(0.7 * sum(Tuple(I))) + 0.3 * i
    end
    φ = grid_zeros(backend, grid, Loc_p())
    @loop backend (I in CartesianIndices(φ)) φ[I] = cos(0.4 * sum(Tuple(I)))
    f_tilde = (array ∘ map)(1:nb) do k
        SVector(ntuple(i -> sin(0.9 * (k + i)), N))
    end

    Qλ = Immersa.Ainv_zeros(backend, grid)
    Immersa.Q_mul!(Qλ, φ, f_tilde, reg; h)

    φ_out = grid_zeros(backend, grid, Loc_p())
    f_out = KernelAbstractions.zeros(backend, SVector{N,T}, nb)
    Immersa.QT_mul!(φ_out, f_out, q, reg; h)

    lhs = sum(i -> sum(no_offset_view(Qλ[i]) .* no_offset_view(q[i])), eachindex(q))
    rhs =
        sum(no_offset_view(φ) .* no_offset_view(φ_out)) +
        sum(dot.(Array(f_tilde), Array(f_out)))

    @test lhs ≈ rhs

    (; q, φ, f_tilde, Qλ, φ_out, f_out)
end

function test_Q(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(40, 40), x0=(-1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 2π, nb + 1)[1:(end-1)]) do t
            SVector(0.5cos(t), 0.5sin(t))
        end

        test_Q(array, grid, xb)
    end
    nothing
end

function test_Q(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(40, 40, 40), x0=(-1.0, -1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 1, nb)) do t
            a = 2π * t
            SVector(0.5cos(a), 0.5sin(a), 0.5 * (2t - 1))
        end

        test_Q(array, grid, xb)
    end
    nothing
end

function test_B(array, grid::Grid{N}, xb; Re=100.0, dt=0.01, n_taylor=3) where {N}
    backend = _backend(array)
    T = Float64
    h = grid.h
    # a = Δt/(2Re). Keeping a/h² ≪ 1 is what makes the Bᴺ Taylor series converge
    # (the paper's νΔt/Δx² ≲ 1 condition); otherwise B is wildly ill-conditioned.
    a = dt / (2Re)
    nb = length(xb)
    reg = Immersa.Reg(backend, T, Immersa.DeltaYang3S(), nb, Val(N))
    Immersa.update_weights!(reg, grid, xb, eachindex(xb))
    work = Immersa.B_work(backend, grid)
    form = Immersa.IBPM()

    zλ() = (
        grid_zeros(backend, grid, Loc_p()),
        KernelAbstractions.zeros(backend, SVector{N,T}, nb),
    )
    function applyB(p, f)
        po, fo = zλ()
        Immersa.B_mul!(po, fo, p, f, reg, work, form; h, a, dt, n_taylor)
        (po, fo)
    end
    ip(x, y) =
        sum(no_offset_view(x[1]) .* no_offset_view(y[1])) +
        sum(dot.(Array(x[2]), Array(y[2])))
    function mkλ(c)
        p, f = zλ()
        @loop backend (I in CartesianIndices(p)) p[I] = sin(c * sum(Tuple(I)))
        f .= (array ∘ map)(1:nb) do k
            SVector(ntuple(i -> cos(c * (k + i)), N))
        end
        (p, f)
    end

    λ = mkλ(0.4)
    μ = mkλ(0.9)

    # 1. Symmetry ⟨Bλ, μ⟩ == ⟨λ, Bμ⟩ — the property that makes CG applicable.
    @test ip(applyB(λ...), μ) ≈ ip(λ, applyB(μ...))

    # 2. The constant-pressure mode is exactly in the null space (G kills a constant),
    #    which is why one pressure DOF must be pinned.
    let (pc, fc) = zλ()
        @loop backend (I in CartesianIndices(pc)) pc[I] = 1
        Bc = applyB(pc, fc)
        @test ip(Bc, Bc) ≈ 0 atol = 1e-18
    end

    # 3. CG (pressure pinned) inverts B: recover a known λ from rhs = B λ.
    Binv = Immersa.CNAB_Binv_Iterative{T}(; abstol=1e-10, reltol=0.0, pin=1)
    p_true, f_true = mkλ(0.6)
    no_offset_view(p_true)[Binv.pin] = 0
    rp, rf = applyB(p_true, f_true)

    p, f = zλ()
    Binv(p, f, rp, rf, reg, work, form; h, a, dt, n_taylor)

    @test no_offset_view(p) ≈ no_offset_view(p_true) rtol = 1e-6
    @test Array(f) ≈ Array(f_true) rtol = 1e-6

    (; p, f, p_true, f_true)
end

function test_B(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(40, 40), x0=(-1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 2π, nb + 1)[1:(end-1)]) do t
            SVector(0.5cos(t), 0.5sin(t))
        end

        test_B(array, grid, xb)
    end
    nothing
end

function test_B(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(40, 40, 40), x0=(-1.0, -1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 1, nb)) do t
            s = 2π * t
            SVector(0.5cos(s), 0.5sin(s), 0.5 * (2t - 1))
        end

        test_B(array, grid, xb)
    end
    nothing
end

function test_P(array, grid::Grid{N}, xb) where {N}
    backend = _backend(array)
    T = Float64
    nb = length(xb)
    reg = Immersa.Reg(backend, T, Immersa.DeltaYang3S(), nb, Val(N))
    Immersa.update_weights!(reg, grid, xb, eachindex(xb))
    proj = Immersa.ManifoldProjection(backend, grid, reg, nb)

    interior(A) = CartesianIndices(Immersa._interior_range(A))

    # Haloed velocity field with a nontrivial interior and a zero halo — the
    # layout the IMAP operators pass to `P_mul!`.
    #
    # NOTE: the locals below are deliberately *not* named `v`/`w`. A Julia inner
    # function assigns to an enclosing local of the same name rather than
    # shadowing it, so a local `v` here would silently overwrite the caller's `v`
    # on every call and alias the two test fields together.
    function field(c)
        out = Immersa.Ainv_zeros(backend, grid)
        for i in eachindex(out)
            A = out[i]
            @loop backend (I in interior(A)) A[I] = sin(c * sum(Tuple(I))) + c * i
        end
        out
    end
    function copyfield(x)
        out = Immersa.Ainv_zeros(backend, grid)
        for i in eachindex(out)
            _set!(out[i], x[i])
        end
        out
    end
    ip(x, y) = sum(i -> sum(no_offset_view(x[i]) .* no_offset_view(y[i])), eachindex(x))
    nrm(x) = sqrt(ip(x, x))
    body_zeros() = KernelAbstractions.zeros(backend, SVector{N,T}, nb)

    v = field(0.7)
    Pv = copyfield(v)
    Immersa.P_mul!(Pv, proj)

    # 0. P actually does something (guards against a silently-zero correction,
    #    which would make every other property below hold trivially).
    @test !isapprox(no_offset_view(Pv[1]), no_offset_view(v[1]))

    # 1. Idempotence P² = P — P projects *onto* the manifold, so re-projecting
    #    an already-projected field is a no-op.
    PPv = copyfield(Pv)
    Immersa.P_mul!(PPv, proj)
    @test all(i -> no_offset_view(PPv[i]) ≈ no_offset_view(Pv[i]), eachindex(Pv))

    # 2. The image satisfies the constraint: Rᵀ(Pv) = E(Pv) = 0. This is the
    #    no-slip condition IMAP enforces in place of a boundary force.
    let Ev = body_zeros(), EPv = body_zeros()
        Immersa.interpolate_body!(Ev, reg, v)
        Immersa.interpolate_body!(EPv, reg, Pv)
        @test maximum(norm, Array(EPv)) < 1e-10 * maximum(norm, Array(Ev))
    end

    # 3. P annihilates range(R): P R f = R f - R(RᵀR)⁻¹(RᵀR) f = 0 for any body
    #    vector f. This is the check that actually exercises the Gram matrix and
    #    its factorization — a wrong RᵀR fails here but can still pass (1) and (2).
    let f = (array ∘ map)(k -> SVector(ntuple(i -> cos(0.9 * (k + i)), N)), 1:nb),
        Rf = Immersa.Ainv_zeros(backend, grid)

        Immersa.regularize!(Rf, reg, f)
        scale = nrm(Rf)
        Immersa.P_mul!(Rf, proj)
        @test nrm(Rf) < 1e-10 * scale
    end

    # 4. Symmetry ⟨Pv, w⟩ == ⟨v, Pw⟩ — P is an *orthogonal* projector (it relies
    #    on E and Eᵀ being exact ℓ² adjoints, as `test_Q` checks). This is what
    #    makes the symmetrized IMAP operator Gᵀ(Σ(a·PLP)ᵏ)G symmetric.
    let w = field(0.3), Pw = copyfield(w)
        @test !isapprox(ip(w, w), ip(v, v))   # guard: the two fields are distinct
        Immersa.P_mul!(Pw, proj)
        @test ip(Pv, w) ≈ ip(v, Pw)
    end

    # 5. The halo stays zero, as the homogeneous operators require.
    @test all(eachindex(Pv)) do i
        A, R = Pv[i], interior(Pv[i])
        sum(abs, no_offset_view(A)) ≈ sum(abs, @view(A[R]))
    end

    (; v, Pv, proj)
end

function test_P(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(40, 40), x0=(-1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 2π, nb + 1)[1:(end-1)]) do t
            SVector(0.5cos(t), 0.5sin(t))
        end

        test_P(array, grid, xb)
    end
    nothing
end

function test_P(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(40, 40, 40), x0=(-1.0, -1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 1, nb)) do t
            s = 2π * t
            SVector(0.5cos(s), 0.5sin(s), 0.5 * (2t - 1))
        end

        test_P(array, grid, xb)
    end
    nothing
end

function test_cnab_imap(array, grid::Grid{2}, xb, ds; Re=40.0, dt=0.002, nsteps=12)
    backend = _backend(array)
    T = Float64
    body = StaticBody(xb, ds)
    u0 = UniformFlow(t -> SVector{2,T}(1, 0))

    mk(; kw...) = CNAB(
        IBProblem(grid, body, T(Re), u0, IMAP()); dt, backend,
        delta=Immersa.DeltaYang3S(), kw...,
    )
    resid(sol) = noslip_residual(sol)

    # Divergence of the *physical* field: `q` carries zero boundary values, so its
    # divergence is nonzero next to ∂D by exactly the boundary flux — the check
    # has to use `u_full`, which holds the prescribed ∂D values.
    function divnorm(sol)
        d = grid_zeros(backend, sol.prob.grid, Loc_p())
        Immersa.divergence!(d, sol.state.u_full; h=sol.prob.grid.h)
        maximum(abs, no_offset_view(d))
    end

    sol = mk()

    # 1. The initial condition is projected onto the manifold at construction:
    #    IMAP preserves the constraint rather than imposing it, so a free-stream
    #    start would violate no-slip by U∞ for the whole run.
    @test resid(sol) < 1e-12

    # 2. No-slip is *conserved*, every step and not just at the end. With the
    #    pressure gradient projected, Rᵀu^{n+1} = Rᵀuⁿ holds exactly, so the
    #    residual never leaves roundoff — it is not re-established by a force
    #    solve as in IBPM. Leaving `G φ` unprojected instead makes this O(1)
    #    after a single step, because p ~ O(1/Δt) at an impulsive start.
    worst = 0.0
    for _ in 1:nsteps
        step!(sol)
        worst = max(worst, resid(sol))
    end
    @test worst < 1e-10

    # 3. Incompressibility, which is what the pressure solve is for.
    @test divnorm(sol) < 1e-4

    # 4. The recovered boundary force is finite, nonzero, and symmetric: a circle
    #    in a uniform x-flow on a y-symmetric grid must produce drag but no lift.
    f = surface_force_sum(sol)
    @test all(isfinite, f)
    @test f[1] > 0
    @test abs(f[2]) < 1e-3 * abs(f[1])

    # 5. The flow is actually developing — a solver that quietly froze would pass
    #    every check above (a frozen projected free stream is on the manifold,
    #    divergence-free, and symmetric).
    @test maximum(abs, no_offset_view(sol.state.φ)) > 0

    (; sol, f)
end

function test_cnab_imap(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(120, 80), x0=(-2.0, -2.0), levels=1),
        nb = 62,                                   # even ⇒ body symmetric about y = 0
        xb = (array ∘ map)(range(0, 2π, nb + 1)[1:(end-1)]) do t
            SVector(0.5cos(t), 0.5sin(t))
        end,
        ds = fill(2π * 0.5 / nb, nb)

        test_cnab_imap(array, grid, xb, ds)
    end
    nothing
end

# 3D is covered by the operator tests (`test_P`, `test_B_imap`, `test_imap_setup`);
# a 3D time-integration case is too slow for the unit suite.
test_cnab_imap(array, ::Val{3}) = nothing

"""
Cross-formulation check: IMAP and IBPM-PV must agree on the *force*, not just the
flow, when run on an identical grid.

This exists because of a bug it would have caught. `recover_force!` reconstructs
the boundary force from the multipliers the projections discard, and an early
version omitted the `(I - P) G φ` term — the pressure force on the body. Every
other IMAP test still passed: the velocity field was untouched (wake length
matched IBPM to four digits at every time), drag was positive, lift was zero by
symmetry. Only the magnitude was wrong, by a factor of 3.6 on a cylinder at
Re = 40. Comparing against the validated formulation is what makes that visible.
"""
function test_imap_vs_ibpm(array, grid::Grid{2}, xb, ds; Re=40.0, dt=0.02, nsteps=50)
    backend = _backend(array)
    T = Float64
    body = StaticBody(xb, ds)
    u0 = UniformFlow(t -> SVector{2,T}(1, 0))

    Cd = map((IMAP(), IBPM())) do form
        sol = CNAB(
            IBProblem(grid, body, T(Re), u0, form); dt, backend,
            delta=Immersa.DeltaYang3S(),
        )
        for _ in 1:nsteps
            step!(sol)
        end
        f = surface_force_sum(sol)
        @test all(isfinite, f)
        @test abs(f[2]) < 1e-3 * abs(f[1])          # symmetric body ⇒ no lift
        2 * f[1]
    end

    # Both formulations solve the same problem, so at matched times the drag must
    # agree to discretization differences (the two use different approximations of
    # A⁻¹ on the same grid, and IMAP starts from a projected initial condition).
    # Measured at ~0.2%; the tolerance is loose enough to be robust and far tighter
    # than any missing force contribution could survive.
    @test abs(Cd[1] - Cd[2]) < 0.02 * abs(Cd[2])

    (; Cd_imap=Cd[1], Cd_ibpm=Cd[2])
end

function test_imap_vs_ibpm(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(92, 60), x0=(-1.5, -1.5), levels=1),
        nb = 40,
        xb = (array ∘ map)(range(0, 2π, nb + 1)[1:(end-1)]) do t
            SVector(0.5cos(t), 0.5sin(t))
        end,
        ds = fill(2π * 0.5 / nb, nb)

        test_imap_vs_ibpm(array, grid, xb, ds)
    end
    nothing
end

# 2D only: this runs two full time integrations, and 3D would dominate the suite.
test_imap_vs_ibpm(array, ::Val{3}) = nothing

function test_B_imap(array, grid::Grid{N}, xb; Re=100.0, dt=0.01, n_taylor=3) where {N}
    backend = _backend(array)
    T = Float64
    h = grid.h
    a = dt / (2Re)
    nb = length(xb)
    reg = Immersa.Reg(backend, T, Immersa.DeltaYang3S(), nb, Val(N))
    Immersa.update_weights!(reg, grid, xb, eachindex(xb))
    proj = Immersa.ManifoldProjection(backend, grid, reg, nb)
    work = Immersa.B_work(backend, grid)
    form = Immersa.IMAP()

    function applyB(p; symmetric)
        po = grid_zeros(backend, grid, Loc_p())
        Immersa.B_mul!(po, p, proj, work, form; h, a, dt, n_taylor, symmetric)
        po
    end
    asymmetry(; kw...) =
        let l = ip(applyB(φ; kw...), ψ), r = ip(φ, applyB(ψ; kw...))
            abs(l - r) / max(abs(l), abs(r))
        end
    ip(x, y) = sum(no_offset_view(x) .* no_offset_view(y))
    function mkφ(c)
        p = grid_zeros(backend, grid, Loc_p())
        @loop backend (I in CartesianIndices(p)) p[I] = sin(c * sum(Tuple(I)))
        p
    end

    φ = mkφ(0.4)
    ψ = mkφ(0.9)

    # 1. Symmetry ⟨Bφ, ψ⟩ == ⟨φ, Bψ⟩ — the property that makes CG applicable,
    #    exactly as for the IBPM `B`.
    @test asymmetry(symmetric=true) < 1e-12

    # 1b. The literal series Σ(a P L)ᵏ gives the *same* symmetric operator here,
    #     and not by luck: `B` feeds it `P G φ`, which lies in range(P), where the
    #     two series are identical ((a P L)ᵏx = (a P L P)ᵏx for P x = x — checked
    #     directly in 4 below). Projecting the gradient is what buys this; the
    #     bare series is asymmetric on a general input, as check 4b shows.
    @test asymmetry(symmetric=false) < 1e-12

    # 2. The constant-pressure mode is exactly in the null space (G kills a
    #    constant), which is why one pressure DOF must be pinned.
    let pc = grid_zeros(backend, grid, Loc_p())
        @loop backend (I in CartesianIndices(pc)) pc[I] = 1
        Bc = applyB(pc; symmetric=true)
        @test ip(Bc, Bc) ≈ 0 atol = 1e-18
    end

    # 3. Positive definiteness off the null space: ⟨Bφ, φ⟩ > 0. With Bᴺ symmetric
    #    positive definite, B = GᵀBᴺG is SPD for any non-constant φ.
    @test ip(applyB(φ; symmetric=true), φ) > 0

    # 4. The two series agree exactly on the constraint manifold: for a projected
    #    input, (a P L P)ᵏ x == (a P L)ᵏ x. This is what makes the symmetrization
    #    a change of operator only *off* the manifold.
    let x = Immersa.Ainv_zeros(backend, grid),
        y1 = Immersa.Ainv_zeros(backend, grid),
        y2 = Immersa.Ainv_zeros(backend, grid),
        tm = Immersa.Ainv_zeros(backend, grid),
        tp = Immersa.Ainv_zeros(backend, grid)

        for i in eachindex(x)
            A = x[i]
            R = CartesianIndices(Immersa._interior_range(A))
            @loop backend (I in R) A[I] = sin(0.6 * sum(Tuple(I))) + 0.2 * i
        end
        # 4b. Off the manifold they are genuinely different operators. This is
        #     the case a moving body will land in (qⁿ on the *previous* step's
        #     manifold), where which series is wanted becomes a formulation
        #     question — it does not affect the symmetry of `B`, which the `P G`
        #     inside `B_mul!` supplies either way.
        Immersa.Ainv_IMAP!(y1, x, tm, tp, proj; a, dt, n_taylor, h, symmetric=true)
        Immersa.Ainv_IMAP!(y2, x, tm, tp, proj; a, dt, n_taylor, h, symmetric=false)
        @test !all(i -> no_offset_view(y1[i]) ≈ no_offset_view(y2[i]), eachindex(y1))

        # 4. On the manifold they agree exactly.
        Immersa.P_mul!(x, proj)
        Immersa.Ainv_IMAP!(y1, x, tm, tp, proj; a, dt, n_taylor, h, symmetric=true)
        Immersa.Ainv_IMAP!(y2, x, tm, tp, proj; a, dt, n_taylor, h, symmetric=false)
        @test all(i -> no_offset_view(y1[i]) ≈ no_offset_view(y2[i]), eachindex(y1))
    end

    # 5. CG (pressure pinned) inverts B: recover a known φ from rhs = B φ.
    Binv = Immersa.CNAB_Binv_Iterative{T}(; abstol=1e-10, reltol=0.0, pin=1)
    φ_true = mkφ(0.6)
    no_offset_view(φ_true)[Binv.pin] = 0
    rhs = applyB(φ_true; symmetric=true)

    φ_got = grid_zeros(backend, grid, Loc_p())
    Binv(φ_got, rhs, proj, work, form; h, a, dt, n_taylor, symmetric=true)
    @test no_offset_view(φ_got) ≈ no_offset_view(φ_true) rtol = 1e-6

    (; φ_got, φ_true, proj)
end

function test_B_imap(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(40, 40), x0=(-1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 2π, nb + 1)[1:(end-1)]) do t
            SVector(0.5cos(t), 0.5sin(t))
        end

        test_B_imap(array, grid, xb)
    end
    nothing
end

function test_B_imap(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(40, 40, 40), x0=(-1.0, -1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 1, nb)) do t
            s = 2π * t
            SVector(0.5cos(s), 0.5sin(s), 0.5 * (2t - 1))
        end

        test_B_imap(array, grid, xb)
    end
    nothing
end

function test_imap_setup(array, grid::Grid{N}, xb, ds) where {N}
    backend = _backend(array)
    T = Float64
    nb = length(xb)
    body = StaticBody(xb, ds)
    u0 = UniformFlow(t -> SVector{N,T}(1, ntuple(_ -> 0, N - 1)...))
    prob = IBProblem(grid, body, T(100), u0, IMAP())
    sol = CNAB(prob; dt=T(0.002), backend, delta=Immersa.DeltaYang3S())
    st = sol.state

    # 1. The unknowns are velocity + pressure only: no boundary-force block.
    #    `rhs_f` is the IBPM force block of the modified-Poisson RHS, and its
    #    absence is the structural difference between the two formulations.
    @test st isa Immersa.IMAPState
    @test !hasproperty(st, :rhs_f)
    @test hasproperty(sol.state, :rhs_φ)

    # 2. The projector lives in the coupler (an operator, not an unknown) and is
    #    built against the initialized geometry.
    @test sol.coupler isa Immersa.PrescribedBodyCoupler
    @test sol.coupler.Binv isa Immersa.IMAPCoupling
    @test sol.coupler.Binv.proj isa Immersa.ManifoldProjection

    # 3. Initialization: the free stream *projected onto the constraint manifold*,
    #    zero pressure, no history. The projection is required — IMAP conserves
    #    the constraint instead of imposing it, so an unprojected free-stream
    #    start would violate no-slip by U∞ for the whole run.
    @test noslip_residual(sol) < 1e-12
    let far = first(CartesianIndices(cell_axes(grid, Loc_u(1), Immersa.ExcludeBoundary())))
        @test st.u_full[1][far] ≈ 1                    # untouched far from the body
    end
    @test !all(no_offset_view(st.u_full[1]) .== 1)     # but changed near it
    @test all(no_offset_view(st.φ) .== 0)
    @test all(no_offset_view(st.rhs_φ) .== 0)
    @test st.nonlin_count == 0

    # 4. The interior unknowns `q` are seeded from the physical field and the
    #    halo is left at zero, as the homogeneous operators require.
    for i in 1:N
        R = CartesianIndices(cell_axes(grid, Loc_u(i), Immersa.ExcludeBoundary()))
        @test all(st.q[i][I] == st.u_full[i][I] for I in R)
        @test sum(abs, no_offset_view(st.q[i])) ≈ sum(abs, @view(st.q[i][R]))
    end

    # 5. The projector in the coupler is wired to the right geometry: an
    #    unprojected free stream violates no-slip by U∞, and one application of
    #    `P` removes it. (This is what `initialize_fields!` does above; here it is
    #    checked directly, starting from the raw free stream.)
    let q = Immersa.Ainv_zeros(backend, grid), ub = KernelAbstractions.zeros(backend, SVector{N,T}, nb)
        for i in eachindex(q)
            R = CartesianIndices(cell_axes(grid, Loc_u(i), Immersa.ExcludeBoundary()))
            @loop backend (I in R) q[i][I] = i == 1 ? 1 : 0
        end
        Immersa.interpolate_body!(ub, sol.reg, q)
        before = maximum(norm, Array(ub))
        @test before > 0.5                      # free stream: no-slip badly violated
        Immersa.P_mul!(q, sol.coupler.Binv.proj)
        Immersa.interpolate_body!(ub, sol.reg, q)
        @test maximum(norm, Array(ub)) < 1e-10 * before
    end

    # 6. Reinitializing is idempotent (no leftover history or pressure).
    initialize_fields!(sol)
    @test st.nonlin_count == 0
    @test all(no_offset_view(st.φ) .== 0)

    sol
end

function test_imap_setup(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(40, 40), x0=(-1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 2π, nb + 1)[1:(end-1)]) do t
            SVector(0.5cos(t), 0.5sin(t))
        end,
        ds = fill(2π * 0.5 / nb, nb)

        test_imap_setup(array, grid, xb, ds)

        # A StretchedGrid must be refused outright rather than silently running
        # with a projection that is not orthogonal in the mass inner product.
        let sgrid = StretchedGrid(;
                dx_min=0.05,
                core=[fill(-0.8, 2) fill(0.8, 2)],
                growth=1.2,
                extent=[fill(-3.0, 2) fill(3.0, 2)],
            ),
            sprob = IBProblem(
                sgrid, StaticBody(xb, ds), 100.0, UniformFlow(t -> SVector(1.0, 0.0)), IMAP()
            )

            @test_throws ArgumentError CNAB(sprob; dt=0.002)
        end
    end
    nothing
end

function test_imap_setup(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(40, 40, 40), x0=(-1.0, -1.0, -1.0), levels=1),
        nb = 20,
        xb = (array ∘ map)(range(0, 1, nb)) do t
            s = 2π * t
            SVector(0.5cos(s), 0.5sin(s), 0.5 * (2t - 1))
        end,
        ds = fill(0.16, nb)

        test_imap_setup(array, grid, xb, ds)
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
        (j, _) in axes_permutations(i),
        Iⱼ in (Rψb[i][j][begin], Rψb[i][j][end])

        R = CartesianIndices(setindex(Rψb[i], Iⱼ:Iⱼ, j))
        @loop backend (I in R) ψ[i][I] = 0
    end

    ψ_expect = map(i -> OffsetArray(ψ[i][Rψ[i]...], Rψ[i]), tupleindices(ψ))
    ψ_got = map(similar, ψ_expect)
    u = ntuple(N) do i
        dims = Ru[i]
        OffsetArray(
            KernelAbstractions.zeros(_backend(array), Float64, length.(dims)...), dims
        )
    end

    plan = Immersa.laplacian_plans(ψ_got, grid.n)

    Immersa.curl!(u, ψ; h=grid.h)
    Immersa.rot!(ψ_got, u; h=grid.h)
    Immersa.EigenbasisTransform(λ -> -1 / (λ / grid.h^2), plan)(ψ_got, ψ_got)

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
                i == j ? (n4:(3n4-1)) : ((n4+1):(3n4-1))
            end,
        )
        @loop backend (I in R_inner) ω²_got[i][I] = 0
    end

    Immersa.multidomain_coarsen!(ω²_got, ω¹; n=grid.n)

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

    Immersa.multidomain_interpolate!(ω_b_got, ω; n=grid.n)

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

    Immersa.set_boundary!(ω_got, ω_b)

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
        ui = map(tupleindices(u[lev])) do i
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

        Immersa.curl!(ui, ψ_expect[lev]; h)
        Immersa.rot!(ω[lev], u[lev]; h)
    end

    for lev in 2:grid.levels, (i, ωᵢ) in pairs(ω[lev])
        R_inner = CartesianIndices(
            ntuple(N) do j
                n4 = grid.n[j] ÷ 4
                i == j ? (n4:(3n4-1)) : ((n4+1):(3n4-1))
            end,
        )
        @loop backend (I in R_inner) ωᵢ[I] = 999
    end

    ψ_b = _boundary_array(_ -> zero(SVector{3}), array, grid, Loc_ω)

    plan = Immersa.laplacian_plans(ω[1], grid.n)

    Immersa.multidomain_poisson!(ω, ψ_got, u, ψ_b, grid, plan)

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

    reg = Immersa.Reg(backend, T, Immersa.DeltaYang3S(), nb, Val(N))
    Immersa.update_weights!(reg, grid, xb, eachindex(xb))

    R = ntuple(i -> cell_axes(grid, Loc_u(i), ExcludeBoundary()), N)

    u = _gridarray(u_true, array, grid, Loc_u, R)

    ub_expect = map(x -> u_true(x)[1:N], Array(xb))
    ub_got = KernelAbstractions.zeros(backend, SVector{N,T}, nb)
    Immersa.interpolate_body!(ub_got, reg, u)

    @test Array(ub_got) ≈ ub_expect

    fu = _gridarray(x -> zero(SVector{N}), array, grid, Loc_u, R)
    fb = KernelAbstractions.allocate(backend, SVector{N,T}, nb)
    fill!(fb, 1 .+ zero(SVector{N,T}))
    Immersa.regularize!(fu, reg, fb)

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
    ω_work = Immersa.grid_view(ω_work_bounds, grid, Loc_ω, ExcludeBoundary())

    f_work = similar(sol.f_tilde)

    # Make sure some vorticity is at the multi-domain boundary.
    for _ in 1:50
        step!(sol)
    end

    sol0 = map(1:length(sol.β)) do _
        s = deepcopy((; u=sol.state.u[1], ω=sol.state.ω[1]))
        step!(sol)
        s
    end

    Immersa.interpolate_body!(f_work, sol.reg, sol.state.u[1])
    unflatten(x) = reinterpret(reshape, T, x)
    @test unflatten(f_work) ≈ unflatten(sol.points.u) atol = 1e-4

    ω = Immersa.grid_view(deepcopy(sol0[end].ω), grid, Loc_ω, ExcludeBoundary())

    for i_step in eachindex(sol0)
        Immersa.nonlinear!(u_work, sol0[i_step].u, sol0[i_step].ω)
        Immersa.rot!(ω_work, u_work; h=grid.h)
        for i in eachindex(ω)
            let ω = ω[i], ω_work = ω_work[i], k = sol.dt * sol.β[i_step]
                @loop backend (I in CartesianIndices(ω)) ω[I] += k * ω_work[I]
            end
        end
    end

    for i in eachindex(ω)
        let ω0 = sol0[end].ω[i], ω1 = sol.state.ω[1][i], ω_work_b = ω_work_bounds[i]
            @loop backend (I in CartesianIndices(ω0)) ω_work_b[I] = ω0[I] + ω1[I]
        end
    end

    Immersa.curl!(u_work, ω_work_bounds; h=grid.h)
    Immersa.rot!(ω_work, u_work; h=grid.h)

    for i in eachindex(ω)
        let ω = ω[i], ω_work = ω_work[i], k = sol.dt / (2sol.prob.Re)
            @loop backend (I in CartesianIndices(ω)) ω[I] -= k * ω_work[I]
        end
    end

    Immersa.regularize!(u_work, sol.reg, sol.f_tilde)
    Immersa.rot!(ω_work, u_work; h=grid.h)

    for i in eachindex(ω)
        let ω = ω[i], ω_work = ω_work[i]
            @loop backend (I in CartesianIndices(ω)) ω[I] -= ω_work[I]
        end
    end

    let ω_got = Immersa.grid_view(sol.state.ω[1], grid, Loc_ω, ExcludeBoundary()), ω_expect = ω
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
        xb = (array ∘ map)(range(0, 2π, nb + 1)[1:(end-1)]) do t
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
    nonlin = map(eachindex(sol.state.nonlin)) do _
        grid_zeros(backend, grid, Loc_ω, ExcludeBoundary(); levels=1:grid.levels)
    end

    sol_i = sol.i
    sol_t = sol.t
    nonlin_count = sol.state.nonlin_count
    for level in 1:grid.levels
        for i in eachindex(ω[level])
            _set!(ω[level][i], sol.state.ω[level][i])
            _set!(ψ[level][i], sol.state.ψ[level][i])
            for k in eachindex(nonlin)
                _set!(nonlin[k][level][i], sol.state.nonlin[k][level][i])
            end
        end
        for i in eachindex(u[level])
            _set!(u[level][i], sol.state.u[level][i])
        end
    end

    io = IOBuffer()
    Immersa.save(io, sol)

    sol.i = -1
    sol.t = NaN
    sol.state.nonlin_count = -1
    for level in 1:grid.levels
        for i in eachindex(sol.state.ω[level])
            fill!(sol.state.ω[level][i], NaN)
            fill!(sol.state.ψ[level][i], NaN)
            for k in eachindex(sol.state.nonlin)
                fill!(sol.state.nonlin[k][level][i], NaN)
            end
        end
        for i in eachindex(sol.state.u[level])
            fill!(sol.state.u[level][i], NaN)
        end
    end

    seekstart(io)
    Immersa.load!(io, sol)

    @test sol.i == sol_i
    @test sol.t == sol_t
    @test sol.state.nonlin_count == nonlin_count
    @test all(
        no_offset_view(sol.state.ω[level][i]) == no_offset_view(ω[level][i]) for
        level in 1:grid.levels for i in eachindex(ω[level])
    )
    @test all(
        no_offset_view(sol.state.ψ[level][i]) == no_offset_view(ψ[level][i]) for
        level in 1:grid.levels for i in eachindex(ψ[level])
    )
    @test all(
        no_offset_view(sol.state.u[level][i]) == no_offset_view(u[level][i]) for
        level in 1:grid.levels for i in eachindex(u[level])
    )
    @test all(
        no_offset_view(sol.state.nonlin[k][level][i]) == no_offset_view(nonlin[k][level][i]) for
        k in 1:sol.state.nonlin_count for level in 1:grid.levels for
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
        sol.state.nonlin_count = 0

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
        sol.state.nonlin_count = 0

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
