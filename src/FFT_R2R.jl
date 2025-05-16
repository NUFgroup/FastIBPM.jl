module FFT_R2R

using LinearAlgebra
using KernelAbstractions
using AbstractFFTs: plan_fft, plan_rfft
import AbstractFFTs
import FFTW

plan_r2r!(A, args...; kw...) = _plan_r2r!(get_backend(A), A, args...; kw...)

function _plan_r2r!(::CPU, args...; kw...)
    FFTW.plan_r2r!(args...; num_threads=Threads.nthreads(), kw...)
end

_plan_r2r!(dev, A, kind, args...; kw...) = bad_plan_r2r!(A, Val.(kind), args...; kw...)

struct R2R{P<:Tuple}
    p::P
end

function bad_plan_r2r!(A, kind::Tuple, dims::Tuple=ntuple(identity, ndims(A)); kw...)
    p = ntuple(i -> bad_plan_r2r!(A, kind[i], dims[i]; kw...), ndims(A))
    R2R(p)
end

function LinearAlgebra.mul!(y, (; p)::R2R, x)
    mul!(y, p[1], x)
    for i in eachindex(p)[2:end]
        mul!(y, p[i], y)
    end
    y
end

struct RODFT00{P<:AbstractFFTs.Plan,A<:AbstractArray,B<:AbstractArray}
    dims::Int
    p::P
    a::A
    b::B
end

function bad_plan_r2r!(A, ::Val{FFTW.RODFT00}, dims::Int; kw...)
    s = size(A)
    a = similar(A, Base.setindex(s, 2(s[dims] + 1), dims))
    b = similar(A, complex(eltype(A)), Base.setindex(s, s[dims] + 2, dims))
    Base.require_one_based_indexing(a, b)

    p = plan_rfft(a, dims; kw...)
    RODFT00(dims, p, a, b)
end

function LinearAlgebra.mul!(y, (; dims, p, a, b)::RODFT00, x)
    n = size(x, dims)
    selectdim(a, dims, 1) .= 0
    selectdim(a, dims, 1 .+ (1:n)) .= x
    selectdim(a, dims, n + 2) .= 0
    selectdim(a, dims, n + 2 .+ (1:n)) .= .-selectdim(x, dims, n:-1:1)
    mul!(b, p, a)
    let b1 = selectdim(b, dims, 1 .+ (1:n))
        @. y = -imag(b1)
    end
    y
end

struct REDFT10{P<:AbstractFFTs.Plan,A<:AbstractArray,B<:AbstractArray}
    dims::Int
    p::P
    a::A
    b::B
end

function bad_plan_r2r!(A, ::Val{FFTW.REDFT10}, dims::Int; kw...)
    s = size(A)
    a = similar(A, Base.setindex(s, 2s[dims], dims))
    b = similar(A, complex(eltype(A)), Base.setindex(s, s[dims] + 1, dims))
    Base.require_one_based_indexing(a, b)

    p = plan_rfft(a, dims; kw...)
    REDFT10(dims, p, a, b)
end

function LinearAlgebra.mul!(y, (; dims, p, a, b)::REDFT10, x)
    n = size(x, dims)
    selectdim(a, dims, 1:n) .= x
    selectdim(a, dims, n+1:2n) .= selectdim(x, dims, n:-1:1)
    mul!(b, p, a)

    k = reshape(0:n-1, ntuple(i -> i == dims ? n : 1, ndims(x)))
    let b1 = selectdim(b, dims, 1:n)
        @. y = real(exp(-1im * π * k / (2n)) * b1)
    end
    y
end

struct REDFT01{P<:AbstractFFTs.Plan,A<:AbstractArray,B<:AbstractArray}
    dims::Int
    p::P
    a::A
    b::B
end

function bad_plan_r2r!(A, ::Val{FFTW.REDFT01}, dims::Int; kw...)
    s = size(A)
    a = similar(A, complex(eltype(A)), Base.setindex(s, 2s[dims], dims))
    b = similar(a)
    Base.require_one_based_indexing(a, b)

    p = plan_fft(a, dims; kw...)
    REDFT01(dims, p, a, b)
end

function LinearAlgebra.mul!(y, (; dims, p, a, b)::REDFT01, x)
    n = size(x, dims)
    k = reshape(0:n-1, ntuple(i -> i == dims ? n : 1, ndims(x)))
    let a1 = selectdim(a, dims, 1:n)
        @. a1 = exp(-1im * π * k / (2n)) * x
    end
    selectdim(a, dims, n+1:2n) .= 0
    mul!(b, p, a)

    let b1 = selectdim(b, dims, 1:n), x1 = selectdim(x, dims, 1:1)
        @. y = 2 * real(b1) - x1
    end
    y
end

end
