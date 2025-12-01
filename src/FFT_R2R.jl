"""
    FFT_R2R.jl

Provides a real-to-real (R2R) Fast Fourier Transform (FFT) interface and implementations
using `FFTW.jl`, with GPU/CPU support via `KernelAbstractions.jl`.

This module extends Julia's linear algebra interface by overloading `mul!` to
support custom FFT-based operations for real-valued arrays.

# Features
- Real-to-real FFT transforms (R2R)
- CPU/GPU dispatch via `KernelAbstractions`
- Integration with Julia’s `LinearAlgebra` API
"""
module FFT_R2R



# ------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------

# For mul! and other matrix operations
using LinearAlgebra 

# For backend dispatching (CPU/GPU)
using KernelAbstractions

# For planning and executing FFTs
using AbstractFFTs: plan_fft, plan_rfft
import AbstractFFTs
import FFTW

# ------------------------------------------------------------------------



"""
    plan_r2r!(A, args...; kw...)

Create a plan for a real-to-real (R2R) FFT on array `A`. 

This function:
- Chooses the appropriate backend (CPU/GPU) automatically.
- Delegates to `_plan_r2r!` for backend-specific planning.

# Arguments
- `A`: the array to transform.
- `args...`: FFTW-compatible transform arguments (e.g., type of R2R transform).
- `kw...`: optional keyword arguments.

# Returns
A plan object that can be used with `mul!(y, plan, x)` to apply the transform efficiently.

# Notes
- On CPU, uses `FFTW.plan_r2r!` with multithreading.
- If the requested transform is not natively supported, a fallback plan (e.g., `RODFT00`, `REDFT10`) is created automatically.
"""
plan_r2r!(A, args...; kw...) = _plan_r2r!(get_backend(A), A, args...; kw...)

function _plan_r2r!(::CPU, args...; kw...)
    FFTW.plan_r2r!(args...; num_threads=Threads.nthreads(), kw...)
end

_plan_r2r!(dev, A, kind, args...; kw...) = bad_plan_r2r!(A, Val.(kind), args...; kw...)

struct R2R{P<:Tuple} # Defines a container for a tuple of FFT plans
    p::P
end



"""
    bad_plan_r2r(A, args...; kw...)

Internal helper to create multi-dimensional R2R FFT plans.

# Returns
Returns a container (`R2R`) holding 1D FFT plans for each dimension.
"""
function bad_plan_r2r!(A, kind::Tuple, dims::Tuple=ntuple(identity, ndims(A)); kw...)
    p = ntuple(i -> bad_plan_r2r!(A, kind[i], dims[i]; kw...), ndims(A))
    R2R(p)
end



"""
    LinearAlgebra.mul!(y, (; p), x)

Applies a multi-dimensional R2R FFT plan to an array `x` and stores the result in `y`.
This overload of `LinearAlgebra.mul!` runs each 1D plan sequentially.
"""
function LinearAlgebra.mul!(y, (; p)::R2R, x)
    mul!(y, p[1], x)
    for i in eachindex(p)[2:end]
        mul!(y, p[i], y)
    end
    y
end



"""
    RODFT00{P,A,B}

Represents a type-I sine transform (FFTW's RODFT00) along a specific dimension.

# Fields
- `dims`: dimension of the transform,
- `p`: the FFT plan,
- `a`, `b`: internal arrays for staging data.
"""
struct RODFT00{P<:AbstractFFTs.Plan,A<:AbstractArray,B<:AbstractArray}
    dims::Int
    p::P
    a::A
    b::B
end



"""
    bad_plan_r2r!(A, ::Val{FFTW.RODFT00}, dims; kw...)

Creates a plan for a type-I sine transform (RODFT00) along dimension `dims` of array `A`.
Internally, it allocates staging arrays and builds an RFFT plan that emulates the desired R2R transform.

# Arguments
- `A`: input array
- `::Val{FFTW.RODFT00}`: indicates the transform type
- `dims`: the dimension along which to apply the transform
- `kw...`: keyword arguments forwarded to `plan_rfft`

# Returns
- `RODFT00` object containing the plan and internal arrays
"""
function bad_plan_r2r!(A, ::Val{FFTW.RODFT00}, dims::Int; kw...)
    s = size(A)
    a = similar(A, Base.setindex(s, 2(s[dims] + 1), dims))
    b = similar(A, complex(eltype(A)), Base.setindex(s, s[dims] + 2, dims))
    Base.require_one_based_indexing(a, b)

    p = plan_rfft(a, dims; kw...)
    RODFT00(dims, p, a, b)
end



"""
    LinearAlgebra.mul!(y, (; dims, p, a, b)::RODFT00, x)

Applies a type-I sine transform (RODFT00) along the specified dimension `dims`. 
Stages `x` into internal array `a`, applies the FFT plan `p` to get `b`, and writes the processed result into `y`.
"""
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



"""
    REDFT10{P,A,B}

Represents a type-II cosine transform (FFTW's REDFT10) along a specific dimension.

# Fields
- `dims`: dimension of the transform,
- `p`: the FFT plan,
- `a`, `b`: internal arrays for staging data.
"""
struct REDFT10{P<:AbstractFFTs.Plan,A<:AbstractArray,B<:AbstractArray}
    dims::Int
    p::P
    a::A
    b::B
end



"""
    bad_plan_r2r!(A, ::Val{FFTW.REDFT10}, dims; kw...)

Creates a plan for a type-II cosine transform (REDFT10) along dimension `dims` of array `A`.
Internally, it allocates staging arrays and builds an RFFT plan that emulates the R2R transform.

# Arguments
- `A`: input array
- `::Val{FFTW.REDFT10}`: indicates the transform type
- `dims`: the dimension along which to apply the transform
- `kw...`: keyword arguments forwarded to `plan_rfft`

# Returns
- `REDFT10` object containing the plan and internal arrays
"""
function bad_plan_r2r!(A, ::Val{FFTW.REDFT10}, dims::Int; kw...)
    s = size(A)
    a = similar(A, Base.setindex(s, 2s[dims], dims))
    b = similar(A, complex(eltype(A)), Base.setindex(s, s[dims] + 1, dims))
    Base.require_one_based_indexing(a, b)

    p = plan_rfft(a, dims; kw...)
    REDFT10(dims, p, a, b)
end



"""
    LinearAlgebra.mul!(y, (; dims, p, a, b)::REDFT10, x)

Applies a type-II cosine transform (REDFT10) along dimension `dims`. 
Mirrors `x`, applies FFT plan `p`, then multiplies by a phase factor to match the transform formula.
"""
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



"""
    REDFT01{P,A,B}

Represents a type-III cosine transform (FFTW's REDFT01) along a specific dimension.

# Fields
- `dims`: dimension of the transform,
- `p`: the FFT plan,
- `a`, `b`: internal arrays for staging data.
"""
struct REDFT01{P<:AbstractFFTs.Plan,A<:AbstractArray,B<:AbstractArray}
    dims::Int
    p::P
    a::A
    b::B
end



"""
    bad_plan_r2r!(A, ::Val{FFTW.REDFT01}, dims; kw...)

Creates a plan for a type-III cosine transform (REDFT01) along dimension `dims` of array `A`.
Internally, it allocates staging arrays and builds an FFT plan that emulates the R2R transform.

# Arguments
- `A`: input array
- `::Val{FFTW.REDFT01}`: indicates the transform type
- `dims`: the dimension along which to apply the transform
- `kw...`: keyword arguments forwarded to `plan_fft`

# Returns
- `REDFT01` object containing the plan and internal arrays
"""
function bad_plan_r2r!(A, ::Val{FFTW.REDFT01}, dims::Int; kw...)
    s = size(A)
    a = similar(A, complex(eltype(A)), Base.setindex(s, 2s[dims], dims))
    b = similar(a)
    Base.require_one_based_indexing(a, b)

    p = plan_fft(a, dims; kw...)
    REDFT01(dims, p, a, b)
end



"""
    LinearAlgebra.mul!(y, (; dims, p, a, b)::REDFT01, x)

Applies a type-III cosine transform (REDFT01) along dimension `dims`. 
Pre-multiplies `x` by a phase factor, applies FFT plan `p`, and extracts the processed result into `y`.
"""
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
