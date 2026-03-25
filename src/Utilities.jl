module Utilities

using Immersa.OffsetTuples
using Printf
using KernelAbstractions
using EllipsisNotation

export axisunit, _nd_tuple, otheraxes, axes_permutations
export Vec, VecZ, vec_kind, sumcross
export outward
export _cycle!
export workgroup_size, @loop
export _set!
export sum_map

"""
    log_timestep(i, t, wall_time)

Print a single-line status message for the current simulation step.

The message is written to `stderr` (so it appears in job logs) in the form:
`iter <i> | wall_time <seconds> | sim_time <t>`.

# Arguments
- `i::Integer`: Iteration or timestep index.
- `t::Real`: Simulation time (typically in flow time units).
- `wall_time::Real`: Elapsed wall-clock time in seconds.

# Notes
- Writes via `@printf(stderr, ...)` so it won’t mix with data printed to `stdout`.
- Use inside callbacks or your time-stepping loop.

# Example
```julia
julia> log_timestep(42, 0.125, 3.57)
# prints to stderr:
# iter     42 | wall_time      3.57 | sim_time  0.125
"""
function log_timestep(i, t, wall_time)
    @printf(stderr, "iter %6d | wall_time %9.2f | sim_time %6.3f\n", i, wall_time, t)
end

"""
    axisunit(Val(N), i)
    axisunit(Val(N))
    axisunit(I::CartesianIndex)

Creates a unit vector of dimension `N` in the `i`-th direction, represented as a `CartesianIndex`.
This is useful for moving along a specific grid axis.

# Arguments (Method 1)
- `::Val{N}`: A `Val` type specifying the total number of dimensions `N`.
- `i`: The dimension for the unit vector.

# Returns (Method 1)
- `CartesianIndex`: A unit vector, e.g., `CartesianIndex((0, 1, 0))` for `N=3, i=2`.

# Arguments (Method 2)
- `::Val{N}`: A `Val` type specifying the dimension.

# Returns (Method 2)
- A function `f(i)` that creates the unit vector in direction `i`.

# Arguments (Method 3)
- `I::CartesianIndex{N}`: An existing `CartesianIndex` used to infer the dimension `N`.

# Returns (Method 3)
- A function `f(i)` that creates the unit vector in direction `i`.
"""
axisunit(::Val{N}, i) where {N} = CartesianIndex(ntuple(==(i), N))
axisunit(::Val{N}) where {N} = Base.Fix1(axisunit, Val(N))
axisunit(::CartesianIndex{N}) where {N} = Base.Fix1(axisunit, Val(N))

"""
    _nd_tuple(a::AbstractArray)

Recursively converts an N-dimensional array into a nested tuple of its elements.
The nesting follows the array's dimensions.

# Arguments
- `a::AbstractArray`: The array to convert. A 1D vector will become a single tuple.
  A 2D matrix will become a tuple of tuples (column-wise).

# Returns
- `Tuple`: A nested tuple structure matching the array's data.
"""
_nd_tuple(a::AbstractVector) = Tuple(a)
_nd_tuple(a::AbstractArray) = Tuple(map(i -> _nd_tuple(a[.., i]), axes(a, ndims(a))))

"""
    otheraxes(i)

Given an axis index `i` (1, 2, or 3), returns the other two axes in cyclic order.

# Arguments
- `i::Int`: The current axis index (1, 2, or 3).

# Returns
- `Tuple{Int, Int}`: The other two axes. (e.g., `i=1` returns `(2, 3)`).
"""
function otheraxes(i)
    j = i % 3 + 1
    k = (i + 1) % 3 + 1
    (j, k)
end

"""
    axes_permutations(i)

Given an axis index `i`, returns the permutations of the other two axes.

# Arguments
- `i::Int`: The current axis index (1, 2, or 3).

# Returns
- `Tuple{Tuple{Int, Int}, Tuple{Int, Int}}`: A tuple containing the forward and
  reverse permutations of the other two axes. (e.g., `i=1` returns `((2, 3), (3, 2))`).
"""
axes_permutations(i) = (otheraxes(i), reverse(otheraxes(i)))

"""
    struct Vec end

Empty struct used as a tag for dispatch, likely to distinguish between
different types of vector-like objects (e.g., a full 3D vector).
"""
struct Vec end

"""
    struct VecZ end

Empty struct used as a tag for dispatch, likely to distinguish
a Z-only component in functions like `sumcross`.
"""
struct VecZ end

"""
    vec_kind(::Tuple)
    vec_kind(::OffsetTuple{3,<:NTuple{1}})

A dispatch function that returns a `Vec` or `VecZ` tag based on the input's type.

# Arguments
- `::Tuple`: Matches any standard tuple.
- `::OffsetTuple{3,<:NTuple{1}}`: Matches a specific `OffsetTuple` (offset 3, 1-element).

# Returns
- `Vec()` for a standard tuple.
- `VecZ()` for the specific `OffsetTuple` type.
"""
vec_kind(::Tuple) = Vec()
vec_kind(::OffsetTuple{3,<:NTuple{1}}) = VecZ()

"""
    sumcross(f, i::Int)
    sumcross(f, i, ::Vec, ::Vec)
    sumcross(f, i::Int, ::Vec, ::VecZ)
    sumcross(f, i, a::VecZ, b::Vec)

Computes antisymmetric combinations (like cross-products or curl components)
of a function `f` applied to 3D axis indices.

The function `f` is expected to take two indices, `f(j, k)`.

# Arguments
- `f`: A function `(Int, Int) -> Value`.
- `i::Int`: The primary axis index (1, 2, or 3).
- `::Vec`, `::VecZ`: Tags used for dispatch to select the correct formula.

# Returns
- A value representing the antisymmetric combination.
- Method 1 (base): `f(j, k) - f(k, j)` where `(j, k) = otheraxes(i)`.
- Method 2 (Vec, Vec): Dispatches to Method 1.
- Method 3 (Vec, VecZ): Specialized form for `i=1` or `i=2`.
- Method 4 (VecZ, Vec): Negation of Method 3 with flipped arguments.
"""
function sumcross(f, i::Int)
    (j, k) = otheraxes(i)
    f(j, k) - f(k, j)
end

sumcross(f, i, ::Vec, ::Vec) = sumcross(f, i)

function sumcross(f, i::Int, ::Vec, ::VecZ)
    @assert i in (1, 2)
    i == 1 ? f(2, 3) : -f(1, 3)
end

sumcross(f, i, a::VecZ, b::Vec) = -sumcross((x, y) -> f(y, x), i, b, a)

"""
    outward(dir)

Maps a direction index (1 or 2) to a sign (-1 or 1).
- `outward(1)` returns `-1`.
- `outward(2)` returns `1`.

# Arguments
- `dir::Int`: The direction index (1 or 2).

# Returns
- `Int`: -1 or 1.
"""
outward(dir) = 2dir - 3

"""
    _cycle!(a::Vector)

Performs an in-place right circular shift on a vector.
The last element becomes the first.

# Arguments
- `a::Vector`: The vector to be modified in-place.

# Returns
- The modified vector `a`.
"""
_cycle!(a::Vector) = length(a) < 2 ? a : pushfirst!(a, pop!(a))

"""
    const workgroup_size = Ref(64)

A mutable, global reference to the workgroup size used by `KernelAbstractions.jl` kernels,
particularly in the `@loop` macro.

This is a `const Ref` so the *reference* cannot be reassigned, but the *value*
can be changed via `workgroup_size[] = 128`.
"""
const workgroup_size = Ref(64)

"""
    @loop backend (I in R) ex

A macro to simplify launching parallel kernels using `KernelAbstractions.jl`.
It automatically defines and launches a kernel that executes the expression `ex`
over the `CartesianIndices` `R`.

# Arguments
- `backend`: The `KernelAbstractions` backend (e.g., `CPU()`, `CUDABackend()`).
- `(I in R)`: The loop specification, where `I` is the index symbol and `R`
  is a `CartesianIndices` range.
- `ex`: The code block (loop body) to execute for each index `I`.
"""
macro loop(backend, inds, ex)
    if !(
        inds isa Expr &&
        inds.head == :call &&
        length(inds.args) == 3 &&
        inds.args[1] == :(in) &&
        inds.args[2] isa Symbol
    )
        throw(ArgumentError("second argument must be in the form `I in R`"))
    end

    I = esc(inds.args[2])
    R = esc(inds.args[3])
    kern = esc(gensym("kern"))
    I0 = esc(gensym("I0"))
    quote
        @kernel function $kern(@Const($I0))
            $I = @index(Global, Cartesian)
            $I += $I0
            $(esc(ex))
        end
        R = $R::CartesianIndices
        R1 = R[begin]
        backend = $(esc(backend))
        $kern(backend, workgroup_size[])(R1 - oneunit(R1); ndrange=size(R))
    end
end

"""
    _set!(b, a)

Performs a parallel copy of the contents of array `a` into array `b`,
using the `@loop` macro. `b` is modified in-place.

# Arguments
- `b`: The destination array (must be KA-compatible).
- `a`: The source array.

# Returns
- The modified array `b`.
"""
function _set!(b, a)
    @loop get_backend(b) (i in CartesianIndices(b)) b[i] = a[i]
    b
end

"""
    sum_map(f, a, b)

Computes `sum(f(a[i], b[i]) for i in eachindex(a))` in a non-allocating manner.
This avoids creating an intermediate array of results, which is more
efficient, especially on GPUs.

# Arguments
- `f`: A function that takes two arguments `(f(::EltypeA, ::EltypeB))`.
- `a`: The first array.
- `b`: The second array.

# Returns
- The sum of `f` applied element-wise to `a` and `b`.
"""
function sum_map(f, a, b)
    s = zero(promote_type(eltype(a), eltype(b)))
    # b not included in eachindex to work on the GPU.
    for i in eachindex(a)
        s += f(a[i], b[i])
    end
    s
end

end # module

