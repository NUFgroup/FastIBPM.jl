using Printf

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
    struct OffsetTuple{O,T<:Tuple}
        x::T
    end

A wrapper for a `Tuple` (`x`) that associates it with a compile-time integer offset `O`.
This allows for indexing that starts at `O` instead of 1.

# Constructors
- `OffsetTuple{O}(x::Tuple)`: Creates an `OffsetTuple` with offset `O` wrapping tuple `x`.
- `OffsetTuple(a::Tuple)`: Convenience constructor. Defaults to `OffsetTuple{1}(a)`.
- `OffsetTuple(a::OffsetTuple)`: Returns the input `a` unchanged.
"""
struct OffsetTuple{O,T<:Tuple}
    x::T
    OffsetTuple{O}(x::T) where {O,T} = new{O,T}(x)
end

OffsetTuple(a::Tuple) = OffsetTuple{1}(a)
OffsetTuple(a::OffsetTuple) = a

"""
    Base.Tuple(a::OffsetTuple)

Converts an `OffsetTuple` back into a standard `Tuple` by unwrapping it.

# Arguments
- `a::OffsetTuple`: The offset tuple to convert.

# Returns
- `Tuple`: The underlying tuple `a.x`.
"""
Base.Tuple(a::OffsetTuple) = a.x

"""
    tupleindices(a::Tuple)
    tupleindices(a::OffsetTuple{O})

Returns a tuple representing the valid indices for `a`.

# Arguments
- `a::Tuple`: A standard tuple.

# Returns (Method 1)
- `Tuple`: A tuple of indices, `(1, 2, ..., length(a))`.

# Arguments
- `a::OffsetTuple{O}`: An offset tuple with offset `O`.

# Returns (Method 2)
- `OffsetTuple{O}`: An offset tuple of indices, `(O, O+1, ..., O + length(a) - 1)`.
"""
tupleindices(a::Tuple) = ntuple(identity, length(a))
function tupleindices(a::OffsetTuple{O}) where {O}
    OffsetTuple{O}(ntuple(i -> i - 1 + O, length(a)))
end

"""
    Base.length(a::OffsetTuple)

Returns the length of the `OffsetTuple`, which is the length of the underlying tuple it wraps.

# Arguments
- `a::OffsetTuple`: The offset tuple.

# Returns
- `Int`: The number of elements in `a.x`.
"""
Base.length(a::OffsetTuple) = length(a.x)

"""
    Base.eachindex(a::OffsetTuple{O})

Returns an iterable range of the valid, offset-aware indices for the `OffsetTuple`.

# Arguments
- `a::OffsetTuple{O}`: An offset tuple with offset `O`.

# Returns
- `UnitRange{Int}`: The range `O:(O + length(a) - 1)`.
"""
Base.eachindex(a::OffsetTuple{O}) where {O} = (1:length(a)) .+ (O - 1)

"""
    Base.getindex(a::OffsetTuple{O}, i::Integer)

Provides offset-based indexing for `OffsetTuple`. Accesses the `i`-th element,
where `i` is expected to be in the offset range (starting from `O`).

# Arguments
- `a::OffsetTuple{O}`: The offset tuple.
- `i::Integer`: The offset index to access.

# Returns
- The element at the corresponding internal index (`i - O + 1`).
"""
Base.getindex(a::OffsetTuple{O}, i::Integer) where {O} = a.x[i-O+1]

"""
    Base.pairs(a::OffsetTuple{O})

Returns an iterator that produces offset-aware `(index, value)` pairs.
The indices will start from the offset `O`.

# Arguments
- `a::OffsetTuple{O}`: The offset tuple.

# Returns
- `Base.Pairs`: An iterator for use in `for` loops.
"""
Base.pairs(a::OffsetTuple{O}) where {O} = Base.Pairs(a, Tuple(tupleindices(a)))

"""
    Base.map(f, a::OffsetTuple{O})

Applies a function `f` to each element of the `OffsetTuple`'s underlying tuple,
returning a new `OffsetTuple` with the same offset `O`.

# Arguments
- `f`: The function to apply.
- `a::OffsetTuple{O}`: The offset tuple.

# Returns
- `OffsetTuple{O}`: A new offset tuple containing the results of `f(x)` for each element `x` in `a`.
"""
Base.map(f, a::OffsetTuple{O}) where {O} = OffsetTuple{O}(map(f, a.x))

"""
    Base.iterate(a::OffsetTuple)
    Base.iterate(a::OffsetTuple, state)

Defines iteration for `OffsetTuple`. It iterates over the *values* of the
underlying tuple, just like a standard tuple.

# Arguments
- `a::OffsetTuple`: The offset tuple to iterate over.
- `state`: The iteration state (optional).

# Returns
- `(value, next_state)` or `nothing`.
"""
Base.iterate(a::OffsetTuple) = iterate(a.x)

"""
    Base.iterate(a::OffsetTuple)
    Base.iterate(a::OffsetTuple, state)

Defines iteration for `OffsetTuple`. It iterates over the *values* of the
underlying tuple, just like a standard tuple.

# Arguments
- `a::OffsetTuple`: The offset tuple to iterate over.
- `state`: The iteration state (optional).

# Returns
- `(value, next_state)` or `nothing`.
"""
Base.iterate(a::OffsetTuple, state) = iterate(a.x, state)

"""
    Adapt.adapt_structure(to, a::OffsetTuple{O})

Extends `Adapt.jl` to handle `OffsetTuple`. It recursively adapts the
underlying tuple `a.x` to the target backend `to` (e.g., a GPU) and
re-wraps the result in an `OffsetTuple` with the same offset `O`.

# Arguments
- `to`: The target backend (e.g., `CuArray`).
- `a::OffsetTuple{O}`: The offset tuple to adapt.

# Returns
- `OffsetTuple{O}`: A new offset tuple with its data adapted to the backend.
"""
function Adapt.adapt_structure(to, a::OffsetTuple{O}) where {O}
    OffsetTuple{O}(Adapt.adapt_structure(to, a.x))
end

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

"""
    struct ArrayPool{B,V<:AbstractVector{UInt8}}
        backend::B
        size::Int
        mem::Vector{V}
        unused::Vector{Int}
    end

A memory pool for reusing large, pre-allocated memory buffers (`Vector{UInt8}`)
to avoid the overhead of repeated allocation/deallocation, especially on GPUs.

# Fields
- `backend::B`: The `KernelAbstractions` backend (e.g., `CPU()`).
- `size::Int`: The size *in bytes* of each memory block in the pool.
- `mem::Vector{V}`: A vector storing all allocated memory blocks.
- `unused::Vector{Int}`: A stack of indices into `mem` pointing to available blocks.
"""
struct ArrayPool{B,V<:AbstractVector{UInt8}}
    backend::B
    size::Int
    mem::Vector{V}
    unused::Vector{Int}
end

"""
    struct ArrayPoolBlock{V<:AbstractVector{UInt8}}
        a::V
        index::Int
    end

A wrapper for a single memory block acquired from an `ArrayPool`.
It tracks the block's data (`a`) and its `index` in the pool's `mem` list
to ensure it can be returned correctly.
"""
struct ArrayPoolBlock{V<:AbstractVector{UInt8}}
    a::V
    index::Int
end

"""
    ArrayPool(backend, size)

Constructs a new, empty `ArrayPool`.

# Arguments
- `backend`: The `KernelAbstractions` backend to use for allocations.
- `size::Int`: The size *in bytes* for each block that will be allocated by this pool.

# Returns
- `ArrayPool`: A new, initialized pool.
"""
function ArrayPool(backend, size)
    V = typeof(KernelAbstractions.zeros(backend, UInt8, 1))
    ArrayPool(backend, size, V[], Int[])
end

"""
    acquire!(pool::ArrayPool)

Acquires a memory block from the `ArrayPool`.

If an unused block is available in the pool, it is reused.
If not, a new block of `pool.size` bytes is allocated using `pool.backend`.

# Arguments
- `pool::ArrayPool`: The memory pool to acquire from.

# Returns
- `ArrayPoolBlock`: A wrapper for the acquired memory block.
"""
function acquire!(pool::ArrayPool)
    if isempty(pool.unused)
        i = 1 + length(pool.mem)
        a = KernelAbstractions.zeros(pool.backend, UInt8, pool.size)
        push!(pool.mem, a)
    else
        i = pop!(pool.unused)
        a = pool.mem[i]
    end
    ArrayPoolBlock(a, i)
end

"""
    release!(pool::ArrayPool, block::ArrayPoolBlock)

Returns an `ArrayPoolBlock` to the pool, marking it as available for reuse.
Includes error checking to prevent double-releasing a block.

# Arguments
- `pool::ArrayPool`: The pool to return the block to.
- `block::ArrayPoolBlock`: The block that is no longer in use.

# Returns
- `nothing`
"""
function release!(pool::ArrayPool, block::ArrayPoolBlock)
    if block.index in pool.unused || !(block.index in eachindex(pool.mem))
        error("invalid block")
    end

    push!(pool.unused, block.index)
    nothing
end

"""
    with_arrays(f, pool::ArrayPool, shapes::Vararg{Any,N})

A higher-order function that safely manages temporary arrays from an `ArrayPool`.

It acquires `N` blocks, creates typed arrays from them based on the `shapes`
specification, calls `f(arrays...)`, and then ensures all blocks are
released back to the pool, even if `f` throws an error.

# Arguments
- `f`: A function to call, `f(array1, array2, ..., arrayN)`.
- `pool::ArrayPool`: The memory pool to use.
- `shapes`: A vararg of `(Type, shape)` tuples specifying the desired
  element type and shape for each temporary array.

# Returns
- The result of `f(arrays...)`.
"""
function with_arrays(f, pool::ArrayPool, shapes::Vararg{Any,N}) where {N}
    blocks = ntuple(_ -> acquire!(pool), Val(N))

    arrays = map(blocks, shapes) do block, (T, shape)
        T::Type
        i = Ref(0)
        _block_array(block, i, T, shape)
    end

    try
        return f(arrays...)
    finally
        foreach(block -> release!(pool, block), blocks)
    end
end

"""
    _block_array(block, i, T, shape)
    _block_array(block, i, T, xs)

Internal helper function for `with_arrays` to create a typed, shaped array
(or nested structure of arrays) from a raw `UInt8` memory block.

It uses `reinterpret` and `reshape` to create a view into the block.
The `i::Ref{Int}` tracks the current byte offset within the block.

# Arguments
- `block::ArrayPoolBlock`: The raw memory block.
- `i::Ref{Int}`: A mutable offset (in bytes) into the block.
- `T::Type`: The desired element type.
- `shape`: The desired shape (e.g., `(32, 64)`) or a nested structure of shapes.

# Returns
- A typed array view or a nested structure of array views.
"""
function _block_array(
    block::ArrayPoolBlock, i, T, shape::Tuple{Vararg{Union{Integer,AbstractRange}}}
)
    n = length(CartesianIndices(shape))
    s = sizeof(T)
    len = s * n
    r = i[] .+ (1:len)
    i[] += len

    @views reshape(reinterpret(T, block.a[r]), shape)
end

function _block_array(block::ArrayPoolBlock, i, T, xs)
    @assert !(xs isa Number)
    map(x -> _block_array(block, i, T, x), xs)
end

"""
    with_arrays_like(f, pool::ArrayPool, arrays...)

A convenience wrapper for `with_arrays`. It allocates temporary arrays
that have the same element type and shape as the provided `arrays`.

# Arguments
- `f`: A function to call with the new temporary arrays.
- `pool::ArrayPool`: The memory pool to use.
- `arrays...`: One or more reference arrays whose `eltype` and `shape`
  will be matched.

# Returns
- The result of `f(temp_array1, temp_array2, ...)`.
"""
function with_arrays_like(f, pool::ArrayPool, arrays...)
    shapes = map(arrays) do a
        (_array_eltype(a), _array_shape(a))
    end
    with_arrays(f, pool, shapes...)
end

"""
    _array_eltype(a::AbstractArray)
    _array_eltype(a)

Recursively determines the base element type of a potentially nested structure
(like a `Vector` of `Array`s).

# Arguments
- `a`: An `AbstractArray` or a nested container (e.g., `Tuple`, `Vector`).

# Returns
- `Type`: The element type of the first `AbstractArray` found.
"""
_array_eltype(a::AbstractArray) = eltype(a)
_array_eltype(a) = _array_eltype(first(a))

"""
    _array_shape(a::AbstractArray)
    _array_shape(a)

Recursively determines the shape (from `axes`) of a
potentially nested structure (like a `Vector` of `Array`s).

# Arguments
- `a`: An `AbstractArray` or a nested container (e.g., `Tuple`, `Vector`).

# Returns
- `Tuple`: The `axes` of the array.
- `Tuple` of `Tuple`s: A nested structure of shapes matching the input structure.
"""
_array_shape(a::AbstractArray) = axes(a)
_array_shape(a) = map(_array_shape, a)