module ArrayPools

using KernelAbstractions

export ArrayPool, with_arrays, with_arrays_like

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

end # module
