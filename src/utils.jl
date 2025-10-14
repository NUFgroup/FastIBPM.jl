using Printf

# Simple timestep logger for examples
function log_timestep(i, t, wall_time)
    @printf(stderr, "iter %6d | wall_time %9.2f | sim_time %6.3f\n", i, wall_time, t)
end
axisunit(::Val{N}, i) where {N} = CartesianIndex(ntuple(==(i), N))
axisunit(::Val{N}) where {N} = Base.Fix1(axisunit, Val(N))
axisunit(::CartesianIndex{N}) where {N} = Base.Fix1(axisunit, Val(N))

struct OffsetTuple{O,T<:Tuple}
    x::T
    OffsetTuple{O}(x::T) where {O,T} = new{O,T}(x)
end

OffsetTuple(a::Tuple) = OffsetTuple{1}(a)
OffsetTuple(a::OffsetTuple) = a

Base.Tuple(a::OffsetTuple) = a.x

tupleindices(a::Tuple) = ntuple(identity, length(a))
function tupleindices(a::OffsetTuple{O}) where {O}
    OffsetTuple{O}(ntuple(i -> i - 1 + O, length(a)))
end

Base.length(a::OffsetTuple) = length(a.x)
Base.eachindex(a::OffsetTuple{O}) where {O} = (1:length(a)) .+ (O - 1)
Base.getindex(a::OffsetTuple{O}, i::Integer) where {O} = a.x[i-O+1]
Base.pairs(a::OffsetTuple{O}) where {O} = Base.Pairs(a, Tuple(tupleindices(a)))

Base.map(f, a::OffsetTuple{O}) where {O} = OffsetTuple{O}(map(f, a.x))

Base.iterate(a::OffsetTuple) = iterate(a.x)
Base.iterate(a::OffsetTuple, state) = iterate(a.x, state)

function Adapt.adapt_structure(to, a::OffsetTuple{O}) where {O}
    OffsetTuple{O}(Adapt.adapt_structure(to, a.x))
end

# Convert an array into a tuple of tuples
_nd_tuple(a::AbstractVector) = Tuple(a)
_nd_tuple(a::AbstractArray) = Tuple(map(i -> _nd_tuple(a[.., i]), axes(a, ndims(a))))

function otheraxes(i)
    j = i % 3 + 1
    k = (i + 1) % 3 + 1
    (j, k)
end

axes_permutations(i) = (otheraxes(i), reverse(otheraxes(i)))

struct Vec end
struct VecZ end
vec_kind(::Tuple) = Vec()
vec_kind(::OffsetTuple{3,<:NTuple{1}}) = VecZ()

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

outward(dir) = 2dir - 3

_cycle!(a::Vector) = length(a) < 2 ? a : pushfirst!(a, pop!(a))

const workgroup_size = Ref(64)

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

function _set!(b, a)
    @loop get_backend(b) (i in CartesianIndices(b)) b[i] = a[i]
    b
end

# Non-allocating sum(map(f, a, b)) for arrays.
function sum_map(f, a, b)
    s = zero(promote_type(eltype(a), eltype(b)))
    # b not included in eachindex to work on the GPU.
    for i in eachindex(a)
        s += f(a[i], b[i])
    end
    s
end

struct ArrayPool{B,V<:AbstractVector{UInt8}}
    backend::B
    size::Int
    mem::Vector{V}
    unused::Vector{Int}
end

struct ArrayPoolBlock{V<:AbstractVector{UInt8}}
    a::V
    index::Int
end

function ArrayPool(backend, size)
    V = typeof(KernelAbstractions.zeros(backend, UInt8, 1))
    ArrayPool(backend, size, V[], Int[])
end

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

function release!(pool::ArrayPool, block::ArrayPoolBlock)
    if block.index in pool.unused || !(block.index in eachindex(pool.mem))
        error("invalid block")
    end

    push!(pool.unused, block.index)
    nothing
end

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

function with_arrays_like(f, pool::ArrayPool, arrays...)
    shapes = map(arrays) do a
        (_array_eltype(a), _array_shape(a))
    end
    with_arrays(f, pool, shapes...)
end

_array_eltype(a::AbstractArray) = eltype(a)
_array_eltype(a) = _array_eltype(first(a))

_array_shape(a::AbstractArray) = axes(a)
_array_shape(a) = map(_array_shape, a)
