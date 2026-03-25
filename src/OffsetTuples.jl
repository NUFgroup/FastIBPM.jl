module OffsetTuples

import Adapt

export OffsetTuple, tupleindices

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

end # module

