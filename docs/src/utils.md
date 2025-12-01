```@meta
CurrentModule = FastIBPM
```

# Utilities
```@docs
FastIBPM.log_timestep
FastIBPM.axisunit
FastIBPM.OffsetTuple
Base.Tuple(::FastIBPM.OffsetTuple)
FastIBPM.tupleindices
Base.length(::FastIBPM.OffsetTuple)
Base.eachindex(::FastIBPM.OffsetTuple)
Base.getindex(::FastIBPM.OffsetTuple, ::Integer)
Base.pairs(::FastIBPM.OffsetTuple)
Base.map(::Any, ::FastIBPM.OffsetTuple)
Base.iterate(::FastIBPM.OffsetTuple)
Base.iterate(::FastIBPM.OffsetTuple, ::Any)
Adapt.adapt_structure(::Any, ::FastIBPM.OffsetTuple)
FastIBPM._nd_tuple
FastIBPM.otheraxes
FastIBPM.axes_permutations
FastIBPM.Vec
FastIBPM.VecZ
FastIBPM.vec_kind
FastIBPM.sumcross
FastIBPM.outward
FastIBPM._cycle!
FastIBPM.workgroup_size
FastIBPM.@loop
FastIBPM._set!
FastIBPM.sum_map
FastIBPM.ArrayPool
FastIBPM.ArrayPoolBlock
FastIBPM.acquire!
FastIBPM.release!
FastIBPM.with_arrays
FastIBPM._block_array
FastIBPM.with_arrays_like
FastIBPM._array_eltype
FastIBPM._array_shape
```
