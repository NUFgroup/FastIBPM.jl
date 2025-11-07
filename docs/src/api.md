```@meta
CurrentModule = FastIBPM
```
# API Reference


# FastIBPM module
```@docs
FastIBPM
```

# CNAB
```@docs
AbstractCoupler
FastIBPM.NothingCoupler
FastIBPM.PrescribedBodyCoupler
FastIBPM.FsiCoupler
FastIBPM.CNAB
FastIBPM.CNAB_Binv_Precomputed
FastIBPM.initial_sol
FastIBPM.zero_vorticity!
FastIBPM.set_time!
FastIBPM.step!
FastIBPM.update_reg!
FastIBPM._A_factor
FastIBPM.Ainv
FastIBPM.prediction_step!
FastIBPM.coupling_step!
FastIBPM._coupling_step!
FastIBPM.B_inverse_rigid
FastIBPM.B_rigid_mul!
FastIBPM.B_deform_mul!
FastIBPM.f_to_f_tilde!
FastIBPM.redist!
FastIBPM.update_redist_weights!
FastIBPM.projection_step!
FastIBPM.apply_vorticity!
FastIBPM.ab_coeffs
FastIBPM._f_tilde_factor
FastIBPM.surface_force!
FastIBPM.surface_force_sum
FastIBPM.CNAB_signature
FastIBPM.save
FastIBPM.load!
```

# Problem Definition
```@docs
FastIBPM.GridKind
FastIBPM.Primal
FastIBPM.Dual
FastIBPM.GridLocation
FastIBPM.Node
FastIBPM.Edge
FastIBPM.Loc_u
FastIBPM.Loc_ω
FastIBPM.Grid
FastIBPM.gridcorner
FastIBPM.gridstep
FastIBPM.coord
FastIBPM._cellcoord
FastIBPM.IncludeBoundary
FastIBPM.ExcludeBoundary
FastIBPM.cell_axes
FastIBPM.grid_length
FastIBPM._on_bndry
FastIBPM.boundary_axes
FastIBPM.boundary_length
FastIBPM._exclude_boundary
FastIBPM.edge_axes
FastIBPM.grid_zeros
FastIBPM.boundary_zeros
FastIBPM.grid_view
FastIBPM.IrrotationalFlow
FastIBPM.UniformFlow
FastIBPM.add_flow!
FastIBPM.BodyPoints
Base.view(::FastIBPM.BodyPoints, ::Any)
FastIBPM.AbstractBody
FastIBPM.IBProblem
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