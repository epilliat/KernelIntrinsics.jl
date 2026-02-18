# API Reference

## Vectorized Memory Access
### Basic Operations
```@docs
vload
vstore!
```
### Dynamic Alignment Operations
...
```@docs
vload_multi
vstore_multi!
```

## Memory Ordering
### Macros
```@docs
@fence
@access
```
### Scopes
```@docs
KernelIntrinsics.Scope
KernelIntrinsics.Workgroup
KernelIntrinsics.Device
KernelIntrinsics.System
```
### Orderings
```@docs
KernelIntrinsics.Ordering
KernelIntrinsics.Weak
KernelIntrinsics.Volatile
KernelIntrinsics.Relaxed
KernelIntrinsics.Acquire
KernelIntrinsics.Release
KernelIntrinsics.AcqRel
KernelIntrinsics.SeqCst
```

## Warp Operations
### Macros
```@docs
@warpsize
@shfl
@warpreduce
@warpfold
@vote
```
### Shuffle Directions
```@docs
KernelIntrinsics.Direction
KernelIntrinsics.Up
KernelIntrinsics.Down
KernelIntrinsics.Xor
KernelIntrinsics.Idx
```
### Vote Modes
```@docs
KernelIntrinsics.Mode
KernelIntrinsics.All
KernelIntrinsics.AnyLane
KernelIntrinsics.Uni
KernelIntrinsics.Ballot
```

## Index
```@index
```