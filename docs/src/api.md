# API Reference

## Vectorized Memory Access

### Basic Operations
```@docs
vload
vstore!
```

### Dynamic Alignment Operations

These functions handle arbitrary starting indices by computing alignment at runtime and dispatching to the appropriate vectorized instruction pattern.
```@docs
vload_multi
vstore_multi!
```

## Memory Ordering

```@docs
@access
@fence
```

## Warp Operations
```@docs
@shfl
@warpreduce
@warpfold
@vote
KernelIntrinsics.Direction
KernelIntrinsics.Up
KernelIntrinsics.Down
KernelIntrinsics.Xor
KernelIntrinsics.Idx
KernelIntrinsics.Mode
KernelIntrinsics.All
KernelIntrinsics.AnyLane
KernelIntrinsics.Uni
KernelIntrinsics.Ballot
```

## Index
```@index
```