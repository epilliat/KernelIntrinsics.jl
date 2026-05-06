# Examples

## Memory Fences

Use `@fence` to ensure memory operations are visible across threads before proceeding:
```julia
using KernelIntrinsics
using KernelAbstractions, CUDA

@kernel function fence_kernel(X, Flag)
    X[1] = 10
    @fence  # Ensure X[1]=10 is visible to all threads before next operations
    Flag[1] = 1
end

X = cu([1])
Flag = cu([0])
fence_kernel(CUDABackend())(X, Flag; ndrange=1)
```

The `@fence` macro generates `fence.acq_rel.gpu` instructions in PTX assembly, ensuring proper memory ordering across the GPU.

## Ordered Memory Access

The `@access` macro provides acquire/release semantics for fine-grained memory ordering:
```julia
@kernel function access_kernel(X, Flag)
    if @index(Global, Linear) == 1
        X[1] = 10
        @access Flag[1] = 1  # Release store
    end

    # Other threads wait for Flag[1] == 1
    while (@access Acquire Flag[1]) != 1  # Acquire load
    end

    # Safely use X[1] here
end

X = cu([i for i in 1:1000])
Flag = cu([0])
access_kernel(CUDABackend())(X, Flag; ndrange=1000)
```

This generates `st.release.gpu` and `ld.acquire.gpu` instructions, providing lock-free synchronization patterns.

## Warp Operations

### Shuffle Operations

Exchange values between threads within a warp:
```julia
@kernel function shfl_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    offset = 1
    shuffled_val = @shfl(Up, val, offset)  # Default: warpsize=32, full mask
    dst[I] = shuffled_val
end

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
shfl_kernel(CUDABackend())(dst, src; ndrange=32)
# dst = [1, 1, 2, 3, 4, ..., 31]
```

### Shuffle with Custom Types

Unlike CUDA.jl, KernelIntrinsics.jl supports shuffle operations on arbitrary user-defined bitstype structs, including nested and composite types, as well as `NTuple`s:
```julia
struct Sub
    a::Float16
    b::UInt8
end

struct ComplexType
    x::Int32
    y::Sub
    z::Float64
end

@kernel function shfl_custom_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    offset = 1
    shuffled_val = @shfl(Up, val, offset)
    dst[I] = shuffled_val
end

# Nested structs
src = cu([ComplexType(i, Sub(i, i), i) for i in 1:32])
dst = cu([ComplexType(0, Sub(0, 0), 0) for i in 1:32])
shfl_custom_kernel(CUDABackend())(dst, src; ndrange=32)

# NTuples
src = cu([(Int32(i), Int32(i + 100)) for i in 1:32])
dst = cu([(Int32(0), Int32(0)) for _ in 1:32])
shfl_custom_kernel(CUDABackend())(dst, src; ndrange=32)
```

### Warp Reduce (Inclusive Scan)

Perform inclusive prefix sum within a warp:
```julia
@kernel function warpreduce_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    @warpreduce(val, +)
    dst[I] = val
end

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
warpreduce_kernel(CUDABackend())(dst, src; ndrange=32)
# dst = [1, 3, 6, 10, ..., 528]  (cumulative sum)
```

### Warp Fold (Reduction)

Reduce all values in a warp to a single result:
```julia
@kernel function warpfold_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    @warpfold(val, +)
    dst[I] = val
end

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
warpfold_kernel(CUDABackend())(dst, src; ndrange=32)
# dst[1] = 528 (sum of 1:32), rest are undefined
```

## Vectorized Memory Access

`vload` and `vstore!` issue hardware vector loads/stores when the underlying pointer is suitably aligned. The vector width depends on element type — `v4` for 32-bit (Int32/Float32), `v2` for 64-bit (Int64/Float64).

### Rebased indexing — block-of-`Nitem` (default)

In the default mode, `idx` is a **1-based block index**: `vload(A, i, Val(N))` loads the `i`-th contiguous block of `N` elements, i.e. `A[(i-1)*N+1 : i*N]`. This is the natural form when each thread owns a fixed-size tile of the array.
```julia
@kernel function rebased_kernel(dst, src, i)
    # i = 2, Nitem = 4 → loads block 2, i.e. elements 5,6,7,8
    values = vload(src, i, Val(4))
    vstore!(dst, i, values)            # writes back to elements 5,6,7,8
end

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
rebased_kernel(CUDABackend())(dst, src, 2; ndrange=1)
# dst[5:8] == [5, 6, 7, 8]
```
When the array's base pointer is `Nitem`-aligned (the common case for top-level GPU allocations), this lowers to a single `ld.global.v4` / `st.global.v4`. Otherwise alignment is resolved internally without user intervention.

### Direct indexing — start at exactly `idx`

Pass `Val(false)` as the fourth argument to load the literal range `A[idx : idx+N-1]`. Use this when the starting position is data-dependent and not necessarily a multiple of `N`.
```julia
@kernel function direct_kernel(dst, src, i)
    # i = 2, Nitem = 4 → loads elements 2,3,4,5 (no rebase)
    values = vload(src, i, Val(4), Val(false))
    vstore!(dst, i, values, Val(false))
end

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
direct_kernel(CUDABackend())(dst, src, 2; ndrange=1)
# dst[2:5] == [2, 3, 4, 5]
```
Direct indexing always goes through the runtime-aligned dispatch path (a mix of `ld.global.v4`, `ld.global.v2`, and scalar loads chosen by the actual offset), so it is correct for any `i` but slightly less aggressive than the aligned rebased fast path.

### Statically known alignment

If the alignment of the slice you load from is known at compile time, pass it as the fifth argument (`Val(k)` with `1 ≤ k ≤ Nitem`) to avoid the runtime check. `Val(1)` means "fully aligned"; `Val(k>1)` means "misaligned by `k`-elements" and emits a fixed pattern of vector + scalar instructions.
```julia
v = view(cu(Int32.(1:32)), 2:32)       # offset by 1 → known misalignment 2
values = vload(v, 1, Val(4), Val(true), Val(2))   # static (1,2,1) pattern, no branch
```

## Inspecting Generated Code

You can verify the generated PTX assembly to confirm proper instruction generation:
```julia
buf = IOBuffer()
CUDA.@device_code_ptx io = buf fence_kernel(CUDABackend())(X, Flag; ndrange=1)
asm = String(take!(buf))
occursin("fence.acq_rel.gpu", asm)  # true
```

Example: verifying vectorized instructions are generated:
```julia
@kernel function test_vload_kernel(a, b, i)
    y = vload(a, i, Val(4))
    b[1] = sum(y)
end

a = cu(Int32.(1:16))
b = cu(zeros(Int32, 4))

buf = IOBuffer()
CUDA.@device_code_ptx io = buf test_vload_kernel(CUDABackend())(a, b, 2; ndrange=1)
asm = String(take!(buf))
occursin("ld.global.v4", asm)  # true
```