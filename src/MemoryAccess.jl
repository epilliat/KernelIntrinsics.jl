module MemoryAccess

using KernelAbstractions

export @fence, @access
export vectorized_load, vectorized_store!
export vectorized_cached_load

export @shfl, @warpreduce, _vote, @vote
#export atomic_store, atomic_load, fence
#export Workgroup, Device, System
#export Acquire, Release, AcqRel, SeqCst, Weak, Volatile, Relaxed

# ============================================================================
# Abstract type definitions for compile-time dispatch
# ============================================================================

"""
    Scope

Abstract type representing the scope of a memory fence.

Subtypes:
- `Workgroup`: Thread block/workgroup scope
- `Device`: Device/GPU scope
- `System`: System scope (includes CPU and other devices)
"""
abstract type Scope end

"""
    Workgroup <: Scope

Thread block/workgroup scope fence.
Synchronizes memory operations within a single thread block/workgroup.
"""
struct Workgroup <: Scope end

"""
    Device <: Scope

Device/GPU scope fence.
Synchronizes memory operations across all thread blocks on a single device.
"""
struct Device <: Scope end

"""
    System <: Scope

System scope fence.
Synchronizes memory operations across all devices, including CPU.
"""
struct System <: Scope end

# ============================================================================
# Abstract type for memory orderings
# ============================================================================

"""
    Ordering

Abstract type representing the memory ordering semantics of a fence.

Subtypes:
- `Weak`: Weak memory ordering (minimal guarantees)
- `Volatile`: Volatile semantics (prevents compiler optimization)
- `Relaxed`: Relaxed ordering (atomicity only, no synchronization)
- `Acquire`: Acquire semantics
- `Release`: Release semantics
- `AcqRel`: Acquire-Release semantics
- `SeqCst`: Sequential consistency
"""
abstract type Ordering end

"""
    Weak <: Ordering

Weak memory ordering.
Provides minimal ordering guarantees, allowing maximum hardware and compiler
reordering flexibility.
"""
struct Weak <: Ordering end

"""
    Volatile <: Ordering

Volatile memory ordering.
Prevents compiler optimizations from caching or reordering operations on the
variable, ensuring each access reads from or writes to memory. Does not provide
atomicity guarantees without additional synchronization.
"""
struct Volatile <: Ordering end

"""
    Relaxed <: Ordering

Relaxed memory ordering.
Ensures atomicity of individual operations but provides no synchronization
guarantees. Operations may be reordered freely by hardware and compiler.
Different threads may observe different orderings of relaxed atomic operations
on different variables.
"""
struct Relaxed <: Ordering end

"""
    Acquire <: Ordering

Acquire memory ordering.
Ensures that memory operations after the fence see all writes that happened
before a corresponding release operation.
"""
struct Acquire <: Ordering end

"""
    Release <: Ordering

Release memory ordering.
Ensures that all memory operations before the fence are visible to threads
that subsequently perform an acquire operation.
"""
struct Release <: Ordering end

"""
    AcqRel <: Ordering

Acquire-Release memory ordering.
Combines both acquire and release semantics.
"""
struct AcqRel <: Ordering end

"""
    SeqCst <: Ordering

Sequential consistency.
Provides the strongest memory ordering guarantees, establishing a total
order of all sequentially consistent operations.
"""
struct SeqCst <: Ordering end

# ============================================================================
# Generic fence interface
# ============================================================================

"""
    fence(scope::Type{<:Scope}, order::Type{<:Ordering})

Emit a memory fence instruction with the specified scope and ordering.

Backend-specific implementations are provided via device-specific modules.

# Arguments
- `scope`: The scope of the fence (Workgroup, Device, or System)
- `order`: The memory ordering (Weak, Volatile, Relaxed, Acquire, Release, AcqRel, or SeqCst)

# Examples
fence(Device, Acquire) # Device-wide acquire fence
fence(Device, Release) # Device-wide release fence
fence(Device, AcqRel) # Device-wide acquire-release fence
fence(Workgroup, SeqCst) # Workgroup-level sequentially consistent fence
fence(System, AcqRel) # System-wide acquire-release fence
fence(Device, Relaxed) # Device-wide relaxed fence

# Notes
Backend implementations may have limitations. For example:
- NVIDIA PTX does not have separate acquire/release fences; both map to acq_rel
- Different backends may have different performance characteristics
- Weak and Volatile orderings may not be directly supported on all hardware
"""
function fence end

function atomic_store! end
function atomic_load end

@inline function vectorized_load(A::AbstractArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    id_base = (idx - 1) * Nitem
    return ntuple(i -> A[id_base+i], Val(Nitem))
end
@inline function vectorized_cached_load(A::AbstractArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    id_base = (idx - 1) * Nitem
    return ntuple(i -> A[id_base+i], Val(Nitem))
end


@inline function vectorized_store!(A::AbstractArray{T}, idx, values::NTuple{Nitem,T}) where {T,Nitem}
    id_base = (idx - 1) * Nitem
    for i in (1:Nitem)
        A[id_base+i] = values[i]
    end
    return
end

include("warp.jl")
include("macros.jl")


end # module MemoryAccess
