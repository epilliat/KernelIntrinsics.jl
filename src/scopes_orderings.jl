# --- Scope ---

"""
    Scope

Abstract type representing the scope of a memory operation or fence.

Subtypes:
- [`Workgroup`](@ref): Thread block/workgroup scope
- [`Device`](@ref): Device/GPU scope
- [`System`](@ref): System scope (includes CPU and other devices)
"""
abstract type Scope end

"""
    Workgroup <: Scope

Thread block/workgroup scope.
Synchronizes memory operations within a single thread block/workgroup.
"""
struct Workgroup <: Scope end

"""
    Device <: Scope

Device/GPU scope.
Synchronizes memory operations across all thread blocks on a single device.
"""
struct Device <: Scope end

"""
    System <: Scope

System scope.
Synchronizes memory operations across all devices, including the CPU.
"""
struct System <: Scope end


# --- Ordering ---

"""
    Ordering

Abstract type representing memory ordering semantics.

Subtypes:
- [`Weak`](@ref): Minimal ordering guarantees
- [`Volatile`](@ref): Prevents compiler caching/reordering; no atomicity implied
- [`Relaxed`](@ref): Atomicity only, no synchronization
- [`Acquire`](@ref): Acquire semantics
- [`Release`](@ref): Release semantics
- [`AcqRel`](@ref): Acquire-Release semantics
- [`SeqCst`](@ref): Sequential consistency
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
Prevents the compiler from caching or reordering the operation, ensuring each
access goes directly to memory. Does not imply atomicity or inter-thread
synchronization.
"""
struct Volatile <: Ordering end

"""
    Relaxed <: Ordering

Relaxed memory ordering.
Ensures atomicity of individual operations but provides no synchronization
guarantees. Operations may be reordered freely by hardware and compiler.
"""
struct Relaxed <: Ordering end

"""
    Acquire <: Ordering

Acquire memory ordering.
Ensures that all memory operations after this point see all writes that
happened before a corresponding [`Release`](@ref) operation.
"""
struct Acquire <: Ordering end

"""
    Release <: Ordering

Release memory ordering.
Ensures that all memory operations before this point are visible to threads
that subsequently perform an [`Acquire`](@ref) operation.
"""
struct Release <: Ordering end

"""
    AcqRel <: Ordering

Acquire-Release memory ordering.
Combines both [`Acquire`](@ref) and [`Release`](@ref) semantics. Used for
read-modify-write operations and fences.
"""
struct AcqRel <: Ordering end

"""
    SeqCst <: Ordering

Sequential consistency.
Provides the strongest memory ordering guarantees, establishing a total
order of all sequentially consistent operations across all threads.
"""
struct SeqCst <: Ordering end


# --- Fence ---

"""
    fence(scope::Type{<:Scope}, ordering::Type{<:Ordering})

Low-level memory fence with the specified scope and ordering. Backend-specific
implementations are loaded via device extension modules.

For the validated high-level interface, see [`@fence`](@ref).

# Arguments
- `scope`: Visibility scope (`Workgroup`, `Device`, or `System`)
- `ordering`: Memory ordering (`Acquire`, `Release`, `AcqRel`, or `SeqCst`)

# Notes
- On NVIDIA PTX, `Acquire` and `Release` both map to `acq_rel` — there is no
  separate acquire-only or release-only fence instruction.
- `Weak`, `Volatile`, and `Relaxed` are not meaningful for fences and are
  rejected by [`@fence`](@ref). They are accepted here only for completeness
  of the backend interface.

# Example
```julia
fence(Device,    AcqRel)  # device-wide acquire-release fence
fence(Workgroup, SeqCst)  # workgroup sequentially consistent fence
fence(System,    AcqRel)  # system-wide acquire-release fence
```
"""
function fence end

function atomic_store! end
function atomic_load end