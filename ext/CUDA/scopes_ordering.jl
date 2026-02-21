import KernelIntrinsics: Scope, Ordering
import KernelIntrinsics: Workgroup, Device, System
import KernelIntrinsics: Acquire, Release, AcqRel, SeqCst, Weak, Volatile, Relaxed

import KernelIntrinsics: fence, atomic_load, atomic_store!

"""
Mapping from KernelIntrinsics fence scopes to PTX scope identifiers.
"""
const SCOPE_TO_PTX = Dict{Type{<:Scope},String}(
    Workgroup => "cta",
    Device => "gpu",
    System => "sys"
)

"""
Mapping from KernelIntrinsics fence orderings to PTX memory ordering semantics.
Note: PTX does not have separate acquire and release fence instructions,
so both Acquire and Release map to acq_rel.
"""
const ORDER_TO_PTX = Dict{Type{<:Ordering},String}(
    Acquire => "acq_rel",  # PTX limitation: no separate acquire
    Release => "acq_rel",  # PTX limitation: no separate release
    AcqRel => "acq_rel",
    SeqCst => "sc"
)


for ScopeType in [Workgroup, Device, System]
    for OrderType in [Acquire, Release, AcqRel, SeqCst]
        scope_str = SCOPE_TO_PTX[ScopeType]
        order_str = ORDER_TO_PTX[OrderType]
        ptx_instr = "fence.$order_str.$scope_str;"

        @eval begin
            """
                fence(::Type{$($ScopeType)}, ::Type{$($OrderType)})

            CUDA implementation: Emits PTX instruction `$($ptx_instr)`.
            """
            CUDA.@device_override @inline function fence(::Type{$ScopeType}, ::Type{$OrderType})
                LLVM.Interop.@asmcall($ptx_instr, "", true, Nothing, Tuple{})
            end
        end
    end
end

# ============================================================================
# Atomic Load Operations
# ============================================================================

"""
Mapping from load orderings to PTX ordering strings.
PTX supports: weak, relaxed, acquire, volatile for loads.
Note: weak and volatile do not use scope qualifiers.
"""
const LOAD_ORDER_TO_PTX = Dict{Type{<:Ordering},String}(
    Weak => "weak",
    Relaxed => "relaxed",
    Volatile => "volatile",
    Acquire => "acquire"
)

"""
Mapping from Julia types to PTX type strings and constraint letters for loads.
Format: JuliaType => (ptx_type, constraint_letter)
"""
const TYPE_TO_PTX = Dict{DataType,Tuple{String,String}}(
    # 8-bit types
    Int8 => ("s8", "r"),   # signed 8-bit (promoted to 32-bit register)
    UInt8 => ("u8", "r"),   # unsigned 8-bit (promoted to 32-bit register)

    # 16-bit types
    Int16 => ("s16", "r"),  # signed 16-bit (promoted to 32-bit register)
    UInt16 => ("u16", "r"),  # unsigned 16-bit (promoted to 32-bit register)

    # 32-bit types
    Int32 => ("s32", "r"),  # signed 32-bit
    UInt32 => ("u32", "r"),  # unsigned 32-bit
    Float32 => ("f32", "f"),  # 32-bit float

    # 64-bit types
    Int64 => ("s64", "l"),  # signed 64-bit
    UInt64 => ("u64", "l"),  # unsigned 64-bit
    Float64 => ("f64", "d")   # 64-bit float
)

# Generate atomic load functions for different types and scopes
for ScopeType in [Workgroup, Device, System]
    scope_str = SCOPE_TO_PTX[ScopeType]

    for OrderType in [Weak, Relaxed, Volatile, Acquire]
        order_str = LOAD_ORDER_TO_PTX[OrderType]

        for (T, (ptx_type, constraint)) in TYPE_TO_PTX
            # Conditional: weak and volatile do not use scope
            ptx_str = if OrderType in [Weak, Volatile]
                "ld.$order_str.$ptx_type \$0, [\$1];"
            else
                "ld.$order_str.$scope_str.$ptx_type \$0, [\$1];"
            end
            constraint_str = "=$constraint,l,~{memory}"

            @eval begin
                """
                    atomic_load(data::CuDeviceArray{$($T),N,AS.Global}, index::Integer, ::Type{$($ScopeType)}, ::Type{$($OrderType)}) where N

                CUDA implementation: Emits PTX `$($ptx_str)`
                Loads element at `index` from `data` with specified memory ordering.
                """
                CUDA.@device_override @inline function atomic_load(
                    data::CuDeviceArray{$T,N,AS.Global},
                    index::Integer,
                    ::Type{$ScopeType},
                    ::Type{$OrderType}
                ) where {N}
                    ptr = data.ptr + (index - 1) * sizeof($T)
                    LLVM.Interop.@asmcall(
                        $ptx_str,
                        $constraint_str,
                        true,
                        $T,
                        Tuple{LLVMPtr{$T,AS.Global}},
                        ptr
                    )
                end
            end
        end
    end
end

# ============================================================================
# Atomic Store Operations
# ============================================================================

"""
Mapping from store orderings to PTX ordering strings.
PTX supports: weak, relaxed, release, volatile for stores.
Note: weak and volatile do not use scope qualifiers.
"""
const STORE_ORDER_TO_PTX = Dict{Type{<:Ordering},String}(
    Weak => "weak",
    Relaxed => "relaxed",
    Volatile => "volatile",
    Release => "release"
)


# Generate atomic store functions for different types and scopes
for ScopeType in [Workgroup, Device, System]
    scope_str = SCOPE_TO_PTX[ScopeType]

    for OrderType in [Weak, Relaxed, Volatile, Release]
        order_str = STORE_ORDER_TO_PTX[OrderType]

        for (T, (ptx_type, constraint)) in TYPE_TO_PTX
            # Conditional: weak and volatile do not use scope
            ptx_str = if OrderType in [Weak, Volatile]
                "st.$order_str.$ptx_type [\$0], \$1;"
            else
                "st.$order_str.$scope_str.$ptx_type [\$0], \$1;"
            end
            constraint_str = "l,$constraint,~{memory}"

            @eval begin
                """
                    atomic_store!(data::CuDeviceArray{$($T),N,AS.Global}, index::Integer, val::$($T), ::Type{$($ScopeType)}, ::Type{$($OrderType)}) where N

                CUDA implementation: Emits PTX `$($ptx_str)`
                Stores `val` to element at `index` in `data` with specified memory ordering.
                """
                CUDA.@device_override @inline function atomic_store!(
                    data::CuDeviceArray{$T,N,AS.Global},
                    index::Integer,
                    val::$T,
                    ::Type{$ScopeType},
                    ::Type{$OrderType}
                ) where {N}
                    ptr = data.ptr + (index - 1) * sizeof($T)
                    LLVM.Interop.@asmcall(
                        $ptx_str,
                        $constraint_str,
                        true,
                        Nothing,
                        Tuple{LLVMPtr{$T,AS.Global},$T},
                        ptr,
                        val
                    )
                end
            end
        end
    end
end


