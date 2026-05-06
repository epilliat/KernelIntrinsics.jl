# Per-backend traits and macros, selected at include time from TEST_BACKEND.
#
# Adding a new backend means:
#   1. Adding an arm to the `HOOKS = if … elseif … end` block below with that
#      backend's IR patterns and capability flags.
#   2. Adding the corresponding `elseif TEST_BACKEND == …` arm to @capture_ir
#      and @allowscalar so they expand to that backend's macros.
#
# IR pattern semantics: each pattern is a `Vector{String}` of substrings that
# must all appear on a *single* line of the captured IR. Single-element vectors
# reduce to a plain "this substring appears somewhere" assertion. Multi-element
# vectors handle cases like Metal's `store <4 x i32> … ptr addrspace(1)` where
# two substrings must coexist on the same line. An empty vector means "this
# backend has no clean IR assertion for that primitive — skip the check."

const HOOKS = if TEST_BACKEND == "cuda"
    (
        ir = (
            vload_v4   = ["ld.global.v4"],
            vload_v2   = ["ld.global.v2"],
            vstore_v4  = ["st.global.v4"],
            vstore_v2  = ["st.global.v2"],
        ),
        supported = (
            float64       = true,
            system_scope  = true,
            vote_ballot   = true,
        ),
    )
elseif TEST_BACKEND == "roc"
    (
        ir = (
            vload_v4   = ["global_load_dwordx4"],
            vload_v2   = ["global_load_dwordx2"],
            vstore_v4  = ["global_store_dwordx4"],
            vstore_v2  = ["global_store_dwordx2"],
        ),
        supported = (
            float64       = true,
            system_scope  = false,   # AMDGPU emits no system-scope fence
            vote_ballot   = false,   # ballot returns UInt64; @vote not implemented in ext
        ),
    )
elseif TEST_BACKEND == "metal"
    (
        ir = (
            vload_v4   = ["load <4 x i32>, ptr addrspace(1)"],
            vload_v2   = ["load <2 x i32>, ptr addrspace(1)"],
            vstore_v4  = ["store <4 x i32>", "addrspace(1)"],
            vstore_v2  = ["store <2 x i32>", "addrspace(1)"],
        ),
        supported = (
            float64       = false,   # MtlArray cannot allocate Float64
            system_scope  = true,
            vote_ballot   = true,
        ),
    )
else
    error("Unknown TEST_BACKEND for backend_hooks: $TEST_BACKEND")
end


# Run `expr` while capturing the active backend's device-code emission to a String.
#
# Usage:
#   asm = @capture_ir my_kernel(backend)(args; ndrange=N)
#   assert_ir(HOOKS.ir.vload_v4, asm)
macro capture_ir(expr)
    # `esc(quote … end)` keeps `io` as the literal symbol the inner
    # `@device_code_*` macro expects (default macro hygiene rewrites `io =`
    # into a gensym'd name and the inner macro then can't find it).
    if TEST_BACKEND == "cuda"
        esc(:(let _buf = IOBuffer()
            CUDA.@device_code_ptx io = _buf $expr
            String(take!(_buf))
        end))
    elseif TEST_BACKEND == "roc"
        esc(:(let _buf = IOBuffer()
            AMDGPU.@device_code_gcn io = _buf $expr
            String(take!(_buf))
        end))
    elseif TEST_BACKEND == "metal"
        esc(:(let _buf = IOBuffer()
            Metal.@device_code_llvm io = _buf $expr
            String(take!(_buf))
        end))
    else
        error("Unknown TEST_BACKEND for @capture_ir: $TEST_BACKEND")
    end
end


# Read/write a scalar element of a device array on the active backend.
#
# Usage:
#   @test (@allowscalar b[1]) == expected
macro allowscalar(expr)
    if TEST_BACKEND == "cuda"
        :(CUDA.@allowscalar $(esc(expr)))
    elseif TEST_BACKEND == "roc"
        :(AMDGPU.@allowscalar $(esc(expr)))
    elseif TEST_BACKEND == "metal"
        :(Metal.@allowscalar $(esc(expr)))
    else
        error("Unknown TEST_BACKEND for @allowscalar: $TEST_BACKEND")
    end
end


# Assert every substring in `patterns` appears on the same line somewhere in
# `asm`. Empty `patterns` skips the assertion (used for backends with no clean
# IR substring for that primitive).
function assert_ir(patterns::AbstractVector{<:AbstractString}, asm::AbstractString)
    isempty(patterns) && return
    @test any(all(occursin(p, line) for p in patterns) for line in eachsplit(asm, '\n'))
end
