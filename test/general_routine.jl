# Include order matters:
#   1. harness.jl       — defines to_device / from_device / launch / warpsz
#                         used by every test file below.
#   2. backend_hooks.jl — defines HOOKS (IR patterns, capability flags) and the
#                         @capture_ir / @allowscalar macros.
#   3. backend-specific tests (access_fences) under tests/<backend>/ — kept
#                         per-backend because their IR vocabularies diverge.
#   4. cross-backend tests (vectorization, shfl, vectorization_custom_test).
include("harness.jl")
include("backend_hooks.jl")

include("tests/$TEST_BACKEND/access_fences.jl")

include("tests/vectorization_test.jl")
include("tests/shfl.jl")
include("tests/match.jl")
include("tests/vectorization_custom_test.jl")
include("tests/sleep.jl")
