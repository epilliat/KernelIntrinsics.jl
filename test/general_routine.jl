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

# AMD only: the shuffle must address the PHYSICAL lane, not the rank among active lanes.
# Can't be a cross-backend test — CUDA's `@shfl` lowers to `shfl.sync` with a full mask, so
# calling it from a divergent branch is undefined there by construction.
TEST_BACKEND == "roc" && include("tests/roc/shfl_lane.jl")

include("tests/vectorization_test.jl")
include("tests/shfl.jl")
include("tests/match.jl")
include("tests/vectorization_custom_test.jl")
include("tests/sleep.jl")

# Dynamic workgroup memory is implemented for CUDA and AMD only; Metal has no
# equivalent opt-in region exposed through KernelAbstractions.
TEST_BACKEND in ("cuda", "roc") && include("tests/dynlocalmem.jl")

# MMA (tensor cores) : chemin HW WMMA validé sur CUDA ; le fallback portable est
# exercé via le même harnais. Le path AMD (MFMA) n'est pas encore câblé.
# MMA tourne sur les deux backends hardware. Restreindre à "cuda" laissait le
# chemin MFMA (gfx942) SANS AUCUNE couverture — c'est exactement pourquoi le bug
# d'accumulateur `undef` n'a pu être vu que côté NVIDIA, alors qu'il frappait les
# deux (même mécanisme d'overlay). Les tests se gardent eux-mêmes par
# `MMA.mma_supported(backend, cfg)`, donc une forme absente du hardware est
# sautée, pas échouée.
# Async global→shared/LDS copy (cp.async on NVIDIA, global_load_lds on AMD) with
# a register fallback. CUDA-only for now: on ROC this file SEGFAULTS libLLVM at
# codegen — the failure is in LLVM-AMDGPU/GPUCompiler (compiling the async_copy
# kernel after tests/vectorization_test.jl), not in the test, and AMD hardware
# support is deferred (async_copy_supported(::ROCBackend)=false, see
# ext/AMDGPU/async_copy.jl). Restore roc once the upstream codegen bugs are fixed.
TEST_BACKEND == "cuda" && include("tests/async_copy.jl")

TEST_BACKEND in ("cuda", "roc") && include("tests/mma.jl")
TEST_BACKEND in ("cuda", "roc") && include("tests/mma_fp8.jl")   # fp8 (charge DLFP8Types)
