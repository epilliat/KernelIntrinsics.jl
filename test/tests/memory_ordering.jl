using Test

# Memory ordering test verbosity: Enable with VERBOSE_MEMORY_ORDERING=true
const VERBOSE = get(ENV, "VERBOSE_MEMORY_ORDERING", "false") == "true"

include("litmus/message_passing.jl")
include("litmus/store.jl")
include("litmus/read.jl")
include("litmus/load_buffer.jl")
include("litmus/store_buffer.jl")
include("litmus/write_2plus2w.jl")

@testset "Memory Ordering Litmus Tests" begin
    @testset "Message Passing" begin
        _, _, weak_relaxed, _, _ = run_test_message_passing(
            backend, n_iterations=100, n_pairs=512, RELAXED=true, VERBOSE=VERBOSE)
        _, _, weak_strong, _, _ = run_test_message_passing(
            backend, n_iterations=100, n_pairs=512, RELAXED=false, VERBOSE=VERBOSE)

        @test weak_strong == 0
    end

    @testset "Store" begin
        _, _, _, weak_relaxed, _ = run_test_store(
            backend, n_iterations=100, n_pairs=512, RELAXED=true, VERBOSE=VERBOSE)
        _, _, _, weak_strong, _ = run_test_store(
            backend, n_iterations=100, n_pairs=512, RELAXED=false, VERBOSE=VERBOSE)

        @test weak_strong == 0
    end

    @testset "Read" begin
        _, _, _, weak_relaxed, _ = run_test_read(
            backend, n_iterations=100, n_pairs=512, RELAXED=true, VERBOSE=VERBOSE)
        _, _, _, weak_strong, _ = run_test_read(
            backend, n_iterations=100, n_pairs=512, RELAXED=false, VERBOSE=VERBOSE)

        @test weak_strong == 0
    end

    @testset "Load Buffer" begin
        _, _, _, weak_relaxed, _ = run_test_load_buffer(
            backend, n_iterations=100, n_pairs=512, RELAXED=true, VERBOSE=VERBOSE)
        _, _, _, weak_strong, _ = run_test_load_buffer(
            backend, n_iterations=100, n_pairs=512, RELAXED=false, VERBOSE=VERBOSE)

        @test weak_strong == 0
    end

    @testset "Store Buffer" begin
        _, _, _, weak_relaxed, _ = run_test_store_buffer(
            backend, n_iterations=100, n_pairs=512, RELAXED=true, VERBOSE=VERBOSE)
        _, _, _, weak_strong, _ = run_test_store_buffer(
            backend, n_iterations=100, n_pairs=512, RELAXED=false, VERBOSE=VERBOSE)

        # Store Buffer: Acquire/Release does NOT prevent all weak behaviors
        # (see GPUHarbor: "release/acquire barrier is not enough to disallow this behavior")
        # We only verify that Acquire/Release reduces weak vs Relaxed
        @test weak_strong <= weak_relaxed
    end

    @testset "2+2 Write" begin
        _, _, _, weak_relaxed, _ = run_test_2plus2w(
            backend, n_iterations=100, n_pairs=512, RELAXED=true, VERBOSE=VERBOSE)
        _, _, _, weak_strong, _ = run_test_2plus2w(
            backend, n_iterations=100, n_pairs=512, RELAXED=false, VERBOSE=VERBOSE)

        @test weak_strong == 0
    end
end
