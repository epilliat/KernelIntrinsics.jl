# Litmus Test: Store
#
# Tests if two stores in one thread can be re-ordered according to a
# store and a load on a second thread.
#
# Pattern:
#   Workgroup 0 Thread 0                Workgroup 1 Thread 0
#       0.1: atomicStore(x, 2)              1.1: let r0 = atomicLoad(y)
#       0.2: atomicStore(y, 1)              1.2: atomicStore(x, 1)
#
# Based on https://github.com/reeselevine/webgpu-litmus/blob/main/shaders/store/store.wgsl

using Adapt
using KernelAbstractions
using KernelIntrinsics
using Test

@kernel inbounds=true function test_store(
    test_locations::AbstractArray{T},
    results_r0::AbstractArray{T},
    n_pairs::T,
    ::Val{RELAXED}=Val(true),
    ::Val{wgXSize}=Val(256),    # workgroupXSize
    ::Val{test_wg}=Val(2),      # testing_workgroups
    ::Val{perm1}=Val(419),      # permute_first
    ::Val{perm2}=Val(1031),     # permute_second
    ::Val{stride}=Val(1)        # mem_stride
) where {T, RELAXED, wgXSize, test_wg, perm1, perm2, stride}

    i = @index(Global, Linear)

    if i <= n_pairs
        local_invocation_id = (i - 1) % wgXSize
        shuffled_workgroup = (i - 1) ÷ wgXSize
        total_ids = wgXSize * test_wg
        id_0 = shuffled_workgroup * wgXSize + local_invocation_id
        new_workgroup = (shuffled_workgroup + 1 + (local_invocation_id % (test_wg - 1))) % test_wg
        id_1 = new_workgroup * wgXSize + ((local_invocation_id * perm1) % wgXSize)

        # Store pattern: Store, Store, Load, Store
        x_0 = (id_0) * stride * 2
        y_0 = ((id_0 * perm2) % total_ids) * stride * 2 + 1
        y_1 = ((id_1 * perm2) % total_ids) * stride * 2 + 1
        x_1 = (id_1) * stride * 2

        if RELAXED
            @access Relaxed test_locations[x_0 + 1] = T(2)
            @access Relaxed test_locations[y_0 + 1] = T(1)
            r0 = @access Relaxed test_locations[y_1 + 1]
            @access Relaxed test_locations[x_1 + 1] = T(1)
        else
            @access Release test_locations[x_0 + 1] = T(2)
            @access Release test_locations[y_0 + 1] = T(1)
            r0 = @access Acquire test_locations[y_1 + 1]
            @access Release test_locations[x_1 + 1] = T(1)
        end

        results_r0[id_1 + 1] = r0
    end
end

function run_test_store(backend; n_iterations::Int=100, n_pairs::Int=512, RELAXED=true, VERBOSE=false)
    if VERBOSE
        println("\n" * "-" ^ 60)
        if RELAXED
            println("Litmus Test: Store (@access Relaxed)\n")
        else
            println("Litmus Test: Store (@access Acquire/Release)\n")
        end
    end

    test_locations = adapt(backend, zeros(Int32, 2048))
    results_r0 = adapt(backend, zeros(Int32, n_pairs))

    r0_1_x_1 = 0
    r0_0_x_2 = 0
    r0_0_x_1 = 0
    r0_1_x_2 = 0

    for iter in 1:n_iterations
        fill!(test_locations, Int32(0))
        fill!(results_r0, Int32(0))

        test_store(backend)(
            test_locations, results_r0,
            Int32(n_pairs), Val(RELAXED);
            ndrange=n_pairs, workgroupsize=256
        )
        KernelAbstractions.synchronize(backend)

        test_cpu = Array(test_locations)

        for i in 1:n_pairs
            # Calculate id_0 to read results (WGSL reads r0 from id_0)
            workgroupXSize = 256
            testing_workgroups = 2
            local_invocation_id = (i - 1) % workgroupXSize
            shuffled_workgroup = (i - 1) ÷ workgroupXSize
            id_0 = shuffled_workgroup * workgroupXSize + local_invocation_id

            r0 = Array(results_r0)[id_0 + 1]

            # Calculate x_0 address to read final value (GPUHarbor reads from x_0, not x_1!)
            x_0_addr = id_0 * 1 * 2 + 1

            # Read the actual final value at x_0 (this is what GPUHarbor calls *x)
            x_final = test_cpu[x_0_addr]

            # Categorize based on both r0 and final x_0 value (GPUHarbor categories)
            if r0 == Int32(1) && x_final == Int32(1)
                r0_1_x_1 += 1
            elseif r0 == Int32(0) && x_final == Int32(2)
                r0_0_x_2 += 1
            elseif r0 == Int32(0) && x_final == Int32(1)
                r0_0_x_1 += 1
            elseif r0 == Int32(1) && x_final == Int32(2)
                r0_1_x_2 += 1
            end
        end
    end

    total = r0_1_x_1 + r0_0_x_2 + r0_0_x_1 + r0_1_x_2

    if VERBOSE
        println("  ╔═══════════════════════════════════════════════════╗")
        println("  ║ RESULTS ($total total tests)                       ║")
        println("  ╠═══════════════════════════════════════════════════╣")
        println("  ║ r0=1, x=1: (sequential)   $(lpad(r0_1_x_1, 10)) ($(lpad(round(100*r0_1_x_1/total, digits=2), 5))%)     ║")
        println("  ║ r0=0, x=2: (sequential)   $(lpad(r0_0_x_2, 10)) ($(lpad(round(100*r0_0_x_2/total, digits=2), 5))%)     ║")
        println("  ║ r0=0, x=1: (interleaved)  $(lpad(r0_0_x_1, 10)) ($(lpad(round(100*r0_0_x_1/total, digits=2), 5))%)     ║")
        println("  ║ r0=1, x=2: (WEAK)         $(lpad(r0_1_x_2, 10)) ($(lpad(round(100*r0_1_x_2/total, digits=2), 5))%)     ║")
        println("  ╚═══════════════════════════════════════════════════╝")
    end
    return (r0_1_x_1, r0_0_x_2, r0_0_x_1, r0_1_x_2, total)
end
