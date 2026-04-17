# Litmus Test: 2+2W (Write)
#
# Tests if two stores in two threads can both be re-ordered.
#
# Pattern:
#   Workgroup 0 Thread 0                Workgroup 1 Thread 0
#       0.1: atomicStore(x, 2)              1.1: atomicStore(y, 2)
#       0.2: atomicStore(y, 1)              1.2: atomicStore(x, 1)
#
# Based on https://github.com/reeselevine/webgpu-litmus/blob/main/shaders/2+2/2+2-write.wgsl

using Adapt
using KernelAbstractions
using KernelIntrinsics
using Test

@kernel inbounds=true function test_write_2plus2w(
    test_locations::AbstractArray{T},
    n_pairs::T,
    ::Val{RELAXED}=Val(true),
    ::Val{wgXSize}=Val(256),
    ::Val{test_wg}=Val(2),
    ::Val{perm1}=Val(419),
    ::Val{perm2}=Val(1031),
    ::Val{stride}=Val(1)
) where {T, RELAXED, wgXSize, test_wg, perm1, perm2, stride}

    i = @index(Global, Linear)

    if i <= n_pairs
        local_invocation_id = (i - 1) % wgXSize
        shuffled_workgroup = (i - 1) ÷ wgXSize
        total_ids = wgXSize * test_wg
        id_0 = shuffled_workgroup * wgXSize + local_invocation_id
        new_workgroup = (shuffled_workgroup + 1 + (local_invocation_id % (test_wg - 1))) % test_wg
        id_1 = new_workgroup * wgXSize + ((local_invocation_id * perm1) % wgXSize)

        # 2+2W pattern: Store, Store, Store, Store (no loads)
        x_0 = (id_0) * stride * 2
        y_0 = ((id_0 * perm2) % total_ids) * stride * 2 + 1
        y_1 = ((id_1 * perm2) % total_ids) * stride * 2 + 1
        x_1 = (id_1) * stride * 2

        if RELAXED
            @access Relaxed test_locations[x_0 + 1] = T(2)
            @access Relaxed test_locations[y_0 + 1] = T(1)
            @access Relaxed test_locations[y_1 + 1] = T(2)
            @access Relaxed test_locations[x_1 + 1] = T(1)
        else
            @access Release test_locations[x_0 + 1] = T(2)
            @access Release test_locations[y_0 + 1] = T(1)
            @access Release test_locations[y_1 + 1] = T(2)
            @access Release test_locations[x_1 + 1] = T(1)
        end
    end
end

function run_test_2plus2w(backend; n_iterations::Int=100, n_pairs::Int=512, RELAXED=true, VERBOSE=true)
    if VERBOSE
        println("\n" * "-" ^ 60)
        if RELAXED
            println("Litmus Test: 2+2W Write (@access Relaxed)\n")
        else
            println("Litmus Test: 2+2W Write (@access Release)\n")
        end
    end

    test_locations = adapt(backend, zeros(Int32, 2048))

    x1_y2 = 0
    x2_y1 = 0
    x1_y1 = 0
    x2_y2 = 0

    for iter in 1:n_iterations
        fill!(test_locations, Int32(0))

        test_write_2plus2w(backend)(
            test_locations,
            Int32(n_pairs), Val(RELAXED);
            ndrange=n_pairs, workgroupsize=256
        )
        KernelAbstractions.synchronize(backend)

        test_cpu = Array(test_locations)

        for i in 1:n_pairs
            # Calculate id_0 to find x_0 and y_0 addresses
            workgroupXSize = 256
            testing_workgroups = 2
            permute_second = 1031
            local_invocation_id = (i - 1) % workgroupXSize
            shuffled_workgroup = (i - 1) ÷ workgroupXSize
            total_ids = workgroupXSize * testing_workgroups
            id_0 = shuffled_workgroup * workgroupXSize + local_invocation_id

            x_0_addr = id_0 * 1 * 2 + 1
            y_0_addr = ((id_0 * permute_second) % total_ids) * 1 * 2 + 1 + 1

            # Read the actual final values at x_0 and y_0 (GPUHarbor's *x and *y)
            mem_x_0 = test_cpu[x_0_addr]
            mem_y_0 = test_cpu[y_0_addr]

            # Categorize based on (mem_x_0, mem_y_0) pairs (GPUHarbor categories)
            if mem_x_0 == Int32(1) && mem_y_0 == Int32(2)
                x1_y2 += 1
            elseif mem_x_0 == Int32(2) && mem_y_0 == Int32(1)
                x2_y1 += 1
            elseif mem_x_0 == Int32(1) && mem_y_0 == Int32(1)
                x1_y1 += 1
            else
                x2_y2 += 1
            end
        end
    end

    total = x1_y2 + x2_y1 + x1_y1 + x2_y2

    if VERBOSE
        println("  ╔═══════════════════════════════════════════════════╗")
        println("  ║ RESULTS ($total iterations)                        ║")
        println("  ╠═══════════════════════════════════════════════════╣")
        println("  ║ x=1, y=2: (sequential)     $(lpad(x1_y2, 10)) ($(lpad(round(100*x1_y2/total, digits=2), 5))%)    ║")
        println("  ║ x=2, y=1: (sequential)     $(lpad(x2_y1, 10)) ($(lpad(round(100*x2_y1/total, digits=2), 5))%)    ║")
        println("  ║ x=1, y=1: (interleaved)    $(lpad(x1_y1, 10)) ($(lpad(round(100*x1_y1/total, digits=2), 5))%)    ║")
        println("  ║ x=2, y=2: (weak)           $(lpad(x2_y2, 10)) ($(lpad(round(100*x2_y2/total, digits=2), 5))%)    ║")
        println("  ╚═══════════════════════════════════════════════════╝")
    end
    return (x1_y2, x2_y1, x1_y1, x2_y2, total)
end
