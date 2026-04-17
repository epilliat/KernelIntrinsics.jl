# Litmus Test: Message Passing (MP)
#
# Tests if two stores in one thread can be re-ordered according to loads on a second thread.
#
# Pattern: 
#   Workgroup 0 Thread 0                Workgroup 1 Thread 0
#       0.1: atomicStore(x, 1)              1.1: let r0 = atomicLoad(y)
#       0.2: atomicStore(y, 1)              1.2: let r1 = atomicLoad(x)
#
# Based on https://github.com/reeselevine/webgpu-litmus/blob/main/shaders/mp/message-passing.wgsl

using Adapt
using KernelAbstractions
using KernelIntrinsics
using Test

@kernel inbounds=true function test_message_passing(
    test_locations::AbstractArray{T},
    results_r0::AbstractArray{T},
    results_r1::AbstractArray{T},
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

        # Message Passing pattern: Store, Store, Load, Load
        x_0 = (id_0) * stride * 2
        y_0 = ((id_0 * perm2) % total_ids) * stride * 2 + 1  # location_offset
        x_1 = (id_1) * stride * 2
        y_1 = ((id_1 * perm2) % total_ids) * stride * 2 + 1  # location_offset

        if RELAXED
            @access Relaxed test_locations[x_0 + 1] = T(1)
            @access Relaxed test_locations[y_0 + 1] = T(1)
            r0 = @access Relaxed test_locations[y_1 + 1]
            r1 = @access Relaxed test_locations[x_1 + 1]
        else
            @access Release test_locations[x_0 + 1] = T(1)
            @access Release test_locations[y_0 + 1] = T(1)
            r0 = @access Acquire test_locations[y_1 + 1]
            r1 = @access Acquire test_locations[x_1 + 1]
        end

        results_r1[id_1 + 1] = r1
        results_r0[id_1 + 1] = r0
    end
end

function run_test_message_passing(backend; n_iterations::Int=100, n_pairs::Int=512, RELAXED=true, VERBOSE=false)
    if VERBOSE
        println("\n" * "-" ^ 60)
        if RELAXED
            println("Litmus Test: Message Passing (@access Relaxed)\n")
        else
            println("Litmus Test: Message Passing (@access Acquire/Release)\n")
        end
    end

    test_locations = adapt(backend, zeros(Int32, 2048))
    results_r0 = adapt(backend, zeros(Int32, n_pairs))
    results_r1 = adapt(backend, zeros(Int32, n_pairs))

    total_r0_0_r1_0 = 0
    total_r0_0_r1_1 = 0
    total_r0_1_r1_0 = 0
    total_r0_1_r1_1 = 0

    for iter in 1:n_iterations
        fill!(test_locations, Int32(0))
        fill!(results_r0, Int32(0))
        fill!(results_r1, Int32(0))

        test_message_passing(backend)(
            test_locations, results_r0, results_r1,
            Int32(n_pairs), Val(RELAXED);
            ndrange=n_pairs, workgroupsize=256
        )
        KernelAbstractions.synchronize(backend)

        for i in 1:n_pairs
            # Calculate id_0 to read results (WGSL reads from id_0)
            workgroupXSize = 256
            testing_workgroups = 2
            local_invocation_id = (i - 1) % workgroupXSize
            shuffled_workgroup = (i - 1) ÷ workgroupXSize
            id_0 = shuffled_workgroup * workgroupXSize + local_invocation_id

            r0 = Array(results_r0)[id_0 + 1]
            r1 = Array(results_r1)[id_0 + 1]

            # Message Passing outcome classification (GPUHarbor categories)
            if r0 == Int32(0) && r1 == Int32(0)
                total_r0_0_r1_0 += 1
            elseif r0 == Int32(0) && r1 == Int32(1)
                total_r0_0_r1_1 += 1
            elseif r0 == Int32(1) && r1 == Int32(0)
                total_r0_1_r1_0 += 1
            elseif r0 == Int32(1) && r1 == Int32(1)
                total_r0_1_r1_1 += 1
            end
        end
    end

    total = total_r0_0_r1_0 + total_r0_0_r1_1 + total_r0_1_r1_0 + total_r0_1_r1_1

    if VERBOSE
        println("  ╔═══════════════════════════════════════════════════╗")
        println("  ║ RESULTS ($total total tests)                       ║")
        println("  ╠═══════════════════════════════════════════════════╣")
        println("  ║ r0=0, r1=0: (sequential)    $(lpad(total_r0_0_r1_0, 10)) ($(lpad(round(100*total_r0_0_r1_0/total, digits=2), 5))%)   ║")
        println("  ║ r0=1, r1=1: (sequential)    $(lpad(total_r0_1_r1_1, 10)) ($(lpad(round(100*total_r0_1_r1_1/total, digits=2), 5))%)   ║")
        println("  ║ r0=0, r1=1: (interleaved)   $(lpad(total_r0_0_r1_1, 10)) ($(lpad(round(100*total_r0_0_r1_1/total, digits=2), 5))%)   ║")
        println("  ║ r0=1, r1=0: (WEAK)          $(lpad(total_r0_1_r1_0, 10)) ($(lpad(round(100*total_r0_1_r1_0/total, digits=2), 5))%)   ║")
        println("  ╚═══════════════════════════════════════════════════╝")
    end
    return (total_r0_0_r1_0, total_r0_0_r1_1, total_r0_1_r1_0, total_r0_1_r1_1, total)
end