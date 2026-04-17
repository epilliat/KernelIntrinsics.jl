# Litmus Test: Load Buffer (LB)
#
# Tests if loads can be buffered and re-ordered on different threads.
#
# Pattern: 
#   Workgroup 0 Thread 0                Workgroup 1 Thread 0
#       0.1: let r0 = atomicLoad(y)         1.1: let r1 = atomicLoad(x)
#       0.2: atomicStore(x, 1)              1.2: atomicStore(y, 1)
#
# Based on https://github.com/reeselevine/webgpu-litmus/blob/main/shaders/lb/load-buffer.wgsl
#
# (See end of file for complete example)

using Adapt
using KernelAbstractions
using KernelIntrinsics
using Test

@kernel inbounds=true function test_load_buffer(
    test_locations::AbstractArray{T},
    results_r0::AbstractArray{T},
    results_r1::AbstractArray{T},
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

        # Load Buffer pattern: Load, Store, Load, Store
        y_0 = (id_0) * stride * 2
        x_0 = ((id_0 * perm2) % total_ids) * stride * 2 + 1  # location_offset
        x_1 = ((id_1 * perm2) % total_ids) * stride * 2 + 1  # location_offset
        y_1 = (id_1) * stride * 2

        if RELAXED
            r0 = @access Relaxed test_locations[y_0 + 1]
            @access Relaxed test_locations[x_0 + 1] = T(1)
            r1 = @access Relaxed test_locations[x_1 + 1]
            @access Relaxed test_locations[y_1 + 1] = T(1)
        else
            r0 = @access Acquire test_locations[y_0 + 1]
            @access Release test_locations[x_0 + 1] = T(1)
            r1 = @access Acquire test_locations[x_1 + 1]
            @access Release test_locations[y_1 + 1] = T(1)
        end

        results_r1[id_1 + 1] = r1
        results_r0[id_0 + 1] = r0
    end
end

function run_test_load_buffer(backend; n_iterations::Int=100, n_pairs::Int=512, RELAXED=true, VERBOSE=false)
    if VERBOSE
        println("\n" * "-" ^ 60)
        if RELAXED
            println("Litmus Test: Load Buffer (@access Relaxed)\n")
        else
            println("Litmus Test: Load Buffer (@access Acquire/Release)\n")
        end
    end

    test_locations = adapt(backend, zeros(Int32, 2048))
    results_r0 = adapt(backend, zeros(Int32, n_pairs))
    results_r1 = adapt(backend, zeros(Int32, n_pairs))

    total_seq0 = 0
    total_seq1 = 0
    total_interleaved = 0
    total_weak = 0

    for iter in 1:n_iterations
        fill!(test_locations, Int32(0))
        fill!(results_r0, Int32(0))
        fill!(results_r1, Int32(0))

        test_load_buffer(backend)(
            test_locations, results_r0, results_r1,
            Int32(n_pairs), Val(RELAXED);
            ndrange=n_pairs, workgroupsize=256
        )
        KernelAbstractions.synchronize(backend)

        for i in 1:n_pairs
            # Calculate id_0 to read results (WGSL reads both r0 and r1 from id_0)
            workgroupXSize = 256
            testing_workgroups = 2
            local_invocation_id = (i - 1) % workgroupXSize
            shuffled_workgroup = (i - 1) ÷ workgroupXSize
            id_0 = shuffled_workgroup * workgroupXSize + local_invocation_id

            r0 = Array(results_r0)[id_0 + 1]
            r1 = Array(results_r1)[id_0 + 1]

            # Load Buffer outcome classification (GPUHarbor categories)
            if r0 == Int32(1) && r1 == Int32(0)
                total_seq0 += 1
            elseif r0 == Int32(0) && r1 == Int32(1)
                total_seq1 += 1
            elseif r0 == Int32(0) && r1 == Int32(0)
                total_interleaved += 1
            elseif r0 == Int32(1) && r1 == Int32(1)
                total_weak += 1
            end
        end
    end

    total = total_seq0 + total_seq1 + total_interleaved + total_weak

    if VERBOSE
        println("  ╔═══════════════════════════════════════════════════╗")
        println("  ║ RESULTS ($total total tests)                       ║")
        println("  ╠═══════════════════════════════════════════════════╣")
        println("  ║ r0=1, r1=0: (seq0)         $(lpad(total_seq0, 10)) ($(lpad(round(100*total_seq0/total, digits=2), 5))%)    ║")
        println("  ║ r0=0, r1=1: (seq1)         $(lpad(total_seq1, 10)) ($(lpad(round(100*total_seq1/total, digits=2), 5))%)    ║")
        println("  ║ r0=0, r1=0: (interleaved)  $(lpad(total_interleaved, 10)) ($(lpad(round(100*total_interleaved/total, digits=2), 5))%)    ║")
        println("  ║ r0=1, r1=1: (WEAK/LB)      $(lpad(total_weak, 10)) ($(lpad(round(100*total_weak/total, digits=2), 5))%)    ║")
        println("  ╚═══════════════════════════════════════════════════╝")
    end
    return (total_seq0, total_seq1, total_interleaved, total_weak, total)
end

"""
As illustration we will represent the different memory operations on threads 0 and 256.

Here are the different memory locations calculated as per the kernel.

┌─────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  i  │ id_0 │ id_1 │ y_0  │ x_0  │ x_1  │ y_1  │
├─────┼──────┼──────┼──────┼──────┼──────┼──────┤
│  1  │   0  │ 256  │   0  │   1  │ 513  │ 512  │
├─────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ 257 │  256 │   0  │ 512  │ 513  │   1  │   0  │
└─────┴──────┴──────┴──────┴──────┴──────┴──────┘

We reprensent the different addresses using boxes using :
    Memory addresses in test_locations :  0     1    512   513
    Memory addresses in results_r0     :  0     256
    Memory addresses in results_r1     :  0     256

We will illustrate the case where all operations on thread 0 happen before thread 256 and are ordered

Step 1: Thread (i=1) loads r0 from location y_0=0
┌─────┬─────┬─────┬─────┐
│  0  │  0  │  0  │  0  │
└─────┴─────┴─────┴─────┘
   ↓
  r_0

Step 2: Thread (i=1) stores 1 to location x_0=1
┌─────┬─────┬─────┬─────┐
│  0  │  1  │  0  │  0  │
└─────┴─────┴─────┴─────┘
         ↑
         1

Step 3: Thread (i=1) loads r1 from location x_1=513
┌─────┬─────┬─────┬─────┐
│  0  │  1  │  0  │  0  │
└─────┴─────┴─────┴─────┘
                     ↓
                    r_1

Step 4: Thread (i=1) stores 1 to location y_1=512
┌─────┬─────┬─────┬─────┐
│  0  │  1  │  1  │  0  │
└─────┴─────┴─────┴─────┘
               ↑
               1

Step 5: Thread (i=1) stores r1 to location id_1=256 of results_r1
┌─────┬─────┐
│  0  │  0  │
└─────┴─────┘
         ↑
        r_1

Step 6: Thread (i=1) stores r0 to location id_0=0 of results_r0
┌─────┬─────┐
│  0  │  0  │
└─────┴─────┘
   ↑
  r_0

Step 7: Thread (i=257) loads r0 from location y_0=512
┌─────┬─────┬─────┬─────┐
│  0  │  1  │  1  │  0  │
└─────┴─────┴─────┴─────┘
               ↓
              r_0

Step 8: Thread (i=257) stores 1 to location x_0=513
┌─────┬─────┬─────┬─────┐
│  0  │  1  │  1  │  1  │
└─────┴─────┴─────┴─────┘
                     ↑
                     1

Step 9: Thread (i=257) loads r1 from location x_1=1
┌─────┬─────┬─────┬─────┐
│  0  │  1  │  1  │  1  │
└─────┴─────┴─────┴─────┘
         ↓
        r_1
    
Step 10: Thread (i=257) stores 1 to location y_1=0
┌─────┬─────┬─────┬─────┐
│  1  │  1  │  1  │  1  │
└─────┴─────┴─────┴─────┘
   ↑
   1
             
Step 11: Thread (i=257) stores r1 to location id_1=0 of results_r1
┌─────┬─────┐
│  1  │  0  │
└─────┴─────┘
   ↑
  r_1

Step 12: Thread (i=257) stores r0 to location id_0=256 of results_r0
┌─────┬─────┐
│  0  │  1  │
└─────┴─────┘
         ↑
        r_0

Step 13: results_r0 and results_r1 are read at location id_0=0 for categorization
    r_0_test = 0
    r_1_test = 1

This is categorized as 'sequential'


Here, if Step 10 happened before Step 1 (reordering) then 
    r_0_test = 1
    r_1_test = 1

This is categorized as 'weak'.
"""
