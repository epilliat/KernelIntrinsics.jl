@kernel function test_vload_rebase(a, b, i)
    y = MemoryAccess.vload(a, i, Val(4))
    b[1] = sum(y)
end

a = CuArray{Int32}(1:16)
b = CUDA.zeros(Int32, 4)
test_vload_rebase(CUDABackend())(a, b, 2, ndrange=1) #sum elements (5,6,7,8) because no rebase
@test CUDA.@allowscalar b[1] == 26
buf = IOBuffer()
CUDA.@device_code_ptx io = buf test_vload_rebase(CUDABackend())(a, b, 2; ndrange=1)

asm = String(take!(copy(buf)))
@test occursin("ld.global.v4", asm)


#%%

# take i = 2 to avoid alignment issues in this test

@kernel function test_vload_pattern(a, b, i)
    mod = (i - 1) % 4 + 1
    values = MemoryAccess.vload_pattern(a, i, Val((1, 2, 1)))
    x = sum(values)
    b[1] = sum(x)#sum(values) + sum(u)
end

a = CuArray{Int32}([i for i in (1:16)])
b = CUDA.zeros(Int32, 1)
test_vload_pattern(CUDABackend())(a, b, 2; ndrange=1)

buf = IOBuffer()
CUDA.@device_code_ptx io = buf test_vload_pattern(CUDABackend())(a, b, 2; ndrange=1)

asm = String(take!(copy(buf)))
occursin("ld.global.v4", asm)
occursin("st.global.v4", asm)
occursin("ld.global.v2", asm)
occursin("st.global.v2", asm)

#%%
@kernel function test_vload_multi(a, b, i)
    mod = (i - 1) % 4 + 1
    values = MemoryAccess.vload_multi(a, i, mod, Val(4))
    x = sum(values)
    b[1] = x[1]#sum(values) + sum(u)
end

a = CuArray{Int32}([i for i in (1:100)])
b = CUDA.zeros(Int32, 8)
test_vload_multi(CUDABackend())(a, b, 2; ndrange=1)
b

buf = IOBuffer()
@device_code_ptx test_vload_multi(CUDABackend())(a, b, 2; ndrange=1)
CUDA.@device_code_ptx io = buf test_vload_multi(CUDABackend())(a, b, 2; ndrange=1)

asm = String(take!(copy(buf)))
@test count_substring(asm, "ld.global.v2") == 4 # (1,2,1) x2, (2,2) x2
@test occursin("ld.global.v4", asm) # (4,)

#%%

@kernel function test_vload(a, b, i)

    x = vload(a, i + 1, Val(4), Val(false))
    b[1] = sum(x)
end

a = CuArray{Int32}(1:16)
b = CUDA.zeros(Int32, 4)
test_vload(CUDABackend())(a, b, 2, ndrange=1)
@test CUDA.@allowscalar b[1] == 18
buf = IOBuffer()
CUDA.@device_code_ptx io = buf test_vload(CUDABackend())(a, b, 2; ndrange=1)

asm = String(take!(copy(buf)))
@test count_substring(asm, "ld.global.v2") == 4 # (1,2,1) x2, (2,2) x2
@test occursin("ld.global.v4", asm) # (4,)



################ STORE #########################
#%%
@kernel function test_vstore_rebase(a, b, i)
    values = (Int32(10), Int32(20), Int32(30), Int32(40))
    MemoryAccess.vstore!(b, i, values, Val(true))
end
a = CuArray{Int32}(1:16)
b = CUDA.zeros(Int32, 16)
test_vstore_rebase(CUDABackend())(a, b, 2; ndrange=1)
@test CUDA.@allowscalar (b[5], b[6], b[7], b[8]) == (10, 20, 30, 40)  # rebase: idx=2 -> base = (2-1)*4+1 = 5
buf = IOBuffer()
CUDA.@device_code_ptx io = buf test_vstore_rebase(CUDABackend())(a, b, 2; ndrange=1)
asm = String(take!(copy(buf)))
@test occursin("st.global.v4", asm)

#%%
@kernel function test_vstore_pattern(a, b, i)
    values = (Int32(1), Int32(2), Int32(3), Int32(4))
    MemoryAccess.vstore_pattern!(b, i, values, Val((1, 2, 1)))
end
a = CuArray{Int32}(1:16)
b = CUDA.zeros(Int32, 16)
test_vstore_pattern(CUDABackend())(a, b, 2; ndrange=1)
@test CUDA.@allowscalar (b[2], b[3], b[4], b[5]) == (1, 2, 3, 4)
buf = IOBuffer()
CUDA.@device_code_ptx io = buf test_vstore_pattern(CUDABackend())(a, b, 2; ndrange=1)
asm = String(take!(copy(buf)))
@test occursin("st.global.v2", asm)

#%%
@kernel function test_vstore_multi(a, b, i)
    mod = (i - 1) % 4 + 1
    values = (Int32(10), Int32(20), Int32(30), Int32(40))
    MemoryAccess.vstore_multi!(b, i, mod, values)
end
a = CuArray{Int32}(1:100)
b = CUDA.zeros(Int32, 100)
test_vstore_multi(CUDABackend())(a, b, 2; ndrange=1)
@test CUDA.@allowscalar (b[2], b[3], b[4], b[5]) == (10, 20, 30, 40)
buf = IOBuffer()
CUDA.@device_code_ptx io = buf test_vstore_multi(CUDABackend())(a, b, 2; ndrange=1)
asm = String(take!(copy(buf)))
@test count_substring(asm, "st.global.v2") == 4  # (1,2,1) x2, (2,2) x2
@test occursin("st.global.v4", asm)  # (4,)

#%%
@kernel function test_vstore_norebase(a, b, i)
    values = (Int32(100), Int32(200), Int32(300), Int32(400))
    vstore!(b, i + 1, values, Val(false))
end
a = CuArray{Int32}(1:16)
b = CUDA.zeros(Int32, 16)
test_vstore_norebase(CUDABackend())(a, b, 2; ndrange=1)
@test CUDA.@allowscalar (b[3], b[4], b[5], b[6]) == (100, 200, 300, 400)  # i+1 = 3, no rebase
buf = IOBuffer()
CUDA.@device_code_ptx io = buf test_vstore_norebase(CUDABackend())(a, b, 2; ndrange=1)
asm = String(take!(copy(buf)))
@test count_substring(asm, "st.global.v2") == 4  # (1,2,1) x2, (2,2) x2
@test occursin("st.global.v4", asm)  # (4,)

#%%
# Round-trip test: load then store
@kernel function test_vload_vstore_roundtrip(a, b, i)
    values = MemoryAccess.vload(a, i, Val(4), Val(true))
    MemoryAccess.vstore!(b, i, values, Val(true))
end
a = CuArray{Int32}(1:16)
b = CUDA.zeros(Int32, 16)
test_vload_vstore_roundtrip(CUDABackend())(a, b, 2; ndrange=1)
@test CUDA.@allowscalar (b[5], b[6], b[7], b[8]) == (5, 6, 7, 8)