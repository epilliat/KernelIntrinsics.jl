# Extension MULTI-DÉCLENCHEUR CUDA + DLFP8Types : chemin hardware fp8/bf8 sur les
# tensor cores NVIDIA (Ada sm_89 / Hopper sm_90). CUDA.jl n'expose PAS fp8 (son WMMA
# haut-niveau ne wrappe que `wmma.mma.sync`, sans fp8) : on émet donc `mma.sync`
# directement en PTX inline (@asmcall), avec le layout de fragment m16n8k32 spécifié
# par la PTX ISA. Comme AMD, aucune dépendance dure, aucun type fp8 possédé par KI —
# le chemin n'existe que si l'utilisateur a chargé DLFP8Types.
#
# NVIDIA `.e4m3`/`.e5m2` = variantes OCP **fn** (Float8_E4M3FN/Float8_E5M2), à ne pas
# confondre avec les **fnuz** d'AMD CDNA. Config à un seul CT ⇒ pas de combos mixtes.
#
# Layout m16n8k32 (PTX ISA), laneid∈0..31, g=laneid>>2, t=laneid&3 :
#   A[16×32] 4 regs (4 fp8/reg) : a0(g, t*4+·) a1(g+8, t*4+·) a2(g, t*4+16+·) a3(g+8, t*4+16+·)
#   B[32×8]  2 regs             : b0(t*4+·, g) b1(t*4+16+·, g)
#   C/D[16×8] f32 4 regs        : c0(g,t*2) c1(g,t*2+1) c2(g+8,t*2) c3(g+8,t*2+1)

module KernelIntrinsicsCUDADLFP8TypesExt

using CUDA
using DLFP8Types: Float8_E4M3FN, Float8_E5M2
import LLVM
using LLVM.Interop: @asmcall

import KernelIntrinsics.MMA: MMAConfig, ColMajor, RowMajor, MatrixA, MatrixB, Accumulator, MulAdd
import KernelIntrinsics.MMA: _load_a, _load_b, _load_c, _fill_c, _mma, _store_d!, mma_supported, _register_ext_shape!

# Fragment mma.sync opaque : NTuple de registres, distribution thread↔élément non exposée.
struct SyncFrag{Use,L,T}
    x::NTuple{L,T}
end


# Coordonnées PHYSIQUES d'un élément logique (u, v) de la tuile selon le layout.
@inline _ij(::Type{ColMajor}, r0, c0, u, v) = (r0 + u + 1, c0 + v + 1)
@inline _ij(::Type{RowMajor}, r0, c0, u, v) = (r0 + v + 1, c0 + u + 1)

# Empile 4 fp8 consécutifs en k (A : ligne m, colonnes kb..kb+3) en un UInt32 LE.
@inline function _packA(A, r0, c0, m, kb, ::Type{LAY}) where {LAY}
    b0 = _ij(LAY, r0, c0, m, kb);     b1 = _ij(LAY, r0, c0, m, kb + 1)
    b2 = _ij(LAY, r0, c0, m, kb + 2); b3 = _ij(LAY, r0, c0, m, kb + 3)
    @inbounds (UInt32(reinterpret(UInt8, A[b0[1], b0[2]]))) |
              (UInt32(reinterpret(UInt8, A[b1[1], b1[2]])) << 8) |
              (UInt32(reinterpret(UInt8, A[b2[1], b2[2]])) << 16) |
              (UInt32(reinterpret(UInt8, A[b3[1], b3[2]])) << 24)
end
# B : colonne n, lignes kb..kb+3.
@inline function _packB(B, r0, c0, kb, n, ::Type{LAY}) where {LAY}
    b0 = _ij(LAY, r0, c0, kb, n);     b1 = _ij(LAY, r0, c0, kb + 1, n)
    b2 = _ij(LAY, r0, c0, kb + 2, n); b3 = _ij(LAY, r0, c0, kb + 3, n)
    @inbounds (UInt32(reinterpret(UInt8, B[b0[1], b0[2]]))) |
              (UInt32(reinterpret(UInt8, B[b1[1], b1[2]])) << 8) |
              (UInt32(reinterpret(UInt8, B[b2[1], b2[2]])) << 16) |
              (UInt32(reinterpret(UInt8, B[b3[1], b3[2]])) << 24)
end

# (type fp8, tag PTX). Une seule forme fp8 côté NVIDIA : m16n8k32, accumulateur f32.
const _MMASYNC_FP8 = ((Float8_E4M3FN, "e4m3"), (Float8_E5M2, "e5m2"))

for (F8, tag) in _MMASYNC_FP8
    ptx = "mma.sync.aligned.m16n8k32.row.col.f32.$tag.$tag.f32 " *
          "{\$0,\$1,\$2,\$3}, {\$4,\$5,\$6,\$7}, {\$8,\$9}, {\$10,\$11,\$12,\$13};"
    call = Symbol("_mma_call_", tag)
    @eval begin
        @inline function $call(a::NTuple{4,UInt32}, b::NTuple{2,UInt32}, c::NTuple{4,Float32})
            @asmcall($ptx, "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", true,
                     Tuple{Float32,Float32,Float32,Float32},
                     Tuple{UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32},
                     a[1], a[2], a[3], a[4], b[1], b[2], c[1], c[2], c[3], c[4])
        end

        CUDA.@device_override @inline function _load_a(::MMAConfig{16,8,32,$F8,Float32,MulAdd}, A, row, col, ::Type{LAY}) where {LAY}
            r0 = row - 1; c0 = col - 1; lane = Int(CUDA.laneid()) - 1
            g = lane >> 2; t = lane & 3
            SyncFrag{MatrixA,4,UInt32}((_packA(A, r0, c0, g, t * 4, LAY), _packA(A, r0, c0, g + 8, t * 4, LAY),
                                       _packA(A, r0, c0, g, t * 4 + 16, LAY), _packA(A, r0, c0, g + 8, t * 4 + 16, LAY)))
        end

        CUDA.@device_override @inline function _load_b(::MMAConfig{16,8,32,$F8,Float32,MulAdd}, B, row, col, ::Type{LAY}) where {LAY}
            r0 = row - 1; c0 = col - 1; lane = Int(CUDA.laneid()) - 1
            g = lane >> 2; t = lane & 3
            SyncFrag{MatrixB,2,UInt32}((_packB(B, r0, c0, t * 4, g, LAY), _packB(B, r0, c0, t * 4 + 16, g, LAY)))
        end

        CUDA.@device_override @inline function _load_c(::MMAConfig{16,8,32,$F8,Float32,MulAdd}, C, row, col, ::Type{LAY}) where {LAY}
            r0 = row - 1; c0 = col - 1; lane = Int(CUDA.laneid()) - 1
            g = lane >> 2; t = lane & 3
            i0 = _ij(LAY, r0, c0, g, t * 2);     i1 = _ij(LAY, r0, c0, g, t * 2 + 1)
            i2 = _ij(LAY, r0, c0, g + 8, t * 2); i3 = _ij(LAY, r0, c0, g + 8, t * 2 + 1)
            @inbounds SyncFrag{Accumulator,4,Float32}((Float32(C[i0[1], i0[2]]), Float32(C[i1[1], i1[2]]),
                                                       Float32(C[i2[1], i2[2]]), Float32(C[i3[1], i3[2]])))
        end

        CUDA.@device_override @inline _fill_c(::MMAConfig{16,8,32,$F8,Float32,MulAdd}, v) =
            SyncFrag{Accumulator,4,Float32}(ntuple(_ -> Float32(v), Val(4)))

        CUDA.@device_override @inline _mma(::MMAConfig{16,8,32,$F8,Float32,MulAdd},
                                           a::SyncFrag{MatrixA}, b::SyncFrag{MatrixB}, c::SyncFrag{Accumulator}) =
            SyncFrag{Accumulator,4,Float32}($call(a.x, b.x, c.x))

        CUDA.@device_override @inline function _store_d!(::MMAConfig{16,8,32,$F8,Float32,MulAdd}, C, row, col,
                                                         d::SyncFrag{Accumulator}, ::Type{LAY}) where {LAY}
            r0 = row - 1; c0 = col - 1; lane = Int(CUDA.laneid()) - 1
            g = lane >> 2; t = lane & 3
            i0 = _ij(LAY, r0, c0, g, t * 2);     i1 = _ij(LAY, r0, c0, g, t * 2 + 1)
            i2 = _ij(LAY, r0, c0, g + 8, t * 2); i3 = _ij(LAY, r0, c0, g + 8, t * 2 + 1)
            @inbounds begin
                C[i0[1], i0[2]] = d.x[1]; C[i1[1], i1[2]] = d.x[2]
                C[i2[1], i2[2]] = d.x[3]; C[i3[1], i3[2]] = d.x[4]
            end
            nothing
        end

        mma_supported(::MMAConfig{16,8,32,$F8,Float32,MulAdd}) = _fp8_supported()
    end
end

# Le fp8 mma.sync exige sm_89/sm_90 ET LLVM.jl ≥ 9.8. Le layout m16n8k32 est spécifié
# par la PTX ISA (indépendant de l'archi), mais l'`@asmcall` de LLVM.jl < 9.8 construit
# mal un asm inline à N≥2 sorties directes : il utilise le type Julia (NTuple homogène
# ⇒ `[4 x float]`) au lieu du struct `{float×4}` qu'impose LLVM, ce qui miscompile le
# `mma.sync` → trappe device. Le correctif (asm_rettyp en struct) est arrivé en 9.8.
# On croise donc la capability avec la version de LLVM.jl, comme AMD croise l'ISA avec
# _MFMA_VALIDATED : ne jamais annoncer un chemin qui trappe. Toolchain ancien ⇒ fp8
# retombe sur le fallback portable (correct). VALIDÉ : 9.10 OK, 9.4 trappe ; le fix est
# présent dans le source depuis 9.8.
@inline _fp8_supported() = CUDA.capability(CUDA.device()) >= v"8.9" && pkgversion(LLVM) >= v"9.8"

function __init__()
    for (F8, _) in _MMASYNC_FP8
        _register_ext_shape!(CUDA.CUDABackend, (M = 16, N = 8, K = 32, compute = F8, acc = Float32),
                             _fp8_supported)
    end
end

end # module KernelIntrinsicsCUDADLFP8TypesExt
