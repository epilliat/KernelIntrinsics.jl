# ============================================================================
# KernelIntrinsics.MMA — matrix-multiply-accumulate (tensor cores) cross-arch
# ============================================================================
#
# Primitives MMA portables, dans l'esprit de la couche warp : une seule API,
# deux chemins derrière (hardware quand l'archi + le type le permettent, fallback
# warp-coopératif pur-Julia sinon). L'appelant écrit le même code ; le choix du
# chemin est résolu par dispatch. Le layout de fragment est OPAQUE — jamais
# d'indexation élément-par-élément dans le contrat.
#
# Le fallback (régime « correction ») est adapté du GeneralFPUOp de GemmKernels.jl
# (BSD-3, © 2020 Thomas Faingnaert and contributors ; Faingnaert, Besard, De
# Sutter, "Flexible Performant GEMM Kernels on GPUs", IEEE TPDS 33(9) 2022,
# doi:10.1109/TPDS.2021.3136457), rendu portable : lane via _laneid(), taille de
# grille via la warpsize (plus de threadIdx()/32 codés en dur), et opérateur
# d'accumulation paramétrable. Voir THIRD_PARTY_LICENSES.

module MMA

# Le sous-module atteint les stubs du parent (KernelIntrinsics) sans imports
# relatifs fragiles : on alias les fonctions via parentmodule. `_laneid` est
# déclaré (et surchargé par backend via @device_override) dans le parent, donc
# l'alias pointe sur le MÊME objet-fonction et le dispatch device fonctionne.
const _laneid = parentmodule(@__MODULE__)._laneid

export MMAConfig
export MatrixA, MatrixB, Accumulator, RowMajor, ColMajor
export MulAdd, Tropical
export load_a, load_b, load_c, fill_c, mma, store_d!
export mma_supported

# ── Tags ────────────────────────────────────────────────────────────────────
abstract type MatrixUse end
struct MatrixA <: MatrixUse end
struct MatrixB <: MatrixUse end
struct Accumulator <: MatrixUse end

abstract type FragLayout end
struct RowMajor <: FragLayout end
struct ColMajor <: FragLayout end

# ── Opérateur d'accumulation paramétrable ────────────────────────────────────
# L'anneau standard (fma) est le cas HW ; les semi-anneaux (tropical, …) sont
# fallback-only mais partagent exactement la même API.
abstract type AccOp end
struct MulAdd   <: AccOp end     # anneau standard : c + a·b
struct Tropical <: AccOp end     # semi-anneau max-plus : max(a+b, c)

# `muladd` (et non `fma`) : défini au-delà des flottants — Complex, et retombe sur
# `a*b+c` pour les types custom. Nécessaire au régime « structures complexes ».
@inline acc(::MulAdd,   a, b, c) = muladd(a, b, c)
@inline acc(::Tropical, a, b, c) = max(a + b, c)

# Identité additive de l'opérateur (valeur de remplissage naturelle de C).
@inline acc_identity(::MulAdd,   ::Type{T}) where {T} = zero(T)
@inline acc_identity(::Tropical, ::Type{T}) where {T} = typemin(T)

# ── Config = operator type porteur ───────────────────────────────────────────
# M,N,K = forme de la tuile MMA ; CT = type de calcul ; AccT = type d'accumulation
# (paramètre de 1re classe) ; OP = opérateur d'accumulation (défaut : MulAdd).
struct MMAConfig{M,N,K,CT,AccT,OP} end
(::Type{MMAConfig{M,N,K,CT,AccT}})() where {M,N,K,CT,AccT} = MMAConfig{M,N,K,CT,AccT,MulAdd}()

# ── Fragment opaque du FALLBACK (NTuple interne, layout non exposé) ──────────
# Le chemin HW renvoie SON propre type de fragment (WMMA.Fragment, …) ; le
# dispatch de `mma`/`store_d!` se fait sur le type de fragment.
struct Frag{Use<:MatrixUse,T,L}
    x::NTuple{L,T}
end

# ── API publique → stubs surchargeables par backend (fallback par défaut) ────
@inline load_a(cfg::MMAConfig, A, idx, layout) = _load_a(cfg, A, idx, layout)
@inline load_b(cfg::MMAConfig, B, idx, layout) = _load_b(cfg, B, idx, layout)
@inline load_c(cfg::MMAConfig, C, idx, layout) = _load_c(cfg, C, idx, layout)
@inline fill_c(cfg::MMAConfig, v)              = _fill_c(cfg, v)
@inline mma(cfg::MMAConfig, a, b, c)           = _mma(cfg, a, b, c)
@inline store_d!(cfg::MMAConfig, C, idx, d, layout) = _store_d!(cfg, C, idx, d, layout)

"""
    mma_supported(cfg::MMAConfig) -> Bool

Vrai si un chemin **hardware** existe pour cette config sur le backend chargé.
Faux ⇒ l'API reste utilisable mais passe par le fallback portable. Réponse
grossière (par type/forme, pas par device exact) — suffisante pour choisir une
tuile côté appelant. Les extensions backend la surchargent.
"""
mma_supported(::MMAConfig) = false

# ============================================================================
# Fallback portable (régime « correction »)
# ============================================================================
# Warpsize à la compilation. Backend-surchargeable : 32 sur CUDA / RDNA / Metal,
# 64 sur CDNA. Défaut 32. `Val{32}()` (et non `Val(32)`) pour un type de retour
# STABLE — sinon l'appel aux fallbacks @generated devient un dispatch dynamique
# (invalide sur GPU).
@inline _mma_wave() = Val{32}()

# Grille lane→tuile mb×nb (mb·nb = warpsize), mb | M, nb | N, la plus équilibrée.
# Utilisée UNIQUEMENT dans les @generated (temps macro), donc pas d'enjeu d'inférence.
function _lane_grid(M, N, WS)
    best = nothing
    for mb in 1:WS
        WS % mb == 0 || continue
        nb = WS ÷ mb
        (M % mb == 0 && N % nb == 0) || continue
        d = abs(mb - nb)
        (best === nothing || d < best[3]) && (best = (mb, nb, d))
    end
    best === nothing &&
        error("MMA fallback: aucune grille lane mb×nb=$WS avec mb|$M, nb|$N")
    return (best[1], best[2])
end

@inline _load_a(cfg::MMAConfig, A, idx, layout) = _load_a_fb(cfg, A, idx, layout, _mma_wave())
@inline _load_b(cfg::MMAConfig, B, idx, layout) = _load_b_fb(cfg, B, idx, layout, _mma_wave())
@inline _load_c(cfg::MMAConfig, C, idx, layout) = _load_c_fb(cfg, C, idx, layout, _mma_wave())
@inline _fill_c(cfg::MMAConfig, v)              = _fill_c_fb(cfg, v, _mma_wave())
@inline _mma(cfg::MMAConfig, a::Frag{MatrixA}, b::Frag{MatrixB}, c::Frag{Accumulator}) =
    _mma_fb(cfg, a, b, c, _mma_wave())
@inline _store_d!(cfg::MMAConfig, C, idx, d::Frag{Accumulator}, layout) =
    _store_d_fb!(cfg, C, idx, d, layout, _mma_wave())

# Décodage tuile-origine (col-major) : idx linéaire 1-based → (row0, col0). Émis
# dans le prologue de chaque @generated.
# load_a : lane possède ses lignes (op_y + mb·mi), toutes les colonnes K.
@generated function _load_a_fb(::MMAConfig{M,N,K,CT,AccT,OP}, A, idx, ::Type{ColMajor},
                               ::Val{WS}) where {M,N,K,CT,AccT,OP,WS}
    mb, _ = _lane_grid(M, N, WS)
    Mr = M ÷ mb
    L = Mr * K
    els = Vector{Any}(undef, L)
    for i in 1:L
        mi = (i - 1) % Mr
        k  = (i - 1) ÷ Mr
        els[i] = :($CT(@inbounds A[r0 + op_y + $(mb * mi) + 1, c0 + $k + 1]))
    end
    quote
        ld = size(A, 1); b0 = idx - 1
        r0 = b0 % ld; c0 = b0 ÷ ld
        op_y = (_laneid() - 1) % $mb
        Frag{MatrixA,$CT,$L}(($(els...),))
    end
end

# load_b : lane possède ses colonnes (op_x + nb·ni), toutes les lignes K.
@generated function _load_b_fb(::MMAConfig{M,N,K,CT,AccT,OP}, B, idx, ::Type{ColMajor},
                               ::Val{WS}) where {M,N,K,CT,AccT,OP,WS}
    mb, nb = _lane_grid(M, N, WS)
    Nc = N ÷ nb
    L = K * Nc
    els = Vector{Any}(undef, L)
    for j in 1:L
        k  = (j - 1) % K
        ni = (j - 1) ÷ K
        els[j] = :($CT(@inbounds B[r0 + $k + 1, c0 + op_x + $(nb * ni) + 1]))
    end
    quote
        ld = size(B, 1); b0 = idx - 1
        r0 = b0 % ld; c0 = b0 ÷ ld
        op_x = (_laneid() - 1) ÷ $mb
        Frag{MatrixB,$CT,$L}(($(els...),))
    end
end

# load_c : lit l'accumulateur depuis la mémoire (mêmes indices que store_d!).
@generated function _load_c_fb(::MMAConfig{M,N,K,CT,AccT,OP}, C, idx, ::Type{ColMajor},
                               ::Val{WS}) where {M,N,K,CT,AccT,OP,WS}
    mb, nb = _lane_grid(M, N, WS)
    Mr = M ÷ mb; Nc = N ÷ nb; L = Mr * Nc
    els = Vector{Any}(undef, L)
    for idx0 in 1:L
        mi = (idx0 - 1) % Mr
        ni = (idx0 - 1) ÷ Mr
        els[idx0] = :($AccT(@inbounds C[r0 + op_y + $(mb * mi) + 1, c0 + op_x + $(nb * ni) + 1]))
    end
    quote
        ld = size(C, 1); b0 = idx - 1
        r0 = b0 % ld; c0 = b0 ÷ ld
        op_y = (_laneid() - 1) % $mb
        op_x = (_laneid() - 1) ÷ $mb
        Frag{Accumulator,$AccT,$L}(($(els...),))
    end
end

@generated function _fill_c_fb(::MMAConfig{M,N,K,CT,AccT,OP}, v,
                               ::Val{WS}) where {M,N,K,CT,AccT,OP,WS}
    mb, nb = _lane_grid(M, N, WS)
    L = (M ÷ mb) * (N ÷ nb)
    # Splat de littéraux (pas de closure/`ntuple(_->…)`) : le corps d'un @generated
    # doit rester « pur ».
    els = Vector{Any}(undef, L)
    for i in 1:L
        els[i] = :($AccT(v))
    end
    quote
        Frag{Accumulator,$AccT,$L}(($(els...),))
    end
end

# mma : c[mi,ni] = acc(a[mi,k], b[k,ni], c[mi,ni]) sur k. Purement par-lane.
@generated function _mma_fb(::MMAConfig{M,N,K,CT,AccT,OP}, a::Frag{MatrixA},
                            b::Frag{MatrixB}, c::Frag{Accumulator},
                            ::Val{WS}) where {M,N,K,CT,AccT,OP,WS}
    mb, nb = _lane_grid(M, N, WS)
    Mr = M ÷ mb; Nc = N ÷ nb; L = Mr * Nc
    outs = Vector{Any}(undef, L)
    for idx0 in 1:L
        mi = (idx0 - 1) % Mr
        ni = (idx0 - 1) ÷ Mr
        e = :(c.x[$idx0])
        for k in 0:(K - 1)
            e = :(acc($OP(), a.x[$(mi + Mr * k + 1)], b.x[$(k + K * ni + 1)], $e))
        end
        outs[idx0] = e
    end
    quote
        Frag{Accumulator,$AccT,$L}(($(outs...),))
    end
end

# store_d! : chaque lane écrit son sous-bloc (op_y+mb·mi, op_x+nb·ni).
@generated function _store_d_fb!(::MMAConfig{M,N,K,CT,AccT,OP}, C, idx, d::Frag{Accumulator},
                                 ::Type{ColMajor}, ::Val{WS}) where {M,N,K,CT,AccT,OP,WS}
    mb, nb = _lane_grid(M, N, WS)
    Mr = M ÷ mb; Nc = N ÷ nb
    sts = Any[]
    for ni in 0:(Nc - 1), mi in 0:(Mr - 1)
        push!(sts, :(@inbounds C[r0 + op_y + $(mb * mi) + 1, c0 + op_x + $(nb * ni) + 1] =
                         d.x[$(mi + Mr * ni + 1)]))
    end
    quote
        ld = size(C, 1); b0 = idx - 1
        r0 = b0 % ld; c0 = b0 ÷ ld
        op_y = (_laneid() - 1) % $mb
        op_x = (_laneid() - 1) ÷ $mb
        $(sts...)
        nothing
    end
end

end # module MMA
