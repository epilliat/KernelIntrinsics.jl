# Chemin AMD pour KernelIntrinsics.MMA.
#
#   1) FALLBACK wave64 — override _mma_wave() = Val{64}() (le fallback portable de
#      src/mma.jl devient correct sur CDNA/MI300, tous types).
#   2) HARDWARE MFMA (CDNA/gfx942) — chemin tensor-core, généré par `@eval` depuis
#      la table _MFMA_OPS ci-dessous (même style que ext/CUDA/shuffle_vote.jl).
#
# Ajouter une forme = ajouter UNE ligne à la table… puis LA VALIDER SUR MI300 :
# les layouts diffèrent réellement d'une op à l'autre (cf. blocked vs interleaved
# ci-dessous, découvert à l'exécution). Ne jamais les supposer.
#
# TODO : fp8/bf8 (4 combos), 32×32×16 i8, SMFMAC (sparse), RDNA3 wave32/WMMA.

import KernelIntrinsics.MMA: MMAConfig, ColMajor, RowMajor, MatrixA, MatrixB, Accumulator, MulAdd
import KernelIntrinsics.MMA: _load_a, _load_b, _load_c, _fill_c, _mma, _store_d!, _mma_wave, mma_supported

# ── 1) Fallback wave64 ───────────────────────────────────────────────────────
@amdgpu_overlay @inline _mma_wave() = Val{64}()

# ── 2) MFMA ──────────────────────────────────────────────────────────────────
# Fragment MFMA opaque : NTuple de VecElement, layout interne CDNA non exposé.
struct MFMAFrag{Use,L,T}
    x::NTuple{L,VecElement{T}}
end

@inline _lane0() = Int(AMDGPU.Device.activelane())   # lane 0..63

# L'origine de tuile arrive DÉJÀ décodée en (row, col) 1-based depuis src/mma.jl :
# plus de `b0 % ld` / `b0 ÷ ld` ici, donc plus de division entière à l'exécution
# dans la boucle-K. Simples décalages 1-based → 0-based.
@inline _base(row, col) = (row - 1, col - 1)

# Table des intrinsics MFMA supportés.
#   nc = éléments accumulateur par lane.
#   Layouts (g = lane÷M, j = index d'élément 0-based) — tous VALIDÉS sur MI300 :
#     opérandes  : A[m,k] m = lane%M, k = na*g + j ;  B[k,n] n = lane%N, k = na*g + j
#     accumulateur, n = lane%N, et
#       :blocked     → m = nc*g + j                   (accum f32/i32)
#       :interleaved → m = nc*j + g                   (accum f64)
#       :blk4        → m = 8*(j÷4) + 4*g + j%4        (famille 32×32)
#   ST = type de STOCKAGE de l'opérande vu par l'intrinsic. Peut différer de CT :
#        - bf16 : LLVM le prend en <4 x i16>            ⇒ ST=Int16, reinterpret
#        - i8   : 8 int8 PACKÉS dans un i64             ⇒ ST=Int64, packing
#        Le facteur de packing pk = sizeof(ST)÷sizeof(CT) est déduit, pas stocké.
#   na = éléments de CT par lane ; nw = na÷pk mots ST réellement passés.
#   archs = architectures GCN où l'op EXISTE d'après l'ISA CDNA. MFMA est propre à
#        CDNA — sur RDNA il n'y a rien de tout ça.
#
# ⚠️ « Exister » ne veut PAS dire « avoir le même layout ». Une erreur d'archi sur
# l'INTRINSIC produit une erreur de compilation bruyante (« Cannot select »), mais
# une erreur d'archi sur le LAYOUT d'accumulateur produit des chiffres
# silencieusement faux : rien ne garantit que gfx908/gfx90a/gfx950 distribuent
# l'accumulateur sur les lanes comme gfx942. Or gfx942 est la SEULE archi où ces
# layouts ont été vérifiés sur hardware (cf. local/mma/proto/probe_layout.jl).
# Donc `mma_supported` croise la table ISA avec les archis réellement validées :
# mieux vaut sous-promettre (⇒ fallback portable, correct partout) que promettre
# un layout supposé. Élargir _MFMA_VALIDATED EXIGE de repasser le prober.
const _CDNA1 = ("gfx908", "gfx90a", "gfx942", "gfx950")   # MFMA de base
const _CDNA2 = ("gfx90a", "gfx942", "gfx950")             # + bf16 .1k, f64
const _CDNA3 = ("gfx942", "gfx950")                       # + i8 K=32

const _MFMA_VALIDATED = ("gfx942",)   # layouts confirmés sur hardware

const _MFMA_OPS = (
    # (M, N, K, CT, AccT, ST, intrinsic, na, nc, acc_layout, archs)
    (16, 16, 16, Float16, Float32, Float16, "llvm.amdgcn.mfma.f32.16x16x16f16", 4, 4, :blocked, _CDNA1),
    (16, 16, 4, Float64, Float64, Float64, "llvm.amdgcn.mfma.f64.16x16x4f64", 1, 4, :interleaved, _CDNA2),
    (32, 32, 8, Float16, Float32, Float16, "llvm.amdgcn.mfma.f32.32x32x8f16", 4, 16, :blk4, _CDNA1),
    (16, 16, 16, Core.BFloat16, Float32, Int16, "llvm.amdgcn.mfma.f32.16x16x16bf16.1k", 4, 4, :blocked, _CDNA2),
    (32, 32, 8, Core.BFloat16, Float32, Int16, "llvm.amdgcn.mfma.f32.32x32x8bf16.1k", 4, 16, :blk4, _CDNA2),
    # gfx942 a REMPLACÉ le i8 K=16 (gfx908/90a) par K=32 à opérandes i64 : le
    # `mfma.i32.16x16x16i8` existe dans LLVM mais finit en « Cannot select » ici.
    (16, 16, 32, Int8, Int32, Int64, "llvm.amdgcn.mfma.i32.16x16x32.i8", 8, 4, :blocked, _CDNA3),
    (32, 32, 16, Int8, Int32, Int64, "llvm.amdgcn.mfma.i32.32x32x16.i8", 8, 16, :blk4, _CDNA3),
)

# Architecture GCN du device courant, suffixes de features retirés
# ("gfx942:sramecc+:xnack-" → "gfx942").
_gfx() = first(split(AMDGPU.device().gcn_arch, ':'))

for (M, N, K, CT, AccT, ST, INTR, na, nc, acclay, ARCHS) in _MFMA_OPS
    pk = sizeof(ST) ÷ sizeof(CT)                    # éléments CT par mot ST
    nw = na ÷ pk                                    # mots ST par lane
    AV = nw == 1 ? ST : NTuple{nw,VecElement{ST}}   # type d'opérande vu par l'intrinsic
    CV = NTuple{nc,VecElement{AccT}}                # type accumulateur

    # Un « mot » ST à partir de pk éléments : conversion/reinterpret si pk==1,
    # sinon packing bit-à-bit (little-endian : élément p aux bits [8p, 8p+8)).
    UT = sizeof(ST) == 8 ? UInt64 : UInt32
    bits = 8 * sizeof(CT)
    word = (el, base) -> begin
        if pk == 1
            ST === CT ? :($CT($(el(base)))) : :(reinterpret($ST, $(el(base))))
        else
            ex = :(zero($UT))
            for p in 0:(pk - 1)
                # `base` est une Expr (index calculé à l'exécution) : on COMPOSE
                # l'expression `base + p`, on ne l'additionne pas au temps macro.
                ex = :($ex | ($UT(reinterpret($(Base.unsigned(CT)), $(el(:($base + $p))))) << $(bits * p)))
            end
            :(reinterpret($ST, $ex))
        end
    end
    unwrap = nw == 1 ? :(f.x[1].value) : :(f.x)     # frag → opérande intrinsic
    # Ligne accumulateur (g = lane÷M, j = index élément 0-based).
    accm = if acclay === :blocked
        :($nc * g + j)                                  # 16×16, accum f32
    elseif acclay === :interleaved
        :($nc * j + g)                                  # 16×16, accum f64
    elseif acclay === :blk4
        :($(M ÷ (nc ÷ 4)) * (j ÷ 4) + 4 * g + j % 4)   # 32×32 : 4 blocs de 4 lignes
    else
        error("layout accumulateur inconnu : $acclay")
    end

    @eval begin
        @inline _mfma_call(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, a::$AV, b::$AV, c::$CV) =
            ccall($INTR, llvmcall, $CV, ($AV, $AV, $CV, Int32, Int32, Int32),
                  a, b, c, Int32(0), Int32(0), Int32(0))

        @inline _operand(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, f::MFMAFrag) = $unwrap

        @amdgpu_overlay @inline _fill_c(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, v) =
            MFMAFrag{Accumulator,$nc,$AccT}(ntuple(_ -> VecElement($AccT(v)), Val($nc)))

        @amdgpu_overlay @inline function _mma(cfg::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd},
                                              a::MFMAFrag{MatrixA}, b::MFMAFrag{MatrixB},
                                              c::MFMAFrag{Accumulator})
            MFMAFrag{Accumulator,$nc,$AccT}(_mfma_call(cfg, _operand(cfg, a), _operand(cfg, b), c.x))
        end
    end

    # ── Accès mémoire : une méthode par (forme, LAYOUT) ──────────────────────
    # RowMajor = échanger les deux composantes d'index, exactement comme dans le
    # fallback de src/mma.jl (et comme WMMA sur CUDA). Sans ça, `A'*B` était une
    # MethodError sur AMD alors qu'elle marchait sur NVIDIA : contrat divergent.
    # Le lane-mapping (m/n/g), lui, ne dépend pas du layout — seul le placement
    # mémoire change.
    for (LAYT, sw) in ((:ColMajor, false), (:RowMajor, true))
        # Élément CT n° `jj` (0-based) de cette lane. Coordonnées LOGIQUES dans la
        # tuile : A → (m, na*g+jj) ; B → (na*g+jj, n) ; accumulateur → (accm, n).
        ax = (u, v) -> sw ? (v, u) : (u, v)
        elA = jj -> (uv = ax(:(m), :($na * g + $jj)); :(A[r0 + $(uv[1]) + 1, c0 + $(uv[2]) + 1]))
        elB = jj -> (uv = ax(:($na * g + $jj), :(n)); :(B[r0 + $(uv[1]) + 1, c0 + $(uv[2]) + 1]))
        accuv = ax(accm, :(n))
        accix = :(C[r0 + $(accuv[1]) + 1, c0 + $(accuv[2]) + 1])
        rdA = word(elA, :((e - 1) * $pk))
        rdB = word(elB, :((e - 1) * $pk))

        @eval begin
            @amdgpu_overlay @inline function _load_a(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, A, row, col, ::Type{$LAYT})
                r0, c0 = _base(row, col); lane = _lane0()
                m = lane % $M; g = lane ÷ $M
                x = ntuple(Val($nw)) do e
                    @inbounds VecElement($rdA)
                end
                MFMAFrag{MatrixA,$nw,$ST}(x)
            end

            @amdgpu_overlay @inline function _load_b(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, B, row, col, ::Type{$LAYT})
                r0, c0 = _base(row, col); lane = _lane0()
                n = lane % $N; g = lane ÷ $N
                x = ntuple(Val($nw)) do e
                    @inbounds VecElement($rdB)
                end
                MFMAFrag{MatrixB,$nw,$ST}(x)
            end

            @amdgpu_overlay @inline function _load_c(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, C, row, col, ::Type{$LAYT})
                r0, c0 = _base(row, col); lane = _lane0()
                # g est l'index de GROUPE dans la partition par colonne (n = lane % N),
                # donc lane ÷ N — pas ÷ M. Les deux coïncidaient tant que la table ne
                # contenait que des formes carrées ; latent pour toute forme M ≠ N.
                n = lane % $N; g = lane ÷ $N
                x = ntuple(Val($nc)) do e
                    j = e - 1
                    @inbounds VecElement($AccT($accix))
                end
                MFMAFrag{Accumulator,$nc,$AccT}(x)
            end

            @amdgpu_overlay @inline function _store_d!(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, C, row, col,
                                                       d::MFMAFrag{Accumulator}, ::Type{$LAYT})
                r0, c0 = _base(row, col); lane = _lane0()
                n = lane % $N; g = lane ÷ $N     # cf. _load_c : ÷ N, pas ÷ M
                for e in 1:$nc
                    j = e - 1
                    @inbounds $accix = d.x[e].value
                end
                nothing
            end
        end
    end

    @eval begin
        # Query hôte : gatée sur l'ARCHITECTURE RÉELLE du device (MFMA = CDNA only).
        # ⚠️ Les overrides device ci-dessus, eux, sont inconditionnels : appeler une
        # config sur une archi qui ne l'a pas échoue à la COMPILATION (« Cannot
        # select ») au lieu de retomber sur le fallback. `mma_supported` est donc le
        # garde-fou que l'appelant doit interroger avant de choisir sa tuile.
        mma_supported(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}) =
            _gfx() in $ARCHS && _gfx() in _MFMA_VALIDATED
    end
end
