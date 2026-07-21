# ============================================================================
# Codegen MFMA partagé (CDNA/AMD) — vit dans le package principal, PAS dans une
# extension, pour une seule raison : `MFMAFrag` et `_emit_mfma_row` doivent être
# UNIQUES et partagés entre l'extension AMDGPU de base et l'extension fp8
# multi-déclencheur (AMDGPU + DLFP8Types). Un `MFMAFrag` défini deux fois donnerait
# deux types distincts et casserait le dispatch de `_mma`. Ce code est dormant tant
# qu'aucune extension AMD ne l'appelle : il ne référence AMDGPU qu'à l'intérieur de
# `quote` (résolus au moment du Core.eval dans l'extension).
#
# Chaque extension fait, pour chacune de ses lignes de table :
#     _emit_mfma_row(@__MODULE__, M, N, K, CT, AccT, ST, INTR, na, nc, acclay, ARCHS)
# ce qui émet _mfma_call/_operand/_fill_c/_mma + _load_a/b/c/_store_d! (× layouts) +
# mma_supported, exactement comme l'ancienne boucle @eval inline le faisait.

# Fragment MFMA opaque : NTuple de VecElement, layout interne CDNA non exposé.
# Backend-agnostique (Base uniquement), d'où sa place ici.
struct MFMAFrag{Use,L,T}
    x::NTuple{L,VecElement{T}}
end

# Archis GCN où les LAYOUTS d'accumulateur ont été confirmés sur hardware réel
# (cf. local/mma/proto/probe_layout.jl). L'ISA dit qu'une op existe ; seul un
# passage du prober prouve la distribution de l'accumulateur sur les lanes. Un
# mauvais layout donne des chiffres SILENCIEUSEMENT faux, contrairement à un
# mauvais intrinsic qui échoue bruyamment. Élargir EXIGE de repasser le prober.
const _MFMA_VALIDATED = ("gfx942",)

# Émet toutes les surcharges device pour UNE ligne de table, dans `mod`.
#
# ⚠️ CE SONT DES MÉTHODES ORDINAIRES, PAS DES OVERLAYS — ne pas remettre
# `@amdgpu_overlay` ici. C'est le jeton `::CDNAMFMA` (cf. src/mma.jl, section
# « Jeton matériel ») qui sépare AMD de CUDA, dont les signatures MMA sont
# autrement identiques (`_fill_c(::MMAConfig{16,16,16,Float16,Float32,MulAdd}, v)`
# existe des deux côtés). Un overlay sur une fonction qui PRODUIT un fragment
# empêche l'inférence de voir à travers l'appel et fait dégénérer l'accumulateur
# porté par la boucle-K en `phi [ undef, %preheader ]` ⇒ résultats SILENCIEUSEMENT
# faux. Seul `_mma_hw()` (singleton, sans donnée) reste un overlay, installé par
# l'extension AMDGPU de base.
#
# `AMDGPU`, `MMAConfig`, `MFMAFrag`, `CDNAMFMA`, les tags et `MulAdd` doivent être
# en portée dans `mod` (les extensions les importent).
function _emit_mfma_row(mod::Module, M, N, K, CT, AccT, ST, INTR, na, nc, acclay, ARCHS)
    pk = sizeof(ST) ÷ sizeof(CT)                    # éléments CT par mot ST
    nw = na ÷ pk                                    # mots ST par lane
    AV = nw == 1 ? ST : NTuple{nw,VecElement{ST}}   # type d'opérande vu par l'intrinsic
    CV = NTuple{nc,VecElement{AccT}}                # type accumulateur

    # Un « mot » ST à partir de pk éléments : conversion/reinterpret si pk==1,
    # sinon packing bit-à-bit (little-endian : élément p aux bits [b·p, b·p+b)).
    UT = sizeof(ST) == 8 ? UInt64 : UInt32
    bits = 8 * sizeof(CT)
    # Entier non signé de MÊME largeur que CT, pour reinterpréter un élément en bits
    # avant packing. `Base.unsigned(CT)` ne marche que pour les entiers (Int8→UInt8) :
    # il échoue sur un fp8 (type flottant). La largeur suffit — reinterpret(UInt8, x)
    # est valable pour tout `primitive type … 8`, entier comme fp8.
    UB = bits == 8 ? UInt8 : bits == 16 ? UInt16 : bits == 32 ? UInt32 : UInt64
    word = (el, base) -> begin
        if pk == 1
            ST === CT ? :($CT($(el(base)))) : :(reinterpret($ST, $(el(base))))
        else
            ex = :(zero($UT))
            for p in 0:(pk - 1)
                # `base` est une Expr (index calculé à l'exécution) : on COMPOSE
                # l'expression `base + p`, on ne l'additionne pas au temps macro.
                ex = :($ex | ($UT(reinterpret($UB, $(el(:($base + $p))))) << $(bits * p)))
            end
            :(reinterpret($ST, $ex))
        end
    end
    unwrap = nw == 1 ? :(f.x[1].value) : :(f.x)     # frag → opérande intrinsic
    # Ligne accumulateur (g = lane÷M, j = index élément 0-based).
    accm = if acclay === :blocked
        :($nc * g + j)                                  # 16×16, accum f32/i32
    elseif acclay === :interleaved
        :($nc * j + g)                                  # 16×16, accum f64
    elseif acclay === :blk4
        :($(M ÷ (nc ÷ 4)) * (j ÷ 4) + 4 * g + j % 4)   # 32×32 : 4 blocs de 4 lignes
    else
        error("layout accumulateur inconnu : $acclay")
    end

    Core.eval(mod, quote
        @inline _mfma_call(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, a::$AV, b::$AV, c::$CV) =
            ccall($INTR, llvmcall, $CV, ($AV, $AV, $CV, Int32, Int32, Int32),
                  a, b, c, Int32(0), Int32(0), Int32(0))

        @inline _operand(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, f::MFMAFrag) = $unwrap

        @inline _fill_c(::CDNAMFMA, ::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, v) =
            MFMAFrag{Accumulator,$nc,$AccT}(ntuple(_ -> VecElement($AccT(v)), Val($nc)))

        @inline function _mma(::CDNAMFMA, cfg::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd},
                              a::MFMAFrag{MatrixA}, b::MFMAFrag{MatrixB},
                              c::MFMAFrag{Accumulator})
            MFMAFrag{Accumulator,$nc,$AccT}(_mfma_call(cfg, _operand(cfg, a), _operand(cfg, b), c.x))
        end
    end)

    # ── Accès mémoire : une méthode par (forme, LAYOUT) ──────────────────────
    # RowMajor = échanger les deux composantes d'index, exactement comme le
    # fallback (src/mma.jl) et WMMA. Le lane-mapping (m/n/g) ne dépend pas du
    # layout — seul le placement mémoire change.
    for (LAYT, sw) in ((:ColMajor, false), (:RowMajor, true))
        # Élément CT n° `jj` (0-based) de cette lane. Coordonnées LOGIQUES dans la
        # tuile : A → (m, na·g+jj) ; B → (na·g+jj, n) ; accumulateur → (accm, n).
        ax = (u, v) -> sw ? (v, u) : (u, v)
        elA = jj -> (uv = ax(:(m), :($na * g + $jj)); :(A[r0 + $(uv[1]) + 1, c0 + $(uv[2]) + 1]))
        elB = jj -> (uv = ax(:($na * g + $jj), :(n)); :(B[r0 + $(uv[1]) + 1, c0 + $(uv[2]) + 1]))
        accuv = ax(accm, :(n))
        accix = :(C[r0 + $(accuv[1]) + 1, c0 + $(accuv[2]) + 1])
        rdA = word(elA, :((e - 1) * $pk))
        rdB = word(elB, :((e - 1) * $pk))

        Core.eval(mod, quote
            @inline function _load_a(::CDNAMFMA, ::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, A, row, col, ::Type{$LAYT})
                r0 = row - 1; c0 = col - 1; lane = Int(AMDGPU.Device.activelane())
                m = lane % $M; g = lane ÷ $M
                x = ntuple(Val($nw)) do e
                    @inbounds VecElement($rdA)
                end
                MFMAFrag{MatrixA,$nw,$ST}(x)
            end

            @inline function _load_b(::CDNAMFMA, ::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, B, row, col, ::Type{$LAYT})
                r0 = row - 1; c0 = col - 1; lane = Int(AMDGPU.Device.activelane())
                n = lane % $N; g = lane ÷ $N
                x = ntuple(Val($nw)) do e
                    @inbounds VecElement($rdB)
                end
                MFMAFrag{MatrixB,$nw,$ST}(x)
            end

            @inline function _load_c(::CDNAMFMA, ::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, C, row, col, ::Type{$LAYT})
                r0 = row - 1; c0 = col - 1; lane = Int(AMDGPU.Device.activelane())
                # g est l'index de GROUPE dans la partition par colonne (n = lane % N),
                # donc lane ÷ N — pas ÷ M. Les deux coïncident pour les formes carrées.
                n = lane % $N; g = lane ÷ $N
                x = ntuple(Val($nc)) do e
                    j = e - 1
                    @inbounds VecElement($AccT($accix))
                end
                MFMAFrag{Accumulator,$nc,$AccT}(x)
            end

            @inline function _store_d!(::CDNAMFMA, ::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, C, row, col,
                                       d::MFMAFrag{Accumulator}, ::Type{$LAYT})
                r0 = row - 1; c0 = col - 1; lane = Int(AMDGPU.Device.activelane())
                n = lane % $N; g = lane ÷ $N     # cf. _load_c : ÷ N, pas ÷ M
                for e in 1:$nc
                    j = e - 1
                    @inbounds $accix = d.x[e].value
                end
                nothing
            end
        end)
    end

    # Query hôte : gatée sur l'ARCHITECTURE RÉELLE du device (MFMA = CDNA only) ET
    # croisée avec _MFMA_VALIDATED. Les archs et la liste validée sont splicées en
    # littéral, donc aucune dépendance de portée dans `mod`.
    Core.eval(mod, quote
        mma_supported(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}) = begin
            g = first(split(AMDGPU.device().gcn_arch, ':'))
            g in $ARCHS && g in $(_MFMA_VALIDATED)
        end
    end)
    return nothing
end
