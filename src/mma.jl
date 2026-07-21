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
export mma_supported, mma_shapes
export mma_shape, compute_type, acc_type, acc_identity

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

# ── Accesseurs de config ─────────────────────────────────────────────────────
# Sans eux, un kernel générique en `cfg` ne peut ni dimensionner sa mémoire
# partagée ni typer ses tampons : il devrait re-déclarer M,N,K,… en paramètres,
# ce qui annule l'intérêt de porter la config comme un seul objet. Tout est
# résolu à la compilation (paramètres de type ⇒ constantes).
@inline mma_shape(::MMAConfig{M,N,K}) where {M,N,K} = (M = M, N = N, K = K)
@inline compute_type(::MMAConfig{M,N,K,CT}) where {M,N,K,CT} = CT
@inline acc_type(::MMAConfig{M,N,K,CT,AccT}) where {M,N,K,CT,AccT} = AccT
@inline acc_op(::MMAConfig{M,N,K,CT,AccT,OP}) where {M,N,K,CT,AccT,OP} = OP()

# ── Fragment opaque du FALLBACK (NTuple interne, layout non exposé) ──────────
# Le chemin HW renvoie SON propre type de fragment (WMMA.Fragment, …) ; le
# dispatch de `mma`/`store_d!` se fait sur le type de fragment.
struct Frag{Use<:MatrixUse,T,L}
    x::NTuple{L,T}
end

# Identité additive de l'opérateur porté par la config — valeur de remplissage
# naturelle de l'accumulateur : `fill_c(cfg, acc_identity(cfg))` est le neutre,
# quel que soit le semi-anneau (0 pour MulAdd, -Inf pour Tropical).
@inline acc_identity(cfg::MMAConfig) = acc_identity(acc_op(cfg), acc_type(cfg))

# ── API publique → stubs surchargeables par backend (fallback par défaut) ────
#
# DEUX points d'entrée, même sémantique :
#   • (row, col) — origine de la tuile en coordonnées 1-based. Forme PRIVILÉGIÉE.
#   • idx        — index linéaire 1-based (col-major), conservé pour compat.
#
# Le (row, col) n'est pas seulement du confort d'appel (il supprime le
# `1 + kb*16*size(A,1)` chez l'appelant) : il supprime surtout DEUX DIVISIONS
# ENTIÈRES à l'exécution par chargement de fragment. La forme linéaire doit
# décoder `r0 = b0 % ld ; c0 = b0 ÷ ld` avec `ld` connu seulement à l'exécution —
# une vraie division GPU, dans la boucle-K, sur le chemin chaud. En partant de
# (row, col) le décodage n'existe pas ; c'est la forme linéaire qui paie, et
# seulement elle. Les stubs internes prennent donc (row, col) comme forme
# canonique et la variante linéaire n'est qu'un adaptateur.
#
# ── Adjoint / Transpose de Julia (comportement MESURÉ, pas supposé) ─────────
# Le fallback et MFMA lisent leurs opérandes par `getindex` 2D : un `A'` de Julia
# y fonctionne donc tel quel, et pour un élément complexe la CONJUGAISON est
# gratuite — c'est `getindex` sur `Adjoint` qui la fait. WMMA, lui, prend un
# `pointer`, qui n'existe pas sur un `Adjoint` : l'appel échoue à la compilation
# (InvalidIRError). Autrement dit la divergence entre backends est BRUYANTE ;
# aucun chemin ne renvoie de chiffres faux.
#
# La forme portable — supportée à l'identique sur les trois chemins — est
# RowMajor sur le tableau parent, pas le wrapper :
#     load_a(cfg, parent, col, row, RowMajor)   ≡   load_a(cfg, parent', row, col, ColMajor)
# (l'origine s'échange en même temps que le layout).
@inline function _rowcol(A, idx)
    ld = size(A, 1); b0 = idx - 1
    return (b0 % ld + 1, b0 ÷ ld + 1)
end

@inline load_a(cfg::MMAConfig, A, row, col, layout) = _load_a(cfg, A, row, col, layout)
@inline load_b(cfg::MMAConfig, B, row, col, layout) = _load_b(cfg, B, row, col, layout)
@inline load_c(cfg::MMAConfig, C, row, col, layout) = _load_c(cfg, C, row, col, layout)
@inline store_d!(cfg::MMAConfig, C, row, col, d, layout) = _store_d!(cfg, C, row, col, d, layout)

@inline load_a(cfg::MMAConfig, A, idx, layout) = ((r, c) = _rowcol(A, idx); _load_a(cfg, A, r, c, layout))
@inline load_b(cfg::MMAConfig, B, idx, layout) = ((r, c) = _rowcol(B, idx); _load_b(cfg, B, r, c, layout))
@inline load_c(cfg::MMAConfig, C, idx, layout) = ((r, c) = _rowcol(C, idx); _load_c(cfg, C, r, c, layout))
@inline store_d!(cfg::MMAConfig, C, idx, d, layout) =
    ((r, c) = _rowcol(C, idx); _store_d!(cfg, C, r, c, d, layout))

@inline fill_c(cfg::MMAConfig, v)    = _fill_c(cfg, v)
@inline mma(cfg::MMAConfig, a, b, c) = _mma(cfg, a, b, c)

"""
    mma_supported(backend, cfg::MMAConfig) -> Bool

Vrai si un chemin **hardware** existe pour cette config sur le device courant de
`backend` (le backend KernelAbstractions). Faux ⇒ l'API reste utilisable mais
passe par le fallback portable. Réponse grossière (par type/forme, pas par device
exact) — suffisante pour choisir une tuile côté appelant. Requête HÔTE (elle
interroge le device), pas utilisable dans un kernel.

Prend le backend en ARGUMENT pour la MÊME raison que `mma_shapes` : les tables de
formes de CUDA et d'AMD se recouvrent, donc les extensions déclaraient des
signatures IDENTIQUES (`mma_supported(::MMAConfig{16,16,16,Float16,Float32,MulAdd})`
existe des deux côtés). Ce sont des méthodes ordinaires : avec CUDA **et** AMDGPU
chargés, la seconde extension écrasait la première et `mma_supported` répondait
pour le mauvais vendeur. Avec le backend en 1er argument, elles SPÉCIALISENT au
lieu de s'écraser.
"""
mma_supported(::Any, ::MMAConfig) = false

"""
    mma_shapes(backend) -> Tuple of NamedTuple

Énumère les configs pour lesquelles un chemin **hardware** existe sur le device
courant de `backend` (le backend KernelAbstractions) : `(M, N, K, compute, acc)`. Requête HÔTE (elle interroge le device),
pas utilisable dans un kernel.

Existe pour que l'appelant (KernelForge) n'ait pas à redupliquer la connaissance
des formes : il choisit sa tuile dans cette liste au lieu de coder en dur une
table qui dériverait de celle-ci.

INVARIANT : tout ce que `mma_shapes(backend)` liste doit vérifier
`mma_supported(backend, cfg)`.
Ne jamais y ajouter une forme non testée sur hardware.

Prend le backend en ARGUMENT (et non zéro argument) pour que les extensions le
SPÉCIALISENT au lieu de l'écraser : une méthode de signature identique à celle du
défaut est un « method overwriting », interdit pendant la précompilation.
"""
mma_shapes(::Any) = ()

# Registre des formes hardware apportées par les extensions OPTIONNELLES (fp8 via
# DLFP8Types, côté AMD comme CUDA). Chaque entrée : (type de backend, forme
# NamedTuple `(M,N,K,compute,acc)`, prédicat de support HÔTE). `mma_shapes` de
# chaque backend y ajoute ses entrées, en plus de sa table native. Rempli à
# l'`__init__` des extensions (jamais à la précompilation : la mutation d'un global
# d'un autre module ne doit pas être figée dans l'image).
#
# On passe par ce registre plutôt que par `Base.get_extension` : depuis une
# extension, `parentmodule(@__MODULE__)` n'est pas `KernelIntrinsics`, donc
# get_extension renvoyait `nothing` et les formes fp8 n'étaient pas énumérées — et,
# contrairement à get_extension, le registre est vérifiable par le garde-fou local.
const _EXT_SHAPE_REGISTRY = Vector{Tuple{Type,NamedTuple,Any}}()

function _register_ext_shape!(backend::Type, shape::NamedTuple, supported)
    any(e -> e[1] === backend && e[2] === shape, _EXT_SHAPE_REGISTRY) && return nothing
    push!(_EXT_SHAPE_REGISTRY, (backend, shape, supported))
    return nothing
end

# Formes enregistrées pour ce type de backend dont le prédicat de support est vrai.
_ext_shapes(backend::Type) =
    NamedTuple[e[2] for e in _EXT_SHAPE_REGISTRY if e[1] === backend && e[3]()]

# ── Jeton matériel (backend token) ───────────────────────────────────────────
#
# POURQUOI CE JETON EXISTE — deux raisons, toutes deux dures :
#
# (1) CORRECTION. Les six points d'entrée `_load_a/_load_b/_load_c/_fill_c/
#     _mma/_store_d!` étaient installés par les extensions via des OVERLAYS
#     (`CUDA.@device_override` / `Base.Experimental.@overlay`). Un overlay sur
#     une fonction qui PRODUIT un fragment empêche l'inférence de voir à travers
#     l'appel : le fragment revient imprécisément typé, SROA n'arrive plus à
#     promouvoir l'accumulateur, et la valeur portée par la boucle-K dégénère en
#     `phi float [ undef, %preheader ]` (le chemin CUDA.WMMA direct, lui, émet
#     `0.000000e+00`). `undef` peut se matérialiser en 0 OU en n'importe quoi :
#     résultats SILENCIEUSEMENT faux, et le symptôme apparaît/disparaît au gré
#     de la forme du code, de la version de LLVM et du nombre de warps.
#
# (2) COLLISION. On ne peut PAS simplement supprimer les overlays : WMMA (CUDA)
#     et MFMA (AMD) définissent des signatures LITTÉRALEMENT identiques
#     (`_fill_c(::MMAConfig{16,16,16,Float16,Float32,MulAdd}, v)` existe des deux
#     côtés). En méthodes ordinaires elles s'écraseraient l'une l'autre, et la
#     méthode AMD (`::Type{ColMajor}`) étant plus spécifique que celle de CUDA
#     (`where {L}`), AMD gagnerait même sur un `CuArray`.
#
# La solution combine les deux : un JETON de backend résolu à la compilation.
# `_mma_hw()` reste le SEUL overlay du chemin MMA — et il renvoie un SINGLETON,
# donc il ne porte aucune donnée à laquelle un `undef` pourrait s'accrocher
# (même forme que `_mma_wave()`, qui fonctionne depuis toujours). Tout le reste
# devient des méthodes ORDINAIRES, discriminées par le jeton en 1er argument.
#
# RÈGLE GÉNÉRALE À RETENIR (valable pour tout le dépôt) : un overlay n'est requis
# que lorsque deux backends définiraient SINON la MÊME signature de méthode —
# c'est le cas de `_shfl`, `_vote`, `_sleep`, `fence`, `_dynlocalmem`, `_laneid`,
# `_warpsize`, qui doivent rester des overlays. Ce n'est PAS le cas dès qu'un
# argument distingue déjà les backends. Ne pas réintroduire d'overlay sur une
# fonction qui produit ou consomme un fragment.
#
# Côté HÔTE, `_mma_hw()` renvoie `NoHW()` (les overlays sont device-only) : le
# comportement hôte reste donc exactement celui d'avant — fallback portable.
abstract type MMAHW end
struct NoHW     <: MMAHW end   # fallback portable (et tout appel côté hôte)
struct NVIDIATC <: MMAHW end   # tensor cores NVIDIA (WMMA / mma.sync)
struct CDNAMFMA <: MMAHW end   # MFMA (CDNA/AMD)

@inline _mma_hw() = NoHW()

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

# Aiguillage : on résout le jeton matériel UNE fois, puis on dispatche dessus.
# Les extensions backend définissent des méthodes ORDINAIRES sur `::NVIDIATC` /
# `::CDNAMFMA` ; ci-dessous, le fallback portable sur `::NoHW`.
@inline _load_a(cfg::MMAConfig, A, row, col, layout) = _load_a(_mma_hw(), cfg, A, row, col, layout)
@inline _load_b(cfg::MMAConfig, B, row, col, layout) = _load_b(_mma_hw(), cfg, B, row, col, layout)
@inline _load_c(cfg::MMAConfig, C, row, col, layout) = _load_c(_mma_hw(), cfg, C, row, col, layout)
@inline _fill_c(cfg::MMAConfig, v)                   = _fill_c(_mma_hw(), cfg, v)
@inline _mma(cfg::MMAConfig, a, b, c)                = _mma(_mma_hw(), cfg, a, b, c)
@inline _store_d!(cfg::MMAConfig, C, row, col, d, layout) =
    _store_d!(_mma_hw(), cfg, C, row, col, d, layout)

# Le jeton d'un backend dont la config n'a PAS de chemin hardware doit retomber
# sur le fallback portable : sans ces méthodes, un CuArray avec une forme non
# tabulée (16×16×8 fp16, Tropical, …) lèverait une MethodError au lieu de
# dégrader gracieusement. Elles ne peuvent pas être ambiguës avec celles des
# extensions : celles-ci lient M,N,K,CT,AccT littéralement.
@inline _load_a(::MMAHW, cfg::MMAConfig, A, row, col, layout) = _load_a(NoHW(), cfg, A, row, col, layout)
@inline _load_b(::MMAHW, cfg::MMAConfig, B, row, col, layout) = _load_b(NoHW(), cfg, B, row, col, layout)
@inline _load_c(::MMAHW, cfg::MMAConfig, C, row, col, layout) = _load_c(NoHW(), cfg, C, row, col, layout)
@inline _fill_c(::MMAHW, cfg::MMAConfig, v)                   = _fill_c(NoHW(), cfg, v)
@inline _mma(::MMAHW, cfg::MMAConfig, a::Frag{MatrixA}, b::Frag{MatrixB}, c::Frag{Accumulator}) =
    _mma(NoHW(), cfg, a, b, c)
@inline _store_d!(::MMAHW, cfg::MMAConfig, C, row, col, d::Frag{Accumulator}, layout) =
    _store_d!(NoHW(), cfg, C, row, col, d, layout)

@inline _load_a(::NoHW, cfg::MMAConfig, A, row, col, layout) = _load_a_fb(cfg, A, row, col, layout, _mma_wave())
@inline _load_b(::NoHW, cfg::MMAConfig, B, row, col, layout) = _load_b_fb(cfg, B, row, col, layout, _mma_wave())
@inline _load_c(::NoHW, cfg::MMAConfig, C, row, col, layout) = _load_c_fb(cfg, C, row, col, layout, _mma_wave())
@inline _fill_c(::NoHW, cfg::MMAConfig, v)                   = _fill_c_fb(cfg, v, _mma_wave())
@inline _mma(::NoHW, cfg::MMAConfig, a::Frag{MatrixA}, b::Frag{MatrixB}, c::Frag{Accumulator}) =
    _mma_fb(cfg, a, b, c, _mma_wave())
@inline _store_d!(::NoHW, cfg::MMAConfig, C, row, col, d::Frag{Accumulator}, layout) =
    _store_d_fb!(cfg, C, row, col, d, layout, _mma_wave())

# Les corps ci-dessous reçoivent l'origine de tuile DÉJÀ décodée en (row, col)
# 1-based ; `r0 = row - 1`, `c0 = col - 1` sont de simples décalages. C'est la
# variante linéaire de l'API publique qui porte le coût du décodage, quand elle
# est utilisée.
#
# ── Layout ──────────────────────────────────────────────────────────────────
# `(row, col)` est TOUJOURS l'origine PHYSIQUE dans le tableau reçu ; le tag de
# layout dit comment la tuile s'y déplie. Pour un élément logique de coordonnées
# (u, v) dans la tuile :
#     ColMajor → tableau[r0 + u + 1, c0 + v + 1]
#     RowMajor → tableau[r0 + v + 1, c0 + u + 1]     (composantes échangées)
#
# Ce n'est pas un choix libre : c'est EXACTEMENT ce que fait déjà WMMA sur CUDA,
# où le pointeur est pris à l'origine physique et la leading dimension reste
# size(A,1) quel que soit le tag. RowMajor y signifie « l'élément (u,v) est à
# ptr[u*ld + v] » — soit tableau[r0+v+1, c0+u+1]. Le fallback et MFMA doivent
# suivre le hardware au bit près, sinon le contrat diverge selon le backend, ce
# qui est précisément le défaut qu'on corrige ici.
#
# Concrètement : une matrice logique M×K stockée row-major se présente comme un
# tableau Julia K×M, et se charge avec RowMajor.
@inline _swap(::Type{ColMajor}) = false
@inline _swap(::Type{RowMajor}) = true

# Ordonne les deux composantes d'index selon le layout (appelé au TEMPS MACRO).
_ax(swap, u, v) = swap ? (v, u) : (u, v)
# load_a : lane possède ses lignes (op_y + mb·mi), toutes les colonnes K.
@generated function _load_a_fb(::MMAConfig{M,N,K,CT,AccT,OP}, A, row, col, ::Type{LAY},
                               ::Val{WS}) where {M,N,K,CT,AccT,OP,LAY,WS}
    mb, _ = _lane_grid(M, N, WS)
    Mr = M ÷ mb
    L = Mr * K
    sw = _swap(LAY)
    els = Vector{Any}(undef, L)
    for i in 1:L
        mi = (i - 1) % Mr
        k  = (i - 1) ÷ Mr
        # coordonnées logiques dans la tuile : (m, k)
        u, v = _ax(sw, :(op_y + $(mb * mi)), :($k))
        els[i] = :($CT(@inbounds A[r0 + $u + 1, c0 + $v + 1]))
    end
    quote
        r0 = row - 1; c0 = col - 1
        op_y = (_laneid() - 1) % $mb
        Frag{MatrixA,$CT,$L}(($(els...),))
    end
end

# load_b : lane possède ses colonnes (op_x + nb·ni), toutes les lignes K.
@generated function _load_b_fb(::MMAConfig{M,N,K,CT,AccT,OP}, B, row, col, ::Type{LAY},
                               ::Val{WS}) where {M,N,K,CT,AccT,OP,LAY,WS}
    mb, nb = _lane_grid(M, N, WS)
    Nc = N ÷ nb
    L = K * Nc
    sw = _swap(LAY)
    els = Vector{Any}(undef, L)
    for j in 1:L
        k  = (j - 1) % K
        ni = (j - 1) ÷ K
        # coordonnées logiques dans la tuile : (k, n)
        u, v = _ax(sw, :($k), :(op_x + $(nb * ni)))
        els[j] = :($CT(@inbounds B[r0 + $u + 1, c0 + $v + 1]))
    end
    quote
        r0 = row - 1; c0 = col - 1
        op_x = (_laneid() - 1) ÷ $mb
        Frag{MatrixB,$CT,$L}(($(els...),))
    end
end

# load_c : lit l'accumulateur depuis la mémoire (mêmes indices que store_d!).
@generated function _load_c_fb(::MMAConfig{M,N,K,CT,AccT,OP}, C, row, col, ::Type{LAY},
                               ::Val{WS}) where {M,N,K,CT,AccT,OP,LAY,WS}
    mb, nb = _lane_grid(M, N, WS)
    Mr = M ÷ mb; Nc = N ÷ nb; L = Mr * Nc
    sw = _swap(LAY)
    els = Vector{Any}(undef, L)
    for idx0 in 1:L
        mi = (idx0 - 1) % Mr
        ni = (idx0 - 1) ÷ Mr
        # coordonnées logiques dans la tuile : (m, n)
        u, v = _ax(sw, :(op_y + $(mb * mi)), :(op_x + $(nb * ni)))
        els[idx0] = :($AccT(@inbounds C[r0 + $u + 1, c0 + $v + 1]))
    end
    quote
        r0 = row - 1; c0 = col - 1
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
            # Les opérandes sont promus dans le TYPE D'ACCUMULATION avant le produit.
            # Sans ça, `Int8 * Int8` se fait en Int8 et déborde dès que le produit
            # dépasse 127 — alors que le hardware (MFMA i8) accumule en i32. Pour les
            # flottants c'est déjà le comportement de la promotion, donc sans effet.
            e = :(acc($OP(), $AccT(a.x[$(mi + Mr * k + 1)]), $AccT(b.x[$(k + K * ni + 1)]), $e))
        end
        outs[idx0] = e
    end
    quote
        Frag{Accumulator,$AccT,$L}(($(outs...),))
    end
end

# store_d! : chaque lane écrit son sous-bloc (op_y+mb·mi, op_x+nb·ni).
@generated function _store_d_fb!(::MMAConfig{M,N,K,CT,AccT,OP}, C, row, col, d::Frag{Accumulator},
                                 ::Type{LAY}, ::Val{WS}) where {M,N,K,CT,AccT,OP,LAY,WS}
    mb, nb = _lane_grid(M, N, WS)
    Mr = M ÷ mb; Nc = N ÷ nb
    sw = _swap(LAY)
    sts = Any[]
    for ni in 0:(Nc - 1), mi in 0:(Mr - 1)
        # coordonnées logiques dans la tuile : (m, n) — miroir exact de _load_c_fb.
        u, v = _ax(sw, :(op_y + $(mb * mi)), :(op_x + $(nb * ni)))
        push!(sts, :(@inbounds C[r0 + $u + 1, c0 + $v + 1] = d.x[$(mi + Mr * ni + 1)]))
    end
    quote
        r0 = row - 1; c0 = col - 1
        op_y = (_laneid() - 1) % $mb
        op_x = (_laneid() - 1) ÷ $mb
        $(sts...)
        nothing
    end
end

# Codegen MFMA partagé (extension AMDGPU de base + extension fp8). Dormant tant
# qu'aucune extension AMD ne l'appelle.
include("mma_amd.jl")

end # module MMA
