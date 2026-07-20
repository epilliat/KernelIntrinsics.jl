# fp8/bf8 sur MFMA (CDNA3/gfx942) — chemin hardware de l'extension multi-déclencheur
# KernelIntrinsicsAMDGPUDLFP8TypesExt. Inclus UNIQUEMENT côté roc (DLFP8Types n'est
# une dépendance que de l'env de test roc), après tests/mma.jl : réutilise `run_tile`,
# `to_device`/`from_device`, `backend`, `warpsz`.
#
# Le sweep `mma_shapes` (dans tests/mma.jl) couvre déjà fp8 numériquement quand
# l'extension est chargée. Ici on ajoute un test de LAYOUT à entrées EXACTES : des
# petits entiers (0..4) sont exactement représentables en E4M3 comme en E5M2, donc
# le produit-accumulé (en Float32) est exact et se compare avec `==`. Un mauvais
# layout d'accumulateur déplacerait des éléments et casserait l'égalité — sans que
# le bruit d'arrondi fp8 puisse masquer quoi que ce soit.

import KernelIntrinsics.MMA as MMA
using DLFP8Types: Float8_E4M3FNUZ, Float8_E5M2FNUZ

@testset "MMA fp8/bf8 layout exact (MFMA)" begin
    @testset "$FT $(Mi)×$(Ni)×$(Ki)" for FT in (Float8_E4M3FNUZ, Float8_E5M2FNUZ),
                                          (Mi, Ni, Ki) in ((16, 16, 32), (32, 32, 16))
        cfg = MMA.MMAConfig{Mi,Ni,Ki,FT,Float32,MMA.MulAdd}()
        MMA.mma_supported(cfg) || continue
        Ah = FT.(Float32.(rand(0:4, Mi, Ki)))
        Bh = FT.(Float32.(rand(0:4, Ki, Ni)))
        C = to_device(zeros(Float32, Mi, Ni))
        run_tile(cfg, C, to_device(Ah), to_device(Bh), 0.0f0)
        ref = Float32.(Ah) * Float32.(Bh)   # entrées entières exactes ⇒ produit exact
        @test from_device(C) == ref
    end
end
