# fp8/bf8 hardware — AMD MFMA (CDNA3, types fnuz) ET NVIDIA mma.sync (Ada/Hopper,
# types fn). Inclus quand DLFP8Types est chargé (déclenche l'extension fp8 du
# backend). Réutilise `run_tile`/`to_device`/`from_device` de tests/mma.jl.
#
# AGNOSTIQUE DU BACKEND : les types + formes fp8 supportés viennent de
# `mma_shapes(backend)` — donc E4M3FNUZ/E5M2FNUZ sur AMD, E4M3FN/E5M2 sur CUDA,
# sans que ce fichier nomme la moindre variante. Test de LAYOUT à entrées EXACTES :
# petits entiers (0..4) représentables dans tous les formats fp8 ⇒ produit-accumulé
# exact (Float32) ⇒ comparaison `==`. Un mauvais layout casserait l'égalité sans que
# le bruit d'arrondi fp8 puisse le masquer.

using DLFP8Types   # déclenche l'extension fp8 du backend (sinon mma_shapes n'en liste aucune)

let fp8 = [s for s in MMA.mma_shapes(backend) if sizeof(s.compute) == 1 && s.compute !== Int8]
    @testset "MMA fp8/bf8 layout exact ($(length(fp8)) formes)" begin
        @testset "$(s.compute) $(s.M)×$(s.N)×$(s.K)" for s in fp8
            cfg = MMA.MMAConfig{s.M,s.N,s.K,s.compute,s.acc,MMA.MulAdd}()
            @test MMA.mma_supported(cfg)
            T = s.compute
            Ah = T.(Float32.(rand(0:4, s.M, s.K)))
            Bh = T.(Float32.(rand(0:4, s.K, s.N)))
            C = to_device(zeros(s.acc, s.M, s.N))
            run_tile(cfg, C, to_device(Ah), to_device(Bh), zero(s.acc))
            @test from_device(C) == Float32.(Ah) * Float32.(Bh)
        end
    end
end
