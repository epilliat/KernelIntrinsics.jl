# ============================================================================
# @sleep — spin-loop backoff hint
# ============================================================================

"""
    _sleep(n::Integer)

Backend dispatch target for [`@sleep`](@ref). The fallback (CPU / no backend
loaded) is a no-op; GPU backends override this with a hardware sleep
instruction.
"""
@inline _sleep(n::Integer) = nothing  # fallback / CPU backend: no-op

"""
    @sleep n

Hint the hardware to back off for approximately `n` units before continuing.
Intended for backoff inside busy-wait / spin loops (e.g. the decoupled-lookback
scan) so a stalled consumer stops hammering the memory subsystem while it waits
on a producer.

`n` is a unitless hint, not a precise duration:
- **AMD (amdgcn):** lowers to `s_sleep`, whose operand is an immediate in units
  of 64 cycles; the runtime `n` is mapped to the nearest power-of-two immediate
  in `1:64`.
- **CUDA (sm_70+):** lowers to `nanosleep.u32` with the operand passed straight
  through (nanoseconds, hardware-rounded).
- **CPU / other backends:** no-op.

This is a hint only — it does not affect results, only how politely a spin loop
waits.

```julia
@sleep 8
```
"""
macro sleep(n)
    return quote
        $_sleep($(esc(n)))
    end
end
