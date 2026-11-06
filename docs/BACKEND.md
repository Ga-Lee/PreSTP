# Inference backend

## Components

| File | Responsibility |
|:--|:--|
| `prestp/backend/wan/token_ops.py` | Patch selection, indexed scattering, latent restoration, and rotary transforms |
| `prestp/backend/wan/sparse_cache.py` | Chunk reservation, selected KV updates, clean-context refresh, and rolling eviction |
| `prestp/backend/wan/causal_model.py` | Attention over the full context with selected queries and output restoration |
| `prestp/backend/wan_wrapper.py` | Patch-mask validation and latent prediction |

## Cache and restoration

- Explicit token indices select the active spatial patches.
- Reference restoration copies the preceding reference frame; cascade restoration propagates the preceding restored frame.
- Each new denoising chunk is prefilled from the last clean KV frame. Subsequent steps overwrite active positions while retaining cached values elsewhere.
- Clean-context refresh restores skipped KV entries sequentially across frames and records the final clean frame for the next chunk.
- Cache state is local to each video and resets between independent generations.
- Rolling eviction removes whole frames and shifts temporal rotary coordinates while preserving spatial coordinates and configured sink frames.
- With `restore=False`, the backend returns zero at skipped output positions; the sampler fills these positions using the spatial reference or last-computed clean estimate.

## Sources

The transformer, attention, text encoder, tokenizer, and VAE components originate from [Wan2.1](https://github.com/Wan-Video/Wan2.1). The causal model, wrappers, and flow scheduler are adapted from [Self-Forcing](https://github.com/guandeh17/Self-Forcing). The pruning implementation was developed with reference to [LIPAR / Streaming LIPAR](https://github.com/DennisMenn/lipar) recovery primitives. Upstream license texts are retained in `licenses/`, and source copyright notices are preserved. Model weights are downloaded separately.
