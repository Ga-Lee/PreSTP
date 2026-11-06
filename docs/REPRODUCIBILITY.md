# Reproduction protocol

## Supported configuration

| Parameter | 5-second | Nominal 30-second |
|:--|:--|:--|
| Model | CausVid autoregressive checkpoint | Same |
| Backend | Modified Self-Forcing/Wan | Same, rolling context |
| Spatial resolution | 832 × 480 | 832 × 480 |
| Latent frames | 21 | 126 |
| Decoded frames / FPS | 81 / 16 | 501 / 16 |
| Latent frames per chunk | 3 | 3 |
| Denoising schedule | Four warped steps: 1000, 750, 500, 250 | Same |
| Timestep shift | 8.0 | 8.0 |
| Context noise | 0 | 0 |
| KV capacity | 21 latent frames | 21 latent frames |
| Spatial / temporal thresholds | 0.97 / 0.997 | 0.97 / 0.997 |
| Mask-reuse gate / re-anchor period | 0.50 / 3 chunks | 0.50 / 3 chunks |

The duration labels follow the experiment names. Saved videos contain 81 or 501 pixel frames at 16 FPS, so their container durations are approximately 5.06 or 31.31 seconds, respectively.

The entire first chunk uses dense token computation. Subsequent chunks use cross-frame prediction, cross-step freezing, and content-adaptive mask reuse. Context refresh is also pruned. The spatial mask is anchored to the previous chunk's final frame. For newly predicted masks, spatially skipped positions use the current chunk's step-zero clean estimate; mask-reused step zero uses the previous clean frame. Frozen positions retain their last computed clean values. The sampler keeps at least one spatial patch per frame when a mask would otherwise be empty.

The reuse gate is evaluated using the preceding spatial mask's pruning fraction, not the union of spatial pruning and temporal freezing. Logged `spatial_pruned_ratio` excludes the initial dense chunk; `frozen_ratio` averages the frozen-within-spatial-keep fraction recorded at later steps. Neither is the full rollout's average skipped-token ratio, and they must not be added as though they used the same denominator and sampling schedule.

## Model configuration

Use the settings above when comparing with the paper's CausVid experiments. Token restoration and pruning-aware KV updates are described in the [backend documentation](BACKEND.md). Absolute-RoPE dense execution and pruning-aware rolling-cache handling use distinct implementation paths.

Base assets and checkpoints are downloaded separately. `run.json` records the loaded generator checkpoint's SHA-256 and software versions. Record the downloaded base-model revision and tokenizer revision separately when running a full benchmark.

## Prompts and random seeds

The paper uses 946 official VBench prompts and five seeds `[0, 1, 2, 3, 42]`, yielding 4,730 videos per method. Generation uses the corresponding extended prompts, while VBench identifies videos by the original benchmark prompt. Keep both fields aligned. A newly generated prompt expansion is not an exact reproduction of the original expansion.

The benchmark helper prepares and exports this mapping. Prompt data is not bundled. Matching the paper's conditioning requires the corresponding extended prompts; using original prompts is an alternative protocol.

```bash
python -m prestp.benchmark prepare \
  --full-info-json data/VBench_full_info.json \
  --extended-prompt-file data/extended_prompts.txt \
  --output data/vbench_prompts.jsonl
```

The extended file must have exactly one nonempty line per official prompt, in the same order. Omitting `--extended-prompt-file` uses original prompts and therefore changes the conditioning protocol.

**Unique filenames:** official prompt lists can contain duplicate names, causing collisions during prompt-based export. The helper rejects collisions rather than overwriting samples or silently deduplicating inputs. Resolve any duplicates consistently in the benchmark metadata and extended prompts before export. If rows are removed, update `--expected-samples` to match the resulting prompt count and seed count. Use a case-sensitive filesystem for official exports because case-only prompt differences can also occur.

For an aligned prompt file:

```bash
python -m prestp.infer \
  --wan-dir weights/Wan2.1-T2V-1.3B \
  --checkpoint weights/CausVid/autoregressive_checkpoint/model.pt \
  --prompts data/vbench_prompts.jsonl \
  --output outputs/vbench_5s --preset 5s \
  --method both --seeds 0,1,2,3,42 --fidelity

python -m prestp.metrics outputs/vbench_5s/metrics.jsonl \
  --expected-samples 4730 --output outputs/vbench_5s/summary.json
```

Independent shards may use `--num-shards N --shard I`, with one output directory per shard. Pass all shard manifests to the metrics command. IDs and seeds are preserved; duplicates and missing dense/PreSTP pairs are rejected. Do not merge different presets, checkpoints, or threshold configurations into a single reported setting. Do not run competing timing jobs on the same device.

## Timing

Each sample's timer begins after prompt encoding, noise/output allocation, and cache initialization. CUDA synchronization brackets the complete denoising loop and context-refresh calls. Decoding, encoding, fidelity evaluation, and file I/O are excluded. One full-length untimed sample is run per selected method before measurements.

For matched samples, aggregate speedup is `sum(dense diffusion seconds) / sum(PreSTP diffusion seconds)`, equivalently the ratio of mean latencies. Do **not** use the mean of per-sample speedup ratios. Throughput is total decoded-frame count divided by total diffusion time, not end-to-end playback throughput. Preserve unrounded measurements and round only for presentation.

## Fidelity

Optional `--fidelity` operates on decoded floating-point videos **before** H.264 compression. It compares matched prompts and seeds:

- LPIPS: AlexNet backbone, frame-wise mean, inputs mapped from `[0,1]` to `[-1,1]`.
- PSNR: one MSE over the entire video; returns 99 when MSE is at most `1e-10`.
- SSIM: a **global** mean/variance/covariance diagnostic, with biased variances and constants `0.01²` and `0.03²`; it is not sliding-window SSIM.
- Long-video early/late LPIPS: first and last quarter of the decoded frames, respectively.

LPIPS frames are evaluated in batches of at most eight to bound activation memory; frame distances are concatenated before averaging, including a smaller final batch.

Fidelity metrics are separate from VBench. Never infer a VBench score from LPIPS or from the small example prompts.

## VBench and VBench-Long

Use the [official VBench repository](https://github.com/Vchitect/VBench) and its original 16-dimension evaluation for 5-second videos; use its `vbench2_beta_long` protocol for long videos. Do not substitute VBench-2.0, which is a different benchmark. Use all dimensions and official normalization/aggregation, preserving full-precision raw results. Record the evaluator revision and model assets with the experiment.

The export helper names files using the original prompt and sample index (`0` through `4`, corresponding to the seed order above), not the expanded generation text. It requires complete five-seed pairs and rejects unsafe filenames instead of silently rewriting benchmark identifiers.

```bash
python -m prestp.benchmark export \
  --metrics outputs/vbench_5s/metrics.jsonl --prompts data/vbench_prompts.jsonl \
  --method prestp --output outputs/vbench_5s_prestp_export
python -m prestp.benchmark export \
  --metrics outputs/vbench_5s/metrics.jsonl --prompts data/vbench_prompts.jsonl \
  --method dense --output outputs/vbench_5s_dense_export
```

These commands use an unsharded run. Export currently accepts one run manifest; keep sharded timing aggregation separate from benchmark export, and do not concatenate manifests whose relative video paths resolve in different directories.

Follow the official evaluator's standard-mode instructions after export. The release does not vendor evaluation models or silently compute a partial score as the full VBench total.
