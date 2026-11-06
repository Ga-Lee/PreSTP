"""Generate matched CausVid dense/PreSTP videos without cluster dependencies."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from prestp.config import Config
from prestp.inputs import parse_seeds, prompt_digest, read_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True, help="UTF-8 text or JSONL with id and prompt")
    parser.add_argument("--output", required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--preset", choices=["5s", "30s"], default="5s")
    parser.add_argument("--config", help="Explicit JSON configuration; overrides --preset")
    parser.add_argument("--method", choices=["dense", "prestp", "both"], default="both")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--fidelity", action="store_true", help="Compute paired metrics before video encoding")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without loading a model")
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config.for_preset(args.preset)
    prompts = read_prompts(args.prompts)
    seeds = parse_seeds(args.seeds)
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        parser.error("Invalid shard configuration")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        prompts = prompts[:args.limit]
    prompts = [item for index, item in enumerate(prompts) if index % args.num_shards == args.shard]
    if not prompts:
        parser.error("Selected shard contains no prompts")
    if args.fidelity and args.method != "both":
        parser.error("--fidelity requires --method both")
    if args.check_only:
        print(json.dumps({"config": asdict(config), "prompts": len(prompts), "seeds": seeds, "pixel_frames": config.pixel_frames}, indent=2))
        return
    output = Path(args.output)
    if output.exists():
        parser.error("Output directory already exists; select a new directory")

    import torch
    from torchvision.io import write_video
    from prestp.pipeline import CausVidPipeline
    from prestp.sampling import generate, generate_combined

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("A CUDA device with BF16 support is required")
    import flash_attn

    pipeline = CausVidPipeline(args.wan_dir, args.checkpoint, config)
    lpips_model = None
    if args.fidelity:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").eval().cuda()
    methods = ["dense", "prestp"] if args.method == "both" else [args.method]

    def sample(method, prompt, seed):
        if method == "dense":
            latents, seconds, _ = generate(pipeline, config, prompt, seed=seed, num_frames=config.latent_frames, is_prune=False)
            ratios = {}
        else:
            latents, seconds, spatial, frozen = generate_combined(
                pipeline, config, prompt, seed=seed, num_frames=config.latent_frames,
                tau=config.spatial_threshold, freeze_tau=config.temporal_threshold,
                chunk0_dense=True, prune_context=config.prune_context, ctx_clean=False,
                step0_reuse=config.reuse_mask, step0_gate=config.reuse_gate,
                step0_reanchor=config.reanchor_period, static_fill="step0")
            ratios = {"spatial_pruned_ratio": spatial, "frozen_ratio": frozen}
        if not torch.isfinite(latents).all():
            raise RuntimeError("Non-finite restored latents; refusing to save this sample")
        return latents, seconds, ratios

    with torch.inference_mode():
        for method in methods:
            warm_latents, _, _ = sample(method, prompts[0]["prompt"], seeds[0])
            del warm_latents
        output.mkdir(parents=True, exist_ok=False)
        fingerprint = hashlib.sha256()
        with Path(args.checkpoint).open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
                fingerprint.update(chunk)
        metadata = {
            "config": asdict(config), "config_digest": config.digest(),
            "checkpoint_sha256": fingerprint.hexdigest(),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "flash_attention_version": getattr(flash_attn, "__version__", "unknown"),
            "seed_order": seeds, "shard": args.shard, "num_shards": args.num_shards,
            "timing": "denoising and context refresh only; one untimed warm-up per method",
        }
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with (output / "metrics.jsonl").open("x", encoding="utf-8") as manifest:
            for item in prompts:
                for seed in seeds:
                    dense_pixels = None
                    for method in methods:
                        latents, seconds, ratios = sample(method, item["prompt"], seed)
                        pixels = (pipeline.vae.decode_to_pixel(latents, use_cache=False) * 0.5 + 0.5).clamp(0, 1)
                        del latents
                        if pixels.shape[1] != config.pixel_frames or not torch.isfinite(pixels).all():
                            raise RuntimeError("Invalid decoded video")
                        extra_metrics = {}
                        if args.fidelity:
                            if method == "dense":
                                dense_pixels = pixels.cpu()
                            else:
                                from prestp.metrics import fidelity
                                extra_metrics = fidelity(dense_pixels, pixels, lpips_model, long=config.preset == "30s")
                        relative = Path(method) / f"{item['id']}-seed{seed}.mp4"
                        target = output / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        video = (pixels[0].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu()
                        write_video(str(target), video, fps=config.fps, video_codec="libx264", options={"crf": "18"})
                        record = {
                            "sample_id": f"{item['id']}-seed{seed}", "prompt_id": item["id"],
                            "seed": seed, "method": method,
                            "diffusion_seconds": seconds, "num_frames": int(pixels.shape[1]),
                            "video": relative.as_posix(), "preset": config.preset,
                            "config_digest": config.digest(), "prompt_digest": prompt_digest(item["prompt"]),
                            "checkpoint_sha256": fingerprint.hexdigest(),
                            **ratios, **extra_metrics,
                        }
                        manifest.write(json.dumps(record, allow_nan=False) + "\n")
                        manifest.flush()
                        print(f"{item['id']} seed={seed} {method}: {seconds:.3f} diffusion seconds", flush=True)
                        del pixels, video
                    del dense_pixels


if __name__ == "__main__":
    main()
