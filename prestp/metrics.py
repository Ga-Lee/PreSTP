"""Strict paired CausVid evaluation and optional decoded-video diagnostics.

``num_frames`` is the decoded pixel-frame count, not the latent-frame count.
Throughput is total decoded frames divided by total diffusion time; it excludes
decoding and file I/O. Fidelity uses a global SSIM diagnostic, not windowed SSIM.
"""

import argparse
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath


_METHODS = ("dense", "prestp")
_CORE_FIDELITY = ("lpips", "psnr", "ssim")
_TAIL_FIDELITY = ("early_lpips", "late_lpips")
_FIDELITY_KEYS = _CORE_FIDELITY + _TAIL_FIDELITY


def _number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _fidelity_record(record):
    flat = {key: record[key] for key in _FIDELITY_KEYS if key in record}
    if "fidelity" in record and flat:
        raise ValueError("use either nested or top-level fidelity, not both")
    if "fidelity" not in record and not flat:
        return None
    values = record.get("fidelity", flat)
    if record["method"] != "prestp":
        raise ValueError("fidelity must be attached to the prestp row")
    if not isinstance(values, dict):
        raise ValueError("fidelity must be an object")
    if set(values) - set(_FIDELITY_KEYS):
        raise ValueError("unknown fidelity field")
    if not all(key in values for key in _CORE_FIDELITY):
        raise ValueError("fidelity requires lpips, psnr, and ssim")
    if ("early_lpips" in values) != ("late_lpips" in values):
        raise ValueError("early_lpips and late_lpips must appear together")
    for key, value in values.items():
        _number(value, key, -1 if key == "ssim" else 0,
                1 if key == "ssim" else None)
    return dict(values)


def _validate_row(row):
    if not isinstance(row, dict):
        raise ValueError("each row must be an object")
    for key in ("sample_id", "method", "video", "preset", "config_digest",
                "prompt_digest", "checkpoint_sha256"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    checkpoint = row["checkpoint_sha256"]
    if len(checkpoint) != 64 or any(character not in "0123456789abcdef" for character in checkpoint):
        raise ValueError("checkpoint_sha256 must be a 64-character lowercase hexadecimal digest")
    if row["method"] not in _METHODS:
        raise ValueError("method must be dense or prestp")
    if row["preset"] not in ("5s", "30s"):
        raise ValueError("preset must be 5s or 30s")
    if type(row.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if type(row.get("num_frames")) is not int or row["num_frames"] <= 0:
        raise ValueError("num_frames must be a positive integer")
    timing = _number(row.get("diffusion_seconds"), "diffusion_seconds", 0)
    if timing == 0:
        raise ValueError("diffusion_seconds must be positive")
    video = row["video"]
    posix = PurePosixPath(video)
    windows = PureWindowsPath(video)
    if (posix.is_absolute() or windows.drive or windows.root
            or ".." in posix.parts or ".." in windows.parts
            or not posix.name or "\x00" in video or "://" in video):
        raise ValueError("video must be a relative file path without traversal")
    return _fidelity_record(row)


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _finite_sum(values, name):
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} is not finite") from error
    return _number(total, name)


def summarize(paths, expected_samples=None):
    """Read JSONL files and return strict, sample-ID-paired timing statistics.

    Pair metadata (including seed and required checkpoint_sha256) must agree.
    Checkpoint hashes must be 64-character lowercase hexadecimal digests.
    All pairs must share config_digest, preset, checkpoint_sha256 and num_frames.
    Prompts and seeds may vary across pairs. IDs are unique per method even
    across files. ``expected_samples`` counts pairs, not rows. Fidelity may be
    nested under ``fidelity`` or stored directly on a prestp row; when supplied,
    all three core scores are required. Missing fidelity is reported explicitly
    through ``fidelity.num_samples`` and never treated as a zero score.
    """
    if expected_samples is not None:
        if type(expected_samples) is not int or expected_samples <= 0:
            raise ValueError("expected_samples must be a positive integer")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = {method: {} for method in _METHODS}
    fidelity_records = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line, object_pairs_hook=_unique_object,
                                     parse_constant=_reject_constant)
                    scores = _validate_row(row)
                    method, sample_id = row["method"], row["sample_id"]
                    if sample_id in rows[method]:
                        raise ValueError(f"duplicate method/sample key: {method}/{sample_id}")
                    rows[method][sample_id] = row
                    if scores is not None:
                        fidelity_records[sample_id] = scores
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    dense_ids = set(rows["dense"])
    prestp_ids = set(rows["prestp"])
    if dense_ids != prestp_ids:
        raise ValueError(f"unpaired samples: missing dense={sorted(prestp_ids - dense_ids)}, "
                         f"missing prestp={sorted(dense_ids - prestp_ids)}")
    if not dense_ids:
        raise ValueError("no paired samples found")
    if expected_samples is not None and len(dense_ids) != expected_samples:
        raise ValueError(f"expected {expected_samples} samples, found {len(dense_ids)}")
    for sample_id in sorted(dense_ids):
        for key in ("seed", "preset", "config_digest", "prompt_digest", "num_frames",
                    "checkpoint_sha256"):
            if rows["dense"][sample_id][key] != rows["prestp"][sample_id][key]:
                raise ValueError(f"sample {sample_id}: mismatching {key}")
    reference_id = min(dense_ids)
    reference = rows["dense"][reference_id]
    for sample_id in sorted(dense_ids):
        for key in ("config_digest", "preset", "checkpoint_sha256", "num_frames"):
            if rows["dense"][sample_id][key] != reference[key]:
                raise ValueError(f"mixed {key} across pairs: samples {reference_id} and {sample_id}; "
                                 "summarize each evaluation setting separately")
    methods = {}
    for method in _METHODS:
        records = list(rows[method].values())
        seconds = _finite_sum((row["diffusion_seconds"] for row in records),
                              "total diffusion time")
        frames = sum(row["num_frames"] for row in records)
        try:
            throughput = frames / seconds
        except OverflowError as error:
            raise ValueError("pixel frame throughput is not finite") from error
        methods[method] = {
            "num_samples": len(records),
            "total_diffusion_seconds": seconds,
            "mean_latency_seconds": seconds / len(records),
            "total_pixel_frames": frames,
            "pixel_frames_per_second": _number(throughput, "pixel frame throughput"),
        }
    speedup = (methods["dense"]["total_diffusion_seconds"]
               / methods["prestp"]["total_diffusion_seconds"])
    means = {}
    counts = {}
    for key in _FIDELITY_KEYS:
        values = [scores[key] for scores in fidelity_records.values() if key in scores]
        if values:
            means[key] = _finite_sum((value / len(values) for value in values), key)
            counts[key] = len(values)
    return {
        "num_samples": len(dense_ids),
        "speedup": _number(speedup, "speedup"),
        "methods": methods,
        "fidelity": {
            "num_samples": len(fidelity_records),
            "means": means,
            "counts": counts,
            "pairs": {key: fidelity_records[key] for key in sorted(fidelity_records)},
        },
    }


def fidelity(dense, prestp, lpips_model, *, long=False):
    """Compute whole-video PSNR, global SSIM and frame-mean Alex LPIPS.

    Inputs are floating decoded RGB tensors [1, T, 3, H, W] in [0, 1]. Dense
    frames are moved internally to the prestp device without mutating inputs.
    Supply an initialized AlexNet LPIPS model in evaluation mode on the prestp
    device. This function never downloads or initializes model weights.
    LPIPS runs in batches of at most eight frames; concatenated per-frame
    distances preserve frame weighting, including the final partial batch.
    Set ``long=True`` to include first/last-quarter LPIPS (floor T/4 frames,
    with at least one frame). Torch is imported only when this function runs.
    """
    import torch

    if not isinstance(dense, torch.Tensor) or not isinstance(prestp, torch.Tensor):
        raise ValueError("videos must be torch tensors")
    if (dense.shape != prestp.shape or dense.ndim != 5 or dense.shape[0] != 1
            or dense.shape[2] != 3 or any(size == 0 for size in dense.shape)):
        raise ValueError("videos must have identical nonempty [1, T, 3, H, W] shapes")
    for video in (dense, prestp):
        if (not video.is_floating_point() or not torch.isfinite(video).all().item()
                or video.min().item() < 0 or video.max().item() > 1):
            raise ValueError("videos must contain finite floating values in [0, 1]")
    if getattr(lpips_model, "pnet_type", "alex") != "alex":
        raise ValueError("LPIPS model must use AlexNet")
    with torch.no_grad():
        dense_float = dense.to(device=prestp.device, dtype=torch.float32)
        prestp_float = prestp.float()
        mse = torch.mean((dense_float - prestp_float) ** 2).item()
        psnr = 99.0 if mse <= 1e-10 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
        dense_flat, prestp_flat = dense_float.flatten(), prestp_float.flatten()
        dense_mean, prestp_mean = dense_flat.mean(), prestp_flat.mean()
        dense_var = dense_flat.var(unbiased=False)
        prestp_var = prestp_flat.var(unbiased=False)
        covariance = ((dense_flat - dense_mean) * (prestp_flat - prestp_mean)).mean()
        luminance_constant, contrast_constant = 0.01 ** 2, 0.03 ** 2
        ssim = (((2 * dense_mean * prestp_mean + luminance_constant)
                 * (2 * covariance + contrast_constant))
                / ((dense_mean ** 2 + prestp_mean ** 2 + luminance_constant)
                   * (dense_var + prestp_var + contrast_constant))).item()
        frames = dense.shape[1]
        distance_batches = []
        for start in range(0, frames, 8):
            stop = min(start + 8, frames)
            batch_distances = lpips_model(dense_float[0, start:stop] * 2 - 1,
                                         prestp_float[0, start:stop] * 2 - 1)
            if (not isinstance(batch_distances, torch.Tensor) or batch_distances.ndim == 0
                    or batch_distances.shape[0] != stop - start
                    or batch_distances.numel() != stop - start):
                raise ValueError("LPIPS model must return one distance per frame")
            batch_distances = batch_distances.flatten()
            if (not torch.isfinite(batch_distances).all().item()
                    or (batch_distances < 0).any().item()):
                raise ValueError("LPIPS distances must be finite and nonnegative")
            distance_batches.append(batch_distances)
        distances = torch.cat(distance_batches)
        result = {"lpips": distances.mean().item(), "psnr": psnr, "ssim": ssim}
        if long:
            quarter = max(1, frames // 4)
            result["early_lpips"] = distances[:quarter].mean().item()
            result["late_lpips"] = distances[-quarter:].mean().item()
    _fidelity_record({"method": "prestp", "fidelity": result})
    return result


def main(argv=None):
    """Write a JSON summary, failing rather than dropping invalid input rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", metavar="INPUT")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int)
    args = parser.parse_args(argv)
    if args.output.resolve() in {Path(path).resolve() for path in args.inputs}:
        parser.error("output must not overwrite an input file")
    try:
        result = summarize(args.inputs, args.expected_samples)
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.write_text(serialized, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
