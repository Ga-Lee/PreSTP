"""Prepare prompts and export videos for external, official VBench evaluation.

Usage (from the release root)::

    python -m prestp.benchmark prepare --full-info-json /data/VBench_full_info.json \
        --extended-prompt-file /data/extended.txt --output prompts.jsonl
    python -m prestp.infer --prompts prompts.jsonl --method both \
        --seeds 0,1,2,3,42 --output run [other inference arguments]
    python -m prestp.benchmark export --metrics run/metrics.jsonl \
        --prompts prompts.jsonl --method prestp --output vbench_videos

The official API matches literal ``{prompt_en}-{index}.mp4`` filenames:
https://github.com/Vchitect/VBench/blob/master/vbench/__init__.py
Original prompt text is used only for naming; extended text is conditioning.
Unsafe names are rejected rather than rewritten. Nonempty extended-file lines
must align exactly by position.
Duplicate official names are rejected with their row IDs: resolve them explicitly
with the same selection in both original and extended inputs, never independently
deduplicate either list. Case-only variants require a case-sensitive export
filesystem. No automatic first/last-wins choice of conditioning is made.

Export requires both methods and all five seeds for every supplied prompt.
Paths are relative to the metrics file's directory; symlinks are not followed.
If present, runner run.json is checked too. Digests establish consistency, not
checkpoint authenticity. Videos are copied, not decoded or evaluated. No scores
or official datasets are generated, downloaded, or included in the export.
Pass the resulting MP4-only directory and the original full-info JSON to the
separately installed official VBench evaluator in vbench_standard mode.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import stat
import unicodedata

from prestp.config import Config
from prestp.inputs import prompt_digest


SEED_ORDER = (0, 1, 2, 3, 42)
METHODS = ("dense", "prestp")


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _loads(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def _read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = _loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Each JSONL record must be an object")
                records.append(record)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"Empty JSONL file: {path}")
    return records


def _unsafe_character(text):
    return any(unicodedata.category(character).startswith("C")
               or unicodedata.category(character) in ("Zl", "Zp")
               for character in text)


def video_name(benchmark_prompt, seed):
    """Return a literal Linux filename; printable punctuation is not rewritten."""
    if type(seed) is not int or seed not in SEED_ORDER:
        raise ValueError(f"Seed must be one of {SEED_ORDER}")
    if (not isinstance(benchmark_prompt, str) or not benchmark_prompt
            or benchmark_prompt != benchmark_prompt.strip()
            or benchmark_prompt in (".", "..")
            or any(character in benchmark_prompt for character in "/\\")
            or _unsafe_character(benchmark_prompt)):
        raise ValueError("Unsafe benchmark prompt for literal VBench filename")
    filename = f"{benchmark_prompt}-{SEED_ORDER.index(seed)}.mp4"
    if len(filename.encode("utf-8")) > 255:
        raise ValueError("VBench filename exceeds 255 UTF-8 bytes")
    return filename


def _collision_key(filename):
    return unicodedata.normalize("NFC", filename)


def _validate_prompts(records):
    prompts = {}
    names = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each prompt must be an object")
        prompt_id = record.get("id")
        if not isinstance(prompt_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", prompt_id):
            raise ValueError("Prompt IDs must be 1-80 letters, numbers, underscores or hyphens")
        if prompt_id in prompts:
            raise ValueError(f"Duplicate prompt ID: {prompt_id}")
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Conditioning prompt must be a nonempty string")
        benchmark_prompt = record.get("benchmark_prompt")
        name = _collision_key(video_name(benchmark_prompt, SEED_ORDER[0]))
        if name in names:
            raise ValueError(f"Benchmark filename collision: prompt IDs {names[name]} and {prompt_id} "
                             "(Unicode normalization included); resolve aligned inputs explicitly")
        names[name] = prompt_id
        prompts[prompt_id] = {"id": prompt_id, "prompt": prompt.strip(),
                              "benchmark_prompt": benchmark_prompt}
    if not prompts:
        raise ValueError("At least one prompt is required")
    return prompts


def prepare(full_info_json, output, extended_prompt_file=None):
    """Write a new runner-compatible JSONL without changing official inputs."""
    full_info = _loads(Path(full_info_json).read_text(encoding="utf-8"))
    if not isinstance(full_info, list) or not full_info:
        raise ValueError("Full-info JSON must be a nonempty list")
    originals = []
    for index, item in enumerate(full_info):
        if not isinstance(item, dict) or "prompt_en" not in item:
            raise ValueError(f"Full-info entry {index} requires prompt_en")
        video_name(item["prompt_en"], SEED_ORDER[0])
        originals.append(item["prompt_en"])
    conditioning = originals
    if extended_prompt_file is not None:
        conditioning = [line.strip() for line in
                        Path(extended_prompt_file).read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        if len(conditioning) != len(originals):
            raise ValueError(f"Extended prompt count {len(conditioning)} must equal "
                             f"full-info count {len(originals)}; alignment is positional")
    records = [{"id": f"{index:05d}", "prompt": prompt, "benchmark_prompt": original}
               for index, (original, prompt) in enumerate(zip(originals, conditioning))]
    _validate_prompts(records)
    serialized = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                         for record in records).encode("utf-8")
    with Path(output).open("xb") as destination:
        destination.write(serialized)
    return len(records)


def _relative_video(value):
    if (not isinstance(value, str) or not value or _unsafe_character(value)
            or "\\" in value or ":" in value or value.startswith("/")
            or PureWindowsPath(value).drive
            or any(part in ("", ".", "..") for part in value.split("/"))
            or Path(value).suffix.lower() != ".mp4"):
        raise ValueError("Video must be a relative MP4 path under the input run, without traversal")
    return value.split("/")


@contextmanager
def _open_video(root, relative):
    """Walk directory descriptors so symlink substitutions cannot escape root."""
    parts = _relative_video(relative)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(root, flags)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
        with os.fdopen(descriptor, "rb") as source:
            status = os.fstat(source.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_size == 0:
                raise ValueError(f"Video must be a nonempty regular file: {relative}")
            yield source
    finally:
        os.close(directory)


def _identity(source):
    status = os.fstat(source.fileno())
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _validate_run_metadata(root, reference):
    path = root / "run.json"
    if not os.path.lexists(path):
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError("run.json must be a regular, non-symlink file")
    metadata = _loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), dict):
        raise ValueError("run.json must contain a config object")
    try:
        config = Config(**metadata["config"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid run.json config: {error}") from error
    for key in ("checkpoint_sha256", "config_digest"):
        if metadata.get(key) != reference[key]:
            raise ValueError(f"run.json mismatching {key}")
    if (config.digest() != reference["config_digest"]
            or config.preset != reference["preset"]
            or config.pixel_frames != reference["num_frames"]):
        raise ValueError("run.json config/preset does not match metrics")
    seeds = metadata.get("seed_order")
    if (not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds)
            or len(seeds) != len(SEED_ORDER) or set(seeds) != set(SEED_ORDER)):
        raise ValueError("run.json must contain all five distinct benchmark seeds")


def _check_case_collisions(destination, prompts):
    groups = {}
    for prompt in prompts.values():
        filename = video_name(prompt["benchmark_prompt"], SEED_ORDER[0])
        groups.setdefault(_collision_key(filename).casefold(), []).append(filename)
    for names in groups.values():
        if len(names) < 2:
            continue
        created = []
        try:
            for filename in names:
                path = destination / filename
                try:
                    with path.open("xb"):
                        pass
                except FileExistsError as error:
                    raise ValueError("Benchmark filename collision on this filesystem; "
                                     "case-only prompt variants need a case-sensitive output") from error
                created.append(path)
        finally:
            for path in created:
                path.unlink()


def export(metrics, prompts, method, output):
    """Preflight complete paired records, then copy one method to a fresh directory."""
    if method not in METHODS:
        raise ValueError("Method must be dense or prestp")
    metrics_path = Path(metrics)
    if metrics_path.is_symlink() or not metrics_path.is_file():
        raise ValueError("Metrics must be a regular, non-symlink file")
    root = metrics_path.parent.resolve(strict=True)
    destination = Path(output)
    if os.path.lexists(destination):
        raise ValueError("Output must be a fresh directory; existing data is never overwritten")
    target = destination.resolve()
    if target.is_relative_to(root):
        raise ValueError("Output must be outside the input run")
    prompt_map = _validate_prompts(_read_jsonl(prompts))
    rows = _read_jsonl(metrics_path)
    matched = {}
    reference = rows[0]
    sources = set()
    plans = {}
    for row in rows:
        prompt_id, seed = row.get("prompt_id"), row.get("seed")
        if not isinstance(prompt_id, str) or prompt_id not in prompt_map:
            raise ValueError("Unknown or missing prompt_id in metrics")
        if type(seed) is not int or seed not in SEED_ORDER:
            raise ValueError(f"Metrics seed must be one of {SEED_ORDER}")
        if row.get("sample_id") != f"{prompt_id}-seed{seed}":
            raise ValueError("sample_id must equal '{prompt_id}-seed{seed}'")
        if row.get("method") not in METHODS:
            raise ValueError("Metrics method must be dense or prestp")
        key = prompt_id, seed, row["method"]
        if key in matched:
            raise ValueError(f"Duplicate sample/method: {key}")
        for field in ("prompt_digest", "checkpoint_sha256", "config_digest"):
            value = row.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if row["prompt_digest"] != prompt_digest(prompt_map[prompt_id]["prompt"]):
            raise ValueError(f"Conditioning prompt_digest mismatch: {prompt_id}")
        if row.get("preset") not in ("5s", "30s"):
            raise ValueError("Metrics preset must be 5s or 30s")
        if (type(row.get("num_frames")) is not int
                or row["num_frames"] != {"5s": 81, "30s": 501}[row["preset"]]):
            raise ValueError("num_frames does not match preset")
        for field in ("checkpoint_sha256", "config_digest", "preset", "num_frames"):
            if row[field] != reference.get(field):
                raise ValueError(f"Inconsistent {field} across metrics")
        relative = row.get("video")
        with _open_video(root, relative) as source:
            identity = _identity(source)
        if identity[:2] in sources:
            raise ValueError("Video collision: multiple records reference the same file")
        sources.add(identity[:2])
        matched[key] = row
        plans[key] = relative, identity
    expected = {(prompt_id, seed, pair_method) for prompt_id in prompt_map
                for seed in SEED_ORDER for pair_method in METHODS}
    if set(matched) != expected:
        raise ValueError(f"Incomplete paired benchmark: missing {len(expected - set(matched))} "
                         "records; every prompt needs both methods and seeds 0,1,2,3,42")
    _validate_run_metadata(root, reference)
    destination.mkdir(mode=0o700, exist_ok=False)
    try:
        _check_case_collisions(destination, prompt_map)
        for prompt_id, prompt in prompt_map.items():
            for seed in SEED_ORDER:
                relative, identity = plans[prompt_id, seed, method]
                filename = video_name(prompt["benchmark_prompt"], seed)
                with _open_video(root, relative) as source:
                    if _identity(source) != identity:
                        raise ValueError(f"Video changed after validation: {relative}")
                    with (destination / filename).open("xb") as copied:
                        shutil.copyfileobj(source, copied)
                    if _identity(source) != identity:
                        raise ValueError(f"Video changed during copy: {relative}")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return len(prompt_map) * len(SEED_ORDER)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Prepare conditioning and literal benchmark prompts")
    preparation.add_argument("--full-info-json", required=True, type=Path)
    preparation.add_argument("--extended-prompt-file", type=Path)
    preparation.add_argument("--output", required=True, type=Path)
    exporting = commands.add_parser("export", help="Export complete five-seed paired runs; no evaluation")
    exporting.add_argument("--metrics", required=True, type=Path)
    exporting.add_argument("--prompts", required=True, type=Path)
    exporting.add_argument("--method", required=True, choices=METHODS)
    exporting.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            count = prepare(args.full_info_json, args.output, args.extended_prompt_file)
            print(f"Prepared {count} prompts: {args.output}")
        else:
            count = export(args.metrics, args.prompts, args.method, args.output)
            print(f"Exported {count} videos: {args.output} (official evaluation not run)")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
