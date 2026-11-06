"""Prompt validation independent of GPU dependencies."""
import hashlib
import json
import re
from pathlib import Path


def read_prompts(path):
    source = Path(path)
    prompts = []
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if source.suffix == ".jsonl":
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError("Each JSONL prompt must have a string id")
            sample_id, prompt = item["id"], item.get("prompt")
        else:
            sample_id, prompt = f"{index:05d}", line.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", sample_id):
            raise ValueError("Prompt IDs must be 1-80 letters, numbers, underscores or hyphens")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Each prompt must be a nonempty string")
        prompts.append({"id": sample_id, "prompt": prompt.strip()})
    if not prompts or len({item["id"] for item in prompts}) != len(prompts):
        raise ValueError("Prompt file must be nonempty and IDs must be unique")
    return prompts


def prompt_digest(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def parse_seeds(value):
    seeds = [int(part.strip()) for part in value.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("Seeds must be distinct integers in [0, 2**32)")
    return seeds
