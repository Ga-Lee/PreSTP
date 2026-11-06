"""Explicit CausVid experiment presets."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    preset: str = "5s"
    latent_frames: int = 21
    fps: int = 16
    spatial_threshold: float = 0.97
    temporal_threshold: float = 0.997
    reuse_gate: float = 0.50
    reanchor_period: int = 3
    attention_window: int = 21
    chunk_frames: int = 3
    context_noise: int = 0
    timestep_shift: float = 8.0
    denoising_steps: tuple = (1000, 750, 500, 250)
    prune_context: bool = True
    reuse_mask: bool = True

    def __post_init__(self):
        for value in (self.latent_frames, self.fps, self.attention_window, self.chunk_frames, self.context_noise):
            if type(value) is not int:
                raise ValueError("Frame counts, FPS, window, chunk size and context noise must be integers")
        expected_frames = {"5s": 21, "30s": 126}
        if self.preset not in expected_frames or self.latent_frames != expected_frames[self.preset]:
            raise ValueError("Use the 5s (21 latent frames) or 30s (126 latent frames) preset")
        if self.fps != 16 or self.chunk_frames != 3 or self.attention_window != 21:
            raise ValueError("This release supports 16 FPS, 3-frame chunks and a 21-frame KV window")
        for value in [self.spatial_threshold, self.temporal_threshold, self.reuse_gate]:
            if not 0 <= value <= 1:
                raise ValueError("Thresholds must be finite and in [0, 1]")
        if not isinstance(self.reanchor_period, int) or isinstance(self.reanchor_period, bool) or self.reanchor_period < 1:
            raise ValueError("reanchor_period must be a positive integer")
        if self.context_noise != 0 or self.timestep_shift != 8.0 or tuple(self.denoising_steps) != (1000, 750, 500, 250):
            raise ValueError("The CausVid paper schedule must not be changed implicitly")
        if not isinstance(self.prune_context, bool) or not isinstance(self.reuse_mask, bool):
            raise ValueError("prune_context and reuse_mask must be booleans")

    @property
    def pixel_frames(self):
        return (self.latent_frames - 1) * 4 + 1

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "denoising_steps" in values:
            values["denoising_steps"] = tuple(values["denoising_steps"])
        return cls(**values)

    @classmethod
    def for_preset(cls, name):
        if name not in {"5s", "30s"}:
            raise ValueError("Unknown preset")
        return cls(preset=name, latent_frames=21 if name == "5s" else 126)
