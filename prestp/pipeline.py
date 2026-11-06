"""CausVid checkpoint adapter for the paper's inference backend."""
from pathlib import Path

import torch

from prestp.backend.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausVidPipeline(torch.nn.Module):
    def __init__(self, model_dir, checkpoint, config, device="cuda"):
        super().__init__()
        model_dir = Path(model_dir)
        checkpoint = Path(checkpoint)
        required = ["config.json", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl"]
        for relative in required:
            if not (model_dir / relative).exists():
                raise FileNotFoundError(f"Missing Wan model asset: {relative}")
        if not checkpoint.is_file():
            raise FileNotFoundError("CausVid checkpoint not found")
        self.generator = WanDiffusionWrapper(model_dir=model_dir, timestep_shift=config.timestep_shift, is_causal=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "generator_ema" in state:
            state = state["generator_ema"]
        elif "generator" in state:
            state = state["generator"]
        else:
            raise ValueError("Expected a CausVid checkpoint containing generator or generator_ema")
        self.generator.load_state_dict(state, strict=True)
        del state
        self.text_encoder = WanTextEncoder(model_dir)
        self.vae = WanVAEWrapper(model_dir)
        self.scheduler = self.generator.get_scheduler()
        timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        self.denoising_step_list = timesteps[1000 - torch.tensor(config.denoising_steps, dtype=torch.long)]
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = config.chunk_frames
        self.generator.model.num_frame_per_block = config.chunk_frames
        self.local_attn_size = -1
        self.kv_cache1 = None
        if config.latent_frames > config.attention_window:
            self.local_attn_size = config.attention_window
            for block in self.generator.model.blocks:
                block.local_attn_size = config.attention_window
                block.self_attn.local_attn_size = config.attention_window
                block.self_attn.max_attention_size = config.attention_window * self.frame_seq_length
        self.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)

    def _initialize_kv_cache(self, batch_size, dtype, device):
        capacity = self.local_attn_size * self.frame_seq_length if self.local_attn_size != -1 else 32760
        self.kv_cache1 = [{
            "k": torch.zeros([batch_size, capacity, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, capacity, 12, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "unrope_prev_k": None,
            "unrope_prev_v": None,
        } for _ in range(self.num_transformer_blocks)]

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        self.crossattn_cache = [{
            "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.num_transformer_blocks)]
