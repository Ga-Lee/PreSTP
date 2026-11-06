"""Paper sampling loops; True masks denote tokens selected for computation."""
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def patchify(lat):
    T, C, H, W = lat.shape
    lat = lat.reshape(T, C, H // 2, 2, W // 2, 2)
    return lat.permute(0, 2, 4, 1, 3, 5).reshape(T, (H // 2) * (W // 2), C * 4)


def cosine_keep_mask(est, prev_last, tau, channels=16, ref_mode="cascade"):
    x = est[0].float()
    T, C, H, W = x.shape
    Hp, Wp = H // 2, W // 2
    toks = patchify(x)
    if prev_last is not None:
        prev_tok = patchify(prev_last[0].permute(1, 0, 2, 3).float())[0]
    else:
        prev_tok = toks[0]
    if ref_mode == "reference":
        refs = prev_tok[None].expand(T, -1, -1)
    else:
        refs = torch.cat([prev_tok[None], toks[:-1]], dim=0)
    cos = F.cosine_similarity(toks, refs, dim=-1)
    keep = (cos < tau).reshape(T, Hp, Wp)

    keep_full = keep.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    keep_full = keep_full[None, :, None].expand(1, T, channels, H, W).contiguous()
    return keep_full


def cross_step_freeze(cur, prev, freeze_tau, channels=16):
    a = patchify(cur[0].float())
    b = patchify(prev[0].float())
    cos = F.cosine_similarity(a, b, dim=-1)
    T = cur.shape[1]
    H, W = cur.shape[-2], cur.shape[-1]
    Hp, Wp = H // 2, W // 2
    frozen = (cos > freeze_tau).reshape(T, Hp, Wp)
    frozen = frozen.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return frozen[None, :, None].expand(1, T, channels, H, W).contiguous()


def ensure_nonempty_keep(mask):
    if bool(mask.any()):
        return mask
    mask = mask.clone()
    h0 = (mask.shape[-2] // 4) * 2
    w0 = (mask.shape[-1] // 4) * 2
    mask[:, :, :, h0:h0 + 2, w0:w0 + 2] = True
    return mask


@torch.no_grad()
def generate_combined(pipeline, config, prompt, *, seed=0, num_frames=21, tau=0.97,
                      freeze_tau=0.997, chunk0_dense=True, prune_context=False,
                      ctx_clean=False, step0_reuse=False, step0_gate=0.50, step0_reanchor=2,
                      static_fill="step0", height=60, width=104, channels=16):
    device = next(pipeline.generator.parameters()).device
    nfpb = pipeline.num_frame_per_block
    B = 1
    set_seed(seed)
    noise = torch.randn([B, num_frames, channels, height, width], device=device, dtype=torch.bfloat16)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])

    pipeline.generator.flow_prev = None
    pipeline.generator.noisy_prev = None
    pipeline.kv_cache1 = None
    pipeline._initialize_kv_cache(batch_size=B, dtype=noise.dtype, device=device)
    pipeline._initialize_crossattn_cache(batch_size=B, dtype=noise.dtype, device=device)

    output = torch.zeros([B, num_frames, channels, height, width], device=device, dtype=noise.dtype)
    steps = pipeline.denoising_step_list
    n = len(steps)
    num_blocks = num_frames // nfpb
    all_true = torch.ones([B, nfpb, channels, height, width], dtype=torch.bool, device=device)

    def gen_call(noisy, timestep, mask, update_prev, csf, restore):
        return pipeline.generator(
            noisy_image_or_video=noisy, conditional_dict=conditional_dict, timestep=timestep,
            kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
            current_start=csf * pipeline.frame_seq_length,
            is_prune=True, is_unprune_kv=True, from_memory=True,
            prune_mask=mask, update_prev=update_prev, fill_mode="reference", restore=restore)

    prev_frame = None
    prev_keep1 = None
    prev_prune = 0.0
    a1_fracs, a2_fracs = [], []
    _synchronize(device); t0 = time.perf_counter()
    csf = 0
    for blk in range(num_blocks):
        cnf = nfpb
        sl = slice(csf, csf + cnf)
        noisy = noise[:, sl]
        active = (blk >= 1) if chunk0_dense else True
        keep1 = None
        frozen = torch.zeros([B, cnf, channels, height, width], dtype=torch.bool, device=device)
        prev_b = prev_frame.transpose(1, 2).expand(B, cnf, channels, height, width) if prev_frame is not None else None


        reuse_step0 = (step0_reuse and active and prev_keep1 is not None
                       and prev_prune >= step0_gate
                       and not (step0_reanchor > 0 and ((blk - 1) % step0_reanchor == 0)))
        x0_buf = None
        x0_step0 = None
        for index, ct in enumerate(steps):
            ts = torch.ones([B, cnf], device=device, dtype=torch.int64) * ct
            last = (index == n - 1)
            if index == 0 and reuse_step0:
                keep1 = prev_keep1
                keep = ensure_nonempty_keep(keep1 & (~frozen))
                _, denoised = gen_call(noisy, ts, keep, last, csf, restore=False)
                denoised = torch.where(~keep1, prev_b, denoised)
            else:
                do_drop = active and index >= 1 and keep1 is not None
                if do_drop:
                    keep = ensure_nonempty_keep(keep1 & (~frozen))
                    _, denoised = gen_call(noisy, ts, keep, last, csf, restore=False)
                    denoised = torch.where(frozen, x0_buf, denoised)


                    static_src = x0_step0 if (static_fill == "step0" and x0_step0 is not None) else prev_b


                    if static_src is None:
                        static_src = x0_step0 if x0_step0 is not None else denoised
                    denoised = torch.where(~keep1, static_src, denoised)
                else:
                    _, denoised = gen_call(noisy, ts, all_true, last, csf, restore=True)
            if active and index == 0 and not reuse_step0:
                keep1 = ensure_nonempty_keep(
                    cosine_keep_mask(denoised, prev_frame, tau, channels, ref_mode="reference")
                )
                x0_step0 = denoised.clone()
            if active and index == 0 and keep1 is not None:
                a1_fracs.append(1.0 - keep1[:, :, 0, ::2, ::2].float().mean().item())
            if active and index >= 1 and x0_buf is not None:

                frozen = frozen | (cross_step_freeze(denoised, x0_buf, freeze_tau, channels) & keep1)
                a2_fracs.append((frozen & keep1)[:, :, 0, ::2, ::2].float().mean().item())
            x0_buf = denoised.clone()
            if not last:
                rn = torch.randn_like(denoised)
                noisy = pipeline.scheduler.add_noise(
                    denoised.flatten(0, 1), rn.flatten(0, 1),
                    steps[index + 1] * torch.ones([B * cnf], device=device, dtype=torch.long)
                ).unflatten(0, denoised.shape[:2])
        output[:, sl] = denoised
        ctx = torch.ones([B, cnf], device=device, dtype=torch.int64) * config.context_noise


        if prune_context and active and keep1 is not None:


            ctx_mask = ensure_nonempty_keep(
                cosine_keep_mask(denoised, prev_frame, tau, channels, ref_mode="reference")
            ) if ctx_clean else keep1
            gen_call(denoised, ctx, ctx_mask, False, csf, restore=False)
        else:
            gen_call(denoised, ctx, all_true, False, csf, restore=True)
        if active and keep1 is not None:
            prev_keep1 = keep1
            prev_prune = 1.0 - keep1[:, :, 0, ::2, ::2].float().mean().item()
        prev_frame = output[:, csf + cnf - 1: csf + cnf].transpose(1, 2).clone()
        csf += cnf
    _synchronize(device); dt = time.perf_counter() - t0
    a1 = sum(a1_fracs) / len(a1_fracs) if a1_fracs else 0.0
    a2 = sum(a2_fracs) / len(a2_fracs) if a2_fracs else 0.0
    return output, dt, a1, a2


@torch.no_grad()
def generate(pipeline, config, prompt, *, seed=0, num_frames=21, is_prune=False,
             tau=0.97, chunk0_dense=True, prune_context=False, fill_mode="reference",
             height=60, width=104, channels=16):


    ref_mode = "reference" if fill_mode == "reference" else "cascade"
    wrapper_fill = "cascade" if fill_mode == "cascade" else "reference"
    device = next(pipeline.generator.parameters()).device
    nfpb = pipeline.num_frame_per_block
    B = 1
    set_seed(seed)
    noise = torch.randn([B, num_frames, channels, height, width], device=device, dtype=torch.bfloat16)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])

    pipeline.generator.flow_prev = None
    pipeline.generator.noisy_prev = None
    pipeline.kv_cache1 = None
    pipeline._initialize_kv_cache(batch_size=B, dtype=noise.dtype, device=device)
    pipeline._initialize_crossattn_cache(batch_size=B, dtype=noise.dtype, device=device)

    output = torch.zeros([B, num_frames, channels, height, width], device=device, dtype=noise.dtype)
    steps = pipeline.denoising_step_list
    n = len(steps)
    num_blocks = num_frames // nfpb
    all_true = torch.ones([B, nfpb, channels, height, width], dtype=torch.bool, device=device)

    def gen_call(noisy, timestep, mask, update_prev, csf):
        kw = dict(noisy_image_or_video=noisy, conditional_dict=conditional_dict, timestep=timestep,
                  kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                  current_start=csf * pipeline.frame_seq_length)
        if is_prune:
            kw.update(is_prune=True, is_unprune_kv=True, from_memory=True,
                      prune_mask=mask, update_prev=update_prev, fill_mode=wrapper_fill)
        return pipeline.generator(**kw)

    prev_frame = None
    mask_fracs = []
    _synchronize(device); t0 = time.perf_counter()
    csf = 0
    for blk in range(num_blocks):
        cnf = nfpb
        sl = slice(csf, csf + cnf)
        noisy = noise[:, sl]
        do_prune = bool(is_prune) and (blk >= 1 if chunk0_dense else True)
        mask_c = None
        for index, ct in enumerate(steps):
            ts = torch.ones([B, cnf], device=device, dtype=torch.int64) * ct
            last = (index == n - 1)
            use_pred = do_prune and index >= 1 and mask_c is not None
            mask = mask_c if use_pred else all_true
            _, denoised = gen_call(noisy, ts, mask, last, csf)
            if do_prune and index == 0:
                mask_c = cosine_keep_mask(denoised, prev_frame, tau, channels, ref_mode=ref_mode)
                mask_fracs.append(mask_c[:, :, 0, ::2, ::2].float().mean().item())
            if not last:
                rn = torch.randn_like(denoised)
                noisy = pipeline.scheduler.add_noise(
                    denoised.flatten(0, 1), rn.flatten(0, 1),
                    steps[index + 1] * torch.ones([B * cnf], device=device, dtype=torch.long)
                ).unflatten(0, denoised.shape[:2])
        output[:, sl] = denoised
        ctx = torch.ones([B, cnf], device=device, dtype=torch.int64) * config.context_noise
        ctx_mask = mask_c if (do_prune and prune_context and mask_c is not None) else all_true
        gen_call(denoised, ctx, ctx_mask, False, csf)
        prev_frame = output[:, csf + cnf - 1: csf + cnf].transpose(1, 2).clone()
        csf += cnf
    _synchronize(device); dt = time.perf_counter() - t0
    frac = sum(mask_fracs) / len(mask_fracs) if mask_fracs else 1.0
    return output, dt, frac
