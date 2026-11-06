"""Explicit token selection, latent restoration, and rotary coordinates."""
import math

import torch


def selected_indices(keep):
    if keep.dtype != torch.bool or keep.ndim != 2 or keep.shape[0] != 1:
        raise ValueError("Token selection requires a boolean [1, tokens] mask")
    return torch.nonzero(keep[0], as_tuple=True)[0]


def expand_tokens(values, indices, length):
    if values.shape[0] != 1 or values.shape[1] != indices.numel():
        raise ValueError("Computed tokens and selected indices must agree")
    output = values.new_zeros((1, length, *values.shape[2:]))
    return output.index_copy(1, indices, values)


def latent_token_mask(keep, patch_size):
    if keep.ndim != 5 or keep.dtype != torch.bool or keep.shape[0] != 1:
        raise ValueError("Latent selection requires a boolean [1, frames, channels, height, width] mask")
    temporal, height, width = patch_size
    if temporal != 1 or keep.shape[-2] % height or keep.shape[-1] % width:
        raise ValueError("The latent grid must align with spatial patches")
    tokens = keep[:, :, :1, ::height, ::width]
    expanded = tokens.repeat_interleave(height, -2).repeat_interleave(width, -1)
    if not torch.equal(keep, expanded.expand_as(keep)):
        raise ValueError("Keep decisions must be constant within each patch and across channels")
    return tokens.flatten(1)


def restore_frames(values, keep, previous=None, mode="reference"):
    """Fill omitted positions from a reference frame or the preceding restored frame.

    Values have shape [batch, frames, ...]; keep broadcasts to that shape.
    Previous, when provided, has shape [batch, 1, ...].
    """
    if mode not in ("reference", "cascade"):
        raise ValueError("Restoration mode must be reference or cascade")
    if values.ndim < 3 or values.shape[1] == 0 or keep.dtype != torch.bool:
        raise ValueError("Restoration requires a nonempty frame grid and boolean mask")
    keep = torch.broadcast_to(keep, values.shape)
    if previous is not None:
        if previous.shape != values[:, :1].shape:
            raise ValueError("Reference frame has an incompatible shape")
        if previous.device != values.device or previous.dtype != values.dtype:
            raise ValueError("Reference frame must match the value device and dtype")
    elif not bool(keep[:, 0].all()):
        raise ValueError("Omitted first-frame positions require a reference")
    reference = values[:, :1] if previous is None else previous
    if mode == "reference":
        return torch.where(keep, values, reference)
    frames = []
    for frame in range(values.shape[1]):
        reference = torch.where(keep[:, frame:frame + 1], values[:, frame:frame + 1], reference)
        frames.append(reference)
    return torch.cat(frames, dim=1)


def restore_latents(values, keep, previous=None, mode="reference"):
    if values.shape != keep.shape:
        raise ValueError("Latent values and mask must have identical shapes")
    restored = restore_frames(values, keep, previous, mode)
    return restored, restored[:, -1:].detach().clone()


def _rotate_pairs(values, phase):
    pairs = values.double().unflatten(-1, (-1, 2))
    real, imaginary = pairs.unbind(-1)
    cosine, sine = phase.real, phase.imag
    rotated = torch.stack((real * cosine - imaginary * sine,
                           real * sine + imaginary * cosine), dim=-1)
    return rotated.flatten(-2).to(values.dtype)


def rotary_tokens(values, grid, frequencies, start_frame=0, indices=None):
    """Apply Wan's three-axis RoPE at explicit flattened token positions."""
    frames, height, width = (int(size) for size in grid)
    length = frames * height * width
    if values.shape[0] != 1 or values.shape[-1] % 2:
        raise ValueError("Rotary tokens require batch size one and an even head dimension")
    if indices is None:
        indices = torch.arange(length, device=values.device)
    if indices.numel() != values.shape[1]:
        raise ValueError("Rotary coordinates must match the number of computed tokens")
    pairs = values.shape[-1] // 2
    spatial_pairs = pairs // 3
    temporal_pairs = pairs - 2 * spatial_pairs
    coordinates = (indices // (height * width) + start_frame,
                   indices // width % height, indices % width)
    ranges = ((0, temporal_pairs), (temporal_pairs, temporal_pairs + spatial_pairs),
              (temporal_pairs + spatial_pairs, pairs))
    phase = torch.cat([frequencies.index_select(0, coordinate)[:, lower:upper]
                       for coordinate, (lower, upper) in zip(coordinates, ranges)], dim=-1)
    return _rotate_pairs(values, phase[None, :, None, :])


def shift_temporal_keys(keys, frames, frequencies):
    """Subtract a frame offset from cached keys without changing spatial phases."""
    if frames < 0 or frames >= frequencies.shape[0]:
        raise ValueError("Temporal shift lies outside the frequency table")
    pairs = keys.shape[-1] // 2
    temporal_pairs = pairs - 2 * (pairs // 3)
    phase = torch.ones(pairs, dtype=frequencies.dtype, device=keys.device)
    phase[:temporal_pairs] = frequencies[frames, :temporal_pairs].reciprocal()
    return _rotate_pairs(keys, phase)


def dense_rotary(values, grids, frequencies, start_frame=0):
    outputs = []
    for batch, grid in enumerate(grids.tolist()):
        length = math.prod(grid)
        rotated = rotary_tokens(values[batch:batch + 1, :length], grid, frequencies, start_frame)
        outputs.append(torch.cat((rotated, values[batch:batch + 1, length:]), dim=1))
    return torch.cat(outputs, dim=0)
