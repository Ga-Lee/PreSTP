"""Chunk-addressed KV storage for PreSTP's selected-query attention."""
from dataclasses import dataclass

import torch

from .token_ops import expand_tokens, restore_frames, rotary_tokens, selected_indices, shift_temporal_keys


@dataclass(frozen=True)
class ChunkSlot:
    start: int
    end: int
    global_end: int
    is_new: bool


class SparseKVCache:
    """A view of one layer's cache; all per-video state belongs to the cache dict."""

    def __init__(self, storage):
        self.storage = storage

    def reserve(self, start, length, frame_tokens, frequencies, shift_frames=0, sink_tokens=0):
        storage = self.storage
        capacity = storage["k"].shape[1]
        global_end = int(storage["global_end_index"].item())
        local_end = int(storage["local_end_index"].item())
        end = start + length
        if length <= 0 or length + sink_tokens > capacity or length % frame_tokens:
            raise ValueError("Chunk and sink must fit inside the frame-aligned KV cache")
        if end == global_end:
            if length > local_end or storage.get("prestp_chunk_start", start) != start:
                raise ValueError("Repeated denoising must address the same chunk")
            return ChunkSlot(local_end - length, local_end, end, False)
        if start != global_end:
            raise ValueError("New chunks must follow the current cache boundary")
        overflow = max(0, local_end + length - capacity)
        if overflow:
            if overflow % frame_tokens:
                raise ValueError("Cache eviction must remove complete frames")
            source_start = sink_tokens + overflow
            remaining = local_end - source_start
            keys = storage["k"][:, source_start:local_end].clone()
            if shift_frames:
                keys = shift_temporal_keys(keys, shift_frames, frequencies)
            storage["k"][:, sink_tokens:sink_tokens + remaining] = keys
            storage["v"][:, sink_tokens:sink_tokens + remaining] = storage["v"][:, source_start:local_end].clone()
            local_end -= overflow
        storage.pop("prestp_prefilled_end", None)
        storage["prestp_chunk_start"] = start
        return ChunkSlot(local_end, local_end + length, end, True)

    def write(self, slot, keys, values, indices=None):
        if keys.shape != values.shape:
            raise ValueError("Keys and values must have identical shapes")
        if indices is None:
            if keys.shape[1] != slot.end - slot.start:
                raise ValueError("A full cache write must cover the reserved chunk")
            self.storage["k"][:, slot.start:slot.end] = keys
            self.storage["v"][:, slot.start:slot.end] = values
        else:
            if keys.shape[1] != indices.numel():
                raise ValueError("Sparse cache write and indices must agree")
            self.storage["k"].index_copy_(1, indices + slot.start, keys)
            self.storage["v"].index_copy_(1, indices + slot.start, values)

    def finish(self, slot, attention_tokens):
        self.storage["global_end_index"].fill_(slot.global_end)
        self.storage["local_end_index"].fill_(slot.end)
        start = max(0, slot.end - attention_tokens)
        return self.storage["k"][:, start:slot.end], self.storage["v"][:, start:slot.end]


def selected_attention_context(query, keys, values, *, storage, grid, frequencies,
                               keep, start, clean, recover, attention_tokens,
                               sink_frames=0, position_window=21):
    """Return selected rotated queries and the complete current attention context.

    Denoising retains earlier computed values within a chunk. A clean refresh
    reconstructs omitted positions from the preceding clean frame, sequentially.
    """
    frames, height, width = (int(size) for size in grid)
    frame_tokens = height * width
    length = frames * frame_tokens
    if keep.shape != (1, length):
        raise ValueError("Selection must describe the complete chunk grid")
    indices = selected_indices(keep)
    if query.shape[1] != indices.numel() or keys.shape != values.shape:
        raise ValueError("Selected queries, keys, values, and mask must agree")
    if keys.shape[1] != indices.numel():
        raise ValueError("Selected key count does not match the keep mask")
    if start % frame_tokens or frames > position_window:
        raise ValueError("Invalid frame position or chunk size")
    frame_start = start // frame_tokens
    rotary_start = min(frame_start, position_window - frames)
    cache = SparseKVCache(storage)
    if clean and recover:
        restored = []
        for computed, name in ((keys, "unrope_prev_k"), (values, "unrope_prev_v")):
            complete = expand_tokens(computed, indices, length)
            shaped = complete.unflatten(1, (frames, frame_tokens))
            reference = storage[name]
            reference = None if reference is None else reference.unsqueeze(1)
            complete = restore_frames(shaped, keep.reshape(1, frames, frame_tokens, 1, 1),
                                      reference, mode="cascade").flatten(1, 2)
            restored.append(complete)
        full_keys, full_values = restored
    elif recover and start > 0:
        if storage["unrope_prev_k"] is None or storage["unrope_prev_v"] is None:
            raise ValueError("A sparse denoising chunk requires a previous clean KV frame")
    elif indices.numel() != length:
        raise ValueError("Sparse queries need clean KV recovery; the first chunk must be dense")
    slot = cache.reserve(start, length, frame_tokens, frequencies,
                         shift_frames=frames if frame_start + frames >= position_window else 0,
                         sink_tokens=sink_frames * frame_tokens)
    rotated_query = rotary_tokens(query, grid, frequencies, rotary_start, indices)
    if clean and recover:
        cache.write(slot, rotary_tokens(full_keys, grid, frequencies, rotary_start), full_values)
        storage["unrope_prev_k"] = full_keys[:, -frame_tokens:].detach().clone()
        storage["unrope_prev_v"] = full_values[:, -frame_tokens:].detach().clone()
        storage.pop("prestp_prefilled_end", None)
    else:
        if recover and start > 0 and storage.get("prestp_prefilled_end") != slot.global_end:
            reference_keys = storage["unrope_prev_k"].repeat(1, frames, 1, 1)
            reference_values = storage["unrope_prev_v"].repeat(1, frames, 1, 1)
            cache.write(slot, rotary_tokens(reference_keys, grid, frequencies, rotary_start), reference_values)
            storage["prestp_prefilled_end"] = slot.global_end
        cache.write(slot, rotary_tokens(keys, grid, frequencies, rotary_start, indices), values, indices)
    context_keys, context_values = cache.finish(slot, attention_tokens)
    return rotated_query, context_keys, context_values
