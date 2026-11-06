import importlib.util
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


HAS_BACKEND = torch is not None and all(importlib.util.find_spec(name) is not None for name in
                                       ("diffusers", "transformers", "einops", "ftfy", "regex"))


def frequency_table(pairs=6):
    positions = torch.arange(128, dtype=torch.float64)
    rates = torch.linspace(0.01, 0.4, pairs, dtype=torch.float64)
    angles = positions[:, None] * rates
    return torch.polar(torch.ones_like(angles), angles)


def empty_cache(capacity=84, heads=2, width=12, dtype=None):
    return {
        "k": torch.zeros(1, capacity, heads, width, dtype=dtype),
        "v": torch.zeros(1, capacity, heads, width, dtype=dtype),
        "global_end_index": torch.tensor([0]),
        "local_end_index": torch.tensor([0]),
        "unrope_prev_k": None,
        "unrope_prev_v": None,
    }


def cpu_attention(query, keys, values, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)).transpose(1, 2)


def tiny_wrapper(dtype=None):
    from prestp.backend.wan.causal_model import CausalWanModel
    from prestp.backend.wan_wrapper import WanDiffusionWrapper
    from prestp.backend.scheduler import FlowMatchScheduler

    torch.manual_seed(123)
    wrapper = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = CausalWanModel(in_dim=2, out_dim=2, dim=24, ffn_dim=48, freq_dim=8,
                                  text_dim=8, num_heads=2, num_layers=2, text_len=4,
                                  local_attn_size=21)
    torch.nn.init.normal_(wrapper.model.head.head.weight, std=0.02)
    wrapper.model.num_frame_per_block = 3
    for block in wrapper.model.blocks:
        block.self_attn.max_attention_size = 84
    wrapper.uniform_timestep = False
    wrapper.seq_len = 84
    wrapper.flow_prev = None
    wrapper.noisy_prev = None
    wrapper.scheduler = FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)
    wrapper.scheduler.set_timesteps(1000, training=True)
    return wrapper.eval().requires_grad_(False).to(dtype=dtype or torch.float32)


def wrapper_caches(dtype=None):
    return [empty_cache(dtype=dtype) for _ in range(2)], [
        {"is_init": False, "k": None, "v": None} for _ in range(2)]


@unittest.skipIf(torch is None, "torch is required for token operations")
class TokenOperationsTests(unittest.TestCase):
    def setUp(self):
        from prestp.backend.wan import token_ops
        self.ops = token_ops
        torch.manual_seed(4)

    def test_indices_and_scatter_keep_original_order(self):
        mask = torch.tensor([[False, True, False, True]])
        indices = self.ops.selected_indices(mask)
        values = torch.tensor([[[2.0], [4.0]]])
        output = self.ops.expand_tokens(values, indices, 4)
        self.assertEqual(output.flatten().tolist(), [0, 2, 0, 4])

    def test_empty_selection(self):
        mask = torch.zeros(1, 4, dtype=torch.bool)
        output = self.ops.expand_tokens(torch.empty(1, 0, 3), self.ops.selected_indices(mask), 4)
        self.assertFalse(output.any())

    def test_bad_selection_shape(self):
        for mask in (torch.ones(4), torch.ones(2, 4, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                self.ops.selected_indices(mask)

    def test_patch_alignment(self):
        keep = torch.ones(1, 3, 2, 4, 4, dtype=torch.bool)
        keep[:, 1, :, :2, :2] = False
        tokens = self.ops.latent_token_mask(keep, (1, 2, 2))
        self.assertEqual(tokens.shape, (1, 12))
        self.assertFalse(tokens[0, 4])
        keep[0, 1, 1, 0, 0] = True
        with self.assertRaises(ValueError):
            self.ops.latent_token_mask(keep, (1, 2, 2))

    def test_restore_reference_and_cascade(self):
        values = torch.tensor([[[10., 11.], [20., 21.], [30., 31.]]])
        keep = torch.tensor([[[True, False], [False, True], [False, False]]])
        previous = torch.tensor([[[1., 2.]]])
        reference = self.ops.restore_frames(values, keep, previous, "reference")
        cascade = self.ops.restore_frames(values, keep, previous, "cascade")
        torch.testing.assert_close(reference, torch.tensor([[[10., 2.], [1., 21.], [1., 2.]]]))
        torch.testing.assert_close(cascade, torch.tensor([[[10., 2.], [10., 21.], [10., 21.]]]))
        torch.testing.assert_close(previous, torch.tensor([[[1., 2.]]]))

    def test_missing_reference_fails(self):
        with self.assertRaisesRegex(ValueError, "require a reference"):
            self.ops.restore_frames(torch.ones(1, 3, 2), torch.zeros(1, 3, 2, dtype=torch.bool))

    def test_first_complete_frame_can_seed_restoration(self):
        values = torch.tensor([[[1., 2.], [3., 4.]]])
        keep = torch.tensor([[[True, True], [False, False]]])
        restored = self.ops.restore_frames(values, keep)
        torch.testing.assert_close(restored, torch.tensor([[[1., 2.], [1., 2.]]]))

    def test_latent_reference_is_detached_and_independent(self):
        values = torch.randn(1, 3, 2, 4, 4, requires_grad=True)
        restored, reference = self.ops.restore_latents(values, torch.ones_like(values, dtype=torch.bool))
        expected = restored[:, -1:].detach().clone()
        restored.zero_()
        torch.testing.assert_close(reference, expected)
        self.assertFalse(reference.requires_grad)

    def test_all_dropped_latents_use_reference(self):
        values = torch.randn(1, 3, 2, 4, 4)
        previous = torch.randn(1, 1, 2, 4, 4)
        output, reference = self.ops.restore_latents(values, torch.zeros_like(values, dtype=torch.bool), previous)
        torch.testing.assert_close(output, previous.expand_as(output))
        torch.testing.assert_close(reference, previous)

    def test_rotary_matches_scalar_coordinate_formula(self):
        values = torch.randn(1, 12, 2, 12, dtype=torch.float64)
        frequencies = frequency_table()
        result = self.ops.rotary_tokens(values, (3, 2, 2), frequencies, 5)
        expected = values.clone()
        for token in range(12):
            coordinates = [5 + token // 4] * 2 + [token // 2 % 2] * 2 + [token % 2] * 2
            for pair, coordinate in enumerate(coordinates):
                complex_value = torch.complex(values[0, token, :, 2 * pair], values[0, token, :, 2 * pair + 1])
                rotated = complex_value * frequencies[coordinate, pair]
                expected[0, token, :, 2 * pair] = rotated.real
                expected[0, token, :, 2 * pair + 1] = rotated.imag
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_sparse_rotary_is_dense_subset(self):
        values = torch.randn(1, 12, 2, 12)
        indices = torch.tensor([0, 3, 5, 11])
        frequencies = frequency_table()
        dense = self.ops.rotary_tokens(values, (3, 2, 2), frequencies, 3)
        sparse = self.ops.rotary_tokens(values[:, indices], (3, 2, 2), frequencies, 3, indices)
        torch.testing.assert_close(sparse, dense[:, indices])

    def test_temporal_shift_preserves_spatial_coordinates(self):
        values = torch.randn(1, 12, 2, 12, dtype=torch.float64)
        frequencies = frequency_table()
        later = self.ops.rotary_tokens(values, (3, 2, 2), frequencies, 8)
        earlier = self.ops.rotary_tokens(values, (3, 2, 2), frequencies, 5)
        shifted = self.ops.shift_temporal_keys(later, 3, frequencies)
        torch.testing.assert_close(shifted, earlier, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(shifted[..., 4:], later[..., 4:], rtol=1e-12, atol=1e-12)

    def test_rotary_dtype_and_empty_input(self):
        frequencies = frequency_table()
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            values = torch.randn(1, 12, 2, 12).to(dtype)
            self.assertEqual(self.ops.rotary_tokens(values, (3, 2, 2), frequencies).dtype, dtype)
        empty = torch.empty(1, 0, 2, 12)
        result = self.ops.rotary_tokens(empty, (3, 2, 2), frequencies, indices=torch.empty(0, dtype=torch.long))
        self.assertEqual(result.shape, empty.shape)


@unittest.skipIf(torch is None, "torch is required for cache operations")
class SparseCacheTests(unittest.TestCase):
    def setUp(self):
        from prestp.backend.wan.sparse_cache import selected_attention_context
        self.context = selected_attention_context
        self.storage = empty_cache()
        self.frequencies = frequency_table()
        torch.manual_seed(17)

    def call(self, start, clean=False, keep=None, recover=True):
        if keep is None:
            keep = torch.ones(1, 12, dtype=torch.bool)
        count = int(keep.sum())
        query, keys, values = [torch.randn(1, count, 2, 12) for _ in range(3)]
        result = self.context(query, keys, values, storage=self.storage, grid=(3, 2, 2),
                              frequencies=self.frequencies, keep=keep, start=start,
                              clean=clean, recover=recover, attention_tokens=84)
        return result, (query, keys, values)

    def test_first_denoise_and_clean_reference(self):
        self.call(0)
        self.assertIsNone(self.storage["unrope_prev_k"])
        _, (_, keys, values) = self.call(0, clean=True)
        torch.testing.assert_close(self.storage["unrope_prev_k"], keys[:, -4:])
        torch.testing.assert_close(self.storage["unrope_prev_v"], values[:, -4:])
        self.assertEqual(self.storage["local_end_index"].item(), 12)

    def test_prefill_and_last_computed_values_survive_mask_changes(self):
        self.call(0, clean=True)
        reference = self.storage["unrope_prev_v"].clone()
        first = torch.zeros(1, 12, dtype=torch.bool)
        first[0, [0, 5]] = True
        _, (_, _, computed) = self.call(12, keep=first)
        expected = reference.repeat(1, 3, 1, 1)
        expected[:, [0, 5]] = computed
        torch.testing.assert_close(self.storage["v"][:, 12:24], expected)
        second = torch.zeros_like(first)
        second[0, [3, 7]] = True
        _, (_, _, updated) = self.call(12, keep=second)
        expected[:, [3, 7]] = updated
        torch.testing.assert_close(self.storage["v"][:, 12:24], expected)
        torch.testing.assert_close(self.storage["unrope_prev_v"], reference)

    def test_clean_refresh_cascades_and_updates_reference(self):
        self.call(0, clean=True)
        reference = self.storage["unrope_prev_v"].clone()
        keep = torch.zeros(1, 12, dtype=torch.bool)
        keep[0, [0, 5, 10]] = True
        _, (_, _, computed) = self.call(12, clean=True, keep=keep)
        expected = []
        previous = reference.clone()
        for frame, spatial in enumerate([0, 1, 2]):
            previous = previous.clone()
            previous[:, spatial] = computed[:, frame]
            expected.append(previous)
        torch.testing.assert_close(self.storage["v"][:, 12:24], torch.cat(expected, dim=1))
        torch.testing.assert_close(self.storage["unrope_prev_v"], expected[-1])

    def test_all_dropped_context_remains_finite(self):
        self.call(0, clean=True)
        result, _ = self.call(12, keep=torch.zeros(1, 12, dtype=torch.bool))
        self.assertEqual(result[0].shape[1], 0)
        self.assertTrue(torch.isfinite(result[1]).all())
        self.assertTrue(torch.isfinite(result[2]).all())

    def test_first_chunk_cannot_have_missing_reference(self):
        with self.assertRaisesRegex(ValueError, "first chunk must be dense"):
            self.call(0, keep=torch.zeros(1, 12, dtype=torch.bool))

    def test_missing_clean_refresh_fails_before_advancing(self):
        self.call(0)
        with self.assertRaisesRegex(ValueError, "previous clean KV"):
            self.call(12)
        self.assertEqual(self.storage["global_end_index"].item(), 12)

    def test_rolling_eviction_and_repeated_steps(self):
        from prestp.backend.wan.token_ops import shift_temporal_keys
        for start in range(0, 84, 12):
            self.call(start, clean=True)
        before_keys = self.storage["k"].clone()
        before_values = self.storage["v"].clone()
        self.call(84)
        torch.testing.assert_close(self.storage["k"][:, :72],
                                   shift_temporal_keys(before_keys[:, 12:], 3, self.frequencies))
        torch.testing.assert_close(self.storage["v"][:, :72], before_values[:, 12:])
        rolled = self.storage["k"][:, :72].clone()
        self.call(84)
        torch.testing.assert_close(self.storage["k"][:, :72], rolled)
        self.assertEqual(self.storage["global_end_index"].item(), 96)
        self.assertEqual(self.storage["local_end_index"].item(), 84)

    def test_noncontiguous_chunk_rejected(self):
        with self.assertRaises(ValueError):
            self.call(24)

    def test_fresh_dictionary_does_not_reuse_previous_prefill(self):
        self.call(0, clean=True)
        self.call(12)
        old_reference = self.storage["unrope_prev_v"].clone()
        self.storage = empty_cache()
        self.call(0, clean=True)
        new_reference = self.storage["unrope_prev_v"].clone()
        self.assertFalse(torch.equal(new_reference, old_reference))
        self.call(12, keep=torch.zeros(1, 12, dtype=torch.bool))
        torch.testing.assert_close(self.storage["v"][:, 12:24], new_reference.repeat(1, 3, 1, 1))

    def test_sink_tokens_are_preserved(self):
        from prestp.backend.wan.sparse_cache import SparseKVCache
        storage = empty_cache(capacity=16)
        storage["k"].normal_()
        storage["v"].normal_()
        storage["global_end_index"].fill_(16)
        storage["local_end_index"].fill_(16)
        sink = storage["k"][:, :4].clone()
        old_values = storage["v"].clone()
        cache = SparseKVCache(storage)
        slot = cache.reserve(16, 4, 4, self.frequencies, shift_frames=1, sink_tokens=4)
        torch.testing.assert_close(storage["k"][:, :4], sink)
        torch.testing.assert_close(storage["v"][:, 4:12], old_values[:, 8:16])
        self.assertEqual((slot.start, slot.end), (12, 16))


@unittest.skipUnless(HAS_BACKEND, "model libraries are required for backend integration")
class SparseBackendIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.self_attention = mock.patch("prestp.backend.wan.causal_model.attention", side_effect=cpu_attention)
        self.cross_attention = mock.patch("prestp.backend.wan.model.flash_attention", side_effect=cpu_attention)
        self.self_attention.start()
        self.addCleanup(self.self_attention.stop)
        self.cross_attention.start()
        self.addCleanup(self.cross_attention.stop)
        self.wrapper = tiny_wrapper()
        self.context = {"prompt_embeds": torch.randn(1, 4, 8)}

    def forward(self, noisy, caches, start, step, keep=None, restore=True, update=False):
        return self.wrapper(noisy, self.context, torch.full((1, 3), step), kv_cache=caches[0],
                            crossattn_cache=caches[1], current_start=start, is_prune=keep is not None,
                            is_unprune_kv=keep is not None, from_memory=True, prune_mask=keep,
                            restore=restore, update_prev=update)

    def test_all_kept_matches_dense_transformer(self):
        dense_cache, sparse_cache = wrapper_caches(), wrapper_caches()
        keep = torch.ones(1, 3, 2, 4, 4, dtype=torch.bool)
        for start in (0, 12, 24):
            for step in (1000, 750, 500, 250, 0):
                noisy = torch.randn(1, 3, 2, 4, 4)
                dense = self.forward(noisy, dense_cache, start, step)
                sparse = self.forward(noisy, sparse_cache, start, step, keep)
                for actual, expected in zip(sparse, dense):
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                for dense_layer, sparse_layer in zip(dense_cache[0], sparse_cache[0]):
                    torch.testing.assert_close(sparse_layer["k"], dense_layer["k"], rtol=1e-5, atol=1e-6)

    def test_sparse_wrapper_runs_beyond_position_window(self):
        caches = wrapper_caches()
        keep = torch.ones(1, 3, 2, 4, 4, dtype=torch.bool)
        for chunk in range(10):
            for step in (1000, 500, 0):
                mask = keep.clone()
                if chunk:
                    mask[:, :, :, :2, :2] = False
                output = self.forward(torch.randn(1, 3, 2, 4, 4), caches, chunk * 12, step,
                                      mask, restore=False)
                self.assertTrue(all(torch.isfinite(value).all() for value in output))
                self.assertTrue(all((value[~mask] == 0).all() for value in output))
        self.assertEqual(caches[0][0]["global_end_index"].item(), 120)
        self.assertEqual(caches[0][0]["local_end_index"].item(), 84)

    def test_wrapper_restores_empty_and_partial_masks(self):
        caches = wrapper_caches()
        all_kept = torch.ones(1, 3, 2, 4, 4, dtype=torch.bool)
        noisy = torch.randn(1, 3, 2, 4, 4)
        self.forward(noisy, caches, 0, 250, all_kept, update=True)
        self.forward(noisy, caches, 0, 0, all_kept)
        for empty in (False, True):
            mask = all_kept.clone()
            mask[:, :, :, :2 if not empty else 4] = False
            output = self.forward(noisy, caches, 12, 500, mask)
            self.assertTrue(all(torch.isfinite(value).all() for value in output))
            torch.testing.assert_close(output[0][~mask], self.wrapper.flow_prev.expand_as(noisy)[~mask])

    def test_state_dict_keys_do_not_include_runtime_state(self):
        before = set(self.wrapper.state_dict())
        caches = wrapper_caches()
        self.forward(torch.randn(1, 3, 2, 4, 4), caches, 0, 0,
                     torch.ones(1, 3, 2, 4, 4, dtype=torch.bool))
        self.assertEqual(before, set(self.wrapper.state_dict()))
        self.assertFalse(any("prestp_" in key for key in before))
        replica = tiny_wrapper()
        replica.load_state_dict(self.wrapper.state_dict(), strict=True)

    def sampler_pipeline(self):
        wrapper = tiny_wrapper(torch.bfloat16)
        context = torch.randn(1, 4, 8).to(torch.bfloat16)

        class Pipeline:
            generator = wrapper
            scheduler = wrapper.scheduler
            num_frame_per_block = 3
            frame_seq_length = 4
            denoising_step_list = [1000, 750, 500, 250]

            def text_encoder(self, text_prompts):
                return {"prompt_embeds": context}

            def _initialize_kv_cache(self, **kwargs):
                self.kv_cache1 = wrapper_caches(torch.bfloat16)[0]

            def _initialize_crossattn_cache(self, **kwargs):
                self.crossattn_cache = wrapper_caches(torch.bfloat16)[1]

        return Pipeline()

    def test_combined_sampler_reuse_freezing_and_fresh_video(self):
        from prestp.sampling import generate_combined
        pipeline = self.sampler_pipeline()
        calls = []

        def record_call(module, args, kwargs):
            calls.append((kwargs["current_start"], kwargs["prune_mask"].clone()))

        handle = pipeline.generator.register_forward_pre_hook(record_call, with_kwargs=True)
        self.addCleanup(handle.remove)
        kwargs = dict(seed=5, num_frames=30, tau=-1.0, freeze_tau=-1.0,
                      prune_context=True, step0_reuse=True, step0_gate=0.0,
                      step0_reanchor=3, channels=2, height=4, width=4)
        result = generate_combined(pipeline, SimpleNamespace(context_noise=0), "example", **kwargs)
        repeated = generate_combined(pipeline, SimpleNamespace(context_noise=0), "example", **kwargs)
        torch.testing.assert_close(result[0], repeated[0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(result[0]).all())
        self.assertGreater(result[2], 0)
        self.assertGreater(result[3], 0)
        self.assertTrue(all(mask.all() for _, mask in calls[:5]))
        self.assertTrue(any(not mask.all() for start, mask in calls if start > 0))

    def test_combined_sampler_all_kept_matches_dense_short_run(self):
        from prestp.sampling import generate, generate_combined
        pipeline = self.sampler_pipeline()
        config = SimpleNamespace(context_noise=0)
        kwargs = dict(seed=7, num_frames=6, channels=2, height=4, width=4)
        dense = generate(pipeline, config, "example", **kwargs)[0]
        sparse = generate_combined(pipeline, config, "example", tau=1.1, freeze_tau=1.1, **kwargs)[0]
        torch.testing.assert_close(sparse, dense, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
