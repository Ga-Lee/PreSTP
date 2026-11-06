from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is optional for non-sampling tests")
class SamplingTests(unittest.TestCase):
    def setUp(self):
        from prestp import sampling
        self.sampling = sampling

    def pipeline(self, constant=False):
        class Generator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
                self.model = SimpleNamespace()
                self.calls = []

            def forward(self, noisy_image_or_video, **kwargs):
                self.calls.append(kwargs)
                prediction = torch.ones_like(noisy_image_or_video) if constant else noisy_image_or_video * 0.5
                return torch.zeros_like(prediction), prediction

        class Scheduler:
            def add_noise(self, clean, noise, timestep):
                return clean * 0.75 + noise * 0.25

        class Pipeline:
            def __init__(self):
                self.generator = Generator()
                self.scheduler = Scheduler()
                self.num_frame_per_block = 3
                self.frame_seq_length = 4
                self.denoising_step_list = [1000, 750, 500, 250]
                self.cache_initializations = 0

            def text_encoder(self, text_prompts):
                return {}

            def _initialize_kv_cache(self, **kwargs):
                self.kv_cache1 = []
                self.cache_initializations += 1

            def _initialize_crossattn_cache(self, **kwargs):
                self.crossattn_cache = []

        return Pipeline()

    def test_patch_and_mask_boundaries(self):
        video = torch.ones(1, 3, 2, 4, 4)
        previous = torch.ones(1, 2, 1, 4, 4)
        with mock.patch.object(self.sampling.F, "cosine_similarity", return_value=torch.full((3, 4), 0.5)):
            self.assertFalse(self.sampling.cosine_keep_mask(video, previous, 0.5, 2).any())
            self.assertFalse(self.sampling.cross_step_freeze(video, video, 0.5, 2).any())
        self.assertEqual(self.sampling.patchify(video[0]).shape, (3, 4, 8))

    def test_empty_fallback_is_patch_aligned(self):
        mask = torch.zeros(1, 3, 2, 4, 4, dtype=torch.bool)
        kept = self.sampling.ensure_nonempty_keep(mask)
        self.assertFalse(mask.any())
        self.assertEqual(kept.sum().item(), 3 * 2 * 4)
        self.assertTrue(kept[:, :, :, 2:4, 2:4].all())
        self.assertIs(self.sampling.ensure_nonempty_keep(kept), kept)

    def test_keep_all_matches_dense_on_fake_backend(self):
        pipeline = self.pipeline()
        config = SimpleNamespace(context_noise=0)
        dense, _, _ = self.sampling.generate(pipeline, config, "test", num_frames=6, channels=2, height=4, width=4)
        pruned, _, _, _ = self.sampling.generate_combined(
            pipeline, config, "test", num_frames=6, tau=1.1, freeze_tau=1.1,
            channels=2, height=4, width=4)
        self.assertTrue(torch.equal(dense, pruned))
        self.assertEqual(pipeline.cache_initializations, 2)

    def test_reuse_reanchor_and_first_chunk(self):
        pipeline = self.pipeline(constant=True)
        result, seconds, spatial, frozen = self.sampling.generate_combined(
            pipeline, SimpleNamespace(context_noise=0), "test", num_frames=21,
            step0_reuse=True, step0_gate=0.5, step0_reanchor=3, prune_context=True,
            channels=2, height=4, width=4)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(seconds, 0)
        self.assertEqual(len(pipeline.generator.calls), 35)
        self.assertTrue(all(call["prune_mask"].all() for call in pipeline.generator.calls[:5]))
        for chunk in (1, 4):
            self.assertTrue(pipeline.generator.calls[chunk * 5]["prune_mask"].all())
        for chunk in (2, 3, 5, 6):
            self.assertFalse(pipeline.generator.calls[chunk * 5]["prune_mask"].all())
        self.assertAlmostEqual(spatial, 0.75)
        self.assertGreater(frozen, 0)


if __name__ == "__main__":
    unittest.main()
