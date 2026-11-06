import importlib.util
import unittest


HAS_BACKEND = all(importlib.util.find_spec(name) is not None for name in
                  ("torch", "diffusers", "transformers", "einops", "ftfy", "regex"))


@unittest.skipUnless(HAS_BACKEND, "model libraries are optional for basic tests")
class BackendSetupTests(unittest.TestCase):
    def test_text_encoder_device_follows_parameters(self):
        import torch
        from prestp.backend.wan_wrapper import WanTextEncoder
        encoder = WanTextEncoder.__new__(WanTextEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.text_encoder = torch.nn.Linear(2, 2)
        self.assertEqual(encoder.device, torch.device("cpu"))

    def test_cache_initialization_is_fresh(self):
        import torch
        from prestp.pipeline import CausVidPipeline
        pipeline = CausVidPipeline.__new__(CausVidPipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline.num_transformer_blocks = 1
        pipeline.local_attn_size = 6
        pipeline.frame_seq_length = 4
        kwargs = {"batch_size": 1, "dtype": torch.bfloat16, "device": "cpu"}
        pipeline._initialize_kv_cache(**kwargs)
        old_cache = pipeline.kv_cache1[0]
        self.assertEqual(old_cache["k"].shape, (1, 24, 12, 128))
        old_cache["k"].fill_(1)
        old_cache["global_end_index"].fill_(24)
        pipeline._initialize_kv_cache(**kwargs)
        self.assertFalse(pipeline.kv_cache1[0]["k"].any())
        self.assertEqual(pipeline.kv_cache1[0]["global_end_index"].item(), 0)
        self.assertIsNone(pipeline.kv_cache1[0]["unrope_prev_k"])
        self.assertIsNot(old_cache, pipeline.kv_cache1[0])
        pipeline._initialize_crossattn_cache(**kwargs)
        self.assertFalse(pipeline.crossattn_cache[0]["is_init"])
        self.assertEqual(pipeline.crossattn_cache[0]["k"].shape, (1, 512, 12, 128))


if __name__ == "__main__":
    unittest.main()
