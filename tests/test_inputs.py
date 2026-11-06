import json
from pathlib import Path
import tempfile
import unittest

from prestp.config import Config
from prestp.inputs import parse_seeds, prompt_digest, read_prompts


class InputTests(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(Config.for_preset("5s").pixel_frames, 81)
        self.assertEqual(Config.for_preset("30s").pixel_frames, 501)
        root = Path(__file__).resolve().parents[1]
        for name in ("5s", "30s"):
            self.assertEqual(Config.load(root / "configs" / f"causvid_{name}.json"), Config.for_preset(name))

    def test_reject_implicit_schedule_changes(self):
        for values in ({"latent_frames": 81}, {"chunk_frames": 1}, {"context_noise": 1},
                       {"latent_frames": 21.0}, {"reanchor_period": 0}, {"reuse_mask": "false"},
                       {"spatial_threshold": float("nan")}, {"temporal_threshold": 1.1},
                       {"timestep_shift": 5}, {"denoising_steps": (1000, 757, 522)}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values)

    def test_digests(self):
        self.assertEqual(Config().digest(), Config().digest())
        self.assertNotEqual(Config().digest(), Config(spatial_threshold=0.96).digest())
        self.assertNotEqual(prompt_digest("a"), prompt_digest("b"))

    def test_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prompts.jsonl"
            target.write_text(json.dumps({"id": "safe_1", "prompt": " a boat "}), encoding="utf-8")
            self.assertEqual(read_prompts(target), [{"id": "safe_1", "prompt": "a boat"}])
            for records in ([{"id": "../escape", "prompt": "a"}], [{"id": None, "prompt": "a"}],
                            [{"id": "safe", "prompt": ""}], [{"id": "safe", "prompt": "a"}] * 2, []):
                target.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
                with self.subTest(records=records), self.assertRaises(ValueError):
                    read_prompts(target)

    def test_seed_order(self):
        self.assertEqual(parse_seeds("0,1,2,3,42"), [0, 1, 2, 3, 42])
        for text in ("", "1,1", "-1", str(2**32), "abc"):
            with self.assertRaises(ValueError):
                parse_seeds(text)


if __name__ == "__main__":
    unittest.main()
