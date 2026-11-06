"""Small aggregation tests and optional CPU/CUDA tensor tests."""

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from prestp.metrics import fidelity, main, summarize


def make_row(method="dense", sample_id="sample-0", **updates):
    row = {
        "sample_id": sample_id,
        "seed": 42,
        "method": method,
        "diffusion_seconds": 4.0 if method == "dense" else 2.0,
        "num_frames": 81,
        "video": f"{method}/{sample_id}.mp4",
        "preset": "5s",
        "config_digest": "example-config-digest",
        "prompt_digest": "example-prompt-digest",
        "checkpoint_sha256": "a" * 64,
    }
    row.update(updates)
    return row


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "metrics.jsonl"

    def write(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return self.path

    def test_weighted_speedup_and_throughput(self):
        rows = [make_row(), make_row("prestp"),
                make_row(sample_id="sample-1", diffusion_seconds=12.0),
                make_row("prestp", "sample-1", diffusion_seconds=3.0)]
        summary = summarize([self.write(rows)], expected_samples=2)
        self.assertEqual(summary["num_samples"], 2)
        self.assertEqual(summary["speedup"], 16 / 5)
        self.assertNotEqual(summary["speedup"], (2 + 4) / 2)
        self.assertEqual(summary["methods"]["dense"]["mean_latency_seconds"], 8)
        self.assertEqual(summary["methods"]["prestp"]["total_pixel_frames"], 162)
        self.assertEqual(summary["methods"]["prestp"]["pixel_frames_per_second"], 162 / 5)
        self.assertEqual(summary["fidelity"]["num_samples"], 0)

    def test_multiple_files(self):
        dense_path = self.write([make_row()])
        prestp_path = self.path.with_name("prestp.jsonl")
        prestp_path.write_text(json.dumps(make_row("prestp")) + "\n", encoding="utf-8")
        self.assertEqual(summarize([prestp_path, dense_path])["speedup"], 2)

    def test_duplicates_including_across_files_and_different_seeds(self):
        for duplicate in (make_row(), make_row(seed=7)):
            with self.subTest(duplicate=duplicate):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    summarize([self.write([make_row(), make_row("prestp"), duplicate])])
        path = self.write([make_row(), make_row("prestp")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([path, path])

    def test_unpaired_empty_and_wrong_expected_count(self):
        for rows in ([], [make_row()], [make_row("prestp")],
                     [make_row(), make_row("prestp", "different")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize([self.write(rows)])
        with self.assertRaises(ValueError):
            summarize([])
        path = self.write([make_row(), make_row("prestp")])
        for expected in (0, -1, True, 1.0, "1", 2):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                summarize([path], expected)

    def test_pair_metadata_must_match(self):
        for key, value in {"seed": 0, "preset": "30s", "num_frames": 501,
                           "config_digest": "different", "prompt_digest": "different",
                           "checkpoint_sha256": "b" * 64}.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                summarize([self.write([make_row(), make_row("prestp", **{key: value})])])

    def test_checkpoint_is_required_on_each_method(self):
        for method in ("dense", "prestp"):
            rows = [make_row(), make_row("prestp")]
            for row in rows:
                if row["method"] == method:
                    del row["checkpoint_sha256"]
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                summarize([self.write(rows)])

    def test_mixed_settings_across_valid_pairs_are_rejected(self):
        changes = {"config_digest": "different-config", "preset": "30s",
                   "checkpoint_sha256": "b" * 64, "num_frames": 501}
        for key, value in changes.items():
            first_pair = [make_row(), make_row("prestp")]
            second_pair = [make_row(method, "sample-1", **{key: value})
                           for method in ("dense", "prestp")]
            with self.subTest(key=key, layout="same file"):
                path = self.write(first_pair + second_pair)
                with self.assertRaisesRegex(ValueError, f"mixed {key} across pairs"):
                    summarize([path], expected_samples=2)
            with self.subTest(key=key, layout="separate files"):
                first_path = self.write(first_pair)
                second_path = self.path.with_name("second.jsonl")
                second_path.write_text("".join(json.dumps(row) + "\n" for row in second_pair),
                                       encoding="utf-8")
                for paths in ([first_path, second_path], [second_path, first_path]):
                    with self.assertRaisesRegex(ValueError, f"mixed {key} across pairs"):
                        summarize(paths, expected_samples=2)

    def test_one_setting_allows_multiple_prompts_and_seeds(self):
        rows = []
        for prompt_id, seed in (("prompt-a", 0), ("prompt-a", 1), ("prompt-b", 0)):
            for method in ("dense", "prestp"):
                rows.append(make_row(method, f"{prompt_id}-seed{seed}", seed=seed,
                                     prompt_id=prompt_id, prompt_digest=f"digest-{prompt_id}"))
        result = summarize([self.write(rows)], expected_samples=3)
        self.assertEqual(result["num_samples"], 3)
        self.assertEqual(result["speedup"], 2)
        self.assertEqual(result["methods"]["dense"]["total_pixel_frames"], 243)

    def test_matching_checkpoint_with_runner_fidelity_fields(self):
        scores = {"lpips": 0.1, "psnr": 30.0, "ssim": 0.9}
        rows = [make_row(), make_row("prestp", spatial_pruned_ratio=0.5,
                                    frozen_ratio=0.25, **scores)]
        result = summarize([self.write(rows)], expected_samples=1)
        self.assertEqual(result["fidelity"]["pairs"], {"sample-0": scores})

    def test_malformed_fields(self):
        invalid = {
            "sample_id": [None, 1, "", "  "],
            "seed": [None, True, 1.5, "42"],
            "method": ["other", None, 1],
            "diffusion_seconds": [None, True, "2", 0, -1, math.inf, -math.inf, math.nan],
            "num_frames": [None, True, 0, -1, 81.0],
            "video": [None, "", ".", "/video.mp4", "../video.mp4", "dir/../video.mp4",
                      "C:\\video.mp4", "C:video.mp4", "\\video.mp4", "..\\video.mp4",
                      "https://example.org/video.mp4", "video\x00.mp4"],
            "preset": [None, "10s"],
            "config_digest": [None, "", 1],
            "prompt_digest": [None, "", 1],
            "checkpoint_sha256": [None, "", "  ", 1, True, "a" * 63, "a" * 65,
                                  "g" * 64, "A" * 64, " " + "a" * 63],
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    summarize([self.write([make_row(**{key: value}), make_row("prestp")])])
        for key in make_row():
            row = make_row()
            del row[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                summarize([self.write([row, make_row("prestp")])])

    def test_invalid_json_is_not_skipped(self):
        for text in ("\n", "{broken}\n", "null\n", "[]\n", "42\n",
                     '{"method":"dense","method":"prestp"}\n'):
            self.path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, ":1:"):
                summarize([self.path])
        self.write([make_row(), make_row("prestp")])
        with self.path.open("a", encoding="utf-8") as output:
            output.write("\n")
        with self.assertRaisesRegex(ValueError, ":3:"):
            summarize([self.path])

    def test_missing_file_is_not_skipped(self):
        with self.assertRaises(OSError):
            summarize([self.path])

    def test_overflow_is_rejected(self):
        rows = [make_row(diffusion_seconds=1e308), make_row("prestp"),
                make_row(sample_id="sample-1", diffusion_seconds=1e308),
                make_row("prestp", "sample-1")]
        with self.assertRaises(ValueError):
            summarize([self.write(rows)])
        with self.assertRaises(ValueError):
            summarize([self.write([make_row("dense", diffusion_seconds=1e308),
                                   make_row("prestp", diffusion_seconds=1e-308)])])

    def test_fidelity_records_and_explicit_coverage(self):
        scores = {"lpips": 0.2, "psnr": 30.0, "ssim": 0.9,
                  "early_lpips": 0.1, "late_lpips": 0.3}
        rows = [make_row(), make_row("prestp", fidelity=scores),
                make_row(sample_id="sample-1"), make_row("prestp", "sample-1")]
        result = summarize([self.write(rows)])["fidelity"]
        self.assertEqual(result["num_samples"], 1)
        self.assertEqual(result["pairs"], {"sample-0": scores})
        self.assertEqual(result["means"], scores)
        self.assertEqual(result["counts"], dict.fromkeys(scores, 1))
        rows[-1].update(lpips=0.4, psnr=20.0, ssim=0.5)
        result = summarize([self.write(rows)])["fidelity"]
        self.assertAlmostEqual(result["means"]["lpips"], 0.3)
        self.assertEqual(result["counts"]["lpips"], 2)
        self.assertEqual(result["counts"]["early_lpips"], 1)

    def test_invalid_fidelity_is_rejected(self):
        base = {"lpips": 0.1, "psnr": 30.0, "ssim": 0.9}
        invalid = [None, [], {}, {"lpips": 0.1}, dict(base, unknown=1),
                   dict(base, early_lpips=0.1), dict(base, late_lpips=0.1)]
        for key in base:
            for value in (None, True, "0.1", math.nan, math.inf):
                invalid.append(dict(base, **{key: value}))
        invalid.extend([dict(base, lpips=-0.1), dict(base, psnr=-1),
                        dict(base, ssim=1.01), dict(base, ssim=-1.01),
                        dict(base, early_lpips=-1, late_lpips=0.1)])
        for scores in invalid:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                summarize([self.write([make_row(), make_row("prestp", fidelity=scores)])])
        for rows in ([make_row(fidelity=base), make_row("prestp")],
                     [make_row(), make_row("prestp", fidelity=base, **base)]):
            with self.assertRaises(ValueError):
                summarize([self.write(rows)])

    def test_cli_and_no_torch_import(self):
        path = self.write([make_row(), make_row("prestp")])
        output = self.path.with_name("summary.json")
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, "-m", "prestp.metrics", str(path),
                   "--output", str(output), "--expected-samples", "1"]
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(output.read_text())["speedup"], 2)
        code = "import sys; import prestp.metrics; assert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", code], cwd=root,
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        command[-1] = "2"
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("expected 2 samples", completed.stderr)

    def test_cli_does_not_clobber_input_or_output_on_validation_error(self):
        path = self.write([make_row(), make_row("prestp")])
        original = path.read_text()
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            main([str(path), "--output", str(path)])
        self.assertEqual(path.read_text(), original)
        output = self.path.with_name("summary.json")
        output.write_text("keep", encoding="utf-8")
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            main([str(path), "--output", str(output), "--expected-samples", "2"])
        self.assertEqual(output.read_text(), "keep")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is optional")
class FidelityTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.dense = torch.zeros(1, 8, 3, 2, 2)

    def model(self, dense, prestp):
        self.assertFalse(self.torch.is_grad_enabled())
        self.assertGreater(dense.shape[0], 0)
        self.assertLessEqual(dense.shape[0], 8)
        self.assertEqual(dense.shape[1:], (3, 2, 2))
        self.assertEqual(dense.shape, prestp.shape)
        self.assertEqual(dense.min().item(), -1)
        return ((dense - prestp) ** 2).mean(dim=(1, 2, 3), keepdim=True)

    def test_identity(self):
        self.assertEqual(fidelity(self.dense, self.dense, self.model),
                         {"lpips": 0.0, "psnr": 99.0, "ssim": 1.0})

    def test_dense_cpu_prestp_cuda_matches_same_device(self):
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA is optional")
        dense = torch.linspace(0, 1, self.dense.numel()).reshape_as(self.dense)
        prestp = (dense * 0.75 + 0.125).cuda()
        original = dense.clone()

        def model(first, second):
            self.assertEqual(first.device, prestp.device)
            self.assertEqual(second.device, prestp.device)
            self.assertFalse(torch.is_grad_enabled())
            return ((first - second) ** 2).mean(dim=(1, 2, 3), keepdim=True)

        with torch.inference_mode():
            expected = fidelity(dense.cuda(), prestp, model, long=True)
            actual = fidelity(dense, prestp, model, long=True)
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=6)
        self.assertEqual(dense.device.type, "cpu")
        self.assertTrue(torch.equal(dense, original))

    def test_whole_video_psnr_global_ssim_and_quarters(self):
        torch = self.torch
        prestp = torch.arange(8).reshape(1, 8, 1, 1, 1).expand_as(self.dense).float() / 8
        result = fidelity(self.dense, prestp, self.model, long=True)
        mse = (prestp ** 2).mean().item()
        expected_psnr = 10 * torch.log10(torch.tensor(1 / mse)).item()
        self.assertEqual(result["psnr"], expected_psnr)
        expected_ssim = (0.0001 * 0.0009 / ((prestp.mean() ** 2 + 0.0001)
                         * (prestp.var(unbiased=False) + 0.0009))).item()
        self.assertAlmostEqual(result["ssim"], expected_ssim, places=7)
        distances = (4 * prestp ** 2).mean(dim=(0, 2, 3, 4))
        self.assertAlmostEqual(result["lpips"], distances.mean().item())
        self.assertAlmostEqual(result["early_lpips"], distances[:2].mean().item())
        self.assertAlmostEqual(result["late_lpips"], distances[-2:].mean().item())

    def test_psnr_threshold(self):
        result = fidelity(self.dense, self.dense + 0.000001, self.model)
        self.assertEqual(result["psnr"], 99.0)

    def test_psnr_preserves_source_float32_logarithm(self):
        torch = self.torch
        for difference in (0.0001, 0.01, 0.3, 1.0):
            with self.subTest(difference=difference):
                prestp = self.dense + difference
                result = fidelity(self.dense, prestp, self.model)
                mse = ((self.dense.float() - prestp.float()) ** 2).mean().item()
                expected = 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
                self.assertEqual(result["psnr"], expected)
                self.assertAlmostEqual(result["psnr"], 10.0 * math.log10(1.0 / mse), delta=1e-5)

    def test_lpips_batches_preserve_frame_weighting_and_quarters(self):
        torch = self.torch
        for frames in (17, 81, 501):
            with self.subTest(frames=frames):
                dense = torch.zeros(1, frames, 3, 2, 2)
                prestp = torch.linspace(0, 1, frames).reshape(1, frames, 1, 1, 1).expand_as(dense)
                sizes = []

                def model(first, second):
                    sizes.append(first.shape[0])
                    return self.model(first, second)

                result = fidelity(dense, prestp, model, long=True)
                distances = ((dense[0] * 2 - 1 - (prestp[0] * 2 - 1)) ** 2).mean(dim=(1, 2, 3))
                quarter = frames // 4
                self.assertEqual(sizes, [8] * (frames // 8) + [frames % 8])
                self.assertEqual(result["lpips"], distances.mean().item())
                self.assertEqual(result["early_lpips"], distances[:quarter].mean().item())
                self.assertEqual(result["late_lpips"], distances[-quarter:].mean().item())

    def test_invalid_lpips_output_in_final_partial_batch_is_rejected(self):
        torch = self.torch
        dense = torch.zeros(1, 9, 3, 2, 2)
        for output in (torch.zeros(8), torch.tensor(0.0), torch.tensor([math.nan]),
                       torch.tensor([math.inf]), torch.tensor([-1.0])):
            with self.subTest(output=output):
                outputs = iter((torch.zeros(8, 1, 1, 1), output))
                with self.assertRaisesRegex(ValueError, "LPIPS"):
                    fidelity(dense, dense, lambda first, second: next(outputs))

    def test_ssim_uses_global_biased_variance_and_covariance(self):
        torch = self.torch
        dense = torch.tensor([0.0, 0.25, 0.5, 1.0]).reshape(1, 4, 1, 1, 1)
        prestp = torch.tensor([0.75, 0.5, 0.25, 0.0]).reshape(1, 4, 1, 1, 1)
        dense = dense.expand(1, 4, 3, 1, 1)
        prestp = prestp.expand_as(dense)
        result = fidelity(dense, prestp, lambda first, second: torch.zeros(4))
        dense_values = dense.flatten().tolist()
        prestp_values = prestp.flatten().tolist()
        count = len(dense_values)
        dense_mean = sum(dense_values) / count
        prestp_mean = sum(prestp_values) / count
        dense_var = sum((value - dense_mean) ** 2 for value in dense_values) / count
        prestp_var = sum((value - prestp_mean) ** 2 for value in prestp_values) / count
        covariance = sum((first - dense_mean) * (second - prestp_mean)
                         for first, second in zip(dense_values, prestp_values)) / count
        expected = ((2 * dense_mean * prestp_mean + 0.0001) * (2 * covariance + 0.0009)
                    / ((dense_mean ** 2 + prestp_mean ** 2 + 0.0001)
                       * (dense_var + prestp_var + 0.0009)))
        self.assertLess(result["ssim"], 0)
        self.assertAlmostEqual(result["ssim"], expected, places=6)

    def test_short_video_quarters_have_at_least_one_frame(self):
        dense = self.dense[:, :1]
        result = fidelity(dense, dense, lambda first, second: self.torch.zeros(1), long=True)
        self.assertEqual(result["early_lpips"], 0)
        self.assertEqual(result["late_lpips"], 0)

    def test_quarters_use_floor_frame_count_for_nonmultiples_of_four(self):
        torch = self.torch
        for frames in (7, 9, 501):
            with self.subTest(frames=frames):
                dense = torch.zeros(1, frames, 3, 1, 1)
                distances = torch.arange(frames, dtype=torch.float32)
                batches = iter(distances.split(8))
                result = fidelity(dense, dense, lambda first, second: next(batches), long=True)
                quarter = frames // 4
                self.assertEqual(result["early_lpips"], (quarter - 1) / 2)
                self.assertEqual(result["late_lpips"], frames - (quarter + 1) / 2)

    def test_invalid_videos(self):
        torch = self.torch
        invalid = [None, self.dense[0], self.dense[:, :0], self.dense[:, :, :1],
                   self.dense.long(), self.dense - 0.1, self.dense + 1.1,
                   self.dense + math.nan, self.dense + math.inf]
        for video in invalid:
            with self.subTest(video=type(video)), self.assertRaises(ValueError):
                fidelity(video, self.dense, self.model)
        with self.assertRaises(ValueError):
            fidelity(torch.zeros(2, 8, 3, 2, 2), torch.zeros(2, 8, 3, 2, 2), self.model)

    def test_bad_lpips_outputs_and_backbone(self):
        torch = self.torch
        for output in (0.1, torch.tensor(0.1), torch.zeros(7), torch.zeros(8, 2),
                       torch.full((8,), math.nan), torch.full((8,), -1.0)):
            with self.subTest(output=output), self.assertRaises(ValueError):
                fidelity(self.dense, self.dense, lambda dense, prestp: output)
        model = mock.Mock(pnet_type="vgg")
        with self.assertRaisesRegex(ValueError, "AlexNet"):
            fidelity(self.dense, self.dense, model)


if __name__ == "__main__":
    unittest.main()
