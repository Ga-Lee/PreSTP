"""CPU-only tests using synthetic prompts and opaque MP4 fixture bytes."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from prestp.benchmark import SEED_ORDER, export, main, prepare, video_name
from prestp.config import Config
from prestp.inputs import prompt_digest, read_prompts


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.full_info = self.root / "full-info.json"
        self.extended = self.root / "extended.txt"
        self.prompts = self.root / "prompts.jsonl"
        self.run = self.root / "run"
        self.run.mkdir()
        self.metrics = self.run / "metrics.jsonl"
        self.output = self.root / "export"
        self.config = Config()
        self.prompt_rows = [{"id": "00000", "prompt": "Extended synthetic teapot scene",
                             "benchmark_prompt": "A striped teapot."}]

    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def write_full_info(self, prompts=None):
        if prompts is None:
            prompts = ["A striped teapot.", "A green kite, drifting."]
        self.full_info.write_text(json.dumps([{"prompt_en": prompt, "dimension": ["test"]}
                                              for prompt in prompts]), encoding="utf-8")

    def build_run(self):
        self.write_jsonl(self.prompts, self.prompt_rows)
        rows = []
        for prompt in self.prompt_rows:
            for seed in reversed(SEED_ORDER):
                for method in ("prestp", "dense"):
                    sample_id = f"{prompt['id']}-seed{seed}"
                    relative = f"{method}/{sample_id}.mp4"
                    video = self.run / relative
                    video.parent.mkdir(exist_ok=True)
                    video.write_bytes(f"Synthetic test video: {method}/{sample_id}".encode())
                    rows.append({"sample_id": sample_id, "prompt_id": prompt["id"],
                                 "seed": seed, "method": method, "video": relative,
                                 "prompt_digest": prompt_digest(prompt["prompt"]),
                                 "checkpoint_sha256": "a" * 64,
                                 "config_digest": self.config.digest(),
                                 "preset": self.config.preset,
                                 "num_frames": self.config.pixel_frames,
                                 "diffusion_seconds": 1.0})
        self.write_jsonl(self.metrics, rows)
        self.write_metadata()
        return rows

    def write_metadata(self):
        metadata = {"config": asdict(self.config), "config_digest": self.config.digest(),
                    "checkpoint_sha256": "a" * 64, "seed_order": list(reversed(SEED_ORDER))}
        (self.run / "run.json").write_text(json.dumps(metadata), encoding="utf-8")

    def assert_export_rejected(self, rows=None):
        if rows is not None:
            self.write_jsonl(self.metrics, rows)
        with self.assertRaises((ValueError, OSError)):
            export(self.metrics, self.prompts, "prestp", self.output)
        self.assertFalse(os.path.lexists(self.output))

    def test_prepare_preserves_originals_and_conditions_on_extended(self):
        self.write_full_info()
        self.extended.write_text("  Extended first scene  \n\nExtended second scene\n", encoding="utf-8")
        original_bytes = self.full_info.read_bytes()
        self.assertEqual(prepare(self.full_info, self.prompts, self.extended), 2)
        prepared = [json.loads(line) for line in self.prompts.read_text().splitlines()]
        self.assertEqual(prepared, [
            {"id": "00000", "prompt": "Extended first scene", "benchmark_prompt": "A striped teapot."},
            {"id": "00001", "prompt": "Extended second scene", "benchmark_prompt": "A green kite, drifting."},
        ])
        self.assertEqual(read_prompts(self.prompts),
                         [{"id": row["id"], "prompt": row["prompt"]} for row in prepared])
        self.assertEqual(self.full_info.read_bytes(), original_bytes)

    def test_prepare_without_extended(self):
        self.write_full_info()
        prepare(self.full_info, self.prompts)
        for line in self.prompts.read_text().splitlines():
            row = json.loads(line)
            self.assertEqual(row["prompt"], row["benchmark_prompt"])

    def test_extended_count_must_equal_not_just_cover_official_count(self):
        self.write_full_info()
        for text in ("one\n", "one\ntwo\nthree\n", "\n"):
            with self.subTest(text=text):
                self.extended.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "must equal"):
                    prepare(self.full_info, self.prompts, self.extended)
                self.assertFalse(self.prompts.exists())

    def test_invalid_full_info(self):
        for value in ([], {}, None, [None], [{}], [{"prompt_en": 7}]):
            with self.subTest(value=value):
                self.full_info.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    prepare(self.full_info, self.prompts)
                self.assertFalse(self.prompts.exists())

    def test_prepare_never_overwrites_existing_or_input_file(self):
        self.write_full_info()
        original = self.full_info.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare(self.full_info, self.full_info)
        self.assertEqual(self.full_info.read_bytes(), original)
        self.prompts.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            prepare(self.full_info, self.prompts)
        self.assertEqual(self.prompts.read_text(), "keep")

    def test_prepare_rejects_dangling_output_symlink(self):
        missing = self.root / "missing"
        self.prompts.symlink_to(missing)
        self.write_full_info()
        with self.assertRaises(FileExistsError):
            prepare(self.full_info, self.prompts)
        self.assertFalse(missing.exists())

    def test_literal_filename_and_fixed_seed_mapping(self):
        for index, seed in enumerate(SEED_ORDER):
            self.assertEqual(video_name("A teapot's café scene.", seed),
                             f"A teapot's café scene.-{index}.mp4")
        for seed in (4, -1, True, "42", 42.0, None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                video_name("A teapot", seed)

    def test_unsafe_names_are_rejected_not_sanitized(self):
        names = ["", " ", ".", "..", "../escape", "a/b", "a\\b", "C:/drive", "C:\\drive",
                 " leading", "trailing ", "line\nbreak", "tab\tname", "nul\x00name", "x\x7f",
                 "x\u202ename", "x\u2028name", "x\u2029name", "x\ud800",
                 "a" * 250, "é" * 125, None, 42]
        for name in names:
            with self.subTest(name=repr(name)), self.assertRaises(ValueError):
                video_name(name, 0)
        self.assertEqual(len(video_name("a" * 249, 0).encode()), 255)

    def test_linux_safe_punctuation_is_preserved_exactly(self):
        names = ['A sign reads "Hello": why?', '<red> | blue * green',
                 "C:drive", ".hidden", "... A scene", "A teapot's café: $5; (new)! [red]"]
        for name in names:
            for index, seed in enumerate(SEED_ORDER):
                with self.subTest(name=name, seed=seed):
                    self.assertEqual(video_name(name, seed), f"{name}-{index}.mp4")

    def test_prepare_and_export_preserve_punctuation(self):
        original = 'A sign asks: "Tea?" <yes> | no * maybe'
        self.write_full_info([original])
        self.extended.write_text("Extended synthetic sign scene\n", encoding="utf-8")
        prepare(self.full_info, self.prompts, self.extended)
        self.prompt_rows = [json.loads(line) for line in self.prompts.read_text().splitlines()]
        self.assertEqual(self.prompt_rows[0]["benchmark_prompt"], original)
        self.build_run()
        self.assertEqual(export(self.metrics, self.prompts, "prestp", self.output), 5)
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {f"{original}-{index}.mp4" for index in range(5)})
        for index, seed in enumerate(SEED_ORDER):
            self.assertEqual((self.output / f"{original}-{index}.mp4").read_bytes(),
                             (self.run / "prestp" / f"00000-seed{seed}.mp4").read_bytes())

    def test_prepare_rejects_filename_collisions(self):
        for names in (["Same", "Same"], ["Café", "Cafe\u0301"]):
            with self.subTest(names=names):
                self.write_full_info(names)
                with self.assertRaisesRegex(ValueError, "collision"):
                    prepare(self.full_info, self.prompts)
                self.assertFalse(self.prompts.exists())

    def test_case_only_names_are_preserved_not_folded(self):
        self.write_full_info(["A synthetic teapot", "a synthetic teapot"])
        self.assertEqual(prepare(self.full_info, self.prompts), 2)
        self.prompt_rows = [json.loads(line) for line in self.prompts.read_text().splitlines()]
        self.build_run()
        probe = self.root / "case-probe"
        probe.touch()
        if (self.root / "CASE-PROBE").exists():
            with self.assertRaisesRegex(ValueError, "case-sensitive"):
                export(self.metrics, self.prompts, "prestp", self.output)
            self.assertFalse(self.output.exists())
        else:
            self.assertEqual(export(self.metrics, self.prompts, "prestp", self.output), 10)
            self.assertEqual(len(list(self.output.iterdir())), 10)

    def test_export_copies_only_selected_method_in_official_seed_order(self):
        self.prompt_rows.append({"id": "named-seed9", "prompt": "Second extended scene",
                                 "benchmark_prompt": "A green kite."})
        self.build_run()
        (self.run / "unmatched.mp4").write_bytes(b"must not be copied")
        before = {path.relative_to(self.run): path.read_bytes()
                  for path in self.run.rglob("*") if path.is_file()}
        self.assertEqual(export(self.metrics, self.prompts, "prestp", self.output), 10)
        expected = {video_name(prompt["benchmark_prompt"], seed)
                    for prompt in self.prompt_rows for seed in SEED_ORDER}
        self.assertEqual({path.name for path in self.output.iterdir()}, expected)
        for prompt in self.prompt_rows:
            for seed in SEED_ORDER:
                source = self.run / "prestp" / f"{prompt['id']}-seed{seed}.mp4"
                copied = self.output / video_name(prompt["benchmark_prompt"], seed)
                self.assertEqual(copied.read_bytes(), source.read_bytes())
                self.assertNotEqual(copied.stat().st_ino, source.stat().st_ino)
        self.assertEqual(before, {path.relative_to(self.run): path.read_bytes()
                                  for path in self.run.rglob("*") if path.is_file()})

    def test_dense_export_and_absent_optional_metadata(self):
        self.build_run()
        (self.run / "run.json").unlink()
        self.assertEqual(export(self.metrics, self.prompts, "dense", self.output), 5)
        self.assertIn(b"dense/00000-seed42", (self.output / "A striped teapot.-4.mp4").read_bytes())

    def test_long_preset(self):
        self.config = Config.for_preset("30s")
        self.build_run()
        self.assertEqual(export(self.metrics, self.prompts, "prestp", self.output), 5)

    def test_missing_seed_pair_and_duplicate_record(self):
        rows = self.build_run()
        for broken in (rows[:-1], rows[:-2], [row for row in rows if row["method"] == "dense"],
                       rows + [rows[0]]):
            with self.subTest(length=len(broken)):
                self.assert_export_rejected(broken)

    def test_bad_record_fields(self):
        rows = self.build_run()
        invalid = [("prompt_id", "missing"), ("prompt_id", None), ("sample_id", "00000"),
                   ("sample_id", "00000-seed0"), ("seed", 4), ("seed", True), ("seed", "42"),
                   ("method", "ours"), ("prompt_digest", "0" * 64),
                   ("checkpoint_sha256", "b" * 64), ("config_digest", "b" * 64),
                   ("checkpoint_sha256", "A" * 64), ("config_digest", "not-a-digest"),
                   ("preset", "30s"), ("preset", "unknown"), ("num_frames", True),
                   ("num_frames", 82)]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                altered = [dict(row) for row in rows]
                altered[0][field] = value
                self.assert_export_rejected(altered)

    def test_consistency_checked_across_prompts_not_only_pairs(self):
        self.prompt_rows.append({"id": "second", "prompt": "Another synthetic scene",
                                 "benchmark_prompt": "A different kite"})
        rows = self.build_run()
        for field in ("checkpoint_sha256", "config_digest"):
            with self.subTest(field=field):
                altered = [dict(row) for row in rows]
                for row in altered:
                    if row["prompt_id"] == "second":
                        row[field] = "b" * 64
                self.assert_export_rejected(altered)

    def test_benchmark_prompt_is_not_used_for_conditioning_digest(self):
        rows = self.build_run()
        for row in rows:
            row["prompt_digest"] = prompt_digest(self.prompt_rows[0]["benchmark_prompt"])
        self.assert_export_rejected(rows)

    def test_prompt_jsonl_integrity(self):
        self.build_run()
        invalid = [[], self.prompt_rows * 2, [{"id": "00000", "prompt": "scene"}],
                   [{**self.prompt_rows[0], "id": "../bad"}],
                   [{**self.prompt_rows[0], "benchmark_prompt": "../bad"}],
                   [{**self.prompt_rows[0], "prompt": "changed"}],
                   self.prompt_rows + [{**self.prompt_rows[0], "id": "other"}],
                   self.prompt_rows + [{"id": "other", "prompt": "scene", "benchmark_prompt": "Other"}]]
        for prompts in invalid:
            with self.subTest(prompts=prompts):
                self.write_jsonl(self.prompts, prompts)
                self.assert_export_rejected()

    def test_json_duplicate_fields_and_nonfinite_constants(self):
        rows = self.build_run()
        original = self.metrics.read_text()
        invalid = ["\n", "null\n", "[]\n", '{"method":"dense","method":"prestp"}\n',
                   original.replace('"diffusion_seconds": 1.0', '"diffusion_seconds": NaN', 1)]
        for value in invalid:
            with self.subTest(value=value[:50]):
                self.metrics.write_text(value, encoding="utf-8")
                self.assert_export_rejected()
        self.write_jsonl(self.metrics, rows)

    def test_unsafe_video_paths(self):
        rows = self.build_run()
        invalid = ["../outside.mp4", "/tmp/outside.mp4", "C:/outside.mp4", "C:outside.mp4",
                   "\\\\server\\video.mp4", "dense/../outside.mp4", "dense\\video.mp4",
                   "dense//video.mp4", "./dense/video.mp4", "dense/file.mp4:stream",
                   "https://example.org/a.mp4", "dense/a\n.mp4", "dense/a\x00.mp4",
                   "dense/a\u202e.mp4", "dense/video.txt", "", None]
        for path in invalid:
            with self.subTest(path=path):
                altered = [dict(row) for row in rows]
                altered[0]["video"] = path
                self.assert_export_rejected(altered)

    def test_missing_empty_and_nonregular_videos(self):
        rows = self.build_run()
        video = self.run / rows[0]["video"]
        video.unlink()
        self.assert_export_rejected()
        video.touch()
        self.assert_export_rejected()
        video.unlink()
        video.mkdir()
        self.assert_export_rejected()
        video.rmdir()
        os.mkfifo(video)
        self.assert_export_rejected()

    def test_file_and_directory_symlink_escape(self):
        rows = self.build_run()
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"private outside bytes")
        video = self.run / rows[0]["video"]
        video.unlink()
        video.symlink_to(outside)
        self.assert_export_rejected()
        video.unlink()
        video.symlink_to(self.run / rows[1]["video"])
        self.assert_export_rejected()
        video.unlink()
        video.write_bytes(b"synthetic video restored")
        (self.run / "linked").symlink_to(self.root, target_is_directory=True)
        rows[0]["video"] = "linked/outside.mp4"
        self.assert_export_rejected(rows)
        self.assertEqual(outside.read_bytes(), b"private outside bytes")

    def test_source_file_aliases_are_collisions(self):
        rows = self.build_run()
        altered = [dict(row) for row in rows]
        altered[1]["video"] = altered[0]["video"]
        self.assert_export_rejected(altered)
        first, second = self.run / rows[0]["video"], self.run / rows[1]["video"]
        second.unlink()
        os.link(first, second)
        self.assert_export_rejected(rows)

    def test_existing_output_and_input_run_are_untouched(self):
        self.build_run()
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_bytes(b"original")
        with self.assertRaises(ValueError):
            export(self.metrics, self.prompts, "prestp", self.output)
        self.assertEqual(sentinel.read_bytes(), b"original")
        with self.assertRaises(ValueError):
            export(self.metrics, self.prompts, "prestp", self.run / "new")
        self.assertFalse((self.run / "new").exists())

    def test_dangling_output_and_metrics_symlink_rejected(self):
        self.build_run()
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(ValueError):
            export(self.metrics, self.prompts, "prestp", self.output)
        self.assertFalse((self.root / "missing").exists())
        self.output.unlink()
        link = self.run / "linked.jsonl"
        link.symlink_to(self.metrics)
        with self.assertRaises(ValueError):
            export(link, self.prompts, "prestp", self.output)

    def test_runner_metadata_mismatches(self):
        self.build_run()
        path = self.run / "run.json"
        original = json.loads(path.read_text())
        for field, value in (("checkpoint_sha256", "b" * 64), ("config_digest", "b" * 64),
                             ("seed_order", [0, 1, 2, 3, 4]), ("seed_order", [False, 1, 2, 3, 42]),
                             ("config", {}), ("config", {**original["config"], "spatial_threshold": 0.5})):
            with self.subTest(field=field):
                path.write_text(json.dumps({**original, field: value}), encoding="utf-8")
                if field == "config" and value == {}:
                    self.assertEqual(export(self.metrics, self.prompts, "dense", self.output), 5)
                    for video in self.output.iterdir():
                        video.unlink()
                    self.output.rmdir()
                else:
                    self.assert_export_rejected()

    def test_failed_copy_removes_only_new_export(self):
        self.build_run()
        originals = {path: path.read_bytes() for path in self.run.rglob("*.mp4")}
        with mock.patch("prestp.benchmark.shutil.copyfileobj", side_effect=OSError("copy failed")):
            self.assert_export_rejected()
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)

    def test_replaced_source_between_validation_and_copy_is_rejected(self):
        rows = self.build_run()
        from prestp import benchmark

        validate = benchmark._validate_run_metadata

        def substitute(root, reference):
            validate(root, reference)
            video = self.run / rows[0]["video"]
            video.unlink()
            video.symlink_to(self.root / "outside.mp4")

        (self.root / "outside.mp4").write_bytes(b"must never copy")
        with mock.patch("prestp.benchmark._validate_run_metadata", side_effect=substitute):
            self.assert_export_rejected()

    def test_cli_prepare_export_and_error_exit(self):
        self.write_full_info(["A striped teapot."])
        with mock.patch("sys.stdout"):
            main(["prepare", "--full-info-json", str(self.full_info), "--output", str(self.prompts)])
        self.build_run()
        with mock.patch("sys.stdout"):
            main(["export", "--metrics", str(self.metrics), "--prompts", str(self.prompts),
                  "--method", "prestp", "--output", str(self.output)])
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            main(["export", "--metrics", str(self.metrics), "--prompts", str(self.prompts),
                  "--method", "prestp", "--output", str(self.output)])
        self.assertEqual(error.exception.code, 2)

    def test_module_help_needs_no_gpu_or_evaluator_imports(self):
        release = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-B", "-m", "prestp.benchmark", "--help"],
                                cwd=release, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("vbench_standard", result.stdout)
        check = subprocess.run([sys.executable, "-B", "-c",
                                "import sys; import prestp.benchmark; "
                                "assert not {'torch', 'vbench', 'transformers'} & sys.modules.keys()"],
                               cwd=release, text=True, capture_output=True)
        self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == "__main__":
    unittest.main()
