import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_check", ROOT / "tools" / "check_release.py")
release_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_check)


class ReleaseSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("README.md", "requirements.txt"):
            (self.root / name).write_text("Public documentation.\n", encoding="utf-8")

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def assert_rule(self, name, content, rule):
        self.write(name, content)
        _, issues = release_check.inspect(self.root)
        self.assertIn(rule, [issue[1] for issue in issues])

    def test_current_source_passes(self):
        files, issues = release_check.inspect(ROOT)
        self.assertFalse(issues, issues)
        self.assertIn(ROOT / "README.md", files)

    def test_quoted_credentials(self):
        key = "api" + "_key"
        self.assert_rule("configs/sample.json", '{"' + key + '": "' + "x" * 32 + '"}',
                         "credential-assignment")

    def test_environment_credentials(self):
        key = "SERVICE_" + "API_KEY"
        self.assert_rule("configs/sample.txt", key + '="' + "x" * 32 + '"',
                         "credential-assignment")

    def test_provider_tokens(self):
        for prefix in ("hf" + "_", "github" + "_pat_", "wandb" + "_v1_"):
            with self.subTest(prefix=prefix):
                self.assert_rule("configs/sample.txt", prefix + "x" * 40, "credential-token")

    def test_url_credentials(self):
        value = "https://" + "user:" + "example-password@" + "example.org/"
        self.assert_rule("docs/sample.md", value, "url-credentials")

    def test_private_paths(self):
        for value in ("/" + "home/example/project/", "C:" + "\\Users\\example\\project"):
            with self.subTest(value=value):
                self.assert_rule("configs/sample.txt", value, "private-filesystem")

    def test_hidden_paths(self):
        self.assert_rule("configs/.cache/config.json", "{}", "hidden-path")

    def test_sensitive_filename(self):
        self.assert_rule("configs/credentials.json", "{}", "sensitive-filename")

    def test_symlink_even_in_bytecode_directory(self):
        target = self.write("docs/public.txt", "Public.")
        link = self.root / "prestp" / "__pycache__" / "entry.pyc"
        link.parent.mkdir(parents=True)
        link.symlink_to(target)
        _, issues = release_check.inspect(self.root)
        self.assertIn((link.relative_to(self.root).as_posix(), "symlink"), issues)

    def test_generated_metadata_not_packaged(self):
        cached = self.write("prestp/__pycache__/sample.pyc", "discard")
        git_config = self.write(".git/config", "[core]\nrepositoryformatversion = 0\n")
        files, issues = release_check.inspect(self.root)
        self.assertFalse(issues)
        self.assertNotIn(cached, files)
        self.assertNotIn(git_config, files)

    def test_email_and_license_attribution(self):
        address = "maintainer" + "@" + "example.org"
        self.write("licenses/UPSTREAM.txt", address)
        _, issues = release_check.inspect(self.root)
        self.assertFalse(issues)
        self.assert_rule("docs/contact.md", address, "email")

    def test_key_material(self):
        private = "-----BEGIN " + "PRIVATE KEY-----"
        public = "ssh-" + "ed25519 " + "A" * 50
        self.assert_rule("configs/key.txt", private, "private-key")
        self.assert_rule("configs/key.txt", public, "ssh-public-key")

    def test_modified_image_requires_review(self):
        self.assert_rule("assets/framework.png", "modified image", "image-review-required")

    def test_unreviewed_image_is_rejected(self):
        self.assert_rule("assets/extra.png", "unreviewed image", "not-source-allowlisted")


if __name__ == "__main__":
    unittest.main()
