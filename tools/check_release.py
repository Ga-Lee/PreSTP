"""Scan source files for release hazards and optionally build a deterministic ZIP."""
import argparse
import hashlib
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"README.md", "requirements.txt",
             "LICENSE", "LICENSE.md", "CITATION.cff", "pyproject.toml", ".gitignore"}
TOP_DIRS = {"prestp", "configs", "examples", "tests", "tools", "docs", "licenses", "assets"}
SUFFIXES = {".py", ".json", ".md", ".txt", ".toml", ".cff"}
REVIEWED_IMAGES = {
    "assets/framework.png": "060299581825748e36dd82d48e7af4d907c8cdd8e1acaa57598819e594b6fae7",
    "assets/qualitative_comparison.png": "d7221782821846bb666d695d1505c11f6acefa6ddfc9400258f8e47434ed458a",
}
SENSITIVE_NAMES = re.compile(r"(?i)^(?:credentials?|secrets?|id_rsa|id_ed25519|id_ecdsa)(?:[._-]|$)")
PATTERNS = {
    "private-key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "credential-token": r"(?:wandb_v1_[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})",
    "credential-assignment": r'''(?i)\b(?:[a-z_]*api[_-]?key|access[_-]?token|secret[_-]?key|password)["']?\s*[:=]\s*["'][^"'\s]{8,}["']''',
    "url-credentials": r"https?://[^/\s:@]+:[^/\s@]+@",
    "ssh-public-key": r"(?:ssh-rsa|ssh-ed25519|ecdsa-sha2-nistp\d+)\s+[A-Za-z0-9+/]{30,}",
    "private-filesystem": r"(?:/(?:Users|home|mnt|root)/(?:[A-Za-z0-9_.-]+/)+|[A-Za-z]:\\Users\\[^\\\s]+\\)",
    "internal-host": r"(?i)https?://[A-Za-z0-9_.-]+\.(?:internal|corp|local)(?:/|\b)",
    "private-network-address": r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b",
    "shared-conversation": r"https?://(?:chatgpt|chat\.openai)\.com/share/",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
}


def inspect(root):
    files, issues = [], []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            issues.append((str(relative), "symlink"))
            continue
        if relative.parts[0] == ".git":
            continue
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if any(part.startswith(".") for part in relative.parts) and relative.as_posix() != ".gitignore":
            issues.append((str(relative), "hidden-path"))
            continue
        if path.is_dir():
            if relative.parts[0] not in TOP_DIRS:
                issues.append((str(relative), "unapproved-directory"))
            continue
        image_digest = REVIEWED_IMAGES.get(relative.as_posix())
        if image_digest is not None:
            if hashlib.sha256(path.read_bytes()).hexdigest() != image_digest:
                issues.append((str(relative), "image-review-required"))
            else:
                files.append(path)
            continue
        approved = ((len(relative.parts) == 1 and path.name in TOP_FILES)
                    or (len(relative.parts) > 1 and relative.parts[0] in TOP_DIRS
                        and path.suffix in SUFFIXES and not path.name.startswith(".")))
        if not approved or path.stat().st_size > 2_000_000:
            issues.append((str(relative), "not-source-allowlisted"))
            continue
        if SENSITIVE_NAMES.match(path.name):
            issues.append((str(relative), "sensitive-filename"))
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append((str(relative), "non-text-file"))
            continue
        for name, pattern in PATTERNS.items():
            if name == "email" and relative.parts[0] == "licenses":
                continue
            for match in re.finditer(pattern, content):
                line = content.count("\n", 0, match.start()) + 1
                issues.append((f"{relative}:{line}", name))
        files.append(path)
    for required in ("README.md", "requirements.txt"):
        if not (root / required).is_file():
            issues.append((required, "missing-required-file"))
    return files, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="New ZIP outside the source directory; never uploaded")
    args = parser.parse_args()
    files, issues = inspect(ROOT)
    if issues:
        for location, rule in issues:
            print(f"FAIL {location}: {rule}")
        raise SystemExit(1)
    print(f"PASS: {len(files)} allowlisted files; text privacy scan and reviewed-image hashes passed.")
    if args.zip:
        output = args.zip.resolve()
        if output == ROOT or ROOT in output.parents:
            parser.error("Place the ZIP outside the source directory")
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                name = "PreSTP_CausVid/" + path.relative_to(ROOT).as_posix()
                entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, path.read_bytes())
        print(f"SHA256 {hashlib.sha256(output.read_bytes()).hexdigest()}")
        print(f"Created {output.name}; no repository history, weights, or bytecode included.")


if __name__ == "__main__":
    main()
