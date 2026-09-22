"""Validate the exact public source-release selection using the standard library."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT_FILES = (
    ".env.example", ".gitattributes", ".gitignore", "AGENTS.md", "CHANGELOG.md", "CONTRIBUTING.md",
    "LICENSE", "Makefile", "NOTICE.md", "README.md", "README.zh-CN.md", "ROADMAP.md",
    "pyproject.toml", "uv.lock",
)
TREES = (".github", "docs", "examples", "scripts", "src", "tests", "benchmarks/sector-etf")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store"}
PRIVATE_PARTS = {".git", ".venv", ".alphaos", "runs", "datasets", "backtests"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".html", ".js", ".css", ".toml", ".lock", ".yml", ".yaml"}
PRIVATE_PATTERNS = (
    re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{25,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def release_paths(root: Path) -> list[Path]:
    """Use an explicit selection; local runs and repository history stay local."""
    root = root.resolve()
    selected = {Path(name) for name in ROOT_FILES}
    for tree in TREES:
        directory = root / tree
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Missing release directory: {tree}")
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if SKIP_PARTS.intersection(relative.parts) or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError(f"Symlink in release selection: {relative}")
            if path.is_file():
                selected.add(relative)
    for relative in selected:
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or unsafe release file: {relative}")
        if PRIVATE_PARTS.intersection(relative.parts):
            raise ValueError(f"Private workspace state in release selection: {relative}")
        if relative.name.startswith(".env") and relative.name != ".env.example":
            raise ValueError(f"Environment file in release selection: {relative}")
        if relative.suffix in {".key", ".pem"}:
            raise ValueError(f"Key file in release selection: {relative}")
        if any((root / parent).is_symlink() for parent in relative.parents):
            raise ValueError(f"Symlinked directory in release selection: {relative}")
    return sorted(selected)


def check(root: Path) -> dict:
    root = root.resolve()
    paths = release_paths(root)
    public = set(paths)
    errors = []
    for relative in paths:
        path = root / relative
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"File exceeds source-release size budget: {relative}")
        if path.suffix not in TEXT_SUFFIXES and relative.name not in {".env.example", "Makefile"}:
            continue
        body = path.read_text(encoding="utf-8")
        if any(pattern.search(body) for pattern in PRIVATE_PATTERNS):
            # Report filenames only, never the possible credential itself.
            errors.append(f"Personal path or credential-shaped text: {relative}")
        if path.suffix != ".md":
            continue
        prose = re.sub(r"```[^\n]*\n.*?```", "", body, flags=re.S)
        links = re.findall(r"\[[^\]\n]*\]\(([^\n]+?)\)", prose)
        links += re.findall(r"(?:src|href)=[\"']([^\"']+)[\"']", prose)
        for link in links:
            target = link.strip()
            if target.startswith("<") and ">" in target:
                target = target[1:target.index(">")]
            else:
                target = target.split(' "', 1)[0]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(root):
                errors.append(f"Link leaves public source tree: {relative}: {target}")
                continue
            linked = resolved.relative_to(root)
            if linked not in public and not any(linked in item.parents for item in public):
                errors.append(f"Link absent from release: {relative}: {target}")
    for expected in ("docs/assets/workbench.png", "docs/assets/benchmark-performance.png",
                     "docs/assets/benchmark-equity.png", "docs/assets/benchmark-costs.png",
                     "src/alpharesearchos/vendor/factorminer/LICENSE"):
        if Path(expected) not in public:
            errors.append(f"Required release asset missing: {expected}")
    return {"ok": not errors, "files": len(paths), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = check(args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"ok": False, "errors": [str(exc)]}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
