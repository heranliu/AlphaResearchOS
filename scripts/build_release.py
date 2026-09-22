"""Build a checked source ZIP containing the public project files."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from check_release import check, release_paths


def build(root: Path, output: Path) -> dict:
    validation = check(root)
    if not validation["ok"]:
        raise ValueError("Release checks failed: " + "; ".join(validation["errors"]))
    content = {path.as_posix(): (root / path).read_bytes() for path in release_paths(root)}
    output = output.resolve()
    if output.is_relative_to(root.resolve()) and "release" not in output.relative_to(root.resolve()).parts:
        raise ValueError("Write releases outside the source tree or into its ignored release/ directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alphaos-release-", dir=output.parent) as temporary:
        archive = Path(temporary) / "source.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for name, value in sorted(content.items()):
                info = zipfile.ZipInfo("AlphaResearchOS/" + name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, value)
        archive.replace(output)
    return {"ok": True, "archive": str(output), "files": len(content),
            "bytes": output.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("release/AlphaResearchOS-GitHub.zip"))
    args = parser.parse_args()
    try:
        result = build(Path(__file__).resolve().parents[1], args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"ok": False, "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
