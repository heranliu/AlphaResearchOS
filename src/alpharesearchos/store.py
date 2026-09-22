"""Atomic artifacts and a process lock for crash-safe resumable experiments."""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextlib.contextmanager
def run_lock(directory: Path):
    # flock releases automatically on process death; no stale pid guessing.
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This run is already active in another process") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
