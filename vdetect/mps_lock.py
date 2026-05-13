from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from typing import Iterator

# Cross-process advisory lock for serialising Apple MPS model loads. Concurrent
# loads can OOM the device; once loaded, inference is safe at study volume.
# Used by the API lifespan and the scripts that batch-load models (enrollment
# worker, phase 2 scorer, deepfake generator).

DEFAULT_LOCK_PATH = "/tmp/vdetect-mps.lock"


def _lock_path() -> str:
    return os.environ.get("VDETECT_MPS_LOCK", DEFAULT_LOCK_PATH)


@contextmanager
def mps_lock(owner: str) -> Iterator[None]:
    path = _lock_path()
    print(f"[{owner}] waiting for MPS lock at {path}...", flush=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        print(f"[{owner}] acquired MPS lock", flush=True)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
