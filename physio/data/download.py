"""Streaming download of MobiPhysio files from Harvard Dataverse.

The full dataset is ~159 GB, so videos are fetched one at a time into a temp
directory, consumed by the pose extractor, and deleted. Only pose sequences are kept.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import requests

from physio import config

CHUNK = 1 << 20  # 1 MiB


def datafile_url(file_id: int) -> str:
    return f"{config.DATAVERSE_API}/access/datafile/{file_id}"


def download_file(file_id: int, dest: Path, expected_size: int | None = None,
                  retries: int = 6, timeout: int = 180, session: requests.Session | None = None) -> Path:
    """Download one Dataverse file to `dest` atomically (via a .part file)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    sess = session or requests.Session()
    part = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with sess.get(datafile_url(file_id), stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(part, "wb") as fh:
                    shutil.copyfileobj(r.raw, fh, length=CHUNK)
            if expected_size is not None and part.stat().st_size != expected_size:
                raise IOError(f"size mismatch for {dest.name}: {part.stat().st_size} != {expected_size}")
            part.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retried, then surfaced
            last_err = exc
            part.unlink(missing_ok=True)
            time.sleep(min(60, 3 * 2 ** attempt))   # 3, 6, 12, 24, 48, 60 s: Dataverse returns 504 under load
    raise RuntimeError(f"download failed for file_id={file_id} ({dest.name}): {last_err}")


def download_pose_model(dest: Path = config.POSE_MODEL_PATH) -> Path:
    """Fetch the MediaPipe pose landmarker model once."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(config.POSE_MODEL_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            shutil.copyfileobj(r.raw, fh, length=CHUNK)
    return dest
