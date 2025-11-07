"""Model snapshot utilities for trainer & agents hot-reload.

Responsibilities:
  - Discover latest model snapshot in a directory (by monotonically increasing suffix or timestamp).
  - Provide atomic save (temp file + rename) to avoid partial reads by agents.
  - Simple naming convention: <base_name>_step<global_step>.pt or timestamp if step unknown.

Agents can periodically poll `find_latest_snapshot` and load if the path changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import re
from typing import Optional, List

import torch

SNAPSHOT_REGEX = re.compile(r"^(?P<base>.+)_step(?P<step>\d+)\.pt$")


@dataclass
class SnapshotInfo:
    path: Path
    step: int


def _list_candidate_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return [p for p in directory.iterdir() if p.is_file() and p.suffix == '.pt']


def find_latest_snapshot(directory: str | Path) -> Optional[SnapshotInfo]:
    """Return the latest snapshot (highest step) if any."""
    d = Path(directory)
    best: Optional[SnapshotInfo] = None
    for f in _list_candidate_files(d):
        m = SNAPSHOT_REGEX.match(f.name)
        if not m:
            continue
        step = int(m.group('step'))
        if best is None or step > best.step:
            best = SnapshotInfo(path=f, step=step)
    return best


def atomic_save(model_state: dict, directory: str | Path, base_name: str, step: int | None = None) -> Path:
    """Atomically save model_state in directory with step-based filename.

    If step is None, a timestamp-based surrogate is used to keep ordering (ms resolution).
    Returns final path.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if step is None:
        step = int(time.time() * 1000)
    filename = f"{base_name}_step{step}.pt"
    final_path = d / filename
    tmp_path = final_path.with_suffix('.tmp')
    torch.save(model_state, tmp_path)
    tmp_path.replace(final_path)
    return final_path
