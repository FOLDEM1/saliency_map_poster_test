from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def load_fill_rgb(stats_path: Path) -> torch.Tensor:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return torch.tensor(stats["mean_rgb"], dtype=torch.float32)


def write_rows(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def prune_incomplete_groups(path: Path, group_keys: list[str], required_steps: int) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return set()

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)

    complete_keys = {
        key
        for key, group_rows in groups.items()
        if max(int(float(row["step"])) for row in group_rows) >= required_steps
    }
    complete_rows = [row for row in rows if tuple(row[name] for name in group_keys) in complete_keys]

    if len(complete_rows) != len(rows):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(complete_rows)

    return complete_keys


def write_run_metadata(path: Path, metadata: dict[str, object]) -> None:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_csv_groups(path: Path, group_keys: Iterable[str]) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    keys = list(group_keys)
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    groups = {tuple(row[key] for key in keys) for row in rows}
    return len(rows), len(groups)
