from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate completeness of occlusion CSV run files.")
    parser.add_argument("csv_paths", type=Path, nargs="+")
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--expected-images", type=int, default=None)
    parser.add_argument("--expected-seeds", type=int, default=None)
    parser.add_argument("--expected-max-step", type=int, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def metadata_for(path: Path) -> dict[str, object]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def validate(path: Path, args: argparse.Namespace) -> dict[str, object]:
    rows = read_rows(path)
    metadata = metadata_for(path)

    images = {row["image_id"] for row in rows}
    seeds = {row.get("seed", "") for row in rows if row.get("seed", "") != ""}
    max_step = max((int(float(row["step"])) for row in rows), default=0)
    expected_rows = args.expected_rows
    if expected_rows is None:
        expected_rows = metadata.get("expected_rows_if_complete")

    checks = {
        "rows_ok": expected_rows is None or len(rows) == expected_rows,
        "images_ok": args.expected_images is None or len(images) == args.expected_images,
        "seeds_ok": args.expected_seeds is None or len(seeds) == args.expected_seeds,
        "max_step_ok": args.expected_max_step is None or max_step == args.expected_max_step,
    }

    return {
        "path": str(path),
        "rows": len(rows),
        "images": len(images),
        "seeds": sorted(seeds),
        "max_step": max_step,
        "expected_rows": expected_rows,
        "checks": checks,
        "ok": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    reports = [validate(path, args) for path in args.csv_paths]
    print(json.dumps(reports, indent=2))
    if not all(report["ok"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
