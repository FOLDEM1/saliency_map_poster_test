from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "Angelou0516/kvasir-seg"
DEFAULT_OUTPUT_DIR = Path("data/raw/kvasir-seg")
SPLITS = ("train", "validation", "test")
SPLIT_FILES = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}


def clear_split_dirs(output_dir: Path) -> None:
    for split in SPLITS:
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def save_hf_split(dataset, split: str, output_dir: Path) -> list[str]:
    images_dir = output_dir / split / "images"
    masks_dir = output_dir / split / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    stems = []
    for index, sample in enumerate(tqdm(dataset[split], desc=f"Saving {split}")):
        stem = f"{index:04d}.png"
        stems.append(stem)
        sample["image"].convert("RGB").save(images_dir / stem)
        sample["mask"].convert("L").save(masks_dir / stem)
    return stems


def write_split_file(output_dir: Path, split: str, stems: list[str]) -> None:
    splits_dir = output_dir.parent.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / SPLIT_FILES[split]).write_text("\n".join(stems) + "\n", encoding="utf-8")


def save_project_splits(
    dataset,
    output_dir: Path,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    records = [(split, index) for split in SPLITS for index in range(len(dataset[split]))]
    rng = random.Random(split_seed)
    rng.shuffle(records)

    total = len(records)
    train_count = round(total * train_ratio)
    val_count = round(total * val_ratio)
    allocations = {
        "train": records[:train_count],
        "validation": records[train_count : train_count + val_count],
        "test": records[train_count + val_count :],
    }

    for target_split, split_records in allocations.items():
        images_dir = output_dir / target_split / "images"
        masks_dir = output_dir / target_split / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        stems = []
        for output_index, (source_split, source_index) in enumerate(tqdm(split_records, desc=f"Saving {target_split}")):
            stem = f"{output_index:04d}.png"
            stems.append(stem)
            sample = dataset[source_split][source_index]
            sample["image"].convert("RGB").save(images_dir / stem)
            sample["mask"].convert("L").save(masks_dir / stem)
        write_split_file(output_dir, target_split, stems)


def save_legacy_hf_splits(dataset, output_dir: Path) -> None:
    for split in SPLITS:
        stems = save_hf_split(dataset, split, output_dir)
        write_split_file(output_dir, split, stems)


def download_dataset(
    output_dir: Path,
    force: bool = False,
    legacy_hf_splits: bool = False,
    split_seed: int = 0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Path:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME)
    clear_split_dirs(output_dir)
    if legacy_hf_splits:
        save_legacy_hf_splits(dataset, output_dir)
    else:
        save_project_splits(dataset, output_dir, split_seed, train_ratio, val_ratio)

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and export the Kvasir-SEG dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for exported dataset files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the output directory before exporting the dataset.",
    )
    parser.add_argument(
        "--legacy-hf-splits",
        action="store_true",
        help="Use the dataset provider's 800/100/100 splits instead of the project 70/15/15 split.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = download_dataset(
        args.output_dir,
        force=args.force,
        legacy_hf_splits=args.legacy_hf_splits,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print("Path to dataset files:", path)


if __name__ == "__main__":
    main()
