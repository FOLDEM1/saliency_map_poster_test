from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    epochs: int
    batch_size: int
    num_workers: int
    max_train_samples: int | None
    max_val_samples: int | None
    max_eval_samples: int | None
    max_occ_samples: int | None
    random_budgets: list[int]
    bo_budget: int
    seeds: list[int]
    sliding_stride: int
    saliency_grid_stride: int
    train_image_size: int = 256
    occ_image_size: int = 256
    mask_size: int = 48
    size_candidates: list[int] | None = None


PROFILES: dict[str, Profile] = {
    "smoke": Profile(
        epochs=1,
        batch_size=2,
        num_workers=0,
        max_train_samples=4,
        max_val_samples=2,
        max_eval_samples=2,
        max_occ_samples=1,
        random_budgets=[4],
        bo_budget=4,
        seeds=[0],
        sliding_stride=64,
        saliency_grid_stride=16,
        train_image_size=128,
        occ_image_size=256,
        size_candidates=[32, 48, 64],
    ),
    "pilot": Profile(
        epochs=5,
        batch_size=8,
        num_workers=2,
        max_train_samples=200,
        max_val_samples=50,
        max_eval_samples=20,
        max_occ_samples=10,
        random_budgets=[25],
        bo_budget=25,
        seeds=[0, 1, 2],
        sliding_stride=32,
        saliency_grid_stride=8,
        size_candidates=[32, 48, 64],
    ),
    "full": Profile(
        epochs=20,
        batch_size=8,
        num_workers=2,
        max_train_samples=None,
        max_val_samples=None,
        max_eval_samples=None,
        max_occ_samples=None,
        random_budgets=[25, 50, 100, 200],
        bo_budget=200,
        seeds=[0, 1, 2, 3, 4],
        sliding_stride=16,
        saliency_grid_stride=8,
        size_candidates=[32, 48, 64],
    ),
}


@dataclass(frozen=True)
class Paths:
    data_root: Path
    checkpoint_dir: Path
    checkpoint: Path
    outputs_dir: Path
    metrics_dir: Path
    occlusion_dir: Path
    figures_dir: Path
    saliency_dir: Path
    predictions_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full BO-SegOcc experiment pipeline.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b0")
    parser.add_argument("--encoder-weights", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fresh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fresh-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-checkpoint", action="store_true", help="Do not delete checkpoints during a fresh run.")
    parser.add_argument("--legacy-hf-splits", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-occlusion", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    # Pomija krok oceny wierności map (Insertion/Deletion). Domyślnie krok się wykonuje.
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def paths_for(args: argparse.Namespace) -> Paths:
    metrics_dir = args.outputs_dir / "metrics"
    return Paths(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint=args.checkpoint_dir / "unet_best.pt",
        outputs_dir=args.outputs_dir,
        metrics_dir=metrics_dir,
        occlusion_dir=args.outputs_dir / "occlusion_runs",
        figures_dir=args.outputs_dir / "figures",
        saliency_dir=args.outputs_dir / "saliency_maps",
        predictions_dir=args.outputs_dir / "predictions",
    )


def join_ints(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def add_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def run_command(command: list[str], dry_run: bool) -> None:
    print("\n==>", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def clean_for_fresh_run(paths: Paths, fresh_data: bool, keep_checkpoint: bool, dry_run: bool) -> None:
    targets = [paths.outputs_dir]
    if not keep_checkpoint:
        targets.append(paths.checkpoint_dir)
    if fresh_data:
        targets.append(paths.data_root)
        targets.append(paths.data_root.parent.parent / "splits")

    for target in targets:
        print(f"clean: {target}", flush=True)
        if not dry_run:
            shutil.rmtree(target, ignore_errors=True)


def write_manifest(args: argparse.Namespace, profile: Profile, paths: Paths, dry_run: bool) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_name": args.profile,
        "profile": asdict(profile),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "args": {
            "seed": args.seed,
            "encoder_name": args.encoder_name,
            "encoder_weights": args.encoder_weights,
            "lr": args.lr,
            "fresh": args.fresh,
            "fresh_data": args.fresh_data,
            "keep_checkpoint": args.keep_checkpoint,
            "legacy_hf_splits": args.legacy_hf_splits,
        },
    }
    manifest_path = paths.metrics_dir / "pipeline_manifest.json"
    print(f"manifest: {manifest_path}", flush=True)
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download_command(args: argparse.Namespace, paths: Paths) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.download_data",
        "--output-dir",
        str(paths.data_root),
        "--force",
        "--split-seed",
        str(args.seed),
    ]
    if args.legacy_hf_splits:
        command.append("--legacy-hf-splits")
    return command


def stats_command(profile: Profile, paths: Paths) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.data_stats",
        "--data-root",
        str(paths.data_root),
        "--image-size",
        str(profile.occ_image_size),
        "--batch-size",
        str(max(profile.batch_size, 1)),
        "--num-workers",
        str(profile.num_workers),
        "--output",
        str(paths.metrics_dir / "train_channel_stats.json"),
    ]


def train_command(args: argparse.Namespace, profile: Profile, paths: Paths) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--data-root",
        str(paths.data_root),
        "--image-size",
        str(profile.train_image_size),
        "--epochs",
        str(profile.epochs),
        "--batch-size",
        str(profile.batch_size),
        "--num-workers",
        str(profile.num_workers),
        "--lr",
        str(args.lr),
        "--encoder-name",
        args.encoder_name,
        "--checkpoint-dir",
        str(paths.checkpoint_dir),
        "--metrics-path",
        str(paths.metrics_dir / "train_history.csv"),
        "--seed",
        str(args.seed),
    ]
    add_optional(command, "--encoder-weights", args.encoder_weights)
    add_optional(command, "--max-train-samples", profile.max_train_samples)
    add_optional(command, "--max-val-samples", profile.max_val_samples)
    return command


def evaluate_command(profile: Profile, paths: Paths, split: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--data-root",
        str(paths.data_root),
        "--split",
        split,
        "--checkpoint",
        str(paths.checkpoint),
        "--image-size",
        str(profile.occ_image_size),
        "--batch-size",
        str(profile.batch_size),
        "--num-workers",
        str(profile.num_workers),
        "--metrics-path",
        str(paths.metrics_dir / f"{split}_metrics.json"),
        "--predictions-dir",
        str(paths.predictions_dir),
    ]
    add_optional(command, "--max-samples", profile.max_eval_samples)
    return command


def random_command(profile: Profile, paths: Paths) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.run_random",
        "--data-root",
        str(paths.data_root),
        "--checkpoint",
        str(paths.checkpoint),
        "--image-size",
        str(profile.occ_image_size),
        "--mask-size",
        str(profile.mask_size),
        "--budgets",
        join_ints(profile.random_budgets),
        "--seeds",
        join_ints(profile.seeds),
        "--stats-path",
        str(paths.metrics_dir / "train_channel_stats.json"),
        "--output",
        str(paths.occlusion_dir / f"random_{profile_name(profile)}.csv"),
    ]
    add_optional(command, "--max-samples", profile.max_occ_samples)
    return command


def sliding_command(profile: Profile, paths: Paths) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.run_sliding",
        "--data-root",
        str(paths.data_root),
        "--checkpoint",
        str(paths.checkpoint),
        "--image-size",
        str(profile.occ_image_size),
        "--mask-size",
        str(profile.mask_size),
        "--stride",
        str(profile.sliding_stride),
        "--stats-path",
        str(paths.metrics_dir / "train_channel_stats.json"),
        "--output",
        str(paths.occlusion_dir / f"sliding_{profile_name(profile)}.csv"),
    ]
    add_optional(command, "--max-samples", profile.max_occ_samples)
    return command


def bo_command(profile: Profile, paths: Paths, variable_size: bool) -> list[str]:
    label = "bo_variable_size" if variable_size else "bo_fixed_size"
    command = [
        sys.executable,
        "-m",
        "src.run_bo",
        "--data-root",
        str(paths.data_root),
        "--checkpoint",
        str(paths.checkpoint),
        "--image-size",
        str(profile.occ_image_size),
        "--mask-size",
        str(profile.mask_size),
        "--budget",
        str(profile.bo_budget),
        "--seeds",
        join_ints(profile.seeds),
        "--stats-path",
        str(paths.metrics_dir / "train_channel_stats.json"),
        "--output",
        str(paths.occlusion_dir / f"{label}_{profile_name(profile)}.csv"),
        "--saliency-dir",
        str(paths.saliency_dir / label),
        "--saliency-grid-stride",
        str(profile.saliency_grid_stride),
    ]
    if variable_size:
        command.extend(["--optimize-size", "--size-candidates", join_ints(profile.size_candidates or [profile.mask_size])])
    add_optional(command, "--max-samples", profile.max_occ_samples)
    return command


def analyze_command(profile: Profile, paths: Paths, bo_label: str, output_label: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.analyze_occlusion",
        "--random-csv",
        str(paths.occlusion_dir / f"random_{profile_name(profile)}.csv"),
        "--sliding-csv",
        str(paths.occlusion_dir / f"sliding_{profile_name(profile)}.csv"),
        "--bo-csv",
        str(paths.occlusion_dir / f"{bo_label}_{profile_name(profile)}.csv"),
        "--random-budget",
        str(max(profile.random_budgets)),
        "--bo-budget",
        str(profile.bo_budget),
        "--default-mask-size",
        str(profile.mask_size),
        "--output-csv",
        str(paths.metrics_dir / f"occlusion_comparison_{output_label}_{profile_name(profile)}.csv"),
        "--summary-json",
        str(paths.metrics_dir / f"occlusion_comparison_{output_label}_{profile_name(profile)}.json"),
    ]


def faithfulness_command(profile: Profile, paths: Paths) -> list[str]:
    # Buduje komendę dla src.evaluate_faithfulness — oceny WIERNOŚCI map saliency
    # metodą Insertion/Deletion. Czyta CSV trzech metod (random/sliding/bo_variable),
    # buduje z nich mapy jednolicie (rasteryzacja + gęstnienie Voronoi, domyślnie
    # włączone) i liczy AUC + krzywe.
    label = profile_name(profile)
    command = [
        sys.executable,
        "-m",
        "src.evaluate_faithfulness",
        "--data-root",
        str(paths.data_root),
        "--checkpoint",
        str(paths.checkpoint),
        "--image-size",
        str(profile.occ_image_size),
        "--default-mask-size",
        str(profile.mask_size),
        # Trzy metody do porównania (BO bierzemy w wariancie variable_size — najlepsza mapa):
        "--random-csv",
        str(paths.occlusion_dir / f"random_{label}.csv"),
        "--sliding-csv",
        str(paths.occlusion_dir / f"sliding_{label}.csv"),
        "--bo-csv",
        str(paths.occlusion_dir / f"bo_variable_size_{label}.csv"),
        "--stats-path",
        str(paths.metrics_dir / "train_channel_stats.json"),
        # Wyniki: surowe AUC (CSV), agregaty (JSON), uśrednione krzywe (PNG):
        "--output-csv",
        str(paths.metrics_dir / f"faithfulness_{label}.csv"),
        "--summary-json",
        str(paths.metrics_dir / f"faithfulness_{label}.json"),
        "--figure",
        str(paths.figures_dir / f"faithfulness_curves_{label}.png"),
    ]
    # Ten sam podzbiór obrazów co reszta okluzji (spójność i niski koszt).
    add_optional(command, "--max-samples", profile.max_occ_samples)
    return command


def validate_command(profile: Profile, paths: Paths, csv_name: str, expected_seeds: int | None, expected_max_step: int) -> list[str]:
    expected_images = profile.max_occ_samples
    command = [
        sys.executable,
        "-m",
        "src.validate_occlusion_runs",
        str(paths.occlusion_dir / csv_name),
        "--expected-max-step",
        str(expected_max_step),
    ]
    add_optional(command, "--expected-images", expected_images)
    add_optional(command, "--expected-seeds", expected_seeds)
    return command


def sliding_grid_size(profile: Profile) -> int:
    half = profile.mask_size // 2
    max_center = profile.occ_image_size - half
    coords = list(range(half, max_center + 1, profile.sliding_stride))
    if coords[-1] != max_center:
        coords.append(max_center)
    return len(coords) * len(coords)


def visualization_commands(profile: Profile, paths: Paths) -> list[list[str]]:
    fixed_bo = paths.occlusion_dir / f"bo_fixed_size_{profile_name(profile)}.csv"
    variable_bo = paths.occlusion_dir / f"bo_variable_size_{profile_name(profile)}.csv"
    random_csv = paths.occlusion_dir / f"random_{profile_name(profile)}.csv"
    sliding_csv = paths.occlusion_dir / f"sliding_{profile_name(profile)}.csv"
    return [
        [
            sys.executable,
            "-m",
            "src.visualize_occlusion",
            "--data-root",
            str(paths.data_root),
            "--checkpoint",
            str(paths.checkpoint),
            "--image-size",
            str(profile.occ_image_size),
            "--mask-size",
            str(profile.mask_size),
            "--csv",
            str(fixed_bo),
            "--stats-path",
            str(paths.metrics_dir / "train_channel_stats.json"),
            "--output",
            str(paths.figures_dir / f"occlusion_example_{profile_name(profile)}.png"),
        ],
        [
            sys.executable,
            "-m",
            "src.visualize_three_occlusions",
            "--data-root",
            str(paths.data_root),
            "--checkpoint",
            str(paths.checkpoint),
            "--image-size",
            str(profile.occ_image_size),
            "--mask-size",
            str(profile.mask_size),
            "--random-csv",
            str(random_csv),
            "--bo-csv",
            str(variable_bo),
            "--sliding-csv",
            str(sliding_csv),
            "--random-budget",
            str(max(profile.random_budgets)),
            "--bo-budget",
            str(profile.bo_budget),
            "--stats-path",
            str(paths.metrics_dir / "train_channel_stats.json"),
            "--output",
            str(paths.figures_dir / f"three_occlusions_{profile_name(profile)}.png"),
        ],
    ]


def profile_name(profile: Profile) -> str:
    for name, candidate in PROFILES.items():
        if candidate == profile:
            return name
    return "custom"


def run_pipeline(args: argparse.Namespace) -> None:
    profile = PROFILES[args.profile]
    paths = paths_for(args)

    if args.fresh:
        clean_for_fresh_run(paths, fresh_data=args.fresh_data, keep_checkpoint=args.keep_checkpoint, dry_run=args.dry_run)

    write_manifest(args, profile, paths, dry_run=args.dry_run)

    if args.skip_train and not args.dry_run and not paths.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {paths.checkpoint}. "
            "Run without --skip-train or pass --checkpoint-dir pointing to an existing unet_best.pt."
        )

    if not args.skip_download:
        run_command(download_command(args, paths), args.dry_run)

    run_command(stats_command(profile, paths), args.dry_run)

    if not args.skip_train:
        run_command(train_command(args, profile, paths), args.dry_run)

    run_command(evaluate_command(profile, paths, "validation"), args.dry_run)
    run_command(evaluate_command(profile, paths, "test"), args.dry_run)

    if not args.skip_occlusion:
        run_command(random_command(profile, paths), args.dry_run)
        run_command(sliding_command(profile, paths), args.dry_run)
        run_command(bo_command(profile, paths, variable_size=False), args.dry_run)
        run_command(bo_command(profile, paths, variable_size=True), args.dry_run)

        label = profile_name(profile)
        run_command(
            validate_command(profile, paths, f"random_{label}.csv", expected_seeds=len(profile.seeds), expected_max_step=max(profile.random_budgets)),
            args.dry_run,
        )
        run_command(validate_command(profile, paths, f"sliding_{label}.csv", expected_seeds=None, expected_max_step=sliding_grid_size(profile)), args.dry_run)
        run_command(validate_command(profile, paths, f"bo_fixed_size_{label}.csv", expected_seeds=len(profile.seeds), expected_max_step=profile.bo_budget), args.dry_run)
        run_command(validate_command(profile, paths, f"bo_variable_size_{label}.csv", expected_seeds=len(profile.seeds), expected_max_step=profile.bo_budget), args.dry_run)

        run_command(analyze_command(profile, paths, "bo_fixed_size", "fixed_size"), args.dry_run)
        run_command(analyze_command(profile, paths, "bo_variable_size", "variable_size"), args.dry_run)

        # Ocena wierności map saliency (Insertion/Deletion). Odpinany krok — jedna linijka.
        if not args.skip_faithfulness:
            run_command(faithfulness_command(profile, paths), args.dry_run)

        if not args.skip_visualizations:
            for command in visualization_commands(profile, paths):
                run_command(command, args.dry_run)

    print("\nPipeline finished.", flush=True)


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
