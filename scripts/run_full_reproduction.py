from __future__ import annotations

import argparse
import shutil
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproduced"

DATASET_RUNS = [
    ("CIFAR-10H", "gold_accuracy", 50),
    ("CIFAR-10H", "reference_distribution", 50),
    ("ChaosNLI", "reference_distribution", 50),
    ("Snapshot_Serengeti", "gold_accuracy", 50),
]


def run(args: list[str]) -> None:
    print("[run]", " ".join(args), flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(args, cwd=ROOT, check=True, env=env)


def clean_reproduced() -> None:
    if REPRO.exists():
        resolved = REPRO.resolve()
        if resolved.name != "reproduced" or resolved.parent != ROOT.resolve():
            raise RuntimeError(f"Refusing to remove unsafe path: {resolved}")
        shutil.rmtree(REPRO)


def copy_expected_stage8_curves() -> None:
    src = ROOT / "expected" / "stage8_curves"
    dst = REPRO / "stage8_curves"
    shutil.copytree(src, dst)


def compute_stage8_curves() -> None:
    for dataset, mode, max_n in DATASET_RUNS:
        run([
            sys.executable,
            "scripts/compute_stage8_curves.py",
            str(ROOT / "data" / "processed" / dataset),
            "--mode",
            mode,
            "--B",
            "200",
            "--seed",
            "20260709",
            "--maxN",
            str(max_n),
            "--outdir",
            str(REPRO / "stage8_curves"),
        ])


def run_downstream_closure() -> None:
    run([sys.executable, "scripts/recompute_final_closure.py"])
    run([sys.executable, "scripts/legacy_reanalysis/paperB_bingol.py"])
    run([sys.executable, "scripts/legacy_reanalysis/paperB_snow.py"])
    run([
        sys.executable,
        "scripts/legacy_reanalysis/paperB_nitti.py",
        "--data",
        str(ROOT / "data" / "legacy_components" / "nitti_data.xlsx"),
    ])
    run([
        sys.executable,
        "scripts/make_figures.py",
        "--final",
        str(REPRO / "final_closure"),
        "--legacy",
        str(REPRO / "legacy_components"),
        "--out",
        str(REPRO / "figures"),
    ])
    run([sys.executable, "scripts/verify_full_reproduction.py"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help=(
            "full recomputes Stage 8 bootstrap curves and is the official "
            "cross-platform reproduction check; quick reuses packaged Stage 8 "
            "curves and reruns downstream closure/figures as a smoke test"
        ),
    )
    args = parser.parse_args()

    clean_reproduced()
    if args.mode == "quick":
        print("[mode] quick: reusing packaged expected/stage8_curves", flush=True)
        copy_expected_stage8_curves()
    else:
        print("[mode] full: recomputing Stage 8 bootstrap curves", flush=True)
        compute_stage8_curves()
    run_downstream_closure()
    print(f"{args.mode.upper()}_MODE_REPRODUCTION: PASS", flush=True)


if __name__ == "__main__":
    main()
