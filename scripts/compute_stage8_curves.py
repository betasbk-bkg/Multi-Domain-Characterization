"""Run Stage 8 C(N) curve and utility-stopping analysis on standardized data.

This is intentionally dataset-agnostic. Use adapters to create standardized
labels_long.csv and gold.csv first.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from common_metrics import bootstrap_curve, summarize_curve, utility_optimum


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", type=Path, help="data/processed/<dataset_name>")
    p.add_argument("--mode", choices=["gold_accuracy", "reference_distribution"], required=True)
    p.add_argument("--metric", choices=["jsd", "tv"], default="jsd")
    p.add_argument("--B", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--maxN", type=int, default=50)
    p.add_argument("--outdir", type=Path, default=Path("reproduced/stage8_curves"))
    args = p.parse_args()

    labels = pd.read_csv(args.dataset_dir / "labels_long.csv")
    labels["item_id"] = labels["item_id"].astype(str)
    per_item = labels.groupby("item_id").size()
    if args.mode == "reference_distribution":
        max_possible = int(min(args.maxN, per_item.quantile(0.25) // 2))
    else:
        max_possible = int(min(args.maxN, per_item.quantile(0.25)))
    if max_possible < 2:
        raise RuntimeError("not enough repeated labels for N sweep")
    Ns = list(range(2, max_possible + 1))
    gold = None
    if args.mode == "gold_accuracy":
        gold_path = args.dataset_dir / "gold.csv"
        if not gold_path.exists():
            raise FileNotFoundError("gold.csv required for gold_accuracy mode")
        gold = pd.read_csv(gold_path)

    curve = bootstrap_curve(labels, gold, args.mode, Ns, args.B, args.seed, metric=args.metric)
    summary = summarize_curve(curve)
    lambdas = np.linspace(0.01, 0.99, 200)
    N_budget = max(Ns)
    util = utility_optimum(summary["N"], summary["C_mean"], lambdas, N_budget=N_budget)

    out = args.outdir / args.dataset_dir.name / args.mode
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / "curve_bootstrap.csv", index=False)
    summary.to_csv(out / "curve_summary.csv", index=False)
    util.to_csv(out / "utility_optimum_primary_budget.csv", index=False)
    (out / "curve_metadata.json").write_text(json.dumps({
        "dataset": args.dataset_dir.name,
        "mode": args.mode,
        "metric": args.metric,
        "B": args.B,
        "seed": args.seed,
        "Ns": Ns,
        "N_budget_primary": N_budget,
        "warning": "Stage 8 curve bootstrap; final closure uses fitted N_sat and bootstrap uncertainty propagation."
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
