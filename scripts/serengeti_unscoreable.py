"""Serengeti expert-label coverage diagnostics quoted in Section IV-A of the manuscript.

An event is unscoreable when the expert label assigned to it appears among no volunteer
label for that event: the majority vote can never return it, so the event scores zero at
every N. This script counts those events, reports the fraction of the available set they
represent at the low and high ends of the grid, and refits the Michaelis-Menten curve with
that fraction removed, which is the sensitivity the manuscript reports.

The removal is applied as a rescaling of the mean curve rather than a re-run of the
bootstrap: an event that scores zero at every N contributes zero to every mean, so
dividing the mean by the scoreable fraction at each N gives the curve that the remaining
events would produce.

    python scripts/serengeti_unscoreable.py

Writes expected/diagnostics/serengeti_unscoreable.csv.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def michaelis_menten(n, c0, amp, k):
    return c0 + amp * n / (k + n)


def fit_k(n, c):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, _ = curve_fit(michaelis_menten, n, c,
                         p0=[c.min(), c.max() - c.min(), 1.0],
                         bounds=([0, 0, 1e-6], [1.2, 1.2, 1000.0]), maxfev=200000)
    return float(p[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/Snapshot_Serengeti"))
    ap.add_argument("--curves", type=Path,
                    default=Path("expected/stage8_curves/Snapshot_Serengeti/gold_accuracy"))
    ap.add_argument("--constants", type=Path,
                    default=Path("expected/final_closure/replicate_constants.csv"))
    ap.add_argument("--out", type=Path, default=Path("expected/diagnostics/serengeti_unscoreable.csv"))
    args = ap.parse_args()

    gold = pd.read_csv(args.data / "gold.csv", dtype=str)
    labels = pd.read_csv(args.data / "labels_long.csv", dtype=str)
    by_item = labels.groupby("item_id")["label"].apply(set)
    per_item = labels.groupby("item_id").size()

    unscoreable = {r.item_id for r in gold.itertuples()
                   if r.gold_label not in by_item.get(r.item_id, set())}
    breakdown = (gold[gold.item_id.isin(unscoreable)]
                 .gold_label.value_counts().to_dict())

    curve = pd.read_csv(args.curves / "curve_summary.csv")
    n = curve["N"].to_numpy(float)
    c = curve["C_mean"].to_numpy(float)
    frac = np.array([sum(1 for i in unscoreable if per_item.get(i, 0) >= int(N))
                     / int((per_item >= int(N)).sum()) for N in n])

    k_median = float(pd.read_csv(args.constants)
                     .query("dataset == 'Snapshot_Serengeti'")["K_median"].iloc[0])
    k_removed = fit_k(n, c / (1.0 - frac))

    row = {
        "n_events": int(len(gold)),
        "n_unscoreable": int(len(unscoreable)),
        "breakdown": "; ".join(f"{k}={v}" for k, v in sorted(breakdown.items(),
                                                            key=lambda kv: -kv[1])),
        "fraction_at_low_N": round(float(frac[0]), 4),
        "fraction_at_grid_bound": round(float(frac[-1]), 4),
        "K_bootstrap_median": round(k_median, 4),
        "K_with_unscoreable_removed": round(k_removed, 4),
        "K_shift_pct": round(100.0 * (k_removed / k_median - 1.0), 1),
    }
    print(f"unscoreable {row['n_unscoreable']} of {row['n_events']} ({row['breakdown']})")
    print(f"fraction {100*row['fraction_at_low_N']:.2f}% at low N, "
          f"{100*row['fraction_at_grid_bound']:.2f}% at the grid bound")
    print(f"K {row['K_bootstrap_median']} -> {row['K_with_unscoreable_removed']} "
          f"({row['K_shift_pct']}%)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
