"""Summarize a replicate-count check: does raising B from 200 to 2000 change the reported
intervals or the fitted constants?

Reads two stage-8 curve directories (the shipped B = 200 curves and a B = 2000 rerun),
refits the Michaelis-Menten family to every replicate curve in each, and reports, per
dataset: the median K at both B, and the 2.5-97.5 percentile interval of the integer
stopping count N* at the four representative weights under the closure semantics
(integer reference M = first grid N reaching 95% of the fitted gain, search over
1..N_support). The manuscript reports that nine of the twelve interval widths are unchanged, that three
widen by one contributor, and that the medians of K agree to within 0.003 on the two
primary datasets.

The B = 2000 intermediates are not shipped (they are about ten times the size of the
B = 200 curves); this summary is. Regenerate the B = 2000 curves with

    python scripts/compute_stage8_curves.py data/processed/<dataset> --mode <mode> \\
        --B 2000 --seed 20260709 --maxN 50 --outdir reproduced/stage8_curves_B2000

for each of the three admitted datasets, then run this script.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATASETS = [
    ("CIFAR-10H", "gold_accuracy", 50),
    ("ChaosNLI", "reference_distribution", 50),
    ("Snapshot_Serengeti", "gold_accuracy", 21),
]
LAMBDAS = (0.25, 0.50, 0.75, 0.90)


def mm(n, c0, a, k):
    return c0 + a * n / (k + n)


def fit_k(n, c):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, _ = curve_fit(mm, n, c, p0=[c.min(), c.max() - c.min(), 1.0],
                         bounds=([0, 0, 1e-6], [1.2, 1.2, 1000.0]), maxfev=20000)
    return p


def nstar_per_replicate(params, support):
    """closure semantics: asymptotic-gain normalization, search over 1..support"""
    c0, a, k = params
    gain = lambda N: mm(N, c0, a, k) - mm(1.0, c0, a, k)
    target = 0.95 * a * (1.0 - 1.0 / (k + 1.0))
    grid = np.arange(1, 1001, dtype=float)
    hit = np.where(gain(grid) >= target)[0]
    if hit.size == 0:
        return [np.nan] * len(LAMBDAS)
    M = float(grid[hit[0]])
    n = np.arange(1, support + 1, dtype=float)
    # asymptotic-gain normalization, matching recompute_final_closure; M is retained only
    # because the reference is still needed to report the ratio elsewhere
    S = gain(n) / (a * (1.0 - 1.0 / (k + 1.0)))
    out = []
    for lam in LAMBDAS:
        eta = (1.0 - lam) / (lam * support)
        u = S - eta * n
        out.append(int(n[int(np.nanargmax(u))]))
    return out


def summarize(curves_dir: Path, dataset: str, mode: str, support: int):
    df = pd.read_csv(curves_dir / dataset / mode / "curve_bootstrap.csv")
    ks, nstars = [], []
    for _, g in df.groupby("bootstrap"):
        n = g["N"].to_numpy(float); c = g["C"].to_numpy(float)
        try:
            p = fit_k(n, c)
        except Exception:
            continue
        ks.append(p[2]); nstars.append(nstar_per_replicate(p, support))
    ks = np.asarray(ks); ns = np.asarray(nstars, dtype=float)
    lo = np.nanpercentile(ns, 2.5, axis=0); hi = np.nanpercentile(ns, 97.5, axis=0)
    return len(ks), float(np.median(ks)), lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b200", type=Path, default=Path("expected/stage8_curves"))
    ap.add_argument("--b2000", type=Path, default=Path("reproduced/stage8_curves_B2000"))
    ap.add_argument("--out", type=Path, default=Path("expected/b2000/b2000_validation_summary.csv"))
    args = ap.parse_args()
    rows = []
    for dataset, mode, support in DATASETS:
        b1, k1, lo1, hi1 = summarize(args.b200, dataset, mode, support)
        b2, k2, lo2, hi2 = summarize(args.b2000, dataset, mode, support)
        w1 = hi1 - lo1; w2 = hi2 - lo2
        row = {"dataset": dataset, "mode": mode, "B_shipped": b1, "B_rerun": b2,
               "K_median_shipped": round(k1, 4), "K_median_rerun": round(k2, 4),
               "K_median_abs_diff": round(abs(k2 - k1), 4)}
        for lam, a1, b1_, a2, b2_ in zip(LAMBDAS, lo1, hi1, lo2, hi2):
            row[f"Nstar_CI_lam{lam:.2f}_shipped"] = f"[{int(a1)}, {int(b1_)}]"
            row[f"Nstar_CI_lam{lam:.2f}_rerun"] = f"[{int(a2)}, {int(b2_)}]"
        row["all_widths_equal"] = bool(np.all(w1 == w2))
        row["max_width_change"] = int(np.max(np.abs(w2 - w1)))
        rows.append(row)
        print(f"{dataset:20s} K {k1:.4f} -> {k2:.4f} (|d|={abs(k2-k1):.4f}) "
              f"widths equal: {row['all_widths_equal']} max change {row['max_width_change']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
