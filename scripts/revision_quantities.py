"""Regenerate the derived quantities the manuscript reports outside the closure tables.

The closure tables are produced by `recompute_final_closure.py` and the appendix constants by
`refit_replicate_constants.py`. A third group of numbers appears only in the manuscript prose:
the runner-up family's asymptote, the effect of shortening the estimation grid, the shortfall of
a fixed annotation redundancy against the achievable normalized gain, and the item coverage of
the Snapshot Serengeti grid. Earlier versions of this package did not produce them, so a reader
could not check them. This script does.

Definitions used here, matching the manuscript:

    S(N)   normalized fitted gain, (C(N) - C(1)) / (C(N95) - C(1))
    N95    smallest N reaching 95% of the fitted asymptotic gain from N = 1
    shortfall(N) = 1 - S(N), the fraction of the achievable gain a fixed redundancy leaves unused

Usage:
    python scripts/revision_quantities.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATASETS = [
    ("CIFAR-10H", "gold_accuracy", 50, 22, 15),
    ("ChaosNLI", "reference_distribution", 50, 27, 15),
    ("Snapshot_Serengeti", "gold_accuracy", 21, 30, 10),
]
LAMBDAS = (0.25, 0.50, 0.75, 0.90)
FIXED_REDUNDANCY = (3, 4, 5)
SAT = 0.95


def michaelis_menten(n, c0, amp, k):
    return c0 + amp * n / (k + n)


def log_saturating(n, c0, amp, k):
    x = np.log1p(n)
    return c0 + amp * x / (k + x)


def fit(model, n, c, p0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, _ = curve_fit(model, n, c, p0=p0,
                              bounds=([0.0, 0.0, 1e-6], [1.2, 1.2, 1e4]), maxfev=200000)
    return params


def load(curves: Path, dataset: str, mode: str):
    df = pd.read_csv(curves / dataset / mode / "curve_summary.csv")
    return df["N"].to_numpy(float), df["C_mean"].to_numpy(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=Path, default=Path("expected/stage8_curves"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path,
                    default=Path("expected/final_closure/revision_quantities.csv"))
    args = ap.parse_args()

    rows = []
    for dataset, mode, support, n95, short in DATASETS:
        n, c = load(args.curves, dataset, mode)

        mm = fit(michaelis_menten, n, c, [c.min(), c.max() - c.min(), 1.0])
        k_full = float(mm[2])

        # runner-up family, to show the null is exact only for the admitted family. The AIC
        # margin is also refitted here as an independent cross-check; the authoritative Table IV
        # values are the closure pipeline's (final_saturation_summary.csv / model_fit_comparison.csv).
        ls = fit(log_saturating, n, c, [c.min(), c.max() - c.min(), 1.0])
        runner_up_asymptote = float(ls[0] + ls[1])

        def aic(model, params):
            rss = float(np.sum((c - model(n, *params)) ** 2))
            return len(n) * np.log(max(rss, 1e-300) / len(n)) + 2 * len(params)

        def inverse_sqrt(nn, L, amp, k):
            return L - amp / np.sqrt(nn + k)

        isq = fit(inverse_sqrt, n, c, [c.max(), 0.5, 1.0])
        aic_mm, aic_ls, aic_isq = aic(michaelis_menten, mm), aic(log_saturating, ls), aic(inverse_sqrt, isq)
        delta_aic = min(aic_ls, aic_isq) - aic_mm

        # grid dependence: refit on a shorter grid, compare K and the continuous ratio
        keep = n <= short
        k_short = float(fit(michaelis_menten, n[keep], c[keep],
                            [c[keep].min(), c[keep].max() - c[keep].min(), 1.0])[2])

        def rho(k, lam):
            eta = (1.0 - lam) / (lam * support)
            return (np.sqrt((k + 1.0) / (SAT * eta)) - k) / (20.0 + 19.0 * k)

        d_rho = max(abs(rho(k_full, l) - rho(k_short, l)) for l in LAMBDAS)

        # shortfall of a fixed redundancy against the achievable normalized gain
        def S(N):
            return ((michaelis_menten(N, *mm) - michaelis_menten(1.0, *mm))
                    / (michaelis_menten(float(n95), *mm) - michaelis_menten(1.0, *mm)))

        shortfall = {N: 1.0 - float(S(float(N))) for N in FIXED_REDUNDANCY}

        # item coverage of the estimation grid
        path = args.processed / dataset / "labels_long.csv"
        if not path.exists():
            path = path.with_suffix(".csv.gz")
        counts = pd.read_csv(path, usecols=["item_id"]).groupby("item_id").size()
        coverage = float(np.mean(counts >= support))

        rows.append({
            "dataset": dataset, "mode": mode,
            "K_full_grid": round(k_full, 4),
            "K_short_grid": round(k_short, 4),
            "short_grid_bound": short,
            "K_shift_pct": round(100.0 * abs(k_short - k_full) / k_full, 1),
            "max_abs_delta_rho95": round(d_rho, 4),
            "runner_up_asymptote": round(runner_up_asymptote, 3),
            "aic_michaelis_crosscheck": round(aic_mm, 2),
            "delta_aic_crosscheck": round(delta_aic, 2),
            "shortfall_at_3": round(shortfall[3], 3),
            "shortfall_at_4": round(shortfall[4], 3),
            "shortfall_at_5": round(shortfall[5], 3),
            "grid_item_coverage": round(coverage, 3),
        })
        r = rows[-1]
        print(f"{dataset:20s} K {r['K_full_grid']:.4f}->{r['K_short_grid']:.4f} "
              f"({r['K_shift_pct']:.1f}%) dRho={r['max_abs_delta_rho95']:.4f} "
              f"runner-up asym={r['runner_up_asymptote']:.3f} "
              f"shortfall 4/5={r['shortfall_at_4']:.3f}/{r['shortfall_at_5']:.3f} "
              f"coverage={r['grid_item_coverage']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
