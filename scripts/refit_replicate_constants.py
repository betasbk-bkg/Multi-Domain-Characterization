"""Refit the Michaelis-Menten family to each bootstrap replicate curve.

The closure tables report a single fitted curve per dataset. The manuscript appendix
instead needs the distribution of the half-saturation constant across replicates,
because the whole map from the cost weight to the stopping ratio is fixed by that one
constant once the family is adopted. This script produces those quantities:

    K            half-saturation constant, median and percentile interval
    N95          derived reference, 20 + 19K, continuous and rounded up
    eta_c        exact crossing for the integer optimizer, S(M) - S(M-1) = (K+1)/((K+M)(K+M-1)),
                 under asymptotic-gain normalization
    eta_c_continuous  continuous approximation, the price at which the unconstrained optimum
                 of (A.5) equals M: (K + 1) / (M + K)^2
    rmse_median  median residual RMSE of the replicate fits
    c0_at_bound  fraction of replicates whose offset reaches its lower bound

`c0_at_bound` is reported because the offset is a nuisance parameter that cancels in
the normalized performance, so it may sit at its bound without affecting any stopping
quantity. It is counted with an explicit tolerance, since whether a bounded optimizer
returns exactly zero or a value a few ulp above it is a property of the solver rather
than of the data.

Usage:
    python scripts/refit_replicate_constants.py
    python scripts/refit_replicate_constants.py --curves reproduced/stage8_curves_B2000
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import math

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Fraction of the asymptotic gain that defines the saturation reference.
SAT_LEVEL = 0.95
# c0 is counted as sitting at its lower bound when it is within this of zero.
C0_BOUND_TOL = 1e-6

DATASETS = [
    ("CIFAR-10H", "gold_accuracy"),
    ("ChaosNLI", "reference_distribution"),
    ("Snapshot_Serengeti", "gold_accuracy"),
]


def michaelis_menten(n, c0, amp, k):
    return c0 + amp * n / (k + n)


def fit_replicates(curves_dir: Path, dataset: str, mode: str):
    path = curves_dir / dataset / mode / "curve_bootstrap.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    ks, rmses, c0s = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, group in df.groupby("bootstrap"):
            n = group["N"].to_numpy(dtype=float)
            c = group["C"].to_numpy(dtype=float)
            try:
                params, _ = curve_fit(
                    michaelis_menten, n, c,
                    p0=[c.min(), c.max() - c.min(), 1.0],
                    bounds=([0.0, 0.0, 1e-6], [1.2, 1.2, 1000.0]),
                    maxfev=20000,
                )
            except Exception:
                continue
            c0s.append(float(params[0]))
            ks.append(float(params[2]))
            rmses.append(float(np.sqrt(np.mean((michaelis_menten(n, *params) - c) ** 2))))
    return np.asarray(ks), np.asarray(rmses), np.asarray(c0s)


def summarise(dataset: str, mode: str, ks, rmses, c0s) -> dict:
    k_med = float(np.median(ks))
    k_lo, k_hi = (float(v) for v in np.percentile(ks, [2.5, 97.5]))
    n95_cont = 20.0 + 19.0 * k_med
    return {
        "dataset": dataset,
        "mode": mode,
        "n_replicates": int(ks.size),
        "K_median": round(k_med, 4),
        "K_lo": round(k_lo, 4),
        "K_hi": round(k_hi, 4),
        "N95_continuous": round(n95_cont, 2),
        "N95_integer": int(np.ceil(n95_cont)),
        # eta_c falls as K rises, so the interval bounds swap.
        # Under asymptotic-gain normalization S(N) = (N-1)/(K+N), the unconstrained optimum
        # is N*(eta) = sqrt((K+1)/eta) - K, so the price at which stopping reaches the
        # reference M is eta_c = (K+1)/(M+K)^2. The reference enters only here, not in the
        # objective, so eta_c moves with the convention while N*(eta) does not.
        # exact crossing for the integer optimizer: the price at which the gain from M-1 to M
        # no longer pays for itself, S(M) - S(M-1) = (K+1)/((K+M)(K+M-1))
        "eta_c": round((k_med + 1.0) / ((k_med + math.ceil(n95_cont)) * (k_med + math.ceil(n95_cont) - 1.0)), 5),
        # continuous approximation, where the unconstrained optimum of (A.5) equals M
        "eta_c_continuous": round((k_med + 1.0) / (math.ceil(n95_cont) + k_med) ** 2, 5),
        
        # interval bounds under the same discrete threshold, each evaluated at its own K
        # and the reference that K implies; eta_c falls as K rises, so the bounds swap
        "eta_c_lo": round((k_hi + 1.0) / ((k_hi + math.ceil(20.0 + 19.0 * k_hi))
                                          * (k_hi + math.ceil(20.0 + 19.0 * k_hi) - 1.0)), 5),
        "eta_c_hi": round((k_lo + 1.0) / ((k_lo + math.ceil(20.0 + 19.0 * k_lo))
                                          * (k_lo + math.ceil(20.0 + 19.0 * k_lo) - 1.0)), 5),
        "rmse_median": round(float(np.median(rmses)), 6),
        "c0_at_bound_frac": round(float(np.mean(c0s < C0_BOUND_TOL)), 3),
        "c0_bound_tol": C0_BOUND_TOL,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, default=Path("expected/stage8_curves"),
                        help="directory holding <dataset>/<mode>/curve_bootstrap.csv")
    parser.add_argument("--out", type=Path,
                        default=Path("expected/final_closure/replicate_constants.csv"))
    args = parser.parse_args()

    rows = []
    for dataset, mode in DATASETS:
        ks, rmses, c0s = fit_replicates(args.curves, dataset, mode)
        rows.append(summarise(dataset, mode, ks, rmses, c0s))
        row = rows[-1]
        print(f"{dataset:20s} B={row['n_replicates']:5d} "
              f"K={row['K_median']:.4f} [{row['K_lo']:.4f}, {row['K_hi']:.4f}] "
              f"N95={row['N95_integer']:3d} eta_c={row['eta_c']:.5f} "
              f"rmse={row['rmse_median']:.6f} c0_at_bound={row['c0_at_bound_frac']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
