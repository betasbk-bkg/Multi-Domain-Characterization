"""Regenerate the derived quantities the manuscript reports outside the closure tables.

The closure tables are produced by `recompute_final_closure.py` and the appendix constants by
`refit_replicate_constants.py`. A third group of numbers appears only in the manuscript prose:
the runner-up family's asymptote, the effect of shortening the estimation grid, the shortfall of
a fixed annotation redundancy against the achievable normalized gain, and the item coverage of
the Snapshot Serengeti grid. Earlier versions of this package did not produce them, so a reader
could not check them. This script does.

Definitions used here, matching the manuscript:

    S(N)   normalized fitted gain, (C(N) - C(1)) / (C(inf) - C(1))
    N95    smallest N reaching 95% of the fitted asymptotic gain from N = 1
    shortfall_asym(N) = 1 - S(N) under the asymptotic-gain normalization the manuscript uses.
    The shortfall_vs_N95 and raw_gap_to_N95 columns give the same comparison against the
    saturation reference instead; they are retained for reference and are not quoted in the
    manuscript, whose reported shortfalls are the asymptotic ones. Values are given in normalized
    and in raw units

Usage:
    python scripts/revision_quantities.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
                              bounds=([0.0, 0.0, 1e-6], [1.2, 1.2, 1000.0]), maxfev=200000)
    return params


def load(curves: Path, dataset: str, mode: str):
    df = pd.read_csv(curves / dataset / mode / "curve_summary.csv")
    return df["N"].to_numpy(float), df["C_mean"].to_numpy(float)


def bootstrap_model_selection(curves: Path, dataset: str, mode: str):
    """Refit the three admissible families to every bootstrap replicate curve and count
    how often each is selected by AIC; also return the bootstrap distribution of the
    AIC margin of the selected family over the runner-up."""
    df = pd.read_csv(curves / dataset / mode / "curve_bootstrap.csv")
    wins = {"michaelis": 0, "log_saturating": 0, "inverse_sqrt": 0}
    margins = []

    def inverse_sqrt(nn, L, amp, k):
        return L - amp / np.sqrt(nn + k)

    def aic(model, params, n, c):
        rss = float(np.sum((c - model(n, *params)) ** 2))
        return len(n) * np.log(max(rss, 1e-300) / len(n)) + 2 * len(params)

    for _, g in df.groupby("bootstrap"):
        n = g["N"].to_numpy(float); c = g["C"].to_numpy(float)
        try:
            a_mm = aic(michaelis_menten, fit(michaelis_menten, n, c, [c.min(), c.max() - c.min(), 1.0]), n, c)
            a_ls = aic(log_saturating, fit(log_saturating, n, c, [c.min(), c.max() - c.min(), 1.0]), n, c)
            a_is = aic(inverse_sqrt, fit(inverse_sqrt, n, c, [c.max(), 0.5, 1.0]), n, c)
        except Exception:
            continue
        scores = {"michaelis": a_mm, "log_saturating": a_ls, "inverse_sqrt": a_is}
        best = min(scores, key=scores.get); wins[best] += 1
        rest = sorted(v for k2, v in scores.items() if k2 != best)
        margins.append(rest[0] - scores[best])
    m = np.asarray(margins)
    return wins, (float(np.percentile(m, 2.5)), float(np.median(m)), float(np.percentile(m, 97.5))), int(m.size)


def reference_fraction_sensitivity(mm_params, support: int, lam: float = 0.90):
    """N* at the representative weight when the saturation reference is read at 90, 95 and
    99 percent of the fitted asymptotic gain. Under asymptotic-gain normalization the
    objective does not contain the reference, so N* is invariant across q and only the
    reported ratio N*/N_q moves; this function records both so the invariance is checked
    rather than asserted."""
    c0, amp, k = mm_params
    gain = lambda N: michaelis_menten(N, c0, amp, k) - michaelis_menten(1.0, c0, amp, k)
    out = {}
    for q in (0.90, 0.95, 0.99):
        target = q * amp * (1.0 - 1.0 / (k + 1.0))  # q * (C(inf) - C(1))
        grid = np.arange(1, 1001, dtype=float)
        idx = np.where(gain(grid) >= target)[0]
        Nq = int(grid[idx[0]]) if idx.size else None
        if Nq is None:
            out[q] = (None, None, None); continue
        n = np.arange(1, support + 1, dtype=float)
        S = gain(n) / (amp * (1.0 - 1.0 / (k + 1.0)))   # asymptotic-gain normalization
        eta = (1.0 - lam) / (lam * support)
        u = S - eta * n
        Ns = int(n[int(np.nanargmax(u))])
        out[q] = (Nq, Ns, Ns / Nq)
    return out


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
            # asymptotic-gain normalization: N*(eta) = sqrt((K+1)/eta) - K
            return (np.sqrt((k + 1.0) / eta) - k) / (20.0 + 19.0 * k)

        d_rho = max(abs(rho(k_full, l) - rho(k_short, l)) for l in LAMBDAS)

        # shortfall of a fixed redundancy, in two units: against the asymptotic gain, which
        # is what the utility normalizes by, and against the gain at the saturation
        # reference, which is what the reported ratio is indexed to
        gain = lambda N: michaelis_menten(N, *mm) - michaelis_menten(1.0, *mm)
        g_inf = mm[1] * (1.0 - 1.0 / (mm[2] + 1.0))
        shortfall = {N: 1.0 - float(gain(float(N)) / g_inf) for N in FIXED_REDUNDANCY}
        shortfall_ref = {N: 1.0 - float(gain(float(N)) / gain(float(n95))) for N in FIXED_REDUNDANCY}
        raw_gap_ref = {N: float(gain(float(n95)) - gain(float(N))) for N in FIXED_REDUNDANCY}

        # item coverage of the estimation grid
        path = args.processed / dataset / "labels_long.csv"
        if not path.exists():
            path = path.with_suffix(".csv.gz")
        counts = pd.read_csv(path, usecols=["item_id"]).groupby("item_id").size()
        coverage = float(np.mean(counts >= support))

        wins, (m_lo, m_med, m_hi), nboot = bootstrap_model_selection(args.curves, dataset, mode)
        qsens = reference_fraction_sensitivity(mm, support)
        # quantities the manuscript quotes in the operator's units
        boot = pd.read_csv(args.curves / dataset / mode / "curve_bootstrap.csv")
        def cv_at(N):
            g = boot[boot["N"] == N]["C"]
            return 100.0 * float(g.std(ddof=1)) / float(g.mean()) if len(g) > 1 else float("nan")
        summ = pd.read_csv(args.curves / dataset / mode / "curve_summary.csv")
        spearman_r = float(spearmanr(summ["N"], summ["C_mean"]).statistic)
        def coverage_at(N):
            return float((counts >= N).mean())
        tail = summ[summ["N"] >= 20]["C_mean"].to_numpy(float)
        high_n_increment = float(np.mean(np.diff(tail))) if tail.size > 1 else float("nan")
        Ginf = float(mm[1] * (1.0 - 1.0 / (mm[2] + 1.0)))   # C(inf) - C(1)
        Sbar = lambda N: (N - 1.0) / (mm[2] + N)            # asymptotic-gain normalization
        rows.append({
            "dataset": dataset, "mode": mode,
            "asymptotic_gain_points": round(100.0 * Ginf, 2),
            # quantities the manuscript quotes about the bootstrap curves themselves
            "cv_at_4_pct": round(cv_at(4), 4),
            "cv_at_5_pct": round(cv_at(5), 4),
            "spearman_r_mean_vs_N": round(spearman_r, 4),
            "coverage_at_18": round(coverage_at(18), 4),
            "mean_increment_above_N20": round(high_n_increment, 6),
            "shortfall_at_4_asymptotic": round(1.0 - Sbar(4.0), 3),
            "shortfall_at_5_asymptotic": round(1.0 - Sbar(5.0), 3),
            "shortfall_at_4_points": round(100.0 * (1.0 - Sbar(4.0)) * Ginf, 2),
            "shortfall_at_5_points": round(100.0 * (1.0 - Sbar(5.0)) * Ginf, 2),
            "boot_select_michaelis": wins["michaelis"],
            "boot_select_log_saturating": wins["log_saturating"],
            "boot_select_inverse_sqrt": wins["inverse_sqrt"],
            "boot_n": nboot,
            "delta_aic_boot_lo": round(m_lo, 1), "delta_aic_boot_median": round(m_med, 1), "delta_aic_boot_hi": round(m_hi, 1),
            "q90_Nref": qsens[0.90][0], "q90_Nstar_lam090": qsens[0.90][1], "q90_ratio_lam090": None if qsens[0.90][2] is None else round(qsens[0.90][2], 3),
            "q95_Nref": qsens[0.95][0], "q95_Nstar_lam090": qsens[0.95][1], "q95_ratio_lam090": None if qsens[0.95][2] is None else round(qsens[0.95][2], 3),
            "q99_Nref": qsens[0.99][0], "q99_Nstar_lam090": qsens[0.99][1], "q99_ratio_lam090": None if qsens[0.99][2] is None else round(qsens[0.99][2], 3),
            "K_full_grid": round(k_full, 4),
            "K_short_grid": round(k_short, 4),
            "short_grid_bound": short,
            "K_shift_pct": round(100.0 * abs(k_short - k_full) / k_full, 1),
            "max_abs_delta_rho95": round(d_rho, 4),
            "runner_up_asymptote": round(runner_up_asymptote, 3),
            "aic_michaelis_crosscheck": round(aic_mm, 2),
            "delta_aic_crosscheck": round(delta_aic, 2),
            "shortfall_asym_at_3": round(shortfall[3], 3),
            "shortfall_asym_at_4": round(shortfall[4], 3),
            "shortfall_asym_at_5": round(shortfall[5], 3),
            "shortfall_vs_N95_at_4": round(shortfall_ref[4], 3),
            "raw_gap_to_N95_at_4": round(raw_gap_ref[4], 4),
            "raw_gap_to_N95_at_5": round(raw_gap_ref[5], 4),
            "grid_item_coverage": round(coverage, 3),
        })
        r = rows[-1]
        print(f"{dataset:20s} K {r['K_full_grid']:.4f}->{r['K_short_grid']:.4f} "
              f"({r['K_shift_pct']:.1f}%) dRho={r['max_abs_delta_rho95']:.4f} "
              f"runner-up asym={r['runner_up_asymptote']:.3f} "
              f"shortfall_asym 4/5={r['shortfall_asym_at_4']:.3f}/{r['shortfall_asym_at_5']:.3f} "
              f"gain={r['asymptotic_gain_points']:.2f}pt short4={r['shortfall_at_4_asymptotic']:.3f}({r['shortfall_at_4_points']:.2f}pt) coverage={r['grid_item_coverage']:.3f} | MM selected {r['boot_select_michaelis']}/{r['boot_n']} dAIC[{r['delta_aic_boot_lo']},{r['delta_aic_boot_median']},{r['delta_aic_boot_hi']}] | q90 N*/Nref={r['q90_ratio_lam090']} q95={r['q95_ratio_lam090']} q99={r['q99_ratio_lam090']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
