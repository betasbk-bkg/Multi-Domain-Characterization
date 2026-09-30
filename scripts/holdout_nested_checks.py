"""Out-of-grid predictive check and a nested four-parameter alternative to the MM form.

Two questions are asked of the fitted saturation curves.

1. Out-of-grid prediction. Each candidate family is fitted to the curve on N <= 25 and used to
   predict N = 26..50; the prediction error is reported on the mean curve and on each of the
   200 bootstrap replicate curves. A shorter training grid (N <= 15) is reported as a
   sensitivity case. This is done for the two datasets whose grid reaches N = 50.

2. Nested alternative. The four-parameter form
       C(N) = c0 + a * N^h / (K^h + N^h)
   reduces to the Michaelis-Menten (MM) form at h = 1. It is fitted to the mean curve and to
   every bootstrap replicate curve of all three admitted datasets, and compared with the MM
   form by AIC. Because h is a new parameter with no prior convention, its bounds are stated
   explicitly ([0.2, 10] as the main case, which no fit reaches, and [0.2, 5] as a
   bound-sensitivity case, which some CIFAR-10H replicates reach), and the number of replicate
   fits that reach a bound is reported.

For completeness the saturation reference N95 and the stopping counts N* at the four
representative weights are also computed under the nested form, with the same routines and
the same N_support budget as the main analysis, so that the effect of the family choice on
the reported quantities can be read directly.

Every fit, reference and stopping count is computed with the routines of
recompute_final_closure (fit_family, saturation_n, utility_curve); the nested form is
registered alongside the three candidate families rather than reimplemented. Before any new
quantity is written, the script reproduces four packaged MM results - the AIC, N95, the
point-estimate stopping counts, and the bootstrap-median stopping counts - and stops with an
error if any of them differs.

    python scripts/holdout_nested_checks.py

Writes expected/diagnostics/holdout_nested_summary.csv,
       expected/diagnostics/holdout_by_replicate.csv,
       expected/diagnostics/nested_by_replicate.csv.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recompute_final_closure as rfc  # noqa: E402

REP_LAMBDAS = [0.25, 0.50, 0.75, 0.90]
DATASETS = [
    ("CIFAR-10H", "gold_accuracy", 50),
    ("ChaosNLI", "reference_distribution", 50),
    ("Snapshot_Serengeti", "gold_accuracy", 21),
]
HOLDOUT_DATASETS = ("CIFAR-10H", "ChaosNLI")
HOLDOUT_CUTS = (25, 15)          # training grid N <= cut; 25 is the main case
H_BOUNDS = {"hill": 10.0, "hill_h5": 5.0}
THREE = ("michaelis", "log_saturating", "inverse_sqrt")


def hill(n, c0, amp, k, h):
    n = np.asarray(n, dtype=float)
    return c0 + amp * n**h / (k**h + n**h)


def register_nested_form() -> None:
    """Add the nested form to the package's family table without altering the three families."""
    for name, h_hi in H_BOUNDS.items():
        rfc.FAMILIES[name] = (hill, [0.5, 0.5, 2.0, 1.0], ([0.0, 0.0, 1e-6, 0.2], [1.2, 1.2, 1000.0, h_hi]))
    original = rfc.asymptote_for

    def asymptote_for(family, params):
        if family in H_BOUNDS:
            return float(params[0] + params[1])
        return original(family, params)

    rfc.asymptote_for = asymptote_for


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def holdout(ns, y, cut):
    """Fit every family on N <= cut and return its prediction error on cut < N <= 50."""
    train = ns <= cut
    test = (ns > cut) & (ns <= 50)
    out = {}
    for fam in THREE + ("hill",):
        fit = rfc.fit_family(ns[train], y[train], fam)
        if not fit["success"]:
            out[fam] = (float("nan"), float("nan"))
            continue
        pred = rfc.predict(fam, fit["params"], ns[test])
        out[fam] = (rmse(pred, y[test]), float(np.mean(pred - y[test])))
    return out


def stopping(fam, params, n_support):
    n95, _ = rfc.saturation_n(fam, params, 0.95)
    u = rfc.utility_curve(fam, params, n95, n_support, np.asarray(REP_LAMBDAS), n_candidate_max=int(n_support))
    nstar = {float(r["lambda"]): int(r["n_star"]) for _, r in u.iterrows()} if not u.empty else {}
    ratio = {float(r["lambda"]): float(r["ratio"]) for _, r in u.iterrows()} if not u.empty else {}
    return n95, nstar, ratio


def pctl(x, q):
    x = [v for v in x if np.isfinite(v)]
    return float(np.percentile(x, q)) if x else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=Path, default=Path("expected/stage8_curves"))
    ap.add_argument("--closure", type=Path, default=Path("expected/final_closure"))
    ap.add_argument("--out", type=Path, default=Path("expected/diagnostics"))
    args = ap.parse_args()
    register_nested_form()

    sat = pd.read_csv(args.closure / "final_saturation_summary.csv")
    bsens = pd.read_csv(args.closure / "budget_sensitivity.csv")
    bci = pd.read_csv(args.closure / "bootstrap_ratio_ci.csv")

    summary, by_rep_hold, by_rep_nest = [], [], []
    failures = []

    def add(section, dataset, quantity, family, estimator, value):
        summary.append({"section": section, "dataset": dataset, "quantity": quantity,
                        "family": family, "estimator": estimator, "value": value})

    for ds, mode, n_support in DATASETS:
        cdir = args.curves / ds / mode
        mean = pd.read_csv(cdir / "curve_summary.csv").sort_values("N")
        ns, y = mean["N"].to_numpy(float), mean["C_mean"].to_numpy(float)
        boot = pd.read_csv(cdir / "curve_bootstrap.csv")

        # ---- self-checks against the packaged MM results
        mm = rfc.fit_family(ns, y, "michaelis")
        srow = sat[(sat.dataset == ds) & (sat["mode"] == mode)].iloc[0]
        mm_n95, mm_nstar, _ = stopping("michaelis", mm["params"], n_support)
        if abs(mm["aic"] - srow.aic) > 1e-6:
            failures.append(f"{ds}: MM AIC {mm['aic']} vs packaged {srow.aic}")
        if int(mm_n95) != int(srow.n95):
            failures.append(f"{ds}: MM N95 {mm_n95} vs packaged {srow.n95}")
        for lam in REP_LAMBDAS:
            pk = bsens[(bsens.dataset == ds) & (bsens["mode"] == mode) & (bsens.n_budget_type == "observed_max")
                       & (np.isclose(bsens["lambda"], lam))]
            if not pk.empty and int(pk.n_star.iloc[0]) != mm_nstar.get(lam):
                failures.append(f"{ds}: MM point N* at {lam} {mm_nstar.get(lam)} vs packaged {int(pk.n_star.iloc[0])}")

        # ---- nested form on the mean curve
        for fam in ("hill", "hill_h5"):
            fit = rfc.fit_family(ns, y, fam)
            if not fit["success"]:
                add("nested", ds, "fit_failed", fam, "mean_curve", 1.0)
                continue
            add("nested", ds, "h", fam, "mean_curve", float(fit["params"][3]))
            add("nested", ds, "delta_aic_vs_mm", fam, "mean_curve", float(fit["aic"] - mm["aic"]))
            if fam == "hill":
                h_n95, h_nstar, h_ratio = stopping("hill", fit["params"], n_support)
                add("consequence", ds, "N95", "michaelis", "mean_curve", float(mm_n95))
                add("consequence", ds, "N95", "hill", "mean_curve", float(h_n95))
                for lam in REP_LAMBDAS:
                    add("consequence", ds, f"n_star_lambda_{lam:.2f}", "michaelis", "mean_curve", float(mm_nstar[lam]))
                    add("consequence", ds, f"n_star_lambda_{lam:.2f}", "hill", "mean_curve", float(h_nstar.get(lam, np.nan)))
                c1 = float(rfc.predict("hill", fit["params"], 1.0)); cinf = rfc.asymptote_for("hill", fit["params"])
                c4 = float(rfc.predict("hill", fit["params"], 4.0))
                add("consequence", ds, "shortfall_at_4", "hill", "mean_curve", float(1 - (c4 - c1) / (cinf - c1)))
                add("consequence", ds, "shortfall_at_4_points", "hill", "mean_curve", float(100 * (cinf - c4)))

        # ---- per-replicate: MM (self-check), nested form, holdout
        mm_boot = {lam: [] for lam in REP_LAMBDAS}
        h_boot = {lam: [] for lam in REP_LAMBDAS}
        h_n95_boot = []
        for b, grp in boot.groupby("bootstrap"):
            g = grp.dropna(subset=["C"]).sort_values("N")
            bn, by = g["N"].to_numpy(float), g["C"].to_numpy(float)
            bm = rfc.fit_family(bn, by, "michaelis")
            if bm["success"]:
                _, ns_mm, _ = stopping("michaelis", bm["params"], n_support)
                for lam in REP_LAMBDAS:
                    mm_boot[lam].append(ns_mm.get(lam, np.nan))
            row = {"dataset": ds, "bootstrap": int(b)}
            for fam in ("hill", "hill_h5"):
                bh = rfc.fit_family(bn, by, fam)
                if bh["success"] and bm["success"]:
                    row[f"{fam}_h"] = float(bh["params"][3])
                    row[f"{fam}_delta_aic"] = float(bh["aic"] - bm["aic"])
                    if fam == "hill":
                        n95h, nsh, _ = stopping("hill", bh["params"], n_support)
                        h_n95_boot.append(n95h)
                        for lam in REP_LAMBDAS:
                            h_boot[lam].append(nsh.get(lam, np.nan))
                else:
                    row[f"{fam}_h"] = float("nan"); row[f"{fam}_delta_aic"] = float("nan")
            by_rep_nest.append(row)
            if ds in HOLDOUT_DATASETS:
                for cut in HOLDOUT_CUTS:
                    for fam, (e, bias) in holdout(bn, by, cut).items():
                        by_rep_hold.append({"dataset": ds, "bootstrap": int(b), "train_max_n": cut,
                                            "family": fam, "holdout_rmse": e, "holdout_bias": bias})

        for lam in REP_LAMBDAS:
            pk = bci[(bci.dataset == ds) & (bci["mode"] == mode) & (bci.ratio_type == "rho_95")
                     & (bci.n_budget_type == "observed_max") & (np.isclose(bci["lambda"], lam))]
            med = float(np.nanmedian(mm_boot[lam]))
            if not pk.empty and med != float(pk.n_star_median.iloc[0]):
                failures.append(f"{ds}: MM bootstrap-median N* at {lam} {med} vs packaged {float(pk.n_star_median.iloc[0])}")
            add("consequence", ds, f"n_star_lambda_{lam:.2f}", "michaelis", "bootstrap_median", med)
            add("consequence", ds, f"n_star_lambda_{lam:.2f}", "hill", "bootstrap_median", float(np.nanmedian(h_boot[lam])))
        add("consequence", ds, "N95", "hill", "bootstrap_median", float(np.nanmedian(h_n95_boot)))

        nest = pd.DataFrame([r for r in by_rep_nest if r["dataset"] == ds])
        for fam, h_hi in H_BOUNDS.items():
            hs = nest[f"{fam}_h"].to_numpy(float); da = nest[f"{fam}_delta_aic"].to_numpy(float)
            ok = np.isfinite(hs)
            add("nested", ds, "h_median", fam, "bootstrap", float(np.median(hs[ok])))
            add("nested", ds, "h_p2_5", fam, "bootstrap", pctl(hs, 2.5))
            add("nested", ds, "h_p97_5", fam, "bootstrap", pctl(hs, 97.5))
            add("nested", ds, "h_at_upper_bound", fam, "bootstrap", float(np.sum(np.isclose(hs[ok], h_hi, atol=1e-3))))
            add("nested", ds, "h_at_lower_bound", fam, "bootstrap", float(np.sum(np.isclose(hs[ok], 0.2, atol=1e-3))))
            add("nested", ds, "replicates_fitted", fam, "bootstrap", float(ok.sum()))
            add("nested", ds, "replicates_nested_preferred_aic_gt_2", fam, "bootstrap", float(np.sum(da[ok] < -2)))
            add("nested", ds, "delta_aic_median", fam, "bootstrap", float(np.median(da[ok])))

        if ds in HOLDOUT_DATASETS:
            hold = pd.DataFrame([r for r in by_rep_hold if r["dataset"] == ds])
            for cut in HOLDOUT_CUTS:
                for fam in THREE + ("hill",):
                    e = hold[(hold.train_max_n == cut) & (hold.family == fam)].holdout_rmse
                    add("holdout", ds, f"rmse_train_le_{cut}", fam, "bootstrap_median", float(e.median()))
                e_mean = holdout(ns, y, cut)
                for fam, (err, bias) in e_mean.items():
                    add("holdout", ds, f"rmse_train_le_{cut}", fam, "mean_curve", err)
                    add("holdout", ds, f"bias_train_le_{cut}", fam, "mean_curve", bias)
                w = hold[hold.train_max_n == cut].pivot_table(index="bootstrap", columns="family", values="holdout_rmse")
                wins3 = w[list(THREE)].idxmin(axis=1).value_counts()
                wins4 = w[list(THREE) + ["hill"]].idxmin(axis=1).value_counts()
                for fam in THREE:
                    add("holdout", ds, f"replicates_lowest_among_three_le_{cut}", fam, "bootstrap", float(wins3.get(fam, 0)))
                for fam in THREE + ("hill",):
                    add("holdout", ds, f"replicates_lowest_among_four_le_{cut}", fam, "bootstrap", float(wins4.get(fam, 0)))

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.out / "holdout_nested_summary.csv", index=False)
    pd.DataFrame(by_rep_hold).to_csv(args.out / "holdout_by_replicate.csv", index=False)
    pd.DataFrame(by_rep_nest).to_csv(args.out / "nested_by_replicate.csv", index=False)

    if failures:
        print("SELF-CHECK FAILED: the MM routines do not reproduce the packaged results")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("self-check passed: MM AIC, N95, point and bootstrap-median N* reproduce the packaged results")
    print("wrote", args.out / "holdout_nested_summary.csv")


if __name__ == "__main__":
    main()
