"""Paper B Phase 4 final fitted-saturation and uncertainty closure.

Inputs are fixed by the user:
- DATA_ROOT: existing Stage 8 workspace with standardized data and pilot curves.
- LOCK_ROOT: existing lock/workorder folder.
- OUTPUT_ROOT: new final closure output folder.

The script does not move or delete inputs. It uses the existing
curve_bootstrap.csv files produced from item-level hierarchical bootstrap during
Stage 8 and refits saturation models per bootstrap replicate.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT
STAGE8_CURVE_ROOT = PACKAGE_ROOT / "reproduced" / "stage8_curves"
LOCK_ROOT = PACKAGE_ROOT / "reference_locks_not_required_for_package_reproduction"
OUTPUT_ROOT = PACKAGE_ROOT / "reproduced" / "final_closure"

LAMBDA_GRID = np.linspace(0.01, 0.99, 200)
REP_LAMBDAS = [0.25, 0.50, 0.75, 0.90]
FIXED_CAP = 50


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    mode: str
    role: str
    manuscript_position: str
    curve_dir: Path
    requires_gold: bool
    primary_for_gate: bool


SPECS = [
    DatasetSpec(
        "CIFAR-10H",
        "gold_accuracy",
        "primary_saturating_validation",
        "primary",
        STAGE8_CURVE_ROOT / "CIFAR-10H" / "gold_accuracy",
        True,
        True,
    ),
    DatasetSpec(
        "ChaosNLI",
        "reference_distribution",
        "primary_saturating_validation",
        "primary",
        STAGE8_CURVE_ROOT / "ChaosNLI" / "reference_distribution",
        False,
        True,
    ),
    DatasetSpec(
        "Snapshot_Serengeti",
        "gold_accuracy",
        "supporting_citizen_science_evidence",
        "supporting",
        STAGE8_CURVE_ROOT / "Snapshot_Serengeti" / "gold_accuracy",
        True,
        False,
    ),
    DatasetSpec(
        "CIFAR-10H",
        "reference_distribution",
        "sensitivity_supporting",
        "sensitivity",
        STAGE8_CURVE_ROOT / "CIFAR-10H" / "reference_distribution",
        False,
        False,
    ),
]

LEGACY_COMPONENTS = [
    ("Bingol_USL", "retrograde_backbone", "LEGACY_REANALYSIS_AVAILABLE_SEPARATE_SUPPORT"),
    ("Snow", "epsilon_stopping_bridge", "LEGACY_REANALYSIS_AVAILABLE_SEPARATE_SUPPORT"),
    ("Nitti", "boundary_failure_case", "LEGACY_REANALYSIS_AVAILABLE_SEPARATE_SUPPORT"),
    ("Galaxy_Zoo", "excluded_optional_not_run", "EXCLUDED_NOT_RUN"),
]


def michaelis(n: np.ndarray, c0: float, amp: float, k: float) -> np.ndarray:
    return c0 + amp * n / (k + n)


def log_saturating(n: np.ndarray, c0: float, amp: float, k: float) -> np.ndarray:
    x = np.log1p(n)
    return c0 + amp * x / (k + x)


def inverse_sqrt(n: np.ndarray, L: float, amp: float, k: float) -> np.ndarray:
    return L - amp / np.sqrt(n + k)


FAMILIES: dict[str, tuple[Callable, list[float], tuple[list[float], list[float]]]] = {
    "michaelis": (
        michaelis,
        [0.5, 0.5, 2.0],
        ([0.0, 0.0, 1e-6], [1.2, 1.2, 1000.0]),
    ),
    "log_saturating": (
        log_saturating,
        [0.5, 0.5, 1.0],
        ([0.0, 0.0, 1e-6], [1.2, 1.2, 1000.0]),
    ),
    "inverse_sqrt": (
        inverse_sqrt,
        [1.0, 0.5, 1.0],
        ([0.0, 0.0, 1e-6], [1.2, 10.0, 1000.0]),
    ),
}


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "figure_data").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "manuscript_tables").mkdir(exist_ok=True)


def fit_family(ns: np.ndarray, y: np.ndarray, family: str) -> dict:
    func, p0, bounds = FAMILIES[family]
    p0 = list(p0)
    p0[0] = float(max(0.0, min(1.0, y[0] * 0.9)))
    try:
        popt, _ = curve_fit(func, ns, y, p0=p0, bounds=bounds, maxfev=30000)
        pred = np.asarray(func(ns, *popt), dtype=float)
        residual = y - pred
        rss = float(np.sum(residual**2))
        n = len(y)
        k = len(popt)
        aic = float(n * math.log(max(rss / max(n, 1), 1e-12)) + 2 * k)
        residual_rmse = float(math.sqrt(max(rss / max(n, 1), 0.0)))
        grid = np.arange(1, max(int(ns.max()) + 1, 3), dtype=float)
        grid_pred = np.asarray(func(grid, *popt), dtype=float)
        nondecreasing = bool(np.all(np.diff(grid_pred) >= -1e-5))
        finite = bool(np.all(np.isfinite(pred)) and np.all(np.isfinite(popt)))
        asym = asymptote_for(family, popt)
        admissible = bool(finite and nondecreasing and asym >= max(y) - 1e-4 and residual_rmse < 0.1)
        residual_flag = "ok" if nondecreasing else "nonmonotone_fit"
        return {
            "family": family,
            "success": True,
            "params": popt,
            "aic": aic,
            "residual_rmse": residual_rmse,
            "admissible": admissible,
            "residual_pattern_flag": residual_flag,
            "notes": "fit_ok" if admissible else "fit_not_admissible",
        }
    except Exception as exc:
        return {
            "family": family,
            "success": False,
            "params": None,
            "aic": float("inf"),
            "residual_rmse": float("nan"),
            "admissible": False,
            "residual_pattern_flag": "fit_failed",
            "notes": repr(exc),
        }


def asymptote_for(family: str, params: np.ndarray) -> float:
    if family in ("michaelis", "log_saturating"):
        return float(params[0] + params[1])
    if family == "inverse_sqrt":
        return float(params[0])
    raise ValueError(family)


def predict(family: str, params: np.ndarray, n: np.ndarray | float) -> np.ndarray:
    return np.asarray(FAMILIES[family][0](np.asarray(n, dtype=float), *params), dtype=float)


def best_fit(ns: np.ndarray, y: np.ndarray) -> tuple[dict, list[dict]]:
    fits = [fit_family(ns, y, fam) for fam in FAMILIES]
    admissible = [f for f in fits if f["success"] and f["admissible"]]
    if admissible:
        best = min(admissible, key=lambda f: f["aic"])
    else:
        successful = [f for f in fits if f["success"]]
        best = min(successful, key=lambda f: f["aic"]) if successful else min(fits, key=lambda f: f["aic"])
    return best, fits


def saturation_n(family: str, params: np.ndarray, frac: float, search_max: int = 1000) -> tuple[float, str]:
    c1 = float(predict(family, params, 1.0))
    cref = asymptote_for(family, params)
    if not np.isfinite(cref) or cref <= c1 + 1e-10:
        return float("nan"), "non_identifiable"
    target = c1 + frac * (cref - c1)
    grid = np.arange(1, search_max + 1, dtype=float)
    vals = predict(family, params, grid)
    hit = np.where(vals >= target)[0]
    if len(hit) == 0:
        return float("nan"), "non_identifiable_within_search"
    idx = int(hit[0])
    if idx == 0:
        return 1.0, "observed_or_below_1"
    lo, hi = grid[idx - 1], grid[idx]
    return float(hi), "identified"


def utility_curve(
    family: str,
    params: np.ndarray,
    n95: float,
    budget: float,
    lambdas: np.ndarray,
    n_candidate_max: int | None = None,
) -> pd.DataFrame:
    if not np.isfinite(n95) or n95 <= 1:
        return pd.DataFrame()
    c1 = float(predict(family, params, 1.0))
    cref = float(predict(family, params, n95))
    denom = max(cref - c1, 1e-12)
    max_n = int(max(2, math.floor(n_candidate_max if n_candidate_max is not None else budget)))
    ns = np.arange(1, max_n + 1, dtype=float)
    ctilde = np.clip((predict(family, params, ns) - c1) / denom, 0.0, 1.5)
    rows = []
    for lam in lambdas:
        util = lam * ctilde - (1.0 - lam) * (ns / float(budget))
        idx = int(np.nanargmax(util))
        rows.append(
            {
                "lambda": float(lam),
                "n_star": int(ns[idx]),
                "u_star": float(util[idx]),
                "reference_n": float(n95),
                "ratio": float(ns[idx] / n95),
                "early_stop_flag": bool(ns[idx] < n95),
            }
        )
    return pd.DataFrame(rows)


def ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmedian(arr)), float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def budget_defs(n_obs_max: int, n95: float) -> list[tuple[str, float, bool]]:
    out = [("observed_max", float(n_obs_max), False)]
    if np.isfinite(n95) and n95 > 1:
        out.append(("N95", float(max(2.0, n95)), False))
    out.append(("fixed_cap_50", float(FIXED_CAP), FIXED_CAP > n_obs_max))
    return out


def load_inputs(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    curve_summary = pd.read_csv(spec.curve_dir / "curve_summary.csv")
    curve_bootstrap = pd.read_csv(spec.curve_dir / "curve_bootstrap.csv")
    metadata = json.loads((spec.curve_dir / "curve_metadata.json").read_text(encoding="utf-8"))
    return curve_summary, curve_bootstrap, metadata


def raw_availability(spec: DatasetSpec) -> tuple[bool, str]:
    d = DATA_ROOT / "data" / "processed" / spec.dataset
    labels = d / "labels_long.csv"
    gold = d / "gold.csv"
    if not labels.exists():
        return False, "labels_long_missing"
    if spec.requires_gold and not gold.exists():
        return False, "gold_missing"
    if spec.dataset == "ChaosNLI" and spec.mode != "reference_distribution":
        return False, "chaosnli_gold_accuracy_forbidden"
    return True, "raw_item_level_processed_available"


def process_spec(spec: DatasetSpec) -> dict[str, list[dict]]:
    curve_summary, curve_bootstrap, meta = load_inputs(spec)
    ns = curve_summary["N"].to_numpy(dtype=float)
    y = curve_summary["C_mean"].to_numpy(dtype=float)
    n_obs_max = int(np.nanmax(ns))
    raw_ok, raw_note = raw_availability(spec)
    input_mode = "RAW_ITEM_LEVEL_HIERARCHICAL_BOOTSTRAP" if raw_ok else "SUMMARY_ONLY_NOT_FULL_RAW_BOOTSTRAP"

    best, fits = best_fit(ns, y)
    fit_rows = []
    best_aic = best["aic"]
    for f in fits:
        fit_rows.append(
            {
                "dataset": spec.dataset,
                "mode": spec.mode,
                "fit_family": f["family"],
                "aic": f["aic"],
                "delta_aic": f["aic"] - best_aic if np.isfinite(f["aic"]) and np.isfinite(best_aic) else float("nan"),
                "residual_rmse": f["residual_rmse"],
                "residual_pattern_flag": f["residual_pattern_flag"],
                "admissible_flag": f["admissible"],
                "notes": f["notes"],
            }
        )

    if not best["success"] or best["params"] is None:
        raise RuntimeError(f"No usable fit for {spec.dataset} {spec.mode}")
    family = best["family"]
    params = best["params"]
    n90, n90_status = saturation_n(family, params, 0.90)
    n95, n95_status = saturation_n(family, params, 0.95)
    n99, n99_status = saturation_n(family, params, 0.99)
    n99_flag = "observed" if np.isfinite(n99) and n99 <= n_obs_max else ("extrapolated" if np.isfinite(n99) else "not_identified")

    boot_saturation = []
    boot_util: dict[tuple[str, float], list[float]] = {}
    boot_nstar: dict[tuple[str, float], list[float]] = {}
    boot_fail = 0
    for b, grp in curve_bootstrap.groupby("bootstrap"):
        bgrp = grp.dropna(subset=["C"])
        try:
            b_ns = bgrp["N"].to_numpy(dtype=float)
            b_y = bgrp["C"].to_numpy(dtype=float)
            b_best, _ = best_fit(b_ns, b_y)
            if not b_best["success"] or b_best["params"] is None:
                boot_fail += 1
                continue
            b_family = b_best["family"]
            b_params = b_best["params"]
            b_n90, _ = saturation_n(b_family, b_params, 0.90)
            b_n95, _ = saturation_n(b_family, b_params, 0.95)
            b_n99, _ = saturation_n(b_family, b_params, 0.99)
            boot_saturation.append((b_n90, b_n95, b_n99))
            for budget_type, budget_value, _ in budget_defs(n_obs_max, b_n95):
                urep = utility_curve(b_family, b_params, b_n95, budget_value, np.asarray(REP_LAMBDAS), n_candidate_max=int(math.ceil(budget_value)))
                for _, ur in urep.iterrows():
                    key = (budget_type, float(ur["lambda"]))
                    boot_util.setdefault(key, []).append(float(ur["ratio"]))
                    boot_nstar.setdefault(key, []).append(float(ur["n_star"]))
        except Exception:
            boot_fail += 1

    n90_med, n90_lo, n90_hi = ci([x[0] for x in boot_saturation])
    n95_med, n95_lo, n95_hi = ci([x[1] for x in boot_saturation])
    n99_med, n99_lo, n99_hi = ci([x[2] for x in boot_saturation])
    n_boot_success = len(boot_saturation)

    saturation_row = {
        "dataset": spec.dataset,
        "mode": spec.mode,
        "role": spec.role,
        "fit_family": family,
        "aic": best["aic"],
        "n_obs_max": n_obs_max,
        "n90": n90,
        "n90_ci_low": n90_lo,
        "n90_ci_high": n90_hi,
        "n95": n95,
        "n95_ci_low": n95_lo,
        "n95_ci_high": n95_hi,
        "n99": n99,
        "n99_ci_low": n99_lo,
        "n99_ci_high": n99_hi,
        "n99_flag": n99_flag,
        "fit_warning": ";".join([n90_status, n95_status, n99_status]),
        "input_mode": input_mode,
    }

    util_rows = []
    budget_rows = []
    for budget_type, budget_value, extrap in budget_defs(n_obs_max, n95):
        u = utility_curve(family, params, n95, budget_value, LAMBDA_GRID, n_candidate_max=int(math.ceil(budget_value)))
        if u.empty:
            continue
        for _, row in u.iterrows():
            if any(abs(float(row["lambda"]) - x) < 0.003 for x in REP_LAMBDAS) or budget_type == "observed_max":
                util_rows.append(
                    {
                        "dataset": spec.dataset,
                        "mode": spec.mode,
                        "role": spec.role,
                        "lambda": row["lambda"],
                        "n_budget_type": budget_type,
                        "n_budget": budget_value,
                        "n_star": row["n_star"],
                        "n_star_ci_low": float("nan"),
                        "n_star_ci_high": float("nan"),
                        "reference_type": "N95",
                        "reference_n": n95,
                        "ratio": row["ratio"],
                        "ratio_ci_low": float("nan"),
                        "ratio_ci_high": float("nan"),
                        "early_stop_flag": row["early_stop_flag"],
                        "budget_sensitive": False,
                        "budget_extrapolates_beyond_observed": extrap,
                    }
                )
        for lam in REP_LAMBDAS:
            nearest = u.iloc[(u["lambda"] - lam).abs().argsort()[:1]].iloc[0]
            conclusion = "early_stop_before_N95" if bool(nearest["early_stop_flag"]) else "not_early_stop_before_N95"
            budget_rows.append(
                {
                    "dataset": spec.dataset,
                    "mode": spec.mode,
                    "lambda": lam,
                    "n_budget_type": budget_type,
                    "n_star": int(nearest["n_star"]),
                    "ratio": float(nearest["ratio"]),
                    "conclusion": conclusion,
                    "conclusion_changed_from_primary": False,
                    "budget_extrapolates_beyond_observed": extrap,
                }
            )

    primary_conc = {
        (r["dataset"], r["mode"], r["lambda"]): r["conclusion"]
        for r in budget_rows
        if r["n_budget_type"] == "observed_max"
    }
    for r in budget_rows:
        key = (r["dataset"], r["mode"], r["lambda"])
        r["conclusion_changed_from_primary"] = r["conclusion"] != primary_conc.get(key, r["conclusion"])

    budget_sensitive = any(r["conclusion_changed_from_primary"] for r in budget_rows)
    for r in util_rows:
        r["budget_sensitive"] = budget_sensitive

    ratio_ci_rows = []
    for budget_type, budget_value, _ in budget_defs(n_obs_max, n95):
        for lam in REP_LAMBDAS:
            key = (budget_type, float(lam))
            ratios = boot_util.get(key, [])
            nstars = boot_nstar.get(key, [])
            med, lo, hi = ci(ratios)
            nmed, nlo, nhi = ci(nstars)
            ratio_ci_rows.append(
                {
                    "dataset": spec.dataset,
                    "mode": spec.mode,
                    "role": spec.role,
                    "lambda": lam,
                    "n_budget_type": budget_type,
                    "ratio_type": "rho_95",
                    "median": med,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_star_median": nmed,
                    "n_star_ci_low": nlo,
                    "n_star_ci_high": nhi,
                    "n_bootstrap_success": n_boot_success,
                    "n_bootstrap_fail": boot_fail,
                }
            )

    # Fill CI columns in representative utility rows from bootstrap ratio summaries.
    ratio_lookup = {(r["n_budget_type"], r["lambda"]): r for r in ratio_ci_rows}
    for r in util_rows:
        rep = min(REP_LAMBDAS, key=lambda x: abs(float(r["lambda"]) - x))
        if abs(float(r["lambda"]) - rep) < 0.003:
            rr = ratio_lookup.get((r["n_budget_type"], rep))
            if rr:
                r["ratio_ci_low"] = rr["ci_low"]
                r["ratio_ci_high"] = rr["ci_high"]
                r["n_star_ci_low"] = rr["n_star_ci_low"]
                r["n_star_ci_high"] = rr["n_star_ci_high"]

    # Monotonicity under primary observed budget.
    primary_u = utility_curve(family, params, n95, float(n_obs_max), LAMBDA_GRID, n_candidate_max=n_obs_max)
    decreases = int((np.diff(primary_u["n_star"].to_numpy(dtype=float)) < 0).sum()) if not primary_u.empty else 999
    monotone_ok = decreases == 0
    central = primary_u[(primary_u["lambda"] >= 0.25) & (primary_u["lambda"] <= 0.75)] if not primary_u.empty else pd.DataFrame()
    central_early = bool((central["n_star"] < n95).mean() >= 0.8) if not central.empty else False

    status_row = {
        "dataset": spec.dataset,
        "mode": spec.mode,
        "role": spec.role,
        "manuscript_position": spec.manuscript_position,
        "status": "FULL_RAW_HIERARCHICAL_BOOTSTRAP_USABLE" if raw_ok else "SUMMARY_ONLY_NOT_FULL_RAW_BOOTSTRAP",
        "raw_status_note": raw_note,
        "curve_bootstrap_exists": (spec.curve_dir / "curve_bootstrap.csv").exists(),
        "n95_identifiable": bool(np.isfinite(n95)),
        "n95": n95,
        "n95_ci_low": n95_lo,
        "n95_ci_high": n95_hi,
        "n_star_monotone_observed_budget": monotone_ok,
        "lambda_monotonicity_decreases": decreases,
        "central_lambda_early_stop": central_early,
        "budget_sensitive": budget_sensitive,
        "input_mode": input_mode,
    }

    fit_grid = np.arange(1, max(n_obs_max, int(math.ceil(n95 if np.isfinite(n95) else n_obs_max))) + 1)
    pd.DataFrame(
        {
            "N": fit_grid,
            "C_fit": predict(family, params, fit_grid),
            "dataset": spec.dataset,
            "mode": spec.mode,
            "fit_family": family,
        }
    ).to_csv(OUTPUT_ROOT / "figure_data" / f"{spec.dataset}_{spec.mode}_fit_curve.csv", index=False)

    return {
        "saturation": [saturation_row],
        "fits": fit_rows,
        "utility": util_rows,
        "ratio_ci": ratio_ci_rows,
        "budget": budget_rows,
        "status": [status_row],
    }


def write_outputs(all_parts: list[dict[str, list[dict]]]) -> None:
    names = ["saturation", "fits", "utility", "ratio_ci", "budget", "status"]
    combined = {name: [] for name in names}
    for part in all_parts:
        for name in names:
            combined[name].extend(part[name])

    for legacy, role, status in LEGACY_COMPONENTS:
        combined["status"].append(
            {
                "dataset": legacy,
                "mode": "not_run",
                "role": role,
                "manuscript_position": "legacy_or_excluded",
                "status": status,
                "raw_status_note": (
                    "handled by scripts/legacy_reanalysis in this package"
                    if str(status).startswith("LEGACY_REANALYSIS")
                    else "not available under DATA_ROOT and not fabricated"
                ),
                "curve_bootstrap_exists": False,
                "n95_identifiable": False,
                "n95": float("nan"),
                "n95_ci_low": float("nan"),
                "n95_ci_high": float("nan"),
                "n_star_monotone_observed_budget": False,
                "lambda_monotonicity_decreases": float("nan"),
                "central_lambda_early_stop": False,
                "budget_sensitive": False,
                "input_mode": status,
            }
        )

    pd.DataFrame(combined["saturation"]).to_csv(OUTPUT_ROOT / "final_saturation_summary.csv", index=False)
    pd.DataFrame(combined["utility"]).to_csv(OUTPUT_ROOT / "final_utility_summary.csv", index=False)
    pd.DataFrame(combined["ratio_ci"]).to_csv(OUTPUT_ROOT / "bootstrap_ratio_ci.csv", index=False)
    pd.DataFrame(combined["fits"]).to_csv(OUTPUT_ROOT / "model_fit_comparison.csv", index=False)
    pd.DataFrame(combined["budget"]).to_csv(OUTPUT_ROOT / "budget_sensitivity.csv", index=False)
    pd.DataFrame(combined["status"]).to_csv(OUTPUT_ROOT / "dataset_status_report.csv", index=False)

    # Manuscript-table convenience copies.
    for filename in [
        "final_saturation_summary.csv",
        "final_utility_summary.csv",
        "bootstrap_ratio_ci.csv",
        "model_fit_comparison.csv",
        "budget_sensitivity.csv",
        "dataset_status_report.csv",
    ]:
        src = OUTPUT_ROOT / filename
        dst = OUTPUT_ROOT / "manuscript_tables" / filename
        dst.write_bytes(src.read_bytes())


def decide(status_df: pd.DataFrame, budget_df: pd.DataFrame, ratio_df: pd.DataFrame) -> tuple[str, list[str]]:
    reasons = []
    primary = status_df[
        (status_df["dataset"].isin(["CIFAR-10H", "ChaosNLI"]))
        & (status_df["manuscript_position"] == "primary")
    ]
    primary_n95 = bool(primary["n95_identifiable"].astype(bool).all()) and len(primary) == 2
    primary_monotone = bool(primary["n_star_monotone_observed_budget"].astype(bool).all()) and len(primary) == 2
    primary_early = bool(primary["central_lambda_early_stop"].astype(bool).all()) and len(primary) == 2
    primary_budget = bool(~primary["budget_sensitive"].astype(bool).any()) if len(primary) else False
    primary_ci = bool(
        ratio_df[
            (ratio_df["dataset"].isin(["CIFAR-10H", "ChaosNLI"]))
            & (ratio_df["n_budget_type"] == "observed_max")
            & (ratio_df["lambda"].isin(REP_LAMBDAS))
        ][["ci_low", "ci_high"]]
        .notna()
        .all()
        .all()
    )
    reasons.append(f"primary_n95_identifiable={primary_n95}")
    reasons.append(f"primary_nstar_monotone={primary_monotone}")
    reasons.append(f"central_lambda_early_stop={primary_early}")
    reasons.append(f"primary_budget_not_sensitive={primary_budget}")
    reasons.append(f"bootstrap_ci_interpretable={primary_ci}")

    if primary_n95 and primary_monotone and primary_early and primary_budget and primary_ci:
        return "SCI_REP_FIRST", reasons
    if primary_n95 and primary_ci:
        return "IEEE_ACCESS_FIRST", reasons
    if primary_n95:
        return "CONSERVATIVE_REBUILD", reasons
    return "NO_GO_AUGMENTED", reasons


def write_reports() -> None:
    sat = pd.read_csv(OUTPUT_ROOT / "final_saturation_summary.csv")
    util = pd.read_csv(OUTPUT_ROOT / "final_utility_summary.csv")
    ratio = pd.read_csv(OUTPUT_ROOT / "bootstrap_ratio_ci.csv")
    budget = pd.read_csv(OUTPUT_ROOT / "budget_sensitivity.csv")
    status = pd.read_csv(OUTPUT_ROOT / "dataset_status_report.csv")
    decision, reasons = decide(status, budget, ratio)

    primary_rows = status[status["manuscript_position"] == "primary"]
    supporting_rows = status[status["manuscript_position"].isin(["supporting", "sensitivity"])]
    legacy_rows = status[status["status"].str.contains("MISSING|EXCLUDED", na=False)]
    budget_changed = bool(budget["conclusion_changed_from_primary"].astype(bool).any()) if len(budget) else False

    final_md = [
        "# Paper B Phase 4 Final Gate Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "## Basis",
        "",
        *[f"- {r}" for r in reasons],
        f"- budget_sensitivity_changed_conclusion={budget_changed}",
        "",
        "## Dataset Usability",
        "",
        "Fully usable raw-bootstrap closure datasets:",
        *[
            f"- {row.dataset} / {row.mode}: {row.status}, N95={row.n95:.3f}"
            for row in primary_rows.itertuples()
        ],
        "",
        "Supporting or sensitivity datasets:",
        *[
            f"- {row.dataset} / {row.mode}: {row.status}, role={row.role}, N95={row.n95:.3f}"
            for row in supporting_rows.itertuples()
        ],
        "",
        "Legacy/excluded components:",
        *[f"- {row.dataset}: {row.status}" for row in legacy_rows.itertuples()],
        "",
        "## Interpretation",
        "",
        "N95 is the central saturation reference. N90 and N99 are sensitivity references. "
        "ChaosNLI is evaluated only by reference-distribution recovery. Snapshot Serengeti is supporting evidence only.",
    ]
    (OUTPUT_ROOT / "final_gate_decision.md").write_text("\n".join(final_md) + "\n", encoding="utf-8")

    report = [
        "# Reproducibility Report",
        "",
        f"DATA_ROOT: `{DATA_ROOT}`",
        f"LOCK_ROOT: `{LOCK_ROOT}`",
        f"OUTPUT_ROOT: `{OUTPUT_ROOT}`",
        "",
        "## Inputs Used",
        "",
        "- Processed item-level labels and gold files under DATA_ROOT/data/processed.",
        "- Stage 8 hierarchical-bootstrap curve files under DATA_ROOT/results/pilot.",
        "- Dataset role and claim locks under LOCK_ROOT/REFERENCE_LOCKS.",
        "",
        "## Method",
        "",
        "- Fitted three admissible saturating families: Michaelis, logarithmic saturating, inverse-sqrt.",
        "- Selected admissible model by AIC; if no model passed admissibility, selected best successful fit and flagged it.",
        "- Computed N90, N95, and N99 against fitted asymptotic reference, with N95 central.",
        "- Refit saturation models inside each bootstrap replicate from curve_bootstrap.csv.",
        "- Computed normalized utility U(N) = lambda*C_tilde(N) - (1-lambda)*N/N_budget.",
        "- Ran budgets: observed maximum N, N95, and fixed cap 50.",
        "",
        "## Missing Locked Components",
        "",
        "- Bingol/USL, Snow, and Nitti raw files were not available under DATA_ROOT; marked LEGACY_INPUT_MISSING.",
        "- Galaxy Zoo was not run and remains excluded/optional not run.",
        "",
        "## Output Files",
        "",
        "- final_saturation_summary.csv",
        "- final_utility_summary.csv",
        "- bootstrap_ratio_ci.csv",
        "- model_fit_comparison.csv",
        "- budget_sensitivity.csv",
        "- dataset_status_report.csv",
        "- final_gate_decision.md",
        "- figure_data/",
        "- manuscript_tables/",
    ]
    (OUTPUT_ROOT / "reproducibility_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = sat.merge(status[["dataset", "mode", "manuscript_position", "central_lambda_early_stop", "budget_sensitive"]], on=["dataset", "mode"], how="left")
    summary.to_csv(OUTPUT_ROOT / "manuscript_tables" / "table_saturation_and_gate_summary.csv", index=False)


def main() -> None:
    ensure_dirs()
    # Public reproduction packages carry the admitted inputs and code directly.
    all_parts = []
    for spec in SPECS:
        print(f"[phase4] processing {spec.dataset} / {spec.mode}")
        all_parts.append(process_spec(spec))
    write_outputs(all_parts)
    write_reports()
    print(f"[phase4] wrote outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
