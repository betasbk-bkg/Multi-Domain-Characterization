from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Figures are authored at the width at which they are printed in the two-column
# manuscript (3.30 in). Authoring at 9.6 in and letting the word processor reduce the
# image to 3.30 in, as versions up to v1.1.0 did, scales all label text by 0.34 and
# puts it below the size a reader can resolve in print.
COLW = 3.30
FIGSIZE = (COLW, 2.05)
DPI = 600
PNG_META = {"Software": "PaperB v1.2.0 full reproduction"}

# Okabe-Ito, chosen so the series remain distinguishable in greyscale and to readers
# with the common forms of colour vision deficiency.
CB = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "legend.framealpha": 0.9,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "legend.handlelength": 1.4,
})


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, metadata=PNG_META)
    plt.close(fig)


def fig1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis("off")
    boxes = [
        (0.06, 0.62, 0.26, 0.23, "Retrograde benchmark\nBingol / USL\nNref = Npeak"),
        (0.37, 0.62, 0.26, 0.23, "Primary closure\nCIFAR-10H, ChaosNLI\nNref = N95"),
        (0.68, 0.62, 0.26, 0.23, "Supporting / boundary\nSnapshot, Snow, Nitti\nconstraints reported"),
        (0.24, 0.22, 0.52, 0.23, "Utility framework\nU(N)=lambda*Ctilde(N)-(1-lambda)N/Nbudget\nOutput: budget-aware N*(lambda)"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.8, color="black"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=5.2)
    arrow = dict(arrowstyle="->", lw=1.4, color="black")
    ax.annotate("", xy=(0.40, 0.45), xytext=(0.19, 0.62), arrowprops=arrow)
    ax.annotate("", xy=(0.50, 0.45), xytext=(0.50, 0.62), arrowprops=arrow)
    ax.annotate("", xy=(0.58, 0.45), xytext=(0.81, 0.62), arrowprops=arrow)
    save(fig, out / "fig1_framework_architecture.png")


def fig2(final: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    curves = [
        ("CIFAR-10H_gold_accuracy_fit_curve.csv", "CIFAR-10H gold accuracy"),
        ("ChaosNLI_reference_distribution_fit_curve.csv", "ChaosNLI distribution recovery"),
        ("Snapshot_Serengeti_gold_accuracy_fit_curve.csv", "Snapshot Serengeti gold accuracy"),
    ]
    # Snapshot Serengeti is fitted on a grid bounded at N_support = 21 while its N95 is
    # 30, so the segment past 21 is an extrapolation of the fit and is drawn dotted.
    grid_bound = {"Snapshot_Serengeti_gold_accuracy_fit_curve.csv": 21}
    for (name, label), colour in zip(curves, CB):
        df = pd.read_csv(final / "figure_data" / name)
        bound = grid_bound.get(name)
        if bound is None:
            ax.plot(df["N"], df["C_fit"], lw=1.3, color=colour, label=label)
        else:
            inside = df[df["N"] <= bound]
            outside = df[df["N"] >= bound]
            ax.plot(inside["N"], inside["C_fit"], lw=1.3, color=colour, label=label)
            ax.plot(outside["N"], outside["C_fit"], lw=1.3, color=colour, ls=":",
                    label="  (extrapolated past N_support)")
    ax.set_xlabel("Number of judgments $N$")
    ax.set_ylabel("Fitted performance $C(N)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, out / "fig2_saturation_curves.png")


def fig3(final: Path, out: Path) -> None:
    util = pd.read_csv(final / "final_utility_summary.csv")
    sat = pd.read_csv(final / "final_saturation_summary.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = {
        ("CIFAR-10H", "gold_accuracy"): "CIFAR-10H (gold accuracy)",
        ("ChaosNLI", "reference_distribution"): "ChaosNLI (reference distribution)",
    }
    for (dataset, mode), label in labels.items():
        df = util[
            (util["dataset"] == dataset)
            & (util["mode"] == mode)
            & (util["n_budget_type"] == "observed_max")
        ].sort_values("lambda")
        ax.plot(df["lambda"], df["n_star"], lw=2.2, label=label)
        row = sat[(sat["dataset"] == dataset) & (sat["mode"] == mode)].iloc[0]
        ax.axhline(row["n95"], ls="--", lw=1.2, alpha=0.65)
    ax.set_xlabel("Performance weight lambda")
    ax.set_ylabel("Utility-optimal N*")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    save(fig, out / "fig3_nstar_lambda_primary.png")


def fig4(legacy: Path, out: Path) -> None:
    data = json.loads((legacy / "paperB_bingol_results.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for fill, rec in sorted(data["PB_B3_utility_Nstar"].items(), key=lambda kv: float(kv[0])):
        xs, ys = [], []
        for lam, vals in sorted(rec["lambda_results"].items(), key=lambda kv: float(kv[0])):
            xs.append(float(lam))
            ys.append(float(vals["ratio_to_Npeak"]))
        ax.plot(xs, ys, marker="o", lw=2.2, label=f"USL {fill}")
    ax.axhline(1.0, ls="--", lw=1.2)
    ax.set_xlabel("Performance weight lambda")
    ax.set_ylabel("N*/Npeak")
    ax.set_ylim(0, 1.12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2)
    save(fig, out / "fig4_bingol_retrograde_ratios.png")


def fig5(final: Path, out: Path) -> None:
    budget = pd.read_csv(final / "budget_sensitivity.csv")
    sat = pd.read_csv(final / "final_saturation_summary.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    primary = [
        ("CIFAR-10H", "gold_accuracy", "CIFAR-10H"),
        ("ChaosNLI", "reference_distribution", "ChaosNLI"),
    ]
    for dataset, mode, short in primary:
        n95 = float(sat[(sat["dataset"] == dataset) & (sat["mode"] == mode)].iloc[0]["n95"])
        for btype in ["observed_max", "N95", "fixed_cap_50"]:
            df = budget[
                (budget["dataset"] == dataset)
                & (budget["mode"] == mode)
                & (budget["n_budget_type"] == btype)
            ].sort_values("lambda")
            if df.empty:
                continue
            ax.plot(df["lambda"], df["n_star"] / n95, marker="o", lw=2.0, label=f"{short} {btype}")
    ax.axhline(1.0, ls="--", lw=1.2)
    ax.set_xlabel("Performance weight lambda")
    ax.set_ylabel("Stopping ratio N*/N95")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=2)
    save(fig, out / "fig5_budget_sensitivity.png")


def fig6(final: Path, out: Path) -> None:
    """Cost-price collapse.

    Every budget definition enters the stopping decision only through the single
    price eta = (1 - lambda) / (lambda * N_budget), so the integer optima computed
    under different budgets fall on one curve when plotted against eta. The curve is
    the closed form of the manuscript appendix evaluated at the fitted K, and the
    dotted vertical line is the budget-free threshold eta_c = 1 / (380 (K + 1)).
    """
    sat = pd.read_csv(final / "final_saturation_summary.csv")
    const = pd.read_csv(final / "replicate_constants.csv")

    def S(n, k):
        return (n - 1.0) / (0.95 * (k + n))

    def nstar(k, lam, budget, grid=1000):
        n = np.arange(1, grid + 1, dtype=float)
        return int(n[int(np.nanargmax(lam * S(n, k) - (1.0 - lam) * n / budget))])

    panels = [
        ("CIFAR-10H", "gold_accuracy", "CIFAR-10H"),
        ("ChaosNLI", "reference_distribution", "ChaosNLI"),
    ]
    lams = np.linspace(0.005, 0.995, 200)
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(2, 1, figsize=(COLW, 3.6))
    for ax, (dataset, mode, short) in zip(axes, panels):
        srow = sat[(sat["dataset"] == dataset) & (sat["mode"] == mode)].iloc[0]
        crow = const[(const["dataset"] == dataset) & (const["mode"] == mode)].iloc[0]
        k = float(crow["K_median"])
        n95 = int(srow["n95"])
        support = int(srow["n_obs_max"])
        eta_c = float(crow["eta_c"])
        budgets = [
            (f"N_support = {support}", support),
            (f"N95 = {n95}", n95),
            ("fixed cap = 50", 50),
        ]
        eta = np.logspace(-3.6, 0.05, 400)
        ax.plot(eta, (np.sqrt((k + 1.0) / (0.95 * eta)) - k) / n95, color="0.25", lw=1.1,
                label=r"closed form $N^{*}(\eta)/N_{95}$")
        for (label, budget), marker, colour in zip(budgets, markers, CB):
            ax.plot((1.0 - lams) / (lams * budget),
                    [nstar(k, lam, budget) / n95 for lam in lams],
                    marker, ms=1.8, mew=0, color=colour, alpha=0.85, label=label)
        ax.axhline(1.0, ls="--", lw=0.7, color="0.4")
        ax.axvline(eta_c, ls=":", lw=0.9, color="#D55E00")
        ax.text(eta_c * 1.15, 0.06, rf"$\eta_c$={eta_c:.5f}", fontsize=5.2, color="#D55E00")
        ax.set_xscale("log")
        ax.set_xlim(2e-4, 1.2)
        ax.set_ylim(0.0, 1.35)
        ax.set_title(f"{short}  ($K$ = {k:.3f})")
        ax.set_ylabel(r"$\rho_{95}=N^{*}/N_{95}$")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right")
    axes[1].set_xlabel(r"cost price  $\eta=(1-\lambda)/(\lambda N_{budget})$")
    save(fig, out / "fig6_eta_collapse.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--final", type=Path, required=True)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    fig1(args.out)
    fig2(args.final, args.out)
    fig3(args.final, args.out)
    fig4(args.legacy, args.out)
    fig5(args.final, args.out)
    fig6(args.final, args.out)


if __name__ == "__main__":
    main()
