from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FIGSIZE = (9.6, 4.8)
DPI = 110
PNG_META = {"Software": "PaperB v1.0.7 full reproduction"}


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
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=13)
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
    for name, label in curves:
        df = pd.read_csv(final / "figure_data" / name)
        ax.plot(df["N"], df["C_fit"], lw=2.2, label=label)
    ax.set_xlabel("Number of judgments N")
    ax.set_ylabel("Fitted performance C(N)")
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
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    save(fig, out / "fig5_budget_sensitivity.png")


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


if __name__ == "__main__":
    main()
