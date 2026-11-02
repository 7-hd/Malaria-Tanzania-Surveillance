"""
src/plot_surveillance.py

Generates publication-quality figures for antimalarial drug resistance 
surveillance in Tanzania using matplotlib.

Outputs:
1. reports/figures/fig1_tanga_longitudinal_trends.png
2. reports/figures/fig2_regional_prevalence_comparison.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import build_tz_resistance_table
from statistical_analysis import wilson_ci

ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


def setup_plot_style():
    """Sets publication-friendly matplotlib parameters."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "figure.dpi": 300,
        }
    )


def plot_tanga_longitudinal(df: pd.DataFrame):
    """Plots longitudinal prevalence with Wilson 95% CIs for Tanga (2010, 2013, 2014)."""
    tanga = df[df["Admin level 1"] == "Tanga"].copy()
    years = [2010, 2013, 2014]

    target_drugs = [
        "Chloroquine",
        "Pyrimethamine",
        "Sulfadoxine",
        "SP (uncomplicated)",
        "SP (IPTp)",
    ]
    colors = {
        "Chloroquine": "#d95f02",
        "Pyrimethamine": "#7570b3",
        "Sulfadoxine": "#1b9e77",
        "SP (uncomplicated)": "#e7298a",
        "SP (IPTp)": "#66a61e",
    }
    markers = {
        "Chloroquine": "o",
        "Pyrimethamine": "s",
        "Sulfadoxine": "^",
        "SP (uncomplicated)": "D",
        "SP (IPTp)": "v",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for drug in target_drugs:
        prevs = []
        yerr_lower = []
        yerr_upper = []

        for yr in years:
            sub = tanga[tanga["Year"] == yr]
            counts = sub[drug].value_counts()
            n_res = counts.get("Resistant", 0)
            n_sens = counts.get("Sensitive", 0)
            n_eval = n_res + n_sens

            if n_eval > 0:
                p = n_res / n_eval
                low, high = wilson_ci(n_res, n_eval)
            else:
                p, low, high = 0.0, 0.0, 0.0

            prevs.append(p * 100)  # Convert to percentage
            yerr_lower.append((p - low) * 100)
            yerr_upper.append((high - p) * 100)

        yerr = [yerr_lower, yerr_upper]

        ax.errorbar(
            years,
            prevs,
            yerr=yerr,
            label=drug,
            color=colors[drug],
            marker=markers[drug],
            linewidth=2,
            markersize=7,
            capsize=4,
            capthick=1.5,
        )

    ax.set_xticks(years)
    ax.set_xticklabels(["2010\n(n=15)", "2013\n(n=152)", "2014\n(n=130)"])
    ax.set_xlabel("Sampling Year (Tanga Region)")
    ax.set_ylabel("Resistance Prevalence (%)")
    ax.set_ylim(-2, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(
        "Figure 1: Antimalarial Resistance Prevalence in Tanga, Tanzania (2010–2014)\n(Error bars represent Wilson 95% Confidence Intervals)",
        pad=15,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=True)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig1_tanga_longitudinal_trends.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to: {out_path}")


def plot_regional_comparison(df: pd.DataFrame):
    """Plots cross-sectional resistance prevalence across Tanzania regions."""
    # Define main region-year groups
    group_order = [
        ("Morogoro", 2010),
        ("Tanga", 2010),
        ("Tanga", 2013),
        ("Lindi", 2013),
        ("Kagera", 2013),
        ("Tanga", 2014),
        ("Kigoma", 2014),
        ("Pwani", 2019),
    ]

    drugs_to_plot = ["Chloroquine", "Sulfadoxine", "SP (IPTp)"]
    labels = [f"{reg}\n({yr})" for reg, yr in group_order]

    x = np.arange(len(group_order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, drug in enumerate(drugs_to_plot):
        prevs = []
        for reg, yr in group_order:
            sub = df[(df["Admin level 1"] == reg) & (df["Year"] == yr)]
            counts = sub[drug].value_counts()
            n_res = counts.get("Resistant", 0)
            n_sens = counts.get("Sensitive", 0)
            n_eval = n_res + n_sens
            prev = (n_res / n_eval * 100) if n_eval > 0 else 0.0
            prevs.append(prev)

        offset = (i - 1) * width
        ax.bar(x + offset, prevs, width, label=drug, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Region & Sampling Year")
    ax.set_ylabel("Resistance Prevalence (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Figure 2: Antimalarial Resistance Prevalence Across Tanzanian Regions (2010–2019)",
        pad=15,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig2_regional_prevalence_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to: {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    df = build_tz_resistance_table()

    plot_tanga_longitudinal(df)
    plot_regional_comparison(df)


if __name__ == "__main__":
    main()
