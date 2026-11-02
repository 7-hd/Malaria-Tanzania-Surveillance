"""
src/pca_population_structure.py

Option B: Performs unsupervised Principal Component Analysis (PCA) on raw 
antimalarial resistance marker genotypes across Tanzanian samples (n=1025).

Evaluates whether parasite population structure clusters by geography 
(Region) or time (Year).

Outputs:
1. reports/pca_variance_explained.tsv
2. reports/figures/fig3_pca_population_structure.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from load_markers import load_markers
from load_data import build_tz_resistance_table

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Define single-locus columns to exclude complex non-synonymous string columns
COMPLEX_COLUMNS = ["kelch13_349-726_ns_changes", "crt_72-76[CVMNK]"]


def encode_marker_genotypes(markers_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Encodes genomic marker columns into numerical matrix:
      - 0.0: Wildtype / Ref
      - 1.0: Mutant / Alt
      - 0.5: Heterozygous / Mixed
      - Imputes missing '-' with locus mode.
    Excludes non-marker metadata and complex string columns.
    """
    ignore_cols = ["Sample"] + COMPLEX_COLUMNS
    marker_cols = [c for c in markers_df.columns if c not in ignore_cols]

    encoded = pd.DataFrame(index=markers_df.index)

    for col in marker_cols:
        series = markers_df[col].astype(str)

        # Detect wildtype/mutant alleles from locus header bracket notation if available
        # Header format example: crt_76[K] -> wildtype reference allele is K
        ref_allele = None
        if "[" in col and "]" in col:
            ref_allele = col.split("[")[1].split("]")[0]

        vals = []
        for v in series:
            if v in ["-", "nan", "None", ""]:
                vals.append(np.nan)
            elif "," in v:
                vals.append(0.5)  # Heterozygous call
            elif ref_allele is not None:
                if v == ref_allele:
                    vals.append(0.0)  # Wildtype
                else:
                    vals.append(1.0)  # Mutant
            else:
                vals.append(0.0)

        encoded_col = pd.Series(vals, index=markers_df.index)
        # Impute missing with mode (or 0.0 fallback)
        mode_val = encoded_col.mode(dropna=True)
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0.0
        encoded[col] = encoded_col.fillna(fill_val)

    # Drop invariant columns (variance == 0)
    variances = encoded.var()
    variant_cols = list(variances[variances > 0].index)
    encoded_variant = encoded[variant_cols]

    return encoded_variant, variant_cols


def run_pca(encoded_df: pd.DataFrame, n_components: int = 5) -> tuple[np.ndarray, PCA]:
    """Fits PCA model on encoded marker matrix."""
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(encoded_df)
    return coords, pca


def plot_pca_results(df: pd.DataFrame, coords: np.ndarray, pca: PCA):
    """Generates Figure 3: PCA scatter plots colored by Region and Year."""
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        # Panel A: Colored by Region
    regions = df["Admin level 1"].unique()
    colors_reg = plt.cm.Set1(np.linspace(0, 1, len(regions)))

    for i, reg in enumerate(regions):
        mask = df["Admin level 1"] == reg
        ax1.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=reg,
            color=colors_reg[i],
            alpha=0.7,
            edgecolors="none",
            s=35,
        )

    ax1.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    ax1.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")
    ax1.set_title("A: Population Structure by Region", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="best", frameon=True)

    # Panel B: Colored by Year
    years = sorted(df["Year"].unique())
    for yr in years:
        mask = df["Year"] == yr
        ax2.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=str(yr),
            alpha=0.7,
            edgecolors="none",
            s=35,
        )

    ax2.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    ax2.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")
    ax2.set_title("B: Population Structure by Sampling Year", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="best", frameon=True)

    plt.suptitle(
        "Figure 3: Principal Component Analysis (PCA) of Resistance Markers in Tanzania (n=1025)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig3_pca_population_structure.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 to: {out_path}")


def main():
    print("=== OPTION B: UNSUPERVISED POPULATION STRUCTURE (PCA) ===")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data & markers
    tz_df = build_tz_resistance_table()
    markers_df = load_markers()

    # Merge markers with tz_df
    merged = pd.merge(tz_df, markers_df, on="Sample", how="inner")
    assert len(merged) == 1025, f"Expected 1025 merged rows, got {len(merged)}"

    # Encode markers
    encoded_df, variant_cols = encode_marker_genotypes(merged)
    print(f"Encoded {len(variant_cols)} variant marker loci (dropped invariant columns).")

    # Run PCA
    coords, pca = run_pca(encoded_df, n_components=5)

    # Save variance report
    var_df = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(5)],
            "Explained_Variance_Ratio": pca.explained_variance_ratio_,
            "Cumulative_Variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    var_path = REPORTS_DIR / "pca_variance_explained.tsv"
    var_df.to_csv(var_path, sep="\t", index=False)
    print(f"Saved PCA variance report to: {var_path}")
    print("\nVariance Explained:")
    print(var_df.to_string(index=False))

    # Plot figure 3
    plot_pca_results(merged, coords, pca)


if __name__ == "__main__":
    main()
