"""
app/app.py

Streamlit interactive dashboard for Malaria Drug-Resistance Genomic Surveillance in Tanzania.
Title: Genomic Surveillance of Antimalarial Drug-Resistance Markers in Tanzania: 
       Spatiotemporal Trends and Population Structure in the MalariaGEN Pf8 Dataset
"""

import sys
from pathlib import Path

# Add project root and src to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from load_data import build_tz_resistance_table
from load_markers import load_markers
from pca_population_structure import encode_marker_genotypes, run_pca
from statistical_analysis import DRUGS, cochran_armitage_test, wilson_ci

# Page configuration
st.set_page_config(
    page_title="Tanzania Malaria Genomic Surveillance (Pf8)",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_app_data():
    """Loads and merges Tanzania resistance status and marker genotypes."""
    tz_df = build_tz_resistance_table()
    markers_df = load_markers()
    merged = pd.merge(tz_df, markers_df, on="Sample", how="inner")
    return merged


@st.cache_data
def get_pca_data(merged_df):
    """Computes PCA components and variance explained."""
    encoded_df, variant_cols = encode_marker_genotypes(merged_df)
    coords, pca = run_pca(encoded_df, n_components=5)

    pca_df = pd.DataFrame(
        {
            "Sample": merged_df["Sample"],
            "Admin level 1": merged_df["Admin level 1"],
            "Year": merged_df["Year"],
            "Study": merged_df["Study"],
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "PC3": coords[:, 2],
            "PC4": coords[:, 3],
            "PC5": coords[:, 4],
        }
    )

    var_df = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(5)],
            "Explained_Variance_%": pca.explained_variance_ratio_ * 100,
            "Cumulative_Variance_%": np.cumsum(pca.explained_variance_ratio_) * 100,
        }
    )

    return pca_df, var_df, variant_cols


def main():
    st.title(
        "Genomic Surveillance of Antimalarial Drug Resistance in Tanzania (Pf8)"
    )
    st.caption(
        "Analysis of spatiotemporal resistance trends and parasite population structure using the MalariaGEN Pf8 dataset (n=1,025 QC-pass samples)."
    )

    merged_df = load_app_data()
    pca_df, var_df, variant_cols = get_pca_data(merged_df)

    # Sidebar Filters
    st.sidebar.header("Filter & Settings")
    selected_drugs = st.sidebar.multiselect(
        "Select Antimalarial Drugs",
        options=DRUGS,
        default=["Chloroquine", "Pyrimethamine", "Sulfadoxine", "SP (IPTp)"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Dataset Info:**\n"
        "- Total QC-pass Samples: 1,025\n"
        "- Regions Covered: 6\n"
        "- Time Range: 2010–2019\n"
        "- Source: MalariaGEN Pf8 (2025)"
    )

    # Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Tanga Longitudinal Trends",
            "Regional Comparison",
            "PCA Population Structure",
            "Data Inspector",
        ]
    )

    # TAB 1: OVERVIEW
    with tab1:
        st.subheader("Project Overview & Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("QC-Pass Samples", "1,025")
        col2.metric("Regions", "6")
        col3.metric("Sampling Years", "2010–2019")
        col4.metric("Evaluated Drugs", "10")

        st.markdown("### Key Findings Summary")
        st.info(
            "1. **Chloroquine Resistance Drop (Tanga 2010–2014)**: Statistically significant collapse in *crt* K76T resistance prevalence "
            "from 100% in 2010 down to 10.8% in 2014 (Cochran-Armitage $Z = -7.689, p = 1.49 \\times 10^{-14}$), demonstrating fitness cost post-withdrawal.\n"
            "2. **Pyrimethamine Fixation**: Near 100% fixation of *dhfr* 108N resistance ($p = 2.65 \\times 10^{-4}$), driven by ongoing SP use for IPTp in pregnant women.\n"
            "3. **Population Structure**: Unsupervised PCA on 21 variant marker loci shows 56.6% of total genetic variance explained by PC1–PC2, revealing temporal shift rather than strong geographic separation."
        )

        st.markdown("### Provenance of Resistance Heuristics")
        heuristics_df = pd.DataFrame(
            [
                {
                    "Drug": "Chloroquine",
                    "Locus": "crt_76[K]",
                    "Provenance": "DOCUMENTED",
                    "Match Rate": "100.0%",
                    "Citation": "Djimde et al., Lancet (2001)",
                },
                {
                    "Drug": "Pyrimethamine",
                    "Locus": "dhfr_108[S]",
                    "Provenance": "DOCUMENTED",
                    "Match Rate": "100.0%",
                    "Citation": "Peterson et al., PNAS (1988)",
                },
                {
                    "Drug": "Sulfadoxine",
                    "Locus": "dhps_437[G]",
                    "Provenance": "EMPIRICAL",
                    "Match Rate": "100.0%",
                    "Citation": "Pf8 Classification & dhps A437G",
                },
            ]
        )
        st.dataframe(heuristics_df, hide_index=True)

    # TAB 2: TANGA LONGITUDINAL TRENDS
    with tab2:
        st.subheader("Longitudinal Surveillance in Tanga Region (2010, 2013, 2014)")

        tanga = merged_df[merged_df["Admin level 1"] == "Tanga"]
        years = [2010, 2013, 2014]

        # Prepare longitudinal data
        trend_records = []
        chart_data = []

        for drug in selected_drugs if selected_drugs else DRUGS:
            r_list, n_eval_list = [], []

            for yr in years:
                sub = tanga[tanga["Year"] == yr]
                counts = sub[drug].value_counts()
                n_res = counts.get("Resistant", 0)
                n_sens = counts.get("Sensitive", 0)
                n_eval = n_res + n_sens

                prev = (n_res / n_eval) if n_eval > 0 else 0.0
                low, high = wilson_ci(n_res, n_eval)

                r_list.append(n_res)
                n_eval_list.append(n_eval)

                chart_data.append(
                    {
                        "Year": yr,
                        "Drug": drug,
                        "Prevalence_%": prev * 100,
                        "CI_Low_%": low * 100,
                        "CI_High_%": high * 100,
                        "CI_Err_Upper": (high - prev) * 100,
                        "CI_Err_Lower": (prev - low) * 100,
                        "Evaluated_N": n_eval,
                        "Resistant_N": n_res,
                    }
                )

            z_stat, p_val = cochran_armitage_test(r_list, n_eval_list, years)
            trend_records.append(
                {
                    "Drug": drug,
                    "2010 Prev (%)": f"{(r_list[0]/n_eval_list[0]*100):.1f}% ({r_list[0]}/{n_eval_list[0]})",
                    "2013 Prev (%)": f"{(r_list[1]/n_eval_list[1]*100):.1f}% ({r_list[1]}/{n_eval_list[1]})",
                    "2014 Prev (%)": f"{(r_list[2]/n_eval_list[2]*100):.1f}% ({r_list[2]}/{n_eval_list[2]})",
                    "CA Z-Stat": f"{z_stat:.3f}",
                    "P-Value": f"{p_val:.4e}",
                    "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No",
                }
            )

        cd_df = pd.DataFrame(chart_data)

        if not cd_df.empty:
            fig = px.line(
                cd_df,
                x="Year",
                y="Prevalence_%",
                color="Drug",
                markers=True,
                error_y="CI_Err_Upper",
                error_y_minus="CI_Err_Lower",
                title="Antimalarial Resistance Prevalence in Tanga with Wilson 95% CIs",
                labels={"Prevalence_%": "Prevalence (%)", "Year": "Sampling Year"},
            )
            fig.update_xaxes(type="category")
            fig.update_layout(height=450)
            st.plotly_chart(fig)

        st.markdown("### Cochran-Armitage Monotonic Trend Test Results")
        st.dataframe(pd.DataFrame(trend_records), hide_index=True)

    # TAB 3: REGIONAL COMPARISON
    with tab3:
        st.subheader("Regional Resistance Prevalence Across Tanzania")

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

        reg_data = []
        for reg, yr in group_order:
            sub = merged_df[(merged_df["Admin level 1"] == reg) & (merged_df["Year"] == yr)]
            for drug in selected_drugs if selected_drugs else DRUGS:
                counts = sub[drug].value_counts()
                n_res = counts.get("Resistant", 0)
                n_sens = counts.get("Sensitive", 0)
                n_eval = n_res + n_sens
                prev = (n_res / n_eval * 100) if n_eval > 0 else 0.0
                reg_data.append(
                    {
                        "Region_Year": f"{reg} ({yr})",
                        "Drug": drug,
                        "Prevalence_%": prev,
                        "Evaluated_N": n_eval,
                    }
                )

        reg_df = pd.DataFrame(reg_data)
        if not reg_df.empty:
            fig_reg = px.bar(
                reg_df,
                x="Region_Year",
                y="Prevalence_%",
                color="Drug",
                barmode="group",
                title="Cross-Sectional Resistance Prevalence by Region & Year",
                labels={"Prevalence_%": "Prevalence (%)", "Region_Year": "Region (Year)"},
            )
            fig_reg.update_layout(height=450)
            st.plotly_chart(fig_reg)

    # TAB 4: PCA POPULATION STRUCTURE
    with tab4:
        st.subheader("Unsupervised Population Structure (PCA on 21 Marker Loci)")

        color_by = st.radio(
            "Color PCA Scatter Plot By:",
            options=["Admin level 1", "Year", "Study"],
            horizontal=True,
        )

        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color=color_by,
            hover_data=["Sample", "Admin level 1", "Year"],
            title=f"PCA of Resistance Marker Genotypes (PC1: {var_df.loc[0, 'Explained_Variance_%']:.1f}% vs PC2: {var_df.loc[1, 'Explained_Variance_%']:.1f}%)",
            labels={
                "PC1": f"PC1 ({var_df.loc[0, 'Explained_Variance_%']:.1f}%)",
                "PC2": f"PC2 ({var_df.loc[1, 'Explained_Variance_%']:.1f}%)",
            },
        )
        fig_pca.update_layout(height=500)
        st.plotly_chart(fig_pca)

        st.markdown("### Variance Explained per Principal Component")
        st.dataframe(var_df, hide_index=True)

    # TAB 5: DATA INSPECTOR
    with tab5:
        st.subheader("Sample Genotype & Resistance Data Inspector")

        region_filter = st.multiselect(
            "Filter Region:",
            options=merged_df["Admin level 1"].unique(),
            default=merged_df["Admin level 1"].unique(),
        )

        filtered_df = merged_df[merged_df["Admin level 1"].isin(region_filter)]
        st.dataframe(filtered_df)

        st.download_button(
            label="Download Filtered Data (TSV)",
            data=filtered_df.to_csv(sep="\t", index=False),
            file_name="tanzania_malaria_surveillance_filtered.tsv",
            mime="text/tab-separated-values",
        )


if __name__ == "__main__":
    main()
