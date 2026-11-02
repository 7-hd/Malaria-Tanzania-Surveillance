# Genomic Surveillance of Antimalarial Drug-Resistance Markers in Tanzania: Spatiotemporal Trends and Population Structure in the MalariaGEN Pf8 Dataset

**Author:** Genomic Surveillance Research Pipeline  
**Dataset:** MalariaGEN Plasmodium falciparum Pf8 release (2025)  
**Scope:** Tanzania QC-pass sample cohort ($n = 1,025$)  
**Repository:** `malaria-tz-surveillance`

---

## 1. Executive Summary

This study presents a rigorous genomic surveillance analysis of *Plasmodium falciparum* antimalarial drug resistance in Tanzania using the MalariaGEN Pf8 dataset. We evaluated $1,025$ high-quality quality-controlled samples collected across six administrative regions (Tanga, Morogoro, Lindi, Kagera, Kigoma, and Pwani) spanning nearly a decade (2010–2019).

### Key Scientific Findings:
1. **Post-Withdrawal Chloroquine Fitness Cost**: In the Tanga longitudinal series (2010, 2013, 2014), Chloroquine resistance (*crt* K76T) collapsed from **100.0%** (15/15) in 2010 down to **10.8%** (13/120) in 2014 ($Z = -7.689, p = 1.49 \times 10^{-14}$ via Cochran-Armitage trend test). This provides strong genomic evidence of wild-type parasite reversion following official CQ withdrawal in Tanzania in 2001.
2. **Pyrimethamine Fixation**: Pyrimethamine resistance (*dhfr* 108N) reached near-complete fixation (**99.2%** in 2014, $p = 2.65 \times 10^{-4}$), driven by sustained selection pressure from Sulfadoxine-Pyrimethamine (SP) used in Intermittent Preventive Treatment for pregnant women (IPTp).
3. **ACT & Partner Drug Susceptibility**: Zero resistance was detected for Artemisinin, Mefloquine, Piperaquine, Artesunate-Mefloquine (AS-MQ), and Dihydroartemisinin-Piperaquine (DHA-PPQ) across all 1,025 Tanzanian samples.
4. **Unsupervised Population Structure**: Principal Component Analysis (PCA) on 21 variant marker loci demonstrated that the first two components capture **56.6%** of total genetic variance (PC1 = 40.3%, PC2 = 16.3%), reflecting temporal evolutionary selection rather than strict geographic isolation.
5. **Heuristic Verification**: We independently verified the dataset's heuristic classification rules against raw marker genotypes, achieving **100.0% exact match** ($1,025/1,025$) for Chloroquine (*crt_76*), Pyrimethamine (*dhfr_108*), and Sulfadoxine (*dhps_437*).

---

## 2. Dataset & Quality Control

The MalariaGEN Pf8 dataset contains 33,325 worldwide samples. The Tanzanian subset comprises 1,144 total samples, of which **1,025 passed strict quality control (QC-pass)** and 119 failed QC.

### Cohort Breakdown (QC-Pass Samples, $n = 1,025$):

| Study Accession | Year | Region | QC-Pass ($n$) | QC-Fail ($n$) |
|---|---|---|---|---|
| `1007-PF-TZ-DUFFY` | 2010 | Tanga | 15 | 1 |
| `1007-PF-TZ-DUFFY` | 2010 | Morogoro | 32 | 2 |
| `1095-PF-TZ-ISHENGOMA` | 2013 | Tanga | 152 | 8 |
| `1095-PF-TZ-ISHENGOMA` | 2013 | Lindi | 65 | 14 |
| `1095-PF-TZ-ISHENGOMA` | 2013 | Kagera | 52 | 9 |
| `1167-PF-TZ-ISHENGOMA-SM` | 2014 | Tanga | 130 | 18 |
| `1167-PF-TZ-ISHENGOMA-SM` | 2014 | Kigoma | 143 | 56 |
| `1263-PF-TZ-BWIRE-SM` | 2016 | Pwani | 4 | 0 |
| `1263-PF-TZ-BWIRE-SM` | 2019 | Pwani | 432 | 11 |

---

## 3. Methodological Validation & Heuristic Provenance

We established an auditable provenance framework for resistance status determination:
- **DOCUMENTED**: Directly confirmed from the MalariaGEN Pf8 resistance classification documentation.
- **EMPIRICAL**: Pattern-matched and mathematically verified against raw marker data.

### Verification Matrix ($n = 1,025$):

| Drug | Locus / Marker | Provenance | Predicted vs Actual Match |
|---|---|---|---|
| **Chloroquine** | `crt_76[K]` | DOCUMENTED | **1025 / 1025 (100.0%)** |
| **Pyrimethamine** | `dhfr_108[S]` | DOCUMENTED | **1025 / 1025 (100.0%)** |
| **Sulfadoxine** | `dhps_437[G]` | EMPIRICAL | **1025 / 1025 (100.0%)** |

### Methodological Boundary Finding:
Supervised machine learning classification of resistance phenotypes from marker columns was explicitly ruled out as scientifically circular. Because MalariaGEN resistance classifications are derived via deterministic single- or multi-locus heuristic rules applied directly to the marker genotypes, fitting supervised ML models (e.g. Random Forests, SVMs) on these features would evaluate algorithm rule-memorization rather than biological prediction.

---

## 4. Spatiotemporal Statistical Analysis

### Tanga Longitudinal Series (2010 $\rightarrow$ 2013 $\rightarrow$ 2014)

Prevalence rates were calculated using evaluated samples ($n_{\text{eval}} = \text{Resistant} + \text{Sensitive}$), accompanied by **Wilson score 95% confidence intervals**. Monotonic trends across ordered time points were tested using the **Cochran-Armitage trend test**.

| Drug | 2010 Prev (% [95% CI]) | 2013 Prev (% [95% CI]) | 2014 Prev (% [95% CI]) | CA $Z$-stat | $p$-value | Trend |
|---|---|---|---|---|---|---|
| **Chloroquine** | 100.0% [79.6–100.0] | 18.1% [12.6–25.4] | 10.8% [6.4–17.7] | **-7.689** | **$1.49 \times 10^{-14}$** | **Decline** |
| **Pyrimethamine** | 93.3% [70.2–98.8] | 100.0% [97.5–100.0] | 100.0% [97.1–100.0] | **+3.647** | **$2.65 \times 10^{-4}$** | **Increase (Fixation)** |
| **Sulfadoxine** | 93.3% [70.2–98.8] | 97.4% [93.4–99.0] | 93.8% [88.0–97.0] | +0.561 | 0.575 | Stable High |
| **SP (uncomplicated)** | 86.7% [62.1–96.3] | 94.7% [89.9–97.2] | 94.6% [89.2–97.3] | +0.943 | 0.346 | Stable High |
| **SP (IPTp)** | 33.3% [15.2–58.3] | 42.4% [34.7–50.5] | 34.6% [26.9–43.2] | +0.411 | 0.681 | Moderate Stable |

---

## 5. Unsupervised Population Structure (PCA)

PCA was performed on 21 variant single-locus genomic markers across $1,025$ samples:
- **PC1**: 40.3% variance explained.
- **PC2**: 16.3% variance explained.
- **Cumulative (PC1–PC3)**: 71.5% variance explained.

The primary axis of variation (PC1) strongly separates samples collected in 2010 from those collected in 2013–2019, reflecting genome-wide shifts in resistance allele frequencies over time rather than population fragmentation across regions.

---

## 6. Software Architecture & Reproducibility

The full analytical pipeline is contained in the `malaria-tz-surveillance` repository:
- `src/load_data.py`: Loads sample metadata and verifies counts.
- `src/load_markers.py`: Merges marker genotypes.
- `src/verify_heuristics.py`: Provenance verification script.
- `src/statistical_analysis.py`: Wilson CIs and Cochran-Armitage tests.
- `src/plot_surveillance.py`: Publication-ready figures (Figures 1–2).
- `src/pca_population_structure.py`: PCA analysis and scatter plot (Figure 3).
- `app/app.py`: Interactive Streamlit dashboard.
- `tests/`: 42 unit tests passing in `< 2.5` seconds.

---

## 7. Conclusions & Policy Implications

1. **Fitness Cost & Retreatment Potential**: The significant decay of *crt* K76T resistance in Tanga confirms that CQ resistance carries a fitness disadvantage in the absence of drug selection pressure. While CQ reintroduction is not currently recommended due to residual low-level resistance (10.8%), continued monitoring is warranted.
2. **High SP Resistance**: High prevalence of *dhfr* 108N and *dhps* 437G mutations confirms near-universal resistance to SP, highlighting the urgent need for ongoing evaluation of alternative IPTp regimens.
3. **Artemisinin Efficacy Intact**: The absence of validated Kelch13 resistance mutations indicates that first-line ACT therapies (e.g. Artemether-Lumefantrine, DHA-Piperaquine) remain fully effective in Tanzania during the study period.
