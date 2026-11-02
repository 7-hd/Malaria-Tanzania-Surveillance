"""
src/statistical_analysis.py

Performs statistical surveillance analysis on the Pf8 Tanzania dataset.
Focuses on the Tanga longitudinal series (2010, 2013, 2014) across all 10 antimalarial drugs.

Outputs:
1. Wilson score 95% confidence intervals for resistance prevalence per drug and year.
2. Cochran-Armitage trend tests for monotonic change in resistance (2010 -> 2013 -> 2014).
3. Missingness (Undetermined) tracking as QC metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from load_data import build_tz_resistance_table

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"

DRUGS = [
    "Chloroquine",
    "Pyrimethamine",
    "Sulfadoxine",
    "Mefloquine",
    "Artemisinin",
    "Piperaquine",
    "SP (uncomplicated)",
    "SP (IPTp)",
    "AS-MQ",
    "DHA-PPQ",
]


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Computes Wilson score interval for a binomial proportion.
    k: number of successes (resistant counts)
    n: total evaluated samples (resistant + sensitive)
    returns: (lower_bound, upper_bound)
    """
    if n == 0:
        return (0.0, 0.0)

    p_hat = k / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z_sq = z**2

    denom = 1 + z_sq / n
    center = (p_hat + z_sq / (2 * n)) / denom
    spread = (z / denom) * np.sqrt((p_hat * (1 - p_hat) / n) + (z_sq / (4 * n**2)))

    lower = max(0.0, float(center - spread))
    upper = min(1.0, float(center + spread))

    return (lower, upper)


def cochran_armitage_test(
    r_list: list[int], n_list: list[int], t_list: list[int]
) -> tuple[float, float]:
    """
    Computes Cochran-Armitage trend test for linear trend in proportions over ordered time points.
    r_list: resistant counts at each time point
    n_list: total evaluated samples (R + S) at each time point
    t_list: numeric time points (e.g. [2010, 2013, 2014])
    returns: (z_statistic, p_value_two_sided)
    """
    r = np.array(r_list, dtype=float)
    n = np.array(n_list, dtype=float)
    t = np.array(t_list, dtype=float)

    N_total = np.sum(n)
    R_total = np.sum(r)

    if N_total == 0 or R_total == 0 or R_total == N_total:
        # Zero variance / completely flat proportions
        return (0.0, 1.0)

    p_bar = R_total / N_total
    q_bar = 1.0 - p_bar

    t_bar = np.sum(n * t) / N_total
    t_diff = t - t_bar

    U = np.sum(r * t_diff)
    var_U = p_bar * q_bar * np.sum(n * (t_diff**2))

    if var_U == 0:
        return (0.0, 1.0)

    z = U / np.sqrt(var_U)
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return (float(z), float(p_val))


def analyze_tanga_longitudinal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters for Tanga region and computes yearly prevalence, Wilson CIs,
    and Cochran-Armitage trend tests across 2010, 2013, and 2014.
    """
    tanga = df[df["Admin level 1"] == "Tanga"].copy()
    years = sorted(tanga["Year"].unique())
    assert years == [2010, 2013, 2014], f"Unexpected Tanga years: {years}"

    prevalence_rows = []
    trend_rows = []

    for drug in DRUGS:
        r_list = []
        n_eval_list = []

        for year in years:
            sub = tanga[tanga["Year"] == year]
            counts = sub[drug].value_counts()

            n_res = counts.get("Resistant", 0)
            n_sens = counts.get("Sensitive", 0)
            n_undet = counts.get("Undetermined", 0)
            n_total = len(sub)
            n_eval = n_res + n_sens

            prev = (n_res / n_eval) if n_eval > 0 else 0.0
            ci_low, ci_high = wilson_ci(n_res, n_eval)
            undet_rate = (n_undet / n_total) if n_total > 0 else 0.0

            prevalence_rows.append(
                {
                    "Drug": drug,
                    "Year": year,
                    "Total_N": n_total,
                    "Evaluated_N": n_eval,
                    "Resistant_N": n_res,
                    "Sensitive_N": n_sens,
                    "Undetermined_N": n_undet,
                    "Prevalence": prev,
                    "Wilson_95CI_Low": ci_low,
                    "Wilson_95CI_High": ci_high,
                    "Undetermined_Rate": undet_rate,
                }
            )

            r_list.append(n_res)
            n_eval_list.append(n_eval)

        # Compute trend test across the 3 years for this drug
        z_stat, p_val = cochran_armitage_test(r_list, n_eval_list, years)
        trend_rows.append(
            {
                "Drug": drug,
                "Time_Points": "2010->2013->2014",
                "N_2010_eval": n_eval_list[0],
                "N_2013_eval": n_eval_list[1],
                "N_2014_eval": n_eval_list[2],
                "Resistant_2010": r_list[0],
                "Resistant_2013": r_list[1],
                "Resistant_2014": r_list[2],
                "CA_Z_stat": z_stat,
                "P_value": p_val,
                "Significant_p05": p_val < 0.05,
            }
        )

    prev_df = pd.DataFrame(prevalence_rows)
    trend_df = pd.DataFrame(trend_rows)

    return prev_df, trend_df


def main():
    print("=== TANGA LONGITUDINAL STATISTICAL ANALYSIS ===")
    df = build_tz_resistance_table()

    prev_df, trend_df = analyze_tanga_longitudinal(df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prev_path = REPORTS_DIR / "tanga_longitudinal_prevalence.tsv"
    trend_path = REPORTS_DIR / "tanga_trend_tests.tsv"

    prev_df.to_csv(prev_path, sep="\t", index=False)
    trend_df.to_csv(trend_path, sep="\t", index=False)

    print(f"\nSaved prevalence & CIs to: {prev_path}")
    print(f"Saved trend test results to: {trend_path}")

    print("\n--- Summary: Significant Trends in Tanga (2010 -> 2013 -> 2014) ---")
    sig_trends = trend_df[trend_df["Significant_p05"]]
    if len(sig_trends) == 0:
        print("No statistically significant monotonic trends observed (p < 0.05).")
    else:
        for _, row in sig_trends.iterrows():
            direction = "INCREASE" if row["CA_Z_stat"] > 0 else "DECREASE"
            print(
                f"  - {row['Drug']:<20}: {direction} (Z = {row['CA_Z_stat']:.3f}, p = {row['P_value']:.4e})"
            )

    print("\n--- Summary: All Drug Trends (Tanga) ---")
    print(
        trend_df[
            [
                "Drug",
                "Resistant_2010",
                "Resistant_2013",
                "Resistant_2014",
                "CA_Z_stat",
                "P_value",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
