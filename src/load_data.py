"""
load_data.py

Loads the three core Pf8 flat files, filters to Tanzania QC-pass samples,
and prints summary counts.

This script is a reproducibility checkpoint: every printed count is checked
against numbers independently verified by hand via `awk` against the raw
files (see project notes). If any assertion fails, STOP — do not proceed
to analysis until the discrepancy is understood and resolved.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"

SAMPLES_PATH = DATA_RAW / "Pf8_samples.txt"
MARKERS_PATH = DATA_RAW / "Pf8_drug_resistance_marker_genotypes.tsv"
CLASSIFICATION_PATH = DATA_RAW / "Pf8_inferred_resistance_status_classification.tsv"

# ---------------------------------------------------------------------------
# Verified ground truth (from manual awk checks against the raw files)
# ---------------------------------------------------------------------------
EXPECTED_TOTAL_TANZANIA = 1144
EXPECTED_TANZANIA_QCPASS = 1025
EXPECTED_TANZANIA_QCFAIL = 119

# (Study, Year, Region) -> QC-pass count
EXPECTED_BREAKDOWN = {
    ("1007-PF-TZ-DUFFY", 2010, "Tanga"): 15,
    ("1007-PF-TZ-DUFFY", 2010, "Morogoro"): 32,
    ("1095-PF-TZ-ISHENGOMA", 2013, "Tanga"): 152,
    ("1095-PF-TZ-ISHENGOMA", 2013, "Lindi"): 65,
    ("1095-PF-TZ-ISHENGOMA", 2013, "Kagera"): 52,
    ("1167-PF-TZ-ISHENGOMA-SM", 2014, "Tanga"): 130,
    ("1167-PF-TZ-ISHENGOMA-SM", 2014, "Kigoma"): 143,
    ("1263-PF-TZ-BWIRE-SM", 2016, "Pwani"): 4,
    ("1263-PF-TZ-BWIRE-SM", 2019, "Pwani"): 432,
}

EXPECTED_RESISTANCE_COUNTS = {
    "Chloroquine":        {"Resistant": 87,   "Sensitive": 886, "Undetermined": 52},
    "Pyrimethamine":      {"Resistant": 1001, "Sensitive": 9,   "Undetermined": 15},
    "Sulfadoxine":        {"Resistant": 851,  "Sensitive": 92,  "Undetermined": 82},
    "Mefloquine":         {"Resistant": 0,    "Sensitive": 945, "Undetermined": 80},
    "Artemisinin":        {"Resistant": 0,    "Sensitive": 981, "Undetermined": 44},
    "Piperaquine":        {"Resistant": 0,    "Sensitive": 957, "Undetermined": 68},
    "SP (uncomplicated)": {"Resistant": 811,  "Sensitive": 97,  "Undetermined": 117},
    "SP (IPTp)":          {"Resistant": 133,  "Sensitive": 814, "Undetermined": 78},
    "AS-MQ":              {"Resistant": 0,    "Sensitive": 1005, "Undetermined": 20},
    "DHA-PPQ":            {"Resistant": 0,    "Sensitive": 999, "Undetermined": 26},
}


def load_samples() -> pd.DataFrame:
    df = pd.read_csv(SAMPLES_PATH, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def filter_tanzania(df: pd.DataFrame) -> pd.DataFrame:
    tz = df[df["Country"].str.strip() == "Tanzania"].copy()
    assert len(tz) == EXPECTED_TOTAL_TANZANIA, (
        f"Tanzania total mismatch: got {len(tz)}, expected {EXPECTED_TOTAL_TANZANIA}"
    )
    return tz


def filter_qcpass(tz: pd.DataFrame) -> pd.DataFrame:
    qc = tz["QC pass"].astype(str).str.strip().str.lower()
    qcpass = tz[qc == "true"].copy()
    qcfail = tz[qc == "false"].copy()

    assert len(qcpass) == EXPECTED_TANZANIA_QCPASS, (
        f"QC-pass count mismatch: got {len(qcpass)}, expected {EXPECTED_TANZANIA_QCPASS}"
    )
    assert len(qcfail) == EXPECTED_TANZANIA_QCFAIL, (
        f"QC-fail count mismatch: got {len(qcfail)}, expected {EXPECTED_TANZANIA_QCFAIL}"
    )
    return qcpass


def check_breakdown(qcpass: pd.DataFrame) -> None:
    qcpass = qcpass.copy()
    qcpass["Year"] = qcpass["Year"].astype(float).astype(int)

    counts = (
        qcpass.groupby(["Study", "Year", "Admin level 1"])
        .size()
        .to_dict()
    )

    print("\n--- Study / Year / Region breakdown (QC-pass) ---")
    for key, expected_count in EXPECTED_BREAKDOWN.items():
        actual_count = counts.get(key, 0)
        status = "OK" if actual_count == expected_count else "MISMATCH"
        print(f"{key}: got {actual_count}, expected {expected_count} [{status}]")
        assert actual_count == expected_count, f"Breakdown mismatch at {key}"

    # Make sure there are no unexpected extra groups
    unexpected = set(counts.keys()) - set(EXPECTED_BREAKDOWN.keys())
    assert not unexpected, f"Unexpected groups found: {unexpected}"

    total = sum(counts.values())
    assert total == EXPECTED_TANZANIA_QCPASS, (
        f"Sum of breakdown ({total}) != total QC-pass ({EXPECTED_TANZANIA_QCPASS})"
    )


def load_classification() -> pd.DataFrame:
    df = pd.read_csv(CLASSIFICATION_PATH, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def check_resistance_counts(qcpass: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    # Join key: 'Sample' in samples file vs first column (likely 'sample') in classification file
    sample_col_class = classification.columns[0]

    merged = qcpass.merge(
        classification,
        left_on="Sample",
        right_on=sample_col_class,
        how="inner",
    )

merged = merged.drop(columns=[sample_col_class])  # drop duplicate lowercase 'sample' col
merged["Year"] = merged["Year"].astype(float).astype(int)  # normalize Year to clean int

    assert len(merged) == EXPECTED_TANZANIA_QCPASS, (
        f"Merged row count mismatch: got {len(merged)}, expected {EXPECTED_TANZANIA_QCPASS} "
        "(possible dropped/duplicated samples in join)"
    )

    print("\n--- Resistance status breakdown (Tanzania, QC-pass) ---")
    for drug, expected_counts in EXPECTED_RESISTANCE_COUNTS.items():
        actual_counts = merged[drug].value_counts().to_dict()
        print(f"\n{drug}:")
        for status, expected_n in expected_counts.items():
            actual_n = actual_counts.get(status, 0)
            flag = "OK" if actual_n == expected_n else "MISMATCH"
            print(f"  {status}: got {actual_n}, expected {expected_n} [{flag}]")
            assert actual_n == expected_n, f"{drug}/{status} mismatch"

    return merged


def main():
    print(f"Loading samples from: {SAMPLES_PATH}")
    samples = load_samples()
    print(f"Total rows loaded: {len(samples)}")

    tz = filter_tanzania(samples)
    print(f"Tanzania total (QC pass + fail): {len(tz)} [expected {EXPECTED_TOTAL_TANZANIA}]")

    tz_qcpass = filter_qcpass(tz)
    print(f"Tanzania QC-pass: {len(tz_qcpass)} [expected {EXPECTED_TANZANIA_QCPASS}]")

    check_breakdown(tz_qcpass)

    classification = load_classification()
    merged = check_resistance_counts(tz_qcpass, classification)

    print("\nAll reproducibility checks PASSED.")

    # Save the verified, filtered dataset for downstream use
    out_path = REPO_ROOT / "data" / "processed" / "tz_qcpass_with_resistance.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved filtered dataset to: {out_path}")


if __name__ == "__main__":
    main()
