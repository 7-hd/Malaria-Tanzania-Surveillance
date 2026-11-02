"""
load_markers.py

Loads the Pf8 drug resistance marker genotype table and joins it to the
verified Tanzania QC-pass + resistance status table (from load_data.py).

This script is intentionally cautious: it hard-asserts only what has been
independently verified (row counts, the PE0230-C worked example from manual
inspection). It PRINTS (but does not yet assert) the raw value encodings for
key loci, so we can confirm how missing/heterozygous calls are represented
before writing further validation logic.
"""

from pathlib import Path
import pandas as pd

import load_data as ld

MARKERS_PATH = ld.DATA_RAW / "Pf8_drug_resistance_marker_genotypes.tsv"

EXPECTED_MARKER_COLUMNS = [
    "Sample", "crt_72[C]", "crt_74[M]", "crt_75[N]", "crt_76[K]",
    "crt_72-76[CVMNK]", "crt_93[T]", "crt_97[H]", "crt_218[I]", "crt_220[A]",
    "crt_271[Q]", "crt_326[N]", "crt_333[T]", "crt_353[G]", "crt_356[I]",
    "crt_371[R]", "dhfr_16[N]", "dhfr_51[N]", "dhfr_59[C]", "dhfr_108[S]",
    "dhfr_164[I]", "dhfr_306[S]", "dhps_436[S]", "dhps_437[G]", "dhps_540[K]",
    "dhps_581[A]", "dhps_613[A]", "exo_415[E]", "mdr1_86[N]", "mdr1_184[Y]",
    "mdr1_1034[S]", "mdr1_1042[N]", "mdr1_1226[F]", "mdr1_1246[D]",
    "arps10_127-128[VD]", "fd_193[D]", "mdr2_484[T]",
    "kelch13_349-726_ns_changes", "mdr1_dup_call", "pm2_dup_call",
]

# Verified by manual terminal inspection (PE0230-C, Tanzania/Tanga/2013)
WORKED_EXAMPLE_SAMPLE = "PE0230-C"
WORKED_EXAMPLE_EXPECTED = {
    "crt_72[C]": "C",
    "crt_74[M]": "M",
    "crt_75[N]": "N",
    "crt_76[K]": "K",
    "crt_72-76[CVMNK]": "CVMNK",
    "dhfr_51[N]": "I",
    "dhfr_59[C]": "R",
    "dhfr_108[S]": "N",
    "dhps_540[K]": "E",
    "dhps_581[A]": "G",
    "mdr1_184[Y]": "F",
    "mdr1_dup_call": "0",
    "pm2_dup_call": "0",
}


def load_markers() -> pd.DataFrame:
    df = pd.read_csv(MARKERS_PATH, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    missing_cols = set(EXPECTED_MARKER_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_MARKER_COLUMNS)
    assert not missing_cols, f"Missing expected columns: {missing_cols}"
    assert not extra_cols, f"Unexpected extra columns: {extra_cols}"

    return df


def check_worked_example(markers: pd.DataFrame) -> None:
    row = markers[markers["Sample"] == WORKED_EXAMPLE_SAMPLE]
    assert len(row) == 1, (
        f"Expected exactly 1 row for {WORKED_EXAMPLE_SAMPLE}, got {len(row)}"
    )
    row = row.iloc[0]

    print(f"\n--- Worked example check: {WORKED_EXAMPLE_SAMPLE} ---")
    for col, expected_val in WORKED_EXAMPLE_EXPECTED.items():
        actual_val = str(row[col]).strip()
        status = "OK" if actual_val == expected_val else "MISMATCH"
        print(f"  {col}: got '{actual_val}', expected '{expected_val}' [{status}]")
        assert actual_val == expected_val, f"{col} mismatch for {WORKED_EXAMPLE_SAMPLE}"


def inspect_key_loci_encoding(merged: pd.DataFrame) -> None:
    """
    Diagnostic only (not yet asserted): print the actual unique values seen
    for a few key loci, so we understand how missing / heterozygous calls
    are encoded in this file before writing further validation logic.
    """
    key_cols = ["crt_76[K]", "dhfr_108[S]", "dhps_540[K]", "kelch13_349-726_ns_changes"]
    print("\n--- Diagnostic: raw value encodings for key loci ---")
    for col in key_cols:
        print(f"\n{col} value counts:")
        print(merged[col].value_counts(dropna=False))


def main():
    print(f"Loading markers from: {MARKERS_PATH}")
    markers = load_markers()
    print(f"Total marker rows loaded: {len(markers)}")

    check_worked_example(markers)

    tz_resistance = ld.build_tz_resistance_table()

    merged = tz_resistance.merge(markers, on="Sample", how="inner")
    assert len(merged) == ld.EXPECTED_TANZANIA_QCPASS, (
        f"Merged row count mismatch: got {len(merged)}, expected {ld.EXPECTED_TANZANIA_QCPASS}"
    )
    print(f"\nFinal joined table row count: {len(merged)} [expected {ld.EXPECTED_TANZANIA_QCPASS}]")

    inspect_key_loci_encoding(merged)

    out_path = ld.REPO_ROOT / "data" / "processed" / "tz_markers_with_resistance.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved joined table to: {out_path}")


if __name__ == "__main__":
    main()
