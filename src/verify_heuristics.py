"""
verify_heuristics.py

Systematically recomputes resistance status from raw marker genotypes for
drugs where a testable rule exists, and compares against the Pf8 inferred
resistance classification file.

IMPORTANT - rule provenance is tracked explicitly:
  DOCUMENTED = rule text independently confirmed from Pf8_resistance_classification.pdf
  EMPIRICAL  = rule discovered by pattern-matching against the data during
               diagnostic inspection; NOT yet cross-checked against PDF rule text
  NOT TESTED = drug has a known/suspected multi-locus or complex rule that has
               not been extracted/verified; no rule is guessed or fabricated here

Output: printed report + saved summary table at
reports/heuristic_verification_summary.tsv
"""

from pathlib import Path
import pandas as pd

import load_data as ld
from load_markers import load_markers

DOCUMENTED = "DOCUMENTED (confirmed in Pf8_resistance_classification.pdf)"
EMPIRICAL = "EMPIRICAL (pattern-matched against classification data; NOT yet cross-checked against PDF rule text)"
NOT_TESTED = "NOT TESTED (multi-locus/complex rule; not extracted from PDF - not guessed)"


RULES = {
    "Chloroquine": {
        "locus": "crt_76[K]",
        "wild_type": "K",
        "mutant": "T",
        "provenance": DOCUMENTED,
        "reference": "Djimde et al., Lancet 358:890-891 (2001); PF3D7_0709000 codon 76",
    },
    "Pyrimethamine": {
        "locus": "dhfr_108[S]",
        "wild_type": "S",
        "mutant": "N",
        "provenance": DOCUMENTED,
        "reference": (
            "Pf8_resistance_classification.pdf, Pyrimethamine section: "
            "'parasites were classified as pyrimethamine resistant if this "
            "[dhfr 108N] mutation was present irrespective of other dhfr and "
            "dhps alleles.' Locus: PF3D7_0417200 (dhfr), codon 108. "
            "Cites: Peterson et al., PNAS 85:9109-13 (1988)."
        ),
    },
     "Sulfadoxine": {
        "locus": "dhps_437[G]",
        "wild_type": "A",
        "mutant": "G",
        "provenance": EMPIRICAL,
        "reference": (
            "Pf8_resistance_classification.pdf confirms locus PF3D7_0810800 "
            "(dhps), codon 437 is used for Sulfadoxine, but the PDF's "
            "Step/Classification workflow table for this drug was not "
            "captured in text extraction (grep cut off before table rows). "
            "wild_type='A', mutant='G' determined empirically from data "
            "(G correlates with Resistant, A with Sensitive/Undetermined), "
            "consistent with the well-established dhps A437G resistance "
            "mutation in the literature. Needs final confirmation against "
            "full PDF table text before being upgraded to DOCUMENTED."
        ),
    },
}

NOT_TESTED_DRUGS = [
    "Mefloquine", "Artemisinin", "Piperaquine",
    "SP (uncomplicated)", "SP (IPTp)", "AS-MQ", "DHA-PPQ",
]


def classify_single_locus(value, wild_type: str, mutant: str) -> str:
    """
    Classify a single-locus genotype call into Sensitive / Resistant / Undetermined.
    Handles heterozygous calls in either allele order (e.g. 'K,T' or 'T,K')
    and missing calls encoded as '-'. Any unexpected pattern is explicitly
    classed as Undetermined rather than silently dropped.
    """
    if pd.isna(value):
        return "Undetermined"
    value = str(value).strip()
    if value == wild_type:
        return "Sensitive"
    if value == mutant:
        return "Resistant"
    if value == "-":
        return "Undetermined"
    alleles = set(value.split(","))
    if alleles == {wild_type, mutant}:
        return "Undetermined"
    return "Undetermined"


def run_verification() -> pd.DataFrame:
    markers = load_markers()
    tz_resistance = ld.build_tz_resistance_table()
    merged = tz_resistance.merge(markers, on="Sample", how="inner")
    assert len(merged) == ld.EXPECTED_TANZANIA_QCPASS, (
        f"Merged row count mismatch: got {len(merged)}, expected {ld.EXPECTED_TANZANIA_QCPASS}"
    )

    results = []

    print("=" * 70)
    print("HEURISTIC VERIFICATION REPORT")
    print("=" * 70)

    for drug, rule in RULES.items():
        locus = rule["locus"]
        predicted = merged[locus].apply(
            lambda v: classify_single_locus(v, rule["wild_type"], rule["mutant"])
        )
        actual = merged[drug]

        match = predicted == actual
        n_match = int(match.sum())
        n_total = len(merged)
        match_rate = n_match / n_total

        print(f"\n--- {drug} ---")
        print(f"Locus tested: {locus}")
        print(f"Rule provenance: {rule['provenance']}")
        print(f"Reference/notes: {rule['reference']}")
        print(f"Match rate: {n_match}/{n_total} ({match_rate:.4%})")

        if n_match != n_total:
            mismatches = merged.loc[~match, ["Sample", locus, drug]].copy()
            mismatches["predicted"] = predicted[~match]
            print(f"Mismatches ({len(mismatches)}):")
            print(mismatches.to_string(index=False))

        results.append({
            "Drug": drug,
            "Locus": locus,
            "Provenance": rule["provenance"],
            "Match": n_match,
            "Total": n_total,
            "MatchRate": round(match_rate, 4),
        })

    print("\n" + "=" * 70)
    print("DRUGS NOT TESTED (multi-locus/complex rules not yet extracted from PDF):")
    for drug in NOT_TESTED_DRUGS:
        print(f"  - {drug}")
        results.append({
            "Drug": drug,
            "Locus": "N/A (multi-locus/complex)",
            "Provenance": NOT_TESTED,
            "Match": None,
            "Total": None,
            "MatchRate": None,
        })

    summary = pd.DataFrame(results)
    out_path = ld.REPO_ROOT / "reports" / "heuristic_verification_summary.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved summary table to: {out_path}")

    return summary


if __name__ == "__main__":
    run_verification()
