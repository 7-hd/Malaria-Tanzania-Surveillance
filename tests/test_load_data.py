"""
test_load_data.py

Pytest wrapper around the reproducibility checks in src/load_data.py.
These tests protect the verified ground-truth counts (from manual awk
checks against the raw Pf8 files) against silent breakage from future
refactoring.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import load_data as ld  # noqa: E402


@pytest.fixture(scope="module")
def samples_df():
    return ld.load_samples()


@pytest.fixture(scope="module")
def tz_df(samples_df):
    return ld.filter_tanzania(samples_df)


@pytest.fixture(scope="module")
def tz_qcpass_df(tz_df):
    return ld.filter_qcpass(tz_df)


@pytest.fixture(scope="module")
def classification_df():
    return ld.load_classification()


@pytest.fixture(scope="module")
def merged_df(tz_qcpass_df, classification_df):
    return ld.check_resistance_counts(tz_qcpass_df, classification_df)


def test_total_rows_loaded(samples_df):
    assert len(samples_df) == 33325


def test_tanzania_total(tz_df):
    assert len(tz_df) == ld.EXPECTED_TOTAL_TANZANIA


def test_tanzania_qcpass_count(tz_qcpass_df):
    assert len(tz_qcpass_df) == ld.EXPECTED_TANZANIA_QCPASS


def test_study_year_region_breakdown(tz_qcpass_df):
    # Will raise AssertionError internally if any mismatch found
    ld.check_breakdown(tz_qcpass_df)


def test_merged_row_count(merged_df):
    assert len(merged_df) == ld.EXPECTED_TANZANIA_QCPASS


def test_no_duplicate_sample_column(merged_df):
    # Only one sample identifier column should remain after cleanup
    sample_like_cols = [c for c in merged_df.columns if c.lower() == "sample"]
    assert len(sample_like_cols) == 1


def test_year_is_integer_type(merged_df):
    assert pd.api.types.is_integer_dtype(merged_df["Year"])


@pytest.mark.parametrize("drug,status,expected", [
    (drug, status, count)
    for drug, statuses in ld.EXPECTED_RESISTANCE_COUNTS.items()
    for status, count in statuses.items()
])
def test_resistance_status_counts(merged_df, drug, status, expected):
    actual = merged_df[drug].value_counts().get(status, 0)
    assert actual == expected, f"{drug}/{status}: got {actual}, expected {expected}"
