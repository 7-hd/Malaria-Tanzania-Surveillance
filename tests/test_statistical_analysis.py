"""
tests/test_statistical_analysis.py

Unit tests for statistical formulas (Wilson score CIs and Cochran-Armitage trend tests).
"""

import pytest
from scipy import stats
from statistical_analysis import cochran_armitage_test, wilson_ci


def test_wilson_ci_bounds():
    # 0 out of 10 -> lower bound must be 0, upper bound > 0
    low, high = wilson_ci(0, 10)
    assert low == 0.0
    assert 0.0 < high < 0.35

    # 10 out of 10 -> upper bound must be 1.0, lower bound < 1.0
    low, high = wilson_ci(10, 10)
    assert 0.65 < low < 1.0
    assert high == 1.0

    # 50 out of 100 -> center should be near 0.5
    low, high = wilson_ci(50, 100)
    assert 0.40 < low < 0.42
    assert 0.58 < high < 0.60


def test_cochran_armitage_flat_trend():
    # Constant proportion -> Z should be ~0, p-val = 1.0
    r = [10, 10, 10]
    n = [100, 100, 100]
    t = [2010, 2013, 2014]
    z, p = cochran_armitage_test(r, n, t)
    assert abs(z) < 1e-6
    assert p == 1.0


def test_cochran_armitage_strong_decline():
    # Strong decreasing trend (matches Tanga Chloroquine pattern)
    r = [15, 25, 13]
    n = [15, 152, 130]
    t = [2010, 2013, 2014]
    z, p = cochran_armitage_test(r, n, t)
    assert z < -5.0  # Strong negative Z
    assert p < 1e-5  # Highly significant p-value
