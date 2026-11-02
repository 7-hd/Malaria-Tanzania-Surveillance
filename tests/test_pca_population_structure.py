
"""
tests/test_pca_population_structure.py

Unit tests for marker genotype encoding and PCA dimensionality reduction.
"""

import numpy as np
import pandas as pd
import pytest
from pca_population_structure import encode_marker_genotypes, run_pca


def test_encode_marker_genotypes():
    # Mock marker dataframe
    mock_data = {
        "Sample": ["S1", "S2", "S3", "S4"],
        "crt_76[K]": ["K", "T", "K,T", "-"],  # Ref, Alt, Het, Missing
        "dhfr_108[S]": ["S", "S", "N", "N"],  # Ref, Ref, Alt, Alt
        "invariant_locus[A]": ["A", "A", "A", "A"],  # Invariant
        "kelch13_349-726_ns_changes": ["none", "H697Y", "none", "-"],  # Complex
    }
    df = pd.DataFrame(mock_data)

    encoded, variant_cols = encode_marker_genotypes(df)

    # Invariant and complex columns should be dropped
    assert "invariant_locus[A]" not in variant_cols
    assert "kelch13_349-726_ns_changes" not in variant_cols
    assert set(variant_cols) == {"crt_76[K]", "dhfr_108[S]"}

    # Value checks for crt_76[K]:
    # 'K' -> 0.0, 'T' -> 1.0, 'K,T' -> 0.5, '-' -> mode (0.0)
    crt_vals = encoded["crt_76[K]"].tolist()
    assert crt_vals[0] == 0.0
    assert crt_vals[1] == 1.0
    assert crt_vals[2] == 0.5
    assert crt_vals[3] in [0.0, 1.0, 0.5]


def test_run_pca():
    # Mock numeric matrix
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(20, 5))

    coords, pca = run_pca(X, n_components=3)

    assert coords.shape == (20, 3)
    assert len(pca.explained_variance_ratio_) == 3
    assert np.sum(pca.explained_variance_ratio_) <= 1.0
