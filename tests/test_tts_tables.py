"""Tests for tts_tables.py against published values of TTS v3.0.

The reference values come from the paper (Moss & Luminet, "The Theory of
Temporal Spheres", v3.0): Table 1 anchor (Luminet et al. 2003), the Molien
series of I*, and Table 4 (p. 41, Sachs-Wolfe approximation, k_max = 250).
"""

import numpy as np
import pytest

from tts_tables import (
    _compute_roe,
    _ultraspherical_Pi,
    alpha_deg,
    compute_spectrum_ratio,
    multiplicity_mk,
    verify_molien_series,
)


def test_table1_anchor_wmap():
    # Consistency anchor: alpha ~ 35 deg at Omega_tot = 1.013 for the
    # WMAP-era background (Omega_m = 0.27), Luminet et al. (2003).
    anchor = alpha_deg(1.013, Omega_m=0.27)
    assert anchor == pytest.approx(35.4, abs=0.1)


def test_molien_multiplicities():
    assert round(multiplicity_mk(0)) == 1
    assert round(multiplicity_mk(12)) == 1
    assert round(multiplicity_mk(45)) == 0
    assert round(multiplicity_mk(60)) == 2


def test_eigenmode_gap():
    # No invariant modes exist for 1 <= k <= 11 (the eigenmode gap).
    assert all(round(multiplicity_mk(k)) == 0 for k in range(1, 12))


def test_molien_series_closed_form():
    result = verify_molien_series(k_max=100)
    assert result["verified"]
    assert result["mismatches"] == []


def test_addition_theorem():
    # sum_ell (2l+1) |Pi^k_ell(chi)|^2 = (k+1)^2 / (2 pi^2), exactly.
    for k in (0, 1, 5, 12):
        for chi in (0.3, 1.0):
            total = sum(
                (2 * ell + 1) * _ultraspherical_Pi(k, ell, chi) ** 2
                for ell in range(k + 1)
            )
            assert total == pytest.approx((k + 1) ** 2 / (2 * np.pi**2), rel=1e-10)


def test_table4_row_10155():
    # Paper Table 4 (p. 41), row Omega_tot = 1.0155:
    # S2 = 0.212, S3 = 1.107, Roe = 1.067.
    assert compute_spectrum_ratio(2, 1.0155) == pytest.approx(0.212, abs=1e-3)
    assert compute_spectrum_ratio(3, 1.0155) == pytest.approx(1.107, abs=1e-3)
    assert _compute_roe(1.0155) == pytest.approx(1.067, abs=1e-3)
