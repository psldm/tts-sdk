#!/usr/bin/env python3
"""
tts_tables.py — Reproduce Tables 1-4 of TTS v3.0 from first principles.

Table 1: Angular radius alpha(Omega_tot) of PDS matched circles (Eqs 60-62)
Table 2: Published circle searches and their coverage (literature compilation)
Table 3: Search floors translated into Omega_tot through Eqs (60)-(62)
Table 4: Computed suppression S_l = C^PDS_l / C^S3_l (Sachs-Wolfe approx)

Tables 1, 3, and 4 are computed from the equations in the paper; Table 2 is
a literature compilation transcribed from the cited searches.
Reference: Moss & Luminet, "The Theory of Temporal Spheres", v3.0, 2026.

Author: Eva Moss
License: MIT
"""

from typing import Any

import numpy as np
from scipy.integrate import quad
import json
import os

# ============================================================
# Conjugacy classes of I* (binary icosahedral group, |I*|=120)
# Half-angles alpha_i and class sizes |C_i| from Appendix E.2
# ============================================================
CONJ_CLASSES = [
    # (|C_i|, half_angle_radians, label)
    (1, 0.0, "C1 identity"),
    (1, np.pi, "C2 central -1"),
    (12, np.pi / 5, "C3 order-5a"),
    (12, 4 * np.pi / 5, "C4 order-5a"),
    (12, 2 * np.pi / 5, "C5 order-5b"),
    (12, 3 * np.pi / 5, "C6 order-5b"),
    (20, np.pi / 3, "C7 order-3"),
    (20, 2 * np.pi / 3, "C8 order-3"),
    (30, np.pi / 2, "C9 order-2"),
]

# ============================================================
# Eq (60): Comoving distance to last scattering surface
# ============================================================


def E_squared(
    z: float, Omega_m: float, Omega_r: float, Omega_tot: float, Omega_Lambda: float
) -> float:
    """E^2(z) from Eq (60)."""
    return (
        Omega_r * (1 + z) ** 4
        + Omega_m * (1 + z) ** 3
        + (1 - Omega_tot) * (1 + z) ** 2
        + Omega_Lambda
    )


def chi_rec(Omega_tot: float, Omega_m: float = 0.315, z_rec: float = 1090) -> float:
    """
    Eq (60): comoving distance to LSS on S^3.

    chi_rec(Omega_tot) = sqrt(Omega_tot - 1) * integral_0^z_rec dz / E(z)

    Uses radiation density Omega_r = 9.1e-5 (photons + neutrinos)
    and Omega_Lambda = 1 - Omega_m - Omega_r - (1 - Omega_tot).

    The result is dimensionless (in units of the curvature radius), so the
    Hubble parameter h cancels and does not appear as an argument; the
    background is fully specified by Omega_m and Omega_tot.
    """
    Omega_r = 9.1e-5
    Omega_Lambda = 1.0 - Omega_m - Omega_r - (1.0 - Omega_tot)

    def integrand(z):
        return 1.0 / np.sqrt(E_squared(z, Omega_m, Omega_r, Omega_tot, Omega_Lambda))

    integral, _ = quad(integrand, 0, z_rec, limit=200)
    return np.sqrt(max(Omega_tot - 1.0, 0)) * integral


# ============================================================
# Eq (61): Injectivity radius of PDS
# ============================================================
R_INJ = np.pi / 10.0  # 0.31416 radians


# ============================================================
# Eq (62): Angular radius of matched circles
# ============================================================


def alpha_deg(Omega_tot: float, Omega_m: float = 0.315, z_rec: float = 1090) -> float:
    """
    Eq (62): cos(alpha) = tan(r_inj) / tan(chi_rec)

    Returns alpha in degrees, or NaN if circles don't exist.
    """
    chi = chi_rec(Omega_tot, Omega_m, z_rec)
    tan_chi = np.tan(chi)
    tan_r = np.tan(R_INJ)

    if abs(tan_chi) < 1e-15:
        return np.nan

    cos_alpha = tan_r / tan_chi

    if cos_alpha < -1 or cos_alpha > 1:
        return np.nan

    return np.degrees(np.arccos(cos_alpha))


# ============================================================
# Table 1: alpha(Omega_tot) for Planck 2018 and WMAP parameters
# ============================================================


def compute_table1() -> dict[str, Any]:
    """
    Table 1: Angular radius of PDS matched circles from Eqs (60)-(62).

    Left block: Planck 2018 background (Omega_m=0.315); right block: WMAP
    first-year background (Omega_m=0.27). The h values quoted in the paper
    (0.674 / 0.71) are labels only — h cancels in the dimensionless chi_rec
    and does not affect these columns.
    """
    omega_vals = np.arange(1.008, 1.021, 0.001)

    results = []
    for om in omega_vals:
        a_planck = alpha_deg(om, Omega_m=0.315)
        chi_planck = np.degrees(chi_rec(om, Omega_m=0.315))

        a_wmap = alpha_deg(om, Omega_m=0.27)
        chi_wmap = np.degrees(chi_rec(om, Omega_m=0.27))

        results.append(
            {
                "Omega_tot": round(om, 4),
                "chi_rec_planck_deg": round(chi_planck, 2),
                "alpha_planck_deg": (
                    round(a_planck, 1) if np.isfinite(a_planck) else None
                ),
                "chi_rec_wmap_deg": round(chi_wmap, 2),
                "alpha_wmap_deg": round(a_wmap, 1) if np.isfinite(a_wmap) else None,
            }
        )

    # Consistency anchor: alpha ~ 35 deg at Omega_tot = 1.013 (Luminet et al. 2003)
    anchor = alpha_deg(1.013, Omega_m=0.27)

    return {"table": results, "anchor_35_deg_at_1.013_wmap": round(anchor, 1)}


# ============================================================
# Table 2: Published circle searches (literature compilation)
# ============================================================


def compute_table2() -> list[dict[str, Any]]:
    """
    Table 2: Published circle searches and their coverage.
    Compiled from original papers, not computed.
    """
    searches = [
        {
            "search": "Cornish et al. 2004 [20]",
            "data": "WMAP1",
            "geometry": "b2b and nearly b2b (centre sep > 170 deg); all phases via FFT",
            "floor_deg": "25 (b2b), 28 (nearly b2b)",
            "notes": "excludes PDS circles of radius 35 deg; authors state search does not rule out finite universe",
        },
        {
            "search": "Key et al. 2007 [22]",
            "data": "WMAP1/3",
            "geometry": "dedicated PDS six-pair combination statistic, twists +/-36 deg, optimal filtering",
            "floor_deg": "16.4 direct; 5 by extrapolation",
            "notes": "deepest PDS-specific claim rests on extrapolated portion of efficiency curve",
        },
        {
            "search": "Aurich et al. 2006 [24]",
            "data": "WMAP1",
            "geometry": "filtered simultaneous search for 3 spherical forms, +/-36 deg",
            "floor_deg": "-",
            "notes": "marginal hint of right-handed PDS at Omega_tot ~ 1.015; no firm conclusion",
        },
        {
            "search": "Lew & Roukema 2008 [27]",
            "data": "WMAP1/3",
            "geometry": "significance test of 11 deg, -36 deg signal of Ref [26]",
            "floor_deg": "-",
            "notes": "data consistent with simply connected space at 68% level",
        },
        {
            "search": "Roukema et al. 2008 [28]",
            "data": "WMAP3",
            "geometry": "spatial cross-correlation xi_C/xi_A, MCMC over orientation, alpha, twist",
            "floor_deg": "-",
            "notes": "optimal twist +39 +/-2.5 deg; chance probability 6-9% in simply connected sky",
        },
        {
            "search": "Bielewicz & Banday 2011 [29]",
            "data": "WMAP7",
            "geometry": "b2b, phased and anti-phased, phi_s scanned",
            "floor_deg": "~10",
            "notes": "lower bound on fundamental domain",
        },
        {
            "search": "Vaudrevange et al. 2012 [30]",
            "data": "WMAP7",
            "geometry": "general geometry: 10<=alpha<=90, centre sep 11-180, all phases",
            "floor_deg": "10",
            "notes": "widest search performed to date",
        },
        {
            "search": "Planck 2013 XXVI [31]",
            "data": "Planck T",
            "geometry": "b2b, phi_s scanned",
            "floor_deg": "20",
            "notes": "R_i > 0.94 chi_rec",
        },
        {
            "search": "Planck 2015 XVIII [32]",
            "data": "Planck T and E",
            "geometry": "b2b, S_max^+/-, phi_s scanned",
            "floor_deg": "~15 (conservative); 10-15 (extrapolated)",
            "notes": "99% CL exclusion of b2b circles larger than alpha_min, assuming orientation-and-mask allows detection; polarization at least as strong as temperature",
        },
    ]
    return searches


# ============================================================
# Table 3: Search floors translated into Omega_tot
# ============================================================


def compute_table3() -> list[dict[str, Any]]:
    """
    Table 3: Invert Eq (62) to map each floor onto the largest Omega_tot
    whose circles it cannot see.

    For a given floor alpha_floor, find Omega_tot such that
    alpha(Omega_tot) = alpha_floor.
    Below that Omega_tot, circles are smaller than the floor or absent.
    """
    floors = [
        ("no circles (alpha=0)", 0.0),
        ("10 deg [29, 30]", 10.0),
        ("15 deg [32]", 15.0),
        ("16.4 deg direct [22]", 16.4),
        ("25 deg [20]", 25.0),
        ("28 deg nearly b2b [20]", 28.0),
    ]

    results = []
    for label, floor_deg in floors:
        # Binary search for Omega_tot where alpha(Omega_tot) = floor_deg
        if floor_deg == 0:
            # Find Omega_tot where circles first appear (alpha -> 0)
            om_planck = _find_circle_threshold(Omega_m=0.315)
            om_wmap = _find_circle_threshold(Omega_m=0.27)
        else:
            om_planck = _invert_alpha(floor_deg, Omega_m=0.315)
            om_wmap = _invert_alpha(floor_deg, Omega_m=0.27)

        results.append(
            {
                "floor": label,
                "Omega_tot_planck": (
                    round(om_planck, 4) if np.isfinite(om_planck) else None
                ),
                "Omega_tot_wmap": round(om_wmap, 4) if np.isfinite(om_wmap) else None,
            }
        )

    return results


def _find_circle_threshold(Omega_m=0.315, z_rec: float = 1090):
    """Find Omega_tot where circles first appear (alpha -> 0)."""
    # alpha -> 0 when cos(alpha) -> 1, i.e. tan(r_inj) = tan(chi_rec)
    # i.e. chi_rec = r_inj = pi/10
    target_chi = R_INJ

    lo, hi = 1.0, 1.1
    for _ in range(100):
        mid = (lo + hi) / 2
        chi = chi_rec(mid, Omega_m, z_rec)
        if chi < target_chi:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _invert_alpha(target_alpha_deg, Omega_m=0.315, z_rec: float = 1090):
    """Find Omega_tot such that alpha(Omega_tot) = target_alpha_deg."""
    target = np.radians(target_alpha_deg)
    target_cos = np.cos(target)

    # alpha increases with Omega_tot, so cos(alpha) decreases.
    # Search in [1.005, 1.02] where circles exist.
    lo, hi = 1.005, 1.02
    for _ in range(200):
        mid = (lo + hi) / 2
        chi = chi_rec(mid, Omega_m, z_rec)
        if abs(chi) < 1e-15:
            return np.nan
        cos_a = np.tan(R_INJ) / np.tan(chi)
        if not np.isfinite(cos_a):
            return np.nan
        # cos(alpha) decreases as Omega_tot increases
        # We want cos(alpha) = target_cos
        if cos_a > target_cos:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ============================================================
# Eq (63): Multiplicity via Burnside's lemma (character sum)
# ============================================================


def su2_character(k: int, alpha: float) -> float:
    """
    Character of the (k+1)-dimensional representation D(k/2) of SU(2),
    evaluated at element with half-angle alpha.

    chi_k(alpha) = sin((k+1)*alpha) / sin(alpha)

    Special cases: chi_k(0) = k+1, chi_k(pi) = (-1)^k * (k+1)
    """
    if abs(alpha) < 1e-12:
        return float(k + 1)
    if abs(alpha - np.pi) < 1e-12:
        return float((-1) ** k * (k + 1))
    return np.sin((k + 1) * alpha) / np.sin(alpha)


def multiplicity_mk(k: int) -> float:
    """
    Eq (63): m_k = (1/|I*|) * sum_i |C_i| * chi_k(alpha_i)

    Multiplicity of trivial representation of I* in D(k/2).
    Returns 0 for odd k (parity selection).
    """
    if k % 2 == 1:
        return 0

    total = 0.0
    for size, alpha, _ in CONJ_CLASSES:
        total += size * su2_character(k, alpha)

    return total / 120.0


# ============================================================
# Eq (64): Angular power spectrum on S^3 and on PDS
# ============================================================


def _ultraspherical_Pi(k: int, ell: int, chi: float) -> float:
    """
    Hyperspherical Bessel function Pi^k_ell(chi) on S^3.

    The orthonormal radial eigenfunctions on S^3 are Jacobi (Gegenbauer)
    polynomials carrying a sin^ell(chi) prefactor:

      Pi^k_ell(chi) = sin^ell(chi) * P^{(a,a)}_n(cos chi) / (h_n * sqrt(4*pi))

    with a = ell + 1/2, n = k - ell, and the Jacobi norm

      h^2_n = 2^{2a+1} * Gamma(n+a+1)^2 / ((2n+2a+1) * n! * Gamma(n+2a+1))

    The sin^ell(chi) factor is essential: the Jacobi weight
    (1-x^2)^{ell+1/2} = sin^{2*ell+1}(chi) splits into the S^3 volume
    measure sin^2(chi) and a sin^{2*ell}(chi) part belonging to the
    eigenfunction itself. With this normalization the addition theorem
    holds exactly:

      sum_{ell=0}^{k} (2*ell+1) * |Pi^k_ell(chi)|^2 = (k+1)^2 / (2*pi^2)

    which fixes the relative projection weights across both k and ell
    entering the spectrum ratio S_ell = C^PDS_ell / C^S3_ell.
    """
    from scipy.special import eval_jacobi, gammaln

    n = k - ell
    if n < 0:
        return 0.0

    x = np.cos(chi)
    s = np.sin(chi)
    if abs(x) > 1 or s <= 0:
        return 0.0

    a = ell + 0.5

    # Jacobi polynomial P^{(a,a)}_n(cos chi)
    jac = eval_jacobi(n, a, a, x)

    # Use log-space to avoid overflow for large k
    # log(h^2) = (2a+1)*log(2) + 2*gammaln(n+a+1) - log(2n+2a+1) - gammaln(n+1) - gammaln(n+2a+1)
    log_h2 = (
        (2 * a + 1) * np.log(2.0)
        + 2 * gammaln(n + a + 1)
        - np.log(2 * n + 2 * a + 1)
        - gammaln(n + 1)
        - gammaln(n + 2 * a + 1)
    )

    # Pi^k_ell(chi) = sin^ell(chi) * P^{(a,a)}_n(cos chi) / (h_n * sqrt(4*pi)),
    # assembled in log-space
    log_pref = ell * np.log(s) - 0.5 * log_h2 - 0.5 * np.log(4.0 * np.pi)
    return jac * np.exp(log_pref)


def _sachs_wolfe_spectrum(k: int) -> float:
    """
    Scale-invariant primordial spectrum P(k) in the convention of Ref [23].
    P(beta) = alpha_s / (beta * (beta^2 - 1))
    where beta = k + 1.
    """
    beta = k + 1
    if beta <= 1:
        return 0.0
    # A_s ~ 2.1e-9, but we use normalized units for ratio computation
    return 1.0 / (beta * (beta**2 - 1))


def _spectrum_pair(
    ell: int, Omega_tot: float, k_max: int = 250, chi: float | None = None
) -> tuple[float, float]:
    """
    Compute (C^PDS_ell, C^S3_ell) from Eqs (64)-(65).

    C^S3_ell  = sum_{k>=ell} P(k+1) * |Pi^k_ell(chi_rec)|^2
    C^PDS_ell = sum_{k>=ell, k even, m_k>0} (120*m_k/(k+1)) * P(k+1) * |Pi^k_ell(chi_rec)|^2

    With orthonormal modes and per-mode primordial power P(k+1), each
    (ell, m) receives exactly one mode per level k on S^3, so the S^3
    level weight is 1 — the full (k+1)^2 multiplicity is already
    encoded in the addition theorem for Pi^k_ell. On the PDS only the
    I*-invariant subspace of dimension m_k*(k+1) survives, spread over
    the (k+1)^2 (ell, m) slots, and each mode carries the volume
    normalization Vol(S^3)/Vol(PDS) = 120 (Eq 64):

      120 * m_k*(k+1)/(k+1)^2 = 120*m_k/(k+1)

    The projection weight |Pi^k_ell(chi_rec)|^2 depends on Omega_tot
    through chi_rec, giving the Omega_tot dependence of the spectra.
    Pass chi to reuse a precomputed chi_rec(Omega_tot).
    """
    if chi is None:
        chi = chi_rec(Omega_tot)

    cs3 = 0.0
    cpds = 0.0

    for k in range(ell, k_max + 1):
        pk = _sachs_wolfe_spectrum(k)
        if pk == 0:
            continue

        pi_val = _ultraspherical_Pi(k, ell, chi)
        w2 = pi_val * pi_val

        # S^3: one orthonormal mode per (ell, m) at each level k
        cs3 += pk * w2

        # PDS: only even k with m_k > 0
        if k % 2 == 1:
            continue
        mk = round(multiplicity_mk(k))
        if mk <= 0:
            continue

        # Level weight: 120 * m_k / (k+1) from Eq (64)
        cpds += (120.0 * mk / (k + 1)) * pk * w2

    return cpds, cs3


def compute_spectrum_ratio(ell: int, Omega_tot: float, k_max: int = 250) -> float:
    """
    Compute S_ell = C^PDS_ell / C^S3_ell using Eqs (64)-(65).

    See _spectrum_pair for the level weights and normalization.
    """
    cpds, cs3 = _spectrum_pair(ell, Omega_tot, k_max)

    if cs3 > 0:
        return cpds / cs3
    return 0.0


# ============================================================
# Table 4: Suppression S_l and Roe as function of Omega_tot
# ============================================================


def compute_table4() -> list[dict[str, Any]]:
    """
    Table 4: Computed suppression S_ell = C^PDS_ell / C^S3_ell
    in the Sachs-Wolfe approximation, Planck 2018 background.

    Roe = odd-to-even ratio of ell(ell+1)*C_ell summed over ell=2-20,
    normalized to the same ratio on S^3.
    """
    omega_vals = [1.008, 1.010, 1.012, 1.013, 1.014, 1.015, 1.0155, 1.016, 1.018, 1.020]

    results = []
    for om in omega_vals:
        # Compute S_2 (quadrupole) and S_3 (octopole)
        s2 = compute_spectrum_ratio(2, om)
        s3 = compute_spectrum_ratio(3, om)

        # Compute Roe: odd-to-even ratio
        roe = _compute_roe(om)

        results.append(
            {
                "Omega_tot": om,
                "S2": round(s2, 4),
                "S3": round(s3, 4),
                "Roe": round(roe, 4) if np.isfinite(roe) else None,
            }
        )

    return results


def _compute_roe(Omega_tot: float, ell_max: int = 20, k_max: int = 250) -> float:
    """
    Roe = odd-to-even ratio of ell*(ell+1)*C_ell summed over ell = 2..ell_max
    on the PDS, normalized to the same ratio on S^3:

      Roe = [sum_odd l(l+1) C^PDS_l / sum_even l(l+1) C^PDS_l]
          / [sum_odd l(l+1) C^S3_l  / sum_even l(l+1) C^S3_l]

    The S^3 normalization ratio is close to, but not exactly, 1: the
    Sachs-Wolfe plateau l(l+1)*C_l is only approximately flat on S^3,
    so the normalization is computed explicitly rather than assumed.
    """
    chi = chi_rec(Omega_tot)

    odd_pds = 0.0
    even_pds = 0.0
    odd_s3 = 0.0
    even_s3 = 0.0

    for ell in range(2, ell_max + 1):
        cpds, cs3 = _spectrum_pair(ell, Omega_tot, k_max, chi=chi)
        w = ell * (ell + 1)

        if ell % 2 == 1:
            odd_pds += w * cpds
            odd_s3 += w * cs3
        else:
            even_pds += w * cpds
            even_s3 += w * cs3

    if even_pds > 0 and odd_s3 > 0 and even_s3 > 0:
        return (odd_pds / even_pds) / (odd_s3 / even_s3)
    return np.nan


# ============================================================
# Molien series verification (Appendix E.5)
# ============================================================


def molien_series_coeff(k: int) -> int:
    """
    Coefficient m_k of the Molien series M(t) = (1 + t^30) / ((1-t^12)(1-t^20)).

    Computed via Burnside's lemma (Eq 63) and verified against
    the closed form (Eq 139).
    """
    return round(multiplicity_mk(k))


def verify_molien_series(k_max: int = 200) -> dict[str, Any]:
    """
    Verify that Burnside's lemma (Eq 63) reproduces the Molien series
    coefficients from the closed form (Eq 139) for all k <= k_max.

    M(t) = (1 + t^30) / ((1 - t^12)(1 - t^20))
    = 1 + t^12 + t^20 + t^24 + t^30 + t^32 + t^36 + ...

    Coefficients: m_k = number of ways to write k = 12a + 20b + 30c
    with c in {0, 1} and a, b >= 0.
    """
    results = []
    for k in range(k_max + 1):
        # Burnside computation (rounded to handle floating point)
        mk_burnside = round(multiplicity_mk(k))

        # Closed-form computation
        mk_closed = 0
        for c in [0, 1]:
            remainder = k - 30 * c
            if remainder < 0:
                continue
            for a in range(remainder // 12 + 1):
                rest = remainder - 12 * a
                if rest >= 0 and rest % 20 == 0:
                    mk_closed += 1

        if mk_burnside != mk_closed:
            results.append(
                {
                    "k": k,
                    "burnside": mk_burnside,
                    "closed_form": mk_closed,
                    "match": False,
                }
            )

    return {
        "verified": len(results) == 0,
        "mismatches": results,
        "k_max": k_max,
    }


# ============================================================
# Klein-CRT correspondence (Appendix E.6)
# ============================================================


def klein_crt_table() -> list[dict[str, Any]]:
    """
    The three Klein invariant degrees and their CRT signatures.

    Each invariant activates exactly one CRT component of Z/60Z = Z/3 x Z/4 x Z/5.
    """
    return [
        {
            "invariant": "f (icosahedral)",
            "degree": 12,
            "mod3": 0,
            "mod4": 0,
            "mod5": 2,
            "ladder_step": "A4 -> A5 (pentagonal)",
        },
        {
            "invariant": "H (Hessian)",
            "degree": 20,
            "mod3": 2,
            "mod4": 0,
            "mod5": 0,
            "ladder_step": "{e} -> Z3 (triangular)",
        },
        {
            "invariant": "T (Jacobian)",
            "degree": 30,
            "mod3": 0,
            "mod4": 2,
            "mod5": 0,
            "ladder_step": "Z3 -> A4 (tetrahedral)",
        },
    ]


# ============================================================
# Complete multiplicity table (Appendix E.7)
# ============================================================


def compute_multiplicity_table(k_max: int = 60) -> list[dict[str, Any]]:
    """
    Complete multiplicity table up to k = k_max.

    k: level
    m_k: invariant multiplicity (Burnside)
    d^PDS_k = m_k * (k+1): total PDS eigenmodes
    (k+1)^2: S^3 multiplicity for comparison
    """
    results = []
    for k in range(k_max + 1):
        mk = multiplicity_mk(k)
        dpds = int(mk * (k + 1))
        s3_mult = (k + 1) ** 2
        results.append(
            {
                "k": k,
                "m_k": int(mk),
                "d_PDS_k": dpds,
                "S3_mult": s3_mult,
            }
        )
    return results


# ============================================================
# Main: reproduce all tables
# ============================================================


def reproduce_all_tables() -> None:
    """Reproduce Tables 1-4 and verification from first principles."""
    print("=" * 72)
    print("  TTS v3.0 — TABLE REPRODUCTION FROM FIRST PRINCIPLES")
    print("  Eqs (60)-(65), Appendix E")
    print("=" * 72)

    # Table 1
    print("\n--- Table 1: alpha(Omega_tot) from Eqs (60)-(62) ---")
    t1 = compute_table1()
    print(
        f"  Anchor: alpha ~ {t1['anchor_35_deg_at_1.013_wmap']} deg at Omega_tot=1.013 (WMAP)"
    )
    print(
        f"  {'Omega_tot':>10} | {'chi_Planck':>10} {'alpha_Planck':>12} | {'chi_WMAP':>10} {'alpha_WMAP':>12}"
    )
    print(f"  {'-'*10} | {'-'*10} {'-'*12} | {'-'*10} {'-'*12}")
    for row in t1["table"]:
        ap = (
            f"{row['alpha_planck_deg']:.1f}"
            if row["alpha_planck_deg"] is not None
            else "-"
        )
        aw = (
            f"{row['alpha_wmap_deg']:.1f}" if row["alpha_wmap_deg"] is not None else "-"
        )
        print(
            f"  {row['Omega_tot']:10.4f} | {row['chi_rec_planck_deg']:10.2f} {ap:>12} | {row['chi_rec_wmap_deg']:10.2f} {aw:>12}"
        )

    # Table 2
    print("\n--- Table 2: Published circle searches ---")
    t2 = compute_table2()
    for s in t2:
        print(f"  {s['search']}")
        print(f"    Data: {s['data']}, Floor: {s['floor_deg']}")
        print(f"    Notes: {s['notes']}")
        print()

    # Table 3
    print("--- Table 3: Floors -> Omega_tot ---")
    t3 = compute_table3()
    print(f"  {'Floor':<30} | {'Omega_tot (Planck)':>18} | {'Omega_tot (WMAP)':>18}")
    print(f"  {'-'*30} | {'-'*18} | {'-'*18}")
    for row in t3:
        op = (
            f"{row['Omega_tot_planck']:.4f}"
            if row["Omega_tot_planck"] is not None
            else "-"
        )
        ow = (
            f"{row['Omega_tot_wmap']:.4f}" if row["Omega_tot_wmap"] is not None else "-"
        )
        print(f"  {row['floor']:<30} | {op:>18} | {ow:>18}")

    # Table 4
    print("\n--- Table 4: Suppression S_l and Roe ---")
    t4 = compute_table4()
    print(f"  {'Omega_tot':>10} | {'S2':>8} | {'S3':>8} | {'Roe':>8}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")
    for row in t4:
        roe = f"{row['Roe']:.4f}" if row["Roe"] is not None else "-"
        print(
            f"  {row['Omega_tot']:10.4f} | {row['S2']:8.4f} | {row['S3']:8.4f} | {roe:>8}"
        )

    # Molien verification
    print("\n--- Molien series verification (Appendix E.5) ---")
    mol = verify_molien_series(k_max=200)
    print(f"  Verified to k={mol['k_max']}: {mol['verified']}")
    if not mol["verified"]:
        for m in mol["mismatches"]:
            print(
                f"  MISMATCH at k={m['k']}: Burnside={m['burnside']}, closed={m['closed_form']}"
            )

    # Klein-CRT
    print("\n--- Klein-CRT correspondence (Appendix E.6) ---")
    for entry in klein_crt_table():
        print(
            f"  {entry['invariant']}: degree={entry['degree']}, "
            f"mod3={entry['mod3']}, mod4={entry['mod4']}, mod5={entry['mod5']}, "
            f"step={entry['ladder_step']}"
        )

    # Multiplicity table
    print("\n--- Multiplicity table (Appendix E.7, k<=60) ---")
    mtable = compute_multiplicity_table(k_max=60)
    print(f"  {'k':>4} | {'m_k':>4} | {'d_PDS_k':>8} | {'(k+1)^2':>8}")
    print(f"  {'-'*4} | {'-'*4} | {'-'*8} | {'-'*8}")
    for row in mtable:
        if row["m_k"] > 0 or row["k"] in [0, 45]:
            print(
                f"  {row['k']:4d} | {row['m_k']:4d} | {row['d_PDS_k']:8d} | {row['S3_mult']:8d}"
            )

    # Save to JSON
    output = {
        "table1": t1,
        "table2": t2,
        "table3": t3,
        "table4": t4,
        "molien_verified": mol,
        "klein_crt": klein_crt_table(),
        "multiplicity_table": mtable,
    }

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "tts_tables.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  All tables saved to: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    reproduce_all_tables()
