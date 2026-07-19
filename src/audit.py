#!/usr/bin/env python3
"""Comprehensive audit of the dodeca-h0 pipeline.

Runs a battery of consistency checks covering dodecahedron geometry,
Pantheon+ data loading, sector assignment, H0 fitting, covariance-matrix
handling, Monte Carlo mocks, and a regression comparison against the
published TTS v3.0 (Moss & Luminet) article numbers.

Usage
-----
    python src/audit.py
"""

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dodecahedron import (
    assign_sectors,
    get_dodecahedron_normals,
    load_pantheon_data,
    radec_to_cartesian,
)
from h0_fit import (
    C_SPEED,
    M_B,
    chi2,
    fit_all_sectors,
    fit_h0_sector,
    load_covariance_matrix,
)
from simulations import generate_mock_data, run_monte_carlo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Pantheon+SH0ES.dat")
COV_PATH = os.path.join(BASE_DIR, "..", "data", "Pantheon+SH0ES_STAT+SYS.cov")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

passed = 0
total = 0
issues: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and print the result of a single audit check.

    Parameters
    ----------
    name : str
        Short identifier of the check (e.g. "1.1a All normals unit length").
    condition : bool
        True if the check passed.
    detail : str, optional
        Diagnostic message printed and stored when the check fails.
    """
    global passed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        issues.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} — {detail}")


def section(title: str) -> None:
    """Print a section header.

    Parameters
    ----------
    title : str
        Section title to display.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_geometry() -> np.ndarray:
    """CHECK 1: validate the dodecahedron face-normal geometry.

    Verifies unit length of the normals, the expected set of pairwise
    angles, and uniform sky coverage of the induced sectors.

    Returns
    -------
    np.ndarray
        Array of shape (12, 3) with the face normals.
    """
    section("CHECK 1: DODECAHEDRON GEOMETRY")

    normals = get_dodecahedron_normals()
    print(f"  Generated {len(normals)} normals, shape: {normals.shape}")

    # 1.1a: unit length
    norms = np.linalg.norm(normals, axis=1)
    check(
        "1.1a All normals have unit length",
        np.allclose(norms, 1.0, atol=1e-10),
        f"min={norms.min():.12f}, max={norms.max():.12f}",
    )

    # 1.1b: pairwise dot products
    dots = normals @ normals.T
    upper = dots[np.triu_indices(12, k=1)]
    unique_dots = np.unique(np.round(upper, 10))
    n_unique = len(unique_dots)
    print(f"  Unique pairwise dot products: {n_unique}")
    for v in unique_dots:
        angle = np.degrees(np.arccos(np.clip(v, -1, 1)))
        print(f"    cos={v:+.6f}  →  angle={angle:.2f}°")
    check(
        "1.1b Three unique angles between normals",
        n_unique == 3,
        f"Found {n_unique} unique values: {unique_dots}",
    )

    # 1.1c: minimum dot product
    min_dot = upper.min()
    max_angle = np.degrees(np.arccos(np.clip(min_dot, -1, 1)))
    print(f"  Min dot product: {min_dot:.6f} (angle={max_angle:.2f}°)")
    check(
        "1.1c Min dot product ≈ 0.447 (angle ≈ 63.4°)",
        abs(min_dot - 0.4472) < 0.01,
        f"min_dot={min_dot:.6f}, expected ~0.447",
    )

    # 1.2: uniform coverage
    n_random = 10000
    rng = np.random.default_rng(12345)
    ra_rand = rng.uniform(0, 360, n_random)
    dec_rand = np.degrees(np.arcsin(rng.uniform(-1, 1, n_random)))
    sector_rand = assign_sectors(ra_rand, dec_rand, normals)
    _, counts_rand = np.unique(sector_rand, return_counts=True)
    fractions = counts_rand / n_random
    print(
        f"  Sector fractions: min={fractions.min():.4f}, max={fractions.max():.4f}, "
        f"expected=1/12={1/12:.4f}"
    )
    check(
        "1.2 Uniform coverage (max deviation < 20%)",
        fractions.max() - fractions.min() < 0.20,
        f"min={fractions.min():.4f}, max={fractions.max():.4f}, "
        f"spread={fractions.max()-fractions.min():.4f}",
    )

    return normals


def check_data_loading() -> pd.DataFrame:
    """CHECK 2: validate Pantheon+ data loading and the z < 0.1 filter.

    The expected count of 741 SNe is the published TTS v3.0 sample size
    and is checked as a regression against the article.

    Returns
    -------
    pd.DataFrame
        Filtered supernova sample (z < 0.1) with columns z, ra, dec, mb.
    """
    section("CHECK 2: DATA LOADING AND FILTERING")

    df_full = pd.read_csv(DATA_PATH, sep=r"\s+")
    n_total = len(df_full)
    print(f"  Total SNe in file: {n_total}")

    df = load_pantheon_data(DATA_PATH)
    n_filtered = len(df)
    print(f"  SNe after z<0.1 filter: {n_filtered}")
    print(f"  z range: [{df['z'].min():.4f}, {df['z'].max():.4f}]")
    print(f"  mb range: [{df['mb'].min():.2f}, {df['mb'].max():.2f}]")

    nan_z = df["z"].isna().sum()
    nan_ra = df["ra"].isna().sum()
    nan_dec = df["dec"].isna().sum()
    nan_mb = df["mb"].isna().sum()
    print(f"  NaN counts: z={nan_z}, ra={nan_ra}, dec={nan_dec}, mb={nan_mb}")

    check(
        "2.1 No NaN values in filtered data",
        nan_z == 0 and nan_ra == 0 and nan_dec == 0 and nan_mb == 0,
    )

    check(
        "2.2 Expected 741 SNe at z<0.1",
        n_filtered == 741,
        f"Got {n_filtered}, expected 741",
    )

    return df


def check_sector_assignment(
    df: pd.DataFrame, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CHECK 3: validate the assignment of SNe to dodecahedron sectors.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered supernova sample.
    normals : np.ndarray
        Face normals of shape (12, 3).

    Returns
    -------
    sector_ids : np.ndarray
        Sector index (0-11) for each SN.
    counts : np.ndarray
        Number of SNe per sector.
    """
    section("CHECK 3: SECTOR ASSIGNMENT")

    sector_ids = assign_sectors(df["ra"].values, df["dec"].values, normals)
    _, counts = np.unique(sector_ids, return_counts=True)

    print(
        f"  {'Sector':>8s}  {'Count':>6s}  {'Mean_RA':>9s}  {'Mean_Dec':>9s}  "
        f"{'Min_AngDist':>12s}  {'Max_AngDist':>12s}"
    )
    print(f"  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*12}  {'-'*12}")

    positions = radec_to_cartesian(df["ra"].values, df["dec"].values)
    cos_dists = np.dot(positions, normals.T)
    cos_dists = np.clip(cos_dists, -1, 1)
    ang_dists = np.degrees(np.arccos(cos_dists))

    for s in range(12):
        mask = sector_ids == s
        n = mask.sum()
        if n > 0:
            mean_ra = df["ra"].values[mask].mean()
            mean_dec = df["dec"].values[mask].mean()
            min_ad = ang_dists[mask, s].min()
            max_ad = ang_dists[mask, s].max()
            print(
                f"  {s:8d}  {n:6d}  {mean_ra:9.2f}  {mean_dec:9.2f}  "
                f"{min_ad:12.2f}  {max_ad:12.2f}"
            )
        else:
            print(f"  {s:8d}  {n:6d}  {'--':>9s}  {'--':>9s}  {'--':>12s}  {'--':>12s}")

    check("3.1 No empty sectors", all(counts > 0))

    max_ang = ang_dists[np.arange(len(sector_ids)), sector_ids].max()
    print(f"  Max angular distance to assigned normal: {max_ang:.2f}°")
    check(
        "3.2 Max angular distance < 45°",
        max_ang < 45.0,
        f"max_ang_dist={max_ang:.2f}°, expected < 45°",
    )

    return sector_ids, counts


def check_h0_fitting(
    df: pd.DataFrame, sector_ids: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """CHECK 4: validate the H0 fitting machinery.

    Checks the single-SN analytical solution, the chi2 scan around the
    minimum and its chi2+1 contour, the effect of the covariance matrix,
    and the reduced chi2 per sector.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered supernova sample.
    sector_ids : np.ndarray
        Sector index for each SN.
    counts : np.ndarray
        Number of SNe per sector.

    Returns
    -------
    cov_full : np.ndarray
        Full Pantheon+ covariance matrix.
    cov_sub : np.ndarray
        Covariance submatrix restricted to the z < 0.1 sample.
    largest_sector : int
        Index of the most populated sector.
    """
    section("CHECK 4: H0 FITTING")

    cov_full = load_covariance_matrix(COV_PATH)
    df_full = pd.read_csv(DATA_PATH, sep=r"\s+")
    mask_z = df_full["zHD"] < 0.1
    orig_indices = np.where(mask_z)[0]
    cov_sub = cov_full[np.ix_(orig_indices, orig_indices)]

    # 4.1: single SN analytical check
    z1 = df["z"].values[0]
    mb1 = df["mb"].values[0]
    H0_analytical = C_SPEED * z1 / (10 ** ((mb1 - M_B - 25) / 5))
    print(f"  Single SN test: z={z1:.6f}, mb={mb1:.4f}")
    print(f"  Analytical H0: {H0_analytical:.6f}")

    H0_fit_1, _, _, _ = fit_h0_sector(np.array([z1]), np.array([mb1]), np.eye(1))
    print(f"  Fitted H0:     {H0_fit_1:.6f}")
    check(
        "4.1 Single SN analytical vs fitted H0",
        abs(H0_fit_1 - H0_analytical) < 1e-6,
        f"analytical={H0_analytical:.6f}, fitted={H0_fit_1:.6f}, "
        f"diff={abs(H0_fit_1-H0_analytical):.2e}",
    )

    # 4.2: chi2 scan for largest sector
    largest_sector = int(np.argmax(counts))
    mask_ls = sector_ids == largest_sector
    indices_ls = np.where(mask_ls)[0]
    z_ls = df["z"].values[indices_ls]
    mb_ls = df["mb"].values[indices_ls]
    cov_ls = cov_sub[np.ix_(indices_ls, indices_ls)]
    cov_inv_ls = np.linalg.pinv(cov_ls)

    H0_best_ls, err_low_ls, err_high_ls, chi2_min_ls = fit_h0_sector(
        z_ls, mb_ls, cov_ls
    )
    print(f"\n  Largest sector: {largest_sector}, N={len(z_ls)}")
    print(
        f"  Fitted H0: {H0_best_ls:.2f}  -{err_low_ls:.2f}  +{err_high_ls:.2f}  "
        f"chi2_min={chi2_min_ls:.2f}"
    )

    h0_scan = np.linspace(50, 100, 500)
    chi2_scan = np.array([chi2(h, z_ls, mb_ls, cov_inv_ls) for h in h0_scan])
    h0_scan_min = h0_scan[np.argmin(chi2_scan)]
    chi2_scan_min = chi2_scan.min()
    print(f"  Scan minimum: H0={h0_scan_min:.2f}, chi2={chi2_scan_min:.2f}")
    check(
        "4.2a Chi2 scan minimum matches fit",
        abs(h0_scan_min - H0_best_ls) < 0.2,
        f"scan={h0_scan_min:.2f}, fit={H0_best_ls:.2f}",
    )

    # Check chi2+1 contour. For an increasing h0_scan, `above` looks like
    # T..T,F..F,T..T: the -1 transition (True -> False) is the LEFT edge
    # (lower H0) and the +1 transition (False -> True) is the RIGHT edge.
    chi2_target = chi2_scan_min + 1.0
    above = chi2_scan > chi2_target
    transitions = np.diff(above.astype(int))
    left_edge = h0_scan[np.where(transitions == -1)[0]]
    right_edge = h0_scan[np.where(transitions == 1)[0]]
    if len(left_edge) > 0 and len(right_edge) > 0:
        scan_low = H0_best_ls - left_edge[-1]
        scan_high = right_edge[0] - H0_best_ls
        print(f"  Scan 1σ: -{scan_low:.2f} +{scan_high:.2f}")
        check(
            "4.2b Chi2+1 contour matches error bars",
            abs(scan_low - err_low_ls) < 0.3 and abs(scan_high - err_high_ls) < 0.3,
            f"scan: -{scan_low:.2f}+{scan_high:.2f}, fit: -{err_low_ls:.2f}+{err_high_ls:.2f}",
        )

    # 4.3: with vs without covariance
    print(f"\n  {'Sector':>8s}  {'H0_cov':>8s}  {'H0_nocov':>8s}  {'Diff':>8s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    max_diff_cov = 0
    for s in range(12):
        mask = sector_ids == s
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        z_s = df["z"].values[indices]
        mb_s = df["mb"].values[indices]
        cov_s = cov_sub[np.ix_(indices, indices)]
        H0_cov, _, _, _ = fit_h0_sector(z_s, mb_s, cov_s)
        H0_nocov, _, _, _ = fit_h0_sector(z_s, mb_s, np.eye(len(indices)))
        diff = abs(H0_cov - H0_nocov)
        max_diff_cov = max(max_diff_cov, diff)
        print(f"  {s:8d}  {H0_cov:8.2f}  {H0_nocov:8.2f}  {diff:8.2f}")
    check(
        "4.3 Covariance matrix has noticeable effect",
        max_diff_cov > 0.1,
        f"max_diff={max_diff_cov:.2f} km/s/Mpc",
    )

    # 4.4: reduced chi2
    print(
        f"\n  {'Sector':>8s}  {'N':>5s}  {'chi2':>8s}  {'chi2/(N-1)':>12s}  {'Status':>10s}"
    )
    print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*12}  {'-'*10}")
    all_chi2_ok = True
    for s in range(12):
        mask = sector_ids == s
        indices = np.where(mask)[0]
        if len(indices) <= 1:
            continue
        z_s = df["z"].values[indices]
        mb_s = df["mb"].values[indices]
        cov_s = cov_sub[np.ix_(indices, indices)]
        _, _, _, chi2_s = fit_h0_sector(z_s, mb_s, cov_s)
        dof = len(indices) - 1
        red_chi2 = chi2_s / dof
        status = "OK" if 0.3 < red_chi2 < 3.0 else "SUSPICIOUS"
        if status != "OK":
            all_chi2_ok = False
        print(
            f"  {s:8d}  {len(indices):5d}  {chi2_s:8.2f}  {red_chi2:12.2f}  {status:>10s}"
        )
    check(
        "4.4 Reduced chi2 ≈ 1 for all sectors",
        all_chi2_ok,
        "Some sectors have suspicious reduced chi2",
    )

    return cov_full, cov_sub, largest_sector


def check_covariance(
    cov_full: np.ndarray,
    cov_sub: np.ndarray,
    sector_ids: np.ndarray,
    largest_sector: int,
) -> None:
    """CHECK 5: validate the covariance matrix and its submatrices.

    Parameters
    ----------
    cov_full : np.ndarray
        Full Pantheon+ covariance matrix.
    cov_sub : np.ndarray
        Covariance submatrix for the z < 0.1 sample.
    sector_ids : np.ndarray
        Sector index for each SN.
    largest_sector : int
        Index of the most populated sector.
    """
    section("CHECK 5: COVARIANCE MATRIX")

    print(f"  Full matrix shape: {cov_full.shape}")
    diag = np.diag(cov_full)
    off_diag = cov_full[~np.eye(cov_full.shape[0], dtype=bool)]
    print(
        f"  Diagonal:   min={diag.min():.6f}, max={diag.max():.6f}, "
        f"mean={diag.mean():.6f}, std={diag.std():.6f}"
    )
    print(
        f"  Off-diag:   min={off_diag.min():.6f}, max={off_diag.max():.6f}, "
        f"mean={off_diag.mean():.6f}"
    )

    eigvals_full = np.linalg.eigvalsh(cov_full)
    min_eig_full = eigvals_full.min()
    print(f"  Min eigenvalue: {min_eig_full:.6e}")
    check(
        "5.1a Full covariance is positive definite",
        min_eig_full > 0,
        f"min eigenvalue = {min_eig_full:.6e}",
    )

    # 5.2: submatrix check
    indices_ls = np.where(sector_ids == largest_sector)[0]
    cov_ls = cov_sub[np.ix_(indices_ls, indices_ls)]
    eigvals_ls = np.linalg.eigvalsh(cov_ls)
    min_eig_ls = eigvals_ls.min()
    print(f"\n  Submatrix for sector {largest_sector} (N={len(indices_ls)}):")
    print(f"  Min eigenvalue: {min_eig_ls:.6e}")

    diag_ls = np.diag(cov_ls)
    diag_full_ls = np.diag(cov_sub)[indices_ls]
    diag_match = np.allclose(diag_ls, diag_full_ls)
    print(f"  Diagonal matches full submatrix: {diag_match}")

    check(
        "5.2a Submatrix is positive definite",
        min_eig_ls > 0,
        f"min eigenvalue = {min_eig_ls:.6e}",
    )
    check("5.2b Submatrix diagonal matches full matrix", diag_match)


def check_monte_carlo(
    df: pd.DataFrame, cov_sub: np.ndarray, normals: np.ndarray
) -> None:
    """CHECK 6: validate mock generation and the Monte Carlo distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered supernova sample.
    cov_sub : np.ndarray
        Covariance submatrix for the z < 0.1 sample.
    normals : np.ndarray
        Face normals of shape (12, 3).
    """
    section("CHECK 6: MONTE CARLO")

    # 6.1: single mock
    df_mock = generate_mock_data(df, cov_sub, H0_true=70.0, random_seed=42)
    print(
        f"  Real data:   mean(mb)={df['mb'].mean():.4f}, std(mb)={df['mb'].std():.4f}"
    )
    print(
        f"  Mock data:   mean(mb)={df_mock['mb_mock'].mean():.4f}, "
        f"std(mb)={df_mock['mb_mock'].std():.4f}"
    )

    real_mean, real_std = df["mb"].mean(), df["mb"].std()
    mock_mean = df_mock["mb_mock"].mean()
    mean_diff_sigma = abs(mock_mean - real_mean) / (real_std / np.sqrt(len(df)))
    check(
        "6.1 Mock and real mb compatible within 2σ",
        mean_diff_sigma < 2.0,
        f"diff={abs(mock_mean-real_mean):.4f}, {mean_diff_sigma:.1f}σ",
    )

    # 6.2: fit mock
    sector_ids_mock = assign_sectors(
        df_mock["ra"].values, df_mock["dec"].values, normals
    )
    df_fit_mock = df_mock.rename(columns={"mb_mock": "mb"})
    results_mock = fit_all_sectors(df_fit_mock, sector_ids_mock, cov_sub)

    print("\n  Mock H0 per sector (H0_true=70.0):")
    all_compatible = True
    for r in results_mock:
        if r["n_sne"] > 0:
            err = max(r["err_low"], r["err_high"])
            compatible = abs(r["H0"] - 70.0) < 2 * err if err > 0 else True
            if not compatible:
                all_compatible = False
            flag = "" if compatible else "  <-- OUTSIDE 2σ"
            print(f"    Sector {r['sector_id']:2d}: H0={r['H0']:.2f} ± {err:.2f}{flag}")
    check(
        "6.2 All mock H0 compatible with H0_true=70 within 2σ",
        all_compatible,
        "Some sectors deviate from H0_true by >2σ",
    )

    # 6.3: MC distribution
    mc_path = os.path.join(OUTPUT_DIR, "mc_results.npz")
    if os.path.exists(mc_path):
        mc_data = dict(np.load(mc_path, allow_pickle=True))
        mock_deltas = mc_data["mock_delta_H0"]
        print(f"\n  MC delta_H0 distribution (n={len(mock_deltas)}):")
        print(f"    mean={mock_deltas.mean():.2f}, std={mock_deltas.std():.2f}")
        print(f"    min={mock_deltas.min():.2f}, max={mock_deltas.max():.2f}")

        # QQ plot
        fig, ax = plt.subplots(figsize=(6, 6))
        sp_stats.probplot(mock_deltas, dist="norm", plot=ax)
        ax.set_title("QQ-plot: MC delta_H0 vs Normal")
        fig.savefig(os.path.join(OUTPUT_DIR, "audit_qqplot.png"), dpi=150)
        plt.close(fig)

        # Shapiro-Wilk test
        sw_stat, sw_p = sp_stats.shapiro(
            mock_deltas[:5000] if len(mock_deltas) > 5000 else mock_deltas
        )
        is_normal = sw_p > 0.01
        print(f"    Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}")
        check(
            "6.3 MC delta_H0 distribution is approximately normal",
            is_normal,
            f"Shapiro-Wilk p={sw_p:.4f} (p<0.01 indicates non-normality)",
        )
    else:
        print("  MC results not found, running 200 mocks for check...")
        mc_results = run_monte_carlo(
            df, cov_sub, normals, n_mocks=200, H0_true=70.0, n_jobs=-1, random_seed=42
        )
        mock_deltas = mc_results["delta_H0"]
        print(f"    mean={mock_deltas.mean():.2f}, std={mock_deltas.std():.2f}")
        print(f"    min={mock_deltas.min():.2f}, max={mock_deltas.max():.2f}")


def check_article_comparison(
    df: pd.DataFrame, sector_ids: np.ndarray, cov_sub: np.ndarray
) -> None:
    """CHECK 7: regression comparison against the published article numbers.

    The hard-coded per-face table below (H0, uncertainty, and SNe count for
    each of the 12 dodecahedron faces) reproduces the values published in
    the TTS v3.0 article (Moss & Luminet). This is a deliberate regression
    check of the pipeline output against those published numbers, not an
    independently derived reference.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered supernova sample.
    sector_ids : np.ndarray
        Sector index for each SN.
    cov_sub : np.ndarray
        Covariance submatrix for the z < 0.1 sample.
    """
    section("CHECK 7: COMPARISON WITH ARTICLE")

    article = {
        1: {"H0": 67.96, "err": 0.83, "N": 40},
        2: {"H0": 69.92, "err": 1.54, "N": 15},
        3: {"H0": 70.09, "err": 0.55, "N": 12},
        4: {"H0": 71.35, "err": 0.99, "N": 49},
        5: {"H0": 70.31, "err": 0.54, "N": 103},
        6: {"H0": 70.82, "err": 0.49, "N": 124},
        7: {"H0": 71.14, "err": 0.78, "N": 55},
        8: {"H0": 70.61, "err": 0.61, "N": 85},
        9: {"H0": 70.32, "err": 1.05, "N": 26},
        10: {"H0": 72.98, "err": 0.86, "N": 57},
        11: {"H0": 73.98, "err": 1.61, "N": 22},
        12: {"H0": 72.02, "err": 0.82, "N": 40},
    }

    h0_results = fit_all_sectors(df, sector_ids, cov_sub)

    print(
        f"  {'Face':>6s}  {'H0_art':>8s}  {'H0_our':>8s}  {'Diff':>8s}  "
        f"{'N_art':>6s}  {'N_our':>6s}"
    )
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}")

    diffs = []
    n_diffs = []
    for face_id in range(1, 13):
        art = article[face_id]
        our = h0_results[face_id - 1]
        diff_h0 = art["H0"] - our["H0"]
        diff_n = art["N"] - our["n_sne"]
        diffs.append(diff_h0)
        n_diffs.append(diff_n)
        print(
            f"  {face_id:6d}  {art['H0']:8.2f}  {our['H0']:8.2f}  {diff_h0:8.2f}  "
            f"{art['N']:6d}  {our['n_sne']:6d}"
        )

    diffs = np.array(diffs)
    n_diffs = np.array(n_diffs)
    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)
    print(
        f"\n  Mean H0 diff (article - ours): {mean_diff:.2f} ± {std_diff:.2f} km/s/Mpc"
    )
    print(f"  Mean N diff (article - ours):   {n_diffs.mean():.1f}")

    check(
        "7.1 Mean H0 difference < 1 km/s/Mpc",
        abs(mean_diff) < 1.0,
        f"mean_diff={mean_diff:.2f} km/s/Mpc",
    )

    # Check N distribution correlation
    our_counts = np.array([h0_results[i]["n_sne"] for i in range(12)])
    art_counts = np.array([article[i + 1]["N"] for i in range(12)])
    corr_n = np.corrcoef(our_counts, art_counts)[0, 1]
    print(f"  Correlation of N distributions: {corr_n:.4f}")
    check(
        "7.2 N distribution correlates with article",
        corr_n > 0.5,
        f"correlation={corr_n:.4f} (different orientation if < 0.5)",
    )


def main() -> None:
    """Run the full audit suite and print the final PASS/FAIL summary."""
    warnings.filterwarnings("ignore")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    normals = check_geometry()
    df = check_data_loading()
    sector_ids, counts = check_sector_assignment(df, normals)
    cov_full, cov_sub, largest_sector = check_h0_fitting(df, sector_ids, counts)
    check_covariance(cov_full, cov_sub, sector_ids, largest_sector)
    check_monte_carlo(df, cov_sub, normals)
    check_article_comparison(df, sector_ids, cov_sub)

    section("FINAL VERDICT")

    print(f"\n  Checks passed: {passed}/{total}")
    print(f"  Checks failed: {total - passed}/{total}")

    if issues:
        print("\n  Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        print("\n  Recommendations:")
        print("    - Review the failed checks above")
        print("    - Verify assumptions and input data")
    else:
        print("\n  All checks passed. The pipeline appears correct.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
