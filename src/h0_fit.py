#!/usr/bin/env python3
"""
H0 fitting module for dodecahedron anisotropy analysis.

Fits the Hubble constant H0 for each of the 12 dodecahedron sectors
using weighted chi-squared minimization with the full covariance matrix.
"""

import sys
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dodecahedron import get_dodecahedron_normals, assign_sectors, load_pantheon_data

C_SPEED = 299792.458
M_B = -19.25


def load_covariance_matrix(filepath: str) -> np.ndarray | None:
    """
    Load the full covariance matrix from file.

    Supports text format (Pantheon+ style: first line is matrix size)
    and .npy/.npz formats. Falls back to identity matrix if file not found.

    Parameters:
        filepath (str): Path to the covariance matrix file.

    Returns:
        numpy.ndarray: Covariance matrix of shape (N, N).
    """
    if not os.path.exists(filepath):
        warnings.warn(
            f"Covariance matrix file '{filepath}' not found. "
            "Using identity matrix (diagonal errors only).",
            stacklevel=2,
        )
        return None

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".npy":
        cov = np.load(filepath)
    elif ext == ".npz":
        data = np.load(filepath)
        key = list(data.keys())[0]
        cov = data[key]
    else:
        with open(filepath, "r") as f:
            first_line = f.readline().strip()

        if len(first_line.split()) == 1:
            size = int(first_line)
            cov_flat = np.loadtxt(filepath, skiprows=1)
            cov = cov_flat.reshape((size, size))
        else:
            cov = np.loadtxt(filepath)
            if cov.ndim == 1:
                size = int(np.sqrt(len(cov)))
                cov = cov.reshape((size, size))

    return cov


def mb_theory(z: np.ndarray | float, H0: float, MB: float = M_B) -> np.ndarray:
    """
    Compute predicted apparent magnitude m_b for given redshift and H0.

    Uses the linear Hubble law approximation valid for z < 0.1:
        m_b = 5 * log10(c * z / H0) + 25 + MB

    Parameters:
        z (array-like): Redshift values.
        H0 (float): Hubble constant in km/s/Mpc.
        MB (float): Absolute magnitude of SNe Ia (default -19.25).

    Returns:
        numpy.ndarray: Predicted apparent magnitudes.
    """
    z = np.asarray(z, dtype=float)
    z_safe = np.maximum(z, 1e-10)
    return 5.0 * np.log10(C_SPEED * z_safe / H0) + 25.0 + MB


def chi2(H0: float, z: np.ndarray, mb_obs: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Compute chi-squared for a given H0 value.

    Parameters:
        H0 (float): Hubble constant in km/s/Mpc.
        z (array-like): Redshift values.
        mb_obs (array-like): Observed apparent magnitudes.
        cov_inv (numpy.ndarray): Inverse covariance matrix.

    Returns:
        float: Chi-squared value.
    """
    residuals = mb_obs - mb_theory(z, H0)
    return residuals.T @ cov_inv @ residuals


def fit_h0_sector(
    z: np.ndarray, mb_obs: np.ndarray, cov_submatrix: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Fit H0 for a single dodecahedron sector.

    Minimizes chi-squared over H0 in [50, 100] km/s/Mpc and estimates
    1-sigma errors from the chi-squared + 1 contour.

    Parameters:
        z (array-like): Redshift values for SNe in this sector.
        mb_obs (array-like): Observed apparent magnitudes.
        cov_submatrix (numpy.ndarray): Covariance submatrix for this sector.

    Returns:
        tuple: (H0_best, H0_err_low, H0_err_high, chi2_min)
    """
    z = np.asarray(z, dtype=float)
    mb_obs = np.asarray(mb_obs, dtype=float)

    cov_inv = np.linalg.pinv(cov_submatrix)

    result = minimize_scalar(
        lambda h: chi2(h, z, mb_obs, cov_inv), bounds=(50, 100), method="bounded"
    )
    H0_best = result.x
    chi2_min = result.fun

    target = chi2_min + 1.0

    def chi2_diff(h):
        return chi2(h, z, mb_obs, cov_inv) - target

    H0_err_low = 0.0
    H0_err_high = 0.0

    try:
        if chi2_diff(50.0) * chi2_diff(H0_best) < 0:
            H0_low = brentq(chi2_diff, 50.0, H0_best, xtol=1e-4)
            H0_err_low = H0_best - H0_low
    except (ValueError, RuntimeError):
        pass

    try:
        if chi2_diff(H0_best) * chi2_diff(100.0) < 0:
            H0_high = brentq(chi2_diff, H0_best, 100.0, xtol=1e-4)
            H0_err_high = H0_high - H0_best
    except (ValueError, RuntimeError):
        pass

    return H0_best, H0_err_low, H0_err_high, chi2_min


def fit_all_sectors(
    df: pd.DataFrame, sector_ids: np.ndarray, cov_full: np.ndarray | None
) -> list[dict[str, Any]]:
    """
    Fit H0 for all 12 dodecahedron sectors.

    Parameters
    ----------
    df : pandas.DataFrame
        Supernova data with columns ['z', 'mb']. Rows must be positionally
        aligned with `cov_full` (row i of `df` corresponds to row/column i
        of the covariance matrix).
    sector_ids : numpy.ndarray
        Sector assignment (0-11) for each SN.
    cov_full : numpy.ndarray or None
        Covariance matrix for exactly the SNe in `df` (N x N). If the SNe
        were filtered from a larger catalog, the covariance must be
        pre-sliced to the filtered set (see ``__main__`` below or
        run_all.py). If None, an identity matrix is used.

    Returns
    -------
    list of dict
        One dict per sector with keys
        [sector_id, n_sne, H0, err_low, err_high, chi2].
    """
    results = []

    for sector in range(12):
        mask = sector_ids == sector
        indices = np.where(mask)[0]

        if len(indices) == 0:
            results.append(
                {
                    "sector_id": sector,
                    "n_sne": 0,
                    "H0": np.nan,
                    "err_low": np.nan,
                    "err_high": np.nan,
                    "chi2": np.nan,
                }
            )
            continue

        z = df["z"].values[indices]
        mb_obs = df["mb"].values[indices]

        if cov_full is not None:
            cov_sub = cov_full[np.ix_(indices, indices)]
        else:
            cov_sub = np.eye(len(indices))

        H0_best, err_low, err_high, chi2_min = fit_h0_sector(z, mb_obs, cov_sub)

        results.append(
            {
                "sector_id": sector,
                "n_sne": len(indices),
                "H0": round(H0_best, 2),
                "err_low": round(err_low, 2),
                "err_high": round(err_high, 2),
                "chi2": round(chi2_min, 2),
            }
        )

    return results


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "Pantheon+SH0ES.dat"
    )
    cov_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "Pantheon+SH0ES_STAT+SYS.cov"
    )

    print("=" * 72)
    print("  DODECAHEDRAL H0 ANISOTROPY FIT")
    print("=" * 72)

    print("\n[1/4] Loading Pantheon+ data...")
    df = load_pantheon_data(data_path)
    print(f"      Loaded {len(df)} SNe with z < 0.1")

    print("\n[2/4] Assigning sectors...")
    normals = get_dodecahedron_normals()
    sector_ids = assign_sectors(df["ra"].values, df["dec"].values, normals)

    print("\n[3/4] Loading covariance matrix...")
    cov_full = load_covariance_matrix(cov_path)
    if cov_full is not None:
        # The covariance rows follow the ordering of the full catalog, while
        # df is the z < 0.1 subset with reset positional indices. Slice the
        # covariance with the original catalog indices of the surviving SNe
        # so that row i of df matches row/column i of the submatrix.
        df_full = pd.read_csv(data_path, sep=r"\s+")
        orig_indices = np.where(df_full["zHD"] < 0.1)[0]
        cov_sub = cov_full[np.ix_(orig_indices, orig_indices)]
        print(f"      Full covariance: {cov_full.shape}")
        print(f"      Sub-covariance:  {cov_sub.shape}")
    else:
        cov_sub = None
        print("      Using identity matrix (diagonal errors)")

    print("\n[4/4] Fitting H0 for 12 sectors...")
    results = fit_all_sectors(df, sector_ids, cov_sub)

    print("\n" + "-" * 72)
    print(
        f"  {'Sector':>6s}  {'N_SNe':>5s}  {'H0':>8s}  {'-1σ':>8s}  {'+1σ':>8s}  {'χ²_min':>8s}"
    )
    print("-" * 72)

    for r in results:
        if r["n_sne"] > 0:
            print(
                f"  {r['sector_id']:6d}  {r['n_sne']:5d}  "
                f"{r['H0']:8.2f}  {r['err_low']:8.2f}  {r['err_high']:8.2f}  "
                f"{r['chi2']:8.2f}"
            )
        else:
            print(
                f"  {r['sector_id']:6d}  {r['n_sne']:5d}  {'--':>8s}  {'--':>8s}  {'--':>8s}  {'--':>8s}"
            )

    print("-" * 72)

    valid = [r for r in results if r["n_sne"] > 0 and not np.isnan(r["H0"])]
    if valid:
        h0_vals = np.array([r["H0"] for r in valid])
        h0_mean = np.mean(h0_vals)
        h0_std = np.std(h0_vals, ddof=1)
        h0_max = np.max(h0_vals)
        h0_min = np.min(h0_vals)
        epsilon = (h0_max - h0_min) / (2 * h0_mean)

        print(f"\n  Mean H0:           {h0_mean:.2f} km/s/Mpc")
        print(f"  Std dev (sectors): {h0_std:.2f} km/s/Mpc")
        print(f"  Max - Min:         {h0_max - h0_min:.2f} km/s/Mpc")
        print(f"  Modulation ε:      {epsilon:.4f}  ({epsilon*100:.1f}%)")
        print("=" * 72)
