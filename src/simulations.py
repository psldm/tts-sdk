#!/usr/bin/env python3
"""
Monte Carlo simulations for dodecahedral H0 anisotropy significance.

Generates mock supernova datasets under the null hypothesis (isotropic H0)
and computes the statistical significance of the observed H0 variation
across dodecahedron sectors.
"""

import sys
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dodecahedron import get_dodecahedron_normals, assign_sectors, load_pantheon_data
from h0_fit import (
    load_covariance_matrix,
    mb_theory,
    fit_all_sectors,
    M_B,
)


def _regularize_covariance(cov: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Ensure covariance matrix is symmetric positive-semidefinite.

    Adds a small diagonal term and symmetrizes the matrix.

    Parameters
    ----------
    cov : numpy.ndarray
        Input covariance matrix.
    epsilon : float
        Regularization strength.

    Returns
    -------
    numpy.ndarray
        Regularized covariance matrix.
    """
    cov = (cov + cov.T) / 2.0
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = np.min(eigvals)
    if min_eig < 0:
        cov = cov + (-min_eig + epsilon) * np.eye(len(cov))
    return cov


def generate_mock_data(
    df: pd.DataFrame,
    cov_sub: np.ndarray,
    H0_true: float = 70.0,
    MB: float = M_B,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a mock supernova dataset under the isotropic null hypothesis.

    Computes theoretical apparent magnitudes from H0_true and adds
    multivariate Gaussian noise drawn from the covariance matrix. Sky
    positions are kept fixed at their observed values.

    Parameters
    ----------
    df : pandas.DataFrame
        Real data with columns ['z', 'ra', 'dec'].
    cov_sub : numpy.ndarray
        Covariance matrix for the SNe in df (N x N).
    H0_true : float
        True Hubble constant of the isotropic mock universe.
    MB : float
        Absolute magnitude of SNe Ia.
    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Copy of df with an added 'mb_mock' column.
    """
    rng = np.random.default_rng(random_seed)

    z = df["z"].values
    mb_true = mb_theory(z, H0_true, MB=MB)

    n_sne = len(z)
    if cov_sub.shape != (n_sne, n_sne):
        raise ValueError(
            f"Covariance shape {cov_sub.shape} does not match " f"number of SNe {n_sne}"
        )

    cov_reg = _regularize_covariance(cov_sub)
    noise = rng.multivariate_normal(np.zeros(n_sne), cov_reg)

    df_mock = df[["z", "ra", "dec"]].copy()
    df_mock["mb_mock"] = mb_true + noise

    return df_mock


def run_single_mock(
    df: pd.DataFrame,
    cov_sub: np.ndarray,
    normals: np.ndarray,
    H0_true: float = 70.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Run a single mock realization: generate data, assign sectors, fit H0.

    Parameters
    ----------
    df : pandas.DataFrame
        Real data with columns ['z', 'ra', 'dec'].
    cov_sub : numpy.ndarray
        Covariance submatrix for the SNe in df.
    normals : numpy.ndarray
        Dodecahedron face normals (12, 3).
    H0_true : float
        True H0 for the isotropic mock.
    seed : int or None
        Random seed for this mock.

    Returns
    -------
    dict
        Keys 'h0_per_sector' (list of 12 floats), 'max_H0', 'min_H0',
        'delta_H0', 'seed'.
    """
    df_mock = generate_mock_data(df, cov_sub, H0_true=H0_true, random_seed=seed)

    sector_ids = assign_sectors(df_mock["ra"].values, df_mock["dec"].values, normals)

    df_fit = df_mock.rename(columns={"mb_mock": "mb"})
    results = fit_all_sectors(df_fit, sector_ids, cov_sub)

    h0_per_sector = [r["H0"] for r in results]
    valid_h0 = [h for h in h0_per_sector if not np.isnan(h)]

    if len(valid_h0) < 2:
        return {
            "h0_per_sector": h0_per_sector,
            "max_H0": np.nan,
            "min_H0": np.nan,
            "delta_H0": np.nan,
            "seed": seed,
        }

    return {
        "h0_per_sector": h0_per_sector,
        "max_H0": float(np.max(valid_h0)),
        "min_H0": float(np.min(valid_h0)),
        "delta_H0": float(np.max(valid_h0) - np.min(valid_h0)),
        "seed": seed,
    }


def run_monte_carlo(
    df: pd.DataFrame,
    cov_sub: np.ndarray,
    normals: np.ndarray,
    n_mocks: int = 10000,
    H0_true: float = 70.0,
    n_jobs: int = -1,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Run Monte Carlo simulations to build the null distribution of delta_H0.

    Parameters
    ----------
    df : pandas.DataFrame
        Real data with columns ['z', 'ra', 'dec'].
    cov_sub : numpy.ndarray
        Covariance submatrix for the SNe in df.
    normals : numpy.ndarray
        Dodecahedron face normals (12, 3).
    n_mocks : int
        Number of mock realizations.
    H0_true : float
        True H0 for the isotropic null hypothesis.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    random_seed : int or None
        Base seed for reproducibility.

    Returns
    -------
    dict
        Keys 'delta_H0', 'max_H0', 'min_H0' (arrays of length n_valid),
        'h0_per_sector' (array of shape n_valid x 12), 'n_mocks',
        'n_valid', 'H0_true'.
    """
    base_rng = np.random.default_rng(random_seed)
    seeds = base_rng.integers(0, 2**31 - 1, size=n_mocks)

    if n_jobs == 1:
        results = []
        for i in tqdm(range(n_mocks), desc="Monte Carlo"):
            results.append(
                run_single_mock(
                    df, cov_sub, normals, H0_true=H0_true, seed=int(seeds[i])
                )
            )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_mock)(
                df, cov_sub, normals, H0_true=H0_true, seed=int(seeds[i])
            )
            for i in tqdm(range(n_mocks), desc="Monte Carlo")
        )

    n_valid = sum(1 for r in results if not np.isnan(r["delta_H0"]))
    if n_valid < n_mocks:
        warnings.warn(
            f"{n_mocks - n_valid}/{n_mocks} mocks produced NaN delta_H0 "
            f"(likely empty sectors). These are excluded from statistics."
        )

    delta_H0 = np.array([r["delta_H0"] for r in results if not np.isnan(r["delta_H0"])])
    max_H0 = np.array([r["max_H0"] for r in results if not np.isnan(r["delta_H0"])])
    min_H0 = np.array([r["min_H0"] for r in results if not np.isnan(r["delta_H0"])])

    h0_per_sector = np.array(
        [r["h0_per_sector"] for r in results if not np.isnan(r["delta_H0"])]
    )

    return {
        "delta_H0": delta_H0,
        "max_H0": max_H0,
        "min_H0": min_H0,
        "h0_per_sector": h0_per_sector,
        "n_mocks": n_mocks,
        "n_valid": n_valid,
        "H0_true": H0_true,
    }


def compute_pvalue(observed_delta: float, mock_deltas: np.ndarray) -> dict[str, Any]:
    """
    Compute p-value and z-score for the observed delta_H0.

    The p-value is the empirical tail fraction sum(mock >= obs) / N. With
    N = 1000 mocks its resolution is limited to ~0.001 and the standard
    error at p ~ 0.2 is ~0.013.

    Parameters
    ----------
    observed_delta : float
        Observed max(H0) - min(H0) across sectors.
    mock_deltas : numpy.ndarray
        Array of delta_H0 from Monte Carlo mocks.

    Returns
    -------
    dict
        Keys 'p_value', 'p_value_two_sided', 'z_score', 'mock_mean',
        'mock_std', 'n_mocks'.
    """
    n_mocks = len(mock_deltas)
    mock_mean = np.mean(mock_deltas)
    mock_std = np.std(mock_deltas, ddof=1)

    p_value = np.sum(mock_deltas >= observed_delta) / n_mocks

    deviation = np.abs(mock_deltas - mock_mean)
    observed_deviation = np.abs(observed_delta - mock_mean)
    p_value_two_sided = np.sum(deviation >= observed_deviation) / n_mocks

    z_score = (observed_delta - mock_mean) / mock_std if mock_std > 0 else 0.0

    return {
        "p_value": p_value,
        "p_value_two_sided": p_value_two_sided,
        "z_score": z_score,
        "mock_mean": mock_mean,
        "mock_std": mock_std,
        "n_mocks": n_mocks,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "Pantheon+SH0ES.dat")
    cov_path = os.path.join(base_dir, "..", "data", "Pantheon+SH0ES_STAT+SYS.cov")
    output_dir = os.path.join(base_dir, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  MONTE CARLO SIGNIFICANCE TEST")
    print("  H0 Anisotropy Across Dodecahedron Sectors")
    print("=" * 72)

    print("\n[1/5] Loading Pantheon+ data...")
    df = load_pantheon_data(data_path)
    print(f"      Loaded {len(df)} SNe with z < 0.1")

    print("\n[2/5] Loading covariance matrix and extracting submatrix...")
    cov_full = load_covariance_matrix(cov_path)
    if cov_full is not None:
        df_full = pd.read_csv(data_path, sep=r"\s+")
        mask_z = df_full["zHD"] < 0.1
        orig_indices = np.where(mask_z)[0]
        cov_sub = cov_full[np.ix_(orig_indices, orig_indices)]
        print(f"      Full covariance: {cov_full.shape}")
        print(f"      Sub-covariance:  {cov_sub.shape}")
    else:
        cov_sub = np.eye(len(df))
        print(f"      Using identity matrix: {cov_sub.shape}")

    print("\n[3/5] Computing observed delta_H0 from real data...")
    normals = get_dodecahedron_normals()
    sector_ids = assign_sectors(df["ra"].values, df["dec"].values, normals)
    real_results = fit_all_sectors(df, sector_ids, cov_sub)

    valid = [r for r in real_results if r["n_sne"] > 0 and not np.isnan(r["H0"])]
    h0_real = np.array([r["H0"] for r in valid])
    observed_delta = float(np.max(h0_real) - np.min(h0_real))
    observed_max = float(np.max(h0_real))
    observed_min = float(np.min(h0_real))

    print(f"      Observed max(H0) = {observed_max:.2f} km/s/Mpc")
    print(f"      Observed min(H0) = {observed_min:.2f} km/s/Mpc")
    print(f"      Observed delta    = {observed_delta:.2f} km/s/Mpc")

    N_MOCKS = 1000
    print(f"\n[4/5] Running {N_MOCKS} Monte Carlo simulations...")
    print("      H0_true = 70.0 km/s/Mpc (isotropic null)")

    mc_results = run_monte_carlo(
        df,
        cov_sub,
        normals,
        n_mocks=N_MOCKS,
        H0_true=70.0,
        n_jobs=-1,
        random_seed=42,
    )

    print(f"      Valid mocks: {mc_results['n_valid']}/{mc_results['n_mocks']}")

    print("\n[5/5] Computing significance...")
    pval = compute_pvalue(observed_delta, mc_results["delta_H0"])

    print("\n" + "-" * 72)
    print("  RESULTS")
    print("-" * 72)
    print(f"  Observed delta_H0:        {observed_delta:.2f} km/s/Mpc")
    print(f"  Mock mean delta_H0:        {pval['mock_mean']:.2f} km/s/Mpc")
    print(f"  Mock std delta_H0:         {pval['mock_std']:.2f} km/s/Mpc")
    print(f"  Z-score:                   {pval['z_score']:.2f}σ")
    print(f"  P-value (one-sided):       {pval['p_value']:.6f}")
    print(f"  P-value (two-sided):       {pval['p_value_two_sided']:.6f}")
    print("-" * 72)

    if pval["p_value"] < 0.01:
        significance = "SIGNIFICANT"
    elif pval["p_value"] < 0.05:
        significance = "MARGINALLY SIGNIFICANT"
    else:
        significance = "NOT SIGNIFICANT"
    print(f"  Conclusion: {significance} at α = 0.05")
    print("=" * 72)

    output_path = os.path.join(output_dir, "mc_results.npz")
    np.savez(
        output_path,
        observed_delta=observed_delta,
        observed_max=observed_max,
        observed_min=observed_min,
        h0_real=h0_real,
        mock_delta_H0=mc_results["delta_H0"],
        mock_max_H0=mc_results["max_H0"],
        mock_min_H0=mc_results["min_H0"],
        mock_h0_per_sector=mc_results["h0_per_sector"],
        p_value=pval["p_value"],
        p_value_two_sided=pval["p_value_two_sided"],
        z_score=pval["z_score"],
        mock_mean=pval["mock_mean"],
        mock_std=pval["mock_std"],
        n_mocks=N_MOCKS,
        H0_true=70.0,
    )
    print(f"\nResults saved to: {output_path}")
