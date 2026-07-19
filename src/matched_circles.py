#!/usr/bin/env python3
"""
Matched Circles Search in CMB for Dodecahedral Topology (Poincare Dodecahedral Space).

Searches for pairs of matched circles around the six antipodal face-axis pairs
of a regular dodecahedron in a fixed fiducial (golden-ratio) orientation,
expressed in Galactic coordinates -- the native frame of the Planck SMICA map
and of the |b| >= 20 deg Galactic mask.  The orientation of the dodecahedron
is NOT marginalised over.

If the Planck SMICA map is unavailable, synthetic maps are generated from a
scale-invariant Sachs-Wolfe-level spectrum (see ``sachs_wolfe_cls``).  The
significance is estimated against isotropic simulations drawn from the same
spectrum.  Default runtime parameters are NSIDE=128 and N_SIM=50 (raised for
the publication run; see the constants in ``__main__``).

References:
  - Cornish+2004, PRL 92, 201302
  - Roukema+2008, A&A 486, 55
  - Planck 2018 I, A&A 641, A1
  - Planck 2015 XVIII, A&A 594, A18
"""

import sys
import os
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dodecahedron import get_dodecahedron_normals

warnings.filterwarnings("ignore")

try:
    import healpy as hp
except ImportError:
    hp = None

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

TWIST_PDS = np.degrees(np.pi / 5)
N_CIRCLE_POINTS = 360


def get_dodecahedron_axes() -> np.ndarray:
    """
    Pair the 12 dodecahedron face normals into 6 antipodal axis pairs.

    The normals come from ``get_dodecahedron_normals`` in a fixed fiducial
    golden-ratio orientation and are interpreted as unit vectors in the
    Galactic frame (the native frame of the SMICA map).  The orientation is
    fiducial and is not marginalised over.

    Returns
    -------
    np.ndarray
        Array of shape (6, 2, 3): for each pair, the two antipodal unit
        normal vectors.
    """
    normals = get_dodecahedron_normals()
    n_pairs = len(normals) // 2
    used = np.zeros(len(normals), dtype=bool)
    axes = np.zeros((n_pairs, 2, 3))
    pair_idx = 0
    for i in range(len(normals)):
        if used[i]:
            continue
        for j in range(i + 1, len(normals)):
            if used[j]:
                continue
            if np.allclose(normals[i], -normals[j], atol=1e-10):
                axes[pair_idx, 0] = normals[i]
                axes[pair_idx, 1] = normals[j]
                used[i] = used[j] = True
                pair_idx += 1
                break
    return axes


def cartesian_to_lonlat(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian unit vectors to Galactic longitude/latitude in degrees.

    Parameters
    ----------
    vecs : np.ndarray
        Unit vector(s) of shape (3,) or (N, 3) in the Galactic frame.

    Returns
    -------
    lon : np.ndarray
        Galactic longitude l in degrees, in [0, 360).
    lat : np.ndarray
        Galactic latitude b in degrees, in [-90, 90].
    """
    vecs = np.asarray(vecs)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, 3)
    lat = np.degrees(np.arcsin(np.clip(vecs[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    lon = np.mod(lon, 360)
    return lon, lat


def sachs_wolfe_cls(lmax: int) -> np.ndarray:
    """
    Scale-invariant Sachs-Wolfe-level TT power spectrum.

    Returns C_l proportional to A_s * 2*pi / (l*(l+1)) with primordial
    amplitude A_s = 2.101e-9 (Planck 2018 value).  No transfer function is
    applied, so the acoustic-peak structure of the full LCDM spectrum is
    absent; this is a flat (scale-invariant) Sachs-Wolfe plateau only.

    Parameters
    ----------
    lmax : int
        Maximum multipole.

    Returns
    -------
    np.ndarray
        C_l array of length lmax + 1 (monopole and dipole set to zero).
    """
    ell = np.arange(lmax + 1)
    cls = np.zeros(lmax + 1)
    cls[2:] = 2.101e-9 * (2 * np.pi) / (ell[2:] * (ell[2:] + 1))
    return cls


def load_or_generate_cmb_map(
    nside: int = 128, random_seed: int = 42, data_path: str | None = None
) -> np.ndarray:
    """
    Load the real Planck SMICA map or generate a synthetic Sachs-Wolfe map.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the returned map.
    random_seed : int
        Random seed for the synthetic realization.
    data_path : str or None
        Path to the Planck FITS file (Galactic coordinates).  If None or
        missing, a synthetic map from ``sachs_wolfe_cls`` is generated.

    Returns
    -------
    np.ndarray
        CMB temperature map (RING ordering).
    """
    if data_path is not None and os.path.exists(data_path):
        try:
            cmb_map = hp.read_map(data_path, field=0, dtype=np.float64)
            current_nside = hp.get_nside(cmb_map)
            if current_nside != nside:
                cmb_map = hp.ud_grade(cmb_map, nside_out=nside)
            return cmb_map
        except Exception as e:
            print(f"      Warning: Could not load Planck map: {e}")
            print("      Falling back to synthetic map...")

    lmax = 3 * nside
    cls = sachs_wolfe_cls(lmax)
    np.random.seed(random_seed)
    cmb_map = hp.synfast(cls, nside, lmax=lmax)
    return cmb_map


def apply_cmb_mask(
    cmb_map: np.ndarray, gal_cut: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a Galactic-latitude cut |b| >= gal_cut to the map.

    Parameters
    ----------
    cmb_map : np.ndarray
        HEALPix temperature map in Galactic coordinates.
    gal_cut : float
        Galactic latitude cut in degrees.

    Returns
    -------
    masked_map : np.ndarray
        Copy of the map with masked pixels set to ``hp.UNSEEN``.
    mask : np.ndarray
        Boolean array, True for valid (unmasked) pixels.
    """
    nside = hp.get_nside(cmb_map)
    npix = len(cmb_map)
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    lat = np.asarray(lat)
    mask = np.abs(lat) >= gal_cut
    masked_map = cmb_map.copy()
    masked_map[~mask] = hp.UNSEEN
    return masked_map, mask


def sample_circle(
    axis: np.ndarray, alpha: float, n_points: int = N_CIRCLE_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample points on a circle of angular radius alpha around an axis.

    Parameters
    ----------
    axis : np.ndarray
        Unit vector (Galactic frame) defining the circle center.
    alpha : float
        Angular radius of the circle in degrees.
    n_points : int
        Number of equally spaced sample points along the circle.

    Returns
    -------
    theta : np.ndarray
        Colatitudes of the sample points in radians.
    phi : np.ndarray
        Longitudes of the sample points in radians, in [0, 2*pi).
    """
    alpha_rad = np.radians(alpha)
    axis = np.asarray(axis) / np.linalg.norm(axis)
    perp1 = np.array([-axis[1], axis[0], 0.0])
    if np.linalg.norm(perp1) < 1e-10:
        perp1 = np.array([1.0, 0.0, 0.0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    phis = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle_pts = np.cos(alpha_rad) * axis[np.newaxis, :] + np.sin(alpha_rad) * (
        np.cos(phis)[:, np.newaxis] * perp1[np.newaxis, :]
        + np.sin(phis)[:, np.newaxis] * perp2[np.newaxis, :]
    )
    norms = np.linalg.norm(circle_pts, axis=1)
    circle_pts = circle_pts / norms[:, np.newaxis]
    theta = np.arccos(np.clip(circle_pts[:, 2], -1, 1))
    phi = np.arctan2(circle_pts[:, 1], circle_pts[:, 0])
    phi = np.mod(phi, 2 * np.pi)
    return theta, phi


def circle_correlation(
    cmb_map: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    alpha: float,
    twist_deg: float,
    nside: int,
    n_points: int = N_CIRCLE_POINTS,
    mask: np.ndarray | None = None,
) -> float:
    """
    Pearson correlation between two circles at a given twist angle.

    Parameters
    ----------
    cmb_map : np.ndarray
        HEALPix temperature map.
    axis1, axis2 : np.ndarray
        Unit vectors (Galactic frame) of the two circle centers.
    alpha : float
        Angular radius of both circles in degrees.
    twist_deg : float
        Relative twist (phase shift) between the circles in degrees.
    nside : int
        HEALPix resolution of the map.
    n_points : int
        Number of sample points per circle.
    mask : np.ndarray or None
        Boolean validity mask; pixels where it is False are excluded.

    Returns
    -------
    float
        Pearson correlation coefficient (0.0 if fewer than 10 valid points).
    """
    theta1, phi1 = sample_circle(axis1, alpha, n_points)
    theta2, phi2 = sample_circle(axis2, alpha, n_points)
    T1 = hp.get_interp_val(cmb_map, theta1, phi1, nest=False)
    T2 = hp.get_interp_val(cmb_map, theta2, phi2, nest=False)
    shift = int(round(twist_deg / 360.0 * n_points)) % n_points
    T2_shifted = np.roll(T2, shift)
    if mask is not None:
        pix1 = hp.ang2pix(nside, theta1, phi1, nest=False)
        pix2 = hp.ang2pix(nside, theta2, phi2, nest=False)
        pix2_shifted = np.roll(pix2, shift)
        valid = mask[pix1] & mask[pix2_shifted]
        if valid.sum() < 10:
            return 0.0
        T1 = T1[valid]
        T2_shifted = T2_shifted[valid]
    common = np.isfinite(T1) & np.isfinite(T2_shifted)
    if common.sum() < 10:
        return 0.0
    corr, _ = pearsonr(T1[common], T2_shifted[common])
    return corr if np.isfinite(corr) else 0.0


def search_circles_for_axis(
    cmb_map: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    alpha_range: np.ndarray,
    nside: int,
    n_points: int = N_CIRCLE_POINTS,
    mask: np.ndarray | None = None,
    twist_step: float = 1.0,
) -> dict:
    """
    Scan circle radius alpha and twist angle for one antipodal axis pair.

    Parameters
    ----------
    cmb_map : np.ndarray
        HEALPix temperature map.
    axis1, axis2 : np.ndarray
        Unit vectors of the antipodal axis pair (Galactic frame).
    alpha_range : np.ndarray
        Circle radii to scan, in degrees.
    nside : int
        HEALPix resolution of the map.
    n_points : int
        Number of sample points per circle.
    mask : np.ndarray or None
        Boolean validity mask.
    twist_step : float
        Step of the twist-angle grid in degrees (0..360).

    Returns
    -------
    dict
        Keys: 'alpha_best', 'twist_best', 'corr_best', 'alpha_scan',
        'twist_scan', 'corr_scan' (2D array over alpha x twist).
    """
    twists = np.arange(0, 360, twist_step)
    best_corr = -1.0
    best_alpha = np.nan
    best_twist = np.nan
    corr_scan = np.zeros((len(alpha_range), len(twists)))
    for i, alpha in enumerate(tqdm(alpha_range, desc="  α scan", leave=False)):
        for j, twist in enumerate(twists):
            corr = circle_correlation(
                cmb_map, axis1, axis2, alpha, twist, nside, n_points, mask
            )
            corr_scan[i, j] = corr
            if corr > best_corr:
                best_corr = corr
                best_alpha = alpha
                best_twist = twist
    return {
        "alpha_best": best_alpha,
        "twist_best": best_twist,
        "corr_best": best_corr,
        "alpha_scan": np.asarray(alpha_range),
        "twist_scan": twists,
        "corr_scan": corr_scan,
    }


def search_all_axes(
    cmb_map: np.ndarray,
    axes: np.ndarray,
    alpha_range: np.ndarray,
    nside: int,
    n_points: int = N_CIRCLE_POINTS,
    mask: np.ndarray | None = None,
    twist_step: float = 1.0,
) -> list[dict]:
    """
    Run the matched-circle scan for all six antipodal axis pairs.

    Parameters
    ----------
    cmb_map : np.ndarray
        HEALPix temperature map.
    axes : np.ndarray
        Axis pairs of shape (6, 2, 3) from ``get_dodecahedron_axes``.
    alpha_range : np.ndarray
        Circle radii to scan, in degrees.
    nside : int
        HEALPix resolution of the map.
    n_points : int
        Number of sample points per circle.
    mask : np.ndarray or None
        Boolean validity mask.
    twist_step : float
        Step of the twist-angle grid in degrees.

    Returns
    -------
    list of dict
        Per-pair results from ``search_circles_for_axis``, each augmented
        with 'pair_idx' and the Galactic coordinates 'lon1', 'lat1',
        'lon2', 'lat2' of the two circle centers (degrees).
    """
    results = []
    for pair_idx in range(len(axes)):
        axis1 = axes[pair_idx, 0]
        axis2 = axes[pair_idx, 1]
        lon1, lat1 = cartesian_to_lonlat(axis1)
        lon2, lat2 = cartesian_to_lonlat(axis2)
        print(
            f"\n  Axis pair {pair_idx + 1}/6: "
            f"center1=(l={lon1[0]:.1f}°, b={lat1[0]:.1f}°), "
            f"center2=(l={lon2[0]:.1f}°, b={lat2[0]:.1f}°)"
        )
        res = search_circles_for_axis(
            cmb_map, axis1, axis2, alpha_range, nside, n_points, mask, twist_step
        )
        res["pair_idx"] = pair_idx
        res["lon1"] = lon1[0]
        res["lat1"] = lat1[0]
        res["lon2"] = lon2[0]
        res["lat2"] = lat2[0]
        results.append(res)
        print(
            f"    Best: α={res['alpha_best']:.1f}°, "
            f"φ={res['twist_best']:.1f}°, r={res['corr_best']:.4f}"
        )
    return results


def run_simulations(
    n_sim: int,
    nside: int,
    axes: np.ndarray,
    alpha_range: np.ndarray,
    mask: np.ndarray | None = None,
    n_points: int = N_CIRCLE_POINTS,
    twist_step: float = 1.0,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Null distribution of the maximum matched-circle correlation.

    Generates isotropic synthetic maps (``sachs_wolfe_cls`` spectrum) and
    repeats the full scan on each.  The twist grid must be identical to the
    one used on the data to avoid biasing the p-value.

    Parameters
    ----------
    n_sim : int
        Number of isotropic simulations.
    nside : int
        HEALPix resolution of the simulated maps.
    axes : np.ndarray
        Axis pairs of shape (6, 2, 3).
    alpha_range : np.ndarray
        Circle radii to scan, in degrees.
    mask : np.ndarray or None
        Boolean validity mask applied to each simulation.
    n_points : int
        Number of sample points per circle.
    twist_step : float
        Step of the twist-angle grid in degrees (same as for the data).
    random_seed : int
        Base random seed; simulation i uses random_seed + i.

    Returns
    -------
    np.ndarray
        Maximum correlation over all pairs for each simulation, length n_sim.
    """
    max_corrs = np.zeros(n_sim)
    for i in tqdm(range(n_sim), desc="Simulations"):
        sim_map = load_or_generate_cmb_map(nside=nside, random_seed=random_seed + i)
        if mask is not None:
            sim_map[~mask] = hp.UNSEEN
        sim_best = -1.0
        for pair_idx in range(len(axes)):
            res = search_circles_for_axis(
                sim_map,
                axes[pair_idx, 0],
                axes[pair_idx, 1],
                alpha_range,
                nside,
                n_points,
                mask,
                twist_step,
            )
            if res["corr_best"] > sim_best:
                sim_best = res["corr_best"]
        max_corrs[i] = sim_best
    return max_corrs


def compute_significance(observed_corr: float, sim_corrs: np.ndarray) -> dict:
    """
    Empirical p-value and z-score of the observed maximum correlation.

    Parameters
    ----------
    observed_corr : float
        Maximum matched-circle correlation observed in the data.
    sim_corrs : np.ndarray
        Null distribution of maximum correlations from simulations.

    Returns
    -------
    dict
        Keys: 'p_value', 'z_score', 'sim_mean', 'sim_std', 'n_sim'.
    """
    n_sim = len(sim_corrs)
    sim_mean = np.mean(sim_corrs)
    sim_std = np.std(sim_corrs, ddof=1)
    p_value = np.sum(sim_corrs >= observed_corr) / n_sim
    z_score = (observed_corr - sim_mean) / sim_std if sim_std > 0 else 0.0
    return {
        "p_value": p_value,
        "z_score": z_score,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "n_sim": n_sim,
    }


def plot_correlation_vs_alpha(all_results: list[dict], output_path: str) -> None:
    """
    Plot maximum correlation versus circle radius for all axis pairs.

    Parameters
    ----------
    all_results : list of dict
        Per-pair results from ``search_all_axes``.
    output_path : str
        Base path for output figures (suffixes are appended).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for i, res in enumerate(all_results):
        max_corr_per_alpha = res["corr_scan"].max(axis=1)
        ax.plot(
            res["alpha_scan"],
            max_corr_per_alpha,
            "o-",
            color=colors[i],
            linewidth=1.2,
            markersize=3,
            label=f"Pair {i+1}",
        )
    ax.axvline(x=29, color="gray", linestyle="--", alpha=0.5, label="PDS α range")
    ax.axvline(x=37, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"Circle radius $\alpha$ [deg]", fontsize=12)
    ax.set_ylabel("Max Pearson r", fontsize=12)
    ax.set_title("Matched Circles: Correlation vs Radius", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_corr_vs_alpha.{fmt}")
    plt.close(fig)


def plot_correlation_vs_twist(best_result: dict, output_path: str) -> None:
    """
    Plot correlation versus twist angle at the best-fit circle radius.

    Parameters
    ----------
    best_result : dict
        Result dict of the best axis pair from ``search_all_axes``.
    output_path : str
        Base path for output figures (suffixes are appended).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    best_alpha_idx = np.argmin(
        np.abs(best_result["alpha_scan"] - best_result["alpha_best"])
    )
    corr_vs_twist = best_result["corr_scan"][best_alpha_idx, :]
    ax.plot(
        best_result["twist_scan"], corr_vs_twist, "-", color="steelblue", linewidth=1.5
    )
    ax.axvline(
        x=TWIST_PDS,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"PDS twist: {TWIST_PDS:.0f}°",
    )
    ax.axvline(
        x=best_result["twist_best"],
        color="darkorange",
        linestyle="-",
        linewidth=1.5,
        alpha=0.7,
        label=f'Best: {best_result["twist_best"]:.0f}°',
    )
    ax.set_xlabel(r"Twist angle $\phi$ [deg]", fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title(
        f"Correlation vs Twist (α = {best_result['alpha_best']:.1f}°)", fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_corr_vs_twist.{fmt}")
    plt.close(fig)


def plot_sim_distribution(
    sim_corrs: np.ndarray, observed_corr: float, p_value: float, output_path: str
) -> None:
    """
    Plot the null distribution of maximum correlations with the observed value.

    Parameters
    ----------
    sim_corrs : np.ndarray
        Null distribution from ``run_simulations``.
    observed_corr : float
        Observed maximum correlation.
    p_value : float
        Empirical p-value.
    output_path : str
        Base path for output figures (suffixes are appended).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(
        min(sim_corrs.min(), observed_corr) * 0.9,
        max(sim_corrs.max(), observed_corr) * 1.1,
        30,
    )
    ax.hist(
        sim_corrs,
        bins=bins,
        color="steelblue",
        edgecolor="white",
        alpha=0.7,
        label=f"Simulations (n={len(sim_corrs)})",
    )
    ax.axvline(
        x=observed_corr,
        color="red",
        linewidth=2.5,
        linestyle="-",
        label=f"Observed: {observed_corr:.4f}",
    )
    ax.axvline(
        x=np.mean(sim_corrs),
        color="gray",
        linewidth=1.5,
        linestyle="--",
        label=f"Sim mean: {np.mean(sim_corrs):.4f}",
    )
    ax.text(
        0.97,
        0.95,
        f"p = {p_value:.4f}\nN_sim = {len(sim_corrs)}",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("Max Pearson r", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Null Distribution: Max Matched-Circle Correlation", fontsize=13)
    ax.legend(fontsize=10)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_sim_dist.{fmt}")
    plt.close(fig)


def plot_cmb_with_circles(
    cmb_map: np.ndarray,
    best_result: dict,
    axes: np.ndarray,
    best_pair_idx: int,
    output_path: str,
) -> None:
    """
    Mollweide view of the CMB map with the best matched circle pair overlaid.

    The color scale is data-driven (+/- 3 standard deviations of the valid
    pixels), so it works for both real SMICA maps (~1e-4 K) and synthetic
    maps of arbitrary amplitude.

    Parameters
    ----------
    cmb_map : np.ndarray
        HEALPix temperature map in Galactic coordinates (may contain
        ``hp.UNSEEN``).
    best_result : dict
        Result dict of the best axis pair.
    axes : np.ndarray
        Axis pairs of shape (6, 2, 3).
    best_pair_idx : int
        Index of the best axis pair.
    output_path : str
        Base path for output figures (suffixes are appended).
    """
    alpha = best_result["alpha_best"]
    axis1 = axes[best_pair_idx, 0]
    axis2 = axes[best_pair_idx, 1]
    theta1, phi1 = sample_circle(axis1, alpha, N_CIRCLE_POINTS)
    theta2, phi2 = sample_circle(axis2, alpha, N_CIRCLE_POINTS)
    lon1 = np.degrees(phi1)
    lat1 = np.degrees(np.pi / 2 - theta1)
    lon2 = np.degrees(phi2)
    lat2 = np.degrees(np.pi / 2 - theta2)
    lon1_m = lon1 - 180.0
    lon1_m[lon1_m > 180] -= 360.0
    lon2_m = lon2 - 180.0
    lon2_m[lon2_m > 180] -= 360.0
    valid = (cmb_map != hp.UNSEEN) & np.isfinite(cmb_map)
    vlim = 3.0 * np.std(cmb_map[valid]) if valid.any() else 1.0
    fig = plt.figure(figsize=(14, 7))
    hp.mollview(
        cmb_map,
        title="CMB with Best Matched Circles",
        hold=True,
        unit=r"$\Delta T$ [K]",
        cmap="RdBu_r",
        min=-vlim,
        max=vlim,
        fig=1,
        cbar=True,
    )
    ax = fig.axes[0]
    ax.plot(
        np.radians(lon1_m),
        np.radians(lat1),
        "-",
        color="lime",
        linewidth=1.5,
        alpha=0.9,
        label=f"Circle 1 (α={alpha:.1f}°)",
    )
    ax.plot(
        np.radians(lon2_m),
        np.radians(lat2),
        "--",
        color="cyan",
        linewidth=1.5,
        alpha=0.9,
        label=f"Circle 2 (α={alpha:.1f}°)",
    )
    ax.legend(loc="lower right", fontsize=8)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_cmb_circles.{fmt}")
    plt.close(fig)


if __name__ == "__main__":
    if hp is None:
        print("ERROR: healpy is required. Install with: pip install healpy")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Fast defaults for the supplementary run; for the publication run these
    # are raised (higher NSIDE, more simulations, finer alpha/twist grids).
    NSIDE = 128
    N_SIM = 50
    ALPHA_RANGE = np.arange(10, 51, 5)
    TWIST_STEP = 5.0  # same twist grid for data and simulations (unbiased p-value)
    GAL_CUT = 20.0

    planck_path = os.path.join(
        base_dir, "..", "data", "COM_CMB_IQU-smica_2048_R3.00_full.fits"
    )

    print("=" * 72)
    print("  MATCHED CIRCLES SEARCH IN CMB")
    print("  Poincaré Dodecahedral Space Topology Test")
    print("=" * 72)
    print(f"  Nside = {NSIDE}  (Npix = {12 * NSIDE**2:,})")
    print(f"  N_sim = {N_SIM}")
    print(
        f"  α range: {ALPHA_RANGE[0]}°–{ALPHA_RANGE[-1]}° (step {ALPHA_RANGE[1]-ALPHA_RANGE[0]:.0f}°)"
    )
    print(f"  Twist step (data & sims): {TWIST_STEP}°")
    print(f"  PDS prediction: α ≈ 29°–37°, φ ≈ {TWIST_PDS:.0f}°")

    print("\n[1/6] Loading CMB map...")
    cmb_map = load_or_generate_cmb_map(
        nside=NSIDE, random_seed=42, data_path=planck_path
    )
    cmb_masked, mask = apply_cmb_mask(cmb_map, gal_cut=GAL_CUT)
    n_valid = mask.sum()
    print(f"      Map size: {len(cmb_map):,} pixels")
    print(
        f"      Valid pixels (|b| ≥ {GAL_CUT}°): {n_valid:,} ({n_valid/len(cmb_map)*100:.1f}%)"
    )

    print(
        "\n[2/6] Computing dodecahedron axis pairs (fixed fiducial Galactic orientation)..."
    )
    axes = get_dodecahedron_axes()
    print(f"      Found {len(axes)} antipodal axis pairs")
    for i in range(len(axes)):
        lon1, lat1 = cartesian_to_lonlat(axes[i, 0])
        lon2, lat2 = cartesian_to_lonlat(axes[i, 1])
        print(
            f"      Pair {i+1}: (l={lon1[0]:7.1f}°, b={lat1[0]:7.1f}°) ↔ "
            f"(l={lon2[0]:7.1f}°, b={lat2[0]:7.1f}°)"
        )

    print("\n[3/6] Searching for matched circles...")
    all_results = search_all_axes(
        cmb_masked,
        axes,
        ALPHA_RANGE,
        NSIDE,
        n_points=N_CIRCLE_POINTS,
        mask=mask,
        twist_step=TWIST_STEP,
    )

    best_overall = max(all_results, key=lambda r: r["corr_best"])
    print("\n  Best overall:")
    print(f"    Axis pair: {best_overall['pair_idx'] + 1}")
    print(
        f"    Center 1:  (l={best_overall['lon1']:.1f}°, b={best_overall['lat1']:.1f}°)"
    )
    print(
        f"    Center 2:  (l={best_overall['lon2']:.1f}°, b={best_overall['lat2']:.1f}°)"
    )
    print(f"    α = {best_overall['alpha_best']:.1f}°")
    print(f"    φ = {best_overall['twist_best']:.1f}°")
    print(f"    r = {best_overall['corr_best']:.4f}")

    pds_alpha_ok = 29 <= best_overall["alpha_best"] <= 37
    pds_twist_ok = abs(best_overall["twist_best"] - TWIST_PDS) <= 10
    print(f"    PDS α compatible: {pds_alpha_ok}")
    print(f"    PDS φ compatible: {pds_twist_ok}")

    print(f"\n[4/6] Running {N_SIM} isotropic simulations...")
    sim_corrs = run_simulations(
        N_SIM,
        NSIDE,
        axes,
        ALPHA_RANGE,
        mask=mask,
        n_points=N_CIRCLE_POINTS,
        twist_step=TWIST_STEP,
        random_seed=100,
    )

    print("\n[5/6] Computing significance...")
    sig = compute_significance(best_overall["corr_best"], sim_corrs)
    print(f"      Observed max r:  {best_overall['corr_best']:.4f}")
    print(f"      Sim mean max r:  {sig['sim_mean']:.4f}")
    print(f"      Sim std max r:   {sig['sim_std']:.4f}")
    print(f"      Z-score:         {sig['z_score']:.2f}σ")
    print(f"      P-value:         {sig['p_value']:.4f}")

    if sig["p_value"] < 0.01:
        verdict = "SIGNIFICANT — possible topology detection"
    elif sig["p_value"] < 0.05:
        verdict = "MARGINALLY SIGNIFICANT"
    else:
        verdict = "NOT SIGNIFICANT — consistent with isotropic ΛCDM"
    print(f"      Verdict: {verdict}")

    print("\n[6/6] Generating figures...")
    fig_base = os.path.join(output_dir, "fig")

    plot_correlation_vs_alpha(all_results, fig_base)
    print(f"  Saved: {fig_base}_corr_vs_alpha.[pdf,png]")

    plot_correlation_vs_twist(best_overall, fig_base)
    print(f"  Saved: {fig_base}_corr_vs_twist.[pdf,png]")

    plot_sim_distribution(
        sim_corrs, best_overall["corr_best"], sig["p_value"], fig_base
    )
    print(f"  Saved: {fig_base}_sim_dist.[pdf,png]")

    plot_cmb_with_circles(
        cmb_masked, best_overall, axes, best_overall["pair_idx"], fig_base
    )
    print(f"  Saved: {fig_base}_cmb_circles.[pdf,png]")

    np.savez(
        os.path.join(output_dir, "matched_circles.npz"),
        best_alpha=best_overall["alpha_best"],
        best_twist=best_overall["twist_best"],
        best_corr=best_overall["corr_best"],
        best_pair=best_overall["pair_idx"],
        sim_corrs=sim_corrs,
        p_value=sig["p_value"],
        z_score=sig["z_score"],
        pds_alpha_ok=pds_alpha_ok,
        pds_twist_ok=pds_twist_ok,
    )
    print(f"\nResults saved to: {os.path.join(output_dir, 'matched_circles.npz')}")

    print(f"\n{'='*72}")
    print("  COMPARISON WITH LITERATURE")
    print(f"{'='*72}")
    print("  Cornish+2004:    No circles >25° found")
    print("  Roukema+2008:    Optimal twist +39°±2.5°")
    print("  Aurich+2006:     Hint of right-handed PDS at Ω_tot≈1.015")
    print("  Planck 2015:     Excluded circles >15°")
    print("")
    print(
        f"  This analysis:   α={best_overall['alpha_best']:.1f}°, "
        f"φ={best_overall['twist_best']:.1f}°, "
        f"r={best_overall['corr_best']:.4f}, "
        f"p={sig['p_value']:.4f}"
    )
    print(f"  Nside={NSIDE}, N_sim={N_SIM}")
    print("  Scale-invariant Sachs–Wolfe spectrum used for synthetic maps")
    print(f"{'='*72}")
