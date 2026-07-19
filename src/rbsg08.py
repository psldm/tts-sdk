#!/usr/bin/env python3
"""
RBSG08 Spatial Cross-Correlation Method for Poincaré Dodecahedral Space.

Implements the method of Roukema, Buliński, Szaniewska & Gaudin (2008, A&A 486, 55)
for detecting dodecahedral topology via cross-correlation of CMB temperature
fluctuations between copies of the last scattering surface in the covering space S³.

Key test: the twist angle φ should automatically converge to ±36° (π/5) if the
Poincaré dodecahedral space is the correct topology of the Universe.

References:
  RBSG08: Roukema+2008, A&A 486, 55-72
  Luminet+2003, Nature 425, 593
  Aurich+2005, CQG 22, 2061
"""

import sys
import os
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation as R3
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

TWIST_PDS_DEG = 36.0
N_HOLONOMIES = 12
N_BINS = 7
R_BINS = np.linspace(0.5, 4.0, N_BINS + 1)
R_BIN_CENTERS = 0.5 * (R_BINS[:-1] + R_BINS[1:])

# Comoving-horizon scale [Gpc] used to convert S³ angular separations
# (scaled by chi_lss/π) into approximate physical separations in Gpc.
HORIZON_SCALE_GPC = 14.0

# Number of SLS points subsampled for each correlation estimate; controls
# the O(n_sample²) pair count and hence the runtime per grid-search point.
N_SAMPLE_DEFAULT = 30


def get_dodecahedron_axes() -> np.ndarray:
    """
    Group the 12 dodecahedron face normals into 6 antipodal pairs.

    Returns
    -------
    np.ndarray
        Array of shape (6, 2, 3); ``axes[i, 0]`` and ``axes[i, 1]`` are
        antipodal unit face normals.
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


def rotation_matrix_4d(
    plane_angle: float, twist_angle: float, axis_3d: np.ndarray
) -> np.ndarray:
    """
    Build a 4×4 holonomy matrix for the Poincaré dodecahedral space.

    The holonomy maps a point across a dodecahedron face:
    rotation by ``plane_angle`` in the (e₀, axis_3d) plane (crossing the
    face) combined with rotation by ``twist_angle`` in the orthogonal
    2-plane.

    Parameters
    ----------
    plane_angle : float
        Angle in the (e₀, axis) plane in radians (π for face crossing).
    twist_angle : float
        Twist angle in the orthogonal plane, in radians.
    axis_3d : np.ndarray
        Unit vector in R³ along the face normal.

    Returns
    -------
    np.ndarray
        4×4 rotation matrix.
    """
    n = np.asarray(axis_3d) / np.linalg.norm(axis_3d)
    u = np.array([-n[1], n[0], 0.0])
    if np.linalg.norm(u) < 1e-10:
        u = np.array([1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    e0 = np.array([1.0, 0.0, 0.0, 0.0])
    e1 = np.array([0.0, n[0], n[1], n[2]])
    e2 = np.array([0.0, u[0], u[1], u[2]])
    e3 = np.array([0.0, v[0], v[1], v[2]])

    M = np.eye(4)
    cp, sp = np.cos(plane_angle), np.sin(plane_angle)
    ct, st = np.cos(twist_angle), np.sin(twist_angle)

    M[0, 0] = cp
    M[0, 1] = -sp
    M[1, 0] = sp
    M[1, 1] = cp
    M[2, 2] = ct
    M[2, 3] = -st
    M[3, 2] = st
    M[3, 3] = ct

    B = np.column_stack([e0, e1, e2, e3])
    M_full = B @ M @ B.T
    return M_full


def build_holonomies(
    orientation_angles: tuple[float, float, float], twist_deg: float
) -> np.ndarray:
    """
    Build the 12 holonomy matrices for a given orientation and twist.

    Parameters
    ----------
    orientation_angles : tuple of float
        ``(l_deg, b_deg, theta_deg)`` — orientation of the first face
        center in galactic coordinates plus rotation about it, in degrees.
    twist_deg : float
        Twist angle in degrees.

    Returns
    -------
    np.ndarray
        Array of shape (12, 4, 4) with one holonomy per face.
    """
    l_deg, b_deg, theta_deg = orientation_angles
    axes = get_dodecahedron_axes()

    rot_lb = R3.from_euler("zxz", [-np.radians(l_deg), np.radians(90 - b_deg), 0])
    rot_theta = R3.from_euler("z", np.radians(theta_deg))

    holonomies = np.zeros((N_HOLONOMIES, 4, 4))
    for i in range(6):
        n_raw = axes[i, 0].copy()
        n_rot = rot_theta.apply(rot_lb.apply(n_raw))
        n_rot = n_rot / np.linalg.norm(n_rot)
        holonomies[2 * i] = rotation_matrix_4d(np.pi, np.radians(twist_deg), n_rot)
        holonomies[2 * i + 1] = rotation_matrix_4d(np.pi, -np.radians(twist_deg), n_rot)

    return holonomies


def sls_position(
    ra_deg: float | np.ndarray, dec_deg: float | np.ndarray, chi_lss: float
) -> np.ndarray:
    """
    Compute the 4D position of points on the last scattering surface.

    Parameters
    ----------
    ra_deg : float or np.ndarray
        Right ascension in degrees.
    dec_deg : float or np.ndarray
        Declination in degrees.
    chi_lss : float
        Comoving distance to the LSS in radians on S³.

    Returns
    -------
    np.ndarray
        Unit 4-vector of shape (4,) for scalar input, otherwise array of
        shape (N, 4).
    """
    ra_rad = np.radians(np.asarray(ra_deg))
    dec_rad = np.radians(np.asarray(dec_deg))
    n_x = np.cos(dec_rad) * np.cos(ra_rad)
    n_y = np.cos(dec_rad) * np.sin(ra_rad)
    n_z = np.sin(dec_rad)
    scalar = np.isscalar(ra_deg)
    if scalar:
        return np.array(
            [
                np.cos(chi_lss),
                np.sin(chi_lss) * n_x,
                np.sin(chi_lss) * n_y,
                np.sin(chi_lss) * n_z,
            ]
        )
    N = len(np.atleast_1d(ra_deg))
    pos = np.zeros((N, 4))
    pos[:, 0] = np.cos(chi_lss)
    pos[:, 1] = np.sin(chi_lss) * n_x
    pos[:, 2] = np.sin(chi_lss) * n_y
    pos[:, 3] = np.sin(chi_lss) * n_z
    return pos


def s3_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """
    Geodesic distance on S³ between two sets of 4D unit vectors.

    Parameters
    ----------
    pos_a, pos_b : np.ndarray
        Arrays of shape (N, 4) with unit 4-vectors.

    Returns
    -------
    np.ndarray
        Pairwise (row-by-row) geodesic distances in radians on S³.
    """
    dots = np.sum(pos_a * pos_b, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


def generate_cmb_map(nside: int = 64, random_seed: int = 42) -> np.ndarray:
    """
    Generate a synthetic Gaussian CMB temperature map.

    The map is a Gaussian random realization of a scale-invariant angular
    power spectrum, C_ℓ ∝ 1/(ℓ(ℓ+1)), normalized to the Sachs-Wolfe level
    (ℓ(ℓ+1)C_ℓ/2π ≈ 2.1e-10). This is a simplified stand-in for pipeline
    validation, NOT the Planck ΛCDM spectrum.

    Parameters
    ----------
    nside : int, optional
        HEALPix resolution parameter (default 64).
    random_seed : int, optional
        Seed for ``numpy.random.default_rng`` (default 42).

    Returns
    -------
    np.ndarray
        HEALPix temperature map of length ``12 * nside**2``.
    """
    if hp is None:
        raise ImportError("healpy required")
    lmax = 3 * nside
    rng = np.random.default_rng(random_seed)
    cls = np.zeros(lmax + 1)
    ell = np.arange(2, lmax + 1)
    cls[2:] = (2 * np.pi) / (ell * (ell + 1)) * (2.1e-10 / (2 * np.pi))

    # healpy 1.19: synfast has no rng/verbose arguments and draws from the
    # legacy global RandomState, so synthesize the alm explicitly with a
    # seeded Generator: a_l0 ~ N(0, C_l) real; a_lm (m>0) complex with
    # variance C_l/2 per component.
    l_arr, m_arr = hp.Alm.getlm(lmax)
    sigma = np.sqrt(cls[l_arr])
    re = rng.standard_normal(l_arr.size)
    im = rng.standard_normal(l_arr.size)
    alm = (re + 1j * im) * (sigma / np.sqrt(2.0))
    m0 = m_arr == 0
    alm[m0] = re[m0] * sigma[m0]
    return hp.alm2map(alm, nside, lmax=lmax)


def apply_cmb_mask(
    cmb_map: np.ndarray, gal_cut: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a galactic-plane mask to a HEALPix map.

    Parameters
    ----------
    cmb_map : np.ndarray
        Input HEALPix temperature map.
    gal_cut : float, optional
        Half-width of the masked galactic latitude band in degrees
        (default 20).

    Returns
    -------
    masked : np.ndarray
        Copy of the map with masked pixels set to ``hp.UNSEEN``.
    mask : np.ndarray
        Boolean array, True for valid (unmasked) pixels.
    """
    nside = hp.get_nside(cmb_map)
    npix = len(cmb_map)
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    mask = np.abs(np.asarray(lat)) >= gal_cut
    masked = cmb_map.copy()
    masked[~mask] = hp.UNSEEN
    return masked, mask


def remove_monopole_dipole(cmb_map: np.ndarray) -> np.ndarray:
    """
    Remove the monopole (ℓ=0) and dipole (ℓ=1) from a HEALPix map.

    The ℓ≤1 spherical-harmonic template — all coefficients with l in
    {0, 1}, m ≤ l, selected via ``hp.Alm.getidx`` — is estimated with
    ``map2alm`` and subtracted; all ℓ≥2 structure is left untouched.

    Parameters
    ----------
    cmb_map : np.ndarray
        Input HEALPix temperature map (may contain ``hp.UNSEEN`` pixels).

    Returns
    -------
    np.ndarray
        Map with monopole and dipole subtracted.
    """
    nside = hp.get_nside(cmb_map)
    lmax = 3
    alm = hp.map2alm(cmb_map, lmax=lmax, use_weights=True)
    template_alm = np.zeros_like(alm)
    for ell in (0, 1):
        for m in range(ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            template_alm[idx] = alm[idx]
    mono_dipole = hp.alm2map(template_alm, nside, lmax=lmax)
    return cmb_map - mono_dipole


def sample_random_points(
    nside: int, mask: np.ndarray, n_points: int = 3000, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample random points on the sphere outside the mask.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter of the mask.
    mask : np.ndarray
        Boolean array, True for valid pixels.
    n_points : int, optional
        Number of points to draw (default 3000; capped at the number of
        valid pixels).
    random_seed : int, optional
        Seed for the random number generator (default 42).

    Returns
    -------
    ra : np.ndarray
        Right ascensions in degrees.
    dec : np.ndarray
        Declinations in degrees.
    chosen : np.ndarray
        HEALPix pixel indices of the sampled points.
    """
    rng = np.random.default_rng(random_seed)
    valid_pix = np.where(mask)[0]
    chosen = rng.choice(valid_pix, size=min(n_points, len(valid_pix)), replace=False)
    theta, phi = hp.pix2ang(nside, chosen)
    ra = np.degrees(phi)
    dec = np.degrees(np.pi / 2 - theta)
    return ra, dec, chosen


def compute_correlations(
    cmb_map: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    pix_indices: np.ndarray,
    holonomies: np.ndarray,
    chi_lss: float,
    n_sample: int = N_SAMPLE_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the auto-correlation ξ_A and cross-correlation ξ_C.

    Correlations are estimated on a random subsample of point pairs,
    binned by comoving separation (converted to Gpc using
    ``HORIZON_SCALE_GPC``).

    Parameters
    ----------
    cmb_map : np.ndarray
        HEALPix temperature map.
    ra, dec : np.ndarray
        Sky coordinates of the sample points in degrees.
    pix_indices : np.ndarray
        HEALPix pixel indices of the sample points.
    holonomies : np.ndarray
        Array of shape (12, 4, 4) with the PDS holonomy matrices.
    chi_lss : float
        Comoving distance to the LSS in radians on S³.
    n_sample : int, optional
        Number of points subsampled for the pair sums
        (default ``N_SAMPLE_DEFAULT``).

    Returns
    -------
    xi_A : np.ndarray
        Auto-correlation per separation bin.
    xi_C : np.ndarray
        Cross-correlation (over holonomy images) per separation bin.
    counts_A : np.ndarray
        Pair counts per bin for ξ_A.
    counts_C : np.ndarray
        Pair counts per bin for ξ_C.
    """
    n_pts = len(ra)
    pos = sls_position(ra, dec, chi_lss)
    delta_T = cmb_map[pix_indices]
    delta_T = delta_T - np.mean(delta_T)

    xi_A = np.zeros(N_BINS)
    xi_C = np.zeros(N_BINS)
    counts_A = np.zeros(N_BINS)
    counts_C = np.zeros(N_BINS)

    n_sample = min(n_pts, n_sample)
    rng = np.random.default_rng(12345)
    idx_a = rng.choice(n_pts, size=n_sample, replace=False)
    idx_b = rng.choice(n_pts, size=n_sample, replace=False)

    pos_a = pos[idx_a]
    pos_b = pos[idx_b]
    dT_a = delta_T[idx_a]
    dT_b = delta_T[idx_b]

    dots_aa = np.clip(pos_a @ pos_b.T, -1, 1)
    dists_aa = np.arccos(dots_aa) * (chi_lss / np.pi) * HORIZON_SCALE_GPC
    for i in range(n_sample):
        for j in range(n_sample):
            if i == j:
                continue
            d = dists_aa[i, j]
            bi = np.digitize(d, R_BINS) - 1
            if 0 <= bi < N_BINS:
                xi_A[bi] += dT_a[i] * dT_b[j]
                counts_A[bi] += 1

    for h in range(N_HOLONOMIES):
        g = holonomies[h]
        pos_im = (g @ pos.T).T
        pos_im_a = pos_im[idx_a]
        dots_cc = np.clip(pos_im_a @ pos_b.T, -1, 1)
        dists_cc = np.arccos(dots_cc) * (chi_lss / np.pi) * HORIZON_SCALE_GPC
        for i in range(n_sample):
            for j in range(n_sample):
                d = dists_cc[i, j]
                bi = np.digitize(d, R_BINS) - 1
                if 0 <= bi < N_BINS:
                    xi_C[bi] += dT_a[i] * dT_b[j]
                    counts_C[bi] += 1

    for b in range(N_BINS):
        if counts_A[b] > 0:
            xi_A[b] /= counts_A[b]
        if counts_C[b] > 0:
            xi_C[b] /= counts_C[b]

    return xi_A, xi_C, counts_A, counts_C


def pseudo_likelihood(
    xi_A: np.ndarray, xi_C: np.ndarray, counts_A: np.ndarray, counts_C: np.ndarray
) -> float:
    """
    Compute the pseudo-likelihood P of the RBSG08 method.

    P = ∏ exp(-(ξ_C - ξ_A)²/(2σ²)) for ξ_C ≤ ξ_A
    P = ∏ (1 + 0.01·(ξ_C-ξ_A)/ξ_A) for ξ_C > ξ_A

    Parameters
    ----------
    xi_A, xi_C : np.ndarray
        Auto- and cross-correlations per separation bin.
    counts_A, counts_C : np.ndarray
        Pair counts per bin.

    Returns
    -------
    float
        Pseudo-likelihood value.
    """
    logP = 0.0
    for b in range(N_BINS):
        if counts_A[b] < 10 or counts_C[b] < 10:
            continue
        if np.abs(xi_A[b]) < 1e-10:
            continue
        diff = xi_C[b] - xi_A[b]
        sigma = (
            0.5 * np.abs(xi_A[b]) * np.sqrt(max(counts_A[b], 1) / max(counts_C[b], 1))
        )
        sigma = max(sigma, 1e-10)
        if diff <= 0:
            logP += -0.5 * (diff / sigma) ** 2
        else:
            logP += np.log(1.0 + 0.01 * diff / np.abs(xi_A[b]))
    return np.exp(logP)


def dist_to_pds_twist(twist_deg: float) -> float:
    """
    Circular distance from a twist angle to the nearest PDS prediction.

    Computes ``min`` over target in {+36°, −36°} of the circular (mod
    360°) angular distance between ``twist_deg`` and the target.

    Parameters
    ----------
    twist_deg : float
        Twist angle in degrees (any real value).

    Returns
    -------
    float
        Circular distance in degrees to the nearest of ±36°.
    """
    dists = []
    for target in (TWIST_PDS_DEG, -TWIST_PDS_DEG):
        delta = abs(twist_deg - target) % 360.0
        dists.append(min(delta, 360.0 - delta))
    return min(dists)


def plot_twist_distribution(results: dict, output_path: str) -> None:
    """
    Plot the histogram of twist angles from the grid search (Figure 3).

    Parameters
    ----------
    results : dict
        Results dictionary with keys ``all_twists`` and ``twist_median``.
    output_path : str
        Base path for output files (without extension).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    twists = results["all_twists"]
    ax.hist(twists, bins=72, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(
        x=TWIST_PDS_DEG,
        color="red",
        linewidth=2.5,
        linestyle="-",
        label=f"+{TWIST_PDS_DEG:.0f}° (PDS)",
    )
    ax.axvline(
        x=360 - TWIST_PDS_DEG,
        color="red",
        linewidth=2.5,
        linestyle="--",
        label=f"−{TWIST_PDS_DEG:.0f}° (PDS)",
    )
    ax.axvline(
        x=results["twist_median"],
        color="darkorange",
        linewidth=2.0,
        linestyle="-",
        label=f"Median: {results['twist_median']:.1f}°",
    )
    ax.set_xlabel(r"Twist angle $\phi$ [deg]", fontsize=12)
    ax.set_ylabel("Grid-search samples", fontsize=12)
    ax.set_title("Grid Search Distribution of Twist Angle φ", fontsize=13)
    ax.legend(fontsize=10)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_twist_dist.{fmt}")
    plt.close(fig)


def plot_alpha_twist_2d(results: dict, output_path: str) -> None:
    """
    Plot the 2D distribution of circle radius α vs twist φ (Figure 2).

    Parameters
    ----------
    results : dict
        Results dictionary with keys ``all_logP``, ``all_alphas``,
        ``all_twists``, ``alphas_highP`` and ``twists_highP``.
    output_path : str
        Base path for output files (without extension).
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    logP = results["all_logP"]
    high_P = (
        logP > np.percentile(logP, 50)
        if len(logP) > 0
        else np.ones(len(logP), dtype=bool)
    )
    ax.scatter(
        results["all_alphas"][~high_P],
        results["all_twists"][~high_P],
        c="gray",
        s=1,
        alpha=0.3,
    )
    ax.scatter(
        results["alphas_highP"], results["twists_highP"], c="steelblue", s=3, alpha=0.7
    )
    ax.axhline(y=TWIST_PDS_DEG, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axhline(
        y=360 - TWIST_PDS_DEG, color="red", linewidth=1.5, linestyle="--", alpha=0.7
    )
    ax.set_xlabel(r"Circle radius $\alpha$ [deg]", fontsize=12)
    ax.set_ylabel(r"Twist angle $\phi$ [deg]", fontsize=12)
    ax.set_title("Grid Search: α vs φ (color = high P)", fontsize=13)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_alpha_twist.{fmt}")
    plt.close(fig)


def plot_correlation_comparison(
    xi_A: np.ndarray, xi_C: np.ndarray, output_path: str
) -> None:
    """
    Plot the auto- vs cross-correlation comparison (Figure 4).

    Parameters
    ----------
    xi_A, xi_C : np.ndarray
        Auto- and cross-correlations per separation bin.
    output_path : str
        Base path for output files (without extension).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = R_BIN_CENTERS
    ax.plot(
        x,
        xi_A,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=8,
        label=r"$\xi_A$ (auto-correlation)",
    )
    ax.plot(
        x,
        xi_C,
        "s--",
        color="crimson",
        linewidth=2,
        markersize=8,
        label=r"$\xi_C$ (cross-correlation)",
    )
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"Separation $r$ [$h^{-1}$ Gpc]", fontsize=12)
    ax.set_ylabel(r"$\langle \delta T \cdot \delta T \rangle$ [$\mu$K²]", fontsize=12)
    ax.set_title("Auto vs Cross Correlation of SLS Copies", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_corr_comparison.{fmt}")
    plt.close(fig)


def plot_mollweide_centers(
    cmb_map: np.ndarray, results: dict, output_path: str
) -> None:
    """
    Plot a Mollweide map with the best-fit dodecahedron face centers (Figure 1).

    Parameters
    ----------
    cmb_map : np.ndarray
        HEALPix temperature map to display.
    results : dict
        Results dictionary with key ``best_params`` =
        ``(l, b, theta, alpha, twist)``.
    output_path : str
        Base path for output files (without extension).
    """
    best = results["best_params"]
    axes = get_dodecahedron_axes()

    rot_lb = R3.from_euler("zxz", [-np.radians(best[0]), np.radians(90 - best[1]), 0])
    rot_theta = R3.from_euler("z", np.radians(best[2]))

    centers = []
    for i in range(6):
        n = axes[i, 0]
        n_rot = rot_theta.apply(rot_lb.apply(n))
        n_rot = n_rot / np.linalg.norm(n_rot)
        dec_c = np.degrees(np.arcsin(np.clip(n_rot[2], -1, 1)))
        ra_c = np.degrees(np.arctan2(n_rot[1], n_rot[0])) % 360
        centers.append((ra_c, dec_c))
        centers.append(((ra_c + 180) % 360, -dec_c))

    fig = plt.figure(figsize=(14, 7))
    hp.mollview(
        cmb_map,
        title="Optimal Dodecahedron Face Centers",
        hold=True,
        unit=r"$\Delta T$ [K]",
        cmap="RdBu_r",
        min=-0.3,
        max=0.3,
        fig=1,
        cbar=True,
    )
    ax = fig.axes[0]
    for ra_c, dec_c in centers:
        ra_m = ra_c - 180.0
        if ra_m > 180:
            ra_m -= 360.0
        ax.plot(
            np.radians(ra_m),
            np.radians(dec_c),
            "o",
            color="lime",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_mollweide_centers.{fmt}")
    plt.close(fig)


if __name__ == "__main__":
    if hp is None:
        print("ERROR: healpy required. pip install healpy")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    NSIDE = 64
    N_POINTS = 200
    N_GRID = 6
    ALPHA_RANGE = np.linspace(10, 50, N_GRID)
    TWIST_RANGE = np.linspace(0, 350, 36)

    print("=" * 72)
    print("  RBSG08 SPATIAL CROSS-CORRELATION METHOD")
    print("  Poincaré Dodecahedral Space Topology Test")
    print("=" * 72)
    print(f"  Nside = {NSIDE}, N_points = {N_POINTS}")
    print(f"  α grid: {len(ALPHA_RANGE)} values, φ grid: {len(TWIST_RANGE)} values")
    print(f"  PDS prediction: φ = ±{TWIST_PDS_DEG:.0f}°")

    print("\n[1/6] Generating synthetic CMB map...")
    cmb_map = generate_cmb_map(nside=NSIDE, random_seed=42)
    cmb_masked, mask = apply_cmb_mask(cmb_map, gal_cut=20.0)
    cmb_clean = remove_monopole_dipole(cmb_masked)
    print(
        f"      Map: {len(cmb_map)} pixels, valid: {mask.sum()} ({mask.sum()/len(mask)*100:.1f}%)"
    )

    print(f"\n[2/6] Sampling {N_POINTS} random points outside mask...")
    ra_pts, dec_pts, pix_pts = sample_random_points(
        NSIDE, mask, n_points=N_POINTS, random_seed=42
    )
    print(f"      Sampled {len(ra_pts)} points")

    print("\n[3/6] Grid search over α and φ (fixed orientation)...")
    l0, b0, theta0 = 0.0, 45.0, 0.0

    best_logP = -np.inf
    best_alpha = np.nan
    best_twist = np.nan
    best_xi_A = None
    best_xi_C = None
    all_twists = []
    all_alphas = []
    all_logPs = []

    for alpha in tqdm(ALPHA_RANGE, desc="  α scan"):
        chi_lss = np.radians(90 - alpha)
        for twist in TWIST_RANGE:
            holonomies = build_holonomies((l0, b0, theta0), twist)
            xi_A, xi_C, cA, cC = compute_correlations(
                cmb_clean, ra_pts, dec_pts, pix_pts, holonomies, chi_lss
            )
            logP = np.log(pseudo_likelihood(xi_A, xi_C, cA, cC) + 1e-300)
            all_twists.append(twist)
            all_alphas.append(alpha)
            all_logPs.append(logP)
            if logP > best_logP:
                best_logP = logP
                best_alpha = alpha
                best_twist = twist
                best_xi_A = xi_A.copy()
                best_xi_C = xi_C.copy()

    all_twists = np.array(all_twists)
    all_alphas = np.array(all_alphas)
    all_logPs = np.array(all_logPs)

    high_P = all_logPs > np.percentile(all_logPs, 50)
    twist_median = np.median(all_twists[high_P])
    alpha_median = np.median(all_alphas[high_P])
    dist_to_36 = dist_to_pds_twist(twist_median)

    print("\n[4/6] Results:")
    print(f"      Best α = {best_alpha:.1f}°, Best φ = {best_twist:.1f}°")
    print(f"      Best logP = {best_logP:.2f}")
    print(f"      Twist median (high P): {twist_median:.1f}°")
    print(f"      Distance to ±{TWIST_PDS_DEG:.0f}°: {dist_to_36:.1f}°")

    if dist_to_36 < 5:
        verdict = (
            f"grid-search twist consistent with PDS prediction "
            f"(±{TWIST_PDS_DEG:.0f}°) at this resolution"
        )
    elif dist_to_36 < 10:
        verdict = (
            f"grid-search twist marginally consistent with PDS "
            f"prediction (±{TWIST_PDS_DEG:.0f}°) at this resolution"
        )
    else:
        verdict = (
            f"grid-search twist inconsistent with PDS prediction "
            f"(±{TWIST_PDS_DEG:.0f}°) at this resolution"
        )
    print(f"      Verdict: {verdict}")

    print("\n[5/6] Generating figures...")
    fig_base = os.path.join(output_dir, "fig")

    results = {
        "all_twists": all_twists,
        "all_alphas": all_alphas,
        "all_logP": all_logPs,
        "twist_median": twist_median,
        "alpha_median": alpha_median,
        "dist_to_36": dist_to_36,
        "best_params": np.array([l0, b0, theta0, best_alpha, best_twist]),
        "best_logP": best_logP,
        "twists_highP": all_twists[high_P] if high_P.sum() > 0 else all_twists,
        "alphas_highP": all_alphas[high_P] if high_P.sum() > 0 else all_alphas,
    }

    plot_twist_distribution(results, fig_base)
    print(f"  Saved: {fig_base}_twist_dist.[pdf,png]")

    plot_alpha_twist_2d(results, fig_base)
    print(f"  Saved: {fig_base}_alpha_twist.[pdf,png]")

    plot_correlation_comparison(best_xi_A, best_xi_C, fig_base)
    print(f"  Saved: {fig_base}_corr_comparison.[pdf,png]")

    plot_mollweide_centers(cmb_clean, results, fig_base)
    print(f"  Saved: {fig_base}_mollweide_centers.[pdf,png]")

    print("\n[6/6] Saving results...")
    np.savez(
        os.path.join(output_dir, "rbsg08_results.npz"),
        best_alpha=best_alpha,
        best_twist=best_twist,
        twist_median=twist_median,
        dist_to_36=dist_to_36,
        all_twists=all_twists,
        all_alphas=all_alphas,
        all_logPs=all_logPs,
        xi_A=best_xi_A,
        xi_C=best_xi_C,
    )
    print(f"  Saved: {os.path.join(output_dir, 'rbsg08_results.npz')}")

    print(f"\n{'='*72}")
    print("  FINAL SUMMARY")
    print(f"{'='*72}")
    print(f"  Observed twist median: φ = {twist_median:.1f}°")
    print(f"  PDS prediction: φ = ±{TWIST_PDS_DEG:.0f}°")
    print(f"  Circular distance: {dist_to_36:.1f}°")
    print(f"  Verdict: {verdict}")
    print("")
    print("  Note: this run uses a synthetic Gaussian CMB realization and")
    print("  validates the pipeline only. Real Planck data (and substantially")
    print("  more computing) are required for any conclusion about the")
    print("  topology of the Universe.")
    print(f"{'='*72}")
