#!/usr/bin/env python3
"""
Cosmicflows-4 peculiar velocity analysis for dodecahedral H0 anisotropy.

Tests whether the dodecahedral H0 pattern seen in SNe Ia is reflected
in the peculiar velocity field of galaxies from Cosmicflows-4.

Method:
  V_pec = V_obs - H0_mean * Distance
  If dodecahedral anisotropy is real, sectors with high H0 should show
  systematic V_pec > 0 (outflow) and sectors with low H0 should show
  V_pec < 0 (inflow).

Data availability
-----------------
The real Cosmicflows-4 catalogue (Tully et al. 2023) is NOT distributed
with this repository. Download it separately (e.g. from the Extragalactic
Distance Database, https://edd.ifa.hawaii.edu/) and save it as
``data/CF4_data.csv`` with columns [ra, dec, distance, v_obs] (optionally
distance_err, distance_method).

If ``data/CF4_data.csv`` is absent, this module falls back to a SYNTHETIC
mock catalogue stored in ``data/CF4_synthetic.csv``. The synthetic data
are suitable ONLY for validating that the pipeline runs end-to-end: they
are drawn as v_obs = 70 * d + N(0, 300 km/s) and therefore contain NO
dodecahedral signal by construction. Any scientific conclusion requires
the real catalogue.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dodecahedron import get_dodecahedron_normals, assign_sectors

warnings.filterwarnings("ignore")

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

CMAP_VPEC = "RdBu_r"
H0_DEFAULT = 70.0

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CF4_REAL_PATH = os.path.join(_DATA_DIR, "CF4_data.csv")
CF4_SYNTHETIC_PATH = os.path.join(_DATA_DIR, "CF4_synthetic.csv")


def generate_synthetic_cf4(
    n_galaxies: int = 56000, random_seed: int = 42, output_path: str | None = None
) -> pd.DataFrame:
    """
    Generate a synthetic mock of the Cosmicflows-4 dataset.

    Matches known catalogue properties: ~56k galaxies, distance methods
    (TF, FP, TRGB, Cepheids, SBF, SNIa), realistic sky coverage with a
    Zone of Avoidance gap. Velocities are drawn as
    v_obs = H0_DEFAULT * d + N(0, 300 km/s), so the mock contains NO
    dodecahedral signal by construction and is intended only for
    pipeline validation.

    Parameters
    ----------
    n_galaxies : int
        Number of galaxies to generate.
    random_seed : int
        Random seed for reproducibility.
    output_path : str or None
        If provided, save CSV to this path. Refuses to write to
        ``CF4_data.csv`` (reserved for the real catalogue).

    Returns
    -------
    pandas.DataFrame
        Columns [ra, dec, distance, distance_err, v_obs, distance_method].
    """
    if output_path is not None and os.path.basename(output_path) == "CF4_data.csv":
        raise ValueError(
            "Refusing to write synthetic data to CF4_data.csv: that filename "
            "is reserved for the real Cosmicflows-4 catalogue. "
            "Use CF4_synthetic.csv instead."
        )

    rng = np.random.default_rng(random_seed)

    ra = rng.uniform(0, 360, n_galaxies)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_galaxies)))

    gal_lat = _galactic_latitude(ra, dec)
    zoa_mask = np.abs(gal_lat) < 10
    n_zoa = zoa_mask.sum()
    ra[zoa_mask] = rng.uniform(0, 360, n_zoa)
    dec[zoa_mask] = np.degrees(np.arcsin(rng.uniform(-0.17, 0.17, n_zoa)))

    methods = ["TF", "FP", "TRGB", "Cepheids", "SBF", "SNIa"]
    method_weights = np.array([0.40, 0.25, 0.10, 0.05, 0.10, 0.10])
    method_weights = method_weights / method_weights.sum()
    distance_method = rng.choice(methods, size=n_galaxies, p=method_weights)

    method_scatter = {
        "TF": 0.20,
        "FP": 0.25,
        "TRGB": 0.08,
        "Cepheids": 0.10,
        "SBF": 0.12,
        "SNIa": 0.07,
    }
    method_max_dist = {
        "TF": 150,
        "FP": 200,
        "TRGB": 30,
        "Cepheids": 40,
        "SBF": 100,
        "SNIa": 300,
    }

    distance = np.zeros(n_galaxies)
    distance_err = np.zeros(n_galaxies)
    for m in methods:
        mask = distance_method == m
        d = rng.uniform(1, method_max_dist[m], mask.sum())
        distance[mask] = d
        distance_err[mask] = d * method_scatter[m]

    v_hubble = H0_DEFAULT * distance
    v_pec_true = rng.normal(0, 300, n_galaxies)
    v_obs = v_hubble + v_pec_true

    df = pd.DataFrame(
        {
            "ra": np.round(ra, 6),
            "dec": np.round(dec, 6),
            "distance": np.round(distance, 3),
            "distance_err": np.round(distance_err, 3),
            "v_obs": np.round(v_obs, 1),
            "distance_method": distance_method,
        }
    )

    if output_path is not None:
        df.to_csv(output_path, index=False)

    return df


def _galactic_latitude(ra_deg, dec_deg):
    """Approximate galactic latitude from equatorial coordinates."""
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    ngp_ra = np.radians(192.85948)
    ngp_dec = np.radians(27.12825)
    sin_b = np.sin(dec_rad) * np.sin(ngp_dec) + np.cos(dec_rad) * np.cos(
        ngp_dec
    ) * np.cos(ra_rad - ngp_ra)
    return np.degrees(np.arcsin(np.clip(sin_b, -1, 1)))


def load_cf4(filepath: str | None = None) -> pd.DataFrame:
    """
    Load Cosmicflows-4 data from a CSV file.

    If ``filepath`` is None, tries the real catalogue at
    ``data/CF4_data.csv`` first; if it is absent, falls back to the
    synthetic mock at ``data/CF4_synthetic.csv`` with a loud warning
    printed to stdout.

    Parameters
    ----------
    filepath : str or None
        Explicit path to a CF4 CSV file, or None for the default
        real-then-synthetic lookup.

    Returns
    -------
    pandas.DataFrame
        Columns [ra, dec, distance, distance_err, v_obs, distance_method,
        v_pec]. The attribute ``df.attrs['is_synthetic']`` is True when
        the synthetic mock was loaded.
    """
    is_synthetic = False
    if filepath is None:
        if os.path.exists(CF4_REAL_PATH):
            filepath = CF4_REAL_PATH
        elif os.path.exists(CF4_SYNTHETIC_PATH):
            filepath = CF4_SYNTHETIC_PATH
            is_synthetic = True
            banner = "!" * 72
            print(banner)
            print("!!  WARNING: real Cosmicflows-4 catalogue (data/CF4_data.csv)")
            print("!!  NOT FOUND. Loading SYNTHETIC mock data instead:")
            print(f"!!      {os.path.abspath(CF4_SYNTHETIC_PATH)}")
            print("!!  The mock is drawn as v_obs = 70*d + N(0, 300 km/s) and")
            print("!!  contains NO dodecahedral signal by construction.")
            print("!!  Results below validate the PIPELINE ONLY, not the science.")
            print("!!  Download the real CF4 catalogue (see module docstring).")
            print(banner)
        else:
            raise FileNotFoundError(
                f"Neither {CF4_REAL_PATH} nor {CF4_SYNTHETIC_PATH} exists. "
                "Download the real CF4 catalogue or run "
                "generate_synthetic_cf4(output_path=CF4_SYNTHETIC_PATH)."
            )
    else:
        is_synthetic = os.path.basename(filepath) == os.path.basename(
            CF4_SYNTHETIC_PATH
        )

    df = pd.read_csv(filepath)

    required = ["ra", "dec", "distance", "v_obs"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[(df["distance"] > 0) & (df["v_obs"] > 0)].copy()

    if "distance_err" not in df.columns:
        df["distance_err"] = df["distance"] * 0.20

    if "distance_method" not in df.columns:
        df["distance_method"] = "unknown"

    df["v_pec"] = np.nan
    df = df.reset_index(drop=True)
    df.attrs["is_synthetic"] = is_synthetic

    return df


def compute_peculiar(df: pd.DataFrame, H0_mean: float = H0_DEFAULT) -> pd.DataFrame:
    """
    Compute peculiar velocities: V_pec = V_obs - H0_mean * Distance.

    Parameters
    ----------
    df : pandas.DataFrame
        CF4 data with 'distance' and 'v_obs' columns.
    H0_mean : float
        Assumed mean Hubble constant [km/s/Mpc].

    Returns
    -------
    pandas.DataFrame
        Copy of the input with the 'v_pec' column filled.
    """
    df = df.copy()
    df["v_pec"] = df["v_obs"] - H0_mean * df["distance"]
    return df


def fit_sector_peculiar(df: pd.DataFrame, sector_ids: np.ndarray) -> list[dict]:
    """
    Compute mean peculiar velocity per dodecahedron sector.

    Parameters
    ----------
    df : pandas.DataFrame
        CF4 data with 'v_pec' column.
    sector_ids : numpy.ndarray
        Sector assignment for each galaxy.

    Returns
    -------
    list of dict
        One entry per sector (0-11) with keys [sector_id, n_gal,
        mean_v_pec, err_v_pec, weighted_mean_v_pec, std_v_pec].
        Empty sectors have n_gal = 0 and NaN statistics.
    """
    results = []
    for sector in range(12):
        mask = sector_ids == sector
        n = mask.sum()
        if n == 0:
            results.append(
                {
                    "sector_id": sector,
                    "n_gal": 0,
                    "mean_v_pec": np.nan,
                    "err_v_pec": np.nan,
                    "weighted_mean_v_pec": np.nan,
                    "std_v_pec": np.nan,
                }
            )
            continue

        v_pec = df["v_pec"].values[mask]
        mean_v = np.mean(v_pec)
        std_v = np.std(v_pec, ddof=1)
        err_v = std_v / np.sqrt(n)

        if "distance_err" in df.columns:
            dist_err = df["distance_err"].values[mask]
            weights = 1.0 / (dist_err**2 + 1e-10)
            weights /= weights.sum()
            w_mean = np.sum(weights * v_pec)
        else:
            w_mean = mean_v

        results.append(
            {
                "sector_id": sector,
                "n_gal": n,
                "mean_v_pec": round(mean_v, 2),
                "err_v_pec": round(err_v, 2),
                "weighted_mean_v_pec": round(w_mean, 2),
                "std_v_pec": round(std_v, 2),
            }
        )

    return results


def test_dodecahedral_pattern(
    peculiar_results: list[dict], h0_results: list[dict] | None = None
) -> dict:
    """
    Test correlation between H0 (from SNe) and V_pec (from CF4) across sectors.

    Both quantities are aligned on sector_id into length-12 arrays
    (NaN for sectors without data) before computing the Spearman
    correlation over the jointly valid sectors.

    Parameters
    ----------
    peculiar_results : list of dict
        V_pec per sector from CF4 (output of fit_sector_peculiar).
    h0_results : list of dict or None
        H0 per sector from Pantheon+, each entry with keys
        [sector_id, H0, n_sne]. If None, loads per-sector H0 from
        outputs/mc_pantheon.npz.

    Returns
    -------
    dict
        Keys [rho, p_value, h0_vals, vpec_vals]; h0_vals and vpec_vals
        are length-12 arrays indexed by sector_id.
    """
    vpec_vals = np.full(12, np.nan)
    for r in peculiar_results:
        vpec_vals[int(r["sector_id"])] = r["mean_v_pec"]

    if h0_results is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mc_path = os.path.join(base_dir, "..", "outputs", "mc_pantheon.npz")
        if os.path.exists(mc_path):
            mc_data = dict(np.load(mc_path, allow_pickle=True))
            h0_vals = np.asarray(mc_data["h0_real"], dtype=float)
        else:
            h0_vals = np.full(12, np.nan)
    else:
        h0_vals = np.full(12, np.nan)
        for r in h0_results:
            if r["n_sne"] > 0:
                h0_vals[int(r["sector_id"])] = r["H0"]

    valid = ~np.isnan(h0_vals) & ~np.isnan(vpec_vals)
    if valid.sum() < 3:
        return {
            "rho": np.nan,
            "p_value": np.nan,
            "h0_vals": h0_vals,
            "vpec_vals": vpec_vals,
        }

    rho, p_val = spearmanr(h0_vals[valid], vpec_vals[valid])

    return {
        "rho": rho,
        "p_value": p_val,
        "h0_vals": h0_vals,
        "vpec_vals": vpec_vals,
    }


def monte_carlo_cf4(
    df: pd.DataFrame, normals: np.ndarray, n_mocks: int = 1000, random_seed: int = 42
) -> dict:
    """
    Monte Carlo test: shuffle V_pec, recompute sector means, compare variance.

    Parameters
    ----------
    df : pandas.DataFrame
        CF4 data with 'v_pec' column.
    normals : numpy.ndarray
        Dodecahedron face normals, shape (12, 3).
    n_mocks : int
        Number of shuffles.
    random_seed : int
        Random seed.

    Returns
    -------
    dict
        Keys [observed_std, mock_stds, p_value, mock_mean, mock_std].
    """
    rng = np.random.default_rng(random_seed)
    sector_ids = assign_sectors(df["ra"].values, df["dec"].values, normals)

    sector_means_obs = []
    for s in range(12):
        mask = sector_ids == s
        if mask.sum() > 0:
            sector_means_obs.append(np.mean(df["v_pec"].values[mask]))
    observed_std = np.std(sector_means_obs, ddof=1)

    v_pec = df["v_pec"].values.copy()
    mock_stds = np.zeros(n_mocks)
    for i in range(n_mocks):
        rng.shuffle(v_pec)
        df_shuffled = df.copy()
        df_shuffled["v_pec"] = v_pec
        sector_means = []
        for s in range(12):
            mask = sector_ids == s
            if mask.sum() > 0:
                sector_means.append(np.mean(df_shuffled["v_pec"].values[mask]))
        mock_stds[i] = np.std(sector_means, ddof=1)

    p_value = np.sum(mock_stds >= observed_std) / n_mocks

    return {
        "observed_std": observed_std,
        "mock_stds": mock_stds,
        "p_value": p_value,
        "mock_mean": np.mean(mock_stds),
        "mock_std": np.std(mock_stds, ddof=1),
    }


def plot_cf4_mollweide(
    df: pd.DataFrame,
    sector_ids: np.ndarray,
    peculiar_results: list[dict],
    output_path: str,
) -> None:
    """
    Mollweide projection of CF4 galaxies colored by sector mean V_pec.

    Parameters
    ----------
    df : pandas.DataFrame
        CF4 data.
    sector_ids : numpy.ndarray
        Sector assignments.
    peculiar_results : list of dict
        V_pec per sector.
    output_path : str
        Base path for output figures.
    """
    vpec_map = {
        r["sector_id"]: r["mean_v_pec"] for r in peculiar_results if r["n_gal"] > 0
    }
    vpec_vals = np.array([vpec_map.get(s, np.nan) for s in sector_ids])

    ra_m = df["ra"].values - 180.0
    ra_m[ra_m > 180] -= 360.0

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)

    vlim = max(abs(np.nanmin(vpec_vals)), abs(np.nanmax(vpec_vals)))
    sc = ax.scatter(
        np.radians(ra_m),
        np.radians(df["dec"].values),
        c=vpec_vals,
        cmap=CMAP_VPEC,
        s=1,
        alpha=0.5,
        edgecolors="none",
        vmin=-vlim,
        vmax=vlim,
    )

    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label(r"$\langle V_{\rm pec} \rangle$ [km/s]", fontsize=11)

    ax.set_title(
        "Cosmicflows-4: Peculiar Velocities by Dodecahedron Sector", fontsize=13, pad=10
    )
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    ax.set_xticklabels([f"{t}°" for t in tick_labels], fontsize=8)

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_cf4_mollweide.{fmt}")
    plt.close(fig)


def plot_cf4_bars(peculiar_results: list[dict], output_path: str) -> None:
    """
    Bar chart of mean V_pec per sector.

    Parameters
    ----------
    peculiar_results : list of dict
        V_pec per sector; empty sectors (n_gal = 0) are skipped, and bar
        positions follow the actual sector_id values.
    output_path : str
        Base path for output figures.
    """
    populated = [r for r in peculiar_results if r["n_gal"] > 0]
    populated = sorted(populated, key=lambda r: r["sector_id"])
    sectors = np.array([r["sector_id"] for r in populated])
    vpec = np.array([r["mean_v_pec"] for r in populated])
    errs = np.array([r["err_v_pec"] for r in populated])
    n_gal = np.array([r["n_gal"] for r in populated])

    colors = plt.cm.RdBu_r((vpec - vpec.min()) / (vpec.max() - vpec.min() + 1e-10))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        sectors,
        vpec,
        yerr=errs,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"linewidth": 1.0},
    )
    ax.axhline(y=0, color="black", linewidth=1.0, alpha=0.5)

    for s, v, e, n in zip(sectors, vpec, errs, n_gal):
        ax.text(
            s, v + e + 0.5, f"n={n}", ha="center", va="bottom", fontsize=6, color="gray"
        )

    ax.set_xticks(sectors)
    ax.set_xticklabels([f"Face {s+1}" for s in sectors], rotation=45, ha="right")
    ax.set_ylabel(r"$\langle V_{\rm pec} \rangle$ [km/s]", fontsize=12)
    ax.set_xlabel("Dodecahedron Sector", fontsize=12)
    ax.set_title("Mean Peculiar Velocity per Dodecahedron Sector (CF4)", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_cf4_bars.{fmt}")
    plt.close(fig)


def plot_h0_vs_vpec(corr_results: dict, output_path: str) -> None:
    """
    Scatter plot: H0 (from SNe) vs mean V_pec (from CF4) for 12 sectors.

    Parameters
    ----------
    corr_results : dict
        Output of test_dodecahedral_pattern().
    output_path : str
        Base path for output figures.
    """
    h0 = corr_results["h0_vals"]
    vpec = corr_results["vpec_vals"]
    rho = corr_results["rho"]
    p_val = corr_results["p_value"]

    valid = ~np.isnan(h0) & ~np.isnan(vpec)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        h0[valid],
        vpec[valid],
        c=np.arange(12)[valid],
        cmap="tab20",
        s=80,
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )

    for i in np.where(valid)[0]:
        ax.annotate(
            f"F{i+1}",
            (h0[i], vpec[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    if valid.sum() >= 3:
        coeffs = np.polyfit(h0[valid], vpec[valid], 1)
        x_fit = np.linspace(h0[valid].min(), h0[valid].max(), 50)
        ax.plot(
            x_fit,
            np.polyval(coeffs, x_fit),
            "--",
            color="gray",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(r"$H_0$ from Pantheon+ SNe [km s$^{-1}$ Mpc$^{-1}$]", fontsize=12)
    ax.set_ylabel(r"$\langle V_{\rm pec} \rangle$ from CF4 [km/s]", fontsize=12)
    ax.set_title(f"H₀ vs V_pec: Spearman ρ = {rho:.3f}, p = {p_val:.3f}", fontsize=13)
    ax.grid(True, alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_h0_vs_vpec.{fmt}")
    plt.close(fig)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  COSMICFLOWS-4 PECULIAR VELOCITY ANALYSIS")
    print("  Dodecahedral H0 Anisotropy Cross-Check")
    print("=" * 72)

    # --- Load or generate data ---
    print("\n[1/6] Loading CF4 data...")
    if not os.path.exists(CF4_REAL_PATH) and not os.path.exists(CF4_SYNTHETIC_PATH):
        print("      No CF4 catalogue found. Generating synthetic mock dataset...")
        generate_synthetic_cf4(
            n_galaxies=56000, random_seed=42, output_path=CF4_SYNTHETIC_PATH
        )
        print(f"      Saved synthetic CF4 to: {CF4_SYNTHETIC_PATH}")
    df = load_cf4()

    print(
        f"      Loaded {len(df)} galaxies"
        + (
            " [SYNTHETIC — pipeline validation only]"
            if df.attrs.get("is_synthetic")
            else ""
        )
    )
    print(
        f"      Distance range: [{df['distance'].min():.1f}, {df['distance'].max():.1f}] Mpc"
    )
    print(
        f"      V_obs range:    [{df['v_obs'].min():.0f}, {df['v_obs'].max():.0f}] km/s"
    )

    # --- Compute peculiar velocities ---
    print(f"\n[2/6] Computing peculiar velocities (H0_mean = {H0_DEFAULT})...")
    df = compute_peculiar(df, H0_mean=H0_DEFAULT)
    print(f"      V_pec mean:  {df['v_pec'].mean():.1f} km/s")
    print(f"      V_pec std:   {df['v_pec'].std():.1f} km/s")
    print(f"      V_pec range: [{df['v_pec'].min():.0f}, {df['v_pec'].max():.0f}] km/s")

    # --- Assign sectors ---
    print("\n[3/6] Assigning galaxies to dodecahedron sectors...")
    normals = get_dodecahedron_normals()
    sector_ids = assign_sectors(df["ra"].values, df["dec"].values, normals)
    unique, counts = np.unique(sector_ids, return_counts=True)
    for s, c in zip(unique, counts):
        print(f"      Sector {s:2d}: {c:5d} galaxies")

    # --- Fit per sector ---
    print("\n[4/6] Computing mean V_pec per sector...")
    peculiar_results = fit_sector_peculiar(df, sector_ids)

    print(
        f"\n  {'Sector':>6s}  {'N_gal':>6s}  {'<V_pec>':>10s}  {'Err':>8s}  {'Std':>8s}  {'W_Mean':>10s}"
    )
    print("  " + "-" * 58)
    for r in peculiar_results:
        if r["n_gal"] > 0:
            print(
                f"  {r['sector_id']:6d}  {r['n_gal']:6d}  "
                f"{r['mean_v_pec']:10.2f}  {r['err_v_pec']:8.2f}  "
                f"{r['std_v_pec']:8.2f}  {r['weighted_mean_v_pec']:10.2f}"
            )

    vpec_vals = np.array([r["mean_v_pec"] for r in peculiar_results if r["n_gal"] > 0])
    print(f"\n  V_pec std across sectors: {np.std(vpec_vals, ddof=1):.2f} km/s")

    # --- Test dodecahedral pattern ---
    print("\n[5/6] Testing dodecahedral pattern: H0 (SNe) vs V_pec (CF4)...")
    corr_results = test_dodecahedral_pattern(peculiar_results)
    print(f"      Spearman ρ = {corr_results['rho']:.4f}")
    print(f"      p-value     = {corr_results['p_value']:.4f}")

    if corr_results["p_value"] < 0.05 and not np.isnan(corr_results["p_value"]):
        print("      → SIGNIFICANT correlation between H0 and V_pec!")
    else:
        print("      → No significant correlation (consistent with isotropy)")

    # --- Monte Carlo ---
    print("\n[6/6] Monte Carlo shuffle test...")
    mc_cf4 = monte_carlo_cf4(df, normals, n_mocks=1000, random_seed=42)
    print(f"      Observed std of sector means: {mc_cf4['observed_std']:.2f} km/s")
    print(f"      Mock mean std:                {mc_cf4['mock_mean']:.2f} km/s")
    print(f"      Mock std of std:              {mc_cf4['mock_std']:.2f} km/s")
    print(f"      p-value:                      {mc_cf4['p_value']:.4f}")

    # --- Visualization ---
    print("\nGenerating figures...")
    fig_base = os.path.join(output_dir, "fig")

    plot_cf4_mollweide(df, sector_ids, peculiar_results, fig_base)
    print(f"  Saved: {fig_base}_cf4_mollweide.[pdf,png]")

    plot_cf4_bars(peculiar_results, fig_base)
    print(f"  Saved: {fig_base}_cf4_bars.[pdf,png]")

    plot_h0_vs_vpec(corr_results, fig_base)
    print(f"  Saved: {fig_base}_h0_vs_vpec.[pdf,png]")

    # --- Save results ---
    np.savez(
        os.path.join(output_dir, "cf4_results.npz"),
        vpec_vals=vpec_vals,
        peculiar_results=np.array(peculiar_results, dtype=object),
        spearman_rho=corr_results["rho"],
        spearman_p=corr_results["p_value"],
        mc_observed_std=mc_cf4["observed_std"],
        mc_p_value=mc_cf4["p_value"],
        is_synthetic=bool(df.attrs.get("is_synthetic", False)),
    )
    print(f"\nResults saved to: {os.path.join(output_dir, 'cf4_results.npz')}")
    print("=" * 72)
