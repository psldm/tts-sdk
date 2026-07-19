#!/usr/bin/env python3
"""
Standard sirens and lensed quasars module for dodecahedral H0 anisotropy.

Combines H0 measurements from:
  - GW standard sirens (LIGO/Virgo/KAGRA): GW170817 + dark sirens
  - Lensed quasars (H0LiCOW/TDCOSMO/SHARP): time-delay cosmography

Tests whether the sparse H0 measurements from sirens/lenses are consistent
with the dodecahedral anisotropy pattern seen in Pantheon+ SNe Ia.

Data caveat
-----------
The individual dark-siren H0 values tabulated in :func:`get_gw_events` are
illustrative numbers compiled from the literature, not official per-event
posteriors released by the LVK collaboration.  They are used only as an
order-of-magnitude cross-check of the sector pattern.  All-sky combined
constraints (GWTC-3 dark sirens, TDCOSMO) carry no directional information
and are excluded from the sector analysis; they are reported separately as
reference values.

References:
  - Abbott+2017 Nature 551, 85 (GW170817)
  - Abbott+2021 ApJ 909, 218 (GWTC-3 dark sirens)
  - Wong+2020 MNRAS 498, 1420 (H0LiCOW)
  - Birrer+2020 A&A 643, A165 (TDCOSMO)
  - Chen+2019 MNRAS 490, 1743 (SHARP)
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import spearmanr, chi2

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


def get_gw_events() -> pd.DataFrame:
    """
    Return H0 measurements from GW standard sirens.

    The GW170817 value is the published bright-siren posterior
    (Abbott+2017 Nature 551, 85).  The individual dark-siren H0 values
    (GW190425, GW190814, GW200105, GW200115) are ILLUSTRATIVE numbers
    compiled from the literature (GWTC-2/GWTC-3 dark-siren analyses);
    they are not official per-event posteriors and should not be used
    for quantitative cosmology beyond the cross-check performed here.
    The GWTC-3 combined dark-siren constraint (Abbott+2021 ApJ 909, 218)
    is an all-sky measurement: it has no sky localization and is flagged
    with ``all_sky=True`` (its ra/dec are placeholders).

    Returns
    -------
    pandas.DataFrame
        Columns [name, ra, dec, H0, err_plus, err_minus, type, ref,
        all_sky].  For all-sky entries the ra/dec values are placeholders
        and must not be used for sector assignment.
    """
    events = [
        {
            "name": "GW170817",
            "ra": 197.45,
            "dec": -23.38,
            "H0": 70.0,
            "err_plus": 12.0,
            "err_minus": 8.0,
            "type": "GW",
            "ref": "Abbott+2017",
            "all_sky": False,
        },
        {
            "name": "GW190425",
            "ra": 260.0,
            "dec": 20.0,
            "H0": 69.0,
            "err_plus": 16.0,
            "err_minus": 8.0,
            "type": "GW",
            "ref": "Abbott+2020",
            "all_sky": False,
        },
        {
            "name": "GW190814_ds",
            "ra": 12.6,
            "dec": -24.9,
            "H0": 72.0,
            "err_plus": 18.0,
            "err_minus": 15.0,
            "type": "GW",
            "ref": "DarkSiren_GWTC2",
            "all_sky": False,
        },
        {
            "name": "GW200105_ds",
            "ra": 150.0,
            "dec": 10.0,
            "H0": 68.0,
            "err_plus": 20.0,
            "err_minus": 12.0,
            "type": "GW",
            "ref": "DarkSiren_GWTC3",
            "all_sky": False,
        },
        {
            "name": "GW200115_ds",
            "ra": 45.0,
            "dec": -5.0,
            "H0": 73.0,
            "err_plus": 22.0,
            "err_minus": 14.0,
            "type": "GW",
            "ref": "DarkSiren_GWTC3",
            "all_sky": False,
        },
        {
            "name": "GWTC3_darksiren_comb",
            "ra": 180.0,  # placeholder: all-sky constraint, no localization
            "dec": 0.0,  # placeholder: all-sky constraint, no localization
            "H0": 68.0,
            "err_plus": 8.0,
            "err_minus": 6.0,
            "type": "GW",
            "ref": "Abbott+2021",
            "all_sky": True,
        },
    ]
    return pd.DataFrame(events)


def get_lens_events() -> pd.DataFrame:
    """
    Return H0 measurements from lensed quasars (time-delay cosmography).

    Individual-lens H0 values are compiled from the H0LiCOW/TDCOSMO/SHARP
    literature for illustration; see the module docstring for the caveat
    on their use.  The TDCOSMO combined constraint (Birrer+2020) averages
    over lenses across the sky: it has no single sky localization and is
    flagged with ``all_sky=True`` (its ra/dec are placeholders).

    Sources:
      H0LiCOW: Wong+2020 MNRAS 498, 1420
      TDCOSMO: Birrer+2020 A&A 643, A165
      SHARP: Chen+2019 MNRAS 490, 1743

    Returns
    -------
    pandas.DataFrame
        Columns [name, ra, dec, H0, err_plus, err_minus, type, ref,
        all_sky].  For all-sky entries the ra/dec values are placeholders
        and must not be used for sector assignment.
    """
    events = [
        {
            "name": "B1608+656",
            "ra": 242.5,
            "dec": 65.5,
            "H0": 71.0,
            "err_plus": 2.9,
            "err_minus": 3.1,
            "type": "lens",
            "ref": "Suyu+2010",
            "all_sky": False,
        },
        {
            "name": "RXJ1131-1231",
            "ra": 172.96,
            "dec": -12.53,
            "H0": 78.3,
            "err_plus": 3.4,
            "err_minus": 3.3,
            "type": "lens",
            "ref": "Suyu+2014",
            "all_sky": False,
        },
        {
            "name": "HE0435-1223",
            "ra": 69.59,
            "dec": -12.30,
            "H0": 71.7,
            "err_plus": 4.3,
            "err_minus": 4.3,
            "type": "lens",
            "ref": "Wong+2017",
            "all_sky": False,
        },
        {
            "name": "PG1115+080",
            "ra": 169.57,
            "dec": 7.77,
            "H0": 68.8,
            "err_plus": 5.3,
            "err_minus": 5.3,
            "type": "lens",
            "ref": "Chen+2019",
            "all_sky": False,
        },
        {
            "name": "J1206+4332",
            "ra": 181.5,
            "dec": 43.53,
            "H0": 68.8,
            "err_plus": 5.3,
            "err_minus": 5.3,
            "type": "lens",
            "ref": "Birrer+2019",
            "all_sky": False,
        },
        {
            "name": "WFI2033-4723",
            "ra": 308.33,
            "dec": -47.38,
            "H0": 71.6,
            "err_plus": 4.3,
            "err_minus": 4.3,
            "type": "lens",
            "ref": "Rusu+2020",
            "all_sky": False,
        },
        {
            "name": "DESJ0408-5354",
            "ra": 62.0,
            "dec": -53.90,
            "H0": 74.2,
            "err_plus": 3.0,
            "err_minus": 3.0,
            "type": "lens",
            "ref": "SHARP_Chen+2019",
            "all_sky": False,
        },
        {
            "name": "PSJ0147+4630",
            "ra": 26.75,
            "dec": 46.50,
            "H0": 73.1,
            "err_plus": 5.7,
            "err_minus": 5.7,
            "type": "lens",
            "ref": "SHARP_Chen+2019",
            "all_sky": False,
        },
        {
            "name": "TDCOSMO_comb",
            "ra": 180.0,  # placeholder: all-sky constraint, no localization
            "dec": 0.0,  # placeholder: all-sky constraint, no localization
            "H0": 73.3,
            "err_plus": 1.8,
            "err_minus": 1.8,
            "type": "lens",
            "ref": "Birrer+2020",
            "all_sky": True,
        },
    ]
    return pd.DataFrame(events)


def get_all_sirens() -> pd.DataFrame:
    """
    Combine GW standard sirens and lensed quasar H0 measurements.

    Returns
    -------
    pandas.DataFrame
        All siren events with columns
        [name, ra, dec, H0, err_plus, err_minus, type, ref, all_sky].
    """
    df_gw = get_gw_events()
    df_lens = get_lens_events()
    df_all = pd.concat([df_gw, df_lens], ignore_index=True)
    return df_all


def assign_sirens_to_sectors(
    df_sirens: pd.DataFrame, normals: np.ndarray
) -> pd.DataFrame:
    """
    Assign each localized siren/lens event to a dodecahedron sector.

    Events flagged ``all_sky=True`` (combined constraints without sky
    localization) are not assigned to any sector: their ``sector_id`` is
    set to None so that they are excluded from all sector-based analyses.

    Parameters
    ----------
    df_sirens : pandas.DataFrame
        Siren events with 'ra', 'dec' and (optionally) 'all_sky' columns.
    normals : numpy.ndarray
        Dodecahedron face normals, shape (12, 3).

    Returns
    -------
    pandas.DataFrame
        Copy of df_sirens with added 'sector_id' column (int for
        localized events, None for all-sky events).
    """
    df = df_sirens.copy()
    if "all_sky" in df.columns:
        all_sky = df["all_sky"].astype(bool).values
    else:
        all_sky = np.zeros(len(df), dtype=bool)

    sectors = assign_sectors(df["ra"].values, df["dec"].values, normals)
    df["sector_id"] = [None if a else int(s) for a, s in zip(all_sky, sectors)]
    return df


def fit_h0_from_sirens(df_sirens: pd.DataFrame, normals: np.ndarray) -> list[dict]:
    """
    Compute weighted mean H0 per dodecahedron sector from sirens/lenses.

    Uses inverse-variance weighting: weight = 1 / sigma^2, where
    sigma = (err_plus + err_minus) / 2.  Events without a sector
    assignment (all-sky combined constraints, sector_id None) are
    skipped.

    Parameters
    ----------
    df_sirens : pandas.DataFrame
        Siren events with 'H0', 'err_plus', 'err_minus', 'sector_id'.
    normals : numpy.ndarray
        Dodecahedron normals (unused, kept for API compatibility).

    Returns
    -------
    list of dict
        One entry per sector with keys
        [sector_id, n_events, H0, err, events_list].
    """
    df = df_sirens[df_sirens["sector_id"].notna()].copy()
    df["sigma"] = (df["err_plus"] + df["err_minus"]) / 2.0
    df["weight"] = 1.0 / (df["sigma"] ** 2)

    results = []
    for sector in range(12):
        mask = df["sector_id"] == sector
        n = int(mask.sum())
        if n == 0:
            results.append(
                {
                    "sector_id": sector,
                    "n_events": 0,
                    "H0": np.nan,
                    "err": np.nan,
                    "events_list": [],
                }
            )
            continue

        sub = df[mask]
        weights = sub["weight"].values
        h0_vals = sub["H0"].values
        w_sum = weights.sum()
        h0_wmean = np.sum(weights * h0_vals) / w_sum
        h0_err = np.sqrt(1.0 / w_sum)

        results.append(
            {
                "sector_id": sector,
                "n_events": n,
                "H0": round(h0_wmean, 2),
                "err": round(h0_err, 2),
                "events_list": sub["name"].tolist(),
            }
        )

    return results


def compare_with_sne(
    sirens_h0_results: list[dict],
    sne_h0_vals: np.ndarray | None = None,
    sne_err_vals: np.ndarray | None = None,
) -> dict:
    """
    Compute Spearman correlation between H0 from sirens and H0 from SNe.

    Only sectors with at least one localized siren/lens event enter the
    correlation (all-sky combined constraints are excluded upstream in
    :func:`assign_sirens_to_sectors`).

    Parameters
    ----------
    sirens_h0_results : list of dict
        H0 per sector from sirens (output of :func:`fit_h0_from_sirens`).
    sne_h0_vals : array-like or None, optional
        H0 per sector from Pantheon+.  If None, loads 'h0_real' from
        outputs/mc_pantheon.npz.
    sne_err_vals : array-like or None, optional
        Per-sector H0 uncertainty for the SNe values.  If None and the
        MC file is available, the per-sector scatter of the isotropic
        mocks (std over mock realizations) is used as the uncertainty
        estimate.

    Returns
    -------
    dict
        Keys [rho, p_value, n_valid, sirens_h0, sirens_err, sne_h0,
        sne_err].
    """
    if sne_h0_vals is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mc_path = os.path.join(base_dir, "..", "outputs", "mc_pantheon.npz")
        if os.path.exists(mc_path):
            mc_data = dict(np.load(mc_path, allow_pickle=True))
            sne_h0_vals = mc_data["h0_real"]
            if sne_err_vals is None and "mock_h0_per_sector" in mc_data:
                with np.errstate(all="ignore"):
                    sne_err_vals = np.nanstd(mc_data["mock_h0_per_sector"], axis=0)
        else:
            sne_h0_vals = np.full(12, np.nan)

    sirens_h0 = np.array([r["H0"] for r in sirens_h0_results])
    sirens_err = np.array([r["err"] for r in sirens_h0_results])
    sne_h0 = np.asarray(sne_h0_vals, dtype=float)
    if sne_err_vals is None:
        sne_err = np.full(12, np.nan)
    else:
        sne_err = np.asarray(sne_err_vals, dtype=float)

    valid = ~np.isnan(sirens_h0) & ~np.isnan(sne_h0)
    n_valid = int(valid.sum())

    if n_valid < 3:
        return {
            "rho": np.nan,
            "p_value": np.nan,
            "n_valid": n_valid,
            "sirens_h0": sirens_h0,
            "sirens_err": sirens_err,
            "sne_h0": sne_h0,
            "sne_err": sne_err,
        }

    rho, p_val = spearmanr(sirens_h0[valid], sne_h0[valid])

    return {
        "rho": rho,
        "p_value": p_val,
        "n_valid": n_valid,
        "sirens_h0": sirens_h0,
        "sirens_err": sirens_err,
        "sne_h0": sne_h0,
        "sne_err": sne_err,
    }


def joint_significance(
    p_sne: float | None = None,
    p_cf4: float | None = None,
    p_sirens: float | None = None,
) -> dict:
    """
    Combine p-values from the three probes using Fisher's method.

    chi2_combined = -2 * sum(ln p_i); under the null hypothesis and for
    INDEPENDENT probes this follows a chi-squared distribution with
    df = 2k (k = number of probes).

    Caveat
    ------
    The probes combined here are NOT independent: both the CF4 and the
    sirens/lenses p-values are obtained by correlating against the same
    per-sector Pantheon+ H0 pattern that produces p_sne.  Fisher's method
    therefore overstates the joint evidence, and p_combined must be read
    as an UPPER BOUND on the significance (i.e. an optimistic, lower
    bound on the true combined p-value), quoted for reference only.

    Parameters
    ----------
    p_sne : float or None, optional
        p-value from SNe analysis.
    p_cf4 : float or None, optional
        p-value from CF4 peculiar velocities.
    p_sirens : float or None, optional
        p-value from sirens/lenses.

    Returns
    -------
    dict
        Keys [chi2_combined, df, p_combined, n_probes, p_values].
    """
    p_values = []
    labels = []

    if p_sne is not None and not np.isnan(p_sne):
        p_values.append(p_sne)
        labels.append("SNe")
    if p_cf4 is not None and not np.isnan(p_cf4):
        p_values.append(p_cf4)
        labels.append("CF4")
    if p_sirens is not None and not np.isnan(p_sirens):
        p_values.append(p_sirens)
        labels.append("Sirens")

    n_probes = len(p_values)
    if n_probes == 0:
        return {
            "chi2_combined": np.nan,
            "df": 0,
            "p_combined": np.nan,
            "n_probes": 0,
            "p_values": {},
        }

    p_values = np.array(p_values)
    p_values = np.clip(p_values, 1e-300, 1.0)
    chi2_combined = -2.0 * np.sum(np.log(p_values))
    df = 2 * n_probes
    p_combined = 1.0 - chi2.cdf(chi2_combined, df)

    return {
        "chi2_combined": chi2_combined,
        "df": df,
        "p_combined": p_combined,
        "n_probes": n_probes,
        "p_values": dict(zip(labels, p_values)),
    }


def plot_sirens_sky(
    df_sirens: pd.DataFrame, normals: np.ndarray, output_path: str
) -> None:
    """
    Mollweide projection showing localized siren/lens events.

    All-sky combined constraints (``all_sky=True``) have no sky position
    and are not plotted.

    Parameters
    ----------
    df_sirens : pandas.DataFrame
        Siren events.
    normals : numpy.ndarray
        Dodecahedron normals (currently unused in the plot).
    output_path : str
        Base path for output files (suffixes are appended).
    """
    if "all_sky" in df_sirens.columns:
        df_plot = df_sirens[~df_sirens["all_sky"].astype(bool)]
    else:
        df_plot = df_sirens

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)

    for _, row in df_plot.iterrows():
        marker = "s" if row["type"] == "GW" else "o"
        color = "darkorange" if row["type"] == "GW" else "steelblue"
        size = 120 if row["type"] == "GW" else 100
        ax.scatter(
            np.radians(row["ra"] - 180.0),
            np.radians(row["dec"]),
            c=color,
            marker=marker,
            s=size,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            row["name"],
            (np.radians(row["ra"] - 180.0), np.radians(row["dec"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6,
            alpha=0.8,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="darkorange",
            markersize=8,
            label="GW Sirens",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="steelblue",
            markersize=8,
            label="Lensed QSOs",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_title("Standard Sirens & Lensed Quasars on the Sky", fontsize=13, pad=10)
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    ax.set_xticklabels([f"{t}°" for t in tick_labels], fontsize=8)

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_sirens_sky.{fmt}")
    plt.close(fig)


def plot_sirens_vs_sne(corr_results: dict, output_path: str) -> None:
    """
    Scatter plot: H0 from sirens vs H0 from SNe per sector.

    Error bars show the real H0 uncertainties: the inverse-variance
    error of the siren sector average (y) and the per-sector SNe H0
    uncertainty (x), when available.

    Parameters
    ----------
    corr_results : dict
        Output of :func:`compare_with_sne`.
    output_path : str
        Base path for output files (suffixes are appended).
    """
    sirens_h0 = corr_results["sirens_h0"]
    sirens_err = corr_results["sirens_err"]
    sne_h0 = corr_results["sne_h0"]
    sne_err = corr_results["sne_err"]
    rho = corr_results["rho"]
    p_val = corr_results["p_value"]

    valid = ~np.isnan(sirens_h0) & ~np.isnan(sne_h0)

    fig, ax = plt.subplots(figsize=(8, 7))

    for i in range(12):
        if valid[i]:
            yerr_i = sirens_err[i] if np.isfinite(sirens_err[i]) else 0.0
            xerr_i = sne_err[i] if np.isfinite(sne_err[i]) else 0.0
            ax.errorbar(
                sne_h0[i],
                sirens_h0[i],
                yerr=yerr_i,
                xerr=xerr_i,
                fmt="o",
                color=f"C{i}",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                ecolor="gray",
                elinewidth=1.0,
                capsize=2,
                zorder=5,
            )
            ax.annotate(
                f"F{i+1}",
                (sne_h0[i], sirens_h0[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    if valid.sum() >= 3:
        coeffs = np.polyfit(sne_h0[valid], sirens_h0[valid], 1)
        x_fit = np.linspace(sne_h0[valid].min(), sne_h0[valid].max(), 50)
        ax.plot(
            x_fit,
            np.polyval(coeffs, x_fit),
            "--",
            color="gray",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlabel(r"$H_0$ from Pantheon+ SNe [km s$^{-1}$ Mpc$^{-1}$]", fontsize=12)
    ax.set_ylabel(r"$H_0$ from Sirens/Lenses [km s$^{-1}$ Mpc$^{-1}$]", fontsize=12)
    ax.set_title(
        f"Sirens vs SNe: Spearman ρ = {rho:.3f}, p = {p_val:.3f}"
        f"  (N_valid = {corr_results['n_valid']})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_sirens_vs_sne.{fmt}")
    plt.close(fig)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  STANDARD SIRENS & LENSED QUASARS")
    print("  Dodecahedral H0 Anisotropy Cross-Check")
    print("=" * 72)

    # --- Load sirens ---
    print("\n[1/5] Loading siren and lens events...")
    df_sirens = get_all_sirens()
    n_all_sky = int(df_sirens["all_sky"].sum())
    print(
        f"      Total events: {len(df_sirens)} "
        f"({n_all_sky} all-sky combined, excluded from sector analysis)"
    )
    print(f"      GW sirens:    {(df_sirens['type'] == 'GW').sum()}")
    print(f"      Lensed QSOs:  {(df_sirens['type'] == 'lens').sum()}")
    print(
        f"\n      {'Name':<20s} {'Type':<6s} {'RA':>8s} {'Dec':>8s} "
        f"{'H0':>8s} {'+err':>8s} {'-err':>8s}"
    )
    print(f"      {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, r in df_sirens[~df_sirens["all_sky"]].iterrows():
        print(
            f"      {r['name']:<20s} {r['type']:<6s} {r['ra']:8.2f} "
            f"{r['dec']:8.2f} {r['H0']:8.1f} {r['err_plus']:8.1f} "
            f"{r['err_minus']:8.1f}"
        )

    print(
        "\n      All-sky combined constraints (reference only, " "no sky localization):"
    )
    for _, r in df_sirens[df_sirens["all_sky"]].iterrows():
        print(
            f"        {r['name']:<20s} H0 = {r['H0']:.1f} "
            f"+{r['err_plus']:.1f} / -{r['err_minus']:.1f}  ({r['ref']})"
        )

    # --- Assign sectors ---
    print("\n[2/5] Assigning localized sirens to dodecahedron sectors...")
    normals = get_dodecahedron_normals()
    df_sirens = assign_sirens_to_sectors(df_sirens, normals)

    print(f"\n      {'Sector':>6s}  {'Events':>8s}")
    print(f"      {'-'*6}  {'-'*8}")
    for s in range(12):
        n = (df_sirens["sector_id"] == s).sum()
        names = df_sirens[df_sirens["sector_id"] == s]["name"].tolist()
        print(f"      {s:6d}  {n:8d}  {', '.join(names) if names else '--'}")

    # --- Fit H0 per sector ---
    print("\n[3/5] Computing weighted H0 per sector...")
    sirens_h0 = fit_h0_from_sirens(df_sirens, normals)

    print(
        f"\n      {'Sector':>6s}  {'N':>3s}  {'H0_sirens':>10s}  "
        f"{'Err':>8s}  {'Events':>30s}"
    )
    print(f"      {'-'*6}  {'-'*3}  {'-'*10}  {'-'*8}  {'-'*30}")
    for r in sirens_h0:
        if r["n_events"] > 0:
            print(
                f"      {r['sector_id']:6d}  {r['n_events']:3d}  "
                f"{r['H0']:10.2f}  {r['err']:8.2f}  "
                f"{', '.join(r['events_list']):30s}"
            )
        else:
            print(
                f"      {r['sector_id']:6d}  {r['n_events']:3d}  "
                f"{'--':>10s}  {'--':>8s}  {'--':>30s}"
            )

    # --- Compare with SNe (localized events only) ---
    print("\n[4/5] Comparing sirens H0 with Pantheon+ SNe H0...")
    print("      (all-sky combined constraints excluded)")
    corr_results = compare_with_sne(sirens_h0)
    print(f"      Valid sectors: {corr_results['n_valid']}/12")
    print(f"      Spearman ρ = {corr_results['rho']:.4f}")
    print(f"      p-value     = {corr_results['p_value']:.4f}")

    if corr_results["p_value"] < 0.05 and not np.isnan(corr_results["p_value"]):
        print("      → SIGNIFICANT correlation!")
    else:
        print("      → No significant correlation")

    # --- Joint significance ---
    print("\n[5/5] Joint significance (Fisher's method)...")

    mc_path = os.path.join(output_dir, "mc_pantheon.npz")
    p_sne = None
    if os.path.exists(mc_path):
        mc_data = dict(np.load(mc_path, allow_pickle=True))
        p_sne = float(mc_data.get("p_value", np.nan))

    cf4_path = os.path.join(output_dir, "cf4_results.npz")
    p_cf4 = None
    if os.path.exists(cf4_path):
        cf4_data = dict(np.load(cf4_path, allow_pickle=True))
        p_cf4 = float(cf4_data.get("mc_p_value", np.nan))

    p_sirens = corr_results["p_value"]

    joint = joint_significance(p_sne=p_sne, p_cf4=p_cf4, p_sirens=p_sirens)

    print(f"      Probes included: {joint['n_probes']}")
    for label, p in joint["p_values"].items():
        print(f"        {label}: p = {p:.4f}")
    print(f"      χ²_combined = {joint['chi2_combined']:.2f}")
    print(f"      df = {joint['df']}")
    print(f"      p_combined  = {joint['p_combined']:.6f}")
    print(
        "      CAVEAT: the probes are NOT independent (CF4 and sirens "
        "correlate against the same Pantheon+ sector pattern);"
    )
    print(
        "      p_combined is an upper bound on the significance and is "
        "quoted for reference only."
    )

    if joint["p_combined"] < 0.01:
        print(
            "      → JOINTLY SIGNIFICANT at α = 0.01 "
            "(subject to independence caveat)"
        )
    elif joint["p_combined"] < 0.05:
        print(
            "      → JOINTLY MARGINALLY SIGNIFICANT at α = 0.05 "
            "(subject to independence caveat)"
        )
    else:
        print("      → NOT jointly significant")

    # --- Visualization ---
    print("\nGenerating figures...")
    fig_base = os.path.join(output_dir, "fig")

    plot_sirens_sky(df_sirens, normals, fig_base)
    print(f"  Saved: {fig_base}_sirens_sky.[pdf,png]")

    plot_sirens_vs_sne(corr_results, fig_base)
    print(f"  Saved: {fig_base}_sirens_vs_sne.[pdf,png]")

    # --- Save results ---
    np.savez(
        os.path.join(output_dir, "sirens_results.npz"),
        sirens_h0=np.array([r["H0"] for r in sirens_h0]),
        sirens_err=np.array([r["err"] for r in sirens_h0]),
        spearman_rho=corr_results["rho"],
        spearman_p=corr_results["p_value"],
        joint_chi2=joint["chi2_combined"],
        joint_df=joint["df"],
        joint_p=joint["p_combined"],
    )
    print(f"\nResults saved to: " f"{os.path.join(output_dir, 'sirens_results.npz')}")
    print("=" * 72)
