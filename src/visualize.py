#!/usr/bin/env python3
"""
Visualization module for dodecahedral H0 anisotropy analysis.

Produces publication-quality figures: Mollweide sky maps, H0 bar charts,
Monte Carlo distributions, rotation tests, and z-cut diagnostics.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dodecahedron import get_dodecahedron_normals, assign_sectors

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.3)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

PLANCK_H0 = 67.4
SH0ES_H0 = 73.0
CMAP_H0 = "RdYlBu_r"


def _ra_to_mollweide(ra_deg):
    """Convert RA [0, 360] to Mollweide x-coordinate [-180, 180]."""
    ra = np.asarray(ra_deg, dtype=float)
    ra_m = ra - 180.0
    ra_m[ra_m > 180] -= 360.0
    return ra_m


def _compute_sector_boundaries(normals, n_grid=300):
    """
    Compute sector boundary paths for Mollweide projection.

    Samples a dense grid on the sphere, assigns each point to the closest
    normal, then extracts contour lines at sector boundaries.

    Parameters
    ----------
    normals : numpy.ndarray
        Dodecahedron face normals, shape (12, 3).
    n_grid : int, optional
        Grid resolution per dimension.

    Returns
    -------
    list of numpy.ndarray
        Each element is an (N, 2) array of (ra_m, dec) points forming a
        boundary segment.
    """
    ra_grid = np.linspace(0, 360, n_grid * 2, endpoint=False)
    dec_grid = np.linspace(-90, 90, n_grid)
    RA, DEC = np.meshgrid(ra_grid, dec_grid)
    ra_flat = RA.ravel()
    dec_flat = DEC.ravel()

    sector_grid = assign_sectors(ra_flat, dec_flat, normals).reshape(RA.shape)

    ra_m = _ra_to_mollweide(ra_grid)
    RA_M, _ = np.meshgrid(ra_m, dec_grid)

    boundaries = []
    for i in range(12):
        for j in range(i + 1, 12):
            mask_i = (sector_grid == i).astype(float)
            mask_j = (sector_grid == j).astype(float)

            boundary_mask = np.zeros_like(sector_grid, dtype=bool)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    shifted_j = np.roll(np.roll(mask_j, di, axis=1), dj, axis=0)
                    boundary_mask |= (mask_i > 0.5) & (shifted_j > 0.5)

            if not boundary_mask.any():
                continue

            try:
                cs = plt.contour(
                    RA_M, DEC, boundary_mask.astype(float), levels=[0.5], colors="none"
                )
                plt.close()
                # matplotlib >= 3.8: ContourSet has no .collections;
                # extract segments from allsegs instead.
                for level_segs in cs.allsegs:
                    for seg in level_segs:
                        if len(seg) > 2:
                            boundaries.append(np.asarray(seg))
            except Exception:
                pass

    return boundaries


def plot_mollweide(
    df: pd.DataFrame,
    sector_ids: np.ndarray,
    h0_results: list,
    normals: np.ndarray,
    output_path: str,
) -> None:
    """
    Mollweide projection of the celestial sphere with SNe colored by H0.

    Parameters
    ----------
    df : pandas.DataFrame
        SNe data with columns ['ra', 'dec'].
    sector_ids : numpy.ndarray
        Sector assignment for each SN.
    h0_results : list of dict
        H0 fit results per sector.
    normals : numpy.ndarray
        Dodecahedron face normals, shape (12, 3).
    output_path : str
        Base path for output (extensions appended).
    """
    h0_map = {r["sector_id"]: r["H0"] for r in h0_results if r["n_sne"] > 0}
    h0_vals = np.array([h0_map.get(s, np.nan) for s in sector_ids])

    ra_m = _ra_to_mollweide(df["ra"].values)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    sc = ax.scatter(
        np.radians(ra_m),
        np.radians(df["dec"].values),
        c=h0_vals,
        cmap=CMAP_H0,
        s=6,
        alpha=0.7,
        edgecolors="none",
        vmin=np.nanmin(h0_vals),
        vmax=np.nanmax(h0_vals),
    )

    boundaries = _compute_sector_boundaries(normals)
    for b in boundaries:
        ax.plot(
            np.radians(b[:, 0]),
            np.radians(b[:, 1]),
            color="black",
            linewidth=0.6,
            alpha=0.5,
        )

    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.75)
    cbar.set_label(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=12)

    ax.set_xlabel("RA", fontsize=12)
    ax.set_ylabel("Dec", fontsize=12)
    ax.set_title("Pantheon+ SNe (z < 0.1) — Dodecahedron Sectors", fontsize=14, pad=12)

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    ax.set_xticklabels([f"{t}°" for t in tick_labels], fontsize=9)

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_mollweide.{fmt}")
    plt.close(fig)


def plot_h0_bars(h0_results: list, output_path: str) -> None:
    """
    Bar chart of H0 per sector with error bars and reference lines.

    Parameters
    ----------
    h0_results : list of dict
        H0 fit results per sector.
    output_path : str
        Base path for output.
    """
    sectors = np.array([r["sector_id"] for r in h0_results if r["n_sne"] > 0])
    h0_vals = np.array([r["H0"] for r in h0_results if r["n_sne"] > 0])
    err_low = np.array([r["err_low"] for r in h0_results if r["n_sne"] > 0])
    err_high = np.array([r["err_high"] for r in h0_results if r["n_sne"] > 0])
    n_sne = np.array([r["n_sne"] for r in h0_results if r["n_sne"] > 0])
    h0_mean = np.mean(h0_vals)

    sort_idx = np.argsort(sectors)
    sectors = sectors[sort_idx]
    h0_vals = h0_vals[sort_idx]
    err_low = err_low[sort_idx]
    err_high = err_high[sort_idx]
    n_sne = n_sne[sort_idx]

    colors = plt.cm.RdYlBu_r(
        (h0_vals - h0_vals.min()) / (h0_vals.max() - h0_vals.min() + 1e-10)
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Use actual sector ids of non-empty sectors as x positions, so that
    # empty sectors do not shift the bars.
    ax.bar(
        sectors,
        h0_vals,
        yerr=[err_low, err_high],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"linewidth": 1.2},
    )

    ax.axhline(
        y=PLANCK_H0,
        color="blue",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"Planck: {PLANCK_H0}",
    )
    ax.axhline(
        y=SH0ES_H0,
        color="green",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"SH0ES: {SH0ES_H0}",
    )
    ax.axhline(
        y=h0_mean,
        color="red",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label=f"Mean: {h0_mean:.1f}",
    )

    for i, (s, n) in enumerate(zip(sectors, n_sne)):
        ax.text(
            s,
            h0_vals[i] + err_high[i] + 0.15,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="gray",
        )

    ax.set_xticks(sectors)
    ax.set_xticklabels([f"Face {s+1}" for s in sectors], rotation=45, ha="right")
    ax.set_ylabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=13)
    ax.set_xlabel("Dodecahedron Sector", fontsize=13)
    ax.set_title("H₀ per Dodecahedron Sector", fontsize=14)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.set_ylim(h0_vals.min() - 1.5, h0_vals.max() + 1.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_h0_bars.{fmt}")
    plt.close(fig)


def plot_mc_distribution(
    mock_deltas: np.ndarray,
    observed_delta: float,
    p_value: float,
    z_score: float,
    output_path: str,
) -> None:
    """
    Histogram of Monte Carlo delta_H0 distribution with observed value.

    Parameters
    ----------
    mock_deltas : numpy.ndarray
        Array of delta_H0 from MC simulations.
    observed_delta : float
        Observed delta_H0.
    p_value : float
        One-sided p-value.
    z_score : float
        Z-score.
    output_path : str
        Base path for output.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, max(mock_deltas.max(), observed_delta) * 1.15, 40)
    ax.hist(
        mock_deltas,
        bins=bins,
        color="steelblue",
        edgecolor="white",
        alpha=0.7,
        label=f"MC mocks (n={len(mock_deltas)})",
    )

    tail_mask = mock_deltas >= observed_delta
    ax.hist(
        mock_deltas[tail_mask],
        bins=bins,
        color="crimson",
        edgecolor="white",
        alpha=0.8,
        label=f"≥ observed ({np.sum(tail_mask)} mocks)",
    )

    ax.axvline(
        x=observed_delta,
        color="red",
        linewidth=2.5,
        linestyle="-",
        label=f"Observed: {observed_delta:.2f}",
    )

    mock_mean = np.mean(mock_deltas)
    ax.axvline(
        x=mock_mean,
        color="gray",
        linewidth=1.5,
        linestyle="--",
        label=f"Mock mean: {mock_mean:.2f}",
    )

    textstr = (
        f"p-value (one-sided) = {p_value:.4f}\n"
        f"z-score = {z_score:.2f}σ\n"
        f"N$_{{\\rm mocks}}$ = {len(mock_deltas)}"
    )
    props = dict(
        boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"
    )
    ax.text(
        0.97,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    ax.set_xlabel(
        r"$\Delta H_0 = \max(H_0) - \min(H_0)$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=13
    )
    ax.set_ylabel("Number of Mock Realizations", fontsize=13)
    ax.set_title("Monte Carlo Null Distribution", fontsize=14)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_mc_dist.{fmt}")
    plt.close(fig)


def plot_rotation_test(
    rotation_deltas: np.ndarray,
    observed_delta: float,
    percentile: float,
    output_path: str,
) -> None:
    """
    Histogram of delta_H0 from random dodecahedron rotations.

    Parameters
    ----------
    rotation_deltas : numpy.ndarray
        Array of delta_H0 from rotations.
    observed_delta : float
        Observed delta_H0.
    percentile : float
        Percentile of observed among rotations.
    output_path : str
        Base path for output.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, max(rotation_deltas.max(), observed_delta) * 1.15, 35)
    ax.hist(
        rotation_deltas,
        bins=bins,
        color="teal",
        edgecolor="white",
        alpha=0.7,
        label=f"Rotations (n={len(rotation_deltas)})",
    )

    ax.axvline(
        x=observed_delta,
        color="red",
        linewidth=2.5,
        linestyle="-",
        label=f"Observed: {observed_delta:.2f}",
    )

    textstr = (
        f"Percentile = {percentile:.1f}%\nN$_{{\\rm rot}}$ = {len(rotation_deltas)}"
    )
    props = dict(
        boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"
    )
    ax.text(
        0.97,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    ax.set_xlabel(r"$\Delta H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=13)
    ax.set_ylabel("Number of Rotations", fontsize=13)
    ax.set_title("Random Dodecahedron Rotation Test", fontsize=14)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_rotation.{fmt}")
    plt.close(fig)


def plot_z_cut(
    z_cuts: np.ndarray, deltas: np.ndarray, n_sne: np.ndarray, output_path: str
) -> None:
    """
    Delta_H0 and N_total as functions of maximum redshift cut.

    Parameters
    ----------
    z_cuts : numpy.ndarray
        Array of z_max values.
    deltas : numpy.ndarray
        delta_H0 for each z_max.
    n_sne : numpy.ndarray
        Total number of SNe for each z_max.
    output_path : str
        Base path for output.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(z_cuts, deltas, "o-", color="darkblue", linewidth=1.5, markersize=5)
    ax1.axhline(
        y=deltas[-1],
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label=f"z_max=0.1: {deltas[-1]:.2f}",
    )
    ax1.set_ylabel(r"$\Delta H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=13)
    ax1.set_title(r"Stability of $\Delta H_0$ with Redshift Cut", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(z_cuts, n_sne, "s-", color="darkgreen", linewidth=1.5, markersize=5)
    ax2.set_xlabel(r"$z_{\rm max}$", fontsize=13)
    ax2.set_ylabel(r"$N_{\rm SNe}$", fontsize=13)
    ax2.grid(True, alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_zcut.{fmt}")
    plt.close(fig)


def create_summary_figure(
    df: pd.DataFrame,
    sector_ids: np.ndarray,
    h0_results: list,
    mc_results: dict | None,
    rotation_results: dict | None,
    z_cut_results: dict | None,
    output_path: str,
) -> None:
    """
    Assemble all plots into a single 3x2 summary figure.

    Parameters
    ----------
    df : pandas.DataFrame
        SNe data.
    sector_ids : numpy.ndarray
        Sector assignments.
    h0_results : list of dict
        H0 fit results.
    mc_results : dict or None
        Monte Carlo results dict.
    rotation_results : dict or None
        Rotation test results.
    z_cut_results : dict or None
        Z-cut results.
    output_path : str
        Base path for output.
    """
    normals = get_dodecahedron_normals()

    fig = plt.figure(figsize=(18, 14))

    # --- Panel 1: Mollweide ---
    ax1 = fig.add_subplot(2, 3, 1, projection="mollweide")
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)
    h0_map = {r["sector_id"]: r["H0"] for r in h0_results if r["n_sne"] > 0}
    h0_vals = np.array([h0_map.get(s, np.nan) for s in sector_ids])
    ra_m = _ra_to_mollweide(df["ra"].values)
    ax1.scatter(
        np.radians(ra_m),
        np.radians(df["dec"].values),
        c=h0_vals,
        cmap=CMAP_H0,
        s=3,
        alpha=0.6,
        edgecolors="none",
        vmin=np.nanmin(h0_vals),
        vmax=np.nanmax(h0_vals),
    )
    boundaries = _compute_sector_boundaries(normals, n_grid=150)
    for b in boundaries:
        ax1.plot(
            np.radians(b[:, 0]),
            np.radians(b[:, 1]),
            color="black",
            linewidth=0.4,
            alpha=0.4,
        )
    ax1.set_title("(a) Sky Map", fontsize=12, pad=8)
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    ax1.set_xticklabels([f"{t}°" for t in tick_labels], fontsize=7)

    # --- Panel 2: H0 bars ---
    ax2 = fig.add_subplot(2, 3, 2)
    sectors = np.array([r["sector_id"] for r in h0_results if r["n_sne"] > 0])
    h0v = np.array([r["H0"] for r in h0_results if r["n_sne"] > 0])
    el = np.array([r["err_low"] for r in h0_results if r["n_sne"] > 0])
    eh = np.array([r["err_high"] for r in h0_results if r["n_sne"] > 0])
    sort_idx = np.argsort(sectors)
    colors2 = plt.cm.RdYlBu_r((h0v - h0v.min()) / (h0v.max() - h0v.min() + 1e-10))
    # x positions are the actual sector ids of non-empty sectors.
    ax2.bar(
        sectors[sort_idx],
        h0v[sort_idx],
        yerr=[el[sort_idx], eh[sort_idx]],
        capsize=3,
        color=colors2[sort_idx],
        edgecolor="black",
        linewidth=0.6,
        error_kw={"linewidth": 0.8},
    )
    h0_mean = np.mean(h0v)
    ax2.axhline(y=PLANCK_H0, color="blue", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.axhline(y=SH0ES_H0, color="green", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.axhline(y=h0_mean, color="red", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.set_xticks(sectors[sort_idx])
    ax2.set_xticklabels([f"{s+1}" for s in sectors[sort_idx]], fontsize=8)
    ax2.set_ylabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=10)
    ax2.set_xlabel("Face", fontsize=10)
    ax2.set_title("(b) H₀ per Sector", fontsize=12)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Observed delta, needed by both the MC and rotation panels.
    obs_delta = np.max(h0v) - np.min(h0v)
    if mc_results is not None:
        obs_delta = mc_results.get("observed_delta", obs_delta)

    # --- Panel 3: MC distribution ---
    ax3 = fig.add_subplot(2, 3, 3)
    if mc_results is not None:
        mock_deltas = mc_results.get(
            "mock_delta_H0", mc_results.get("delta_H0", np.array([]))
        )
        pval = mc_results.get("p_value", 0)
        zsc = mc_results.get("z_score", 0)
        bins3 = np.linspace(0, max(mock_deltas.max(), obs_delta) * 1.15, 30)
        ax3.hist(
            mock_deltas, bins=bins3, color="steelblue", edgecolor="white", alpha=0.7
        )
        tail = mock_deltas >= obs_delta
        ax3.hist(
            mock_deltas[tail], bins=bins3, color="crimson", edgecolor="white", alpha=0.8
        )
        ax3.axvline(x=obs_delta, color="red", linewidth=2, linestyle="-")
        ax3.axvline(x=np.mean(mock_deltas), color="gray", linewidth=1, linestyle="--")
        ax3.text(
            0.97,
            0.95,
            f"p = {pval:.3f}\nz = {zsc:.1f}σ",
            transform=ax3.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    ax3.set_xlabel(r"$\Delta H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("(c) Monte Carlo", fontsize=12)

    # --- Panel 4: Rotation test ---
    ax4 = fig.add_subplot(2, 3, 4)
    if rotation_results is not None:
        rot_deltas = rotation_results.get("deltas", np.array([]))
        rot_pct = rotation_results.get("percentile", 50)
        if len(rot_deltas) > 0:
            bins4 = np.linspace(0, max(rot_deltas.max(), obs_delta) * 1.15, 25)
            ax4.hist(rot_deltas, bins=bins4, color="teal", edgecolor="white", alpha=0.7)
            ax4.axvline(x=obs_delta, color="red", linewidth=2, linestyle="-")
            ax4.text(
                0.97,
                0.95,
                f"Pct = {rot_pct:.1f}%",
                transform=ax4.transAxes,
                fontsize=9,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
    ax4.set_xlabel(r"$\Delta H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=10)
    ax4.set_ylabel("Count", fontsize=10)
    ax4.set_title("(d) Rotation Test", fontsize=12)

    # --- Panel 5: z-cut ---
    ax5 = fig.add_subplot(2, 3, 5)
    if z_cut_results is not None:
        zc = z_cut_results.get("z_cuts", np.array([]))
        zd = z_cut_results.get("deltas", np.array([]))
        if len(zc) > 0:
            ax5.plot(zc, zd, "o-", color="darkblue", linewidth=1.2, markersize=4)
            ax5.axhline(y=zd[-1], color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax5.set_xlabel(r"$z_{\rm max}$", fontsize=10)
    ax5.set_ylabel(r"$\Delta H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=10)
    ax5.set_title("(e) Redshift Cut Stability", fontsize=12)
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: text summary ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    lines = [
        "Dodecahedral H₀ Anisotropy",
        "Pantheon+ SNe (z < 0.1)",
        "",
        f"N$_{{\\rm SNe}}$ = {len(df)}",
        f"Mean H₀ = {h0_mean:.1f} km/s/Mpc",
        f"ΔH₀ = {np.max(h0v) - np.min(h0v):.2f} km/s/Mpc",
        f"ε = {(np.max(h0v) - np.min(h0v)) / (2 * h0_mean):.4f}",
    ]
    if mc_results is not None:
        lines.append(f"p-value = {mc_results.get('p_value', 0):.4f}")
        lines.append(f"z-score = {mc_results.get('z_score', 0):.1f}σ")
    for i, line in enumerate(lines):
        ax6.text(
            0.5,
            0.95 - i * 0.08,
            line,
            transform=ax6.transAxes,
            fontsize=11,
            ha="center",
            va="top",
            fontweight="bold" if i < 2 else "normal",
        )
    ax6.set_title("(f) Summary", fontsize=12)

    plt.tight_layout(pad=2.0)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_summary.{fmt}")
    plt.close(fig)


def plot_h0_comparison(h0_pantheon: list, h0_union3: list, output_path: str) -> None:
    """
    Compare H0 per sector between Pantheon+ and Union3 catalogs.

    Sectors are matched by sector_id, so catalogs with different sets of
    empty sectors remain correctly aligned.

    Parameters
    ----------
    h0_pantheon : list of dict
        H0 results for Pantheon+.
    h0_union3 : list of dict
        H0 results for Union3.
    output_path : str
        Base path for output.
    """
    p_map = {r["sector_id"]: r for r in h0_pantheon if r["n_sne"] > 0}
    u_map = {r["sector_id"]: r for r in h0_union3 if r["n_sne"] > 0}
    common = sorted(set(p_map) & set(u_map))

    h0p = np.array([p_map[s]["H0"] for s in common])
    h0u = np.array([u_map[s]["H0"] for s in common])
    ep_low = np.array([p_map[s]["err_low"] for s in common])
    ep_high = np.array([p_map[s]["err_high"] for s in common])
    eu_low = np.array([u_map[s]["err_low"] for s in common])
    eu_high = np.array([u_map[s]["err_high"] for s in common])

    valid = ~np.isnan(h0p) & ~np.isnan(h0u)
    if valid.sum() >= 3:
        corr = np.corrcoef(h0p[valid], h0u[valid])[0, 1]
    else:
        corr = np.nan

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(common))
    w = 0.35
    ax.bar(
        x - w / 2,
        h0p,
        w,
        yerr=[ep_low, ep_high],
        capsize=3,
        color="steelblue",
        edgecolor="black",
        linewidth=0.6,
        error_kw={"linewidth": 0.8},
        label="Pantheon+",
    )
    ax.bar(
        x + w / 2,
        h0u,
        w,
        yerr=[eu_low, eu_high],
        capsize=3,
        color="darkorange",
        edgecolor="black",
        linewidth=0.6,
        error_kw={"linewidth": 0.8},
        label="Union3",
    )

    ax.axhline(y=PLANCK_H0, color="blue", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(y=SH0ES_H0, color="green", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Face {s+1}" for s in common], rotation=45, ha="right")
    ax.set_ylabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=13)
    ax.set_xlabel("Dodecahedron Sector", fontsize=13)
    ax.set_title(f"H₀ Comparison: Pantheon+ vs Union3  (r = {corr:.3f})", fontsize=14)
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}_h0_comparison.{fmt}")
    plt.close(fig)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "Pantheon+SH0ES.dat")
    cov_path = os.path.join(base_dir, "..", "data", "Pantheon+SH0ES_STAT+SYS.cov")
    mc_path = os.path.join(base_dir, "..", "outputs", "mc_results.npz")
    output_dir = os.path.join(base_dir, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    from dodecahedron import load_pantheon_data
    from h0_fit import load_covariance_matrix, fit_all_sectors

    print("Loading data and computing results...")
    df = load_pantheon_data(data_path)
    normals = get_dodecahedron_normals()
    sector_ids = assign_sectors(df["ra"].values, df["dec"].values, normals)

    cov_full = load_covariance_matrix(cov_path)
    if cov_full is not None:
        df_full = pd.read_csv(data_path, sep=r"\s+")
        mask_z = df_full["zHD"] < 0.1
        orig_indices = np.where(mask_z)[0]
        cov_sub = cov_full[np.ix_(orig_indices, orig_indices)]
    else:
        cov_sub = np.eye(len(df))

    h0_results = fit_all_sectors(df, sector_ids, cov_sub)

    mc_data = None
    if os.path.exists(mc_path):
        mc_data = dict(np.load(mc_path, allow_pickle=True))

    print("Generating Mollweide map...")
    plot_mollweide(df, sector_ids, h0_results, normals, os.path.join(output_dir, "fig"))

    print("Generating H0 bar chart...")
    plot_h0_bars(h0_results, os.path.join(output_dir, "fig"))

    if mc_data is not None and "mock_delta_H0" in mc_data:
        print("Generating MC distribution...")
        plot_mc_distribution(
            mc_data["mock_delta_H0"],
            float(mc_data["observed_delta"]),
            float(mc_data["p_value"]),
            float(mc_data["z_score"]),
            os.path.join(output_dir, "fig"),
        )

    print("Generating summary figure...")
    create_summary_figure(
        df,
        sector_ids,
        h0_results,
        mc_data,
        None,
        None,
        os.path.join(output_dir, "fig"),
    )

    print(f"\nAll figures saved to: {output_dir}/")
