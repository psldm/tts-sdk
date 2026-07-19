#!/usr/bin/env python3
"""
Main analysis pipeline for dodecahedral H0 anisotropy.

Runs the complete workflow:
  1. Data loading and sector assignment
  2. H0 fitting per sector
  3. Monte Carlo significance tests
  4. Redshift cut stability tests
  5. Visualization
  6. Summary output

Usage:
  python run_all.py --catalog pantheon    # Pantheon+ only (default)
  python run_all.py --catalog union3      # Union3 only
  python run_all.py --catalog all         # Both catalogs
"""

import sys
import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dodecahedron import (
    get_dodecahedron_normals,
    assign_sectors,
    load_pantheon_data,
    load_union3_data,
)
from h0_fit import (
    load_covariance_matrix,
    fit_all_sectors,
)
from simulations import (
    run_monte_carlo,
    compute_pvalue,
)
from visualize import (
    plot_mollweide,
    plot_h0_bars,
    plot_mc_distribution,
    plot_z_cut,
    create_summary_figure,
    plot_h0_comparison,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_MOCKS = 1000
H0_TRUE = 70.0
RANDOM_SEED = 42

summary_lines = []


def log(msg, also_print=True):
    if also_print:
        print(msg)
    summary_lines.append(msg)


def section(title):
    sep = "=" * 72
    log(f"\n{sep}")
    log(f"  {title}")
    log(sep)


def run_z_cut_test(df, cov_sub, normals, n_steps=10):
    z_all = df["z"].values
    z_min = max(0.02, z_all.min())
    z_max = z_all.max()
    z_cuts = np.linspace(z_min, z_max, n_steps)
    deltas, n_sne_list = [], []
    for zc in z_cuts:
        mask = z_all <= zc
        indices = np.where(mask)[0]
        n_sne_list.append(len(indices))
        if len(indices) < 24:
            deltas.append(np.nan)
            continue
        df_sub = df.iloc[indices].reset_index(drop=True)
        cov_z = cov_sub[np.ix_(indices, indices)]
        sector_ids_z = assign_sectors(
            df_sub["ra"].values, df_sub["dec"].values, normals
        )
        results_z = fit_all_sectors(df_sub, sector_ids_z, cov_z)
        valid = [r["H0"] for r in results_z if r["n_sne"] > 0 and not np.isnan(r["H0"])]
        deltas.append(
            float(np.max(valid) - np.min(valid)) if len(valid) >= 2 else np.nan
        )
    return {"z_cuts": z_cuts, "deltas": np.array(deltas), "n_sne": np.array(n_sne_list)}


def run_catalog(
    catalog_name, df, cov_sub, normals, fig_prefix, mc_prefix, summary_prefix
):
    """Run the full pipeline for a single catalog."""
    t_start = time.time()

    log(f"\n{'#'*72}")
    log(f"  CATALOG: {catalog_name}")
    log(f"{'#'*72}")

    section("STEP 1: SECTOR ASSIGNMENT")
    sector_ids = assign_sectors(df["ra"].values, df["dec"].values, normals)
    unique, counts = np.unique(sector_ids, return_counts=True)
    for s, c in zip(unique, counts):
        log(f"  Sector {s:2d}: {c:3d} SNe")

    section("STEP 2: H0 FITTING PER SECTOR")
    h0_results = fit_all_sectors(df, sector_ids, cov_sub)
    log(
        f"\n  {'Sector':>6s}  {'N_SNe':>5s}  {'H0':>8s}  {'-1σ':>8s}  {'+1σ':>8s}  {'χ²_min':>8s}"
    )
    log("  " + "-" * 58)
    for r in h0_results:
        if r["n_sne"] > 0:
            log(
                f"  {r['sector_id']:6d}  {r['n_sne']:5d}  "
                f"{r['H0']:8.2f}  {r['err_low']:8.2f}  {r['err_high']:8.2f}  "
                f"{r['chi2']:8.2f}"
            )

    valid = [r for r in h0_results if r["n_sne"] > 0 and not np.isnan(r["H0"])]
    h0_vals = np.array([r["H0"] for r in valid])
    h0_mean = np.mean(h0_vals)
    h0_std = np.std(h0_vals, ddof=1)
    h0_max = np.max(h0_vals)
    h0_min = np.min(h0_vals)
    observed_delta = float(h0_max - h0_min)
    epsilon = observed_delta / (2 * h0_mean)

    log(f"\n  Mean H0:           {h0_mean:.2f} km/s/Mpc")
    log(f"  Std dev (sectors): {h0_std:.2f} km/s/Mpc")
    log(
        f"  Max H0:            {h0_max:.2f} km/s/Mpc  (Sector {valid[np.argmax(h0_vals)]['sector_id']})"
    )
    log(
        f"  Min H0:            {h0_min:.2f} km/s/Mpc  (Sector {valid[np.argmin(h0_vals)]['sector_id']})"
    )
    log(f"  Delta H0:          {observed_delta:.2f} km/s/Mpc")
    log(f"  Modulation ε:      {epsilon:.4f}  ({epsilon*100:.1f}%)")

    section("STEP 3: MONTE CARLO SIGNIFICANCE TEST")
    log(f"Running {N_MOCKS} Monte Carlo simulations...")
    mc_results = run_monte_carlo(
        df,
        cov_sub,
        normals,
        n_mocks=N_MOCKS,
        H0_true=H0_TRUE,
        n_jobs=-1,
        random_seed=RANDOM_SEED,
    )
    log(f"  Valid mocks: {mc_results['n_valid']}/{mc_results['n_mocks']}")
    pval = compute_pvalue(observed_delta, mc_results["delta_H0"])
    log(f"\n  Observed delta_H0:        {observed_delta:.2f} km/s/Mpc")
    log(f"  Mock mean delta_H0:        {pval['mock_mean']:.2f} km/s/Mpc")
    log(f"  Mock std delta_H0:         {pval['mock_std']:.2f} km/s/Mpc")
    log(f"  Z-score:                   {pval['z_score']:.2f}σ")
    log(f"  P-value (one-sided):       {pval['p_value']:.6f}")
    log(f"  P-value (two-sided):       {pval['p_value_two_sided']:.6f}")
    sig = (
        "SIGNIFICANT"
        if pval["p_value"] < 0.01
        else ("MARGINALLY SIGNIFICANT" if pval["p_value"] < 0.05 else "NOT SIGNIFICANT")
    )
    log(f"  Conclusion: {sig} at α = 0.05")

    np.savez(
        mc_prefix + ".npz",
        observed_delta=observed_delta,
        observed_max=h0_max,
        observed_min=h0_min,
        h0_real=h0_vals,
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
        H0_true=H0_TRUE,
    )
    log(f"\n  MC results saved to: {mc_prefix}.npz")

    section("STEP 4: REDSHIFT CUT STABILITY")
    z_cut_results = run_z_cut_test(df, cov_sub, normals, n_steps=10)
    log(f"\n  {'z_max':>8s}  {'N_SNe':>6s}  {'Delta_H0':>10s}")
    log("  " + "-" * 30)
    for zc, ns, d in zip(
        z_cut_results["z_cuts"], z_cut_results["n_sne"], z_cut_results["deltas"]
    ):
        log(f"  {zc:8.4f}  {ns:6d}  {'--' if np.isnan(d) else f'{d:10.2f}'}")

    section("STEP 5: VISUALIZATION")
    log("Generating Mollweide sky map...")
    plot_mollweide(df, sector_ids, h0_results, normals, fig_prefix)
    log(f"  Saved: {fig_prefix}_mollweide.[pdf,png]")
    log("Generating H0 bar chart...")
    plot_h0_bars(h0_results, fig_prefix)
    log(f"  Saved: {fig_prefix}_h0_bars.[pdf,png]")
    log("Generating MC distribution...")
    plot_mc_distribution(
        mc_results["delta_H0"],
        observed_delta,
        pval["p_value"],
        pval["z_score"],
        fig_prefix,
    )
    log(f"  Saved: {fig_prefix}_mc_dist.[pdf,png]")
    log("Generating z-cut stability plot...")
    plot_z_cut(
        z_cut_results["z_cuts"],
        z_cut_results["deltas"],
        z_cut_results["n_sne"],
        fig_prefix,
    )
    log(f"  Saved: {fig_prefix}_zcut.[pdf,png]")
    log("Generating summary figure...")
    mc_data = dict(np.load(mc_prefix + ".npz", allow_pickle=True))
    create_summary_figure(
        df, sector_ids, h0_results, mc_data, None, z_cut_results, fig_prefix
    )
    log(f"  Saved: {fig_prefix}_summary.[pdf,png]")

    t_elapsed = time.time() - t_start
    log(f"\n  [{catalog_name}] Runtime: {t_elapsed:.1f} seconds")

    return {
        "h0_results": h0_results,
        "h0_vals": h0_vals,
        "observed_delta": observed_delta,
        "pval": pval,
        "sig": sig,
        "h0_mean": h0_mean,
        "epsilon": epsilon,
    }


def main():
    parser = argparse.ArgumentParser(description="Dodecahedral H0 Anisotropy Pipeline")
    parser.add_argument(
        "--catalog",
        choices=["pantheon", "union3", "all"],
        default="pantheon",
        help="Catalog to use (default: pantheon)",
    )
    args = parser.parse_args()

    t_start = time.time()
    log("DODECAHEDRAL H0 ANISOTROPY — FULL ANALYSIS PIPELINE")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Catalog: {args.catalog}")
    log(f"Output directory: {OUTPUT_DIR}")

    normals = get_dodecahedron_normals()
    log(f"Generated {len(normals)} dodecahedron face normals")

    results = {}

    # --- Pantheon+ ---
    if args.catalog in ("pantheon", "all"):
        data_path = os.path.join(DATA_DIR, "Pantheon+SH0ES.dat")
        cov_path = os.path.join(DATA_DIR, "Pantheon+SH0ES_STAT+SYS.cov")

        log("\nLoading Pantheon+ catalog...")
        df_p = load_pantheon_data(data_path)
        log("  Total SNe in file: 1701")
        log(f"  SNe with z < 0.1:  {len(df_p)}")
        log(f"  z range: [{df_p['z'].min():.4f}, {df_p['z'].max():.4f}]")

        cov_full = load_covariance_matrix(cov_path)
        if cov_full is not None:
            df_full = pd.read_csv(data_path, sep=r"\s+")
            mask_z = df_full["zHD"] < 0.1
            orig_indices = np.where(mask_z)[0]
            cov_p = cov_full[np.ix_(orig_indices, orig_indices)]
            log(f"  Covariance: {cov_p.shape}")
        else:
            cov_p = np.eye(len(df_p))
            log(f"  Using identity matrix: {cov_p.shape}")

        results["pantheon"] = run_catalog(
            "Pantheon+",
            df_p,
            cov_p,
            normals,
            os.path.join(OUTPUT_DIR, "fig_pantheon"),
            os.path.join(OUTPUT_DIR, "mc_pantheon"),
            os.path.join(OUTPUT_DIR, "summary_pantheon"),
        )

    # --- Union3 ---
    if args.catalog in ("union3", "all"):
        union3_path = os.path.join(DATA_DIR, "union3_inputs.pickle")

        log("\nLoading Union3 catalog...")
        df_u = load_union3_data(union3_path)
        log("  Total SNe in file: 2087")
        log(f"  SNe with z < 0.1:  {len(df_u)}")
        log(f"  z range: [{df_u['z'].min():.4f}, {df_u['z'].max():.4f}]")

        cov_u = np.eye(len(df_u))
        log(f"  Using diagonal covariance: {cov_u.shape}")

        results["union3"] = run_catalog(
            "Union3",
            df_u,
            cov_u,
            normals,
            os.path.join(OUTPUT_DIR, "fig_union3"),
            os.path.join(OUTPUT_DIR, "mc_union3"),
            os.path.join(OUTPUT_DIR, "summary_union3"),
        )

    # --- Comparison ---
    if args.catalog == "all" and "pantheon" in results and "union3" in results:
        section("CATALOG COMPARISON")
        h0p = results["pantheon"]["h0_vals"]
        h0u = results["union3"]["h0_vals"]
        valid = ~np.isnan(h0p) & ~np.isnan(h0u)
        if valid.sum() >= 3:
            r = np.corrcoef(h0p[valid], h0u[valid])[0, 1]
            log(f"  Pearson r between Pantheon+ and Union3 H0: {r:.4f}")
        else:
            r = np.nan
            log("  Not enough valid sectors for correlation")

        log(f"\n  {'Metric':>25s}  {'Pantheon+':>12s}  {'Union3':>12s}")
        log(f"  {'-'*25}  {'-'*12}  {'-'*12}")
        for key in ["h0_mean", "observed_delta", "epsilon"]:
            vp = results["pantheon"][key]
            vu = results["union3"][key]
            log(f"  {key:>25s}  {vp:12.2f}  {vu:12.2f}")
        log(
            f"  {'p-value':>25s}  {results['pantheon']['pval']['p_value']:12.4f}  {results['union3']['pval']['p_value']:12.4f}"
        )
        log(
            f"  {'Conclusion':>25s}  {results['pantheon']['sig']:>12s}  {results['union3']['sig']:>12s}"
        )

        log("\nGenerating comparison plot...")
        plot_h0_comparison(
            results["pantheon"]["h0_results"],
            results["union3"]["h0_results"],
            os.path.join(OUTPUT_DIR, "fig"),
        )
        log(f"  Saved: {os.path.join(OUTPUT_DIR, 'fig')}_h0_comparison.[pdf,png]")

    # --- Final ---
    section("FINAL SUMMARY")
    t_elapsed = time.time() - t_start
    log(f"Total runtime: {t_elapsed:.1f} seconds")
    log("=" * 72)

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{args.catalog}.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    log(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
