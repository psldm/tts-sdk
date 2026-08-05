"""A4/A5 wrapper: end-to-end matched-circle search using the working-tree module.

Usage: run_search.py {R2|R3}
  R2: twist_step=5.0   R3: twist_step=1.0
No repo source is edited; all logic is imported from src/matched_circles.py.
Replicates the module __main__ flow (same grids, same seeds: data map seed 42,
sim i seed 100+i) and adds: live pulse (<=60 s), sanity gates, per-pair
zero-cell fractions, and (R3) exact twist=36/324 tables.

Outputs: tts-sdk/outputs/matched_circles_{TAG}.npz + fig_{TAG}_* figures.
"""
import sys, os, time, datetime
import numpy as np

SRC = "/Users/air_eva/Desktop/tts_sdk/tts-sdk/src"
OUT = "/Users/air_eva/Desktop/tts_sdk/tts-sdk/outputs"
MAP = "/Users/air_eva/Desktop/tts_sdk/tts-sdk/data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
sys.path.insert(0, SRC)

import healpy as hp
from matched_circles import (
    load_or_generate_cmb_map, apply_cmb_mask, get_dodecahedron_axes,
    cartesian_to_lonlat, circle_correlation, N_CIRCLE_POINTS, TWIST_PDS,
    plot_correlation_vs_alpha, plot_correlation_vs_twist,
    plot_sim_distribution, plot_cmb_with_circles, compute_significance,
)

TAG = sys.argv[1] if len(sys.argv) > 1 else "R2"
TWIST_STEP = {"R2": 5.0, "R3": 1.0}[TAG]
NSIDE, GAL_CUT, N_SIM = 128, 20.0, 50
ALPHA_RANGE = np.arange(10, 51, 5)
TWISTS = np.arange(0, 360, TWIST_STEP)
N_TW = len(TWISTS)
TOTAL_ITERS = (1 + N_SIM) * 6 * len(ALPHA_RANGE) * N_TW

t_start = time.time()
state = {"done": 0, "run_max": -1.0, "last_pulse": 0.0}

def pulse(phase, pair, alpha, force=False):
    now = time.time()
    if not force and now - state["last_pulse"] < 55:
        return
    state["last_pulse"] = now
    elapsed = now - t_start
    rate = state["done"] / elapsed if elapsed > 0 else 0
    remaining = (TOTAL_ITERS - state["done"]) / rate if rate > 0 else float("nan")
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {TAG} {phase} | "
          f"pair {pair} | alpha={alpha} | iter {state['done']}/{TOTAL_ITERS} "
          f"({state['done']/TOTAL_ITERS*100:.1f}%) | run max r={state['run_max']:.4f} | "
          f"elapsed {elapsed:.0f}s | remaining ~{remaining:.0f}s", flush=True)

def scan_map(m, mask, axes, phase):
    """Full 6-pair x alpha x twist scan; returns list of per-pair dicts."""
    results = []
    for pidx in range(6):
        a1, a2 = axes[pidx, 0], axes[pidx, 1]
        corr_scan = np.zeros((len(ALPHA_RANGE), N_TW))
        for ia, alpha in enumerate(ALPHA_RANGE):
            for it, tw in enumerate(TWISTS):
                c = circle_correlation(m, a1, a2, float(alpha), float(tw),
                                       NSIDE, N_CIRCLE_POINTS, mask)
                corr_scan[ia, it] = c
                if c > state["run_max"]:
                    state["run_max"] = c
            state["done"] += N_TW
            pulse(phase, f"{pidx+1}/6", int(alpha))
        besti = np.unravel_index(np.argmax(corr_scan), corr_scan.shape)
        lon1, lat1 = cartesian_to_lonlat(a1)
        lon2, lat2 = cartesian_to_lonlat(a2)
        results.append({
            "pair_idx": pidx,
            "alpha_best": float(ALPHA_RANGE[besti[0]]),
            "twist_best": float(TWISTS[besti[1]]),
            "corr_best": float(corr_scan[besti]),
            "alpha_scan": np.asarray(ALPHA_RANGE, dtype=float),
            "twist_scan": TWISTS.copy(),
            "corr_scan": corr_scan,
            "lon1": lon1[0], "lat1": lat1[0], "lon2": lon2[0], "lat2": lat2[0],
        })
    return results

print(f"=== {TAG} run: NSIDE={NSIDE}, GAL_CUT={GAL_CUT}, alpha={ALPHA_RANGE.tolist()}, "
      f"twist step {TWIST_STEP} ({N_TW} values), N_SIM={N_SIM}, seeds: data 42, sims 100+i ===", flush=True)
print(f"start: {datetime.datetime.now().isoformat()}", flush=True)

cmb = load_or_generate_cmb_map(nside=NSIDE, random_seed=42, data_path=MAP)
cmb_masked, mask = apply_cmb_mask(cmb, gal_cut=GAL_CUT)
print(f"map loaded: npix={len(cmb)}, valid={mask.sum()} ({mask.sum()/len(mask)*100:.1f}%), "
      f"masked-map std(valid)={np.std(cmb_masked[mask]):.6e} K", flush=True)
axes = get_dodecahedron_axes()

# ---------- data scan ----------
data_results = scan_map(cmb_masked, mask, axes, "data")
best = max(data_results, key=lambda r: r["corr_best"])
zero_frac = np.array([np.mean(r["corr_scan"] == 0.0) for r in data_results])
print("\nDATA per-pair best:", flush=True)
for r in data_results:
    print(f"  pair {r['pair_idx']+1}: r={r['corr_best']:.6f} at alpha={r['alpha_best']:.0f}, "
          f"twist={r['twist_best']:.0f} | zero-cell fraction {zero_frac[r['pair_idx']]:.3f}", flush=True)
print(f"DATA best overall: pair {best['pair_idx']+1}, alpha={best['alpha_best']:.0f}, "
      f"twist={best['twist_best']:.0f}, r={best['corr_best']:.6f}", flush=True)

# ---------- simulations ----------
sim_corrs = np.zeros(N_SIM)
sim_best_pair = np.zeros(N_SIM, dtype=int)
data_max_state = state["run_max"]
for i in range(N_SIM):
    sim_map = load_or_generate_cmb_map(nside=NSIDE, random_seed=100 + i)  # synthetic (no data_path)
    sim_map[~mask] = hp.UNSEEN
    sim_results = scan_map(sim_map, mask, axes, f"sim {i+1}/{N_SIM}")
    sb = max(sim_results, key=lambda r: r["corr_best"])
    sim_corrs[i] = sb["corr_best"]
    sim_best_pair[i] = sb["pair_idx"]
    print(f"  sim {i+1:2d}/{N_SIM}: max r = {sim_corrs[i]:.6f} (pair {sb['pair_idx']+1})", flush=True)

sig = compute_significance(best["corr_best"], sim_corrs)
pds_alpha_ok = 29 <= best["alpha_best"] <= 37
pds_twist_ok = abs(best["twist_best"] - TWIST_PDS) <= 10

print(f"\n=== {TAG} SIGNIFICANCE ===", flush=True)
n_ge = int(np.sum(sim_corrs >= best["corr_best"]))
print(f"observed max r = {best['corr_best']:.6f}", flush=True)
print(f"sim mean = {sig['sim_mean']:.6f}, sim std = {sig['sim_std']:.6f}", flush=True)
print(f"p = count(sim >= obs)/N = {n_ge}/{N_SIM} = {sig['p_value']:.4f}", flush=True)
print(f"z = (obs - mean)/std = ({best['corr_best']:.6f} - {sig['sim_mean']:.6f})/{sig['sim_std']:.6f} "
      f"= {sig['z_score']:.4f}", flush=True)
print(f"pds_alpha_ok={pds_alpha_ok}, pds_twist_ok={pds_twist_ok}", flush=True)

# ---------- sanity gates ----------
gate_fail = []
any_one_data = any(np.any(r["corr_scan"] == 1.0) for r in data_results)
if any_one_data or np.any(sim_corrs == 1.0):
    gate_fail.append("GATE FAIL: correlation value exactly 1.0 present")
if not (np.std(sim_corrs) > 0):
    gate_fail.append("GATE FAIL: simulation maxima degenerate (std == 0)")
print("\nGATES:", flush=True)
print(f"  exact-1.0 in data scan: {any_one_data}; in sim maxima: {bool(np.any(sim_corrs == 1.0))}", flush=True)
print(f"  sim std = {np.std(sim_corrs, ddof=1):.6f} (>0 required)", flush=True)
for pidx in range(6):
    print(f"  pair {pidx+1} zero-cell fraction (data): {zero_frac[pidx]:.3f}", flush=True)
for g in gate_fail:
    print(g, flush=True)

# ---------- R3 extras: exact twist=36/324 ----------
extras = {}
if TWIST_STEP == 1.0:
    i36 = int(np.where(TWISTS == 36.0)[0][0])
    i324 = int(np.where(TWISTS == 324.0)[0][0])
    r_at_36 = np.stack([r["corr_scan"][:, i36] for r in data_results])    # (6, 9)
    r_at_324 = np.stack([r["corr_scan"][:, i324] for r in data_results])  # (6, 9)
    extras = {"r_at_twist36": r_at_36, "r_at_twist324": r_at_324}
    print("\nR3: data r at exact twist=36 (rows=pairs, cols=alpha 10..50):", flush=True)
    print(np.array2string(r_at_36, precision=4, max_line_width=200), flush=True)
    print("R3: data r at exact twist=324 (-36):", flush=True)
    print(np.array2string(r_at_324, precision=4, max_line_width=200), flush=True)

# ---------- save ----------
npz_path = os.path.join(OUT, f"matched_circles_{TAG}.npz")
np.savez(
    npz_path,
    tag=TAG, nside=NSIDE, gal_cut=GAL_CUT, n_sim=N_SIM,
    twist_step=TWIST_STEP, alpha_range=ALPHA_RANGE, twist_grid=TWISTS,
    seed_data=42, seed_sim_base=100,
    best_alpha=best["alpha_best"], best_twist=best["twist_best"],
    best_corr=best["corr_best"], best_pair=best["pair_idx"],
    per_pair_best_corr=np.array([r["corr_best"] for r in data_results]),
    per_pair_best_alpha=np.array([r["alpha_best"] for r in data_results]),
    per_pair_best_twist=np.array([r["twist_best"] for r in data_results]),
    data_corr_scan=np.stack([r["corr_scan"] for r in data_results]),
    zero_cell_fraction=zero_frac,
    sim_corrs=sim_corrs, sim_best_pair=sim_best_pair,
    p_value=sig["p_value"], z_score=sig["z_score"],
    sim_mean=sig["sim_mean"], sim_std=sig["sim_std"],
    pds_alpha_ok=pds_alpha_ok, pds_twist_ok=pds_twist_ok,
    run_start_iso=datetime.datetime.fromtimestamp(t_start).isoformat(),
    run_end_iso=datetime.datetime.now().isoformat(),
    elapsed_s=time.time() - t_start,
    **extras,
)
print(f"\nsaved: {npz_path}", flush=True)

fig_base = os.path.join(OUT, f"fig_{TAG}")
plot_correlation_vs_alpha(data_results, fig_base)
plot_correlation_vs_twist(best, fig_base)
plot_sim_distribution(sim_corrs, best["corr_best"], sig["p_value"], fig_base)
plot_cmb_with_circles(cmb_masked, best, axes, best["pair_idx"], fig_base)
print(f"figures: {fig_base}_corr_vs_alpha/_corr_vs_twist/_sim_dist/_cmb_circles .pdf/.png", flush=True)

print(f"end: {datetime.datetime.now().isoformat()}; elapsed {time.time()-t_start:.1f} s", flush=True)
if gate_fail:
    sys.exit(2)
