"""A3: timing probe for R2/R3 projection + UNSEEN-leakage measurement.

Loads the (now intact) in-repo map once at NSIDE=128 with the |b|>=20 mask,
times 100 calls of circle_correlation (alpha=30, twist=0, pair 1 = index 0),
and projects R2/R3 wall-clock from the measured mean per-call time.

Also records, as measured fact, whether hp.get_interp_val leaks huge values
(UNSEEN-contaminated) into ring samples on a masked map (mechanism behind the
degenerate 2026-07-06 artifact).
"""
import sys, os, time, datetime
import numpy as np

sys.path.insert(0, "/Users/air_eva/Desktop/tts_sdk/tts-sdk/src")
import healpy as hp
from matched_circles import (
    load_or_generate_cmb_map, apply_cmb_mask, get_dodecahedron_axes,
    circle_correlation, sample_circle, N_CIRCLE_POINTS,
)

MAP = "/Users/air_eva/Desktop/tts_sdk/tts-sdk/data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
NSIDE, GAL_CUT = 128, 20.0

print(f"start: {datetime.datetime.now().isoformat()}")
t0 = time.perf_counter()
cmb = load_or_generate_cmb_map(nside=NSIDE, random_seed=42, data_path=MAP)
t_load = time.perf_counter() - t0
print(f"map load+ud_grade(2048->128): {t_load:.2f} s; npix={len(cmb)}; std={np.std(cmb):.6e}")

cmb_masked, mask = apply_cmb_mask(cmb, gal_cut=GAL_CUT)
print(f"mask |b|>={GAL_CUT}: valid={mask.sum()}/{len(mask)} ({mask.sum()/len(mask)*100:.1f}%)")

axes = get_dodecahedron_axes()

# --- UNSEEN leakage measurement (mechanism fact) ---
for pidx in (0, 2):  # off-plane pair 1, Galactic-plane pair 3
    th, ph = sample_circle(axes[pidx, 0], 30.0, N_CIRCLE_POINTS)
    T = hp.get_interp_val(cmb_masked, th, ph, nest=False)
    n_huge = int(np.sum(np.abs(T) >= 1e10))
    print(f"ring pair {pidx+1} (alpha=30): interp samples with |T|>=1e10: {n_huge}/{N_CIRCLE_POINTS}")

# --- timing: 100 calls, alpha=30, twist=0, pair 1 (index 0) ---
_ = circle_correlation(cmb_masked, axes[0,0], axes[0,1], 30.0, 0.0, NSIDE, N_CIRCLE_POINTS, mask)  # warm-up
t0 = time.perf_counter()
N_CALLS = 100
for _i in range(N_CALLS):
    r = circle_correlation(cmb_masked, axes[0,0], axes[0,1], 30.0, 0.0, NSIDE, N_CIRCLE_POINTS, mask)
t_total = time.perf_counter() - t0
mean_call = t_total / N_CALLS
print(f"circle_correlation x{N_CALLS} (alpha=30, twist=0, pair 1): total {t_total:.3f} s, mean {mean_call*1000:.3f} ms/call; last r={r:.6f}")

# --- synfast timing (additive fact for the sim loop) ---
t0 = time.perf_counter()
sim = load_or_generate_cmb_map(nside=NSIDE, random_seed=100)
t_synfast = time.perf_counter() - t0
print(f"one synfast sim map (nside={NSIDE}, lmax={3*NSIDE}): {t_synfast:.3f} s")

# --- projections (per task formula: total = call count x measured mean) ---
for label, n_twist in (("R2 (twist step 5.0)", 72), ("R3 (twist step 1.0)", 360)):
    calls = (1 + 50) * 6 * 9 * n_twist
    proj = calls * mean_call
    print(f"{label}: calls=(1+50)*6*9*{n_twist}={calls}; projected {proj:.0f} s = {proj/60:.1f} min "
          f"(+ 50 synfast = {50*t_synfast:.0f} s)")
print(f"end: {datetime.datetime.now().isoformat()}")
