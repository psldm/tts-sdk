# RESULTS: matched-circle search verification runs (2026-08-04)

Verification runs R1–R4 executed 2026-08-04, following the read-only audit
recorded in `verification/2026-08-04/REPORT_circle_search.md` (the pre-run
audit, copied verbatim into this repository). This document records
what the runs produced. Facts only. Every number is either
formula + substitution + result, or carries provenance
(`file:line`, or artifact + timestamp + md5); see §10.

Run scripts and full console logs are preserved in
`verification/2026-08-04/` (`r1_map_load.py`, `r_timing_probe.py`,
`run_search.py`, `R2_run.log`, `R3_run.log`). The wrapper `run_search.py`
imports the working-tree `src/matched_circles.py` unmodified (same grids and
seeds as its `__main__`: data map seed 42, simulation i seed 100+i) and adds
only orchestration: live pulse, sanity gates, and result serialization.

---

## 1. Environment

Recorded by `r1_map_load.py` at run time (2026-08-04T20:31:22):

- Hardware: Apple M2, 8 cores, 24 GB RAM (`sysctl`). All runs single-process.
- OS: macOS 26.6 (arm64) — `platform.platform()` = `macOS-26.6-arm64-arm-64bit`.
- Interpreter: CPython 3.12.13, `tts-sdk/venv/bin/python`.
- Packages: numpy 2.5.1, scipy 1.18.0, healpy 1.19.0, astropy 8.0.1,
  matplotlib 3.11.0.
- Code state: commit `fc50da3` plus the uncommitted working-tree hunks in
  `src/matched_circles.py` (traceback print; UNSEEN/overflow filter at
  lines 315–318). Used as-is, no source edits.

## 2. R1 findings: map-load behavior and degeneracy mechanism (measured)

Script: `r1_map_load.py`; full output in the script log (§10, item L1).

1. **Truncated in-repo copy** (28,732,322 B):
   `hp.read_map(field=0, dtype=np.float64)` raises
   `ValueError: cannot reshape array of size 718092 into shape (50331648,)`
   (full traceback in log). In `load_or_generate_cmb_map`
   (`src/matched_circles.py:171-188`) any exception lands in the
   `except` branch → warning printed → **synthetic-map fallback**. This settles
   audit Open Question 3: with the truncated file in place, the entry point
   could not run on real data.
2. **Intact copy** (2,013,312,960 B): reads successfully. npix = 50,331,648
   (NSIDE 2048); UNSEEN-sentinel pixels: 0; pixels with |T| ≥ 1e10: 0; valid
   min = −5.754999e−03, max = +7.898794e−03, std = 1.083652e−04 (K).
3. **UNSEEN leakage through interpolation** (measured by `r_timing_probe.py`):
   on the masked map (|b| ≥ 20°, masked pixels = `hp.UNSEEN`),
   `hp.get_interp_val` along the α = 30° ring of Galactic-plane pair 3 returns
   **174 of 360** samples with |T| ≥ 1e10; for off-plane pair 1: 0 of 360.
   The committed (HEAD) `circle_correlation` has no UNSEEN/overflow filter —
   the filter exists only in the uncommitted hunk (lines 315–318). Sentinel
   values of order 1e30 entering `pearsonr` drive r → 1.0. This is the
   measured mechanism consistent with the retired 2026-07-06 artifact
   (r = 1.0 for data and all 30 simulations; now in
   `_attic_2026-08-04/` (internal session archive, not published), MANIFEST
   entry "degenerate run").

## 3. R4 record: map replacement

| step | value |
| --- | --- |
| Source | `../data/COM_CMB_IQU-smica_2048_R3.00_full.fits` (outer working folder, outside this repository), 2,013,312,960 B |
| Source md5 (pre-copy check) | `ee2fc49a2eb70c2eca0d582e4aae5d05` — equals the audit-recorded value; gate passed |
| Destination before | `data/COM_CMB_IQU-smica_2048_R3.00_full.fits`, 28,732,322 B (mtime 2026-07-06 17:28) |
| Destination after | 2,013,312,960 B (mtime 2026-08-04 20:31) |
| Destination md5 after copy | `ee2fc49a2eb70c2eca0d582e4aae5d05` — equals source; copy verified |

The truncated version was not separately preserved: it is a byte prefix of the
intact file (audit: prefix md5 `174079e28d67340feb5208ef0e43ebb9`) and is
reconstructible by `head -c 28732322`.

## 4. Timing (measured)

Probe: `r_timing_probe.py`, map loaded once at NSIDE = 128 with the
|b| ≥ 20° mask; 100 calls of `circle_correlation(alpha=30, twist=0, pair 1)`.

- Map load + `ud_grade`(2048→128): 3.50 s.
- Mean per call: **1.074 ms** (100 calls, total 0.107 s).
- One `synfast` simulation map (NSIDE 128, lmax 384): 0.027 s.
- Projection per the prescribed formula (total = call count × measured mean,
  call count = (1 + N_SIM) × 6 pairs × 9 alphas × N_twist):
  - R2: (1+50)×6×9×72 = 198,288 calls → 213 s projected;
  - R3: (1+50)×6×9×360 = 991,440 calls → 1,065 s projected.
- Measured wall-clock: **R2 = 69.5 s**, **R3 = 319.1 s** (single core).
  Actuals are below projection because for Galactic-plane pairs 3 and 4
  ~36% of (α, twist) cells return early (< 10 valid points ⇒ 0.0,
  `src/matched_circles.py:310-311,319-320`), faster than the measured
  full-cost call; both labels: "projected from measured per-call time"
  vs "measured".

## 5. R2 results (original configuration, intact map)

Run: `run_search.py R2`, start 2026-08-04T20:34:12, end 20:35:22,
elapsed 69.5 s. Config: NSIDE=128, GAL_CUT=20°, α ∈ {10,…,50} step 5°,
twist ∈ {0,…,355} step 5° (72 values), N_SIM=50, seeds: data 42, sims 100+i.
Map: valid pixels 129,536/196,608 (65.9%), masked-map std(valid) =
9.064910e−05 K. Artifact: `outputs/matched_circles_R2.npz` (40,446 B,
md5 `d5347b73c6ab636193c430aa7f43f4ae`); figures `fig_R2_*` (8 files);
console log L2.

**Best per pair (data), with zero-cell fractions:**

| pair | best r | at α | at twist | zero-cell fraction |
| --- | --- | --- | --- | --- |
| 1 | 0.318045 | 45° | 175° | 0.000 |
| 2 | 0.312447 | 10° | 280° | 0.000 |
| 3 | 0.627607 | 30° | 265° | 0.361 |
| 4 | **0.815240** | 25° | 115° | 0.361 |
| 5 | 0.480632 | 15° | 295° | 0.008 |
| 6 | 0.718831 | 45° | 5° | 0.008 |

Elevated zero-cell fractions for pairs 3 and 4 (the Galactic-plane pairs) —
expected per the sanity-gate design; gate passed.

**All 50 simulation maxima** (isotropic Sachs–Wolfe `synfast`, seeds 100–149,
in seed order):

```text
0.779856 0.794221 0.840133 0.855732 0.872415 0.875209 0.866123 0.908187
0.807228 0.787912 0.967815 0.884911 0.827764 0.831061 0.857412 0.892848
0.983428 0.848756 0.802154 0.970548 0.766773 0.886000 0.894502 0.794765
0.710599 0.875433 0.819273 0.776268 0.903628 0.820710 0.850529 0.871019
0.869694 0.820699 0.825421 0.911203 0.873413 0.814926 0.880648 0.927991
0.890510 0.879857 0.759797 0.916126 0.948934 0.815423 0.840807 0.746124
0.920934 0.874689
```

**Significance (with substitution):**

- observed max r = 0.815240 (pair 4, α = 25°, twist = 115°);
- sim mean = 0.854808, sim std (ddof=1) = 0.059035;
- p = count(sim ≥ obs)/N = 38/50 = **0.7600**;
- z = (obs − mean)/std = (0.815240 − 0.854808)/0.059035 = **−0.6702**;
- `pds_alpha_ok` = False (25 ∉ [29, 37]); `pds_twist_ok` = False (|115 − 36| > 10).

**Sanity gates:** no correlation value exactly 1.0 in data scan or sim maxima;
sim std = 0.059035 > 0. All gates passed.

## 6. R3 results (fine twist grid, ±36° on grid)

Run: `run_search.py R3`, start 2026-08-04T20:35:39, end 20:40:58,
elapsed 319.1 s. Config identical to R2 except twist step 1.0°
(360 values; 36 and 324 are exact grid points). Artifact:
`outputs/matched_circles_R3.npz` (168,544 B, md5
`17a1ce12eca3c5f49d05241367b5987e`); figures `fig_R3_*`; console log L3.

**Best per pair (data):**

| pair | best r | at α | at twist | zero-cell fraction |
| --- | --- | --- | --- | --- |
| 1 | 0.340416 | 15° | 197° | 0.000 |
| 2 | 0.328796 | 15° | 246° | 0.000 |
| 3 | 0.694088 | 30° | 264° | 0.362 |
| 4 | 0.815240 | 25° | 115° | 0.362 |
| 5 | 0.608090 | 50° | 347° | 0.006 |
| 6 | **0.836802** | 45° | 6° | 0.006 |

(Pair 4's best equals the R2 value exactly: its optimum 115° lies on both
grids. The global best moved to pair 6, twist 6°, found by the finer grid.)

**All 50 simulation maxima** (same seeds 100–149, finer twist grid):

```text
0.868349 0.837625 0.895416 0.903571 0.895626 0.878568 0.866123 0.953121
0.913345 0.844787 0.967815 0.884911 0.890602 0.836949 0.863021 0.892848
0.983428 0.922535 0.808863 0.981877 0.853768 0.929471 0.903363 0.853832
0.837142 0.921226 0.828297 0.819523 0.909507 0.949443 0.911116 0.973644
0.905373 0.906691 0.825421 0.911203 0.960850 0.878125 0.932537 0.962404
0.915450 0.879857 0.809545 0.916126 0.959687 0.842371 0.890879 0.893001
0.920934 0.874689
```

**Significance (with substitution):**

- observed max r = 0.836802 (pair 6, α = 45°, twist = 6°);
- sim mean = 0.895297, sim std (ddof=1) = 0.046425;
- p = count(sim ≥ obs)/N = 45/50 = **0.9000**;
- z = (obs − mean)/std = (0.836802 − 0.895297)/0.046425 = **−1.2600**;
- `pds_alpha_ok` = False (45 ∉ [29, 37]); `pds_twist_ok` = False (|6 − 36| > 10).

**Sanity gates:** no exact 1.0 anywhere; sim std = 0.046425 > 0. Passed.

**Pearson r at the exact PDS twists** (data, rows = pairs 1–6,
columns = α = 10°…50° step 5°; 0.000 marks cells with < 10 valid points):

twist = +36°:

| pair \ α | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0944 | 0.0566 | 0.1169 | 0.1011 | 0.0327 | −0.0056 | −0.1157 | −0.0304 | −0.1692 |
| 2 | −0.3899 | −0.0015 | −0.0425 | 0.0621 | 0.0826 | −0.0044 | 0.0679 | −0.0018 | −0.1722 |
| 3 | 0.000 | 0.000 | 0.000 | 0.1600 | −0.2481 | −0.1179 | −0.0115 | 0.0156 | −0.1090 |
| 4 | 0.000 | 0.000 | 0.000 | −0.0083 | 0.0803 | 0.1522 | −0.0093 | −0.1216 | −0.0344 |
| 5 | −0.0485 | 0.1892 | −0.1303 | 0.1029 | 0.1397 | 0.0589 | −0.2433 | 0.2142 | −0.3549 |
| 6 | −0.1672 | −0.3395 | −0.4134 | −0.1788 | −0.2944 | −0.1389 | −0.0823 | −0.0604 | −0.4014 |

twist = 324° (−36°):

| pair \ α | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0097 | −0.1539 | −0.0839 | 0.1187 | −0.0452 | 0.1047 | −0.0933 | −0.1995 | 0.1789 |
| 2 | 0.0674 | 0.0041 | 0.0737 | −0.1121 | −0.0173 | −0.1901 | −0.2346 | 0.0240 | −0.2114 |
| 3 | 0.000 | 0.000 | 0.000 | 0.4784 | −0.1053 | 0.1832 | 0.0454 | −0.0955 | 0.0968 |
| 4 | 0.000 | 0.000 | 0.000 | −0.2112 | 0.0136 | −0.3382 | 0.0904 | 0.1452 | −0.1429 |
| 5 | 0.0710 | 0.0163 | −0.1423 | −0.2163 | 0.3725 | 0.1487 | −0.2384 | −0.0192 | 0.0480 |
| 6 | −0.1408 | 0.0605 | −0.1656 | −0.0424 | −0.0436 | 0.0586 | 0.0780 | −0.0062 | −0.0221 |

Largest value at twist = +36°: r = 0.2142 (pair 5, α = 45°).
Largest at twist = 324°: r = 0.4784 (pair 3, α = 25°).
For comparison, the simulation mean of the *maximum* statistic is 0.8953.

## 7. Updated "this work" table row

| Field | Value |
| --- | --- |
| Label | `tts-sdk` `matched_circles.py` (commit `fc50da3` + UNSEEN filter), runs R2/R3, 2026-08-04 |
| Data (map, release) | Planck PR3 (R3.00) SMICA temperature (`I_STOKES`, `K_CMB`; file md5 `ee2fc49a2eb70c2eca0d582e4aae5d05`), NSIDE 2048 → 128, Galactic \|b\| ≥ 20° cut; no beam or noise treatment |
| Statistic | max Pearson r between two 360-point interpolated circles; maximized over α, twist, 6 axis pairs |
| α coverage | 10°–50°, step 5° |
| Phase coverage | R2: 0°–355°, step 5°; R3: 0°–359°, step 1° — **+36° and −36° (324°) exact grid points in R3** |
| Axis coverage | 6 antipodal face-axis pairs, fixed fiducial golden-ratio orientation (Galactic frame); no orientation marginalisation |
| Sensitivity floor / threshold | empirical null from 50 isotropic Sachs–Wolfe `synfast` simulations (p granularity 0.02); no analytic threshold |
| Outcome | R3: max r = 0.836802 (α = 45°, φ = 6°, pair 6), p = 45/50 = 0.90, z = −1.26; at exact φ = +36°: max r = 0.2142, at φ = −36°: max r = 0.4784. R2: max r = 0.815240, p = 38/50 = 0.76, z = −0.67. Observed maxima lie below the simulation means. **No detection.** |

## 8. Summary paragraph (for the reply to Prof. Luminet)

> An independent matched-circle search was run on the Planck PR3 SMICA
> temperature map (NSIDE 2048, degraded to 128; Galactic |b| ≥ 20° cut; no
> beam or noise treatment), correlating antipodal circle pairs around the six
> face axes of a fixed fiducial golden-ratio dodecahedron in Galactic
> coordinates. The statistic is the Pearson correlation of 360-point
> interpolated circle pairs, maximized over circle radius α = 10°–50° (step
> 5°), relative phase 0°–359° (step 1°, so the PDS holonomy phases +36° and
> −36° are exact grid points), and the six axis pairs. The observed maximum
> is r = 0.8368 at α = 45°, phase 6°, which is below the mean maximum (0.8953)
> of 50 isotropic Sachs–Wolfe simulations processed identically:
> p = 45/50 = 0.90, z = −1.26. At the exact PDS phases the largest observed
> correlations are r = 0.214 (+36°) and r = 0.478 (−36°). No matched-circle
> signal is detected at any scanned α, phase, or axis pair in this
> configuration.

## 9. Caveats (stated plainly)

1. The simulation null uses a scale-invariant Sachs–Wolfe spectrum only
   (`C_ℓ ∝ 1/(ℓ(ℓ+1))`, `src/matched_circles.py:125-147`): no transfer
   function, no acoustic structure, no Doppler, no ISW.
2. No instrument beam and no noise are modeled, in data treatment or
   simulations.
3. The mask is a plain geometric |b| ≥ 20° cut; no Planck confidence or
   point-source mask. The two Galactic-plane axis pairs lose ~36% of their
   (α, twist) cells to the < 10-valid-points rule.
4. Single fixed fiducial dodecahedron orientation; no orientation
   marginalisation. A signal in a rotated orientation would not be found.
5. The statistic is plain Pearson r, not the weighted statistic of
   Cornish et al. (2004) or the six-pair combination of Key et al. (2007).
6. NSIDE = 128 working resolution; structure below the pixel scale (~27′)
   is smoothed by `ud_grade`.
7. N_SIM = 50: the p-value granularity is 1/50 = 0.02.
8. R2 and R3 maximize over the scanned grid only; the p-value accounts for
   this look-elsewhere within the grid because simulations are scanned
   identically.

## 10. Provenance register

- Code: `src/matched_circles.py` working tree = commit `fc50da3` + uncommitted
  hunks (lines 179, 181, 315–318); all line references above to this file
  state.
- L1: `verification/2026-08-04/r1_map_load.py` output (in-session,
  2026-08-04T20:31); reproduced in this document §2.
- L2: `verification/2026-08-04/R2_run.log` (start
  2026-08-04T20:34:12).
- L3: `verification/2026-08-04/R3_run.log` (start
  2026-08-04T20:35:39).
- A-R2: `outputs/matched_circles_R2.npz`, 40,446 B, md5
  `d5347b73c6ab636193c430aa7f43f4ae` (contains config, grids, seeds, full data
  `corr_scan` (6×9×72), per-pair bests, zero-cell fractions, all 50 sim
  maxima, p, z, timestamps).
- A-R3: `outputs/matched_circles_R3.npz`, 168,544 B, md5
  `17a1ce12eca3c5f49d05241367b5987e` (same schema, 6×9×360 scan, plus
  `r_at_twist36`, `r_at_twist324` (6×9) exact-phase tables).
- Retired artifacts: `_attic_2026-08-04/MANIFEST.md` (internal session
  archive, not published; 32 entries, 2026-08-04): the degenerate 2026-07-06
  npz with its 4 figure pairs, caches, `.DS_Store`.
- Map replacement record: §3 of this document.
- Rule: any number not derivable from the above does not appear in this
  document.
