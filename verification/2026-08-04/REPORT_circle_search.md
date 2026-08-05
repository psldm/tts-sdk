# REPORT: Audit of the TTS SDK matched-circle search

**Scope.** Read-only engineering audit of the matched-circle search implemented in
this repository, performed 2026-08-04. Facts only; no interpretation, no theory
claims. Every numerical statement carries provenance: `file:line` of code, a
stored output file with its date and checksum, or an explicit computation
(formula + substitution + result) executed read-only during this audit.
No code, data, or configs were modified. The only file created is this report.

**Roots.** Outer working folder: `/Users/air_eva/Desktop/tts_sdk/` (not a git
repository). The SDK proper is the nested git repository `tts-sdk/`
(single commit `fc50da3`, 2026-07-19 15:44:34 −0700, "TTS v3.0 supplementary
code: dodecahedral H0 anisotropy and CMB matched-circle search"; remote
`https://github.com/psldm/tts-sdk.git`) `[run: git log / git remote -v]`.
All relative paths below are from the outer folder unless noted.

**Provenance labels.**
- `[code]` — read from source, cited as `file:line` (working tree as of this audit).
- `[artifact]` — value read from a stored output file; labeled
  "stored output, not re-verified in this session" where applicable.
- `[run]` — obtained by a read-only command executed during this audit (command given).
- `[claim]` — stated in a README/docstring, not independently verified here.

---

## 1. Inventory

### 1.1 Directory tree (depth 2, service dirs omitted) `[run: find]`

```
tts_sdk/                                  (outer folder, 4.5 GB total, not a git repo)
├── data/
│   ├── COM_CMB_IQU-smica_2048_R3.00_full.fits    (full Planck SMICA map, 2.0 GB)
│   └── DataRelease/                              (git clone of PantheonPlusSH0ES/DataRelease)
├── tts-sdk/                              (the SDK; its own git repo, remote psldm/tts-sdk)
│   ├── README.md, LICENSE (MIT), CITATION.cff (v3.0.0, date-released 2026-07-19)
│   ├── CIRCLES_SDK_SUMMARY.md            (untracked; prior v3.0 audit document)
│   ├── pyproject.toml, requirements.txt
│   ├── data/                             (git-ignored except README.md/.gitkeep)
│   ├── src/                              (11 Python modules, flat layout)
│   ├── tests/                            (conftest + 2 test modules)
│   ├── notebooks/analysis.ipynb          (Pantheon+ H0 walkthrough; 0 occurrences of "circle" [run: grep -c])
│   ├── outputs/                          (git-ignored except summary*.txt/.gitkeep)
│   └── venv/
├── .claude/, .kilo/, .offline-markdown-preview/   (tooling; not part of the analyses)
```

Omitted from the tree: `venv/`, `node_modules/`, caches, `.git` internals.
The DataRelease clone contains 3073 files under `Pantheon+_Data/` and 14 under
`SH0ES_Data/` `[run: find | count]`.

### 1.2 Datasets

| # | File | Size (bytes) | Format | Provenance as recorded | Download date recorded? |
|---|------|--------------|--------|------------------------|--------------------------|
| D1 | `data/COM_CMB_IQU-smica_2048_R3.00_full.fits` | 2,013,312,960 | FITS (HEALPix bintable) | `tts-sdk/data/README.md:15-19`: Planck Legacy Archive, Maps → CMB → SMICA, PR3/2018. FITS header `[run: astropy]`: `DATE='2018-04-10'`, `METHOD='SMICA'`, `AST-COMP='CMB'`, `NSIDE=2048`, `ORDERING='NESTED'`, `COORDSYS='GALACTIC'`, 10 columns (`TTYPE1='I_STOKES'`, `TUNIT1='K_CMB'`), HDU2 = 4097 R × 2 C (beam/transfer table). File intact: declared HDU1 data 2,013,265,920 B ≤ file size. md5 `ee2fc49a2eb70c2eca0d582e4aae5d05` `[run: md5]` | No. File mtime 2026-07-06 18:21 `[run: ls -la]`. No checksum recorded in repo → Open Questions |
| D2 | `tts-sdk/data/COM_CMB_IQU-smica_2048_R3.00_full.fits` | 28,732,322 | FITS (truncated) | Same header as D1 but **truncated**: astropy warns "actual file length (28732322) is smaller than the expected size (2013275520)" `[run: astropy]`. Byte-identical to the first 28,732,322 bytes of D1: md5 of D1's prefix = md5 of D2 = `174079e28d67340feb5208ef0e43ebb9` `[run: head -c | md5]`. **This is the path the search code loads** (see §2.3) | No. mtime 2026-07-06 17:28. Reason for truncation not recorded → Open Questions |
| D3 | `tts-sdk/data/Pantheon+SH0ES.dat` | 579,283 | whitespace table | `tts-sdk/data/README.md:8-13`: PantheonPlusSH0ES/DataRelease. md5 `2049b142e6aad384470b3364aa10f3fa` — identical to the copy in the local clone `data/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/` `[run: md5 both]` | No explicit record; clone evidence: `.git/HEAD` mtime 2026-07-05 `[run: ls -la]` |
| D4 | `tts-sdk/data/Pantheon+SH0ES_STAT+SYS.cov` | 33,284,960 | text covariance | Same source; md5 `041bdb6638841794fc2d7caa88dd66eb` — identical to clone copy `[run: md5 both]` | Same as D3 |
| D5 | `data/DataRelease/` | (clone) | git repo | Upstream `https://github.com/PantheonPlusSH0ES/DataRelease.git`, HEAD `c447f0f` (upstream commit of 2022-12-21) `[run: git log/remote]` | Clone date evidence: `.git/HEAD` mtime 2026-07-05 20:44, `.git/FETCH_HEAD` 2026-07-05 23:00 |
| D6 | `tts-sdk/data/union3_inputs.pickle` | 2,236,190 | pickle (gzip) | `tts-sdk/data/README.md:21-25`: Union3/UNITY1.5, Rubin et al. 2023, arXiv:2311.12098, github.com/rubind/union3_release | No; mtime 2026-07-06 |
| D7 | `tts-sdk/data/union3_samples.npz` | 100,164,320 | npz | same release (README names only the pickle explicitly) | No; mtime 2026-07-06 |
| D8 | `tts-sdk/data/union3_release.zip` | 91,653,568 | zip | same release | No; mtime 2026-07-06 |
| D9 | `tts-sdk/data/union3_mu.fits` | 8,640 | FITS | same release | No; mtime 2026-07-06 |
| D10 | `tts-sdk/data/CF4_synthetic.csv` | 2,520,460 | CSV | **Synthetic**, generated by `src/cosmicflows.py` when the real `CF4_data.csv` is absent; "isotropic mock with no dodecahedral signal by construction" (`tts-sdk/data/README.md:34-37`) | Generated locally; mtime 2026-07-06 |

No dataset in the repository has a recorded download date or a recorded checksum;
identification above is by md5 computed during this audit and by FITS headers.

### 1.3 Scripts/modules (`tts-sdk/src/`), one line each

| Module | Purpose (from module docstring / README structure block `README.md:66-76`) |
|--------|---------------------------------------------------------------------------|
| `dodecahedron.py` | 12 dodecahedron face normals (golden-ratio fiducial), sector assignment, Pantheon+/Union3 loaders |
| `h0_fit.py` | per-sector H0 chi-square fits with full covariance |
| `simulations.py` | Monte-Carlo isotropic null for ΔH0 |
| `run_all.py` | end-to-end H0 pipeline (Pantheon+/Union3), CLI |
| `visualize.py` | publication figures for the H0 pipeline |
| `audit.py` | internal-consistency self-audit (7 blocks) |
| `tts_tables.py` | reproduces paper Tables 1–4 from Eqs (60)–(65); includes the α(Ω_tot) map, Eq (62) |
| **`matched_circles.py`** | **the CMB matched-circle search audited in §2** |
| `rbsg08.py` | Roukema et al. (2008)-style covering-space cross-correlation on a **synthetic** map (`rbsg08.py:256-258`, `748-772`; grid `NSIDE=64, N_POINTS=200, ALPHA_RANGE=linspace(10,50,6), TWIST_RANGE=linspace(0,350,36)` at `rbsg08.py:757-761`) |
| `cosmicflows.py` | CosmicFlows-4 peculiar-velocity cross-check (synthetic fallback) |
| `sirens.py` | GW-siren / lensed-quasar H0 comparison (hardcoded literature values) |

### 1.4 Non-circle analyses and their entry points (inventory only, not deep-audited)

- Pantheon+/Union3 H0 anisotropy: `python src/run_all.py --catalog {pantheon|union3|all}` (`README.md:100-104`).
- Paper tables: `python src/tts_tables.py` (`README.md:106-107`).
- Self-audit: `python src/audit.py` (`README.md:109-110`).
- Cross-checks: `python src/cosmicflows.py`, `python src/sirens.py`, `python src/rbsg08.py` (`README.md:116-119`).
- Tests: `pytest tests/` — `tests/` contains only `test_dodecahedron.py` and
  `test_tts_tables.py`; **zero occurrences of "circle" in `tests/`** `[run: grep -rn -i circle tests/]`.
  There is no automated test of the matched-circle code.

### 1.5 Git state of `tts-sdk/` `[run: git status/diff]`

- Single commit `fc50da3` (2026-07-19). Untracked: `CIRCLES_SDK_SUMMARY.md`.
- **One modified tracked file: `src/matched_circles.py`** (file mtime 2026-07-19 22:08),
  two uncommitted hunks:
  1. `load_or_generate_cmb_map`: adds `import traceback` / `traceback.print_exc()`
     on map-load failure (working-tree lines 179, 181);
  2. `circle_correlation`: adds explicit exclusion of the HEALPix `UNSEEN`
     sentinel (−1.6375e30) and of `|T| ≥ 1e10` before the Pearson correlation
     (working-tree lines 315–318, comment: "np.isfinite does not catch it").
- Consequence recorded as fact: the committed (HEAD) version of the search does
  **not** contain the UNSEEN exclusion; the working-tree version does. All line
  numbers in §2 refer to the **working tree**.

---

## 2. Pipeline (`tts-sdk/src/matched_circles.py`, working tree)

**Entry point:** `python src/matched_circles.py` (`__main__` block,
`matched_circles.py:770-933`; also listed in `README.md:112-114`).

### 2.1 Correlation statistic, exactly as coded

Per circle pair, radius α, twist φ (`circle_correlation`, `matched_circles.py:264-322`):

1. Sample n = 360 equally spaced points on each circle of angular radius α
   about `axis1` and `axis2` (`sample_circle`, lines 221-261; `N_CIRCLE_POINTS = 360`, line 62).
2. `T1 = hp.get_interp_val(cmb_map, theta1, phi1, nest=False)` and likewise `T2`
   (lines 301-302) — bilinear HEALPix interpolation of the temperature along each ring.
3. Twist applied as an integer cyclic shift:
   `shift = int(round(twist_deg / 360.0 * n_points)) % n_points` (line 303);
   `T2_shifted = np.roll(T2, shift)` (line 304). With n_points = 360 this is
   `shift = round(twist_deg)` — the phase is quantized to whole degrees.
4. Mask handling: points whose pixel (or twist-shifted partner pixel) falls in
   the masked region are dropped (lines 305-313); < 10 valid points ⇒ return 0.0
   (lines 310-311, 319-320).
5. UNSEEN/overflow guard (uncommitted hunk, lines 314-318).
6. Statistic: **Pearson product-moment correlation coefficient**
   `corr, _ = pearsonr(T1[common], T2_shifted[common])` (line 321; `scipy.stats.pearsonr`, imported line 33).

Aggregation:
- per axis pair: `S(pair) = max over α, φ of r` — brute-force double loop
  (`search_circles_for_axis`, lines 361-375); **no FFT over the phase variable**;
- over pairs: `best_overall = max(all_results, key=corr_best)` over the 6 pairs (line 838);
- significance: `p = Σ[sim_corrs ≥ observed] / n_sim` and
  `z = (observed − sim_mean)/sim_std` (`compute_significance`, lines 529-533).

**Name:** the quantity correlated is plain Pearson r of the two interpolated
temperature rings. It is **not** the weighted circle statistic
S = 2⟨T₁T₂⟩/⟨T₁²+T₂²⟩ of Cornish et al. (2004), and **not** the six-pair
combined statistic of Key et al. (2007); no per-point or per-m weighting is
applied anywhere in the file. (The docstring's own references list, lines 17-21,
cites Cornish+2004 and Roukema+2008 as background, not as the implemented statistic.)

### 2.2 Input map, resolution, mask, beam, noise

- **Map path:** `data/COM_CMB_IQU-smica_2048_R3.00_full.fits` inside the repo
  (built at lines 787-789) — i.e. **the truncated 28.7 MB copy (D2)**, not the
  intact 2.0 GB file (D1) one level up.
- **Release/component separation:** Planck PR3 (R3.00, header `DATE='2018-04-10'`)
  SMICA; field 0 = `I_STOKES` in `K_CMB` (`hp.read_map(..., field=0, dtype=np.float64)`,
  line 173; column identity from the FITS header `[run: astropy]`).
- **Resolution:** map is degraded from NSIDE 2048 to `NSIDE = 128` via
  `hp.ud_grade` (lines 174-176; `NSIDE=128` at line 781). No explicit harmonic
  truncation is applied to the real map.
- **ell_max (synthetic path/simulations only):** `lmax = 3 * nside` = 384
  (line 184).
- **Fallback:** on any exception while reading the FITS, the code prints a
  warning and generates a **synthetic** map, `hp.synfast(sachs_wolfe_cls(lmax), nside)`
  with seed 42 (lines 178-188, 804-806). The synthetic spectrum is
  `C_ℓ = 2.101e-9 · 2π / (ℓ(ℓ+1))` for ℓ ≥ 2, zero for ℓ = 0,1
  (lines 144-147; verified numerically: C₂ = 2.101e-9·2π/6 = 2.2002e-9 `[run]`);
  the docstring states "No transfer function is applied" (lines 130-132).
- **Mask:** geometric Galactic-latitude cut only, `|b| ≥ 20°`
  (`apply_cmb_mask`, lines 191-218; `GAL_CUT = 20.0`, line 785). **No** Planck
  confidence/common mask, **no** point-source mask.
- **Beam:** none — no smoothing or beam deconvolution anywhere in the file.
- **Noise:** none — simulations are signal-only `synfast` realizations (line 187);
  no noise model is applied to data or simulations.

### 2.3 Grid of circle radii α

`ALPHA_RANGE = np.arange(10, 51, 5)` (line 783) =
{10, 15, 20, 25, 30, 35, 40, 45, 50}° — min 10°, max 50°, step 5°, 9 values
`[run: verified numerically]`.

### 2.4 Search space over circle-pair axes

- Axes are the **6 antipodal face-axis pairs of a fixed fiducial golden-ratio
  dodecahedron**, interpreted directly in Galactic coordinates
  (`get_dodecahedron_axes`, lines 65-97, pairing the 12 normals of
  `dodecahedron.py:20-45`; docstring lines 5-9: "The orientation of the
  dodecahedron is NOT marginalised over").
- **There is no all-sky axis grid and no orientation scan** — one fixed
  orientation, 6 axis directions total (used at lines 817, 828-836).
- The 6 pair centres, computed from the repo's own functions `[run]`:

  | pair (1-based) | centre 1 (l, b) | centre 2 (l, b) |
  |---|---|---|
  | 1 | (90.00°, +58.28°) | (270.00°, −58.28°) |
  | 2 | (90.00°, −58.28°) | (270.00°, +58.28°) |
  | 3 | (58.28°, 0.00°) | (238.28°, 0.00°) |
  | 4 | (301.72°, 0.00°) | (121.72°, 0.00°) |
  | 5 | (0.00°, +31.72°) | (180.00°, −31.72°) |
  | 6 | (0.00°, −31.72°) | (180.00°, +31.72°) |

  Pairs 3 and 4 lie exactly in the Galactic plane (b = 0°), i.e. their circles
  intersect the |b| < 20° mask maximally. The 6 antipodal face axes of the
  dodecahedron are therefore all "handled", but only in this one fiducial
  orientation.

### 2.5 Relative phase (twist) handling

- Twist grid: `twists = np.arange(0, 360, twist_step)` (line 361). The
  `__main__` run sets `TWIST_STEP = 5.0` (line 784, passed at lines 835 and 864),
  giving 72 values {0, 5, …, 355}° `[run]`.
- The statistic **is maximized over the twist grid** by the explicit loop at
  lines 367-375 (**not** via FFT).
- **+36° and −36° (=324°) are NOT in the scanned set** under the `__main__`
  configuration: neither 36 nor 324 is a multiple of 5
  (`36.0 in np.arange(0,360,5)` → False; `324.0 in ...` → False `[run]`).
  Nearest scanned points: 35° and 325° (1° away) `[run]`.
- The function-level default `twist_step: float = 1.0` (line 333) **would**
  include 36° and 324° exactly `[run]`, but the entry point overrides it to 5.0.
- Phase quantization: because the shift is `round(twist_deg)` points on a
  360-point ring (line 303), the effective phase resolution floor is 1° even
  for non-integer twist values.

### 2.6 Detection threshold

- **No analytic or pre-calibrated false-alarm threshold exists in the code.**
  Significance is an empirical p-value of the observed maximum against `N_SIM`
  isotropic simulations: `p = Σ[sim ≥ obs]/n_sim` (line 532).
- Null calibration: `run_simulations` (lines 449-510) generates `N_SIM = 50`
  (line 782) synthetic `synfast` maps from the same Sachs–Wolfe spectrum,
  seeds `100 + i` (base seed 100 at line 865; increment at lines 483, 492),
  applies the same mask and identical α/twist grids (docstring lines 462-464:
  "The twist grid must be identical to the one used on the data to avoid
  biasing the p-value"), and records the max correlation per simulation.
- Verdict tiers, as coded (lines 876-881): p < 0.01 → "SIGNIFICANT";
  p < 0.05 → "MARGINALLY SIGNIFICANT"; else "NOT SIGNIFICANT".
- With N_SIM = 50 the smallest resolvable p is 0 and the granularity is 0.02.
  (Arithmetic: 1/50 = 0.02.)

### 2.7 Publication-run parameters

The docstring (lines 14-15) and `__main__` comment (lines 779-780) state the
defaults are "raised for the publication run (higher NSIDE, more simulations,
finer alpha/twist grids)" `[claim]`. **No publication-run configuration, log,
or output is present in the repository** → Open Questions.

### 2.8 Ω_tot

`matched_circles.py` contains no Ω_tot parameter; the search scans α directly
(§2.3). The α ↔ Ω_tot mapping (Eq 62) is implemented separately in
`tts_tables.py` (audited only as inventory here). Any statement of the search's
coverage in Ω_tot terms is a derived quantity, not something the search code
computes.

---

## 3. Coverage summary (what this search does and does not scan)

As coded at the `__main__` configuration (`matched_circles.py:781-785`):

| Dimension | Covered | Not covered |
|---|---|---|
| Circle radius α | 10°–50°, step 5° (9 values) | α < 10°, α > 50°, and everything between grid points |
| Twist φ | 0°–355°, step 5° (72 values), maximized over | the exact PDS values ±36° (nearest grid points 35°/325°); sub-degree phases |
| Axis directions | 6 antipodal face-axis pairs of one fixed golden-ratio orientation in Galactic frame | any other orientation of the dodecahedron; any all-sky axis grid; non-back-to-back circle pairs |
| Map | Planck PR3 SMICA temperature (I_STOKES), degraded to NSIDE=128, |b| ≥ 20° cut | polarization; other component separations (NILC/SEVEM/Commander); Planck confidence masks; beam/noise treatment |
| Null model | 50 signal-only synfast maps of a scale-invariant Sachs–Wolfe spectrum (no transfer function) | ΛCDM acoustic spectrum; instrument noise; mask-induced anisotropy of the null beyond the shared mask |
| Ω_tot | not a search parameter (see §2.8) | — |

---

## 4. Results (null summary with provenance)

### 4.1 Stored artifacts of the circle search

The only stored results of `matched_circles.py` are
`tts-sdk/outputs/matched_circles.npz` and four figure pairs
(`fig_corr_vs_alpha`, `fig_corr_vs_twist`, `fig_sim_dist`, `fig_cmb_circles`,
`.pdf`+`.png` each), all with mtime **2026-07-06 16:42** `[run: ls -la]` —
one run. There is **no text log** of the circle search (the `outputs/summary*.txt`
mechanism covers only the H0 pipeline). All values below are
**"stored output, not re-verified in this session."**

`outputs/matched_circles.npz` (mtime 2026-07-06 16:42:18, 2,570 B,
md5 `2a70c358f0a22695619b5fcacdebf553`) `[run: np.load]`:

| Key | Stored value |
|---|---|
| `best_alpha` | 20 |
| `best_twist` | 0.0 |
| `best_corr` | 1.0 |
| `best_pair` | 2 (0-based `pair_idx`, saved at `matched_circles.py:908` from the loop at lines 423-436 ⇒ pair 3 in the 1-based numbering of §2.4: centres (58.28°, 0°)/(238.28°, 0°), the Galactic-plane pair) |
| `sim_corrs` | 30 values, **all exactly 1.0** (min = max = mean = 1.0) |
| `p_value` | 1.0 |
| `z_score` | 0.0 |
| `pds_alpha_ok` | False |
| `pds_twist_ok` | False |

### 4.2 Facts about this stored run (no interpretation)

1. The data maximum (r = 1.0) and **every one** of the 30 simulation maxima
   (r = 1.0) saturate the statistic exactly; p = 1.0 follows arithmetically
   (30/30 sims ≥ observed).
2. `sim_corrs` has length **30**, whereas the current code sets `N_SIM = 50`
   (`matched_circles.py:782`) — the stored run was made with a different
   configuration than the current source.
3. The npz mtime (2026-07-06) **predates the repository's only commit**
   (2026-07-19) and the uncommitted UNSEEN-exclusion fix
   (`src/matched_circles.py` mtime 2026-07-19 22:08). The exact code version
   that produced the npz is not recoverable from the repository.
4. The map path the entry point reads (`matched_circles.py:787-789`) points at
   the truncated 28.7 MB copy (D2, §1.2); the intact 2.0 GB map (D1) is outside
   the repo and is not referenced by any code path `[run: grep]`.
5. The PDS-compatibility flags stored are both False, per their definitions
   `pds_alpha_ok = (29 ≤ α_best ≤ 37)`, `pds_twist_ok = (|φ_best − 36| ≤ 10)`
   (lines 851-852) with α_best = 20, φ_best = 0.

**Net factual statement:** the repository does **not** contain a valid
(non-degenerate) null-result artifact for the matched-circle search. The single
stored run is saturated at r = 1.0 in both data and all simulations and carries
`p = 1.0` by construction. No maximum-statistic-vs-threshold comparison from a
healthy run exists in the repo.

---

## 5. Draft "this work" table row

Fields left blank could not be established from the repository; per the audit
rules, blanks are not filled with estimates.

| Field | Value |
|---|---|
| Label | `tts-sdk` `matched_circles.py` (v3.0.0, commit `fc50da3` + uncommitted UNSEEN fix in working tree) |
| Data (map, release) | Planck PR3 (R3.00) SMICA temperature (`I_STOKES`, `K_CMB`), NSIDE 2048 → degraded to 128; Galactic |b| ≥ 20° cut; no beam/noise treatment. NOTE: in-repo copy of the map is truncated (28.7 MB of 2.0 GB); intact copy exists outside the repo |
| Statistic | max Pearson r between two 360-point interpolated circles; maximized over α, twist, and 6 axis pairs; no weighting; no FFT |
| α coverage | 10°–50°, step 5° |
| Phase (twist) coverage | 0°–355°, step 5°, maximized over grid; exact ±36° not on grid (nearest 35°/325°); 1° quantization floor |
| Axis coverage | 6 antipodal face-axis pairs of one fixed fiducial golden-ratio orientation (Galactic frame); no orientation marginalisation; back-to-back pairs only |
| Sensitivity floor / threshold | no analytic threshold; empirical p vs 50 isotropic Sachs–Wolfe synfast sims (p-granularity 0.02); stored run's null distribution degenerate (all sims r = 1.0) — **no valid calibration on record** |
| Outcome | stored run (2026-07-06): α_best = 20°, φ_best = 0°, pair 3, r = 1.0, p = 1.0 — **degenerate; not a valid null result**. No verified science outcome for the circle search exists in the repository |

---

## 6. Proposed verification runs (PROPOSE ONLY — awaiting explicit approval)

All runs below are designed to require **no modification of repo code, data, or
configs**: wrappers live in the session scratchpad and import repo modules;
outputs go to the scratchpad. Each long run prints a pulse line at least every
60 s (timestamp, current α, iteration count, current max statistic), per the
task's live-pulse requirement. Interpreter: `tts-sdk/venv/bin/python`
(CPython with numpy/scipy/healpy installed).

**R1 — Map-load smoke test (truncated vs intact copy).**
Determines, as fact, what `hp.read_map(field=0)` does with the truncated D2
file (raise → synthetic fallback, or partial read), and confirms D1 loads.
Command (scratchpad script `r1_map_load.py`, ~15 lines: try `hp.read_map` on
both paths, print exception or map stats).
Expected runtime: < 2 min. CPU: 1 core. RAM: ≤ ~2.5 GB peak (full-map read:
50,331,648 px × 8 B = 403 MB for I column, plus astropy/ud_grade temporaries).

**R2 — End-to-end re-run of the search with the intact map and working-tree code.**
Re-verifies the headline pipeline end to end: wrapper imports
`matched_circles` functions, calls `load_or_generate_cmb_map(nside=128,
data_path=<outer data/D1>)`, then `apply_cmb_mask`, `search_all_axes`,
`run_simulations(N_SIM=50, seed=100)`, `compute_significance`, with the exact
`__main__` grids (α = 10–50/5, twist step 5°, |b| ≥ 20°), saving npz + figures
to the scratchpad and printing the 60-s pulse.
Expected runtime: dominated by (1 + 50) × 6 pairs × 9 α × 72 twists ≈ 198,000
`circle_correlation` calls plus 50 synfast maps; order-of-magnitude estimate
15–90 min on 1 core (no wall-clock for this workload is recorded in the repo;
the estimate is not a measurement). RAM: ≤ ~2.5 GB during map load, ~0.1 GB
during the scan.

**R3 (optional) — Same as R2 with `twist_step=1.0`.**
Puts ±36° exactly on the scanned grid (matching the function default, line 333).
Runtime ≈ 5 × R2 (360 vs 72 twist values).

**R4 (optional, requires separate approval as a data change) — replace the
truncated in-repo map D2 with a copy of D1** so that the unmodified entry point
`python src/matched_circles.py` runs on real data as committed. This modifies
`tts-sdk/data/` and is therefore outside read-only scope; flagged only.

**STOP.** No run above has been executed. Awaiting explicit approval and
selection (R1 alone is cheap and settles the truncated-map behavior; R2 is the
end-to-end re-verification).

---

## 7. Open questions

1. **Which code version produced `outputs/matched_circles.npz`?** Its mtime
   (2026-07-06) predates the only git commit (2026-07-19); N_SIM was 30, not the
   current 50. Unanswerable from the repo.
2. **Why is the in-repo SMICA copy truncated at 28,732,322 bytes?** It is a
   byte-exact prefix of the intact file (§1.2 D2). No record of the copy
   operation exists. The entry point reads this truncated path.
3. **Behavior of `hp.read_map` on the truncated file** (exception → synthetic
   fallback, or partial data) is not established; proposed as R1.
4. **Publication-run parameters** ("raised for the publication run",
   `matched_circles.py:14-15, 779-780`): no config, log, or output of any
   higher-resolution run exists in the repository. If a publication figure/number
   for the circle search exists in the paper, its provenance is not in this repo.
5. **Twist grid vs the PDS prediction:** the `__main__` grid (step 5°) does not
   contain ±36° exactly, while the `pds_twist_ok` flag tests |φ_best − 36| ≤ 10.
   Whether the 5° step (vs the 1° function default) is intentional is not
   recorded. Proposed fix path: run R3; decision belongs to the authors.
6. **No beam, no noise, no Planck confidence mask, Sachs–Wolfe-only null:**
   stated in code/docstrings (§2.2); whether this is adequate for the reply's
   claims is an authors' decision, not established by this audit.
7. **No detection threshold calibration:** with N_SIM = 50 the p-value
   granularity is 0.02; no false-alarm-rate target is defined anywhere in the code.
8. **No text log of circle-search runs:** only the npz and figures (one run,
   2026-07-06). Consider adding a `summary_circles.txt` writer (proposal only).
9. **No automated tests** cover `matched_circles.py` (§1.4).
10. **Download provenance:** no dataset has a recorded download date or checksum
    in the repo; Pantheon+ files were matched to the local DataRelease clone by
    md5 during this audit (§1.2), the SMICA file only to its own FITS header.
11. **Repo naming history:** `outputs/summary_pantheon.txt` header records the
    output path under a directory named `dodeca-h0`; the remote is now
    `psldm/tts-sdk`. A prior audit document (`CIRCLES_SDK_SUMMARY.md`, untracked)
    exists in the repo root and was used as a map for—but not as a source of—the
    facts in this report; every number cited here was re-verified independently
    this session except where labeled `[claim]`.

---

*Audit performed 2026-08-04, read-only. Tools: `find`, `ls`, `git`, `md5`,
`cmp`/`head -c`, `grep`, `astropy.io.fits` (header-only), `numpy.load`, and
execution of two pure geometry/grid helper functions from the repo
(`get_dodecahedron_axes`, `cartesian_to_lonlat`, `sachs_wolfe_cls`) with no side
effects. No repository file was modified; the only file created is this report.*
