# Dodecahedral Anisotropy in the Local Hubble Flow and CMB Matched-Circle Search

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19490216-blue)](https://doi.org/10.5281/zenodo.19490216)

**Eva Moss** (RAD Systems) · **Jean-Pierre Luminet** (CNRS / LUTH, Observatoire de Paris)

Supplementary code for *"The Theory of Temporal Spheres"* (TTS v3.0), submitted.

## Abstract

The Poincaré dodecahedral space S³/I* predicts a discrete set of twelve
preferred directions on the sky and a specific matched-circle signature in
the cosmic microwave background. This repository implements two empirical
tests of that geometry: (i) a sector-wise measurement of the local Hubble
constant using Pantheon+ Type Ia supernovae (z < 0.1), with the full
STAT+SYS covariance and a Monte Carlo significance analysis under the
isotropic null; and (ii) a matched-circle correlation search on the Planck
SMICA temperature map with the π/5 holonomy of the PDS. It also reproduces
Tables 1–4 of the paper from first principles: circle geometry from
Eqs. (60)–(62), and the Sachs–Wolfe suppression spectrum S_ℓ from the
Molien series of the binary icosahedral group (Eqs. 63–65).

## Results

Running the Pantheon+ pipeline in this repository (741 SNe with z < 0.1,
12 dodecahedral sectors, full covariance) gives:

| Metric | Value |
| ------ | ----- |
| Mean H₀ | 71.24 km/s/Mpc |
| ΔH₀ (max − min across sectors) | 3.92 km/s/Mpc |
| Modulation ε | 0.0275 (2.8%) |
| MC p-value (one-sided, 1000 mocks) | 0.215 |
| Z-score | 0.61σ |

**The dodecahedral H₀ variation measured by this pipeline is not
statistically significant** (p = 0.215): under the isotropic null,
variations of this size occur in ~21% of random realizations.

This is not in contradiction with the ~6.8σ directional anisotropy reported
by Hu et al. (2024, [arXiv:2411.08528](https://arxiv.org/abs/2411.08528)),
but the two analyses answer different questions. Hu et al. fit a *dipolar*
hemisphere-type template with an optimised direction over the full Pantheon+
redshift range; this pipeline fixes twelve *a priori* dodecahedral sectors
in a fiducial orientation, restricts to z < 0.1, and uses the extreme-value
statistic ΔH₀ = max − min, which is dominated by the lowest-count sectors.
A fixed-template test has no orientation freedom to exploit and therefore
less statistical power; the honest conclusion is that this pipeline neither
confirms nor excludes the modulation amplitude ε = 0.05 ± 0.02 discussed in
the paper. A full-sky template fit marginalised over orientation is the
natural next step.

The theoretical tables (`tts_tables.py`) reproduce the published values of
TTS v3.0 Table 4 (Sachs–Wolfe approximation, k_max = 250), including the
quadrupole suppression S₂ = 0.212 and parity ratio R_oe = 1.067 at
Ω_tot = 1.0155, and verify the Molien series of I* against its closed form
up to k = 200.

## Matched-circle search

Verification runs of 2026-08-04 on the Planck PR3 SMICA temperature map
(NSIDE 2048 → 128, |b| ≥ 20° Galactic cut; map checksum in
[data/README.md](data/README.md)). Full records, per-pair tables, all 50
simulation maxima, and caveats: [RESULTS_circle_search.md](RESULTS_circle_search.md).
Run scripts, full console logs, and the pre-run audit:
[verification/2026-08-04/](verification/2026-08-04/).

| Quantity | R2 (twist step 5°) | R3 (twist step 1°) |
| --- | --- | --- |
| Observed max r | 0.815240 (α=25°, φ=115°, pair 4) | 0.836802 (α=45°, φ=6°, pair 6) |
| Simulation mean ± std (N=50) | 0.854808 ± 0.059035 | 0.895297 ± 0.046425 |
| p = count(sim ≥ obs)/50 | 38/50 = 0.76 | 45/50 = 0.90 |
| z | −0.67 | −1.26 |
| Max r at exact φ = +36° | (36° not on grid) | 0.2142 |
| Max r at exact φ = −36° (324°) | (not on grid) | 0.4784 |
| Wall-clock (Apple M2, 1 core) | 69.5 s | 319.1 s |

No matched-circle signal is detected at any scanned α (10°–50°), phase
(R3: 0°–359° in 1° steps, ±36° on grid), or axis pair.

Reproduce R2 (this is exactly the script's default configuration; writes
`outputs/matched_circles.npz` and unsuffixed `fig_*` files):

```bash
python src/matched_circles.py
```

Reproduce R3 (same functions, twist step 1.0°, same seeds):

```bash
python - <<'PY'
import sys, numpy as np
sys.path.insert(0, 'src')
from matched_circles import (load_or_generate_cmb_map, apply_cmb_mask,
    get_dodecahedron_axes, search_all_axes, run_simulations,
    compute_significance)
NSIDE, ALPHA, STEP = 128, np.arange(10, 51, 5), 1.0
cmb = load_or_generate_cmb_map(nside=NSIDE, random_seed=42,
    data_path='data/COM_CMB_IQU-smica_2048_R3.00_full.fits')
m, mask = apply_cmb_mask(cmb, gal_cut=20.0)
axes = get_dodecahedron_axes()
res = search_all_axes(m, axes, ALPHA, NSIDE, mask=mask, twist_step=STEP)
best = max(res, key=lambda r: r['corr_best'])
sims = run_simulations(50, NSIDE, axes, ALPHA, mask=mask,
    twist_step=STEP, random_seed=100)
sig = compute_significance(best['corr_best'], sims)
print(f"max r={best['corr_best']:.6f} at alpha={best['alpha_best']:.0f}, "
      f"twist={best['twist_best']:.0f}; p={sig['p_value']:.4f}, "
      f"z={sig['z_score']:.4f}")
PY
```

Both runs require the intact SMICA map in `data/` (md5-verify it per
data/README.md): if the file is missing or unreadable the script falls back
to a synthetic map and the result does not test real data.

## Repository structure

```text
tts-sdk/
├── src/
│   ├── dodecahedron.py     # geometry: 12 face normals, sector assignment, catalog loaders
│   ├── h0_fit.py           # per-sector H0 fits with full covariance
│   ├── simulations.py      # Monte Carlo null distribution (isotropic mocks)
│   ├── visualize.py        # publication figures (Mollweide, bars, MC, summary)
│   ├── run_all.py          # end-to-end pipeline (Pantheon+ / Union3)
│   ├── audit.py            # self-audit: 7 blocks of internal consistency checks
│   ├── tts_tables.py       # Tables 1-4 of TTS v3.0 from first principles
│   ├── matched_circles.py  # CMB matched-circle search (Planck SMICA, π/5 twist)
│   ├── rbsg08.py           # Roukema et al. (2008) cross-correlation test (synthetic map)
│   ├── cosmicflows.py      # CosmicFlows-4 peculiar-velocity cross-check
│   └── sirens.py           # gravitational-wave sirens / lensed-quasar comparison
├── tests/                  # pytest suite
├── notebooks/analysis.ipynb
├── data/                   # not committed — see data/README.md
└── outputs/                # generated figures and result arrays
```

## Installation

```bash
git clone https://github.com/psldm/tts-sdk.git
cd tts-sdk
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the data files as described in [data/README.md](data/README.md).

## Reproducibility

Every published number in the Results table above is regenerated by:

```bash
# Full Pantheon+ pipeline: sectors → H0 fits → 1000-mock MC → z-cut → figures
python src/run_all.py --catalog pantheon      # ~1 min on 8 cores

# Union3 cross-check and catalog comparison
python src/run_all.py --catalog all

# Tables 1-4 of the paper (writes outputs/tts_tables.json)
python src/tts_tables.py                      # ~2 min

# Internal consistency audit (geometry, covariance, MC normality, ...)
python src/audit.py

# CMB matched-circle search (requires the SMICA map; reduced defaults
# NSIDE=128, N_SIM=50 — increase in-script for a publication run)
python src/matched_circles.py

# Optional cross-checks
python src/cosmicflows.py
python src/sirens.py
python src/rbsg08.py

# Test suite
pytest tests/
```

All Monte Carlo runs are seeded (seed 42) and reproduce bit-identical
summary statistics on a fixed software stack (see `requirements.txt`).

## Data

| Dataset | Source |
| ------- | ------ |
| Pantheon+ SNe Ia + STAT+SYS covariance | [PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease) |
| Planck SMICA CMB map (PR3) | [Planck Legacy Archive](https://pla.esac.esa.int/) |
| Union3 / UNITY1.5 | [rubind/union3_release](https://github.com/rubind/union3_release) (Rubin et al. 2023) |
| CosmicFlows-4 | [Extragalactic Distance Database](https://edd.ifa.hawaii.edu/) (Tully et al. 2023) |

Caveats stated plainly: `rbsg08.py` and the fallback path of
`matched_circles.py` operate on *synthetic* scale-invariant Gaussian maps
(no acoustic physics) and validate the machinery, not the hypothesis;
`sirens.py` uses illustrative literature-derived per-event H₀ values; the
CosmicFlows-4 module requires the real catalog for any scientific claim
(see data/README.md).

## Citation

If you use this code, please cite:

```bibtex
@article{MossLuminet2026TTS,
  author  = {Moss, Eva and Luminet, Jean-Pierre},
  title   = {The Theory of Temporal Spheres},
  year    = {2026},
  note    = {v3.0, submitted}
}

@misc{Moss2026E8,
  author = {Moss, Eva},
  title  = {From Dodecahedron to E8: A Dictionary between Cosmic Topology
            and Exceptional Geometry},
  year   = {2026},
  doi    = {10.5281/zenodo.19490216}
}

@misc{Moss2026CRT,
  author = {Moss, Eva},
  title  = {The Subgroup Ladder is the Chinese Remainder Theorem:
            Base 60 as the Canonical Coordinate of 2I},
  year   = {2026},
  doi    = {10.5281/zenodo.20135714}
}
```

## License

MIT — see [LICENSE](LICENSE).
