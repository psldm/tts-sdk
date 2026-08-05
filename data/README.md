# Data

The data files are not committed to this repository because of their size.
Download them into this directory before running the pipeline.

## Pantheon+ (required for the H0 anisotropy pipeline)

From the official [PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease):

```bash
wget https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
wget https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov
```

Reference checksums (recorded in the 2026-08-04 audit, `REPORT_circle_search.md`
§1.2; files verified identical to the DataRelease copies at upstream commit
`c447f0f`):

| file | bytes | md5 |
| --- | --- | --- |
| `Pantheon+SH0ES.dat` | 579,283 | `2049b142e6aad384470b3364aa10f3fa` |
| `Pantheon+SH0ES_STAT+SYS.cov` | 33,284,960 | `041bdb6638841794fc2d7caa88dd66eb` |

## Planck SMICA CMB map (required for the matched-circle search)

`COM_CMB_IQU-smica_2048_R3.00_full.fits` from the
[Planck Legacy Archive](https://pla.esac.esa.int/) (Maps → CMB maps → SMICA,
PR3/2018 release, ~2 GB).

**This 2 GB map is not shipped in the repository** (`data/*` is git-ignored).
Download it from the PLA into this `data/` directory before running
`src/matched_circles.py`. Verify the file after download — an incomplete
download fails to load and the code then silently falls back to a synthetic
map (see the warning it prints):

| file | bytes | md5 |
| --- | --- | --- |
| `COM_CMB_IQU-smica_2048_R3.00_full.fits` | 2,013,312,960 | `ee2fc49a2eb70c2eca0d582e4aae5d05` |

FITS header of the reference copy: `NSIDE=2048`, `ORDERING='NESTED'`,
`COORDSYS='GALACTIC'`, `DATE='2018-04-10'`, field 0 = `I_STOKES` in `K_CMB`.

## Union3 (optional, `run_all.py --catalog union3`)

`union3_inputs.pickle` from the Union3/UNITY1.5 release accompanying
Rubin et al. (2023), [arXiv:2311.12098](https://arxiv.org/abs/2311.12098):
<https://github.com/rubind/union3_release>.

## CosmicFlows-4 (optional, `cosmicflows.py`)

Real CF4 distances and velocities: Tully et al. (2023), via the
[Extragalactic Distance Database](https://edd.ifa.hawaii.edu/) — export a CSV
named `CF4_data.csv` with columns
`ra, dec, distance, distance_err, v_obs, distance_method`.

If `CF4_data.csv` is absent, the module generates `CF4_synthetic.csv` — an
isotropic mock with no dodecahedral signal by construction — and prints a
prominent warning. Synthetic data validate the pipeline only; they carry no
cosmological information.
