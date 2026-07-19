# Data

The data files are not committed to this repository because of their size.
Download them into this directory before running the pipeline.

## Pantheon+ (required for the H0 anisotropy pipeline)

From the official [PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease):

```bash
wget https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
wget https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov
```

## Planck SMICA CMB map (required for the matched-circle search)

`COM_CMB_IQU-smica_2048_R3.00_full.fits` from the
[Planck Legacy Archive](https://pla.esac.esa.int/) (Maps → CMB maps → SMICA,
PR3/2018 release, ~2 GB).

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
