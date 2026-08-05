"""A1 (R1): map-load smoke test on the truncated and intact SMICA copies.

Read-only. Records: exception (full traceback) or successful read; on success:
npix, UNSEEN count, |T|>=1e10 count, min/max/std of remaining valid pixels.
Also prints the environment (interpreter + package versions).
"""
import sys
import traceback
import platform
import datetime

import numpy as np
import scipy
import healpy as hp
import astropy
import matplotlib

TRUNCATED = "/Users/air_eva/Desktop/tts_sdk/tts-sdk/data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
INTACT = "/Users/air_eva/Desktop/tts_sdk/data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
UNSEEN = -1.6375e30

print("=== ENVIRONMENT ===")
print(f"timestamp: {datetime.datetime.now().isoformat()}")
print(f"platform:  {platform.platform()}")
print(f"python:    {sys.version.split()[0]} ({sys.executable})")
for mod in (np, scipy, hp, astropy, matplotlib):
    print(f"{mod.__name__}: {mod.__version__}")

for label, path in [("TRUNCATED (in-repo)", TRUNCATED), ("INTACT (outer data/)", INTACT)]:
    print(f"\n=== {label}: {path}")
    try:
        m = hp.read_map(path, field=0, dtype=np.float64)
        npix = len(m)
        n_unseen = int(np.sum(np.abs(m - UNSEEN) < 1e20))
        n_huge = int(np.sum(np.abs(m) >= 1e10))
        valid = m[(np.abs(m - UNSEEN) >= 1e20) & (np.abs(m) < 1e10) & np.isfinite(m)]
        print(f"READ OK: npix={npix} (nside={hp.get_nside(m)})")
        print(f"UNSEEN-sentinel pixels (|T-(-1.6375e30)|<1e20): {n_unseen}")
        print(f"|T|>=1e10 pixels: {n_huge}")
        print(f"valid pixels: {len(valid)}")
        if len(valid):
            print(f"valid min={valid.min():.6e} max={valid.max():.6e} std={valid.std():.6e}")
    except Exception:
        print("EXCEPTION:")
        traceback.print_exc(file=sys.stdout)
