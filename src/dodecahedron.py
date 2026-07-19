#!/usr/bin/env python3
"""
Dodecahedron geometry module for H0 anisotropy analysis.

Provides the 12 face normals of a regular dodecahedron, celestial-to-Cartesian
coordinate conversion, nearest-face sector assignment, and loaders for the
Pantheon+ and Union3 supernova catalogs.

The face-normal orientation is the fixed golden-ratio (icosahedron-vertex)
configuration; it is a fiducial choice and is not marginalised over.
"""

import gzip
import pickle as pkl

import numpy as np
import pandas as pd


def get_dodecahedron_normals() -> np.ndarray:
    """
    Generate unit normal vectors for the 12 faces of a regular dodecahedron.

    The normals are the vertices of the dual icosahedron: the icosahedron
    vertices are the face normals of the dodecahedron.

    Returns
    -------
    numpy.ndarray
        Array of shape (12, 3) containing unit normal vectors.
    """
    phi = (1 + np.sqrt(5)) / 2

    vertices = []
    # (0, +/-1, +/-phi)
    vertices.extend([[0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi]])
    # (+/-1, +/-phi, 0)
    vertices.extend([[1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0]])
    # (+/-phi, 0, +/-1)
    vertices.extend([[phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]])

    vertices = np.array(vertices)
    normals = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]

    return normals


def radec_to_cartesian(
    ra_deg: np.ndarray | float, dec_deg: np.ndarray | float
) -> np.ndarray:
    """
    Convert equatorial coordinates (RA, Dec) to Cartesian unit vectors.

    Parameters
    ----------
    ra_deg : float or array-like
        Right ascension in degrees.
    dec_deg : float or array-like
        Declination in degrees.

    Returns
    -------
    numpy.ndarray
        Array of shape (3,) for scalar input, or (N, 3) for array input,
        containing Cartesian unit vectors.
    """
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    if np.isscalar(ra_deg):
        return np.array([x, y, z])
    return np.column_stack([x, y, z])


def assign_sectors(ra: np.ndarray, dec: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Assign supernovae to dodecahedron sectors by angular proximity.

    For each supernova, the angular distance to each of the 12 face normals
    is computed and the closest sector is selected.

    Parameters
    ----------
    ra : array-like
        Right ascension values in degrees.
    dec : array-like
        Declination values in degrees.
    normals : numpy.ndarray
        Array of shape (12, 3) with face normals.

    Returns
    -------
    numpy.ndarray
        Sector IDs (0-11) for each supernova.
    """
    positions = radec_to_cartesian(ra, dec)

    # Dot products give cos(theta); the largest value is the nearest normal.
    cos_distances = np.dot(positions, normals.T)
    cos_distances = np.clip(cos_distances, -1.0, 1.0)
    sector_ids = np.argmax(cos_distances, axis=1)

    return sector_ids


def load_pantheon_data(filepath: str) -> pd.DataFrame:
    """
    Load and filter Pantheon+ supernova data.

    Parameters
    ----------
    filepath : str
        Path to the Pantheon+ catalog file (whitespace-separated).

    Returns
    -------
    pandas.DataFrame
        SNe with z < 0.1 and columns ['z', 'ra', 'dec', 'mb']. Positional
        row order follows the original catalog order of the surviving SNe,
        which is required for covariance-matrix alignment.
    """
    df = pd.read_csv(filepath, sep=r"\s+")

    df_filtered = df[df["zHD"] < 0.1].copy()

    df_result = df_filtered.rename(
        columns={"zHD": "z", "RA": "ra", "DEC": "dec", "m_b_corr": "mb"}
    )[["z", "ra", "dec", "mb"]].reset_index(drop=True)

    return df_result


def _parse_sexagesimal(ra_val, dec_val) -> tuple[float, float]:
    """
    Convert RA/Dec to decimal degrees.

    Handles both sexagesimal strings ('HH:MM:SS.S', '+DD:MM:SS.S') and
    already-decimal float values.

    Parameters
    ----------
    ra_val : str or float
        RA value.
    dec_val : str or float
        Dec value.

    Returns
    -------
    tuple of float
        (ra_deg, dec_deg) in decimal degrees.
    """
    if isinstance(ra_val, (int, float, np.floating)):
        return float(ra_val), float(dec_val)

    ra_parts = str(ra_val).strip().split(":")
    ra_deg = 15.0 * (
        float(ra_parts[0]) + float(ra_parts[1]) / 60.0 + float(ra_parts[2]) / 3600.0
    )

    dec_str = str(dec_val).strip()
    sign = -1 if dec_str[0] == "-" else 1
    dec_body = dec_str[1:] if dec_str[0] in "+-" else dec_str
    dec_parts = dec_body.split(":")
    dec_deg = sign * (
        float(dec_parts[0]) + float(dec_parts[1]) / 60.0 + float(dec_parts[2]) / 3600.0
    )

    return ra_deg, dec_deg


def load_union3_data(pickle_path: str) -> pd.DataFrame:
    """
    Load and filter Union3 supernova data from the release pickle file.

    Parameters
    ----------
    pickle_path : str
        Path to the Union3 inputs pickle file (gzipped).

    Returns
    -------
    pandas.DataFrame
        SNe with z < 0.1 and columns ['z', 'ra', 'dec', 'mb'].
    """
    with gzip.open(pickle_path, "rb") as f:
        data = pkl.load(f)

    d0 = data[0]

    z_helio = d0["z_helio_list"]
    mb = d0["mB_list"]
    ra_strs = d0["RA"]
    dec_strs = d0["Dec"]

    ra_deg = np.zeros(len(ra_strs))
    dec_deg = np.zeros(len(dec_strs))
    for i, (ra_s, dec_s) in enumerate(zip(ra_strs, dec_strs)):
        ra_deg[i], dec_deg[i] = _parse_sexagesimal(ra_s, dec_s)

    df_all = pd.DataFrame({"z": z_helio, "ra": ra_deg, "dec": dec_deg, "mb": mb})
    df_filtered = df_all[df_all["z"] < 0.1].copy().reset_index(drop=True)

    return df_filtered


if __name__ == "__main__":
    import os

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "Pantheon+SH0ES.dat"
    )
    try:
        data = load_pantheon_data(data_path)
        print(f"Loaded {len(data)} supernovae with z < 0.1")

        normals = get_dodecahedron_normals()
        print(f"Generated {len(normals)} dodecahedron face normals")

        sector_ids = assign_sectors(data["ra"].values, data["dec"].values, normals)
        unique, counts = np.unique(sector_ids, return_counts=True)

        print("\nSector | N_SNe")
        print("-------|------")
        for sector_id, count in zip(unique, counts):
            print(f"  {sector_id:4d} | {count:5d}")
        print(f"\nTotal supernovae assigned: {len(sector_ids)}")
    except FileNotFoundError:
        print("Pantheon+ data file not found; see data/README.md for download links.")
