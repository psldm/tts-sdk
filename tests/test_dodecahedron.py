"""Tests for the dodecahedron geometry module."""

import numpy as np

from dodecahedron import (
    assign_sectors,
    get_dodecahedron_normals,
    radec_to_cartesian,
)


def test_normals_are_unit_vectors():
    normals = get_dodecahedron_normals()
    assert normals.shape == (12, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_normals_form_antipodal_pairs():
    normals = get_dodecahedron_normals()
    dots = normals @ normals.T
    # Every face normal has exactly one antipodal partner (dot = -1).
    assert np.allclose(dots.min(axis=1), -1.0)


def test_radec_conversion_cardinal_points():
    assert np.allclose(radec_to_cartesian(0, 0), [1, 0, 0])
    assert np.allclose(radec_to_cartesian(90, 0), [0, 1, 0])
    assert np.allclose(radec_to_cartesian(0, 90), [0, 0, 1])


def test_assign_sectors_recovers_normal_directions():
    normals = get_dodecahedron_normals()
    # A point placed exactly at each face normal must land in that sector.
    dec = np.degrees(np.arcsin(np.clip(normals[:, 2], -1.0, 1.0)))
    ra = np.degrees(np.arctan2(normals[:, 1], normals[:, 0])) % 360
    sector_ids = assign_sectors(ra, dec, normals)
    assert list(sector_ids) == list(range(12))
