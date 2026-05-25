import numpy as np

from cbct_siddon.phantom import build_phantom, modified_shepp_logan_table


def test_shape():
    n = 64
    v = build_phantom(n)
    assert v.shape == (n, n, n)


def test_support():
    v = build_phantom(64)
    assert v[0, 0, 0] == 0.0
    assert v[-1, -1, -1] == 0.0


def test_center_positive():
    v = build_phantom(128)
    c = v.shape[0] // 2
    assert v[c, c, c] > 0.0


def test_interior_structure():
    """Phantom has interior low-density and high-density regions."""
    v = build_phantom(64)
    assert v.max() >= 0.9
    assert np.count_nonzero(v > 0.15) > 500
    assert np.count_nonzero(v < 0.5) > 500


def test_table_shape():
    e = modified_shepp_logan_table()
    assert e.shape == (10, 10)
