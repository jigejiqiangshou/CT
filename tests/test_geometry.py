import numpy as np

from cbct_siddon.geometry import ConeBeamGeometry


def test_source_on_circle():
    geom = ConeBeamGeometry(dso=2.0)
    s = geom.source_position(0.0)
    r = np.hypot(s[0], s[1])
    assert np.isclose(r, geom.dso)
    assert np.isclose(s[2], 0.0)


def test_ray_direction():
    geom = ConeBeamGeometry(dso=2.0, dsd=4.0)
    beta = 0.5
    s = geom.source_position(beta)
    p = geom.detector_pixel(beta, 0.0, 0.0)
    d = p - s
    d /= np.linalg.norm(d)
    u_iso = -s / np.linalg.norm(s)
    assert np.allclose(d, u_iso, atol=1e-10)


def test_detector_behind_source():
    geom = ConeBeamGeometry(dso=2.0, dsd=4.0)
    beta = 0.0
    s = geom.source_position(beta)
    p = geom.detector_pixel(beta, 0.0, 0.0)
    dist = np.linalg.norm(p - s)
    assert np.isclose(dist, geom.dsd, rtol=1e-6)
