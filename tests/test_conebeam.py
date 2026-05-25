import numpy as np

from cbct_siddon.geometry import ConeBeamGeometry
from cbct_siddon.phantom import build_phantom
from cbct_siddon.project import forward_project


def _small_geom() -> ConeBeamGeometry:
    return ConeBeamGeometry(
        dso=2.0,
        dsd=4.0,
        det_nu=16,
        det_nv=16,
        du=0.08,
        dv=0.08,
        n_angles=4,
        volume_n=32,
    )


def test_projection_shape():
    geom = _small_geom()
    vol = build_phantom(geom.volume_n)
    proj = forward_project(vol, geom)
    assert proj.shape == (geom.n_angles, geom.det_nv, geom.det_nu)


def test_no_nan_inf():
    geom = _small_geom()
    vol = build_phantom(geom.volume_n)
    proj = forward_project(vol, geom)
    assert np.all(np.isfinite(proj))


def test_air_view():
    geom = _small_geom()
    vol = np.zeros((geom.volume_n,) * 3)
    proj = forward_project(vol, geom)
    assert np.allclose(proj, 0.0)


def test_single_view_symmetry():
    geom = ConeBeamGeometry(
        dso=2.0,
        dsd=4.0,
        det_nu=32,
        det_nv=32,
        du=0.06,
        dv=0.06,
        n_angles=1,
        volume_n=48,
    )
    vol = build_phantom(geom.volume_n)
    proj = forward_project(vol, geom)[0]
    assert np.allclose(proj, proj[:, ::-1], rtol=0.15, atol=0.05)


def test_monotonicity_skull():
    geom = ConeBeamGeometry(
        dso=2.0,
        dsd=4.0,
        det_nu=24,
        det_nv=24,
        du=0.08,
        dv=0.08,
        n_angles=1,
        volume_n=48,
    )
    vol = build_phantom(geom.volume_n)
    proj = forward_project(vol, geom)[0]
    center = proj[geom.det_nv // 2, geom.det_nu // 2]
    corner = proj[0, 0]
    assert center > corner
