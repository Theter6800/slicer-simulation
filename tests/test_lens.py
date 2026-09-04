"""
Test 3 & 4: Ideal thin lens focusing and paraxial slope shift.
"""

import numpy as np
import pytest
from optics.ray import RayBundle, RayStatus
from optics.elements import ThinLens


def test_parallel_on_axis_rays_focus_at_f():
    """
    TEST 3: Parallel on-axis rays through an ideal thin lens must focus at z = z_lens + f.
    """
    f = 100.0  # mm
    z_lens = 50.0  # mm
    lens = ThinLens(name="TestLens", z=z_lens, focal_length=f, diameter=25.4)

    # Collimated parallel beam along z axis within clear aperture (radius 12.7 mm)
    n_rays = 15
    x = np.linspace(-7.0, 7.0, n_rays)
    y = np.linspace(-7.0, 7.0, n_rays)
    z = np.zeros(n_rays)

    r = np.column_stack([x, y, z])
    k = np.zeros((n_rays, 3))
    k[:, 2] = 1.0  # k = [0, 0, 1]

    bundle = RayBundle(r=r, k=k)
    lens.trace(bundle)

    # Propagate to focal plane at z = z_lens + f
    z_focus = z_lens + f
    bundle.propagate_to_z(z_focus)

    # Check that transverse coordinates (x, y) at focal plane are essentially 0
    active = bundle.active_mask
    assert np.all(active), "All rays within diameter should pass lens"

    # Paraxial formula: u_out = -x/f. At z = z_lens + f, dx = u_out * f = -x, so x_final = x + dx = 0.
    np.testing.assert_allclose(
        bundle.x[active],
        0.0,
        atol=1e-7,
        err_msg="Parallel on-axis rays must converge to x = 0 at focal plane",
    )
    np.testing.assert_allclose(
        bundle.y[active],
        0.0,
        atol=1e-7,
        err_msg="Parallel on-axis rays must converge to y = 0 at focal plane",
    )


def test_off_axis_ray_paraxial_transformation():
    """
    TEST 4: Thin-lens transformation produces expected paraxial slopes.
    u_out = u_in - x/f
    v_out = v_in - y/f
    """
    f = 75.0
    z_lens = 100.0
    lens = ThinLens(name="Lens75", z=z_lens, focal_length=f, diameter=50.0)

    # Incident ray with non-zero slope: u_in = 0.05, v_in = -0.02
    u_in = 0.05
    v_in = -0.02
    kz = 1.0 / np.sqrt(u_in**2 + v_in**2 + 1.0)
    kx = u_in * kz
    ky = v_in * kz

    # Incident position at lens plane: x = 5.0 mm, y = -3.0 mm
    r0 = np.array([[5.0, -3.0, z_lens]])
    k0 = np.array([[kx, ky, kz]])

    bundle = RayBundle(r=r0, k=k0)
    lens.trace(bundle)

    # Expected output slopes
    expected_u_out = u_in - 5.0 / f
    expected_v_out = v_in - (-3.0) / f

    actual_u_out = bundle.kx[0] / bundle.kz[0]
    actual_v_out = bundle.ky[0] / bundle.kz[0]

    assert np.isclose(actual_u_out, expected_u_out, atol=1e-8), (
        f"Expected u_out {expected_u_out}, got {actual_u_out}"
    )
    assert np.isclose(actual_v_out, expected_v_out, atol=1e-8), (
        f"Expected v_out {expected_v_out}, got {actual_v_out}"
    )
