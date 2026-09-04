"""
Test 1 & 2: Reflection physics and mirror tilt behavior.
"""

import numpy as np
import pytest
from optics.ray import RayBundle, RayStatus
from optics.elements import PlaneMirror


def test_angle_of_incidence_equals_angle_of_reflection():
    """
    TEST 1: Angle of incidence = angle of reflection for general 3D surface normal.
    """
    # Create mirror tilted at 10 deg tip_x and 15 deg tilt_y with ample clear aperture
    mirror = PlaneMirror(
        name="TestMirror",
        center=(0.0, 0.0, 100.0),
        width=100.0,
        height=100.0,
        tip_x_deg=10.0,
        tilt_y_deg=15.0,
    )

    # Test an array of arbitrary incidence angles
    rng = np.random.default_rng(123)
    n_test = 50

    # Small divergence angles aimed towards mirror center
    kx = rng.uniform(-0.1, 0.1, n_test)
    ky = rng.uniform(-0.1, 0.1, n_test)
    kz = np.sqrt(1.0 - kx**2 - ky**2)
    k_in = np.column_stack([kx, ky, kz])

    r_in = np.zeros((n_test, 3))  # Starting at z = 0

    bundle = RayBundle(r=r_in, k=k_in)
    mirror.trace(bundle)

    # Active reflected rays
    reflected_mask = bundle.active_mask
    assert np.all(reflected_mask), "All rays aimed at mirror center should hit"

    k_out = bundle.k
    normal = mirror.normal

    # Angle of incidence theta_i: cos(theta_i) = |k_in . n|
    cos_theta_i = np.abs(np.dot(k_in, normal))
    # Angle of reflection theta_r: cos(theta_r) = |k_out . n|
    cos_theta_r = np.abs(np.dot(k_out, normal))

    np.testing.assert_allclose(
        cos_theta_i,
        cos_theta_r,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Angle of incidence must equal angle of reflection",
    )

    # Also verify the vector reflection equation: k_out = k_in - 2 * (k_in . n) * n
    expected_k_out = k_in - 2.0 * np.sum(k_in * normal, axis=1, keepdims=True) * normal
    expected_k_out = expected_k_out / np.linalg.norm(expected_k_out, axis=1, keepdims=True)
    np.testing.assert_allclose(
        k_out,
        expected_k_out,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Reflected vector must strictly satisfy specular reflection law",
    )


def test_mirror_tilt_deflection():
    """
    TEST 2: Rotating a mirror by delta should change reflected chief-ray angle by approximately 2 * delta.
    """
    delta_deg = 2.5  # 2.5 degree tilt
    delta_rad = np.radians(delta_deg)

    # Flat mirror perpendicular to optical axis (facing -z)
    mirror_flat = PlaneMirror(
        name="FlatMirror",
        center=(0.0, 0.0, 100.0),
        width=50.0,
        height=50.0,
        tip_x_deg=0.0,
        tilt_y_deg=0.0,
    )

    # Tilted mirror by delta_deg about y-axis
    mirror_tilted = PlaneMirror(
        name="TiltedMirror",
        center=(0.0, 0.0, 100.0),
        width=50.0,
        height=50.0,
        tip_x_deg=0.0,
        tilt_y_deg=delta_deg,
    )

    # Incident chief ray along +z: k = [0, 0, 1]
    r_in = np.array([[0.0, 0.0, 0.0]])
    k_in = np.array([[0.0, 0.0, 1.0]])

    bundle_flat = RayBundle(r=r_in.copy(), k=k_in.copy())
    bundle_tilted = RayBundle(r=r_in.copy(), k=k_in.copy())

    mirror_flat.trace(bundle_flat)
    mirror_tilted.trace(bundle_tilted)

    k_out_flat = bundle_flat.k[0]
    k_out_tilted = bundle_tilted.k[0]

    # Angle between reflected rays
    dot_prod = np.clip(np.dot(k_out_flat, k_out_tilted), -1.0, 1.0)
    angular_deflection_rad = np.arccos(dot_prod)
    angular_deflection_deg = np.degrees(angular_deflection_rad)

    # Reflected beam deflection angle is exactly 2 * delta for pure rotation
    expected_deflection_deg = 2.0 * delta_deg
    assert np.isclose(angular_deflection_deg, expected_deflection_deg, atol=1e-5), (
        f"Expected {expected_deflection_deg} deg deflection, got {angular_deflection_deg} deg"
    )
