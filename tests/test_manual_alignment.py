"""
Unit tests for Manual Channel Alignment controls and diagnostics.
"""
import numpy as np
import pytest
from optics.slicer import SliceMirror
from optics.pupil import PupilMirror
from optics.elements import ThinLens3D
from optics.geometry3d import normalize

def test_slicer_fine_angle_adjustment():
    """Verify that fine angular increments (down to 0.001 deg) produce predictable ray deflections."""
    slicer = SliceMirror(
        slice_id=0,
        width=2.0,
        height=0.5,
        center_x=0.0,
        center_y=0.0,
        z=50.0,
        tip_x_deg=0.0,
        tilt_y_deg=0.0,
        rot_z_deg=0.0
    )
    
    k_in = np.array([0.0, 0.0, 1.0])
    
    # 0 deg: reflected ray is [0, 0, -1]
    k_ref0 = slicer.get_reflected_chief_ray(k_in)
    np.testing.assert_allclose(k_ref0, [0.0, 0.0, -1.0], atol=1e-6)
    
    # Adjust tip_x by 0.001 deg
    slicer.set_tilt(tip_x_deg=0.001, tilt_y_deg=0.0, rot_z_deg=0.0)
    k_ref1 = slicer.get_reflected_chief_ray(k_in)
    
    # Reflected beam angle deflection should be 2 * 0.001 deg
    # dot product between -k_ref0 (which is [0,0,1]) and k_ref1
    # or angle between k_ref0 and k_ref1
    dot_val = np.clip(np.dot(k_ref0, k_ref1), -1.0, 1.0)
    deflection_rad = np.arccos(dot_val)
    expected_rad = np.radians(2 * 0.001)
    assert np.isclose(deflection_rad, expected_rad, rtol=1e-3)
    
    # At distance L = 70 mm, transverse shift should be L * tan(2 * 0.001 deg)
    shift_mm = 70.0 * np.tan(expected_rad)
    assert 0.002 < shift_mm < 0.003  # ~2.44 microns

def test_pupil_get_local_hit():
    """Verify local (u, v) and miss distance calculation on the pupil mirror face."""
    pupil = PupilMirror(
        channel_id=0,
        name="Pupil_0",
        center=np.array([0.0, -20.0, 90.0]),
        normal=normalize(np.array([0.0, 0.5, -0.8660254])), # tilted
        width=12.0,
        height=12.0
    )
    
    # Direct hit at center along normal
    r_hit_center = pupil.center - 10.0 * pupil.normal
    k_in = pupil.normal
    u, v, miss, in_bounds, hit_pos = pupil.get_local_hit(r_hit_center, k_in)
    assert in_bounds is True
    assert np.isclose(miss, 0.0, atol=1e-5)
    assert np.isclose(u, 0.0, atol=1e-5)
    assert np.isclose(v, 0.0, atol=1e-5)
    
    # Off-center hit by u=+2.0, v=-3.0
    r_target = pupil.center + 2.0 * pupil.u_vec - 3.0 * pupil.v_vec
    r_start = r_target - 10.0 * pupil.normal
    u_off, v_off, miss_off, in_bounds_off, hit_pos_off = pupil.get_local_hit(r_start, k_in)
    assert in_bounds_off is True
    assert np.isclose(u_off, 2.0, atol=1e-5)
    assert np.isclose(v_off, -3.0, atol=1e-5)
    assert np.isclose(miss_off, np.sqrt(4.0 + 9.0), atol=1e-5)
    
    # Out of bounds hit (u = 10.0, half-width is 6.0)
    r_oob = pupil.center + 10.0 * pupil.u_vec
    r_start_oob = r_oob - 10.0 * pupil.normal
    u_oob, v_oob, miss_oob, in_bounds_oob, _ = pupil.get_local_hit(r_start_oob, k_in)
    assert in_bounds_oob is False
    assert np.isclose(u_oob, 10.0, atol=1e-5)

def test_slicer_ideal_angles_calculation():
    """Verify that get_ideal_angles_for_target calculates tip/tilt that points exactly at pupil center."""
    slicer = SliceMirror(
        slice_id=0,
        width=2.0,
        height=0.5,
        center_x=0.0,
        center_y=0.5,
        z=50.0,
        tip_x_deg=0.0,
        tilt_y_deg=0.0
    )
    pupil_center = np.array([0.0, -25.0, 95.0])
    k_in = np.array([0.0, 0.0, 1.0])
    
    tip_ideal, tilt_ideal = slicer.get_ideal_angles_for_target(pupil_center, k_in=k_in)
    
    # Apply ideal angles
    slicer.set_tilt(tip_x_deg=tip_ideal, tilt_y_deg=tilt_ideal)
    k_ref = slicer.get_reflected_chief_ray(k_in)
    
    # Reflected ray direction must align with vector from slicer center to pupil center
    v_target = pupil_center - slicer.center
    v_target_dir = normalize(v_target)
    cos_err = np.dot(k_ref, v_target_dir)
    assert np.isclose(cos_err, 1.0, atol=1e-5)

def test_reflection_diagnostics():
    """Verify reflection diagnostics computation: normal, theta_i, theta_r, pointing error."""
    slicer = SliceMirror(
        slice_id=0,
        width=2.0,
        height=0.5,
        center_x=0.0,
        center_y=0.0,
        z=50.0,
        tip_x_deg=-15.0,
        tilt_y_deg=5.0
    )
    diag = slicer.get_reflection_diagnostics(k_in=np.array([0.0, 0.0, 1.0]))
    
    assert np.isclose(diag['theta_i_deg'], diag['theta_r_deg'], atol=1e-5)
    assert 'normal' in diag
    assert 'k_ref' in diag
    assert diag['verified'] is True

def test_condenser_lens_local_hit():
    """Verify ThinLens3D local hit coordinates on condenser aperture."""
    lens = ThinLens3D(
        name="Condenser",
        center=(0.0, -30.0, 150.0),
        focal_length=75.0,
        diameter=50.0,
        normal=(0.0, -0.2, 0.9797959) # slightly tilted relay axis
    )
    r_start = lens.center - 50.0 * lens.normal
    k_in = lens.normal
    
    u, v, miss, in_bounds, hit_pos = lens.get_local_hit(r_start, k_in)
    assert in_bounds is True
    assert np.isclose(miss, 0.0, atol=1e-4)
