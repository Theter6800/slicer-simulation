"""
Test the 7 Critical Debug Checks for the 3D Branched Non-Sequential Architecture.
"""

import numpy as np
import pytest
from optics.ray import RayBundle, RayStatus
from optics.elements import ThinLens3D, PlaneMirror3D
from optics.slicer import SliceMirror, SlicerArray
from optics.pupil import PupilMirror, PupilRelaySystem
from optics.geometry3d import normalize, reflect_vector, reflection_bisector
from optics.presets import create_branched_slicer_system


def test_check_1_slicer_tilt_deflection():
    """
    CHECK 1: Change slicer tilt by +1 degree.
    Its reflected chief ray must change by approximately +2 degrees.
    """
    s_base = SliceMirror(
        slice_id=0,
        width=10.0,
        height=10.0,
        center_x=0.0,
        center_y=0.0,
        z=100.0,
        tip_x_deg=2.0,
        tilt_y_deg=0.0,
    )
    s_tilted = SliceMirror(
        slice_id=0,
        width=10.0,
        height=10.0,
        center_x=0.0,
        center_y=0.0,
        z=100.0,
        tip_x_deg=3.0,  # +1.0 deg tip
        tilt_y_deg=0.0,
    )

    k_in = np.array([0.0, 0.0, 1.0])
    k_ref_base = s_base.get_reflected_chief_ray(k_in)
    k_ref_tilted = s_tilted.get_reflected_chief_ray(k_in)

    # Angle between reflected rays
    dot_val = np.clip(np.dot(k_ref_base, k_ref_tilted), -1.0, 1.0)
    angular_change_deg = np.degrees(np.arccos(dot_val))

    assert np.isclose(angular_change_deg, 2.0, atol=1e-4), (
        f"Expected ~2.0 deg deflection for +1.0 deg mirror tilt, got {angular_change_deg:.4f} deg"
    )


def test_check_2_pupil_center_on_reflected_chief_ray():
    """
    CHECK 2: Pupil mirror center must lie on its slicer reflected chief ray.
    Compute perpendicular error; target approximately zero.
    """
    s = SliceMirror(
        slice_id=0,
        width=10.0,
        height=10.0,
        center_x=-5.0,
        center_y=5.0,
        z=190.0,
        tip_x_deg=2.5,
        tilt_y_deg=-1.5,
    )
    k_in = np.array([0.0, 0.0, 1.0])
    k_ref = s.get_reflected_chief_ray(k_in)

    L = 65.0  # Pupil distance along reflected beam
    P = s.compute_pupil_position(distance=L, k_in=k_in)

    # Vector from slicer center S to pupil center P
    SP = P - s.center
    # Perpendicular error = || SP - (SP . k_ref) * k_ref ||
    proj = np.dot(SP, k_ref) * k_ref
    perp_error = np.linalg.norm(SP - proj)

    assert np.isclose(perp_error, 0.0, atol=1e-12), (
        f"Perpendicular error between pupil center and reflected chief ray is {perp_error}, expected 0"
    )
    assert np.isclose(np.linalg.norm(SP), L, atol=1e-10), "Distance S to P must equal L"


def test_check_3_pupil_chief_ray_points_to_final_lens():
    """
    CHECK 3: Chief ray after pupil mirror must point toward the requested final lens location.
    """
    s = SliceMirror(
        slice_id=0,
        width=10.0,
        height=10.0,
        center_x=0.0,
        center_y=5.0,
        z=190.0,
        tip_x_deg=3.0,
        tilt_y_deg=0.0,
    )
    k_in = np.array([0.0, 0.0, 1.0])
    k_ref = s.get_reflected_chief_ray(k_in)
    P = s.compute_pupil_position(distance=50.0, k_in=k_in)

    final_lens_center = np.array([0.0, 0.0, 310.0])

    pm = PupilMirror(
        channel_id=0,
        name="TestPupil",
        center=P,
        normal=np.array([0.0, 0.0, -1.0]),
        width=15.0,
        height=15.0,
    )
    # Aim pupil mirror towards final lens center
    pm.aim_towards(k_ref, final_lens_center)

    # Reflected direction from pupil mirror
    k_out = reflect_vector(k_ref, pm.normal)

    # Expected direction from P to final lens center
    expected_k_out = normalize(final_lens_center - P)

    dot_target = np.clip(np.dot(k_out, expected_k_out), -1.0, 1.0)
    angle_error_deg = np.degrees(np.arccos(dot_target))

    assert np.isclose(angle_error_deg, 0.0, atol=1e-6), (
        f"Pupil reflected chief ray deviates from final lens center by {angle_error_deg} deg"
    )


def test_check_4_no_backward_ray_interaction():
    """
    CHECK 4: No ray should interact with an optical surface behind it (t > epsilon).
    """
    surface = PlaneMirror3D(
        name="TestSurf",
        center=(0.0, 0.0, 50.0),
        normal=(0.0, 0.0, -1.0),
        width=20.0,
        height=20.0,
    )

    # Ray starting at z = 60 moving along +z (surface is behind the ray at z = 50)
    r = np.array([[0.0, 0.0, 60.0]])
    k = np.array([[0.0, 0.0, 1.0]])

    t, hit_pos, local_uv, in_bounds = surface.intersect(r, k)

    assert t[0] < 0, "Distance to surface behind the ray must be negative"
    assert not in_bounds[0], "Surface behind ray must not produce a valid forward hit"


def test_check_5_nearest_surface_intersection():
    """
    CHECK 5: Nearest-surface intersection must determine the next interaction.
    """
    # Two surfaces at z = 100 and z = 150
    surf1 = PlaneMirror3D(name="Near", center=(0.0, 0.0, 100.0), normal=(0.0, 0.0, -1.0), width=30.0, height=30.0)
    surf2 = PlaneMirror3D(name="Far", center=(0.0, 0.0, 150.0), normal=(0.0, 0.0, -1.0), width=30.0, height=30.0)

    r = np.array([[0.0, 0.0, 0.0]])
    k = np.array([[0.0, 0.0, 1.0]])

    t1, _, _, hit1 = surf1.intersect(r, k)
    t2, _, _, hit2 = surf2.intersect(r, k)

    assert hit1[0] and hit2[0], "Ray intersects both planes"
    assert t1[0] < t2[0], "surf1 is closer than surf2"
    nearest = surf1 if t1[0] < t2[0] else surf2
    assert nearest.name == "Near", "Nearest positive intersection must be selected"


def test_check_6_moving_pupil_mirror_changes_ray_path():
    """
    CHECK 6: Changing a pupil mirror x/y/z position should physically change the ray path.
    """
    slicer = SlicerArray(name="TestSlicer", z=190.0, layout="1x2")

    relay1 = PupilRelaySystem(name="Relay1")
    relay2 = PupilRelaySystem(name="Relay2")

    # Mirror at L = 50 mm vs L = 80 mm
    P1 = slicer.slices[0].compute_pupil_position(distance=50.0)
    P2 = slicer.slices[0].compute_pupil_position(distance=80.0)

    pm1 = PupilMirror(channel_id=0, name="P1", center=P1, normal=(0.0, 0.0, -1.0), width=20.0, height=20.0)
    pm2 = PupilMirror(channel_id=0, name="P2", center=P2, normal=(0.0, 0.0, -1.0), width=20.0, height=20.0)

    relay1.add_mirror(pm1)
    relay2.add_mirror(pm2)

    assert not np.allclose(P1, P2), "Pupil mirrors should be at different 3D locations"

    # Launch ray striking slice 0
    r_test = np.array([[0.0, slicer.slices[0].center_y, 180.0]])
    k_test = np.array([[0.0, 0.0, 1.0]])

    b1 = RayBundle(r=r_test.copy(), k=k_test.copy())
    b2 = RayBundle(r=r_test.copy(), k=k_test.copy())

    slicer.trace(b1)
    slicer.trace(b2)

    # Intersect with respective pupil mirrors
    t1, hit1, _, in1 = pm1.intersect(b1.r, b1.k)
    t2, hit2, _, in2 = pm2.intersect(b2.r, b2.k)

    assert in1[0] and in2[0], "Ray should strike both pupil mirrors"
    assert not np.allclose(hit1[0], hit2[0]), "Moving pupil mirror must physically change the 3D ray hit points"
    assert t2[0] > t1[0], "Pupil mirror placed further away must yield larger propagation distance"


def test_check_7_rotating_final_lens_rotates_optical_axis():
    """
    CHECK 7: Rotating the final coupling lens must rotate its optical axis and ray transformation.
    """
    # Lens 1: normal along +z
    lens_flat = ThinLens3D(
        name="FlatLens",
        center=(0.0, 0.0, 200.0),
        normal=(0.0, 0.0, 1.0),
        focal_length=100.0,
        diameter=50.0,
    )

    # Lens 2: tilted by 15 degrees about x-axis (normal tilted in y-z plane)
    theta_rad = np.radians(15.0)
    normal_tilted = np.array([0.0, -np.sin(theta_rad), np.cos(theta_rad)])
    lens_tilted = ThinLens3D(
        name="TiltedLens",
        center=(0.0, 0.0, 200.0),
        normal=normal_tilted,
        focal_length=100.0,
        diameter=50.0,
    )

    # Incident off-axis ray: x = 0, y = 5.0
    r_in = np.array([[0.0, 5.0, 100.0]])
    k_in = np.array([[0.0, 0.0, 1.0]])

    b_flat = RayBundle(r=r_in.copy(), k=k_in.copy())
    b_tilted = RayBundle(r=r_in.copy(), k=k_in.copy())

    t_f, hit_f, uv_f, in_f = lens_flat.intersect(b_flat.r, b_flat.k)
    lens_flat.interact(b_flat, np.array([0]), hit_f, uv_f)

    t_t, hit_t, uv_t, in_t = lens_tilted.intersect(b_tilted.r, b_tilted.k)
    lens_tilted.interact(b_tilted, np.array([0]), hit_t, uv_t)

    # Refracted ray directions should be significantly different
    assert not np.allclose(b_flat.k[0], b_tilted.k[0], atol=1e-3), (
        "Rotating the final coupling lens must change the refracted ray direction vector"
    )


def test_one_sided_pupil_placement_all_sides():
    """
    CHECK 8: All pupil mirrors must lie strictly on the chosen side of the optical axis.
    """
    from optics.pupil import generate_one_sided_pupil_positions

    # Test "lower": all y < 0
    p_lower = generate_one_sided_pupil_positions(side="lower", n_channels=2, transverse_offset=25.0)
    assert len(p_lower) == 2
    for p in p_lower:
        assert p[1] <= -20.0, f"Expected lower side y <= -20, got {p[1]}"

    # Test "upper": all y > 0
    p_upper = generate_one_sided_pupil_positions(side="upper", n_channels=4, transverse_offset=25.0)
    assert len(p_upper) == 4
    for p in p_upper:
        assert p[1] >= 15.0, f"Expected upper side y >= 15, got {p[1]}"

    # Test "left": all x < 0
    p_left = generate_one_sided_pupil_positions(side="left", n_channels=2, transverse_offset=25.0)
    for p in p_left:
        assert p[0] <= -20.0, f"Expected left side x <= -20, got {p[0]}"

    # Test "right": all x > 0
    p_right = generate_one_sided_pupil_positions(side="right", n_channels=2, transverse_offset=25.0)
    for p in p_right:
        assert p[0] >= 20.0, f"Expected right side x >= 20, got {p[0]}"


def test_slicer_analytical_aiming_at_pupils():
    """
    CHECK 9: Slicer mirrors actively aimed at assigned pupil mirrors must send chief rays
    directly to those pupil mirror centers with zero perpendicular error.
    """
    slicer = SlicerArray(name="Slicer", z=190.0, layout="1x2")
    pupil_centers = [
        np.array([-7.0, -25.0, 250.0]),
        np.array([+7.0, -25.0, 250.0]),
    ]

    k_in = np.array([0.0, 0.0, 1.0])
    for i, s in enumerate(slicer.slices):
        P_target = pupil_centers[i]
        s.aim_at_pupil(P_target, k_in=k_in)
        k_ref = s.get_reflected_chief_ray(k_in)

        # Vector from S to P
        SP = P_target - s.center
        proj = np.dot(SP, k_ref) * k_ref
        perp_error = np.linalg.norm(SP - proj)

        assert np.isclose(perp_error, 0.0, atol=1e-12), (
            f"Slice {i} chief ray perpendicular error to pupil center is {perp_error}, expected 0"
        )


def test_2channel_one_sided_end_to_end_system():
    """
    CHECK 10: 2-Channel one-sided asymmetric setup end-to-end trace:
    Slice 1 -> Pupil 1 -> New Relay Axis -> Condenser -> Fiber
    Slice 2 -> Pupil 2 -> New Relay Axis -> Condenser -> Fiber
    """
    system = create_branched_slicer_system(
        n_channels=2,
        pupil_layout_side="lower",
        pupil_transverse_offset=25.0,
        condenser_focal_length=75.0,
        pupil_aim_mode="parallel",
    )

    assert system.pupil_relay is not None
    assert len(system.pupil_relay.mirrors) == 2

    # Verify all pupil mirrors are strictly on the lower side (y < 0)
    for m in system.pupil_relay.mirrors:
        assert m.center[1] < 0, f"Pupil mirror {m.name} must be on lower side, got y={m.center[1]}"

    # Verify slicer chief rays are directed into lower half-space
    for s in system.slicer.slices:
        k_ref = s.get_reflected_chief_ray()
        assert k_ref[1] < 0, f"Slice {s.slice_id} chief ray must point downwards (k_y < 0), got {k_ref[1]}"

    # 1. Test chief rays entering each slicer mirror
    s0 = system.slicer.slices[0]
    s1 = system.slicer.slices[1]
    r_in = np.array([s0.center, s1.center])
    k_in = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    bundle = RayBundle(r=r_in, k=k_in)

    system.slicer.trace(bundle)
    coupling_res = system._trace_non_sequential_post_slicer(bundle, tot_power=1.0)

    # Verify both chief rays reach coupling optic and fiber plane
    assert coupling_res.power_reaching_coupling_optic > 0, "Chief rays must reach common condenser lens"
    assert coupling_res.n_rays_reaching_plane == 2, "Both channels must reach the fiber plane"
    # Verify exact focusing onto fiber core and NA acceptance
    assert coupling_res.n_accepted == 2, f"Both chief rays must couple into fiber, got {coupling_res.n_accepted}"
    assert np.all(coupling_res.r_coords <= 0.5), f"Radial position exceeds 0.5mm: {coupling_res.r_coords}"
    assert np.all(coupling_res.theta_angles_deg <= 12.71), f"Angle exceeds NA=0.22 (12.71 deg): {coupling_res.theta_angles_deg}"


def test_4channel_one_sided_end_to_end_system():
    """
    CHECK 11: 4-Channel one-sided asymmetric setup:
    All 4 slices aimed at 2x2 pupil cluster on lower side, redirected to condenser and fiber.
    """
    system = create_branched_slicer_system(
        n_channels=4,
        pupil_layout_side="lower",
        pupil_transverse_offset=25.0,
        condenser_focal_length=75.0,
        pupil_aim_mode="parallel",
    )

    assert len(system.pupil_relay.mirrors) == 4
    # All pupil mirrors lie strictly in lower half-space
    for m in system.pupil_relay.mirrors:
        assert m.center[1] < 0, f"Pupil mirror {m.name} must be on lower side, got y={m.center[1]}"

    centers = [s.center for s in system.slicer.slices]
    bundle = RayBundle(r=np.array(centers), k=np.repeat([[0.0, 0.0, 1.0]], 4, axis=0))

    system.slicer.trace(bundle)
    coupling_res = system._trace_non_sequential_post_slicer(bundle, tot_power=1.0)

    assert coupling_res.n_rays_reaching_plane == 4, "All 4 channels must reach the fiber plane"
    assert coupling_res.n_accepted == 4, f"All 4 channels must couple into fiber, got {coupling_res.n_accepted}"
    assert np.all(coupling_res.r_coords <= 0.5), f"Radial position exceeds 0.5mm: {coupling_res.r_coords}"
    assert np.all(coupling_res.theta_angles_deg <= 12.71), f"Angle exceeds NA: {coupling_res.theta_angles_deg}"
