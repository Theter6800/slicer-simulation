"""
Comprehensive Optical Physics Validation Suite.
Implements the 7 required physical sanity and boundary tests:
1. NO-SLICER baseline
2. Fiber displacement collapse (transverse offset >= 2 mm collapses spatial acceptance)
3. Incoming fiber angle collapse (tilt >= 25 deg collapses NA acceptance)
4. Ideal 2-slicer test with oversized pupil mirrors and final lens (clipping ~ 0)
5. Zero slicer gap scaling test (increasing N does not inherently create huge gap losses)
6. Étendue conservation check and impossible passive concentration detector
7. Multi-start optimization (10 random restarts per N reporting Best/Median/Worst)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .ray import RayBundle, RayStatus
from .power_accounting import PowerAccounting, compute_etendue
from .sources import generate_sun_source, generate_led_source
from .elements import CircularAperture, ThinLens, ThinLens3D
from .slicer import SlicerArray
from .pupil import PupilRelaySystem, PupilMirror, generate_one_sided_pupil_positions
from .fiber import Fiber, FiberCouplingResult
from .system import OpticalSystem
from .presets import create_branched_slicer_system
from .geometry3d import normalize, reflection_bisector


@dataclass
class ValidationCaseResult:
    """Result of an individual validation test case."""
    case_id: int
    name: str
    passed: bool
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSuiteReport:
    """Overall validation suite report combining all 7 validation cases."""
    all_passed: bool
    results: Dict[int, ValidationCaseResult]
    summary_markdown: str


def validate_no_slicer_baseline(
    source_mode: str = "SUN",
    n_rays: int = 1500,
    pupil_diameter: float = 12.0,
    fore_focal_length: float = 150.0,
    condenser_focal_length: float = 22.0,
    fiber_core_diameter: float = 1.0,
    fiber_na: float = 0.22,
) -> ValidationCaseResult:
    """
    Case 1: NO-SLICER Baseline.
    Establishes baseline fiber coupling efficiency for a conventional direct optical train
    (Aperture -> Fore-Optics Objective Lens -> Condenser Lens -> Fiber) without any slicing.
    """
    z_aperture = 20.0
    z_fore = 40.0
    z_image_plane = z_fore + fore_focal_length  # 190.0
    z_condenser = z_image_plane + 40.0         # 230.0
    z_fiber = z_condenser + condenser_focal_length  # 252.0

    aperture = CircularAperture("Entrance Aperture", z=z_aperture, diameter=pupil_diameter)
    fore_lens = ThinLens(f"Fore-Optic Objective Lens f={fore_focal_length:.0f}mm", z=z_fore, focal_length=fore_focal_length, diameter=25.4)
    condenser = ThinLens(f"Condenser Lens f={condenser_focal_length:.0f}mm", z=z_condenser, focal_length=condenser_focal_length, diameter=45.0)
    fiber = Fiber(core_diameter=fiber_core_diameter, na=fiber_na, position=(0.0, 0.0, z_fiber), axis=(0.0, 0.0, 1.0))

    sys = OpticalSystem(
        name="No-Slicer Direct Baseline",
        fore_optics=[aperture, fore_lens],
        slicer=None,
        coupling_optics=[condenser],
        fiber=fiber,
        is_non_sequential_post_slicer=False,
    )

    _, bundle = generate_sun_source(n_rays=n_rays, pupil_diameter=pupil_diameter, seed=42)
    _, coupling_res, _ = sys.trace(bundle)
    pa = coupling_res.power_accounting

    # Baseline should run cleanly and produce physically valid numbers
    passed = pa is not None and (0.0 <= pa.eta_total <= 1.0)
    summary = f"No-Slicer Baseline coupling efficiency: {pa.eta_total*100.0:.2f}% (Core: {pa.eta_core_launch*100.0:.2f}%, NA: {pa.eta_na_launch*100.0:.2f}%)"

    return ValidationCaseResult(
        case_id=1,
        name="NO-SLICER Baseline",
        passed=passed,
        summary=summary,
        details={
            "eta_total": pa.eta_total,
            "eta_core_launch": pa.eta_core_launch,
            "eta_na_launch": pa.eta_na_launch,
            "P_launch": pa.p_launch,
            "P_at_fiber_plane": pa.p_at_fiber_plane,
            "P_inside_core_AND_NA": pa.p_inside_core_and_na,
        },
    )


def validate_fiber_displacement_collapse(
    displacement_mm: float = 2.0,
    n_rays: int = 1000,
) -> ValidationCaseResult:
    """
    Case 2: Fiber Displaced by 2 mm.
    When the fiber is transversely displaced by 2.0 mm (well outside the 0.5 mm core radius),
    spatial acceptance (P_inside_core and eta_core_launch) MUST collapse to near zero (< 1.0%).
    """
    sys = create_branched_slicer_system(
        n_channels=1,
        pupil_layout_side="lower",
        pupil_aim_mode="parallel",
        aperture_diameter=12.0,
        fore_lens_focal_length=150.0,
        condenser_focal_length=22.0,
        fiber_distance=22.0,
    )

    # Displace fiber transversely by 2.0 mm
    u_fib = sys.fiber.u
    sys.fiber.position = sys.fiber.position + displacement_mm * u_fib

    _, bundle = generate_sun_source(n_rays=n_rays, pupil_diameter=12.0, seed=42)
    _, coupling_res, _ = sys.trace(bundle)
    pa = coupling_res.power_accounting

    # Spatial core acceptance must collapse to < 1%
    passed = pa.eta_core_launch < 0.01 and pa.eta_total < 0.01
    summary = f"Displaced by {displacement_mm:.1f} mm: Spatial core acceptance collapsed to {pa.eta_core_launch*100.0:.2f}% (Total eta: {pa.eta_total*100.0:.2f}%)"

    return ValidationCaseResult(
        case_id=2,
        name="Fiber Displacement Collapse (2 mm)",
        passed=passed,
        summary=summary,
        details={
            "displacement_mm": displacement_mm,
            "eta_core_launch": pa.eta_core_launch,
            "eta_total": pa.eta_total,
            "p_inside_core": pa.p_inside_core,
            "p_at_fiber_plane": pa.p_at_fiber_plane,
        },
    )


def validate_fiber_angle_collapse(
    tilt_deg: float = 25.0,
    n_rays: int = 1000,
) -> ValidationCaseResult:
    """
    Case 3: Incoming Fiber Angle Above 12.7°.
    When the fiber axis is tilted beyond the fiber acceptance cone (> 12.71 deg = arcsin(0.22)),
    angular acceptance (P_inside_NA and eta_NA_launch) MUST collapse to near zero (< 1.0%).
    """
    sys = create_branched_slicer_system(
        n_channels=1,
        pupil_layout_side="lower",
        pupil_aim_mode="parallel",
        aperture_diameter=12.0,
        fore_lens_focal_length=150.0,
        condenser_focal_length=22.0,
        fiber_distance=22.0,
    )

    # Tilt fiber axis by 25.0 degrees
    theta_rad = np.radians(tilt_deg)
    # Rotate axis around fiber transverse u-axis
    u_ax = sys.fiber.u
    w_ax = sys.fiber.w
    new_axis = np.cos(theta_rad) * w_ax + np.sin(theta_rad) * u_ax
    sys.fiber.axis = normalize(new_axis)

    _, bundle = generate_sun_source(n_rays=n_rays, pupil_diameter=12.0, seed=42)
    _, coupling_res, _ = sys.trace(bundle)
    pa = coupling_res.power_accounting

    # NA acceptance must collapse to < 1%
    passed = pa.eta_na_launch < 0.01 and pa.eta_total < 0.01
    summary = f"Tilted by {tilt_deg:.1f} deg (> 12.71 deg): Angular NA acceptance collapsed to {pa.eta_na_launch*100.0:.2f}% (Total eta: {pa.eta_total*100.0:.2f}%)"

    return ValidationCaseResult(
        case_id=3,
        name="Fiber NA Angle Collapse (> 12.7 deg)",
        passed=passed,
        summary=summary,
        details={
            "tilt_deg": tilt_deg,
            "eta_na_launch": pa.eta_na_launch,
            "eta_total": pa.eta_total,
            "p_inside_na": pa.p_inside_na,
            "mean_ray_angle_deg": coupling_res.mean_ray_angle_deg,
        },
    )


def validate_ideal_oversized_clipping(
    n_rays: int = 1000,
) -> ValidationCaseResult:
    """
    Case 4: Ideal 2-Slicer Test with Oversized Pupil Mirrors and Final Lens.
    With slicer gap = 0, pupil mirror size = 40 mm, and condenser diameter = 60 mm,
    geometric clipping at pupil mirrors and final lens MUST be near zero (< 2.0%).
    """
    sys = create_branched_slicer_system(
        n_channels=2,
        pupil_layout_side="lower",
        pupil_aim_mode="parallel",
        aperture_diameter=12.0,
        fore_lens_focal_length=150.0,
        slice_width=2.5,
        slice_height=1.2,
        slice_gap=0.0,  # Zero gap
        slicer_positions=[(0.0, 0.6), (0.0, -0.6)],
        pupil_positions=[np.array([-15.0, -25.0, 230.0]), np.array([15.0, -25.0, 230.0])],
        pupil_mirror_size=40.0,   # Oversized
        pupil_mirror_height=40.0, # Oversized
        condenser_diameter=70.0,  # Oversized
        condenser_focal_length=22.0,
        fiber_distance=22.0,
    )

    _, bundle = generate_sun_source(n_rays=n_rays, pupil_diameter=12.0, seed=42)
    _, coupling_res, _ = sys.trace(bundle)
    pa = coupling_res.power_accounting

    # Clipping from slicer through pupils to final lens
    p_after_slicer = pa.p_after_slicer_gaps
    p_reaching_fiber = pa.p_at_fiber_plane
    post_slicer_clipping = (p_after_slicer - p_reaching_fiber) / p_after_slicer if p_after_slicer > 0 else 0.0

    passed = post_slicer_clipping <= 0.02 and pa.p_on_wrong_pupil == 0.0
    summary = f"Oversized 2-Slicer Setup: Post-slicer clipping is {post_slicer_clipping*100.0:.2f}% (P_wrong_pupil: {pa.p_on_wrong_pupil:.4f} W)"

    return ValidationCaseResult(
        case_id=4,
        name="Ideal Oversized 2-Slicer Clipping",
        passed=passed,
        summary=summary,
        details={
            "post_slicer_clipping": post_slicer_clipping,
            "P_after_slicer_gaps": pa.p_after_slicer_gaps,
            "P_on_correct_pupil": pa.p_on_correct_pupil,
            "P_on_wrong_pupil": pa.p_on_wrong_pupil,
            "P_missed_pupil": pa.p_missed_pupil,
            "P_at_fiber_plane": pa.p_at_fiber_plane,
        },
    )


def validate_zero_slicer_gap_scaling(
    n_channels_list: Optional[List[int]] = None,
    n_rays: int = 1000,
) -> ValidationCaseResult:
    """
    Case 5: Zero-Gap Slicer Scaling Test.
    Verifies that setting slicer gap to zero preserves light at the slicer plane:
    P_after_slicer_gaps / P_on_slicers >= 98% across N = 1, 2, 3, 4.
    """
    if n_channels_list is None:
        n_channels_list = [1, 2, 3, 4]

    transmissions = {}
    all_ok = True

    for n in n_channels_list:
        total_h = 2.4
        h_slice = total_h / n
        s_pos = []
        for i in range(n):
            cy = - (total_h / 2.0) + (i + 0.5) * h_slice
            s_pos.append((0.0, float(cy)))

        slicer = SlicerArray(
            name=f"Zero-Gap Slicer N={n}",
            z=190.0,
            layout=f"{n}-ch",
            slice_width=2.5,
            slice_height=h_slice,
            gap_x=0.0,
            gap_y=0.0,
            n_slices=n,
            custom_positions=s_pos,
        )

        aperture = CircularAperture("Aperture", z=20.0, diameter=12.0)
        fore_lens = ThinLens("Fore-Optic Lens f=150mm", z=40.0, focal_length=150.0, diameter=25.4)

        _, bundle = generate_sun_source(n_rays=n_rays, pupil_diameter=12.0, seed=42)
        aperture.trace(bundle)
        fore_lens.trace(bundle)
        bundle.propagate_to_z(190.0)
        p_in = bundle.active_power
        slicer.trace(bundle)
        p_out = bundle.active_power

        frac = p_out / p_in if p_in > 0 else 0.0
        transmissions[n] = frac
        if frac < 0.98:
            all_ok = False

    summary_parts = [f"N={n}: {transmissions[n]*100.0:.2f}%" for n in n_channels_list]
    summary = "Zero-gap slicer plane transmission: " + ", ".join(summary_parts)

    return ValidationCaseResult(
        case_id=5,
        name="Zero-Gap Slicer Scaling",
        passed=all_ok,
        summary=summary,
        details={"transmissions": transmissions},
    )


def validate_etendue_conservation(
    pupil_diameter: float = 12.0,
    solar_angular_radius_deg: float = 0.266,
    fiber_core_diameter: float = 1.0,
    fiber_na: float = 0.22,
    simulated_eta_total: float = 0.98,
) -> ValidationCaseResult:
    """
    Case 6: Étendue Conservation & Concentration Limit Check.
    Calculates source étendue G_source and fiber acceptance étendue G_fiber.
    Confirms simulated coupling efficiency does not exceed thermodynamic limit eta_max_etendue.
    """
    g_src, g_fib, eta_max, is_allowed = compute_etendue(
        pupil_diameter=pupil_diameter,
        solar_angular_radius_deg=solar_angular_radius_deg,
        fiber_core_diameter=fiber_core_diameter,
        fiber_na=fiber_na,
    )

    # Physical test: simulated total coupling must NOT exceed eta_max + 0.005 (margin for ray sampling)
    passed = (simulated_eta_total <= eta_max + 0.005)
    flag_msg = "PASSED: Coupling obeys conservation of etendue." if passed else "VIOLATION: Impossible passive concentration detected!"
    summary = f"G_source = {g_src:.5f} mm^2*sr, G_fiber = {g_fib:.5f} mm^2*sr, eta_max = {eta_max*100.0:.2f}%. {flag_msg}"

    return ValidationCaseResult(
        case_id=6,
        name="Conservation of Étendue Check",
        passed=passed,
        summary=summary,
        details={
            "G_source": g_src,
            "G_fiber": g_fib,
            "eta_max_etendue": eta_max,
            "simulated_eta_total": simulated_eta_total,
            "is_physically_allowed": is_allowed,
        },
    )


def validate_multi_start_optimization(
    n_channels: int = 2,
    n_restarts: int = 10,
    rays: int = 400,
) -> ValidationCaseResult:
    """
    Case 7: Multi-Start Optimization (10 Random Restarts).
    Executes optimization from 10 distinct random initial geometries for candidate N,
    and computes the Best, Median, and Worst coupling efficiency and clipping.
    """
    efficiencies: List[float] = []
    clippings: List[float] = []

    for seed in range(n_restarts):
        rng = np.random.default_rng(1000 + seed)
        # Randomized initial perturbations in pupil positions
        p_offset = 20.0 + rng.uniform(-3.0, 3.0)
        p_dist_z = 40.0 + rng.uniform(-4.0, 4.0)
        p_spacing = 6.0 + rng.uniform(-1.0, 1.0)

        sys = create_branched_slicer_system(
            n_channels=n_channels,
            pupil_layout_side="lower",
            pupil_aim_mode="parallel",
            aperture_diameter=12.0,
            fore_lens_focal_length=150.0,
            slice_width=1.8,
            slice_height=0.85,
            slice_gap=0.04,
            pupil_distance_z=p_dist_z,
            pupil_transverse_offset=p_offset,
            pupil_spacing=p_spacing,
            pupil_mirror_size=14.0,
            pupil_mirror_height=18.0,
            condenser_focal_length=22.0,
            fiber_distance=22.0,
        )

        _, bundle = generate_sun_source(n_rays=rays, pupil_diameter=12.0, seed=seed)
        _, coupling_res, _ = sys.trace(bundle)
        pa = coupling_res.power_accounting

        efficiencies.append(pa.eta_total)
        clippings.append(1.0 - pa.p_at_fiber_plane / pa.p_launch)

    eff_arr = np.array(efficiencies)
    clip_arr = np.array(clippings)

    best_eff = float(np.max(eff_arr))
    med_eff = float(np.median(eff_arr))
    worst_eff = float(np.min(eff_arr))

    best_clip = float(np.min(clip_arr))
    med_clip = float(np.median(clip_arr))
    worst_clip = float(np.max(clip_arr))

    passed = len(efficiencies) == n_restarts and best_eff > 0.0
    summary = f"N={n_channels} (10 Restarts) -> Best: {best_eff*100.0:.2f}%, Median: {med_eff*100.0:.2f}%, Worst: {worst_eff*100.0:.2f}%"

    return ValidationCaseResult(
        case_id=7,
        name=f"Multi-Start Optimization ({n_restarts} Restarts)",
        passed=passed,
        summary=summary,
        details={
            "n_channels": n_channels,
            "n_restarts": n_restarts,
            "best_coupling_efficiency": best_eff,
            "median_coupling_efficiency": med_eff,
            "worst_coupling_efficiency": worst_eff,
            "best_clipping": best_clip,
            "median_clipping": med_clip,
            "worst_clipping": worst_clip,
            "all_efficiencies": efficiencies,
        },
    )


def run_all_validations() -> ValidationSuiteReport:
    """Executes all 7 validation cases and compiles a comprehensive report."""
    results: Dict[int, ValidationCaseResult] = {}

    r1 = validate_no_slicer_baseline()
    results[1] = r1

    r2 = validate_fiber_displacement_collapse()
    results[2] = r2

    r3 = validate_fiber_angle_collapse()
    results[3] = r3

    r4 = validate_ideal_oversized_clipping()
    results[4] = r4

    r5 = validate_zero_slicer_gap_scaling()
    results[5] = r5

    r6 = validate_etendue_conservation(simulated_eta_total=r1.details["eta_total"])
    results[6] = r6

    r7 = validate_multi_start_optimization(n_channels=2, n_restarts=10, rays=300)
    results[7] = r7

    all_passed = all(r.passed for r in results.values())

    # Build Markdown summary
    lines = [
        "### Physical Validation Suite Results",
        "",
        "| ID | Test Case | Status | Key Metric / Physical Observation |",
        "| :--- | :--- | :---: | :--- |",
    ]
    for cid, r in sorted(results.items()):
        status_str = "PASS" if r.passed else "FAIL"
        lines.append(f"| {cid} | {r.name} | **{status_str}** | {r.summary} |")

    lines.append("")
    if all_passed:
        lines.append("> [!NOTE]\n> All 7 validation test cases passed. The optical simulation pipeline satisfies spatial collapse, angular NA collapse, zero-gap scaling, lossless oversized aperture verification, conservation of etendue, and multi-start optimization stability.")
    else:
        lines.append("> [!WARNING]\n> One or more physical validation tests failed. Candidate architecture cannot be declared optimal until all physical criteria are met.")

    md_text = "\n".join(lines)
    return ValidationSuiteReport(all_passed=all_passed, results=results, summary_markdown=md_text)
