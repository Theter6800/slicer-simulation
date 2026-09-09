"""
Unit tests for the 18 physics and reporting corrections:
- Tie tolerance logic and dual recommendations
- Joint fiber phase-space acceptance (Core AND NA)
- N_effective calculation and active channel thresholding
- Balanced multi-channel mode with power penalty
- Phase-space reformatting score calculation
- Dominant limitation classification
- Estimated physical efficiency with real surface losses
- 11-point validation gate
- Alignment tolerance study
"""

import numpy as np
import pytest
from optics.sources import generate_sun_source
from optics.presets import create_branched_slicer_system
from optics.system import OpticalSystem
from optics.power_accounting import PowerAccounting, SlicePowerShare
from optics.metrics import compute_phase_space_reformatting_score, PhaseSpaceReformattingScore
from optics.design_optimizer import (
    OptimizationConfig,
    SlicerPupilOptimizer,
    compute_analytical_optical_checks,
    run_alignment_tolerance_study,
)
from optics.validation import verify_validation_gate, build_direct_baseline_system


def make_test_pa(
    p_launch=1.0,
    p_on_image_plane=1.0,
    p_intercepted_by_slicers=1.0,
    p_lost_at_slicer_gaps=0.0,
    p_on_correct_pupil=1.0,
    p_on_condenser=1.0,
    p_at_fiber_plane=0.8,
    p_inside_core=0.4,
    p_inside_na=0.4,
    p_inside_core_and_na=0.3,
    slice_shares=None,
):
    return PowerAccounting(
        p_launch=p_launch,
        p_after_aperture=p_launch,
        p_on_image_plane=p_on_image_plane,
        p_intercepted_by_slicers=p_intercepted_by_slicers,
        p_lost_at_slicer_gaps=p_lost_at_slicer_gaps,
        p_missed_slicer_array=p_on_image_plane - p_intercepted_by_slicers,
        p_on_correct_pupil=p_on_correct_pupil,
        p_on_wrong_pupil=0.0,
        p_missed_all_pupils=0.0,
        p_blocked_by_other_optics=0.0,
        p_on_condenser=p_on_condenser,
        p_missed_condenser=0.0,
        p_at_fiber_plane=p_at_fiber_plane,
        p_inside_core=p_inside_core,
        p_inside_na=p_inside_na,
        p_inside_core_and_na=p_inside_core_and_na,
        wrong_pupil_policy="reject_as_stray",
        slice_shares=slice_shares or {},
    )


def test_joint_acceptance_and_power_accounting_properties():
    """Verify eta_both_launch and eta_both_conditional properties."""
    pa = make_test_pa(
        p_launch=1.0,
        p_at_fiber_plane=0.8,
        p_inside_core=0.4,
        p_inside_na=0.4,
        p_inside_core_and_na=0.3,
    )
    assert np.isclose(pa.eta_both_launch, 0.3)
    assert np.isclose(pa.eta_both_conditional, 0.3 / 0.8)
    assert np.isclose(pa.eta_total, 0.3)
    assert np.isclose(pa.eta_coupling_conditional, 0.3 / 0.8)


def test_n_effective_and_power_fractions():
    """Verify compute_n_effective and slice_power_fractions."""
    shares = {
        0: SlicePowerShare(slice_id=0, p_incident=0.75, p_reflected=0.75),
        1: SlicePowerShare(slice_id=1, p_incident=0.05, p_reflected=0.05),
        2: SlicePowerShare(slice_id=2, p_incident=0.00, p_reflected=0.00),
    }
    pa = make_test_pa(
        p_launch=1.0,
        p_intercepted_by_slicers=0.8,
        slice_shares=shares,
    )
    assert len(pa.slice_power_fractions) == 3
    assert np.isclose(pa.slice_power_fractions[0], 0.75 / 0.8)
    assert np.isclose(pa.slice_power_fractions[1], 0.05 / 0.8)
    assert np.isclose(pa.slice_power_fractions[2], 0.0)

    # Channel 2 is 0% < 5%, so active channels should be 2
    n_eff = pa.compute_n_effective(active_threshold=0.05)
    assert n_eff[0] == 2
    assert pa.n_effective == 2


def test_estimated_physical_efficiency():
    """Verify compute_estimated_physical_efficiency with real surface reflections."""
    shares = {0: SlicePowerShare(slice_id=0, p_incident=0.8, p_reflected=0.8)}
    pa = make_test_pa(
        p_launch=1.0,
        p_inside_core_and_na=0.30,
        slice_shares=shares,
    )
    # For slicer: T_fore(0.96) * R_slicer(0.98) * R_pupil(0.98) * T_condenser(0.96)
    eff_slicer = pa.compute_estimated_physical_efficiency(r_slicer=0.98, r_pupil=0.98, t_lens=0.96, is_direct=False)
    assert np.isclose(eff_slicer, 0.30 * 0.96 * 0.98 * 0.98 * 0.96)

    # For direct: T_fore(0.96) * T_condenser(0.96) = 0.9216
    eff_direct = pa.compute_estimated_physical_efficiency(r_slicer=0.98, r_pupil=0.98, t_lens=0.96, is_direct=True)
    assert np.isclose(eff_direct, 0.30 * 0.96 * 0.96)


def test_dominant_limitation_classification():
    """Verify dominant limitation classification rules."""
    # Coverage limited: intercepted / image < 0.90
    pa1 = make_test_pa(p_on_image_plane=1.0, p_intercepted_by_slicers=0.5)
    assert pa1.classify_dominant_limitation() == "COVERAGE-LIMITED"

    # Pupil limited: pupil / intercepted < 0.85
    pa2 = make_test_pa(p_on_image_plane=1.0, p_intercepted_by_slicers=1.0, p_on_correct_pupil=0.6)
    assert pa2.classify_dominant_limitation() == "PUPIL-LIMITED"

    # Condenser limited: condenser / pupil < 0.90
    pa3 = make_test_pa(p_on_image_plane=1.0, p_intercepted_by_slicers=1.0, p_on_correct_pupil=1.0, p_on_condenser=0.6)
    assert pa3.classify_dominant_limitation() == "CONDENSER-LIMITED"

    # NA limited: eta_na < 0.70 and eta_core >= 0.85
    pa4 = make_test_pa(
        p_on_image_plane=1.0,
        p_intercepted_by_slicers=1.0,
        p_on_correct_pupil=1.0,
        p_on_condenser=1.0,
        p_at_fiber_plane=1.0,
        p_inside_core=0.90,
        p_inside_na=0.40,
        p_inside_core_and_na=0.38,
    )
    assert pa4.classify_dominant_limitation() == "NA-LIMITED"

    # Tied with baseline
    pa5 = make_test_pa()
    assert pa5.classify_dominant_limitation(tie_with_baseline=True) == "NO MATERIAL IMPROVEMENT"


def test_phase_space_reformatting_score():
    """Verify phase-space reformatting score calculation and diagnosis."""
    from optics.metrics import SpotMetrics, AngularMetrics

    spot_0 = SpotMetrics(
        z=0.0, centroid_x=0.0, centroid_y=0.0, rms_radius=0.4,
        encircled_50_radius=0.2, encircled_80_radius=0.35, encircled_90_radius=0.45, encircled_95_radius=0.5,
        diameter_50=0.4, diameter_80=0.7, diameter_90=0.9, diameter_95=1.0,
        bbox_width=0.9, bbox_height=0.9, peak_x=0.0, peak_y=0.0, total_power=1.0, n_active_rays=100
    )
    ang_0 = AngularMetrics(
        theta_rms_deg=8.0, theta_50_deg=6.0, theta_80_deg=10.0, theta_90_deg=11.0,
        theta_max_deg=12.0, mean_theta_deg=7.5, na_max_numerical=0.20
    )

    # Case: Spatial compressed (R90 smaller), angular expanded (theta90 larger)
    spot_n = SpotMetrics(
        z=0.0, centroid_x=0.0, centroid_y=0.0, rms_radius=0.3,
        encircled_50_radius=0.15, encircled_80_radius=0.25, encircled_90_radius=0.30, encircled_95_radius=0.4,
        diameter_50=0.3, diameter_80=0.5, diameter_90=0.6, diameter_95=0.8,
        bbox_width=0.6, bbox_height=0.6, peak_x=0.0, peak_y=0.0, total_power=1.0, n_active_rays=100
    )
    ang_n = AngularMetrics(
        theta_rms_deg=11.0, theta_50_deg=9.0, theta_80_deg=13.0, theta_90_deg=15.0,
        theta_max_deg=18.0, mean_theta_deg=10.5, na_max_numerical=0.25
    )

    score = compute_phase_space_reformatting_score(
        spot_n=spot_n,
        spot_0=spot_0,
        ang_n=ang_n,
        ang_0=ang_0,
        eta_both_cond_n=0.30,
        eta_both_cond_0=0.32,
        n_channels=2,
    )
    assert score.spatial_compressed is True
    assert score.angular_expanded is True
    assert score.joint_improved is False
    assert "Spatial compression obtained at cost of angular expansion." in score.interpretation


def test_tie_tolerance_in_optimizer():
    """Verify tie tolerance logic when comparing N=0 against N=1..2."""
    cfg = OptimizationConfig(
        n_min=0,
        n_max=2,
        source_mode="SUN",
        fore_focal_length=150.0,
        condenser_focal_length=22.0,
        pupil_distance_z=40.0,
        pupil_transverse_offset=20.0,
        exploration_rays=100,
        validation_rays=300,
        max_de_iter=2,
        popsize=4,
        polish=False,
        random_seed=42,
        absolute_tie_tolerance=0.01,  # 1 percentage point for unit test
    )
    opt = SlicerPupilOptimizer(cfg)
    res = opt.run_multi_n_study(0, 2)

    assert hasattr(res, "is_tied")
    assert hasattr(res, "candidate_ties")
    assert hasattr(res, "engineering_recommendation_n")
    assert hasattr(res, "engineering_recommendation_reason")
    assert "Core AND NA conditional" in res.comparison_table.columns
    assert "N_effective" in res.comparison_table.columns
    assert "Dominant Limitation" in res.comparison_table.columns


def test_analytical_etendue_interpretation():
    """Verify that etendue checks accurately report the ratio and non-ceiling interpretation."""
    checks = compute_analytical_optical_checks(
        d_image=1.3,
        f_fore=150.0,
        f_condenser=22.0,
    )
    assert checks.etendue_ratio > 1.0  # Fiber etendue substantially exceeds input etendue
    assert "rather than by a fundamental fiber étendue ceiling" in checks.summary


def test_11_point_validation_gate():
    """Verify that verify_validation_gate checks all 11 criteria and reports PROVISIONAL DESIGN RESULT."""
    sys = create_branched_slicer_system(
        n_channels=2,
        slice_width=10.0,
        slice_height=5.0,
        slice_gap=0.04,
        condenser_focal_length=22.0,
        fiber_distance=22.0,
    )
    _, bundle = generate_sun_source(n_rays=500, pupil_diameter=12.0, seed=42)
    sys.trace(bundle)

    gate = verify_validation_gate(sys)
    assert len(gate.checklist) == 11
    assert "PROVISIONAL DESIGN RESULT" in gate.status_banner
    assert gate.certified_optimal is False


def test_alignment_tolerance_study():
    """Verify run_alignment_tolerance_study generates tolerance report with 5% loss limits."""
    sys = create_branched_slicer_system(
        n_channels=2,
        slice_width=10.0,
        slice_height=5.0,
        slice_gap=0.04,
        condenser_focal_length=22.0,
        fiber_distance=22.0,
    )
    report = run_alignment_tolerance_study(sys, n_rays=200, seed=42)
    assert report.nominal_coupling > 0.0
    assert len(report.tolerance_5pct_loss) > 0
    assert report.most_sensitive_parameter != ""
    assert len(report.tolerance_table) > 0
