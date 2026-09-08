"""
Unit tests for the 7 physical validation cases in optics/validation.py.
"""

import pytest
import numpy as np

from optics.validation import (
    validate_no_slicer_baseline,
    validate_fiber_displacement_collapse,
    validate_fiber_angle_collapse,
    validate_ideal_oversized_clipping,
    validate_zero_slicer_gap_scaling,
    validate_etendue_conservation,
    validate_multi_start_optimization,
    run_all_validations,
)
from optics.power_accounting import compute_etendue


def test_case_1_no_slicer_baseline():
    """Case 1: Baseline optical train without slicer."""
    res = validate_no_slicer_baseline(n_rays=500)
    assert res.passed
    assert 0.0 < res.details["eta_total"] <= 1.0
    assert res.details["P_inside_core_AND_NA"] > 0.0


def test_case_2_fiber_displacement_collapse():
    """Case 2: 2 mm fiber displacement collapses spatial acceptance to < 1%."""
    res = validate_fiber_displacement_collapse(displacement_mm=2.0, n_rays=500)
    assert res.passed
    assert res.details["eta_core_launch"] < 0.01
    assert res.details["eta_total"] < 0.01


def test_case_3_fiber_angle_collapse():
    """Case 3: Fiber angle > 12.7 deg collapses angular NA acceptance to < 1%."""
    res = validate_fiber_angle_collapse(tilt_deg=25.0, n_rays=500)
    assert res.passed
    assert res.details["eta_na_launch"] < 0.01
    assert res.details["eta_total"] < 0.01


def test_case_4_ideal_oversized_clipping():
    """Case 4: Oversized pupils and final lens produce near-zero clipping (< 2%)."""
    res = validate_ideal_oversized_clipping(n_rays=500)
    assert res.passed
    assert res.details["post_slicer_clipping"] <= 0.02
    assert res.details["P_on_wrong_pupil"] == 0.0


def test_case_5_zero_slicer_gap_scaling():
    """Case 5: Zero-gap slicer preserves >= 98% power across N=1, 2, 3, 4."""
    res = validate_zero_slicer_gap_scaling(n_channels_list=[1, 2, 3, 4], n_rays=500)
    assert res.passed
    for n, frac in res.details["transmissions"].items():
        assert frac >= 0.98


def test_case_6_etendue_conservation():
    """Case 6: Étendue calculation and physical limit enforcement."""
    # When simulated coupling is within limit:
    res_pass = validate_etendue_conservation(pupil_diameter=12.0, simulated_eta_total=0.95)
    assert res_pass.passed
    assert res_pass.details["is_physically_allowed"]

    # When simulated coupling exceeds impossible concentration limit:
    # Large pupil diameter (50 mm) where G_source > G_fiber
    g_src, g_fib, eta_max, is_allowed = compute_etendue(pupil_diameter=50.0)
    assert not is_allowed
    assert eta_max < 1.0

    # Simulated 99% coupling on a system with eta_max ~ 89.8% must fail the check
    res_fail = validate_etendue_conservation(pupil_diameter=50.0, simulated_eta_total=0.99)
    assert not res_fail.passed


def test_case_7_multi_start_optimization():
    """Case 7: Multi-start optimization across 10 random initial geometries."""
    res = validate_multi_start_optimization(n_channels=2, n_restarts=10, rays=200)
    assert res.passed
    assert len(res.details["all_efficiencies"]) == 10
    assert res.details["best_coupling_efficiency"] >= res.details["median_coupling_efficiency"]
    assert res.details["median_coupling_efficiency"] >= res.details["worst_coupling_efficiency"]


def test_fiber_acceptance_tests_a_to_e():
    """Fiber sanity tests A through E."""
    from optics.validation import validate_fiber_acceptance_tests_a_to_e
    res = validate_fiber_acceptance_tests_a_to_e()
    assert res.passed
    assert all(res.details.values())


def test_run_all_validations_master_report():
    """Master validation report runs all cases and passes."""
    report = run_all_validations()
    assert report.all_passed
    assert len(report.results) >= 7
    for cid, r in report.results.items():
        assert r.passed
    assert "### Physical Validation Suite Results" in report.summary_markdown
