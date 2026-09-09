"""
Tests for Strict Power Accounting, Per-Channel Consistency,
Wrong-Pupil Policy Handling, and Architecture Validation Gate.
"""

import numpy as np
import pytest

from optics.sources import generate_sun_source
from optics.elements import CircularAperture, ThinLens
from optics.fiber import Fiber
from optics.system import OpticalSystem
from optics.presets import create_branched_slicer_system
from optics.power_accounting import PowerAccounting, SlicePowerShare
from optics.validation import (
    validate_ideal_oversized_clipping,
    build_direct_baseline_system,
    verify_n2_reformatting_benefit,
    verify_validation_gate,
    run_all_validations,
)


def test_wrong_pupil_policy_reject_as_stray():
    """Verify that reject_as_stray clips cross-channel hits at pupil plane."""
    sys = create_branched_slicer_system(
        n_channels=2,
        pupil_layout_side="lower",
        pupil_aim_mode="parallel",
        aperture_diameter=12.0,
        fore_lens_focal_length=150.0,
        slice_width=1.8,
        slice_height=0.85,
        slice_gap=0.04,
        pupil_distance_z=40.0,
        pupil_transverse_offset=20.0,
        pupil_spacing=5.0,
        pupil_mirror_size=14.0,
        pupil_mirror_height=18.0,
        condenser_focal_length=22.0,
        fiber_distance=22.0,
        wrong_pupil_policy="reject_as_stray",
    )
    assert sys.wrong_pupil_policy == "reject_as_stray"

    _, bundle = generate_sun_source(n_rays=5000, pupil_diameter=12.0, seed=42)
    sys.trace(bundle)

    pa = sys.last_power_accounting
    assert pa is not None
    assert pa.wrong_pupil_policy == "reject_as_stray"
    # Condenser input power cannot exceed correct pupil power
    assert pa.p_condenser_input <= pa.p_on_correct_pupil + 1e-6
    # Strict power conservation must pass
    is_valid, issues = pa.verify_power_conservation(tol=1e-6)
    assert is_valid, f"Conservation issues: {issues}"


def test_wrong_pupil_policy_propagate_physically():
    """Verify that propagate_physically allows wrong-pupil hits to proceed to condenser."""
    sys = create_branched_slicer_system(
        n_channels=2,
        pupil_layout_side="lower",
        pupil_aim_mode="parallel",
        aperture_diameter=12.0,
        fore_lens_focal_length=150.0,
        slice_width=1.8,
        slice_height=0.85,
        slice_gap=0.04,
        pupil_distance_z=40.0,
        pupil_transverse_offset=20.0,
        pupil_spacing=5.0,
        pupil_mirror_size=14.0,
        pupil_mirror_height=18.0,
        condenser_focal_length=22.0,
        fiber_distance=22.0,
        wrong_pupil_policy="propagate_physically",
    )
    assert sys.wrong_pupil_policy == "propagate_physically"

    _, bundle = generate_sun_source(n_rays=5000, pupil_diameter=12.0, seed=42)
    sys.trace(bundle)

    pa = sys.last_power_accounting
    assert pa is not None
    assert pa.wrong_pupil_policy == "propagate_physically"
    # Pupil output power is sum of correct and wrong pupil hits
    assert abs(pa.p_pupil_output - (pa.p_on_correct_pupil + pa.p_on_wrong_pupil)) < 1e-9
    # Strict conservation check
    is_valid, issues = pa.verify_power_conservation(tol=1e-6)
    assert is_valid, f"Conservation issues: {issues}"


def test_per_channel_power_accounting_and_sum_conservation():
    """Verify per-channel quantities are preserved and sum exactly to global totals."""
    sys = create_branched_slicer_system(
        n_channels=2,
        pupil_layout_side="lower",
        pupil_aim_mode="parallel",
        aperture_diameter=12.0,
        fore_lens_focal_length=150.0,
        slice_width=1.8,
        slice_height=0.85,
        slice_gap=0.04,
        pupil_distance_z=40.0,
        pupil_transverse_offset=20.0,
        pupil_spacing=5.0,
        pupil_mirror_size=14.0,
        pupil_mirror_height=18.0,
        condenser_focal_length=22.0,
        fiber_distance=22.0,
        wrong_pupil_policy="reject_as_stray",
    )

    _, bundle = generate_sun_source(n_rays=10000, pupil_diameter=12.0, seed=42)
    sys.trace(bundle)

    pa = sys.last_power_accounting
    assert pa is not None
    assert len(pa.slice_shares) == 2

    # Fiber power per slice must NOT be zero
    for s_id, share in pa.slice_shares.items():
        assert share.p_fiber > 0.0, f"Channel {s_id} reported zero fiber power!"
        assert share.p_accepted > 0.0, f"Channel {s_id} reported zero accepted power!"
        assert share.p_core >= share.p_accepted
        assert share.p_na >= share.p_accepted

    # Sum across channels must match global quantities within 1e-6 relative tolerance
    is_valid, issues = pa.verify_per_channel_consistency(tol=1e-6)
    assert is_valid, f"Per-channel sum discrepancies: {issues}"

    sum_fib = sum(s.p_fiber for s in pa.slice_shares.values())
    sum_acc = sum(s.p_accepted for s in pa.slice_shares.values())
    sum_core = sum(s.p_core for s in pa.slice_shares.values())
    sum_na = sum(s.p_na for s in pa.slice_shares.values())

    assert abs(sum_fib - pa.p_at_fiber_plane) / pa.p_launch < 1e-6
    assert abs(sum_acc - pa.p_inside_core_and_na) / pa.p_launch < 1e-6
    assert abs(sum_core - pa.p_inside_core) / pa.p_launch < 1e-6
    assert abs(sum_na - pa.p_inside_na) / pa.p_launch < 1e-6


def test_ideal_zero_loss_multi_slicer_test_reports_8_columns():
    """Verify validate_ideal_oversized_clipping reports all 8 required columns."""
    res = validate_ideal_oversized_clipping(n_channels_list=[1, 2, 3, 4], n_rays=1000)
    assert res.passed

    required_keys = [
        "N",
        "P_slicer",
        "P_pupil",
        "P_condenser",
        "P_fiber",
        "eta_core_conditional",
        "eta_NA_conditional",
        "eta_total",
    ]

    for n in [1, 2, 3, 4]:
        d = res.details["results_by_n"][n]
        for k in required_keys:
            assert k in d, f"Missing key {k} for N={n}"
        # Oversized optics must show zero clipping and zero cross-talk
        assert d["transmission"] >= 0.98
        assert d["P_wrong_pupil"] <= 1e-4


def test_direct_baseline_n0_system():
    """Verify N=0 direct optical train baseline system."""
    sys0 = build_direct_baseline_system(
        pupil_diameter=12.0,
        fore_focal_length=150.0,
        condenser_focal_length=22.0,
        fiber_core_diameter=1.0,
        fiber_na=0.22,
    )

    _, bundle = generate_sun_source(n_rays=5000, pupil_diameter=12.0, seed=42)
    sys0.trace(bundle)

    pa = sys0.last_power_accounting
    assert pa is not None
    assert 0.0 < pa.eta_total < 1.0
    is_valid, issues = pa.verify_power_conservation(tol=1e-6)
    assert is_valid, f"Baseline conservation issues: {issues}"
    is_ch_valid, ch_issues = pa.verify_per_channel_consistency(tol=1e-6)
    assert is_ch_valid, f"Baseline channel issues: {ch_issues}"


def test_high_ray_reformatting_benefit_and_validation_gate():
    """Verify high-ray reformatting benefit report and 8-point validation gate banner."""
    sys2 = create_branched_slicer_system(
        n_channels=2,
        pupil_layout_side="lower",
        pupil_aim_mode="parallel",
        aperture_diameter=12.0,
        fore_lens_focal_length=150.0,
        slice_width=1.8,
        slice_height=0.85,
        slice_gap=0.04,
        pupil_distance_z=40.0,
        pupil_transverse_offset=20.0,
        pupil_spacing=5.0,
        pupil_mirror_size=14.0,
        pupil_mirror_height=18.0,
        condenser_focal_length=22.0,
        fiber_distance=22.0,
        wrong_pupil_policy="reject_as_stray",
    )

    _, bundle = generate_sun_source(n_rays=5000, pupil_diameter=12.0, seed=42)
    sys2.trace(bundle)

    # Run high-ray verification across seeds (fast sample: 5,000 rays x 3 seeds for unit test)
    reformat_rep = verify_n2_reformatting_benefit(
        sys2,
        n_rays=5000,
        n_seeds=3,
        seed_start=100,
    )
    assert reformat_rep.n_seeds == 3
    assert len(reformat_rep.eta_n0_per_seed) == 3
    assert len(reformat_rep.eta_n2_per_seed) == 3

    # Check validation gate (11-point criteria)
    gate = verify_validation_gate(sys2, reformatting_report=reformat_rep)
    assert len(gate.checklist) == 11

    # Key physics criteria should be True
    assert gate.checklist["1_fiber_sanity"][0] is True
    assert gate.checklist["2_power_conservation"][0] is True
    assert gate.checklist["3_per_channel_sums"][0] is True
    assert gate.checklist["5_n_effective_reported"][0] is True
    assert gate.checklist["6_same_source_rays"][0] is True
    assert gate.checklist["8_tie_significance"][0] is True
    assert gate.checklist["9_joint_acceptance"][0] is True
    assert gate.checklist["10_zero_stale_text"][0] is True
    assert gate.checklist["11_etendue_ideal_consistency"][0] is True

    # Criterion 7 will be False if N=2 does not beat N=0 baseline
    if not reformat_rep.confirms_improvement:
        assert gate.certified_optimal is False
        assert "PROVISIONAL DESIGN RESULT" in gate.status_banner
