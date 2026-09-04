"""
Unit tests for SlicerPupilOptimizer and design optimization routines.
"""

import numpy as np
import pytest
from optics.design_optimizer import (
    OptimizationConfig,
    SlicerPupilOptimizer,
    SingleNOptimizationResult,
    MultiNStudyResult,
)
from optics.geometry3d import normalize


def test_optimizer_initial_vector_and_bounds():
    """Verify initial vector shape and bounds structure for N channels."""
    config = OptimizationConfig(pupil_layout_side="lower", z_slicer=190.0)
    optimizer = SlicerPupilOptimizer(config)

    for n in [1, 2, 4]:
        x0, bounds = optimizer.get_initial_vector_and_bounds(n, spot_radius=2.5)
        # Vector has length 5 * N: 2N slicer (x, y) + 3N pupil (x, y, z)
        assert len(x0) == 5 * n
        assert len(bounds) == 5 * n

        # Check bounds enclose x0
        for i, (val, (b_low, b_high)) in enumerate(zip(x0, bounds)):
            assert b_low <= val <= b_high, f"X0[{i}] = {val} outside [{b_low}, {b_high}]"

        # Check one-sided constraint on pupil y (strictly negative for 'lower')
        slicer_pos, pupil_pos = optimizer.unpack_vector(x0, n)
        assert len(slicer_pos) == n
        assert len(pupil_pos) == n
        for p in pupil_pos:
            assert p[1] < 0.0, f"Pupil y={p[1]} must be negative for 'lower' side"
            assert p[2] > config.z_slicer, "Pupil z must be downstream of slicer"


def test_optimizer_build_candidate_system_derived_orientations():
    """Verify that build_candidate_system derives mirror tilts analytically."""
    config = OptimizationConfig(
        pupil_layout_side="lower",
        pupil_aim_mode="parallel",
        z_slicer=190.0,
        condenser_focal_length=75.0,
    )
    optimizer = SlicerPupilOptimizer(config)

    n = 2
    x0, _ = optimizer.get_initial_vector_and_bounds(n, spot_radius=2.5)
    w, h = optimizer.get_slice_dimensions(n, spot_radius=2.5)

    sys = optimizer.build_candidate_system(x0, n, w, h)
    assert sys.slicer is not None
    assert len(sys.slicer.slices) == n
    assert sys.pupil_relay is not None
    assert len(sys.pupil_relay.mirrors) == n

    # Check slicers aim at their pupils
    for i, s in enumerate(sys.slicer.slices):
        pm = sys.pupil_relay.mirrors[i]
        k_in = np.array([0.0, 0.0, 1.0])
        k_ref = s.get_reflected_chief_ray(k_in)
        target_dir = normalize(pm.center - s.center)
        # Dot product should be 1.0
        assert np.isclose(np.dot(k_ref, target_dir), 1.0, atol=1e-4)


def test_optimizer_merit_function_penalties():
    """Verify penalty calculation for overlapping slicers, overlapping pupils, and wrong side."""
    config = OptimizationConfig(pupil_layout_side="lower", exploration_rays=200)
    optimizer = SlicerPupilOptimizer(config)

    n = 2
    x0, _ = optimizer.get_initial_vector_and_bounds(n, spot_radius=2.5)
    w, h = optimizer.get_slice_dimensions(n, spot_radius=2.5)

    pre_bundle = optimizer.get_pre_slicer_bundle(200)
    source_p = float(optimizer._cached_source_bundle.total_power)

    # Valid X0 merit
    merit_clean = optimizer.evaluate_merit_function(x0, n, w, h, pre_bundle, source_p)

    # Case A: Overlapping Slicers (put both slicers at exactly the same (x, y))
    x_overlap = x0.copy()
    x_overlap[2] = x_overlap[0]
    x_overlap[3] = x_overlap[1]
    merit_slicer_overlap = optimizer.evaluate_merit_function(x_overlap, n, w, h, pre_bundle, source_p)
    # Minimizing objective: penalty makes value more positive (worse)
    assert merit_slicer_overlap > merit_clean

    # Case B: Overlapping Pupils (put both pupils at exactly the same 3D coordinate)
    x_pupil_overlap = x0.copy()
    x_pupil_overlap[7:10] = x_pupil_overlap[4:7]
    merit_pupil_overlap = optimizer.evaluate_merit_function(x_pupil_overlap, n, w, h, pre_bundle, source_p)
    assert merit_pupil_overlap > merit_clean

    # Case C: Violation of one-sided region (put pupil on positive y when side is 'lower')
    x_wrong_side = x0.copy()
    x_wrong_side[5] = +20.0  # y_p1 positive
    merit_wrong_side = optimizer.evaluate_merit_function(x_wrong_side, n, w, h, pre_bundle, source_p)
    assert merit_wrong_side > merit_clean


def test_optimizer_single_n_execution():
    """Test full single N optimization run and verify physical coupling metrics."""
    config = OptimizationConfig(
        n_min=1,
        n_max=1,
        max_de_iter=3,          # Fast test run
        popsize=4,
        polish=False,
        exploration_rays=150,
        validation_rays=300,
    )
    optimizer = SlicerPupilOptimizer(config)

    res = optimizer.optimize_fixed_n(1)
    assert isinstance(res, SingleNOptimizationResult)
    assert res.n_channels == 1
    assert 0.0 <= res.coupling_efficiency <= 1.0 + 1e-6
    assert 0.0 <= res.core_accepted_fraction <= 1.0 + 1e-6
    assert 0.0 <= res.na_accepted_fraction <= 1.0 + 1e-6
    assert len(res.slicer_table) == 1
    assert len(res.pupil_table) == 1
    assert "Final Fiber Accepted" in res.loss_budget


def test_optimizer_multi_n_study_comparison():
    """Test discrete comparison across N=1 and N=2 channels."""
    config = OptimizationConfig(
        n_min=1,
        n_max=2,
        max_de_iter=2,          # Fast test run
        popsize=3,
        polish=False,
        exploration_rays=100,
        validation_rays=200,
    )
    optimizer = SlicerPupilOptimizer(config)

    study = optimizer.run_multi_n_study(n_min=1, n_max=2)
    assert isinstance(study, MultiNStudyResult)
    assert study.best_n in [1, 2]
    assert len(study.comparison_table) == 2
    assert study.best_result is not None
    assert study.best_result.coupling_efficiency >= 0.0
    assert len(study.winner_explanation) > 0


def test_sun_broad_image_high_coupling_optimization():
    """Verify physical power accounting and coupling metrics for Sun source."""
    config = OptimizationConfig(
        n_min=1,
        n_max=1,
        source_mode="SUN",
        fore_focal_length=150.0,
        z_slicer=190.0,
        max_de_iter=5,
        popsize=4,
        polish=True,
        exploration_rays=200,
        validation_rays=500,
    )
    optimizer = SlicerPupilOptimizer(config)
    res = optimizer.optimize_fixed_n(1)
    assert res.coupling_efficiency >= 0.25, f"Expected coupling >= 25%, got {res.coupling_efficiency*100:.2f}%"
    assert res.power_accounting is not None
    assert res.power_accounting.p_launch > 0.0
    assert res.power_accounting.p_inside_core_and_na > 0.0
