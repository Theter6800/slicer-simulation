"""
Tests for Fore-Optics Subsystem, Architectural Optimization, Magnification Sweep,
Sensitivity Analysis, and Analytical Physical Checks.
"""

import pytest
import numpy as np
import pandas as pd

from optics.fore_optics import (
    ForeOpticsMode,
    ForeOpticsConfig,
    ForeOpticsSystem,
    build_fore_optics_system,
    compute_theoretical_focal_length_for_d90,
    optimize_hardware_fore_optics,
    AVAILABLE_HARDWARE_FOCAL_LENGTHS,
)
from optics.design_optimizer import (
    SlicerPupilOptimizer,
    OptimizationConfig,
    SingleNOptimizationResult,
    MultiNStudyResult,
    compute_analytical_optical_checks,
    compute_slicer_necessity_diagnostic,
    run_magnification_slice_sweep,
    run_high_ray_uncertainty_validation,
)
from optics.sensitivity import ToleranceSensitivityEngine
from optics.presets import create_branched_slicer_system
from optics.sources import generate_sun_source


def test_fore_optics_theoretical_mode_calibration():
    """Verify that theoretical mode creates exact focal length for target D90 and locks z_image."""
    target_d90 = 5.0  # mm
    f_eff = compute_theoretical_focal_length_for_d90(target_d90)
    assert f_eff > 100.0

    cfg = ForeOpticsConfig(
        mode=ForeOpticsMode.THEORETICAL,
        target_d90_mm=target_d90,
        z_fore_start=40.0,
    )
    sys = build_fore_optics_system(cfg, target_d90=target_d90, n_rays_sample=1000)

    # Image plane must strictly equal z_fore_start + f_eff
    assert pytest.approx(sys.z_image, abs=1e-3) == 40.0 + f_eff
    # Measured spot D90 must be close to target
    assert pytest.approx(sys.d90_image, rel=0.15) == target_d90


def test_fore_optics_real_hardware_mode_discrete_selection():
    """Verify that hardware mode uses only available laboratory lenses."""
    for f in AVAILABLE_HARDWARE_FOCAL_LENGTHS:
        cfg = ForeOpticsConfig(
            mode=ForeOpticsMode.REAL_HARDWARE,
            lens1_focal_length=f,
            z_fore_start=50.0,
        )
        sys = build_fore_optics_system(cfg, n_rays_sample=500)
        assert sys.z_image == 50.0 + f
        assert sys.effective_focal_length == f

    # Optimizer selects best hardware lens from catalog
    best_hw = optimize_hardware_fore_optics(target_d90=1.3)
    assert best_hw.config.mode == ForeOpticsMode.REAL_HARDWARE
    assert best_hw.effective_focal_length in AVAILABLE_HARDWARE_FOCAL_LENGTHS or len(best_hw.elements) > 2


def test_strict_image_plane_locking_in_presets():
    """Verify that z_slicer is strictly locked to z_fore + fore_lens_focal_length."""
    sys = create_branched_slicer_system(
        n_channels=2,
        fore_lens_focal_length=200.0,
        fore_lens_z=35.0,
    )
    # Slicer z must be strictly 35.0 + 200.0 = 235.0
    assert sys.slicer.z == 235.0
    for s in sys.slicer.slices:
        assert s.z == 235.0


def test_analytical_optical_checks_and_slicer_necessity():
    """Verify analytical checks on etendue, NA acceptance angle, and slicer necessity diagnostic."""
    checks = compute_analytical_optical_checks(
        fiber_core_diameter=1.0,
        fiber_na=0.22,
        aperture_diameter=12.0,
        source_angular_radius_deg=0.266,
    )
    assert pytest.approx(checks.theta_max_deg, abs=0.05) == 12.71
    assert checks.fiber_etendue > checks.input_etendue
    assert checks.is_theoretically_concentrable is True

    # Diagnostic for small image (D90 = 1.3 mm, slicer width = 10 mm)
    diag_small = compute_slicer_necessity_diagnostic(d90=1.3, slicer_width=10.0, slicer_height=10.0)
    assert diag_small["category"] == "NO_SLICER_NEEDED"
    assert diag_small["ratio_d90_to_width"] < 0.5

    # Diagnostic for large image (D90 = 25 mm, slicer width = 10 mm)
    diag_large = compute_slicer_necessity_diagnostic(d90=25.0, slicer_width=10.0, slicer_height=5.0)
    assert diag_large["category"] == "SLICER_STRONGLY_JUSTIFIED"
    assert diag_large["needed_slices"] >= 5


def test_n0_baseline_evaluation_and_dual_winner_reporting():
    """Verify that N=0 is treated as a real candidate and dual winners are reported."""
    cfg = OptimizationConfig(
        n_min=1,
        n_max=2,
        fore_focal_length=150.0,
        exploration_rays=150,
        validation_rays=300,
        max_de_iter=3,
        popsize=3,
        polish=False,
    )
    opt = SlicerPupilOptimizer(cfg)
    study = opt.run_multi_n_study()

    # N=0 must be in results
    assert 0 in study.results
    res0 = study.results[0]
    assert res0.n_channels == 0
    assert res0.coupling_efficiency > 0.0

    # Dual winners must be reported
    assert study.overall_winner_n in [0, 1, 2]
    assert study.best_slicer_n in [1, 2]
    assert isinstance(study.slicer_beats_baseline, bool)
    assert isinstance(study.relative_slicer_gain, float)
    assert len(study.ideal_vs_real_table) == 3
    assert "Throughput" in study.comparison_table.columns
    assert len(study.winner_explanation) > 20


def test_magnification_slice_sweep():
    """Verify 2D magnification sweep across image diameters and slice count."""
    res = run_magnification_slice_sweep(
        d90_values=[1.3, 15.0],
        n_values=[0, 1, 2],
        n_rays=150,
    )
    assert len(res.coupling_matrix) == 2
    assert "N=0" in res.coupling_matrix.columns
    assert "N=1" in res.coupling_matrix.columns
    assert "N=2" in res.coupling_matrix.columns
    assert len(res.throughput_matrix) == 2
    assert len(res.core_acc_matrix) == 2
    assert len(res.summary_text) > 10


def test_tolerance_sensitivity_engine():
    """Verify mechanical and optical perturbation analysis and degradation curves."""
    sys = create_branched_slicer_system(n_channels=2, fore_lens_focal_length=150.0)
    engine = ToleranceSensitivityEngine(sys, n_rays=300, seed=42)
    report = engine.run_analysis()

    assert report.nominal_efficiency > 0.0
    assert len(report.curves) == 6
    assert "slicer_tip" in report.curves
    assert "slicer_tilt" in report.curves
    assert "pupil_tip" in report.curves
    assert "pupil_pos" in report.curves
    assert "fiber_pos" in report.curves
    assert "lens_axial" in report.curves

    assert len(report.recommended_tolerances) == 6
    assert len(report.summary_table) == 6
    assert len(report.most_sensitive_component) > 0


def test_high_ray_uncertainty_validation():
    """Verify high ray count uncertainty validation reporting 95% confidence intervals."""
    sys_a = create_branched_slicer_system(n_channels=2, fore_lens_focal_length=150.0)
    sys_b = create_branched_slicer_system(n_channels=1, fore_lens_focal_length=150.0)

    report = run_high_ray_uncertainty_validation(
        sys_a,
        sys_b,
        n_rays=500,
        n_seeds=3,
        seed_start=42,
    )
    assert report.n_seeds == 3
    assert report.eta_a_ci95[0] <= report.eta_a_ci95[1]
    assert report.eta_b_ci95[0] <= report.eta_b_ci95[1]
    assert isinstance(report.is_statistically_significant, bool)
    assert "95% CI" in report.summary


def test_freeze_slicer_physical_dimensions_to_hardware():
    """Directive 1: Slicer physical aperture is permanently fixed to measured 10.0 mm x 10.0 mm."""
    cfg = OptimizationConfig(min_slicer_gap=0.04)
    opt = SlicerPupilOptimizer(cfg)

    # Regardless of spot radius (small 0.5 mm or giant 40.0 mm), slice width is permanently 10.0 mm
    for spot_r in [0.5, 2.5, 10.0, 40.0]:
        w1, h1 = opt.get_slice_dimensions(1, spot_radius=spot_r)
        assert w1 == 10.0
        assert h1 == 10.0

        w2, h2 = opt.get_slice_dimensions(2, spot_radius=spot_r)
        assert w2 == 10.0
        assert pytest.approx(h2, abs=1e-4) == (10.0 - 0.04) / 2.0
        # Total array height is 2*h + gap = 10.0 mm
        assert pytest.approx(2.0 * h2 + 0.04, abs=1e-4) == 10.0

        w4, h4 = opt.get_slice_dimensions(4, spot_radius=spot_r)
        assert w4 == 10.0
        assert pytest.approx(h4, abs=1e-4) == (10.0 - 3.0 * 0.04) / 4.0
        # Total array height is 4*h + 3*gap = 10.0 mm
        assert pytest.approx(4.0 * h4 + 3.0 * 0.04, abs=1e-4) == 10.0


def test_shared_geometry_object_consistency():
    """Directive 2: Single shared geometry object guarantees z_slicer, z_image, and tables cannot disagree."""
    cfg = OptimizationConfig(
        fore_focal_length=85.0,
        z_fore=45.0,
    )
    geom = cfg.to_geometry()
    # Image plane strictly equals z_fore + fore_focal_length
    assert geom.z_image == 45.0 + 85.0
    assert geom.z_slicer == geom.z_image
    assert cfg.z_slicer == geom.z_slicer

    sys = create_branched_slicer_system(
        n_channels=2,
        fore_lens_focal_length=85.0,
        fore_lens_z=45.0,
    )
    assert sys.slicer.z == 130.0
    assert sys.z_image_plane == 130.0
    assert sys.geometry is not None
    assert sys.geometry.z_slicer == 130.0
    assert sys.geometry.z_image == 130.0


def test_ideal_vs_real_physical_enforcement():
    """Directive 3: Enforce eta_ideal >= eta_real across all N candidates."""
    cfg = OptimizationConfig(
        n_min=1,
        n_max=2,
        fore_focal_length=150.0,
        exploration_rays=150,
        validation_rays=300,
        max_de_iter=2,
        popsize=3,
        polish=False,
    )
    opt = SlicerPupilOptimizer(cfg)
    study = opt.run_multi_n_study()

    for n_arch, res in study.results.items():
        assert res.eta_ideal >= res.coupling_efficiency
        assert res.implementation_penalty >= 0.0

    for _, row in study.ideal_vs_real_table.iterrows():
        eta_i = float(row["eta_ideal (%)"].replace("%", ""))
        eta_r = float(row["eta_real (%)"].replace("%", ""))
        pen = float(row["Implementation Penalty (%)"].replace("%", ""))
        assert eta_i >= eta_r - 1e-6
        assert pen >= -1e-6
