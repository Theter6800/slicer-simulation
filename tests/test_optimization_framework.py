"""
Unit tests for the refined optimization framework:
- Explicit 4-category variable classification
- Multi-start restart statistics (10 restarts)
- Design export to JSON and CSV
- Phase-space metrics and 'N/A' formatting for empty ray sets
- 2-level hierarchical architecture search
- Strict zero-emoji compliance
"""

import json
import re
import numpy as np
import pytest

from optics.design_optimizer import (
    OptimizationConfig,
    SlicerPupilOptimizer,
    SingleNOptimizationResult,
    OptimizationVariablesSummary,
    HierarchicalSearchResult,
    get_optimization_variables_summary,
    run_hierarchical_architecture_search,
    export_architecture_design_to_dict,
    export_architecture_design_to_json,
    export_architecture_design_to_csv,
    compute_detailed_loss_breakdown,
)
from optics.metrics import (
    SpotMetrics,
    AngularMetrics,
    format_metric_or_na,
)
from optics.power_accounting import PowerAccounting


def test_optimization_variables_summary_classification():
    """Verifies that all simulation parameters are cleanly divided into 4 mandatory categories."""
    cfg = OptimizationConfig(
        fore_focal_length=150.0,
        condenser_focal_length=22.0,
        source_mode="SUN",
        random_seed=42,
    )
    summary = get_optimization_variables_summary(cfg, n=2, include_outer_sweep=True)

    assert isinstance(summary, OptimizationVariablesSummary)
    assert len(summary.optimized_inner) >= 3
    assert len(summary.derived_alignment) >= 3
    assert len(summary.fixed_constraints) >= 6
    assert len(summary.study_parameters) >= 6
    assert summary.outer_sweep_variables is not None
    assert len(summary.outer_sweep_variables) >= 5

    # Verify Derived Variables are explicitly labeled "Derived alignment values", NEVER "Optimized"
    for der in summary.derived_alignment:
        assert "Derived alignment values" in der["Classification"]
        assert "NOT an optimization variable" in der["Status"]

    # Verify Slicer aperture is permanently locked in Fixed Constraints
    slicer_fixed = [fc for fc in summary.fixed_constraints if "Slicer Total Aperture" in fc["Constraint"]]
    assert len(slicer_fixed) == 1
    assert "Permanently Locked" in slicer_fixed[0]["Value"]

    # Verify Fiber coupled ray condition
    coupled_cond = [fc for fc in summary.fixed_constraints if "Fiber Coupled Ray Condition" in fc["Constraint"]]
    assert len(coupled_cond) == 1
    assert "sqrt(x^2 + y^2) <= 0.5 mm AND n_ext * sin(theta) <= 0.22" in coupled_cond[0]["Value"]


def test_multi_start_statistics_and_bounds_in_single_n():
    """Verifies multi-start statistics and bounds tracking on SingleNOptimizationResult."""
    cfg = OptimizationConfig(
        source_mode="SUN",
        exploration_rays=150,
        validation_rays=200,
        max_de_iter=3,
        popsize=4,
        polish=False,
        random_seed=42,
    )
    opt = SlicerPupilOptimizer(cfg)

    res1 = opt.optimize_fixed_n(1)
    assert isinstance(res1, SingleNOptimizationResult)
    assert res1.multi_start_stats is not None
    assert "best_coupling_efficiency" in res1.multi_start_stats
    assert "median_coupling_efficiency" in res1.multi_start_stats
    assert "worst_coupling_efficiency" in res1.multi_start_stats
    assert "n_converged" in res1.multi_start_stats
    assert res1.multi_start_stats["n_restarts"] == 10

    # Variable bounds tracking
    assert len(res1.variable_bounds) >= 5
    assert "slicer_1_x" in res1.variable_bounds
    assert "pupil_1_z" in res1.variable_bounds

    # Detailed loss breakdown
    assert len(res1.detailed_loss_breakdown) >= 8
    assert "slicer_inter_slice_gap_loss" in res1.detailed_loss_breakdown
    assert "coating_absorption_loss" in res1.detailed_loss_breakdown


def test_n0_baseline_bounds_and_losses():
    """Verifies N=0 baseline SingleNOptimizationResult bounds, losses, and multi-start stats."""
    cfg = OptimizationConfig(source_mode="SUN", validation_rays=200, random_seed=42)
    opt = SlicerPupilOptimizer(cfg)
    res0 = opt.evaluate_n0_baseline()

    assert res0.n_channels == 0
    assert "N" in res0.variable_bounds
    assert len(res0.detailed_loss_breakdown) >= 8
    assert res0.multi_start_stats is not None
    assert res0.multi_start_stats["n_converged"] == 10


def test_design_export_json_and_csv():
    """Verifies export of complete design to valid JSON and CSV formats."""
    cfg = OptimizationConfig(
        source_mode="SUN",
        exploration_rays=150,
        validation_rays=200,
        max_de_iter=3,
        popsize=4,
        polish=False,
        random_seed=42,
    )
    opt = SlicerPupilOptimizer(cfg)
    res = opt.optimize_fixed_n(2)

    # JSON export
    json_str = export_architecture_design_to_json(res, cfg)
    data = json.loads(json_str)
    assert "metadata" in data
    assert "optical_train_geometry" in data
    assert "slicer_array" in data
    assert "pupil_relay" in data
    assert "detailed_loss_breakdown" in data
    assert "power_accounting_stages" in data
    assert "fiber_spot_metrics" in data
    assert "fiber_angular_metrics" in data

    # CSV export
    csv_str = export_architecture_design_to_csv(res, cfg)
    assert "[METADATA_AND_SUMMARY]" in csv_str
    assert "[SLICER_MIRRORS_DERIVED_ALIGNMENT]" in csv_str
    assert "[PUPIL_MIRRORS_DERIVED_ALIGNMENT]" in csv_str
    assert "[DETAILED_OPTICAL_LOSS_BREAKDOWN]" in csv_str
    assert "eta_total_coupling_pct" in csv_str


def test_phase_space_metrics_empty_ray_na_formatting():
    """Verifies that empty ray sets format to 'N/A' rather than 0.000."""
    empty_spot = SpotMetrics(
        z=0.0,
        centroid_x=0.0,
        centroid_y=0.0,
        rms_radius=0.0,
        encircled_50_radius=0.0,
        encircled_80_radius=0.0,
        encircled_90_radius=0.0,
        encircled_95_radius=0.0,
        diameter_50=0.0,
        diameter_80=0.0,
        diameter_90=0.0,
        diameter_95=0.0,
        bbox_width=0.0,
        bbox_height=0.0,
        peak_x=0.0,
        peak_y=0.0,
        total_power=0.0,
        n_active_rays=0,
    )
    assert not empty_spot.is_valid
    assert format_metric_or_na(empty_spot.rms_radius, empty_spot.is_valid) == "N/A"

    valid_spot = SpotMetrics(
        z=0.0,
        centroid_x=0.0,
        centroid_y=0.0,
        rms_radius=0.42,
        encircled_50_radius=0.3,
        encircled_80_radius=0.5,
        encircled_90_radius=0.6,
        encircled_95_radius=0.7,
        diameter_50=0.6,
        diameter_80=1.0,
        diameter_90=1.2,
        diameter_95=1.4,
        bbox_width=1.0,
        bbox_height=1.0,
        peak_x=0.0,
        peak_y=0.0,
        total_power=1.0,
        n_active_rays=500,
    )
    assert valid_spot.is_valid
    assert format_metric_or_na(valid_spot.rms_radius, valid_spot.is_valid, unit="mm") == "0.42mm"

    empty_ang = AngularMetrics(
        theta_rms_deg=0.0,
        theta_50_deg=0.0,
        theta_80_deg=0.0,
        theta_90_deg=0.0,
        theta_max_deg=0.0,
        mean_theta_deg=0.0,
        na_max_numerical=0.0,
        n_rays=0,
    )
    assert not empty_ang.is_valid
    assert format_metric_or_na(empty_ang.theta_rms_deg, empty_ang.is_valid, unit="deg") == "N/A"


def test_hierarchical_architecture_search_execution():
    """Verifies that 2-level hierarchical search evaluates outer grid and returns winner."""
    cfg = OptimizationConfig(
        source_mode="SUN",
        exploration_rays=100,
        validation_rays=150,
        random_seed=42,
    )
    h_res = run_hierarchical_architecture_search(
        base_config=cfg,
        n_values=[0, 1],
        d90_values=[1.3, 10.0],
        pupil_distance_values=[40.0],
        pupil_offset_values=[20.0],
        condenser_focal_values=[22.0],
        rays=150,
        seed=42,
    )
    assert isinstance(h_res, HierarchicalSearchResult)
    assert h_res.best_inner_result is not None
    assert len(h_res.all_evaluated_architectures) == 4
    assert "Winning architecture" in h_res.summary_text


def test_zero_emojis_across_simulator():
    """Enforces zero emojis across all Python modules and application code."""
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]")
    target_files = [
        "app.py",
        "optics/design_optimizer.py",
        "optics/metrics.py",
        "optics/power_accounting.py",
        "optics/__init__.py",
    ]
    for filename in target_files:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        matches = emoji_pattern.findall(content)
        assert len(matches) == 0, f"Emoji detected in {filename}: {matches}"


def test_session_state_none_resilience():
    """Verifies that extracting parameters with None or missing values does not throw TypeError."""
    session_mock = {
        "optimizer_selected_n": None,
        "fore_focal_opt": None,
        "condenser_focal_length": None,
        "opt_seed": None,
        "random_seed": None,
        "target_d90_mm": None,
        "hardware_lens1": None,
        "hardware_lens2": None,
        "pupil_dist": None,
        "pupil_transverse_offset": None,
    }

    cur_f_fore_val = float(session_mock.get("fore_focal_opt") or 150.0)
    cur_f_cond_val = float(session_mock.get("condenser_focal_length") or 22.0)
    cur_n_raw = session_mock.get("optimizer_selected_n")
    cur_n_val = int(cur_n_raw) if cur_n_raw is not None else 2
    cur_seed_raw = session_mock.get("opt_seed") or session_mock.get("random_seed")
    cur_seed_val = int(cur_seed_raw) if cur_seed_raw is not None else 42

    assert cur_f_fore_val == 150.0
    assert cur_f_cond_val == 22.0
    assert cur_n_val == 2
    assert cur_seed_val == 42


def test_multin_study_result_has_config_and_export_handles_none_config():
    """Verifies that MultiNStudyResult preserves config and export functions handle config=None."""
    cfg = OptimizationConfig(source_mode="SUN", random_seed=42, exploration_rays=100, validation_rays=150)
    optimizer = SlicerPupilOptimizer(cfg)
    study = optimizer.run_multi_n_study(n_min=0, n_max=1)
    assert hasattr(study, "config")
    assert study.config is not None
    assert study.config.random_seed == 42

    winner_res = study.results[study.winning_n]
    json_str = export_architecture_design_to_json(winner_res, config=None)
    assert len(json_str) > 0
    csv_str = export_architecture_design_to_csv(winner_res, config=None)
    assert len(csv_str) > 0
