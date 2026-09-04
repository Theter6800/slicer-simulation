"""
Test 6: Energy accounting and optical power conservation.
"""

import numpy as np
import pytest
from optics.sources import generate_led_source, generate_sun_source
from optics.presets import create_slicer_condenser_system, create_old_lens_system
from optics.ray import RayStatus


def test_energy_conservation_across_system_stages():
    """
    TEST 6: Energy accounting:
    At every stage: input power = surviving power + lost/clipped power within numerical tolerance.
    """
    # 1. Generate realistic LED ray bundle
    launched, at_ap = generate_led_source(
        n_rays=5000,
        source_diameter=2.0,
        aperture_diameter=15.0,
        aperture_pos=(0.0, 0.0, 40.0),
        seed=42,
    )

    total_launched = launched.total_power
    assert np.isclose(total_launched, 1.0, atol=1e-10), "Launched power should be normalized to 1.0"

    # 2. Build optical system
    system = create_slicer_condenser_system(
        aperture_diameter=15.0,
        z_aperture=40.0,
        z_fore1=70.0,
        z_fore2=150.0,
        z_slicer=200.0,
        z_condenser=250.0,
        z_fiber=320.0,
    )

    # 3. Trace
    final_bundle, coupling_res, metrics = system.trace(launched)

    # Check each recorded stage
    for stage in system.stages:
        # Check conservation: active_power + clipped_power == total_power
        stage_sum = stage.active_power + stage.clipped_power
        assert np.isclose(stage_sum, stage.total_power, atol=1e-10), (
            f"Stage '{stage.name}' violates energy conservation: "
            f"active ({stage.active_power}) + clipped ({stage.clipped_power}) != total ({stage.total_power})"
        )

    # Check loss budget table
    for row in metrics.loss_budget_table:
        surv = row["Surviving Power"]
        lost = row["Power Lost"]
        assert surv >= -1e-12, "Surviving power cannot be negative"
        assert lost >= -1e-12, "Power lost cannot be negative"

    # Final breakdown at fiber face:
    # power arriving at fiber plane = accepted + rejected_by_position + rejected_by_na + rejected_by_both
    p_plane = coupling_res.power_reaching_fiber_plane
    p_accepted = coupling_res.accepted_power

    mask_plane = (
        (final_bundle.status == RayStatus.ACCEPTED_BY_FIBER)
        | (final_bundle.status == RayStatus.REJECTED_BY_POSITION)
        | (final_bundle.status == RayStatus.REJECTED_BY_NA)
        | (final_bundle.status == RayStatus.REJECTED_BY_BOTH)
    )
    p_classified = float(np.sum(final_bundle.power[mask_plane]))
    assert np.isclose(p_plane, p_classified, atol=1e-10), (
        f"Sum of fiber classified power ({p_classified}) must equal power reaching fiber plane ({p_plane})"
    )

    # Overall system balance:
    # total_launched = accepted + clipped_along_chain + rejected_at_fiber
    mask_clipped = final_bundle.status == RayStatus.CLIPPED
    p_clipped = float(np.sum(final_bundle.power[mask_clipped]))
    p_fiber_rejected = (
        coupling_res.power_reaching_fiber_plane - coupling_res.accepted_power
    )

    total_accounted = p_accepted + p_clipped + p_fiber_rejected
    assert np.isclose(total_accounted, total_launched, atol=1e-10), (
        f"Total accounted power ({total_accounted}) must equal launched power ({total_launched})"
    )
