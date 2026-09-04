"""
Test 7: Slicer channel ID retention, gap clipping, and channel steering.
"""

import numpy as np
import pytest
from optics.ray import RayBundle, RayStatus
from optics.slicer import SlicerArray, SliceMirror
from optics.elements import ThinLens
from optics.fiber import Fiber


def test_slicer_channel_id_persistence_downstream():
    """
    TEST 7: A ray hitting slice N must retain channel ID N downstream through subsequent optics.
    """
    slicer = SlicerArray(
        name="TestSlicer",
        z=100.0,
        layout="2x2",
        slice_width=10.0,
        slice_height=10.0,
        gap_x=1.0,
        gap_y=1.0,
    )
    # The 4 slices are centered around:
    # 0: (-5.5,  5.5)
    # 1: ( 5.5,  5.5)
    # 2: (-5.5, -5.5)
    # 3: ( 5.5, -5.5)

    # Launch 4 test rays aimed at the center of each slice
    r_in = np.array([
        [-5.5,  5.5, 0.0],  # hits slice 0
        [ 5.5,  5.5, 0.0],  # hits slice 1
        [-5.5, -5.5, 0.0],  # hits slice 2
        [ 5.5, -5.5, 0.0],  # hits slice 3
    ])
    k_in = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    bundle = RayBundle(r=r_in, k=k_in)
    slicer.trace(bundle)

    # Verify each ray was assigned correct channel_id
    assert np.array_equal(bundle.channel_id, [0, 1, 2, 3]), (
        f"Expected channels [0, 1, 2, 3], got {bundle.channel_id}"
    )

    # Propagate through a downstream lens
    lens = ThinLens(name="RelayLens", z=150.0, focal_length=75.0, diameter=60.0)
    lens.trace(bundle)

    # Channel IDs must remain unchanged
    assert np.array_equal(bundle.channel_id, [0, 1, 2, 3]), (
        f"Channel IDs must persist downstream of relay lens, got {bundle.channel_id}"
    )

    # Propagate to fiber
    fiber = Fiber(core_diameter=20.0, na=0.5, position=(0.0, 0.0, 200.0))
    fiber.evaluate_coupling(bundle)

    assert np.array_equal(bundle.channel_id, [0, 1, 2, 3]), (
        "Channel IDs must persist through fiber evaluation"
    )


def test_slicer_gap_clipping():
    """
    Verify that rays hitting the gap between slicer mirrors are clipped.
    """
    slicer = SlicerArray(
        name="GapTestSlicer",
        z=100.0,
        layout="2x2",
        slice_width=10.0,
        slice_height=10.0,
        gap_x=2.0,  # 2 mm gap around x = 0
        gap_y=2.0,  # 2 mm gap around y = 0
    )

    # Ray hitting exactly at (0, 0) which is in the central gap
    r_gap = np.array([[0.0, 0.0, 0.0]])
    k_gap = np.array([[0.0, 0.0, 1.0]])

    bundle = RayBundle(r=r_gap, k=k_gap)
    slicer.trace(bundle)

    assert bundle.status[0] == RayStatus.CLIPPED, "Ray striking inter-slice gap must be CLIPPED"
    assert bundle.channel_id[0] == -1, "Gap ray must have unassigned channel (-1)"
