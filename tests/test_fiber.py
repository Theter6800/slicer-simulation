"""
Test 5: Fiber spatial and angular acceptance criteria.
"""

import numpy as np
import pytest
from optics.ray import RayBundle, RayStatus
from optics.fiber import Fiber


def test_fiber_coupling_acceptance_and_rejection():
    """
    TEST 5:
    - Centered zero-angle ray must be accepted.
    - Centered 13-degree ray must be rejected for NA = 0.22 in air.
    - Ray at radius > 0.5 mm must be rejected regardless of angle.
    """
    fiber = Fiber(core_diameter=1.0, na=0.22, position=(0.0, 0.0, 200.0), n_external=1.0)
    z_fiber = 200.0

    # Ray 1: Centered at r = 0, angle = 0 deg (on-axis) -> ACCEPTED
    r1 = [0.0, 0.0, z_fiber]
    k1 = [0.0, 0.0, 1.0]

    # Ray 2: Centered at r = 0 at fiber face, angle = 13 deg in x-z plane -> REJECTED BY NA
    # theta = 13 deg > arcsin(0.22) ~ 12.71 deg
    theta_13_rad = np.radians(13.0)
    r2 = [0.0, 0.0, z_fiber]
    k2 = [np.sin(theta_13_rad), 0.0, np.cos(theta_13_rad)]

    # Ray 3: Centered at r = 0 at fiber face, angle = 10 deg (inside NA = 0.22) -> ACCEPTED
    theta_10_rad = np.radians(10.0)
    r3 = [0.0, 0.0, z_fiber]
    k3 = [np.sin(theta_10_rad), 0.0, np.cos(theta_10_rad)]

    # Ray 4: Off-axis at r = 0.6 mm (> 0.5 mm core radius), angle = 0 deg -> REJECTED BY POSITION
    r4 = [0.6, 0.0, z_fiber]
    k4 = [0.0, 0.0, 1.0]

    # Ray 5: Off-axis at r = 0.6 mm, angle = 13 deg -> REJECTED BY BOTH
    r5 = [0.6, 0.0, z_fiber]
    k5 = [np.sin(theta_13_rad), 0.0, np.cos(theta_13_rad)]

    r = np.array([r1, r2, r3, r4, r5])
    k = np.array([k1, k2, k3, k4, k5])

    bundle = RayBundle(r=r, k=k)
    res = fiber.evaluate_coupling(bundle)

    assert bundle.status[0] == RayStatus.ACCEPTED_BY_FIBER, "Centered 0-deg ray must be accepted"
    assert bundle.status[1] == RayStatus.REJECTED_BY_NA, "Centered 13-deg ray must be rejected by NA"
    assert bundle.status[2] == RayStatus.ACCEPTED_BY_FIBER, "Centered 10-deg ray must be accepted"
    assert bundle.status[3] == RayStatus.REJECTED_BY_POSITION, "Ray at r=0.6mm must be rejected by position"
    assert bundle.status[4] == RayStatus.REJECTED_BY_BOTH, "Ray at r=0.6mm, 13-deg must fail both conditions"

    assert res.n_accepted == 2
    assert res.n_rejected_by_na == 1
    assert res.n_rejected_by_position == 1
    assert res.n_rejected_by_both == 1
