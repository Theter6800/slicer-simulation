"""
Rigorous Optical Power-Accounting and Étendue Conservation Model.
Tracks absolute optical power relative to launched power across every stage,
calculates explicit efficiencies (both relative to launch and conditional on fiber arrival),
and verifies the second law of thermodynamics (conservation of étendue).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np


@dataclass
class PowerAccounting:
    """
    Absolute and conditional optical power accounting across all physical stages.
    All power quantities are in Watts (or normalized units where P_launch = 1.0).
    """
    p_launch: float
    p_after_aperture: float
    p_on_slicers: float
    p_after_slicer_gaps: float
    p_on_correct_pupil: float
    p_on_wrong_pupil: float
    p_missed_pupil: float
    p_on_final_lens: float
    p_at_fiber_plane: float
    p_inside_core: float
    p_inside_na: float
    p_inside_core_and_na: float

    # 1. Efficiencies relative to total launched power (Absolute System Metrics)
    @property
    def eta_total(self) -> float:
        """Absolute physical coupling efficiency = P_inside_core_AND_NA / P_launch."""
        return float(self.p_inside_core_and_na / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def eta_core_launch(self) -> float:
        """Absolute spatial core acceptance = P_inside_core / P_launch."""
        return float(self.p_inside_core / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def eta_na_launch(self) -> float:
        """Absolute angular NA acceptance = P_inside_NA / P_launch."""
        return float(self.p_inside_na / self.p_launch) if self.p_launch > 0 else 0.0

    # 2. Conditional efficiencies relative to power reaching fiber plane
    @property
    def eta_core_conditional(self) -> float:
        """Conditional spatial acceptance = P_inside_core / P_at_fiber_plane."""
        return float(self.p_inside_core / self.p_at_fiber_plane) if self.p_at_fiber_plane > 0 else 0.0

    @property
    def eta_na_conditional(self) -> float:
        """Conditional angular acceptance = P_inside_NA / P_at_fiber_plane."""
        return float(self.p_inside_na / self.p_at_fiber_plane) if self.p_at_fiber_plane > 0 else 0.0

    @property
    def eta_coupling_conditional(self) -> float:
        """Conditional coupling efficiency = P_inside_core_AND_NA / P_at_fiber_plane."""
        return float(self.p_inside_core_and_na / self.p_at_fiber_plane) if self.p_at_fiber_plane > 0 else 0.0

    # 3. Intermediate stage loss fractions relative to P_launch
    @property
    def loss_aperture_clipping(self) -> float:
        return max(0.0, float(self.p_launch - self.p_after_aperture) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_slicer_gaps(self) -> float:
        return max(0.0, float(self.p_on_slicers - self.p_after_slicer_gaps) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_pupil_missed(self) -> float:
        return max(0.0, float(self.p_missed_pupil) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_cross_channel_pupil(self) -> float:
        return max(0.0, float(self.p_on_wrong_pupil) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_final_lens_clipping(self) -> float:
        return max(0.0, float(self.p_on_correct_pupil - self.p_on_final_lens) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_spatial_core_rejection(self) -> float:
        return max(0.0, float(self.p_at_fiber_plane - self.p_inside_core) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_angular_na_rejection(self) -> float:
        return max(0.0, float(self.p_at_fiber_plane - self.p_inside_na) / self.p_launch) if self.p_launch > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert power accounting to dictionary with formatted percentages and stage powers."""
        return {
            "P_launch": self.p_launch,
            "P_after_aperture": self.p_after_aperture,
            "P_on_slicers": self.p_on_slicers,
            "P_after_slicer_gaps": self.p_after_slicer_gaps,
            "P_on_correct_pupil": self.p_on_correct_pupil,
            "P_on_wrong_pupil": self.p_on_wrong_pupil,
            "P_missed_pupil": self.p_missed_pupil,
            "P_on_final_lens": self.p_on_final_lens,
            "P_at_fiber_plane": self.p_at_fiber_plane,
            "P_inside_core": self.p_inside_core,
            "P_inside_NA": self.p_inside_na,
            "P_inside_core_AND_NA": self.p_inside_core_and_na,
            "eta_total": self.eta_total,
            "eta_core_launch": self.eta_core_launch,
            "eta_NA_launch": self.eta_na_launch,
            "eta_core_conditional": self.eta_core_conditional,
            "eta_NA_conditional": self.eta_na_conditional,
            "eta_coupling_conditional": self.eta_coupling_conditional,
        }


def compute_etendue(
    pupil_diameter: float = 12.0,
    solar_angular_radius_deg: float = 0.266,
    fiber_core_diameter: float = 1.0,
    fiber_na: float = 0.22,
) -> Tuple[float, float, float, bool]:
    """
    Calculate optical étendue for source/telescope entrance and receiver optical fiber.

    Parameters:
    - pupil_diameter: Entrance aperture / entrance pupil diameter (mm)
    - solar_angular_radius_deg: Half-angle radius of the Sun (0.266 deg)
    - fiber_core_diameter: Multimode fiber core diameter (mm, e.g. 1.0 mm)
    - fiber_na: Fiber numerical aperture (e.g. 0.22)

    Returns:
    - G_source: Source étendue (mm^2 * sr)
    - G_fiber: Fiber acceptance étendue (mm^2 * sr)
    - eta_max_etendue: Theoretical maximum possible passive coupling efficiency min(1.0, G_fiber / G_source)
    - is_physically_allowed: True if G_source <= G_fiber (100% passive transmission is permitted)
    """
    # 1. Source Étendue
    a_pupil = np.pi * (pupil_diameter / 2.0)**2
    alpha_rad = np.radians(solar_angular_radius_deg)
    omega_sun = np.pi * (np.sin(alpha_rad))**2
    g_source = float(a_pupil * omega_sun)

    # 2. Fiber Étendue
    r_core = fiber_core_diameter / 2.0
    a_fiber = np.pi * (r_core**2)
    omega_fiber = np.pi * (fiber_na**2)
    g_fiber = float(a_fiber * omega_fiber)

    # 3. Maximum theoretically permissible passive concentration / transmission
    eta_max = float(min(1.0, g_fiber / g_source)) if g_source > 0 else 1.0
    is_allowed = g_source <= g_fiber

    return g_source, g_fiber, eta_max, is_allowed
