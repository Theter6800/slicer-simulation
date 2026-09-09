"""
Rigorous Optical Power-Accounting and Étendue Conservation Model.
Tracks absolute optical power relative to launched power across 16 physical stages,
enforces conservation (P_in = P_surviving + P_lost) at every interface,
calculates explicit efficiencies (both absolute and conditional),
and verifies the second law of thermodynamics (conservation of étendue).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


@dataclass
class SlicePowerShare:
    """Detailed power accounting for an individual slicer channel i."""
    slice_id: int
    p_incident: float = 0.0
    p_reflected: float = 0.0
    p_correct_pupil: float = 0.0
    p_wrong_pupil: float = 0.0
    p_condenser: float = 0.0
    p_fiber: float = 0.0
    p_core: float = 0.0
    p_na: float = 0.0
    p_accepted: float = 0.0

    @property
    def p_incident_slice(self) -> float:
        return self.p_incident

    @property
    def p_after_slice(self) -> float:
        return self.p_reflected

    @property
    def p_fiber_plane(self) -> float:
        return self.p_fiber


@dataclass
class PowerAccounting:
    """
    Absolute and conditional optical power accounting across all 16 physical stages.
    All power quantities are in Watts (or normalized units where P_launch = 1.0).
    """
    p_launch: float
    p_after_aperture: float
    p_on_image_plane: float
    p_intercepted_by_slicers: float
    p_lost_at_slicer_gaps: float
    p_missed_slicer_array: float
    p_on_correct_pupil: float
    p_on_wrong_pupil: float
    p_missed_all_pupils: float
    p_blocked_by_other_optics: float
    p_on_condenser: float
    p_missed_condenser: float
    p_at_fiber_plane: float
    p_inside_core: float
    p_inside_na: float
    p_inside_core_and_na: float

    # Policy for handling rays striking adjacent/wrong pupil mirrors:
    # "reject_as_stray": terminated as stray at pupil plane (default for optimization)
    # "propagate_physically": reflected physically and traced downstream to condenser
    wrong_pupil_policy: str = "reject_as_stray"

    # Per-slice power tracking dictionary: channel_id -> SlicePowerShare
    slice_shares: Dict[int, SlicePowerShare] = field(default_factory=dict)

    # Pupil surviving power depending on wrong_pupil_policy
    @property
    def p_pupil_output(self) -> float:
        """Surviving optical power leaving the pupil relay."""
        if self.wrong_pupil_policy == "propagate_physically":
            return self.p_on_correct_pupil + self.p_on_wrong_pupil
        return self.p_on_correct_pupil

    @property
    def p_condenser_input(self) -> float:
        """Power arriving at the input of the condenser lens."""
        return self.p_pupil_output

    # Backwards-compatibility properties
    @property
    def p_on_slicers(self) -> float:
        return self.p_on_image_plane

    @property
    def p_after_slicer_gaps(self) -> float:
        return max(0.0, self.p_intercepted_by_slicers - self.p_lost_at_slicer_gaps)

    @property
    def p_missed_pupil(self) -> float:
        return self.p_missed_all_pupils

    @property
    def p_on_final_lens(self) -> float:
        return self.p_on_condenser

    # 1. Efficiencies relative to total launched power (Absolute System Metrics)
    @property
    def eta_slicer_launch(self) -> float:
        """Fraction of launched power intercepted by slicers = P_intercepted / P_launch."""
        return float(self.p_intercepted_by_slicers / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def eta_correct_pupil_launch(self) -> float:
        """Fraction of launched power on assigned pupil mirrors = P_on_correct_pupil / P_launch."""
        return float(self.p_on_correct_pupil / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def eta_condenser_launch(self) -> float:
        """Fraction of launched power reaching condenser = P_on_condenser / P_launch."""
        return float(self.p_on_condenser / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def eta_core_launch(self) -> float:
        """Absolute spatial core acceptance = P_inside_core / P_launch."""
        return float(self.p_inside_core / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def eta_na_launch(self) -> float:
        """Absolute angular NA acceptance = P_inside_NA / P_launch."""
        return float(self.p_inside_na / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def eta_total(self) -> float:
        """Absolute physical coupling efficiency = P_inside_core_AND_NA / P_launch."""
        return float(self.p_inside_core_and_na / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def eta_both_launch(self) -> float:
        """Absolute joint spatial core and angular NA acceptance = P_inside_core_AND_NA / P_launch."""
        return self.eta_total

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
    def eta_NA_conditional(self) -> float:
        """Alias for eta_na_conditional."""
        return self.eta_na_conditional

    @property
    def eta_both_conditional(self) -> float:
        """Conditional coupling efficiency = P_inside_core_AND_NA / P_at_fiber_plane."""
        return float(self.p_inside_core_and_na / self.p_at_fiber_plane) if self.p_at_fiber_plane > 0 else 0.0

    @property
    def eta_coupling_conditional(self) -> float:
        """Alias for eta_both_conditional."""
        return self.eta_both_conditional

    def compute_n_effective(self, active_threshold: float = 0.05) -> Tuple[int, Dict[int, float], bool]:
        """
        Computes effective active slicer channels based on power fraction:
        power_fraction_i = P_incident_on_slice_i / P_total_intercepted_by_slicers.
        A channel is active if power_fraction_i >= active_threshold (default 5%).
        Returns: (n_effective, per_slice_fractions, is_underutilized_warning).
        """
        if not self.slice_shares:
            return 0, {}, False

        p_intercept = self.p_intercepted_by_slicers
        if p_intercept <= 1e-12:
            p_intercept = sum(s.p_incident for s in self.slice_shares.values())

        fractions: Dict[int, float] = {}
        active_count = 0
        for s_id, s_share in self.slice_shares.items():
            frac = float(s_share.p_incident / p_intercept) if p_intercept > 1e-12 else 0.0
            fractions[s_id] = frac
            if frac >= active_threshold:
                active_count += 1

        n_configured = len(self.slice_shares)
        is_warning = (active_count < n_configured) and (n_configured > 1)
        return active_count, fractions, is_warning

    @property
    def n_effective(self) -> int:
        """Count of active slicer channels receiving >= 5% of intercepted power."""
        n_eff, _, _ = self.compute_n_effective(active_threshold=0.05)
        return n_eff

    @property
    def slice_power_fractions(self) -> Dict[int, float]:
        """Per-channel power fraction relative to total intercepted power."""
        _, fracs, _ = self.compute_n_effective()
        return fracs

    def compute_estimated_physical_efficiency(
        self,
        r_slicer: float = 0.98,
        r_pupil: float = 0.98,
        t_lens: float = 0.96,
        is_direct: bool = False,
    ) -> float:
        """
        Calculates estimated physical efficiency incorporating realistic Fresnel and coating losses:
        - For direct baseline (N=0): T_fore * T_condenser = t_lens * t_lens = 0.96 * 0.96 = 0.9216.
        - For slicer systems (N >= 1): T_fore * R_slicer * R_pupil * T_condenser = 0.96 * 0.98 * 0.98 * 0.96 ~= 0.8844.
        Returns: eta_estimated_physical = eta_total * T_physical.
        """
        if is_direct or len(self.slice_shares) == 0:
            t_phys = t_lens * t_lens
        else:
            t_phys = t_lens * r_slicer * r_pupil * t_lens
        return float(self.eta_total * t_phys)

    def classify_dominant_limitation(self, tie_with_baseline: bool = False) -> str:
        """
        Classifies the dominant physical loss mechanism for the architecture:
        - COVERAGE-LIMITED: Significant image power misses slicers (P_intercepted / P_image < 0.90)
        - PUPIL-LIMITED: Significant power misses assigned pupil mirrors (P_pupil / P_intercepted < 0.85)
        - CONDENSER-LIMITED: Significant power misses final coupling optic (P_condenser / P_pupil < 0.90)
        - SPATIAL-FIBER-LIMITED: Most fiber-plane losses occur because r > 0.5 mm (eta_core < 0.70 and eta_na >= 0.85)
        - NA-LIMITED: Most fiber-plane losses occur because theta exceeds acceptance (eta_na < 0.70 and eta_core >= 0.85)
        - JOINT PHASE-SPACE LIMITED: Spatial and angular conditions interact strongly (both < 0.85)
        - NO MATERIAL IMPROVEMENT: Performance statistically indistinguishable from N=0
        """
        if tie_with_baseline:
            return "NO MATERIAL IMPROVEMENT"

        # Check slicer coverage
        if self.p_on_image_plane > 1e-9:
            slic_frac = self.p_intercepted_by_slicers / self.p_on_image_plane
            if slic_frac < 0.90:
                return "COVERAGE-LIMITED"

        # Check pupil interception
        if self.p_intercepted_by_slicers > 1e-9:
            pupil_in = max(1e-9, self.p_after_slicer_gaps)
            pupil_frac = self.p_on_correct_pupil / pupil_in
            if pupil_frac < 0.85:
                return "PUPIL-LIMITED"

        # Check condenser interception
        p_pupil_surv = self.p_pupil_output
        if p_pupil_surv > 1e-9:
            cond_frac = self.p_on_condenser / p_pupil_surv
            if cond_frac < 0.90:
                return "CONDENSER-LIMITED"

        # Fiber plane acceptance
        c_cond = self.eta_core_conditional
        na_cond = self.eta_na_conditional
        if c_cond < 0.70 and na_cond >= 0.85:
            return "SPATIAL-FIBER-LIMITED"
        elif na_cond < 0.70 and c_cond >= 0.85:
            return "NA-LIMITED"
        elif c_cond < 0.85 or na_cond < 0.85:
            return "JOINT PHASE-SPACE LIMITED"

        return "OPTIMIZED"

    # 3. Intermediate stage loss fractions relative to P_launch
    @property
    def loss_aperture_clipping(self) -> float:
        return max(0.0, float(self.p_launch - self.p_after_aperture) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_slicer_miss(self) -> float:
        return max(0.0, float(self.p_missed_slicer_array) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_slicer_gaps(self) -> float:
        return max(0.0, float(self.p_lost_at_slicer_gaps) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_pupil_missed(self) -> float:
        return max(0.0, float(self.p_missed_all_pupils) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_cross_channel_pupil(self) -> float:
        return max(0.0, float(self.p_on_wrong_pupil) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_condenser_missed(self) -> float:
        return max(0.0, float(self.p_missed_condenser) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_blocked_optics(self) -> float:
        return max(0.0, float(self.p_blocked_by_other_optics) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_spatial_core_rejection(self) -> float:
        return max(0.0, float(self.p_at_fiber_plane - self.p_inside_core) / self.p_launch) if self.p_launch > 0 else 0.0

    @property
    def loss_angular_na_rejection(self) -> float:
        return max(0.0, float(self.p_at_fiber_plane - self.p_inside_na) / self.p_launch) if self.p_launch > 0 else 0.0

    def verify_power_conservation(self, tol: float = 1e-6) -> Tuple[bool, List[str]]:
        """
        Enforce P_in = P_surviving + P_lost at every optical interface with strict relative tolerance.
        Returns (is_valid, list_of_discrepancies).
        """
        issues: List[str] = []
        p_norm = max(self.p_launch, 1e-12)

        # Check 1: Fore-optics Aperture Stop
        if self.p_after_aperture > self.p_launch + tol * p_norm:
            issues.append(f"Aperture creates power: P_after ({self.p_after_aperture:.6f}) > P_launch ({self.p_launch:.6f})")

        # Check 2: Image Plane Slicer Interception
        p_slicer_sum = self.p_intercepted_by_slicers + self.p_missed_slicer_array
        if abs(self.p_on_image_plane - p_slicer_sum) / p_norm > tol:
            issues.append(
                f"Image plane mismatch: P_on_image ({self.p_on_image_plane:.6f}) != "
                f"P_intercepted ({self.p_intercepted_by_slicers:.6f}) + P_missed ({self.p_missed_slicer_array:.6f})"
            )

        # Check 3: Slicer Active Facets and Gaps
        p_refl = self.p_after_slicer_gaps
        if abs(self.p_intercepted_by_slicers - (p_refl + self.p_lost_at_slicer_gaps)) / p_norm > tol:
            issues.append(
                f"Slicer gap mismatch: P_intercepted ({self.p_intercepted_by_slicers:.6f}) != "
                f"P_refl ({p_refl:.6f}) + P_gaps ({self.p_lost_at_slicer_gaps:.6f})"
            )

        # Check 4: Pupil Relay Interception & Gaps
        p_pupil_in = p_refl
        p_pupil_tallied = self.p_on_correct_pupil + self.p_on_wrong_pupil + self.p_missed_all_pupils
        if abs(p_pupil_in - p_pupil_tallied) / p_norm > tol:
            issues.append(
                f"Pupil relay input mismatch: P_in ({p_pupil_in:.6f}) != "
                f"P_correct ({self.p_on_correct_pupil:.6f}) + P_wrong ({self.p_on_wrong_pupil:.6f}) + P_missed ({self.p_missed_all_pupils:.6f})"
            )

        # Check 5: Pupil to Condenser Transmission (depends on wrong_pupil_policy)
        p_cond_in = self.p_condenser_input
        p_cond_tallied = self.p_on_condenser + self.p_missed_condenser + self.p_blocked_by_other_optics
        if abs(p_cond_in - p_cond_tallied) / p_norm > tol:
            issues.append(
                f"Condenser transmission mismatch ({self.wrong_pupil_policy}): P_cond_in ({p_cond_in:.6f}) != "
                f"P_on_cond ({self.p_on_condenser:.6f}) + P_missed ({self.p_missed_condenser:.6f}) + P_blocked ({self.p_blocked_by_other_optics:.6f})"
            )

        # Check 6: Condenser to Fiber Plane Arrival
        if abs(self.p_on_condenser - self.p_at_fiber_plane) / p_norm > tol:
            issues.append(
                f"Fiber plane arrival mismatch: P_on_condenser ({self.p_on_condenser:.6f}) != "
                f"P_at_fiber_plane ({self.p_at_fiber_plane:.6f})"
            )

        # Check 7: Fiber Acceptance Bounds
        if self.p_inside_core > self.p_at_fiber_plane + tol * p_norm:
            issues.append(f"Core power exceeds fiber plane power: {self.p_inside_core:.6f} > {self.p_at_fiber_plane:.6f}")
        if self.p_inside_na > self.p_at_fiber_plane + tol * p_norm:
            issues.append(f"NA power exceeds fiber plane power: {self.p_inside_na:.6f} > {self.p_at_fiber_plane:.6f}")
        if self.p_inside_core_and_na > min(self.p_inside_core, self.p_inside_na) + tol * p_norm:
            issues.append(
                f"Combined core+NA power ({self.p_inside_core_and_na:.6f}) exceeds "
                f"min(core={self.p_inside_core:.6f}, NA={self.p_inside_na:.6f})"
            )

        return (len(issues) == 0, issues)

    def verify_per_channel_consistency(self, tol: float = 1e-6) -> Tuple[bool, List[str]]:
        """
        Verify that per-channel power shares sum exactly to the global stage quantities:
        - sum_i(P_fiber_plane_i) = P_at_fiber
        - sum_i(P_accepted_i) = P_accepted
        - sum_i(P_core_i) = P_inside_core
        - sum_i(P_NA_i) = P_inside_na
        Returns (is_valid, list_of_discrepancies).
        """
        if not self.slice_shares:
            return True, []

        issues: List[str] = []
        p_norm = max(self.p_launch, 1e-12)

        sum_fiber = float(sum(s.p_fiber for s in self.slice_shares.values()))
        sum_accepted = float(sum(s.p_accepted for s in self.slice_shares.values()))
        sum_core = float(sum(s.p_core for s in self.slice_shares.values()))
        sum_na = float(sum(s.p_na for s in self.slice_shares.values()))

        if abs(sum_fiber - self.p_at_fiber_plane) / p_norm > tol:
            issues.append(
                f"Per-channel fiber plane sum mismatch: sum_i(P_fiber_i)={sum_fiber:.6f} W != "
                f"P_at_fiber_plane={self.p_at_fiber_plane:.6f} W (delta={abs(sum_fiber - self.p_at_fiber_plane)/p_norm:.2e})"
            )

        if abs(sum_accepted - self.p_inside_core_and_na) / p_norm > tol:
            issues.append(
                f"Per-channel accepted sum mismatch: sum_i(P_accepted_i)={sum_accepted:.6f} W != "
                f"P_accepted={self.p_inside_core_and_na:.6f} W (delta={abs(sum_accepted - self.p_inside_core_and_na)/p_norm:.2e})"
            )

        if abs(sum_core - self.p_inside_core) / p_norm > tol:
            issues.append(
                f"Per-channel core sum mismatch: sum_i(P_core_i)={sum_core:.6f} W != "
                f"P_inside_core={self.p_inside_core:.6f} W (delta={abs(sum_core - self.p_inside_core)/p_norm:.2e})"
            )

        if abs(sum_na - self.p_inside_na) / p_norm > tol:
            issues.append(
                f"Per-channel NA sum mismatch: sum_i(P_na_i)={sum_na:.6f} W != "
                f"P_inside_na={self.p_inside_na:.6f} W (delta={abs(sum_na - self.p_inside_na)/p_norm:.2e})"
            )

        return (len(issues) == 0, issues)

    def verify_all(self, tol: float = 1e-6) -> Tuple[bool, List[str]]:
        """Run both stage conservation and per-channel consistency checks."""
        ok_stage, issues_stage = self.verify_power_conservation(tol=tol)
        ok_ch, issues_ch = self.verify_per_channel_consistency(tol=tol)
        all_issues = issues_stage + issues_ch
        return (ok_stage and ok_ch, all_issues)

    def to_dict(self) -> Dict[str, Any]:
        """Convert power accounting to dictionary with formatted percentages and stage powers."""
        is_cons, cons_issues = self.verify_power_conservation()
        return {
            "P_launch": self.p_launch,
            "P_after_aperture": self.p_after_aperture,
            "P_on_image_plane": self.p_on_image_plane,
            "P_intercepted_by_slicers": self.p_intercepted_by_slicers,
            "P_lost_at_slicer_gaps": self.p_lost_at_slicer_gaps,
            "P_missed_slicer_array": self.p_missed_slicer_array,
            "P_on_correct_pupil": self.p_on_correct_pupil,
            "P_on_wrong_pupil": self.p_on_wrong_pupil,
            "P_missed_all_pupils": self.p_missed_all_pupils,
            "P_blocked_by_other_optics": self.p_blocked_by_other_optics,
            "P_on_condenser": self.p_on_condenser,
            "P_missed_condenser": self.p_missed_condenser,
            "P_at_fiber_plane": self.p_at_fiber_plane,
            "P_inside_core": self.p_inside_core,
            "P_inside_NA": self.p_inside_na,
            "P_inside_core_AND_NA": self.p_inside_core_and_na,
            # Absolute efficiencies
            "eta_slicer_launch": self.eta_slicer_launch,
            "eta_correct_pupil_launch": self.eta_correct_pupil_launch,
            "eta_condenser_launch": self.eta_condenser_launch,
            "eta_core_launch": self.eta_core_launch,
            "eta_NA_launch": self.eta_na_launch,
            "eta_total": self.eta_total,
            "eta_both_launch": self.eta_both_launch,
            # Conditional efficiencies
            "eta_core_conditional": self.eta_core_conditional,
            "eta_NA_conditional": self.eta_na_conditional,
            "eta_both_conditional": self.eta_both_conditional,
            "eta_coupling_conditional": self.eta_coupling_conditional,
            # Effective channel metrics & physical efficiency
            "N_effective": self.n_effective,
            "slice_power_fractions": self.slice_power_fractions,
            "eta_estimated_physical": self.compute_estimated_physical_efficiency(),
            "dominant_limitation": self.classify_dominant_limitation(),
            # Conservation status
            "is_power_conserved": is_cons,
            "conservation_issues": cons_issues,
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
