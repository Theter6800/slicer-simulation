"""
Tolerance and Alignment Sensitivity Engine.
Evaluates the physical coupling efficiency degradation eta / eta_0 under realistic
mechanical and optical alignment perturbations:
- Slicer tip / tilt: +/- 0.01 deg, +/- 0.05 deg, +/- 0.1 deg
- Pupil mirror tip / tilt: +/- 0.01 deg, +/- 0.05 deg, +/- 0.1 deg
- Pupil mirror position: +/- 0.1 mm, +/- 0.5 mm, +/- 1.0 mm
- Fiber position: +/- 0.05 mm, +/- 0.10 mm, +/- 0.25 mm
- Condenser lens axial position: +/- 0.5 mm, +/- 1.0 mm, +/- 2.0 mm
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import pandas as pd

from .system import OpticalSystem
from .ray import RayBundle
from .sources import generate_sun_source
from .geometry3d import normalize


@dataclass
class SensitivityCurve:
    """Coupling efficiency degradation curve for a specific perturbed parameter."""
    component_name: str
    parameter_name: str
    units: str
    perturbations: List[float]
    efficiencies: List[float]
    normalized_efficiencies: List[float]  # eta / eta_0
    tolerance_90pct: float               # max perturbation maintaining >= 90% eta_0
    max_drop_pct: float                  # maximum observed percentage drop
    sensitivity_slope: float             # |d(eta/eta0) / d(param)| near origin


@dataclass
class SensitivityReport:
    """Comprehensive sensitivity analysis report."""
    nominal_efficiency: float
    curves: Dict[str, SensitivityCurve]
    most_sensitive_component: str
    recommended_tolerances: Dict[str, str]
    summary_table: pd.DataFrame


def _rotate_normal(normal: np.ndarray, angle_deg: float, axis: str = "x") -> np.ndarray:
    """Rotate a unit normal vector by angle_deg around x or y axis."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    n = normal.copy()
    if axis == "x":
        rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ], dtype=np.float64)
    elif axis == "y":
        rot = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ], dtype=np.float64)
    else:  # z
        rot = np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    return normalize(rot @ n)


class ToleranceSensitivityEngine:
    """
    Simulates mechanical alignment errors and generates degradation curves.
    """

    DEFAULT_SLICER_TIP_TILT_DEG = [-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]
    DEFAULT_PUPIL_TIP_TILT_DEG = [-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]
    DEFAULT_PUPIL_POS_MM = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
    DEFAULT_FIBER_POS_MM = [-0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25]
    DEFAULT_LENS_AXIAL_MM = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    def __init__(
        self,
        base_system: OpticalSystem,
        n_rays: int = 1500,
        seed: int = 42,
    ):
        self.base_system = base_system
        self.n_rays = n_rays
        self.seed = seed
        # Generate nominal source bundle
        _, self.source_bundle = generate_sun_source(n_rays=n_rays, pupil_diameter=12.0, seed=seed)

    def evaluate_system(self, sys: OpticalSystem) -> float:
        """Trace source bundle through system and return total coupling efficiency."""
        b = self.source_bundle.clone()
        sys.trace(b)
        pa = sys.last_power_accounting
        if pa is not None:
            return float(pa.eta_total)
        m = sys.last_metrics
        return float(m.geometric_coupling_efficiency) if m else 0.0

    def run_analysis(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> SensitivityReport:
        """Runs the full battery of tolerance perturbation sweeps."""
        eta_0 = self.evaluate_system(copy.deepcopy(self.base_system))
        if eta_0 <= 1e-6:
            eta_0 = 1e-6

        curves: Dict[str, SensitivityCurve] = {}

        # 1. Slicer Tip Sweep
        if progress_callback:
            progress_callback(0.1, "Analyzing Slicer Tip sensitivity...")
        slicer_tip_effs = []
        for val in self.DEFAULT_SLICER_TIP_TILT_DEG:
            if abs(val) < 1e-6:
                slicer_tip_effs.append(eta_0)
                continue
            sys_clone = copy.deepcopy(self.base_system)
            if sys_clone.slicer is not None and len(sys_clone.slicer.slices) > 0:
                for s in sys_clone.slicer.slices:
                    s.normal = _rotate_normal(s.normal, val, axis="x")
            slicer_tip_effs.append(self.evaluate_system(sys_clone))

        curves["slicer_tip"] = self._build_curve(
            "Slicer Mirror Tip", "tip_x", "deg", self.DEFAULT_SLICER_TIP_TILT_DEG, slicer_tip_effs, eta_0
        )

        # 2. Slicer Tilt Sweep
        if progress_callback:
            progress_callback(0.25, "Analyzing Slicer Tilt sensitivity...")
        slicer_tilt_effs = []
        for val in self.DEFAULT_SLICER_TIP_TILT_DEG:
            if abs(val) < 1e-6:
                slicer_tilt_effs.append(eta_0)
                continue
            sys_clone = copy.deepcopy(self.base_system)
            if sys_clone.slicer is not None and len(sys_clone.slicer.slices) > 0:
                for s in sys_clone.slicer.slices:
                    s.normal = _rotate_normal(s.normal, val, axis="y")
            slicer_tilt_effs.append(self.evaluate_system(sys_clone))

        curves["slicer_tilt"] = self._build_curve(
            "Slicer Mirror Tilt", "tilt_y", "deg", self.DEFAULT_SLICER_TIP_TILT_DEG, slicer_tilt_effs, eta_0
        )

        # 3. Pupil Mirror Tip Sweep
        if progress_callback:
            progress_callback(0.4, "Analyzing Pupil Mirror Tip sensitivity...")
        pupil_tip_effs = []
        for val in self.DEFAULT_PUPIL_TIP_TILT_DEG:
            if abs(val) < 1e-6:
                pupil_tip_effs.append(eta_0)
                continue
            sys_clone = copy.deepcopy(self.base_system)
            if sys_clone.pupil_relay is not None and len(sys_clone.pupil_relay.mirrors) > 0:
                for pm in sys_clone.pupil_relay.mirrors:
                    new_n = _rotate_normal(pm.normal, val, axis="x")
                    pm.set_pose(pm.center, new_n)
            pupil_tip_effs.append(self.evaluate_system(sys_clone))

        curves["pupil_tip"] = self._build_curve(
            "Pupil Mirror Tip", "tip_x", "deg", self.DEFAULT_PUPIL_TIP_TILT_DEG, pupil_tip_effs, eta_0
        )

        # 4. Pupil Mirror Position Sweep (Transverse dx)
        if progress_callback:
            progress_callback(0.55, "Analyzing Pupil Mirror Position sensitivity...")
        pupil_pos_effs = []
        for val in self.DEFAULT_PUPIL_POS_MM:
            if abs(val) < 1e-6:
                pupil_pos_effs.append(eta_0)
                continue
            sys_clone = copy.deepcopy(self.base_system)
            if sys_clone.pupil_relay is not None and len(sys_clone.pupil_relay.mirrors) > 0:
                for pm in sys_clone.pupil_relay.mirrors:
                    pm.set_pose(pm.center + np.array([val, 0.0, 0.0]), pm.normal)
            pupil_pos_effs.append(self.evaluate_system(sys_clone))

        curves["pupil_pos"] = self._build_curve(
            "Pupil Mirror Position", "transverse_x", "mm", self.DEFAULT_PUPIL_POS_MM, pupil_pos_effs, eta_0
        )

        # 5. Fiber Transverse Position Sweep
        if progress_callback:
            progress_callback(0.7, "Analyzing Fiber Transverse Alignment sensitivity...")
        fiber_pos_effs = []
        for val in self.DEFAULT_FIBER_POS_MM:
            if abs(val) < 1e-6:
                fiber_pos_effs.append(eta_0)
                continue
            sys_clone = copy.deepcopy(self.base_system)
            sys_clone.fiber.position = sys_clone.fiber.position + val * sys_clone.fiber.u
            fiber_pos_effs.append(self.evaluate_system(sys_clone))

        curves["fiber_pos"] = self._build_curve(
            "Fiber Transverse Position", "transverse_u", "mm", self.DEFAULT_FIBER_POS_MM, fiber_pos_effs, eta_0
        )

        # 6. Condenser Lens Axial Position Sweep
        if progress_callback:
            progress_callback(0.85, "Analyzing Condenser Axial Position sensitivity...")
        lens_axial_effs = []
        for val in self.DEFAULT_LENS_AXIAL_MM:
            if abs(val) < 1e-6:
                lens_axial_effs.append(eta_0)
                continue
            sys_clone = copy.deepcopy(self.base_system)
            if sys_clone.final_lens_3d is not None:
                sys_clone.final_lens_3d.center = sys_clone.final_lens_3d.center + val * sys_clone.final_lens_3d.normal
            elif sys_clone.coupling_optics:
                sys_clone.coupling_optics[0].z += val
            lens_axial_effs.append(self.evaluate_system(sys_clone))

        curves["lens_axial"] = self._build_curve(
            "Condenser Lens Axial", "axial_z", "mm", self.DEFAULT_LENS_AXIAL_MM, lens_axial_effs, eta_0
        )

        # Identify most sensitive component
        highest_slope = -1.0
        most_sens_key = "fiber_pos"
        for k, c in curves.items():
            if c.sensitivity_slope > highest_slope:
                highest_slope = c.sensitivity_slope
                most_sens_key = k

        most_sensitive_name = curves[most_sens_key].component_name

        # Recommended tolerances (to keep eta >= 90% of eta_0)
        rec_tols: Dict[str, str] = {}
        summary_rows = []
        for k, c in curves.items():
            unit = c.units
            tol_val = c.tolerance_90pct
            tol_str = f"+/- {tol_val:.3f} {unit}" if tol_val < 1.0 else f"+/- {tol_val:.1f} {unit}"
            rec_tols[c.component_name] = tol_str
            summary_rows.append({
                "Component / Parameter": c.component_name,
                "Tested Range": f"{min(c.perturbations):.2f} to +{max(c.perturbations):.2f} {unit}",
                "Max Degradation": f"{c.max_drop_pct:.1f}%",
                "90% Retained Tolerance": tol_str,
                "Relative Sensitivity Slope": f"{c.sensitivity_slope:.2f} / {unit}",
            })

        df_summary = pd.DataFrame(summary_rows)

        if progress_callback:
            progress_callback(1.0, "Sensitivity analysis complete.")

        return SensitivityReport(
            nominal_efficiency=eta_0,
            curves=curves,
            most_sensitive_component=most_sensitive_name,
            recommended_tolerances=rec_tols,
            summary_table=df_summary,
        )

    def _build_curve(
        self,
        component_name: str,
        parameter_name: str,
        units: str,
        perturbations: List[float],
        efficiencies: List[float],
        eta_0: float,
    ) -> SensitivityCurve:
        norm_effs = [float(e / max(eta_0, 1e-9)) for e in efficiencies]
        drops = [float((1.0 - n) * 100.0) for n in norm_effs]
        max_drop = float(max(drops))

        # Find 90% threshold
        tol_90 = float("inf")
        for p, n in zip(perturbations, norm_effs):
            if abs(p) > 1e-6 and n < 0.90:
                tol_90 = min(tol_90, abs(p))
        if np.isinf(tol_90):
            tol_90 = float(max(abs(p) for p in perturbations))

        # Estimate sensitivity slope near 0
        slopes = []
        for p, n in zip(perturbations, norm_effs):
            if 1e-6 < abs(p) <= max(abs(pert) for pert in perturbations) / 2.0:
                slopes.append(abs(1.0 - n) / abs(p))
        slope = float(np.mean(slopes)) if slopes else 0.0

        return SensitivityCurve(
            component_name=component_name,
            parameter_name=parameter_name,
            units=units,
            perturbations=perturbations,
            efficiencies=efficiencies,
            normalized_efficiencies=norm_effs,
            tolerance_90pct=tol_90,
            max_drop_pct=max_drop,
            sensitivity_slope=slope,
        )
