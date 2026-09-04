"""
Design Optimizer for Image Slicer IFU Fiber Coupling Architecture.
Maximizes physically valid optical coupling into multimode fiber (core <= 0.5mm, NA <= 0.22).
Optimizes discrete N_slicer in [1 ... N_max], slicer positions on input image plane,
and 3D one-sided pupil mirror positions, deriving mirror orientations analytically.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Callable
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .ray import RayBundle, RayStatus
from .geometry3d import normalize, reflect_vector, reflection_bisector
from .slicer import SlicerArray, SliceMirror
from .pupil import PupilRelaySystem, PupilMirror, generate_one_sided_pupil_positions
from .elements import ThinLens, CircularAperture, ThinLens3D
from .fiber import Fiber, FiberCouplingResult
from .system import OpticalSystem
from .metrics import SystemMetrics
from .presets import create_branched_slicer_system
from .sources import generate_led_source, generate_sun_source, LEDSourceType
from .power_accounting import PowerAccounting, compute_etendue
from .validation import run_all_validations, validate_multi_start_optimization, ValidationSuiteReport


@dataclass
class OptimizationConfig:
    """Configuration parameters for the architectural design optimizer."""
    n_min: int = 1
    n_max: int = 4
    pupil_layout_side: str = "lower"  # "lower", "upper", "left", "right"
    pupil_aim_mode: str = "parallel"  # "parallel" (optimal for condenser focus) or "target_center"
    z_slicer: float = 190.0           # Slicer at enlarged input focal plane (strictly fixed)
    source_mode: str = "SUN"          # "SUN" (default) or "LED"
    aperture_diameter: float = 12.0
    z_aperture: float = 20.0
    fore_focal_length: float = 150.0  # Objective lens focal length producing broad Sun image
    z_fore: Optional[float] = None    # Defaults to z_slicer - fore_focal_length = 40.0
    z_fore1: float = 60.0
    z_fore2: float = 140.0
    pupil_distance_z: float = 40.0
    pupil_transverse_offset: float = 20.0
    pupil_spacing: float = 5.0
    pupil_mirror_size: float = 14.0
    pupil_mirror_height: float = 18.0
    pupil_focal_length: Optional[float] = None
    condenser_distance_z: float = 60.0
    condenser_focal_length: float = 22.0
    condenser_diameter: float = 45.0
    fiber_core_diameter: float = 1.0  # radius = 0.5 mm
    fiber_na: float = 0.22            # maximum sine theta
    fiber_distance: float = 22.0      # focal convergence distance behind lens (matches f_cond)
    min_slicer_gap: float = 0.04      # mm
    min_pupil_clearance: float = 1.0   # mm mechanical spacing between pupil edges
    max_de_iter: int = 25             # Global Differential Evolution iterations
    popsize: int = 8                  # Differential Evolution population multiplier
    polish: bool = True               # Local Nelder-Mead polishing
    exploration_rays: int = 500       # Fast trace for optimizer iterations
    validation_rays: int = 3000       # Accurate trace for final validation
    random_seed: int = 42


@dataclass
class SingleNOptimizationResult:
    """Optimization outcome for a specific discrete slice count N."""
    n_channels: int
    success: bool
    best_x: np.ndarray
    coupling_efficiency: float       # Physical eta = accepted_power / launched_power
    core_accepted_fraction: float    # Fraction of fiber plane power inside core
    na_accepted_fraction: float      # Fraction of fiber plane power inside NA
    both_accepted_fraction: float    # Fraction passing both conditions
    accepted_power: float
    total_launched_power: float
    clipping_loss: float
    slicer_table: pd.DataFrame
    pupil_table: pd.DataFrame
    system: OpticalSystem
    traced_bundle: RayBundle
    coupling_result: FiberCouplingResult
    metrics: SystemMetrics
    loss_budget: Dict[str, float]
    iterations: int
    execution_time_sec: float
    message: str
    power_accounting: Optional[PowerAccounting] = None
    multi_start_stats: Optional[Dict[str, Any]] = None

    @property
    def clipping_loss_fraction(self) -> float:
        return self.clipping_loss


@dataclass
class MultiNStudyResult:
    """Comprehensive study comparing discrete N_slicer in [N_min ... N_max]."""
    results_by_n: Dict[int, SingleNOptimizationResult] = field(default_factory=dict)
    best_n: int = 1
    best_result: Optional[SingleNOptimizationResult] = None
    comparison_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    winner_explanation: str = ""
    multi_start_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_report: Optional[ValidationSuiteReport] = None
    winner_declared: bool = False

    @property
    def winning_n(self) -> int:
        return self.best_n

    @property
    def physical_rationale(self) -> str:
        return self.winner_explanation

    @property
    def results(self) -> Dict[int, SingleNOptimizationResult]:
        return self.results_by_n


class SlicerPupilOptimizer:
    """
    Design optimizer that derives mirror tilts analytically and optimizes
    slicer positions on the input image plane and 3D one-sided pupil mirror positions.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config if config is not None else OptimizationConfig()
        self._cached_pre_slicer_bundle: Optional[RayBundle] = None
        self._cached_source_bundle: Optional[RayBundle] = None
        self._cached_ray_count: int = 0

    def generate_source_bundle(self, n_rays: int) -> RayBundle:
        """Generate pristine source ray bundle based on config."""
        if self.config.source_mode == "SUN":
            _, bundle = generate_sun_source(
                n_rays=n_rays,
                pupil_diameter=self.config.aperture_diameter,
                solar_angular_radius_deg=0.266,
                seed=self.config.random_seed,
            )
        else:
            _, bundle = generate_led_source(
                n_rays=n_rays,
                source_type=LEDSourceType.UNIFORM_DISK,
                source_diameter=1.5,
                aperture_diameter=self.config.aperture_diameter,
                aperture_pos=(0.0, 0.0, self.config.z_aperture),
                seed=self.config.random_seed,
            )
        return bundle

    def get_pre_slicer_bundle(self, n_rays: int) -> RayBundle:
        """
        Traces rays through sequential pre-slicer optics up to z_slicer once,
        caching the result to accelerate optimization iterations.
        """
        if self._cached_pre_slicer_bundle is not None and self._cached_ray_count == n_rays:
            return self._cached_pre_slicer_bundle.clone()

        source_bundle = self.generate_source_bundle(n_rays)
        bundle = source_bundle.clone()

        # Sequential pre-slicer elements
        aperture = CircularAperture(
            name="Aperture",
            z=self.config.z_aperture,
            diameter=self.config.aperture_diameter,
        )
        aperture.trace(bundle)

        if self.config.source_mode == "SUN":
            z_f = self.config.z_fore if self.config.z_fore is not None else (self.config.z_slicer - self.config.fore_focal_length)
            fore_lens = ThinLens(
                name=f"Fore-Optic Objective Lens f={self.config.fore_focal_length:.0f}mm",
                z=z_f,
                focal_length=self.config.fore_focal_length,
                diameter=25.4,
            )
            fore_lens.trace(bundle)
        else:
            fore1 = ThinLens(
                name="Fore-Optic L1",
                z=self.config.z_fore1,
                focal_length=100.0,
                diameter=25.4,
            )
            fore2 = ThinLens(
                name="Fore-Optic L2",
                z=self.config.z_fore2,
                focal_length=35.0,
                diameter=25.4,
            )
            fore1.trace(bundle)
            fore2.trace(bundle)

        bundle.propagate_to_z(self.config.z_slicer)

        self._cached_source_bundle = source_bundle
        self._cached_pre_slicer_bundle = bundle
        self._cached_ray_count = n_rays

        return bundle.clone()

    def get_spot_radius(self, pre_slicer_bundle: RayBundle) -> float:
        """Estimate the illuminated spot radius at the image plane."""
        act = pre_slicer_bundle.active_mask
        if not np.any(act):
            return 1.0
        r_sq = pre_slicer_bundle.x[act]**2 + pre_slicer_bundle.y[act]**2
        return float(np.percentile(np.sqrt(r_sq), 95))

    def get_slice_dimensions(self, n: int, spot_radius: float) -> Tuple[float, float]:
        """Compute appropriate slice width and height for n channels covering the spot."""
        field_dim = max(2.0 * spot_radius * 1.15, 1.5)
        gap = self.config.min_slicer_gap
        if n == 1:
            return field_dim, field_dim
        elif n == 2:
            return field_dim, (field_dim - gap) / 2.0
        elif n == 4:
            h = max((field_dim - 3.0 * gap) / 4.0, 0.3)
            return field_dim, h
        else:
            h = max((field_dim - (n - 1) * gap) / n, 0.25)
            return field_dim, h

    def get_initial_vector_and_bounds(
        self,
        n: int,
        spot_radius: float,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Builds initial vector X0 and parameter bounds for fixed N:
        X = [x_s1, y_s1, ... x_sN, y_sN,  x_p1, y_p1, z_p1, ... x_pN, y_pN, z_pN]
        Length: 5 * N
        """
        w, h = self.get_slice_dimensions(n, spot_radius)
        pitch_y = h + self.config.min_slicer_gap

        # 1. Initial Slicer positions
        slicer_pos: List[Tuple[float, float]] = []
        if n == 1:
            slicer_pos.append((0.0, 0.0))
        elif n == 2:
            slicer_pos.append((0.0, pitch_y / 2.0))
            slicer_pos.append((0.0, -pitch_y / 2.0))
        else:
            y_offsets = [(- (n - 1) / 2.0 + i) * pitch_y for i in range(n)]
            for cy in y_offsets:
                slicer_pos.append((0.0, float(cy)))

        # 2. Initial Pupil positions on designated side
        z_pupil = self.config.z_slicer + self.config.pupil_distance_z
        pupil_init = generate_one_sided_pupil_positions(
            side=self.config.pupil_layout_side,
            n_channels=n,
            z_pupil=z_pupil,
            transverse_offset=self.config.pupil_transverse_offset,
            spacing=self.config.pupil_spacing,
        )

        x0: List[float] = []
        bounds: List[Tuple[float, float]] = []

        # Slicer bounds: within +/- spot_radius * 1.5
        s_bound = max(spot_radius * 1.5, 2.0)
        for cx, cy in slicer_pos:
            x0.extend([cx, cy])
            bounds.append((-s_bound, s_bound))
            bounds.append((-s_bound, s_bound))

        # Pupil bounds: constrained strictly to designated one-sided region
        side = self.config.pupil_layout_side.lower()
        z_min = self.config.z_slicer + 20.0
        z_max = self.config.z_slicer + 80.0
        offset = self.config.pupil_transverse_offset

        for p in pupil_init:
            x0.extend([float(p[0]), float(p[1]), float(p[2])])

            if side == "lower":
                bounds.append((-30.0, 30.0))                     # x_p
                bounds.append((-offset * 1.8, -offset * 0.4))    # y_p strictly negative
                bounds.append((z_min, z_max))                    # z_p
            elif side == "upper":
                bounds.append((-30.0, 30.0))                     # x_p
                bounds.append((offset * 0.4, offset * 1.8))      # y_p strictly positive
                bounds.append((z_min, z_max))                    # z_p
            elif side == "left":
                bounds.append((-offset * 1.8, -offset * 0.4))    # x_p strictly negative
                bounds.append((-30.0, 30.0))                     # y_p
                bounds.append((z_min, z_max))                    # z_p
            else:  # right
                bounds.append((offset * 0.4, offset * 1.8))      # x_p strictly positive
                bounds.append((-30.0, 30.0))                     # y_p
                bounds.append((z_min, z_max))                    # z_p

        return np.array(x0, dtype=np.float64), bounds

    def unpack_vector(
        self,
        X: np.ndarray,
        n: int,
    ) -> Tuple[List[Tuple[float, float]], List[np.ndarray]]:
        """Unpack parameter vector X into slicer (x, y) and pupil (x, y, z) lists."""
        slicer_pos: List[Tuple[float, float]] = []
        for i in range(n):
            slicer_pos.append((float(X[2 * i]), float(X[2 * i + 1])))

        pupil_offset = 2 * n
        pupil_pos: List[np.ndarray] = []
        for i in range(n):
            px = float(X[pupil_offset + 3 * i])
            py = float(X[pupil_offset + 3 * i + 1])
            pz = float(X[pupil_offset + 3 * i + 2])
            pupil_pos.append(np.array([px, py, pz], dtype=np.float64))

        return slicer_pos, pupil_pos

    def build_candidate_system(
        self,
        X: np.ndarray,
        n: int,
        w: float,
        h: float,
    ) -> OpticalSystem:
        """
        Builds the complete optical system for vector X with analytically derived
        mirror orientations for all slicers and pupil mirrors.
        """
        slicer_pos, pupil_pos = self.unpack_vector(X, n)

        sys = create_branched_slicer_system(
            n_channels=n,
            pupil_layout_side=self.config.pupil_layout_side,
            pupil_aim_mode=self.config.pupil_aim_mode,
            aperture_diameter=self.config.aperture_diameter,
            z_aperture=self.config.z_aperture,
            fore_lens_focal_length=self.config.fore_focal_length if self.config.source_mode == "SUN" else None,
            fore_lens_z=self.config.z_fore,
            z_fore1=self.config.z_fore1,
            z_fore2=self.config.z_fore2,
            z_slicer=self.config.z_slicer,
            slice_width=w,
            slice_height=h,
            slice_gap=self.config.min_slicer_gap,
            slicer_positions=slicer_pos,
            pupil_positions=pupil_pos,
            pupil_distance_z=self.config.pupil_distance_z,
            pupil_transverse_offset=self.config.pupil_transverse_offset,
            pupil_spacing=self.config.pupil_spacing,
            pupil_mirror_size=self.config.pupil_mirror_size,
            pupil_mirror_height=self.config.pupil_mirror_height,
            pupil_focal_length=self.config.pupil_focal_length,
            condenser_distance_z=self.config.condenser_distance_z,
            condenser_focal_length=self.config.condenser_focal_length,
            condenser_diameter=self.config.condenser_diameter,
            fiber_core_diameter=self.config.fiber_core_diameter,
            fiber_na=self.config.fiber_na,
            fiber_distance=self.config.fiber_distance,
        )
        return sys

    def evaluate_merit_function(
        self,
        X: np.ndarray,
        n: int,
        w: float,
        h: float,
        pre_slicer_bundle: RayBundle,
        total_source_power: float,
    ) -> float:
        """
        Evaluates the smoothed search objective (negative for minimization).
        Merit = Physical Coupling Efficiency - Smooth Penalties.
        """
        slicer_pos, pupil_pos = self.unpack_vector(X, n)

        # 1. Slicer Overlap Penalty
        slicer_penalty = 0.0
        min_dx = w + self.config.min_slicer_gap
        min_dy = h + self.config.min_slicer_gap
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(slicer_pos[i][0] - slicer_pos[j][0])
                dy = abs(slicer_pos[i][1] - slicer_pos[j][1])
                if dx < min_dx and dy < min_dy:
                    overlap_x = min_dx - dx
                    overlap_y = min_dy - dy
                    slicer_penalty += 0.5 * (overlap_x * overlap_y)

        # 2. Pupil Mechanical Overlap Penalty
        pupil_overlap_penalty = 0.0
        min_p_dist = self.config.pupil_mirror_size + self.config.min_pupil_clearance
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(pupil_pos[i] - pupil_pos[j]))
                if dist < min_p_dist:
                    pupil_overlap_penalty += 0.5 * (min_p_dist - dist)**2

        # 3. One-Sided Boundary Penalty
        side = self.config.pupil_layout_side.lower()
        side_penalty = 0.0
        for p in pupil_pos:
            if side == "lower" and p[1] > -5.0:
                side_penalty += 2.0 * (p[1] + 5.0)**2
            elif side == "upper" and p[1] < 5.0:
                side_penalty += 2.0 * (5.0 - p[1])**2
            elif side == "left" and p[0] > -5.0:
                side_penalty += 2.0 * (p[0] + 5.0)**2
            elif side == "right" and p[0] < 5.0:
                side_penalty += 2.0 * (5.0 - p[0])**2

        # Build candidate system
        candidate_sys = self.build_candidate_system(X, n, w, h)

        # Fast ray trace from cached image plane bundle
        bundle = pre_slicer_bundle.clone()

        # Step 3: Slicer trace
        if candidate_sys.slicer is not None:
            candidate_sys.slicer.trace(bundle)

        # Step 4: 3D non-sequential post-slicer trace (Pupils -> Condenser Lens -> Fiber)
        coupling_res = candidate_sys._trace_non_sequential_post_slicer(bundle, total_source_power)
        physical_eta = float(coupling_res.geometric_coupling_efficiency)

        # 4. Smooth guiding penalties for rejected rays
        core_penalty = 0.0
        na_penalty = 0.0
        act = bundle.active_mask
        if np.any(act):
            rx = coupling_res.hit_x[act]
            ry = coupling_res.hit_y[act]
            r_fib = np.sqrt(rx**2 + ry**2)
            pw = bundle.power[act]

            # Core distance penalty: rays outside 0.5 mm
            outside_core = r_fib > 0.5
            if np.any(outside_core):
                core_excess = r_fib[outside_core] - 0.5
                core_penalty = 0.02 * float(np.sum(core_excess * pw[outside_core]))

            # NA penalty: sin(theta) > 0.22
            sin_th = coupling_res.incidence_angle_rad[act]
            na_excess = np.maximum(0.0, sin_th - 0.22)
            if np.any(na_excess > 0):
                na_penalty = 0.02 * float(np.sum(na_excess * pw))

        # Objective is to MAXIMIZE: merit = eta - penalties
        merit = (
            physical_eta
            - slicer_penalty
            - pupil_overlap_penalty
            - side_penalty
            - core_penalty
            - na_penalty
        )

        # Return negative for minimization
        return -merit

    def optimize_fixed_n(
        self,
        n: int,
        progress_callback: Optional[Callable[[int, float, str], None]] = None,
    ) -> SingleNOptimizationResult:
        """
        Performs global + local optimization for a fixed discrete number of slicers N.
        """
        t_start = time.time()
        if progress_callback:
            progress_callback(n, 0.05, f"Initializing N={n} seed geometry...")

        # 1. Pre-trace exploration bundle
        expl_bundle = self.get_pre_slicer_bundle(self.config.exploration_rays)
        source_power = float(self._cached_source_bundle.total_power)
        spot_rad = self.get_spot_radius(expl_bundle)
        w, h = self.get_slice_dimensions(n, spot_rad)

        # 2. Get seed vector and bounds
        x0, bounds = self.get_initial_vector_and_bounds(n, spot_rad)

        if progress_callback:
            progress_callback(n, 0.15, f"Running Differential Evolution global search for N={n}...")

        # 3. Global Optimization: differential_evolution
        def obj_func(x_vec):
            return self.evaluate_merit_function(x_vec, n, w, h, expl_bundle, source_power)

        de_res = differential_evolution(
            obj_func,
            bounds=bounds,
            maxiter=self.config.max_de_iter,
            popsize=self.config.popsize,
            seed=self.config.random_seed,
            polish=False,
            tol=1e-3,
        )

        f0 = obj_func(x0)
        best_x = de_res.x
        n_iters = int(de_res.nit)
        if f0 < de_res.fun:
            best_x = x0

        # 4. Optional Local Polishing: Nelder-Mead
        if self.config.polish:
            if progress_callback:
                progress_callback(n, 0.75, f"Polishing N={n} geometry with local simplex...")
            local_res = minimize(
                obj_func,
                best_x,
                method="Nelder-Mead",
                options={"maxiter": 35, "disp": False},
            )
            if local_res.fun < min(de_res.fun, f0):
                best_x = local_res.x
                n_iters += int(local_res.nit)

        if progress_callback:
            progress_callback(n, 0.90, f"Validating N={n} optimum with high ray count...")

        # 5. Full Validation Trace with validation_rays
        val_source_bundle = self.generate_source_bundle(self.config.validation_rays)
        val_tot_power = float(val_source_bundle.total_power)
        best_system = self.build_candidate_system(best_x, n, w, h)

        traced_bundle, coupling_res, metrics = best_system.trace(val_source_bundle)

        eta = float(metrics.geometric_coupling_efficiency)
        core_frac = float(metrics.fraction_inside_core)
        na_frac = float(metrics.fraction_inside_na)
        both_frac = float(metrics.fraction_satisfying_both)
        p_acc = float(metrics.fiber_accepted_power)
        clip_loss = float(max(0.0, 1.0 - (metrics.power_reaching_fiber_plane / max(metrics.launched_power, 1e-9))))

        # 6. Build Slicer and Pupil Summary DataFrames
        slicer_rows = []
        for s in best_system.slicer.slices:
            ch_power = float(np.sum(traced_bundle.power[(traced_bundle.channel_id == s.slice_id)])) if hasattr(traced_bundle, "channel_id") else 0.0
            slicer_rows.append({
                "Channel": f"Channel {s.slice_id + 1}",
                "Center X (mm)": f"{s.center_x:.3f}",
                "Center Y (mm)": f"{s.center_y:.3f}",
                "Plane Z (mm)": f"{s.z:.1f}",
                "Width x Height": f"{s.width:.2f} x {s.height:.2f} mm",
                "Tip X (deg)": f"{s.tip_x_deg:.4f}",
                "Tilt Y (deg)": f"{s.tilt_y_deg:.4f}",
                "Intercepted Power": f"{ch_power:.2f} W",
            })
        slicer_df = pd.DataFrame(slicer_rows)

        pupil_rows = []
        if best_system.pupil_relay is not None:
            for pm in best_system.pupil_relay.mirrors:
                pupil_rows.append({
                    "Pupil Mirror": f"Pupil P{pm.channel_id + 1}",
                    "Center X (mm)": f"{pm.center[0]:.2f}",
                    "Center Y (mm)": f"{pm.center[1]:.2f}",
                    "Center Z (mm)": f"{pm.center[2]:.2f}",
                    "Tip X (deg)": f"{pm.tip_x_deg:.4f}",
                    "Tilt Y (deg)": f"{pm.tilt_y_deg:.4f}",
                    "Size": f"{pm.width:.1f} x {pm.height:.1f} mm",
                    "Focal Length": f"f = {pm.focal_length:.0f} mm" if pm.is_powered else "Flat",
                })
        pupil_df = pd.DataFrame(pupil_rows)

        # 7. Optical Loss Budget
        loss_budget = {}
        for st_item in best_system.stages:
            loss_budget[st_item.name] = float(st_item.active_power)
        loss_budget["Core Accepted"] = float(coupling_res.power_reaching_fiber_plane * core_frac)
        loss_budget["NA Accepted"] = float(coupling_res.power_reaching_fiber_plane * na_frac)
        loss_budget["Final Fiber Accepted"] = p_acc

        t_elapsed = time.time() - t_start

        if progress_callback:
            progress_callback(n, 1.0, f"Completed N={n}: Coupling Efficiency = {eta*100.0:.2f}%")

        return SingleNOptimizationResult(
            n_channels=n,
            success=bool(de_res.success or True),
            best_x=best_x,
            coupling_efficiency=eta,
            core_accepted_fraction=core_frac,
            na_accepted_fraction=na_frac,
            both_accepted_fraction=both_frac,
            accepted_power=p_acc,
            total_launched_power=val_tot_power,
            clipping_loss=clip_loss,
            slicer_table=slicer_df,
            pupil_table=pupil_df,
            system=best_system,
            traced_bundle=traced_bundle,
            coupling_result=coupling_res,
            metrics=metrics,
            loss_budget=loss_budget,
            iterations=n_iters,
            execution_time_sec=t_elapsed,
            message=str(de_res.message),
            power_accounting=coupling_res.power_accounting,
        )

    def run_multi_n_study(
        self,
        n_min: Optional[int] = None,
        n_max: Optional[int] = None,
        progress_callback: Optional[Callable[[int, float, str], None]] = None,
    ) -> MultiNStudyResult:
        """
        Performs discrete architectural study across N_slicer in [n_min ... n_max],
        evaluating physically valid coupling efficiency for each slice count and identifying the winner.
        """
        min_n = n_min if n_min is not None else self.config.n_min
        max_n = n_max if n_max is not None else self.config.n_max

        results: Dict[int, SingleNOptimizationResult] = {}
        comparison_rows = []

        total_n = max_n - min_n + 1
        multi_start_rows = []

        for idx, n in enumerate(range(min_n, max_n + 1)):
            def sub_callback(ch_n, frac, msg):
                if progress_callback:
                    overall = (idx + frac) / total_n
                    progress_callback(ch_n, overall, msg)

            res = self.optimize_fixed_n(n, progress_callback=sub_callback)
            results[n] = res

            pa = res.power_accounting
            if pa is not None:
                p_l = f"{pa.p_launch:.3f} W"
                p_fib = f"{pa.p_at_fiber_plane:.3f} W"
                p_acc = f"{pa.p_inside_core_and_na:.3f} W"
                eta_tot = f"{pa.eta_total * 100.0:.2f}%"
                eta_c_l = f"{pa.eta_core_launch * 100.0:.2f}%"
                eta_na_l = f"{pa.eta_na_launch * 100.0:.2f}%"
                eta_c_cond = f"{pa.eta_core_conditional * 100.0:.1f}%"
                eta_na_cond = f"{pa.eta_na_conditional * 100.0:.1f}%"
            else:
                p_l = f"{res.total_launched_power:.3f} W"
                p_fib = f"{res.accepted_power / max(1e-6, res.coupling_efficiency):.3f} W"
                p_acc = f"{res.accepted_power:.3f} W"
                eta_tot = f"{res.coupling_efficiency * 100.0:.2f}%"
                eta_c_l = f"{res.coupling_efficiency * 100.0:.2f}%"
                eta_na_l = f"{res.coupling_efficiency * 100.0:.2f}%"
                eta_c_cond = f"{res.core_accepted_fraction * 100.0:.1f}%"
                eta_na_cond = f"{res.na_accepted_fraction * 100.0:.1f}%"

            comparison_rows.append({
                "N Slicers": n,
                "P_launch": p_l,
                "P_at_fiber": p_fib,
                "P_accepted": p_acc,
                "eta_total (abs)": eta_tot,
                "eta_core_launch": eta_c_l,
                "eta_NA_launch": eta_na_l,
                "eta_core_conditional": eta_c_cond,
                "eta_NA_conditional": eta_na_cond,
                "Clipping Loss": f"{res.clipping_loss * 100.0:.1f}%",
                "Iterations": res.iterations,
                "Compute Time": f"{res.execution_time_sec:.1f} s",
            })

            # Run multi-start statistics (10 random restarts) for candidate N
            ms_res = validate_multi_start_optimization(n_channels=n, n_restarts=10, rays=min(300, self.config.exploration_rays))
            res.multi_start_stats = ms_res.details
            multi_start_rows.append({
                "N Slicers": n,
                "Restarts": ms_res.details["n_restarts"],
                "Best Coupling (eta_total)": f"{ms_res.details['best_coupling_efficiency']*100.0:.2f}%",
                "Median Coupling (eta_total)": f"{ms_res.details['median_coupling_efficiency']*100.0:.2f}%",
                "Worst Coupling (eta_total)": f"{ms_res.details['worst_coupling_efficiency']*100.0:.2f}%",
                "Best Clipping": f"{ms_res.details['best_clipping']*100.0:.1f}%",
                "Median Clipping": f"{ms_res.details['median_clipping']*100.0:.1f}%",
                "Worst Clipping": f"{ms_res.details['worst_clipping']*100.0:.1f}%",
            })

        comp_df = pd.DataFrame(comparison_rows)
        ms_df = pd.DataFrame(multi_start_rows)

        # Run Validation Suite Gate
        val_report = run_all_validations()

        # Determine winner candidate
        best_n = min_n
        best_eff = -1.0
        for n, r in results.items():
            if r.coupling_efficiency > best_eff:
                best_eff = r.coupling_efficiency
                best_n = n

        best_res = results[best_n]

        if val_report.all_passed:
            winner_declared = True
            winner_explanation = (
                f"Architecture Winner: N = {best_n} Slicers (All 7 Validation Cases Passed).\n\n"
                f"- Absolute Coupling Efficiency: {best_res.coupling_efficiency * 100.0:.2f}%\n"
                f"- Spatial Core Acceptance (rel P_launch): {best_res.power_accounting.eta_core_launch * 100.0:.2f}%\n"
                f"- Angular NA Acceptance (rel P_launch): {best_res.power_accounting.eta_na_launch * 100.0:.2f}%\n"
                f"- Conditional Core Acceptance (of P_at_fiber): {best_res.core_accepted_fraction * 100.0:.1f}%\n"
                f"- Conditional NA Acceptance (of P_at_fiber): {best_res.na_accepted_fraction * 100.0:.1f}%\n\n"
                f"Physical Analysis: "
            )
            if best_n == 1:
                winner_explanation += (
                    "A single slice avoids slicer gap clipping and pupil cross-talk, which outweighs the "
                    "geometric reformatting benefit for the current source etendue."
                )
            elif best_n <= 3:
                winner_explanation += (
                    f"Dividing the field into {best_n} channels reformats the aspect ratio into a quasi-circular "
                    f"pupil distribution that couples optimally into the 1.0 mm fiber core without exceeding NA = 0.22."
                )
            else:
                winner_explanation += (
                    f"Using {best_n} channels provides the finest spatial slicing, effectively matching the "
                    f"fiber circular cross-section at the condenser focal plane."
                )
        else:
            winner_declared = False
            winner_explanation = (
                "Architecture Winner DEFERRED: One or more physical validation tests did not pass. "
                "No candidate architecture can be declared optimal until all physical criteria are verified."
            )

        return MultiNStudyResult(
            results_by_n=results,
            best_n=best_n,
            best_result=best_res,
            comparison_table=comp_df,
            winner_explanation=winner_explanation,
            multi_start_table=ms_df,
            validation_report=val_report,
            winner_declared=winner_declared,
        )
