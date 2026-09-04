"""
Automatic alignment, SciPy optimization, and parameter sweep routines.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize
from .ray import RayBundle
from .slicer import SlicerArray
from .pupil import PupilRelaySystem
from .fiber import Fiber
from .system import OpticalSystem


def aim_slicer_mirror(
    slicer: SlicerArray,
    slice_id: int,
    target_pos: Tuple[float, float, float],
    incoming_k: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Tuple[float, float]:
    """
    Calculate and apply the exact tip_x and tilt_y for slice_id so its chief ray
    reflects directly towards target_pos (x, y, z) using the reflection bisector.
    
    Returns:
        (tip_x_deg, tilt_y_deg)
    """
    # Locate slice
    target_slice = None
    for s in slicer.slices:
        if s.slice_id == slice_id:
            target_slice = s
            break

    if target_slice is None:
        raise ValueError(f"Slice ID {slice_id} not found in slicer array")

    slicer_center = np.array([target_slice.center_x, target_slice.center_y, target_slice.z], dtype=np.float64)
    target = np.array(target_pos, dtype=np.float64)

    # Outgoing desired direction vector k_out
    d = target - slicer_center
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-6:
        raise ValueError("Target position coincides with slice position")
    k_out = d / norm_d

    # Incoming direction vector k_in
    k_in = np.array(incoming_k, dtype=np.float64)
    k_in = k_in / np.linalg.norm(k_in)

    # Reflection law: k_out = k_in - 2 * (k_in . n) * n
    # Bisector normal n pointing against incidence:
    # n = (k_in - k_out) / ||k_in - k_out||
    diff = k_in - k_out
    norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-12:
        # Straight transmission / no reflection needed
        n = np.array([0.0, 0.0, -1.0])
    else:
        n = diff / norm_diff

    # Ensure normal points towards incoming beam (-z direction)
    if n[2] > 0:
        n = -n

    # Nominal un-tilted mirror has normal n0 = [0, 0, -1]
    # R = Ry(tilt_y) * Rx(tip_x)
    # n = R @ [0, 0, -1] = [-sin(tilt_y)*cos(tip_x), sin(tip_x), -cos(tilt_y)*cos(tip_x)]
    # Therefore:
    # n_y = sin(tip_x) => tip_x = arcsin(n_y)
    # n_x = -sin(tilt_y)*cos(tip_x) => tilt_y = arcsin(-n_x / cos(tip_x))
    ny = np.clip(n[1], -1.0, 1.0)
    tip_x_rad = np.arcsin(ny)
    cos_tip = np.cos(tip_x_rad)

    if np.abs(cos_tip) > 1e-6:
        arg_tilt = np.clip(-n[0] / cos_tip, -1.0, 1.0)
        tilt_y_rad = np.arcsin(arg_tilt)
    else:
        tilt_y_rad = 0.0

    tip_x_deg = float(np.degrees(tip_x_rad))
    tilt_y_deg = float(np.degrees(tilt_y_rad))

    target_slice.tip_x_deg = tip_x_deg
    target_slice.tilt_y_deg = tilt_y_deg

    return tip_x_deg, tilt_y_deg


def optimize_slicer_tilts(
    system: OpticalSystem,
    initial_bundle: RayBundle,
    maxiter: int = 50,
) -> Dict[str, Any]:
    """
    Numerically optimize tip and tilt of all enabled slicers to maximize fiber coupled power.
    Uses SciPy minimize (Nelder-Mead).
    """
    if system.slicer is None:
        return {"success": False, "message": "No slicer present in system"}

    enabled_slices = [s for s in system.slicer.slices if s.enabled]
    if not enabled_slices:
        return {"success": False, "message": "No slicer mirrors are enabled"}

    # Initial parameter vector: [tip_0, tilt_0, tip_1, tilt_1, ...]
    x0 = []
    for s in enabled_slices:
        x0.extend([s.tip_x_deg, s.tilt_y_deg])
    x0 = np.array(x0, dtype=np.float64)

    def objective(params: np.ndarray) -> float:
        for idx, s in enumerate(enabled_slices):
            s.tip_x_deg = params[2 * idx]
            s.tilt_y_deg = params[2 * idx + 1]

        # Trace
        bundle, res, metrics = system.trace(initial_bundle)
        # We want to MAXIMIZE accepted power, while softly penalizing beam divergence outside core
        p_acc = res.accepted_power
        # Soft penalty on spot size and ray angle
        penalty = 0.001 * res.spot_rms_radius + 0.0005 * res.mean_ray_angle_deg
        return -(p_acc - penalty)

    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "disp": False},
    )

    # Set optimal values
    for idx, s in enumerate(enabled_slices):
        s.tip_x_deg = float(res.x[2 * idx])
        s.tilt_y_deg = float(res.x[2 * idx + 1])

    # Re-trace with optimal values
    final_bundle, final_res, final_metrics = system.trace(initial_bundle)

    return {
        "success": bool(res.success),
        "iterations": int(res.nit),
        "optimized_efficiency": float(final_res.geometric_coupling_efficiency),
        "optimized_power": float(final_res.accepted_power),
        "final_res": final_res,
    }


def optimize_fiber_position(
    system: OpticalSystem,
    initial_bundle: RayBundle,
    maxiter: int = 40,
) -> Dict[str, Any]:
    """
    Optimize fiber position (x, y, z) to maximize coupled power.
    """
    fib = system.fiber
    x0 = np.array([fib.position[0], fib.position[1], fib.position[2]], dtype=np.float64)

    def objective(pos: np.ndarray) -> float:
        fib.position[0] = pos[0]
        fib.position[1] = pos[1]
        fib.position[2] = pos[2]

        bundle, res, metrics = system.trace(initial_bundle)
        penalty = 0.001 * res.spot_rms_radius
        return -(res.accepted_power - penalty)

    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "disp": False},
    )

    fib.position[0] = float(res.x[0])
    fib.position[1] = float(res.x[1])
    fib.position[2] = float(res.x[2])

    bundle, final_res, final_metrics = system.trace(initial_bundle)

    return {
        "success": bool(res.success),
        "optimal_x": float(fib.position[0]),
        "optimal_y": float(fib.position[1]),
        "optimal_z": float(fib.position[2]),
        "optimized_efficiency": float(final_res.geometric_coupling_efficiency),
        "final_res": final_res,
    }


def run_1d_sweep(
    system: OpticalSystem,
    initial_bundle: RayBundle,
    param_name: str,
    values: np.ndarray,
    setter_func: Callable[[OpticalSystem, float], None],
) -> Dict[str, np.ndarray]:
    """
    Run 1D parameter sweep and return coupling efficiency and spot RMS vs parameter values.
    """
    n_pts = len(values)
    eff_arr = np.zeros(n_pts)
    power_arr = np.zeros(n_pts)
    rms_arr = np.zeros(n_pts)
    mean_angle_arr = np.zeros(n_pts)

    for i, val in enumerate(values):
        setter_func(system, val)
        _, res, _ = system.trace(initial_bundle)
        eff_arr[i] = res.geometric_coupling_efficiency
        power_arr[i] = res.accepted_power
        rms_arr[i] = res.spot_rms_radius
        mean_angle_arr[i] = res.mean_ray_angle_deg

    return {
        "param_name": param_name,
        "values": values,
        "efficiency": eff_arr,
        "accepted_power": power_arr,
        "spot_rms": rms_arr,
        "mean_angle": mean_angle_arr,
    }


def run_2d_sweep(
    system: OpticalSystem,
    initial_bundle: RayBundle,
    param1_values: np.ndarray,
    param2_values: np.ndarray,
    setter_func: Callable[[OpticalSystem, float, float], None],
) -> np.ndarray:
    """
    Run 2D parameter sweep and return 2D grid of coupling efficiency.
    Shape: (len(param1_values), len(param2_values))
    """
    grid = np.zeros((len(param1_values), len(param2_values)))
    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            setter_func(system, v1, v2)
            _, res, _ = system.trace(initial_bundle)
            grid[i, j] = res.geometric_coupling_efficiency
    return grid
