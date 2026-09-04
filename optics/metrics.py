"""
Optical metrics, spot analysis, loss budget accounting, and étendue sanity checking.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .ray import RayBundle, RayStatus


@dataclass
class SpotMetrics:
    """Spot / beam footprint geometric metrics at a specified plane."""
    z: float
    centroid_x: float
    centroid_y: float
    rms_radius: float
    encircled_80_radius: float
    encircled_80_diameter: float
    diameter_90: float
    bbox_width: float
    bbox_height: float
    total_power: float
    n_active_rays: int


def compute_spot_metrics(
    bundle: RayBundle,
    z_plane: Optional[float] = None,
    channel_id: Optional[int] = None,
) -> SpotMetrics:
    """
    Compute geometric spot metrics at z_plane (propagating virtually if provided).
    Optionally filter by channel_id.
    """
    mask = bundle.active_mask
    if channel_id is not None:
        mask = mask & (bundle.channel_id == channel_id)

    indices = np.where(mask)[0]
    if len(indices) == 0:
        target_z = z_plane if z_plane is not None else 0.0
        return SpotMetrics(
            z=target_z,
            centroid_x=0.0,
            centroid_y=0.0,
            rms_radius=0.0,
            encircled_80_radius=0.0,
            encircled_80_diameter=0.0,
            diameter_90=0.0,
            bbox_width=0.0,
            bbox_height=0.0,
            total_power=0.0,
            n_active_rays=0,
        )

    r_sub = bundle.r[indices].copy()
    k_sub = bundle.k[indices]
    p_sub = bundle.power[indices]
    tot_p = float(np.sum(p_sub))

    if z_plane is not None:
        dz = z_plane - r_sub[:, 2]
        safe_kz = np.where(np.abs(k_sub[:, 2]) < 1e-12, 1e-12, k_sub[:, 2])
        t = dz / safe_kz
        x = r_sub[:, 0] + k_sub[:, 0] * t
        y = r_sub[:, 1] + k_sub[:, 1] * t
        target_z = float(z_plane)
    else:
        x = r_sub[:, 0]
        y = r_sub[:, 1]
        target_z = float(r_sub[0, 2])

    cx = float(np.average(x, weights=p_sub)) if tot_p > 0 else float(np.mean(x))
    cy = float(np.average(y, weights=p_sub)) if tot_p > 0 else float(np.mean(y))

    dx = x - cx
    dy = y - cy
    dist = np.sqrt(dx**2 + dy**2)

    var_r = float(np.average(dist**2, weights=p_sub)) if tot_p > 0 else float(np.mean(dist**2))
    rms = np.sqrt(max(var_r, 0.0))

    # Encircled energy radii
    sort_idx = np.argsort(dist)
    cum_p = np.cumsum(p_sub[sort_idx])
    
    idx_80 = np.searchsorted(cum_p, 0.8 * tot_p) if tot_p > 0 else int(0.8 * len(dist))
    r_80 = float(dist[sort_idx[min(idx_80, len(dist) - 1)]])

    idx_90 = np.searchsorted(cum_p, 0.9 * tot_p) if tot_p > 0 else int(0.9 * len(dist))
    r_90 = float(dist[sort_idx[min(idx_90, len(dist) - 1)]])

    bbox_w = float(np.ptp(x)) if len(x) > 0 else 0.0
    bbox_h = float(np.ptp(y)) if len(y) > 0 else 0.0

    return SpotMetrics(
        z=target_z,
        centroid_x=cx,
        centroid_y=cy,
        rms_radius=float(rms),
        encircled_80_radius=r_80,
        encircled_80_diameter=2.0 * r_80,
        diameter_90=2.0 * r_90,
        bbox_width=bbox_w,
        bbox_height=bbox_h,
        total_power=tot_p,
        n_active_rays=len(indices),
    )


def scan_beam_size_vs_z(
    bundle: RayBundle,
    z_min: float,
    z_max: float,
    n_points: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Scan beam geometric properties vs z to identify input focal plane,
    beam waists, and slicer placement locations.
    """
    z_vals = np.linspace(z_min, z_max, n_points)
    rms_arr = np.zeros(n_points)
    r80_arr = np.zeros(n_points)
    d90_arr = np.zeros(n_points)
    bbox_w_arr = np.zeros(n_points)
    bbox_h_arr = np.zeros(n_points)

    for i, z in enumerate(z_vals):
        m = compute_spot_metrics(bundle, z_plane=z)
        rms_arr[i] = m.rms_radius
        r80_arr[i] = m.encircled_80_radius
        d90_arr[i] = m.diameter_90
        bbox_w_arr[i] = m.bbox_width
        bbox_h_arr[i] = m.bbox_height

    return {
        "z": z_vals,
        "rms_radius": rms_arr,
        "encircled_80_radius": r80_arr,
        "diameter_90": d90_arr,
        "bbox_width": bbox_w_arr,
        "bbox_height": bbox_h_arr,
    }


def find_z_for_target_footprint(
    bundle: RayBundle,
    target_width: float,
    z_min: float,
    z_max: float,
    n_points: int = 100,
) -> float:
    """Find the z plane closest to producing the desired beam footprint width."""
    scan = scan_beam_size_vs_z(bundle, z_min, z_max, n_points)
    diffs = np.abs(scan["bbox_width"] - target_width)
    best_idx = int(np.argmin(diffs))
    return float(scan["z"][best_idx])


@dataclass
class EtendueCheck:
    """First-order optical étendue sanity check."""
    fiber_etendue_mm2_sr: float
    source_etendue_mm2_sr: float
    etendue_ratio: float           # G_fiber / G_source
    max_theoretical_coupling: float
    is_thermodynamically_valid: bool
    warning_message: str


def compute_etendue_check(
    source_type: str,
    aperture_diameter: float,
    source_diameter: float = 1.0,
    source_distance: float = 50.0,
    solar_angular_radius_deg: float = 0.266,
    fiber_core_diameter: float = 1.0,
    fiber_na: float = 0.22,
) -> EtendueCheck:
    """
    First-order étendue estimate:
    G_fiber = A_fiber * pi * NA^2 = pi^2 * (D_core / 2)^2 * NA^2
    G_sun = A_aperture * Omega_sun = pi * (D_ap / 2)^2 * (pi * sin^2(alpha_sun))
    G_led = A_led * Omega_aperture
    """
    # Fiber étendue
    r_core = fiber_core_diameter / 2.0
    a_fiber = np.pi * (r_core**2)
    omega_fiber = np.pi * (fiber_na**2)
    g_fiber = a_fiber * omega_fiber

    # Source étendue
    r_ap = aperture_diameter / 2.0
    a_ap = np.pi * (r_ap**2)

    if source_type.upper() == "SUN":
        alpha_rad = np.radians(solar_angular_radius_deg)
        omega_source = np.pi * (np.sin(alpha_rad)**2)
        g_source = a_ap * omega_source
    else:  # LED
        r_led = max(source_diameter / 2.0, 0.05)
        a_led = np.pi * (r_led**2)
        # Numerical aperture of fore-optics aperture viewed from LED
        theta_ap = np.arctan2(r_ap, max(source_distance, 1.0))
        omega_source = np.pi * (np.sin(theta_ap)**2)
        g_source = a_led * omega_source

    ratio = g_fiber / g_source if g_source > 0 else 1.0
    max_coupling = min(1.0, ratio)

    if ratio < 1.0:
        warning = (
            f"Thermodynamic Constraint: Fiber étendue ({g_fiber:.4f} mm²·sr) is smaller than "
            f"source étendue ({g_source:.4f} mm²·sr). Maximum passive coupling is theoretically "
            f"limited to ~{max_coupling * 100.0:.1f}%. The image slicer reshapes phase-space geometry "
            f"to better match fiber aspect ratio/acceptance, but CANNOT defeat étendue conservation."
        )
        valid = True
    else:
        warning = (
            f"Étendue Compatible: Fiber étendue ({g_fiber:.4f} mm²·sr) exceeds or matches "
            f"source étendue ({g_source:.4f} mm²·sr). Up to 100% coupling is theoretically allowed."
        )
        valid = True

    return EtendueCheck(
        fiber_etendue_mm2_sr=float(g_fiber),
        source_etendue_mm2_sr=float(g_source),
        etendue_ratio=float(ratio),
        max_theoretical_coupling=float(max_coupling),
        is_thermodynamically_valid=valid,
        warning_message=warning,
    )


@dataclass
class SystemMetrics:
    """Comprehensive performance metrics of the entire optical chain."""
    launched_rays: int
    launched_power: float
    aperture_transmitted_power: float
    slicer_intercepted_power: float
    slicer_gap_loss_power: float
    power_per_slice: Dict[int, float]
    power_reaching_coupling_optic: float
    power_reaching_fiber_plane: float
    fiber_accepted_power: float
    fiber_rejected_position_power: float
    fiber_rejected_na_power: float
    fiber_rejected_both_power: float
    
    geometric_coupling_efficiency: float
    efficiency_rel_coupling_optic: float
    efficiency_rel_fiber_plane: float
    
    fraction_inside_core: float
    fraction_inside_na: float
    fraction_satisfying_both: float
    
    spot_rms_radius: float
    encircled_80_diameter: float
    mean_ray_angle_deg: float
    max_ray_angle_deg: float
    
    per_channel_stats: Dict[int, Dict[str, float]]
    loss_budget_table: List[Dict[str, Any]]
