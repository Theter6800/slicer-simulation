"""
Optical metrics, spot analysis, loss budget accounting, phase-space classification, and étendue checking.
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
    encircled_50_radius: float
    encircled_80_radius: float
    encircled_90_radius: float
    encircled_95_radius: float
    diameter_50: float
    diameter_80: float
    diameter_90: float
    diameter_95: float
    bbox_width: float
    bbox_height: float
    peak_x: float
    peak_y: float
    total_power: float
    n_active_rays: int

    # Backwards compatibility properties
    @property
    def encircled_80_diameter(self) -> float:
        return self.diameter_80


@dataclass
class AngularMetrics:
    """Ray angular distribution metrics at the fiber face."""
    theta_rms_deg: float
    theta_50_deg: float
    theta_80_deg: float
    theta_90_deg: float
    theta_max_deg: float
    mean_theta_deg: float
    na_max_numerical: float


def compute_spot_metrics(
    bundle: RayBundle,
    z_plane: Optional[float] = None,
    channel_id: Optional[int] = None,
) -> SpotMetrics:
    """
    Compute geometric spot metrics at z_plane (propagating virtually if provided).
    Optionally filter by channel_id.
    Includes D50, D80, D90, D95, centroid, RMS radius, bbox dimensions, and peak irradiance location.
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

    def get_encircled_radius(fraction: float) -> float:
        if tot_p <= 0 or len(dist) == 0:
            return 0.0
        idx = np.searchsorted(cum_p, fraction * tot_p)
        return float(dist[sort_idx[min(idx, len(dist) - 1)]])

    r_50 = get_encircled_radius(0.50)
    r_80 = get_encircled_radius(0.80)
    r_90 = get_encircled_radius(0.90)
    r_95 = get_encircled_radius(0.95)

    bbox_w = float(np.ptp(x)) if len(x) > 0 else 0.0
    bbox_h = float(np.ptp(y)) if len(y) > 0 else 0.0

    # Peak irradiance location using 2D histogram
    if len(x) >= 4 and bbox_w > 1e-6 and bbox_h > 1e-6:
        n_bins = min(40, max(10, int(np.sqrt(len(x)))))
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins, weights=p_sub)
        max_idx = np.unravel_index(np.argmax(hist), hist.shape)
        peak_x = float(0.5 * (x_edges[max_idx[0]] + x_edges[max_idx[0] + 1]))
        peak_y = float(0.5 * (y_edges[max_idx[1]] + y_edges[max_idx[1] + 1]))
    else:
        peak_x = cx
        peak_y = cy

    return SpotMetrics(
        z=target_z,
        centroid_x=cx,
        centroid_y=cy,
        rms_radius=float(rms),
        encircled_50_radius=r_50,
        encircled_80_radius=r_80,
        encircled_90_radius=r_90,
        encircled_95_radius=r_95,
        diameter_50=2.0 * r_50,
        diameter_80=2.0 * r_80,
        diameter_90=2.0 * r_90,
        diameter_95=2.0 * r_95,
        bbox_width=bbox_w,
        bbox_height=bbox_h,
        peak_x=peak_x,
        peak_y=peak_y,
        total_power=tot_p,
        n_active_rays=len(indices),
    )


def compute_angular_metrics(
    incidence_angles_rad: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> AngularMetrics:
    """
    Computes angular distribution metrics at the fiber face (theta in degrees and NA).
    """
    if len(incidence_angles_rad) == 0:
        return AngularMetrics(
            theta_rms_deg=0.0,
            theta_50_deg=0.0,
            theta_80_deg=0.0,
            theta_90_deg=0.0,
            theta_max_deg=0.0,
            mean_theta_deg=0.0,
            na_max_numerical=0.0,
        )

    theta_deg = np.degrees(incidence_angles_rad)
    w = weights if weights is not None else np.ones_like(theta_deg)
    tot_w = float(np.sum(w))

    if tot_w <= 0:
        tot_w = float(len(theta_deg))
        w = np.ones_like(theta_deg)

    mean_th = float(np.average(theta_deg, weights=w))
    rms_th = float(np.sqrt(max(0.0, np.average(theta_deg**2, weights=w))))
    max_th = float(np.max(theta_deg))

    sort_idx = np.argsort(theta_deg)
    cum_w = np.cumsum(w[sort_idx])

    def get_percentile_angle(frac: float) -> float:
        idx = np.searchsorted(cum_w, frac * tot_w)
        return float(theta_deg[sort_idx[min(idx, len(theta_deg) - 1)]])

    th_50 = get_percentile_angle(0.50)
    th_80 = get_percentile_angle(0.80)
    th_90 = get_percentile_angle(0.90)

    return AngularMetrics(
        theta_rms_deg=rms_th,
        theta_50_deg=th_50,
        theta_80_deg=th_80,
        theta_90_deg=th_90,
        theta_max_deg=max_th,
        mean_theta_deg=mean_th,
        na_max_numerical=float(np.sin(np.radians(max_th))),
    )


def classify_fiber_phase_space(
    r_fiber_mm: np.ndarray,
    theta_fiber_deg: np.ndarray,
    core_radius_max_mm: float = 0.5,
    theta_na_max_deg: float = 12.71,
) -> Dict[str, Any]:
    """
    Classifies rays reaching the fiber plane into 4 phase-space categories:
    - GREEN:  r <= 0.5 mm AND theta <= 12.71 deg (Both pass: fully coupled)
    - ORANGE: r <= 0.5 mm AND theta > 12.71 deg  (Core accepted, NA rejected)
    - BLUE:   r > 0.5 mm  AND theta <= 12.71 deg (Core rejected, NA accepted)
    - RED:    r > 0.5 mm  AND theta > 12.71 deg  (Both rejected)
    """
    core_pass = r_fiber_mm <= core_radius_max_mm
    na_pass = theta_fiber_deg <= theta_na_max_deg

    green_mask = core_pass & na_pass
    orange_mask = core_pass & (~na_pass)
    blue_mask = (~core_pass) & na_pass
    red_mask = (~core_pass) & (~na_pass)

    return {
        "green_mask": green_mask,
        "orange_mask": orange_mask,
        "blue_mask": blue_mask,
        "red_mask": red_mask,
        "n_green": int(np.sum(green_mask)),
        "n_orange": int(np.sum(orange_mask)),
        "n_blue": int(np.sum(blue_mask)),
        "n_red": int(np.sum(red_mask)),
        "total_rays": len(r_fiber_mm),
    }


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
    r_core = fiber_core_diameter / 2.0
    a_fiber = np.pi * (r_core**2)
    omega_fiber = np.pi * (fiber_na**2)
    g_fiber = a_fiber * omega_fiber

    r_ap = aperture_diameter / 2.0
    a_ap = np.pi * (r_ap**2)

    if source_type.upper() == "SUN":
        alpha_rad = np.radians(solar_angular_radius_deg)
        omega_source = np.pi * (np.sin(alpha_rad)**2)
        g_source = a_ap * omega_source
    else:  # LED
        r_led = max(source_diameter / 2.0, 0.05)
        a_led = np.pi * (r_led**2)
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

    # Enhanced diagnostics fields
    pre_slicer_spot_metrics: Optional[SpotMetrics] = None
    fiber_angular_metrics: Optional[AngularMetrics] = None


@dataclass
class PhaseSpaceReformattingScore:
    """Phase-space reformatting assessment comparing architecture N to baseline N=0."""
    n_channels: int
    delta_r90_mm: float
    delta_theta90_deg: float
    delta_eta_both_conditional: float
    spatial_compressed: bool
    angular_expanded: bool
    joint_improved: bool
    interpretation: str
    details: Dict[str, Any]

    @property
    def r_90(self) -> float:
        return float(self.details.get("R90_N", 0.0))

    @property
    def r_rms(self) -> float:
        return float(self.details.get("RMS_r_N", self.details.get("R90_N", 0.0)))

    @property
    def theta_90_deg(self) -> float:
        return float(self.details.get("theta90_N", 0.0))

    @property
    def theta_rms_deg(self) -> float:
        return float(self.details.get("RMS_theta_N", self.details.get("theta90_N", 0.0)))

    @property
    def delta_r_90_vs_n0(self) -> float:
        return self.delta_r90_mm

    @property
    def delta_theta_90_deg_vs_n0(self) -> float:
        return self.delta_theta90_deg

    @property
    def delta_eta_both_cond_vs_n0(self) -> float:
        return self.delta_eta_both_conditional

    @property
    def diagnosis(self) -> str:
        return self.interpretation

    def __getattr__(self, name: str) -> Any:
        if name in self.details:
            return self.details[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def compute_phase_space_reformatting_score(
    spot_n: SpotMetrics,
    spot_0: SpotMetrics,
    ang_n: AngularMetrics,
    ang_0: AngularMetrics,
    eta_both_cond_n: float,
    eta_both_cond_0: float,
    n_channels: int = 1,
) -> PhaseSpaceReformattingScore:
    """
    Computes phase-space reformatting metrics comparing N to N=0:
    - Delta_R90 = R90(N) - R90(0)
    - Delta_theta90 = theta90(N) - theta90(0)
    - Delta_eta_both_conditional = eta_both_conditional(N) - eta_both_conditional(0)
    Generates dynamic physical interpretation without hardcoding.
    """
    delta_r90 = float(spot_n.encircled_90_radius - spot_0.encircled_90_radius)
    delta_th90 = float(ang_n.theta_90_deg - ang_0.theta_90_deg)
    delta_eta = float(eta_both_cond_n - eta_both_cond_0)

    spat_comp = delta_r90 < -0.01  # shrunk by at least 10 um
    ang_exp = delta_th90 > 0.1     # widened by at least 0.1 deg
    joint_imp = delta_eta > 0.005  # improved conditional acceptance by > 0.5%

    if spat_comp and ang_exp:
        interp = "Spatial compression obtained at cost of angular expansion."
    elif joint_imp:
        interp = "Net positive phase-space reformatting achieved."
    elif (not spat_comp) and ang_exp:
        interp = "Angular broadening without spatial compression."
    else:
        interp = "No useful phase-space reformatting demonstrated."

    return PhaseSpaceReformattingScore(
        n_channels=n_channels,
        delta_r90_mm=delta_r90,
        delta_theta90_deg=delta_th90,
        delta_eta_both_conditional=delta_eta,
        spatial_compressed=spat_comp,
        angular_expanded=ang_exp,
        joint_improved=joint_imp,
        interpretation=interp,
        details={
            "R90_N": spot_n.encircled_90_radius,
            "R90_0": spot_0.encircled_90_radius,
            "RMS_r_N": getattr(spot_n, "rms_radius", 0.0),
            "RMS_r_0": getattr(spot_0, "rms_radius", 0.0),
            "theta90_N": ang_n.theta_90_deg,
            "theta90_0": ang_0.theta_90_deg,
            "RMS_theta_N": getattr(ang_n, "theta_rms_deg", 0.0),
            "RMS_theta_0": getattr(ang_0, "theta_rms_deg", 0.0),
            "eta_both_cond_N": eta_both_cond_n,
            "eta_both_cond_0": eta_both_cond_0,
        },
    )

