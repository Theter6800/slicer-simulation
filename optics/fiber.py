"""
Multimode optical fiber model with arbitrary 3D pose (center and axis).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .ray import RayBundle, RayStatus
from .geometry3d import normalize, build_orthonormal_basis


@dataclass
class FiberCouplingResult:
    """Detailed results of fiber coupling analysis."""
    total_launched_power: float
    power_reaching_fiber_plane: float
    power_reaching_coupling_optic: float
    accepted_power: float
    
    geometric_coupling_efficiency: float       # accepted / launched
    efficiency_rel_coupling_optic: float       # accepted / reaching_coupling_optic
    efficiency_rel_fiber_plane: float          # accepted / reaching_fiber_plane
    
    n_rays_reaching_plane: int
    n_accepted: int
    n_rejected_by_position: int
    n_rejected_by_na: int
    n_rejected_by_both: int
    
    spot_rms_radius: float                     # mm
    encircled_80_diameter: float               # mm
    mean_ray_angle_deg: float                  # degrees
    max_ray_angle_deg: float                   # degrees
    
    # Per-ray arrays at fiber face
    r_coords: np.ndarray                       # radial distance from fiber axis (mm)
    theta_angles_deg: np.ndarray               # angle relative to fiber axis (deg)
    hit_x: np.ndarray                          # transverse local u on fiber face (mm)
    hit_y: np.ndarray                          # transverse local v on fiber face (mm)
    classifications: np.ndarray                # RayStatus values for each ray
    powers: np.ndarray                         # power of each ray reaching fiber

    power_inside_core: float = 0.0             # spatial core condition only
    power_inside_na: float = 0.0               # angular NA condition only
    power_inside_core_and_na: float = 0.0      # both conditions satisfied
    power_accounting: Optional[Any] = None     # Full stage-by-stage PowerAccounting instance


class Fiber:
    """
    Multimode optical fiber receiver model with full 3D position and orientation.
    Default core diameter = 1.0 mm, NA = 0.22, n_external = 1.0 (air).
    """

    def __init__(
        self,
        core_diameter: float = 1.0,
        na: float = 0.22,
        position: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 300.0),
        axis: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
        n_external: float = 1.0,
    ):
        self.core_diameter = float(core_diameter)
        self.core_radius = self.core_diameter / 2.0
        self.na = float(na)
        self.position = np.asarray(position, dtype=np.float64).reshape(3)
        self.axis = normalize(np.asarray(axis, dtype=np.float64).reshape(3))
        self.n_external = float(n_external)
        self.u_axis, self.v_axis, self.w_axis = build_orthonormal_basis(self.axis)

    def set_pose(
        self,
        position: Tuple[float, float, float] | np.ndarray,
        axis: Tuple[float, float, float] | np.ndarray,
    ) -> None:
        self.position = np.asarray(position, dtype=np.float64).reshape(3)
        self.axis = normalize(np.asarray(axis, dtype=np.float64).reshape(3))
        self.u_axis, self.v_axis, self.w_axis = build_orthonormal_basis(self.axis)

    @property
    def u(self) -> np.ndarray:
        return self.u_axis

    @property
    def v(self) -> np.ndarray:
        return self.v_axis

    @property
    def w(self) -> np.ndarray:
        return self.w_axis

    @property
    def max_acceptance_angle_rad(self) -> float:
        arg = np.clip(self.na / self.n_external, -1.0, 1.0)
        return float(np.arcsin(arg))

    @property
    def max_acceptance_angle_deg(self) -> float:
        return float(np.degrees(self.max_acceptance_angle_rad))

    def evaluate_coupling(
        self,
        bundle: RayBundle,
        power_reaching_coupling_optic: Optional[float] = None,
    ) -> FiberCouplingResult:
        """
        Intersect active rays with the 3D fiber face plane,
        evaluate spatial (r <= core_radius) and angular (theta <= asin(NA)) acceptance.
        """
        total_launched = bundle.total_power
        mask = bundle.active_mask
        arrived_indices = np.where(mask)[0]
        n_arrived = len(arrived_indices)

        if power_reaching_coupling_optic is None:
            power_reaching_coupling_optic = total_launched

        if n_arrived == 0:
            return FiberCouplingResult(
                total_launched_power=total_launched,
                power_reaching_fiber_plane=0.0,
                power_reaching_coupling_optic=power_reaching_coupling_optic,
                accepted_power=0.0,
                geometric_coupling_efficiency=0.0,
                efficiency_rel_coupling_optic=0.0,
                efficiency_rel_fiber_plane=0.0,
                n_rays_reaching_plane=0,
                n_accepted=0,
                n_rejected_by_position=0,
                n_rejected_by_na=0,
                n_rejected_by_both=0,
                spot_rms_radius=0.0,
                encircled_80_diameter=0.0,
                mean_ray_angle_deg=0.0,
                max_ray_angle_deg=0.0,
                r_coords=np.array([], dtype=np.float64),
                theta_angles_deg=np.array([], dtype=np.float64),
                hit_x=np.array([], dtype=np.float64),
                hit_y=np.array([], dtype=np.float64),
                classifications=np.array([], dtype=np.int32),
                powers=np.array([], dtype=np.float64),
            )

        # Propagate rays to the 3D fiber plane
        r_sub = bundle.r[arrived_indices]
        k_sub = bundle.k[arrived_indices]

        k_dot_axis = np.dot(k_sub, self.axis)
        denom = np.where(np.abs(k_dot_axis) < 1e-12, 1e-12, k_dot_axis)
        p0_minus_r = self.position - r_sub
        t = np.dot(p0_minus_r, self.axis) / denom

        # Update ray positions to hit points on fiber plane
        hit_pos = r_sub + t[:, np.newaxis] * k_sub
        bundle.r[arrived_indices] = hit_pos
        bundle.record_snapshot("Fiber Face")

        # Local transverse coordinates on fiber face
        rel = hit_pos - self.position
        u = np.dot(rel, self.u_axis)
        v = np.dot(rel, self.v_axis)
        r = np.sqrt(u**2 + v**2)

        # Angle relative to fiber axis
        # Cosine of angle between ray propagation direction and fiber forward normal
        cos_theta = np.clip(k_dot_axis, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)

        # Acceptance criteria
        spatial_ok = r <= self.core_radius
        sin_theta = np.sin(theta_rad)
        na_ok = (self.n_external * sin_theta) <= self.na

        both_ok = spatial_ok & na_ok
        spatial_ok_na_rej = spatial_ok & (~na_ok)
        spatial_rej_na_ok = (~spatial_ok) & na_ok
        both_rej = (~spatial_ok) & (~na_ok)

        # Update ray statuses
        bundle.status[arrived_indices[both_ok]] = RayStatus.ACCEPTED_BY_FIBER
        bundle.status[arrived_indices[spatial_ok_na_rej]] = RayStatus.REJECTED_BY_NA
        bundle.status[arrived_indices[spatial_rej_na_ok]] = RayStatus.REJECTED_BY_POSITION
        bundle.status[arrived_indices[both_rej]] = RayStatus.REJECTED_BY_BOTH

        # Power calculations
        ray_powers = bundle.power[arrived_indices]
        p_fiber_plane = float(np.sum(ray_powers))
        p_inside_core = float(np.sum(ray_powers[spatial_ok]))
        p_inside_na = float(np.sum(ray_powers[na_ok]))
        p_inside_core_and_na = float(np.sum(ray_powers[both_ok]))
        p_accepted = p_inside_core_and_na

        eff_geom = p_accepted / total_launched if total_launched > 0 else 0.0
        eff_optic = p_accepted / power_reaching_coupling_optic if power_reaching_coupling_optic > 0 else 0.0
        eff_plane = p_accepted / p_fiber_plane if p_fiber_plane > 0 else 0.0

        # Spot metrics
        mean_u = np.average(u, weights=ray_powers) if p_fiber_plane > 0 else 0.0
        mean_v = np.average(v, weights=ray_powers) if p_fiber_plane > 0 else 0.0
        variance_r = np.average((u - mean_u)**2 + (v - mean_v)**2, weights=ray_powers) if p_fiber_plane > 0 else 0.0
        spot_rms = float(np.sqrt(variance_r))

        sort_order = np.argsort(r)
        cum_power = np.cumsum(ray_powers[sort_order])
        idx_80 = np.searchsorted(cum_power, 0.8 * p_fiber_plane) if p_fiber_plane > 0 else 0
        r_80 = r[sort_order[min(idx_80, len(sort_order) - 1)]] if len(sort_order) > 0 else 0.0

        mean_angle = float(np.average(theta_deg, weights=ray_powers)) if p_fiber_plane > 0 else 0.0
        max_angle = float(np.max(theta_deg)) if len(theta_deg) > 0 else 0.0

        return FiberCouplingResult(
            total_launched_power=total_launched,
            power_reaching_fiber_plane=p_fiber_plane,
            power_reaching_coupling_optic=power_reaching_coupling_optic,
            accepted_power=p_accepted,
            geometric_coupling_efficiency=eff_geom,
            efficiency_rel_coupling_optic=eff_optic,
            efficiency_rel_fiber_plane=eff_plane,
            power_inside_core=p_inside_core,
            power_inside_na=p_inside_na,
            power_inside_core_and_na=p_inside_core_and_na,
            power_accounting=None,
            n_rays_reaching_plane=n_arrived,
            n_accepted=int(np.sum(both_ok)),
            n_rejected_by_position=int(np.sum(spatial_rej_na_ok)),
            n_rejected_by_na=int(np.sum(spatial_ok_na_rej)),
            n_rejected_by_both=int(np.sum(both_rej)),
            spot_rms_radius=spot_rms,
            encircled_80_diameter=2.0 * float(r_80),
            mean_ray_angle_deg=mean_angle,
            max_ray_angle_deg=max_angle,
            r_coords=r,
            theta_angles_deg=theta_deg,
            hit_x=u,
            hit_y=v,
            classifications=bundle.status[arrived_indices].copy(),
            powers=ray_powers,
        )
