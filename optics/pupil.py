"""
3D Pupil mirror relay system with independent 3D poses, powered concave mirrors,
and auto-placement / auto-aiming routines.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from .ray import RayBundle, RayStatus
from .elements import Surface3D, rotation_matrix_xyz
from .geometry3d import (
    normalize,
    build_orthonormal_basis,
    reflect_vector,
    reflection_bisector,
)
from .slicer import SlicerArray


@dataclass
class PupilPlaneInfo:
    """Characteristics of a channel pupil beam."""
    channel_id: int
    z_pupil: float
    center_x: float
    center_y: float
    rms_radius: float
    diameter_80: float
    n_rays: int
    chief_ray_direction: np.ndarray


class PupilMirror(Surface3D):
    """
    An individual 3D Pupil Mirror placed along a slicer channel's reflected branch.
    Supports both flat steering and powered spherical concave refocusing.
    """

    def __init__(
        self,
        channel_id: int,
        name: str,
        center: np.ndarray | Tuple[float, float, float],
        normal: np.ndarray | Tuple[float, float, float],
        width: float = 12.0,
        height: float = 12.0,
        focal_length: Optional[float] = None,
        is_circular: bool = False,
        enabled: bool = True,
    ):
        super().__init__(name=name, center=center, normal=normal, is_active=enabled)
        self.channel_id = int(channel_id)
        self.width = float(width)
        self.height = float(height)
        self.focal_length = float(focal_length) if focal_length is not None else None
        self.is_powered = (self.focal_length is not None and self.focal_length > 0.0)
        self.radius_of_curvature = 2.0 * self.focal_length if self.is_powered else 0.0
        self.is_circular = is_circular
        self.enabled = enabled
        self.tip_x_deg: float = 0.0
        self.tilt_y_deg: float = 0.0
        self.rot_z_deg: float = 0.0
        self._sync_angles_from_normal()

    @property
    def center_x(self) -> float:
        return float(self.center[0])

    @property
    def center_y(self) -> float:
        return float(self.center[1])

    @property
    def z(self) -> float:
        return float(self.center[2])

    def _sync_angles_from_normal(self) -> None:
        """Extract tip_x and tilt_y angles from current normal vector."""
        n = self.normal
        ny = float(np.clip(n[1], -1.0, 1.0))
        tip_x_rad = float(np.arcsin(ny))
        cos_tip = np.cos(tip_x_rad)
        if np.abs(cos_tip) > 1e-6:
            tilt_y_rad = float(np.arctan2(-n[0], -n[2]))
        else:
            tilt_y_rad = 0.0
        self.tip_x_deg = float(np.degrees(tip_x_rad))
        self.tilt_y_deg = float(np.degrees(tilt_y_rad))

    def set_angles(self, tip_x_deg: float, tilt_y_deg: float, rot_z_deg: float = 0.0) -> None:
        """Set pupil mirror orientation from tip, tilt, and roll angles in degrees."""
        self.tip_x_deg = float(tip_x_deg)
        self.tilt_y_deg = float(tilt_y_deg)
        self.rot_z_deg = float(rot_z_deg)
        R = rotation_matrix_xyz(
            np.radians(self.tip_x_deg),
            np.radians(self.tilt_y_deg),
            np.radians(self.rot_z_deg),
        )
        n0 = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        new_normal = normalize(R @ n0)
        self.set_pose(self.center, new_normal)

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set pupil mirror 3D position."""
        new_center = np.array([float(x), float(y), float(z)], dtype=np.float64)
        self.set_pose(new_center, self.normal)

    def get_local_hit(self, r_in: np.ndarray, k_in: np.ndarray) -> Tuple[float, float, float, bool, np.ndarray]:
        """
        Intersect a single ray (r_in, k_in) with this pupil mirror.
        Returns:
            u: local transverse coordinate on pupil mirror face (mm)
            v: local orthogonal coordinate on pupil mirror face (mm)
            miss_distance: sqrt(u^2 + v^2) (mm)
            in_bounds: bool, whether hit falls inside clear aperture
            hit_pos: 3D coordinates of hit point (3,)
        """
        r_arr = np.asarray(r_in, dtype=np.float64).reshape(1, 3)
        k_arr = normalize(np.asarray(k_in, dtype=np.float64).reshape(1, 3))
        t, hit_pos, local_uv, in_bounds = self.intersect(r_arr, k_arr)
        u = float(local_uv[0, 0])
        v = float(local_uv[0, 1])
        miss_distance = float(np.sqrt(u**2 + v**2))
        return u, v, miss_distance, bool(in_bounds[0]), hit_pos[0]

    def aim_towards(self, k_in: np.ndarray, target_point: np.ndarray) -> np.ndarray:
        """
        Analytically calculate and set the mirror normal so an incoming ray along k_in
        reflects directly toward target_point.
        """
        d = target_point - self.center
        k_out = normalize(d)
        new_normal = reflection_bisector(k_in, k_out)
        self.set_pose(self.center, new_normal)
        self._sync_angles_from_normal()
        return new_normal

    def intersect(self, r: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.enabled:
            n_r = len(r)
            return (
                np.full(n_r, -1.0),
                np.zeros((n_r, 3)),
                np.zeros((n_r, 2)),
                np.zeros(n_r, dtype=bool),
            )

        k_dot_n = np.dot(k, self.normal)
        denom = np.where(np.abs(k_dot_n) < 1e-12, 1e-12, k_dot_n)
        p0_minus_r = self.center - r
        num = np.dot(p0_minus_r, self.normal)
        t = num / denom

        hit_pos = r + t[:, np.newaxis] * k
        rel = hit_pos - self.center
        u = np.dot(rel, self.u_axis)
        v = np.dot(rel, self.v_axis)
        local_uv = np.column_stack([u, v])

        valid_t = t > 1e-5
        if self.is_circular:
            in_aperture = (u**2 + v**2) <= (self.width / 2.0)**2
        else:
            in_aperture = (np.abs(u) <= self.width / 2.0) & (np.abs(v) <= self.height / 2.0)

        in_bounds = valid_t & in_aperture
        return t, hit_pos, local_uv, in_bounds

    def interact(
        self,
        bundle: RayBundle,
        hit_indices: np.ndarray,
        hit_pos: np.ndarray,
        local_uv: np.ndarray,
    ) -> None:
        if len(hit_indices) == 0:
            return

        bundle.r[hit_indices] = hit_pos
        k_in = bundle.k[hit_indices]

        if self.is_powered and self.radius_of_curvature > 0:
            # Concave spherical mirror power
            u = local_uv[:, 0]
            v = local_uv[:, 1]
            R = self.radius_of_curvature
            # Local normal perturbation
            n_eff = self.normal - (u[:, np.newaxis] / R) * self.u_axis - (v[:, np.newaxis] / R) * self.v_axis
            n_eff = normalize(n_eff)
            k_dot_n = np.sum(k_in * n_eff, axis=-1, keepdims=True)
            k_out = k_in - 2.0 * k_dot_n * n_eff
            bundle.k[hit_indices] = normalize(k_out)
        else:
            # Flat specular reflection
            k_out = reflect_vector(k_in, self.normal)
            bundle.k[hit_indices] = k_out


def generate_one_sided_pupil_positions(
    side: str = "lower",
    n_channels: int = 2,
    z_pupil: float = 250.0,
    transverse_offset: float = 25.0,
    spacing: float = 14.0,
) -> List[np.ndarray]:
    """
    Generate 3D center positions P_i for pupil mirrors positioned strictly
    on one chosen side of the original optical axis.

    side in {"lower", "upper", "left", "right"}:
    - "lower": all y < 0, offset = -abs(transverse_offset), mirrors spaced along x
    - "upper": all y > 0, offset = +abs(transverse_offset), mirrors spaced along x
    - "left": all x < 0, offset = -abs(transverse_offset), mirrors spaced along y
    - "right": all x > 0, offset = +abs(transverse_offset), mirrors spaced along y
    """
    side_clean = side.lower().strip()
    positions: List[np.ndarray] = []

    if n_channels == 4:
        # Compact 2x2 cluster on the chosen side to fit well inside NA acceptance
        dx_vals = [-spacing / 2.0, spacing / 2.0]
        dy_vals = [-spacing / 2.0, spacing / 2.0]
        if side_clean == "lower":
            y_base = -abs(transverse_offset)
            for dy in dy_vals:
                for dx in dx_vals:
                    positions.append(np.array([dx, y_base + dy, z_pupil], dtype=np.float64))
        elif side_clean == "upper":
            y_base = abs(transverse_offset)
            for dy in dy_vals:
                for dx in dx_vals:
                    positions.append(np.array([dx, y_base + dy, z_pupil], dtype=np.float64))
        elif side_clean == "left":
            x_base = -abs(transverse_offset)
            for dx in dx_vals:
                for dy in dy_vals:
                    positions.append(np.array([x_base + dx, dy, z_pupil], dtype=np.float64))
        elif side_clean == "right":
            x_base = abs(transverse_offset)
            for dx in dx_vals:
                for dy in dy_vals:
                    positions.append(np.array([x_base + dx, dy, z_pupil], dtype=np.float64))
        else:
            y_base = -abs(transverse_offset)
            for dy in dy_vals:
                for dx in dx_vals:
                    positions.append(np.array([dx, y_base + dy, z_pupil], dtype=np.float64))
    else:
        # 1xN array on the chosen side (e.g. 2 channels)
        offsets = [(- (n_channels - 1) / 2.0 + i) * spacing for i in range(n_channels)]
        if side_clean == "lower":
            y_pos = -abs(transverse_offset)
            for dx in offsets:
                positions.append(np.array([dx, y_pos, z_pupil], dtype=np.float64))
        elif side_clean == "upper":
            y_pos = abs(transverse_offset)
            for dx in offsets:
                positions.append(np.array([dx, y_pos, z_pupil], dtype=np.float64))
        elif side_clean == "left":
            x_pos = -abs(transverse_offset)
            for dy in offsets:
                positions.append(np.array([x_pos, dy, z_pupil], dtype=np.float64))
        elif side_clean == "right":
            x_pos = abs(transverse_offset)
            for dy in offsets:
                positions.append(np.array([x_pos, dy, z_pupil], dtype=np.float64))
        else:
            y_pos = -abs(transverse_offset)
            for dx in offsets:
                positions.append(np.array([dx, y_pos, z_pupil], dtype=np.float64))

    return positions


class PupilRelaySystem:
    """
    Collection of per-channel Pupil Mirrors positioned along each slicer branch.
    """

    def __init__(self, name: str = "3D Pupil Relay", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.mirrors: List[PupilMirror] = []

    def add_mirror(self, mirror: PupilMirror) -> None:
        self.mirrors.append(mirror)

    @property
    def centroid(self) -> np.ndarray:
        """Centroid of all active pupil mirrors in global 3D space."""
        if not self.mirrors:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        centers = [m.center for m in self.mirrors if m.enabled]
        if not centers:
            centers = [m.center for m in self.mirrors]
        return np.mean(centers, axis=0)

    def auto_place_mirrors(
        self,
        slicer: SlicerArray,
        distance: float = 60.0,
        k_in: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> None:
        """
        Position each pupil mirror P_i along the reflected chief ray of its slice S_i:
        P_i = S_i + distance * k_ref_i
        """
        for m in self.mirrors:
            for s in slicer.slices:
                if s.slice_id == m.channel_id:
                    new_pos = s.compute_pupil_position(distance, k_in)
                    m.set_pose(new_pos, m.normal)
                    break

    def aim_all_at_target(
        self,
        slicer: SlicerArray,
        target_point: np.ndarray,
        k_in: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> None:
        """
        Analytically orient all pupil mirrors so their chief rays reflect toward target_point.
        """
        for m in self.mirrors:
            for s in slicer.slices:
                if s.slice_id == m.channel_id:
                    k_ref = s.get_reflected_chief_ray(k_in)
                    m.aim_towards(k_ref, target_point)
                    break

    def redirect_all_to_relay_axis(
        self,
        slicer: SlicerArray,
        condenser_center: np.ndarray,
        relay_axis_direction: Optional[np.ndarray] = None,
        mode: str = "parallel",
    ) -> None:
        """
        Analytically orient all pupil mirrors so their chief rays redirect along the new common relay axis.

        mode:
        - "parallel": Reflects chief rays parallel to relay_axis_direction (or normalize(condenser_center - centroid)).
          This aligns the chief rays parallel to the condenser lens axis so the lens focuses them all into the fiber core (r <= 0.5mm).
        - "target_center": Reflects chief rays directly towards the physical center of the condenser lens.
        """
        condenser_c = np.asarray(condenser_center, dtype=np.float64).reshape(3)
        if relay_axis_direction is not None:
            a_relay = normalize(relay_axis_direction)
        else:
            a_relay = normalize(condenser_c - self.centroid)

        for m in self.mirrors:
            for s in slicer.slices:
                if s.slice_id == m.channel_id:
                    k_slicer_to_pupil = normalize(m.center - s.center)
                    if mode == "parallel":
                        n_p = reflection_bisector(k_slicer_to_pupil, a_relay)
                        m.set_pose(m.center, n_p)
                    else:
                        m.aim_towards(k_slicer_to_pupil, condenser_c)
                    break


class PupilAnalysis:
    """
    Locates minimum beam waists along each channel branch.
    """

    @staticmethod
    def find_pupil_planes(
        bundle: RayBundle,
        z_start: float,
        z_end: float,
        n_steps: int = 70,
    ) -> List[PupilPlaneInfo]:
        unique_channels = np.unique(bundle.channel_id[bundle.active_mask])
        results: List[PupilPlaneInfo] = []
        z_scan = np.linspace(z_start, z_end, n_steps)

        for ch in sorted(unique_channels):
            if ch < 0:
                continue

            ch_mask = (bundle.channel_id == ch) & bundle.active_mask
            if not np.any(ch_mask):
                continue

            r_init = bundle.r[ch_mask]
            k_init = bundle.k[ch_mask]
            p_init = bundle.power[ch_mask]
            tot_p = float(np.sum(p_init))

            rms_vals = np.zeros(n_steps)
            cx_vals = np.zeros(n_steps)
            cy_vals = np.zeros(n_steps)

            for i, z_plane in enumerate(z_scan):
                dz = z_plane - r_init[:, 2]
                safe_kz = np.where(np.abs(k_init[:, 2]) < 1e-12, 1e-12, k_init[:, 2])
                t = dz / safe_kz
                x_at_z = r_init[:, 0] + k_init[:, 0] * t
                y_at_z = r_init[:, 1] + k_init[:, 1] * t

                cx = float(np.average(x_at_z, weights=p_init)) if tot_p > 0 else float(np.mean(x_at_z))
                cy = float(np.average(y_at_z, weights=p_init)) if tot_p > 0 else float(np.mean(y_at_z))
                cx_vals[i] = cx
                cy_vals[i] = cy

                var_r = float(np.average((x_at_z - cx)**2 + (y_at_z - cy)**2, weights=p_init)) if tot_p > 0 else float(np.mean((x_at_z - cx)**2 + (y_at_z - cy)**2))
                rms_vals[i] = np.sqrt(max(var_r, 0.0))

            min_idx = int(np.argmin(rms_vals))
            best_z = float(z_scan[min_idx])
            best_cx = float(cx_vals[min_idx])
            best_cy = float(cy_vals[min_idx])
            min_rms = float(rms_vals[min_idx])

            dz_best = best_z - r_init[:, 2]
            safe_kz_best = np.where(np.abs(k_init[:, 2]) < 1e-12, 1e-12, k_init[:, 2])
            t_best = dz_best / safe_kz_best
            x_best = r_init[:, 0] + k_init[:, 0] * t_best
            y_best = r_init[:, 1] + k_init[:, 1] * t_best
            dist_c = np.sqrt((x_best - best_cx)**2 + (y_best - best_cy)**2)

            sort_idx = np.argsort(dist_c)
            cum_p = np.cumsum(p_init[sort_idx])
            idx_80 = np.searchsorted(cum_p, 0.8 * tot_p) if tot_p > 0 else int(0.8 * len(dist_c))
            r_80 = float(dist_c[sort_idx[min(idx_80, len(sort_idx) - 1)]])

            chief_k = np.average(k_init, axis=0, weights=p_init) if tot_p > 0 else np.mean(k_init, axis=0)
            chief_k = normalize(chief_k)

            results.append(
                PupilPlaneInfo(
                    channel_id=int(ch),
                    z_pupil=best_z,
                    center_x=best_cx,
                    center_y=best_cy,
                    rms_radius=min_rms,
                    diameter_80=2.0 * r_80,
                    n_rays=int(np.sum(ch_mask)),
                    chief_ray_direction=chief_k,
                )
            )

        return results
