"""
Optical elements: Sequential elements (Aperture, ThinLens, PlaneMirror)
and 3D non-sequential elements (Surface3D, ThinLens3D, PlaneMirror3D, SphericalMirror3D).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from .ray import RayBundle, RayStatus
from .geometry3d import (
    normalize,
    build_orthonormal_basis,
    reflect_vector,
    global_to_local,
    local_to_global,
)


def rotation_matrix_xyz(tip_x_rad: float, tilt_y_rad: float, rot_z_rad: float = 0.0) -> np.ndarray:
    """
    Compute 3x3 rotation matrix using Tait-Bryan angles:
    R = Rz(rot_z) * Ry(tilt_y) * Rx(tip_x)
    """
    cx, sx = np.cos(tip_x_rad), np.sin(tip_x_rad)
    cy, sy = np.cos(tilt_y_rad), np.sin(tilt_y_rad)
    cz, sz = np.cos(rot_z_rad), np.sin(rot_z_rad)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


# ==========================================
# SEQUENTIAL BASE & ELEMENTS (PRE-SLICER)
# ==========================================
class OpticalElement(ABC):
    """Abstract base class for sequential optical elements."""

    def __init__(self, name: str, z: float, enabled: bool = True):
        self.name = name
        self.z = float(z)
        self.enabled = enabled

    @abstractmethod
    def trace(self, bundle: RayBundle) -> None:
        pass


class CircularAperture(OpticalElement):
    """Circular clear aperture at plane z."""

    def __init__(
        self,
        name: str,
        z: float,
        diameter: float,
        decenter_x: float = 0.0,
        decenter_y: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__(name=name, z=z, enabled=enabled)
        self.diameter = float(diameter)
        self.radius = self.diameter / 2.0
        self.decenter_x = float(decenter_x)
        self.decenter_y = float(decenter_y)

    def trace(self, bundle: RayBundle) -> None:
        if not self.enabled:
            return

        bundle.propagate_to_z(self.z)
        mask = bundle.active_mask
        dx = bundle.x[mask] - self.decenter_x
        dy = bundle.y[mask] - self.decenter_y
        r_sq = dx**2 + dy**2

        active_indices = np.where(mask)[0]
        clipped_indices = active_indices[r_sq > (self.radius**2)]
        bundle.status[clipped_indices] = RayStatus.CLIPPED
        bundle.record_snapshot(f"{self.name} (Aperture)")


class ThinLens(OpticalElement):
    """Paraxial thin lens model along nominal z optical axis."""

    def __init__(
        self,
        name: str,
        z: float,
        focal_length: float,
        diameter: float = 25.4,
        decenter_x: float = 0.0,
        decenter_y: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__(name=name, z=z, enabled=enabled)
        self.focal_length = float(focal_length)
        self.diameter = float(diameter)
        self.radius = self.diameter / 2.0
        self.decenter_x = float(decenter_x)
        self.decenter_y = float(decenter_y)

    def trace(self, bundle: RayBundle) -> None:
        if not self.enabled:
            return

        bundle.propagate_to_z(self.z)
        mask = bundle.active_mask
        active_idx = np.where(mask)[0]
        if len(active_idx) == 0:
            return

        x_loc = bundle.x[active_idx] - self.decenter_x
        y_loc = bundle.y[active_idx] - self.decenter_y
        r_sq = x_loc**2 + y_loc**2

        clipped_mask = r_sq > (self.radius**2)
        if np.any(clipped_mask):
            bundle.status[active_idx[clipped_mask]] = RayStatus.CLIPPED

        surviving_idx = active_idx[~clipped_mask]
        if len(surviving_idx) == 0:
            bundle.record_snapshot(f"{self.name} (Lens)")
            return

        x_surv = x_loc[~clipped_mask]
        y_surv = y_loc[~clipped_mask]

        kz = bundle.k[surviving_idx, 2]
        safe_kz = np.where(np.abs(kz) < 1e-12, 1e-12, kz)
        u_in = bundle.k[surviving_idx, 0] / safe_kz
        v_in = bundle.k[surviving_idx, 1] / safe_kz

        u_out = u_in - (x_surv / self.focal_length)
        v_out = v_in - (y_surv / self.focal_length)

        dir_z_sign = np.sign(kz)
        dir_z_sign[dir_z_sign == 0] = 1.0

        denom = np.sqrt(u_out**2 + v_out**2 + 1.0)
        bundle.k[surviving_idx, 0] = (u_out / denom) * dir_z_sign
        bundle.k[surviving_idx, 1] = (v_out / denom) * dir_z_sign
        bundle.k[surviving_idx, 2] = (1.0 / denom) * dir_z_sign

        bundle.record_snapshot(f"{self.name} (Lens f={self.focal_length:.1f}mm)")


class PlaneMirror(OpticalElement):
    """Sequential plane mirror model."""

    def __init__(
        self,
        name: str,
        center: Tuple[float, float, float],
        width: float,
        height: float,
        tip_x_deg: float = 0.0,
        tilt_y_deg: float = 0.0,
        rot_z_deg: float = 0.0,
        is_circular: bool = False,
        enabled: bool = True,
    ):
        x0, y0, z0 = center
        super().__init__(name=name, z=z0, enabled=enabled)
        self.center = np.array([x0, y0, z0], dtype=np.float64)
        self.width = float(width)
        self.height = float(height)
        self.tip_x_deg = float(tip_x_deg)
        self.tilt_y_deg = float(tilt_y_deg)
        self.rot_z_deg = float(rot_z_deg)
        self.is_circular = is_circular

        self.R = rotation_matrix_xyz(
            np.radians(self.tip_x_deg),
            np.radians(self.tilt_y_deg),
            np.radians(self.rot_z_deg),
        )
        self.normal = self.R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)

    def set_angles(self, tip_x_deg: float, tilt_y_deg: float, rot_z_deg: float = 0.0) -> None:
        self.tip_x_deg = float(tip_x_deg)
        self.tilt_y_deg = float(tilt_y_deg)
        self.rot_z_deg = float(rot_z_deg)
        self.R = rotation_matrix_xyz(
            np.radians(self.tip_x_deg),
            np.radians(self.tilt_y_deg),
            np.radians(self.rot_z_deg),
        )
        self.normal = self.R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)

    def intersect(self, r: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_dot_n = np.dot(k, self.normal)
        parallel_mask = np.abs(k_dot_n) < 1e-12

        p0_minus_r = self.center - r
        num = np.dot(p0_minus_r, self.normal)
        safe_k_dot_n = np.where(parallel_mask, 1e-12, k_dot_n)
        t = num / safe_k_dot_n
        t[parallel_mask] = -1.0

        hit_pos = r + t[:, np.newaxis] * k
        rel_pos = hit_pos - self.center
        local_pos = rel_pos @ self.R
        return t, hit_pos, local_pos

    def trace(self, bundle: RayBundle) -> None:
        if not self.enabled:
            return

        mask = bundle.active_mask
        active_idx = np.where(mask)[0]
        if len(active_idx) == 0:
            return

        r_sub = bundle.r[active_idx]
        k_sub = bundle.k[active_idx]

        t, hit_pos, local_pos = self.intersect(r_sub, k_sub)
        u = local_pos[:, 0]
        v = local_pos[:, 1]

        valid_t = t > 1e-6
        if self.is_circular:
            in_aperture = (u**2 + v**2) <= (self.width / 2.0)**2
        else:
            in_aperture = (np.abs(u) <= self.width / 2.0) & (np.abs(v) <= self.height / 2.0)

        hit_mask = valid_t & in_aperture
        missed_mask = ~hit_mask

        if np.any(missed_mask):
            bundle.status[active_idx[missed_mask]] = RayStatus.MISSED

        hit_indices = active_idx[hit_mask]
        if len(hit_indices) > 0:
            bundle.r[hit_indices] = hit_pos[hit_mask]
            k_in = bundle.k[hit_indices]
            k_dot_n = np.sum(k_in * self.normal, axis=1, keepdims=True)
            k_out = k_in - 2.0 * k_dot_n * self.normal
            norms = np.linalg.norm(k_out, axis=1, keepdims=True)
            bundle.k[hit_indices] = k_out / norms

        bundle.record_snapshot(f"{self.name} (Mirror)")


# ==========================================
# 3D NON-SEQUENTIAL OPTICAL SURFACES
# ==========================================
class Surface3D(ABC):
    """
    Base class for arbitrary 3D optical surfaces in a non-sequential scene.
    Has full 3D position and orientation.
    """

    def __init__(
        self,
        name: str,
        center: np.ndarray | Tuple[float, float, float],
        normal: np.ndarray | Tuple[float, float, float],
        is_active: bool = True,
    ):
        self.name = name
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        self.normal = normalize(np.asarray(normal, dtype=np.float64).reshape(3))
        self.is_active = is_active
        # Build local basis: u (transverse), v (orthogonal), w (surface normal)
        self.u_axis, self.v_axis, self.w_axis = build_orthonormal_basis(self.normal)

    @property
    def u_vec(self) -> np.ndarray:
        return self.u_axis

    @property
    def v_vec(self) -> np.ndarray:
        return self.v_axis

    def set_pose(
        self,
        center: np.ndarray | Tuple[float, float, float],
        normal: np.ndarray | Tuple[float, float, float],
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        self.normal = normalize(np.asarray(normal, dtype=np.float64).reshape(3))
        self.u_axis, self.v_axis, self.w_axis = build_orthonormal_basis(self.normal)

    @abstractmethod
    def intersect(self, r: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Intersect rays with surface.
        r: (N, 3), k: (N, 3)
        Returns:
            t: shape (N,), distance along ray
            hit_pos: shape (N, 3), 3D point of intersection
            local_uv: shape (N, 2), local (u, v) coordinates on surface
            in_bounds: shape (N,), bool mask if hit falls inside clear aperture
        """
        pass

    @abstractmethod
    def interact(
        self,
        bundle: RayBundle,
        hit_indices: np.ndarray,
        hit_pos: np.ndarray,
        local_uv: np.ndarray,
    ) -> None:
        """Apply optical interaction (reflection, refraction, absorption) in place."""
        pass


class ThinLens3D(Surface3D):
    """
    3D Thin Lens placed at arbitrary position and orientation.
    Has center, optical axis normal, clear diameter, and focal length f.
    Transforms rays into local (u, v) frame, applies paraxial thin-lens transformation,
    then transforms back into global 3D space.
    """

    def __init__(
        self,
        name: str,
        center: np.ndarray | Tuple[float, float, float],
        normal: np.ndarray | Tuple[float, float, float],
        focal_length: float,
        diameter: float = 50.0,
        is_active: bool = True,
    ):
        super().__init__(name=name, center=center, normal=normal, is_active=is_active)
        self.focal_length = float(focal_length)
        self.diameter = float(diameter)
        self.radius = self.diameter / 2.0

    def intersect(self, r: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Plane equation: (p - center) . normal = 0
        # t = (center - r) . normal / (k . normal)
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

        # Check positive distance and circular aperture
        valid_t = t > 1e-5
        in_aperture = (u**2 + v**2) <= (self.radius**2)
        in_bounds = valid_t & in_aperture

        return t, hit_pos, local_uv, in_bounds

    def get_local_hit(self, r_in: np.ndarray, k_in: np.ndarray) -> Tuple[float, float, float, bool, np.ndarray]:
        """
        Intersect a single ray with this 3D lens.
        Returns:
            u: local transverse coordinate on lens face (mm)
            v: local orthogonal coordinate on lens face (mm)
            radial_distance: sqrt(u^2 + v^2) (mm)
            in_aperture: bool, whether hit falls inside clear diameter
            hit_pos: 3D coordinates of hit point (3,)
        """
        r_arr = np.asarray(r_in, dtype=np.float64).reshape(1, 3)
        k_arr = normalize(np.asarray(k_in, dtype=np.float64).reshape(1, 3))
        t, hit_pos, local_uv, in_bounds = self.intersect(r_arr, k_arr)
        u = float(local_uv[0, 0])
        v = float(local_uv[0, 1])
        radial_distance = float(np.sqrt(u**2 + v**2))
        return u, v, radial_distance, bool(in_bounds[0]), hit_pos[0]

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

        # Refract rays using paraxial thin lens formula in local frame (u, v, w)
        k_sub = bundle.k[hit_indices]
        ku = np.dot(k_sub, self.u_axis)
        kv = np.dot(k_sub, self.v_axis)
        kw = np.dot(k_sub, self.w_axis)

        safe_kw = np.where(np.abs(kw) < 1e-12, 1e-12, kw)
        su_in = ku / safe_kw
        sv_in = kv / safe_kw

        u_loc = local_uv[:, 0]
        v_loc = local_uv[:, 1]

        # Paraxial slope deflection
        su_out = su_in - (u_loc / self.focal_length)
        sv_out = sv_in - (v_loc / self.focal_length)

        sgn_w = np.sign(kw)
        sgn_w[sgn_w == 0] = 1.0

        denom = np.sqrt(su_out**2 + sv_out**2 + 1.0)
        k_loc_u = (su_out / denom) * sgn_w
        k_loc_v = (sv_out / denom) * sgn_w
        k_loc_w = (1.0 / denom) * sgn_w

        # Transform back to global coordinates
        k_glob_out = (
            k_loc_u[:, np.newaxis] * self.u_axis +
            k_loc_v[:, np.newaxis] * self.v_axis +
            k_loc_w[:, np.newaxis] * self.w_axis
        )
        bundle.k[hit_indices] = normalize(k_glob_out)


class PlaneMirror3D(Surface3D):
    """
    3D Plane Mirror with arbitrary 3D center, normal, and rectangular or circular clear aperture.
    """

    def __init__(
        self,
        name: str,
        center: np.ndarray | Tuple[float, float, float],
        normal: np.ndarray | Tuple[float, float, float],
        width: float,
        height: float,
        is_circular: bool = False,
        is_active: bool = True,
    ):
        super().__init__(name=name, center=center, normal=normal, is_active=is_active)
        self.width = float(width)
        self.height = float(height)
        self.is_circular = is_circular

    def intersect(self, r: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        k_out = reflect_vector(k_in, self.normal)
        bundle.k[hit_indices] = k_out


class SphericalMirror3D(Surface3D):
    """
    3D Powered Concave Spherical Mirror.
    Radius of curvature R = 2 * focal_length.
    Provides optical power (refocusing or collimating each channel independently).
    """

    def __init__(
        self,
        name: str,
        center: np.ndarray | Tuple[float, float, float],
        normal: np.ndarray | Tuple[float, float, float],
        focal_length: float,
        width: float,
        height: float,
        is_circular: bool = False,
        is_active: bool = True,
    ):
        super().__init__(name=name, center=center, normal=normal, is_active=is_active)
        self.focal_length = float(focal_length)
        self.radius_of_curvature = 2.0 * self.focal_length
        self.width = float(width)
        self.height = float(height)
        self.is_circular = is_circular

    def intersect(self, r: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Intersect with vertex tangent plane
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
        u = local_uv[:, 0]
        v = local_uv[:, 1]
        R = self.radius_of_curvature

        # Spherical concave mirror local normal variation:
        # Facing incoming beam along -w:
        # Effective normal n_eff = normal - (u/R) * u_axis - (v/R) * v_axis
        n_eff = self.normal - (u[:, np.newaxis] / R) * self.u_axis - (v[:, np.newaxis] / R) * self.v_axis
        n_eff = normalize(n_eff)

        k_in = bundle.k[hit_indices]
        k_dot_neff = np.sum(k_in * n_eff, axis=-1, keepdims=True)
        k_out = k_in - 2.0 * k_dot_neff * n_eff
        bundle.k[hit_indices] = normalize(k_out)
