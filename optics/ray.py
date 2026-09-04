"""
RayBundle and RayStatus implementation for vectorized 3D geometric optics.

Coordinate conventions:
z = nominal optical axis / bench direction (mm)
x = horizontal transverse direction (mm)
y = vertical transverse direction (mm)
"""

from __future__ import annotations
from enum import IntEnum
from typing import Optional, List, Dict, Any
import numpy as np


class RayStatus(IntEnum):
    ACTIVE = 0
    CLIPPED = 1
    MISSED = 2
    ACCEPTED_BY_FIBER = 3
    REJECTED_BY_POSITION = 4
    REJECTED_BY_NA = 5
    REJECTED_BY_BOTH = 6

    def label(self) -> str:
        names = {
            RayStatus.ACTIVE: "Active",
            RayStatus.CLIPPED: "Clipped",
            RayStatus.MISSED: "Missed",
            RayStatus.ACCEPTED_BY_FIBER: "Accepted by Fiber",
            RayStatus.REJECTED_BY_POSITION: "Rejected by Position",
            RayStatus.REJECTED_BY_NA: "Rejected by NA",
            RayStatus.REJECTED_BY_BOTH: "Rejected by Both",
        }
        return names.get(self, "Unknown")


class RayBundle:
    """
    Vectorized representation of N optical rays.
    All lengths are in mm, angles in radians, power normalized or in Watts.
    """

    def __init__(
        self,
        r: np.ndarray,
        k: np.ndarray,
        power: Optional[np.ndarray] = None,
        wavelength: Optional[np.ndarray] = None,
        channel_id: Optional[np.ndarray] = None,
        status: Optional[np.ndarray] = None,
    ):
        """
        Initialize ray bundle.
        r: shape (N, 3), [x, y, z] positions in mm
        k: shape (N, 3), [kx, ky, kz] unit direction vectors
        power: shape (N,), optical power per ray
        wavelength: shape (N,), wavelength in nm (optional, default 550 nm)
        channel_id: shape (N,), slicer channel ID (-1 = unassigned/direct)
        status: shape (N,), integer status from RayStatus
        """
        self.r = np.asarray(r, dtype=np.float64)
        if self.r.ndim == 1:
            self.r = self.r.reshape(1, 3)
        self.n_rays = self.r.shape[0]

        self.k = np.asarray(k, dtype=np.float64)
        if self.k.ndim == 1:
            self.k = self.k.reshape(1, 3)
        self._normalize_k()

        if power is None:
            self.power = np.full(self.n_rays, 1.0 / max(self.n_rays, 1), dtype=np.float64)
        else:
            self.power = np.asarray(power, dtype=np.float64)

        if wavelength is None:
            self.wavelength = np.full(self.n_rays, 550.0, dtype=np.float64)
        else:
            self.wavelength = np.asarray(wavelength, dtype=np.float64)

        if channel_id is None:
            self.channel_id = np.full(self.n_rays, -1, dtype=np.int32)
        else:
            self.channel_id = np.asarray(channel_id, dtype=np.int32)

        if status is None:
            self.status = np.full(self.n_rays, RayStatus.ACTIVE, dtype=np.int32)
        else:
            self.status = np.asarray(status, dtype=np.int32)

        # Snapshots list storing dictionary of:
        # {'label': str, 'r': np.ndarray, 'status': np.ndarray}
        self.history: List[Dict[str, Any]] = []

    def _normalize_k(self) -> None:
        """Ensure all direction vectors have unit length."""
        norms = np.linalg.norm(self.k, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.k = self.k / norms

    @property
    def x(self) -> np.ndarray:
        return self.r[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.r[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.r[:, 2]

    @property
    def kx(self) -> np.ndarray:
        return self.k[:, 0]

    @property
    def ky(self) -> np.ndarray:
        return self.k[:, 1]

    @property
    def kz(self) -> np.ndarray:
        return self.k[:, 2]

    @property
    def active_mask(self) -> np.ndarray:
        return self.status == RayStatus.ACTIVE

    @property
    def accepted_mask(self) -> np.ndarray:
        return self.status == RayStatus.ACCEPTED_BY_FIBER

    @property
    def active_power(self) -> float:
        return float(np.sum(self.power[self.active_mask]))

    @property
    def total_power(self) -> float:
        return float(np.sum(self.power))

    @property
    def trajectories(self) -> List[np.ndarray]:
        """
        Reconstruct 3D polyline trajectory for each ray across recorded snapshots.
        Returns a list of length n_rays, where each element is an array of shape (N_snapshots, 3).
        """
        if not self.history:
            return [self.r[i : i + 1].copy() for i in range(self.n_rays)]
        n_planes = len(self.history)
        trajs = []
        for r_i in range(self.n_rays):
            pts = np.array([self.history[p]["r"][r_i] for p in range(n_planes)], dtype=np.float64)
            trajs.append(pts)
        return trajs

    def clone(self) -> RayBundle:
        """Create an independent deep copy of this RayBundle."""
        bundle = RayBundle(
            r=self.r.copy(),
            k=self.k.copy(),
            power=self.power.copy(),
            wavelength=self.wavelength.copy(),
            channel_id=self.channel_id.copy(),
            status=self.status.copy(),
        )
        bundle.history = [
            {
                "label": h["label"],
                "r": h["r"].copy(),
                "status": h["status"].copy(),
                "channel_id": h["channel_id"].copy() if "channel_id" in h else None,
            }
            for h in self.history
        ]
        return bundle

    def record_snapshot(self, label: str) -> None:
        """Record current spatial positions for ray trace visualization."""
        self.history.append({
            "label": label,
            "r": self.r.copy(),
            "status": self.status.copy(),
            "channel_id": self.channel_id.copy(),
        })

    def propagate_distance(self, dist: float | np.ndarray) -> None:
        """Propagate active rays along their direction vectors by given distance."""
        mask = self.active_mask
        if isinstance(dist, np.ndarray):
            self.r[mask] += self.k[mask] * dist[mask, np.newaxis]
        else:
            self.r[mask] += self.k[mask] * dist

    def propagate_to_z(self, target_z: float) -> None:
        """
        Propagate active rays to a planar surface at z = target_z perpendicular to z-axis.
        Rays traveling parallel to the plane (kz == 0) are marked MISSED.
        """
        mask = self.active_mask
        kz = self.k[mask, 2]
        valid_kz = np.abs(kz) > 1e-12

        # Rays with kz ≈ 0 cannot reach target z
        active_indices = np.where(mask)[0]
        invalid_indices = active_indices[~valid_kz]
        if len(invalid_indices) > 0:
            self.status[invalid_indices] = RayStatus.MISSED

        valid_indices = active_indices[valid_kz]
        if len(valid_indices) > 0:
            dz = target_z - self.r[valid_indices, 2]
            t = dz / self.k[valid_indices, 2]
            self.r[valid_indices] += self.k[valid_indices] * t[:, np.newaxis]
            # Ensure exact target_z to avoid floating point drift
            self.r[valid_indices, 2] = target_z
