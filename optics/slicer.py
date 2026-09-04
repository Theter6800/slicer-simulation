"""
Image Slicer array and slice mirror implementation with 3D branching geometry.
Inspired by Ellen Lee's image slicer IFU principles.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from .ray import RayBundle, RayStatus
from .elements import OpticalElement, rotation_matrix_xyz
from .geometry3d import normalize, reflect_vector, reflection_bisector


@dataclass
class SliceMirror:
    """An individual mirror slice in an image slicer array with 3D pose."""
    slice_id: int
    width: float              # mm (transverse x dimension)
    height: float             # mm (transverse y dimension)
    center_x: float           # mm
    center_y: float           # mm
    z: float                  # mm (piston position)
    tip_x_deg: float = 0.0    # deg (rotation about x, steers reflected ray along y)
    tilt_y_deg: float = 0.0   # deg (rotation about y, steers reflected ray along x)
    rot_z_deg: float = 0.0    # deg (rotation about z)
    focal_length: Optional[float] = None  # mm (optional powered mirror)
    enabled: bool = True
    _custom_normal: Optional[np.ndarray] = None

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y, self.z], dtype=np.float64)

    @property
    def normal(self) -> np.ndarray:
        """Surface normal in global coordinates (nominal un-tilted points towards -z)."""
        if self._custom_normal is not None:
            return normalize(self._custom_normal)
        R = rotation_matrix_xyz(
            np.radians(self.tip_x_deg),
            np.radians(self.tilt_y_deg),
            np.radians(self.rot_z_deg),
        )
        n0 = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        n = R @ n0
        return normalize(n)

    def aim_at_pupil(
        self,
        pupil_center: np.ndarray,
        k_in: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> np.ndarray:
        """
        Analytically calculate and set the slicer normal so that the incoming chief ray k_in
        reflects directly toward pupil_center:
        k_target = normalize(P_i - S_i)
        n_s = reflection_bisector(k_in, k_target)
        """
        p_arr = np.asarray(pupil_center, dtype=np.float64).reshape(3)
        diff = p_arr - self.center
        k_target = normalize(diff)
        n_s = reflection_bisector(k_in, k_target)
        self._custom_normal = n_s

        # Update tip_x and tilt_y angles for user introspection
        ny = float(np.clip(n_s[1], -1.0, 1.0))
        tip_x_rad = float(np.arcsin(ny))
        cos_tip = np.cos(tip_x_rad)
        if np.abs(cos_tip) > 1e-6:
            tilt_y_rad = float(np.arctan2(-n_s[0], -n_s[2]))
        else:
            tilt_y_rad = 0.0

        self.tip_x_deg = float(np.degrees(tip_x_rad))
        self.tilt_y_deg = float(np.degrees(tilt_y_rad))
        return n_s

    def set_tilt(self, tip_x_deg: float, tilt_y_deg: float, rot_z_deg: float = 0.0) -> None:
        """Manually set mirror tip, tilt, and roll angles, updating normal vector."""
        self.tip_x_deg = float(tip_x_deg)
        self.tilt_y_deg = float(tilt_y_deg)
        self.rot_z_deg = float(rot_z_deg)
        R = rotation_matrix_xyz(
            np.radians(self.tip_x_deg),
            np.radians(self.tilt_y_deg),
            np.radians(self.rot_z_deg),
        )
        n0 = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._custom_normal = normalize(R @ n0)

    def get_ideal_angles_for_target(
        self,
        pupil_center: np.ndarray,
        k_in: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> Tuple[float, float]:
        """
        Analytically calculate the ideal (tip_x, tilt_y) angles in degrees
        required to deflect chief ray k_in directly toward pupil_center.
        """
        p_arr = np.asarray(pupil_center, dtype=np.float64).reshape(3)
        diff = p_arr - self.center
        k_target = normalize(diff)
        n_ideal = reflection_bisector(k_in, k_target)

        ny = float(np.clip(n_ideal[1], -1.0, 1.0))
        tip_x_rad = float(np.arcsin(ny))
        cos_tip = np.cos(tip_x_rad)
        if np.abs(cos_tip) > 1e-6:
            tilt_y_rad = float(np.arctan2(-n_ideal[0], -n_ideal[2]))
        else:
            tilt_y_rad = 0.0

        return float(np.degrees(tip_x_rad)), float(np.degrees(tilt_y_rad))

    def get_reflection_diagnostics(
        self,
        k_in: np.ndarray = np.array([0.0, 0.0, 1.0]),
        target_pos: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Calculate full reflection diagnostics including incident and reflection angles,
        and optional pointing error toward target_pos.
        """
        k_in_u = normalize(np.asarray(k_in, dtype=np.float64).reshape(3))
        n = self.normal
        k_ref = reflect_vector(k_in_u, n)

        # Angle of incidence: angle between -k_in and n
        cos_i = np.clip(np.dot(-k_in_u, n), -1.0, 1.0)
        theta_i_deg = float(np.degrees(np.arccos(cos_i)))

        # Angle of reflection: angle between k_ref and n
        cos_r = np.clip(np.dot(k_ref, n), -1.0, 1.0)
        theta_r_deg = float(np.degrees(np.arccos(cos_r)))

        verified = bool(np.isclose(theta_i_deg, theta_r_deg, atol=1e-4))

        pointing_error_deg = 0.0
        if target_pos is not None:
            v_t = normalize(np.asarray(target_pos, dtype=np.float64).reshape(3) - self.center)
            cos_t = np.clip(np.dot(k_ref, v_t), -1.0, 1.0)
            pointing_error_deg = float(np.degrees(np.arccos(cos_t)))

        return {
            "k_in": k_in_u,
            "normal": n,
            "k_ref": k_ref,
            "tip_x_deg": self.tip_x_deg,
            "tilt_y_deg": self.tilt_y_deg,
            "rot_z_deg": self.rot_z_deg,
            "theta_i_deg": theta_i_deg,
            "theta_r_deg": theta_r_deg,
            "pointing_error_deg": pointing_error_deg,
            "verified": verified,
        }

    def get_reflected_chief_ray(self, k_in: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
        """Compute the reflected chief ray direction from this slice."""
        return reflect_vector(k_in, self.normal)

    def compute_pupil_position(self, distance: float, k_in: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
        """
        Calculate the 3D position of the associated pupil mirror placed along the reflected ray path:
        P_i = S_i + L_i * k_ref_i
        """
        k_ref = self.get_reflected_chief_ray(k_in)
        return self.center + distance * k_ref


class SlicerArray(OpticalElement):
    """
    Slicer array containing multiple SliceMirror elements.
    Places rectangular/square mirrors at the input focal plane,
    slicing the extended image into field channels and steering each channel along its own 3D branch.
    """

    def __init__(
        self,
        name: str = "ImageSlicer",
        z: float = 200.0,
        layout: str = "2x2",
        slice_width: float = 10.0,
        slice_height: float = 10.0,
        gap_x: float = 0.5,
        gap_y: float = 0.5,
        n_slices: Optional[int] = None,
        custom_positions: Optional[List[Tuple[float, float]]] = None,
        enabled: bool = True,
    ):
        super().__init__(name=name, z=z, enabled=enabled)
        self.layout = layout
        self.slice_width = float(slice_width)
        self.slice_height = float(slice_height)
        self.gap_x = float(gap_x)
        self.gap_y = float(gap_y)
        self.n_slices = n_slices
        self.custom_positions = custom_positions
        self.slices: List[SliceMirror] = []
        self._build_layout()

    def _build_layout(self) -> None:
        """Construct the slice mirrors based on the selected layout."""
        self.slices.clear()
        w = self.slice_width
        h = self.slice_height
        gx = self.gap_x
        gy = self.gap_y
        pitch_x = w + gx
        pitch_y = h + gy

        if self.custom_positions is not None:
            for i, (cx, cy) in enumerate(self.custom_positions):
                self.slices.append(
                    SliceMirror(
                        slice_id=i,
                        width=w,
                        height=h,
                        center_x=float(cx),
                        center_y=float(cy),
                        z=self.z,
                        tip_x_deg=0.0,
                        tilt_y_deg=0.0,
                    )
                )
            return

        # Determine count if explicit n_slices provided or layout is integer
        num_s = None
        if self.n_slices is not None:
            num_s = int(self.n_slices)
        elif self.layout.isdigit():
            num_s = int(self.layout)
        elif self.layout.endswith("-channel"):
            try:
                num_s = int(self.layout.split("-")[0])
            except Exception:
                num_s = None

        if num_s is not None and num_s > 0:
            if num_s == 1:
                self.slices.append(
                    SliceMirror(slice_id=0, width=w, height=h, center_x=0.0, center_y=0.0, z=self.z)
                )
            elif num_s == 4 and self.layout == "2x2":
                configs = [
                    (0, -pitch_x / 2.0,  pitch_y / 2.0,  2.0, -2.0),
                    (1,  pitch_x / 2.0,  pitch_y / 2.0,  2.0,  2.0),
                    (2, -pitch_x / 2.0, -pitch_y / 2.0, -2.0, -2.0),
                    (3,  pitch_x / 2.0, -pitch_y / 2.0, -2.0,  2.0),
                ]
                for s_id, cx, cy, tip_x, tilt_y in configs:
                    self.slices.append(
                        SliceMirror(slice_id=s_id, width=w, height=h, center_x=cx, center_y=cy, z=self.z, tip_x_deg=tip_x, tilt_y_deg=tilt_y)
                    )
            else:
                # Linear stack along y
                y_offsets = [(- (num_s - 1) / 2.0 + i) * pitch_y for i in range(num_s)]
                for i, cy in enumerate(y_offsets):
                    self.slices.append(
                        SliceMirror(
                            slice_id=i,
                            width=w,
                            height=h,
                            center_x=0.0,
                            center_y=cy,
                            z=self.z,
                            tip_x_deg=0.0,
                            tilt_y_deg=0.0,
                        )
                    )
            return

        if self.layout == "1x2" or self.layout == "2-channel":
            # 2 Slices (Top and Bottom or Left and Right)
            # S1 tilted up (+y), S2 tilted down (-y)
            self.slices.append(
                SliceMirror(
                    slice_id=0,
                    width=w,
                    height=h,
                    center_x=0.0,
                    center_y=pitch_y / 2.0,
                    z=self.z,
                    tip_x_deg=2.5,   # steers upward in y
                    tilt_y_deg=0.0,
                )
            )
            self.slices.append(
                SliceMirror(
                    slice_id=1,
                    width=w,
                    height=h,
                    center_x=0.0,
                    center_y=-pitch_y / 2.0,
                    z=self.z,
                    tip_x_deg=-2.5,  # steers downward in y
                    tilt_y_deg=0.0,
                )
            )

        elif self.layout == "2x2":
            # 4 mirrors in 2x2 grid around (0, 0)
            # Outward 3D tilts so channels diverge cleanly in 3D
            configs = [
                # slice_id, cx, cy, tip_x_deg, tilt_y_deg
                (0, -pitch_x / 2.0,  pitch_y / 2.0,  2.0, -2.0),  # Top-left -> (+y, -x)
                (1,  pitch_x / 2.0,  pitch_y / 2.0,  2.0,  2.0),  # Top-right -> (+y, +x)
                (2, -pitch_x / 2.0, -pitch_y / 2.0, -2.0, -2.0),  # Bottom-left -> (-y, -x)
                (3,  pitch_x / 2.0, -pitch_y / 2.0, -2.0,  2.0),  # Bottom-right -> (-y, +x)
            ]
            for s_id, cx, cy, tip_x, tilt_y in configs:
                self.slices.append(
                    SliceMirror(
                        slice_id=s_id,
                        width=w,
                        height=h,
                        center_x=cx,
                        center_y=cy,
                        z=self.z,
                        tip_x_deg=tip_x,
                        tilt_y_deg=tilt_y,
                    )
                )

        elif self.layout == "1x4":
            y_offsets = [(-1.5 + i) * pitch_y for i in range(4)]
            tilts_x = [-3.0, -1.0, 1.0, 3.0]
            for i, cy in enumerate(y_offsets):
                self.slices.append(
                    SliceMirror(
                        slice_id=i,
                        width=w * 4.0,
                        height=h,
                        center_x=0.0,
                        center_y=cy,
                        z=self.z,
                        tip_x_deg=tilts_x[i],
                        tilt_y_deg=0.0,
                    )
                )

        else:
            # Single slice
            self.slices.append(
                SliceMirror(
                    slice_id=0,
                    width=w,
                    height=h,
                    center_x=0.0,
                    center_y=0.0,
                    z=self.z,
                    tip_x_deg=0.0,
                    tilt_y_deg=0.0,
                )
            )

    def set_slice_tilt(self, slice_id: int, tip_x_deg: float, tilt_y_deg: float) -> None:
        """Update tilt of an individual slice."""
        for s in self.slices:
            if s.slice_id == slice_id:
                s.tip_x_deg = float(tip_x_deg)
                s.tilt_y_deg = float(tilt_y_deg)
                break

    def set_slice_enabled(self, slice_id: int, enabled: bool) -> None:
        """Enable or disable an individual slice."""
        for s in self.slices:
            if s.slice_id == slice_id:
                s.enabled = bool(enabled)
                break

    def aim_all_at_pupils(
        self,
        pupil_positions: List[np.ndarray],
        k_in: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> None:
        """
        Aim each slicer mirror analytically at its assigned pupil position.
        """
        for i, s in enumerate(self.slices):
            if i < len(pupil_positions):
                s.aim_at_pupil(pupil_positions[i], k_in=k_in)

    def trace(self, bundle: RayBundle) -> None:
        """
        Trace bundle to the slicer plane, assign channel IDs, and reflect active rays.
        """
        if not self.enabled:
            return

        bundle.propagate_to_z(self.z)
        bundle.record_snapshot(f"{self.name} (Incident)")

        mask = bundle.active_mask
        active_indices = np.where(mask)[0]
        if len(active_indices) == 0:
            return

        x = bundle.x[active_indices]
        y = bundle.y[active_indices]

        assigned_slice = np.full(len(active_indices), -1, dtype=np.int32)

        for s in self.slices:
            if not s.enabled:
                continue

            half_w = s.width / 2.0
            half_h = s.height / 2.0

            in_slice = (
                (x >= s.center_x - half_w)
                & (x <= s.center_x + half_w)
                & (y >= s.center_y - half_h)
                & (y <= s.center_y + half_h)
            )
            assign_mask = in_slice & (assigned_slice == -1)
            assigned_slice[assign_mask] = s.slice_id

        # Rays striking the inter-slice gaps are CLIPPED
        unassigned_mask = assigned_slice == -1
        if np.any(unassigned_mask):
            bundle.status[active_indices[unassigned_mask]] = RayStatus.CLIPPED

        # Reflect rays for each slice along its 3D branch
        for s in self.slices:
            if not s.enabled:
                continue

            s_hit_mask = assigned_slice == s.slice_id
            if not np.any(s_hit_mask):
                continue

            sub_indices = active_indices[s_hit_mask]
            bundle.channel_id[sub_indices] = s.slice_id

            n = s.normal
            k_in = bundle.k[sub_indices]
            k_out = reflect_vector(k_in, n)

            # If mirror has power
            if s.focal_length is not None and s.focal_length != 0.0:
                dx_loc = bundle.x[sub_indices] - s.center_x
                dy_loc = bundle.y[sub_indices] - s.center_y
                kz_ref = k_out[:, 2]
                safe_kz = np.where(np.abs(kz_ref) < 1e-12, 1e-12, kz_ref)
                u = k_out[:, 0] / safe_kz - (dx_loc / s.focal_length)
                v = k_out[:, 1] / safe_kz - (dy_loc / s.focal_length)
                denom = np.sqrt(u**2 + v**2 + 1.0)
                sgn = np.sign(kz_ref)
                sgn[sgn == 0] = 1.0
                k_out[:, 0] = (u / denom) * sgn
                k_out[:, 1] = (v / denom) * sgn
                k_out[:, 2] = (1.0 / denom) * sgn

            bundle.k[sub_indices] = normalize(k_out)

        bundle.record_snapshot(f"{self.name} (Reflected Channels)")
