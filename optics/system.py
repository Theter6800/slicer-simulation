"""
OpticalSystem pipeline managing sequential pre-slicer optics and
non-sequential 3D branched post-slicer ray tracing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from .ray import RayBundle, RayStatus
from .elements import OpticalElement, ThinLens, CircularAperture, Surface3D, ThinLens3D
from .slicer import SlicerArray
from .pupil import PupilRelaySystem, PupilMirror
from .fiber import Fiber, FiberCouplingResult
from .metrics import SystemMetrics, SpotMetrics, AngularMetrics, compute_spot_metrics, compute_angular_metrics
from .power_accounting import PowerAccounting, SlicePowerShare


@dataclass
class OpticalStage:
    """Snapshot of system state at an optical interface."""
    name: str
    z: float
    active_power: float
    clipped_power: float
    total_power: float
    n_active_rays: int


@dataclass
class OpticalGeometry:
    """
    Single source of truth for optical bench positions and dimensions.
    Guarantees slicer z, image-plane z, 3D view, build sheet, and component table
    cannot disagree.
    """
    z_aperture: float = 20.0
    z_fore: float = 40.0
    fore_focal_length: float = 150.0
    pupil_distance_z: float = 40.0
    pupil_transverse_offset: float = 20.0
    condenser_distance_z: float = 60.0
    condenser_focal_length: float = 22.0
    fiber_distance: float = 22.0
    slicer_width: float = 10.0
    slicer_height: float = 10.0
    min_slicer_gap: float = 0.04

    @property
    def z_image(self) -> float:
        return float(self.z_fore + self.fore_focal_length)

    @property
    def z_slicer(self) -> float:
        return self.z_image

    @property
    def z_pupil(self) -> float:
        return float(self.z_slicer + self.pupil_distance_z)

    @property
    def z_condenser(self) -> float:
        return float(self.z_pupil + self.condenser_distance_z)

    @property
    def z_fiber(self) -> float:
        return float(self.z_condenser + self.fiber_distance)


class OpticalSystem:
    """
    Complete optical system supporting:
    1. Pre-slicer sequential optical train (LED -> Aperture -> Fore-optics Lenses).
    2. Slicer array branching into 3D field channels.
    3. Post-slicer non-sequential 3D ray tracing (Pupil Mirrors -> Common Final Lens -> Fiber).
    """

    def __init__(
        self,
        name: str = "3D Branched Optical Bench",
        fore_optics: Optional[List[OpticalElement]] = None,
        slicer: Optional[SlicerArray] = None,
        pupil_relay: Optional[PupilRelaySystem] = None,
        final_lens_3d: Optional[ThinLens3D] = None,
        coupling_optics: Optional[List[OpticalElement]] = None,
        fiber: Optional[Fiber] = None,
        is_non_sequential_post_slicer: bool = True,
        wrong_pupil_policy: str = "reject_as_stray",
        geometry: Optional[OpticalGeometry] = None,
    ):
        self.name = name
        self.fore_optics: List[OpticalElement] = fore_optics if fore_optics is not None else []
        self.slicer: Optional[SlicerArray] = slicer
        self.pupil_relay: Optional[PupilRelaySystem] = pupil_relay
        self.final_lens_3d: Optional[ThinLens3D] = final_lens_3d
        self.coupling_optics: List[OpticalElement] = coupling_optics if coupling_optics is not None else []
        self.fiber: Fiber = fiber if fiber is not None else Fiber()
        self.is_non_sequential_post_slicer = is_non_sequential_post_slicer
        self.wrong_pupil_policy = wrong_pupil_policy
        self.geometry = geometry

        self.stages: List[OpticalStage] = []
        self.cross_channel_hits: int = 0
        self.last_bundle: Optional[RayBundle] = None
        self.last_coupling_result: Optional[FiberCouplingResult] = None
        self.last_metrics: Optional[SystemMetrics] = None
        self.last_power_accounting: Optional[PowerAccounting] = None
        self.pre_slicer_rays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.pre_slicer_spot_metrics: Optional[SpotMetrics] = None

    @property
    def z_image_plane(self) -> float:
        """Strict single-source-of-truth image plane axial coordinate."""
        if self.slicer is not None:
            return float(self.slicer.z)
        if self.geometry is not None:
            return float(self.geometry.z_image)
        if self.fore_optics:
            for elem in reversed(self.fore_optics):
                if hasattr(elem, "focal_length"):
                    return float(elem.z + elem.focal_length)
            return float(self.fore_optics[-1].z + 50.0)
        return 190.0

    def trace(self, initial_bundle: RayBundle) -> Tuple[RayBundle, FiberCouplingResult, SystemMetrics]:
        bundle = initial_bundle.clone()
        self.stages.clear()
        self.cross_channel_hits = 0
        tot_power = bundle.total_power

        # 1. Source launched state
        self.stages.append(
            OpticalStage(
                name="Source Launched",
                z=bundle.z[0] if len(bundle.z) > 0 else 0.0,
                active_power=bundle.active_power,
                clipped_power=0.0,
                total_power=tot_power,
                n_active_rays=int(np.sum(bundle.active_mask)),
            )
        )

        # 2. Sequential Pre-Slicer Optics
        p_after_aperture = tot_power
        has_aperture = False
        for elem in self.fore_optics:
            elem.trace(bundle)
            active_p = bundle.active_power
            if "aperture" in elem.name.lower():
                p_after_aperture = active_p
                has_aperture = True
            self.stages.append(
                OpticalStage(
                    name=elem.name,
                    z=elem.z,
                    active_power=active_p,
                    clipped_power=tot_power - active_p,
                    total_power=tot_power,
                    n_active_rays=int(np.sum(bundle.active_mask)),
                )
            )
        if not has_aperture and len(self.fore_optics) > 0:
            p_after_aperture = bundle.active_power

        # 3. Propagate to Image Plane & Pre-Slicer Diagnostic
        z_slicer_plane = self.z_image_plane
        bundle.propagate_to_z(z_slicer_plane)
        p_on_image_plane = bundle.active_power

        # Compute spot metrics at image plane before slicing
        pre_slicer_metrics = compute_spot_metrics(bundle, z_plane=z_slicer_plane)
        act_mask = bundle.active_mask
        self.pre_slicer_rays = (bundle.x[act_mask].copy(), bundle.y[act_mask].copy(), bundle.power[act_mask].copy())
        self.pre_slicer_spot_metrics = pre_slicer_metrics

        # 4. Slicer Array: Spatial sectioning, gap loss, array miss, and 3D reflection
        slice_shares: Dict[int, SlicePowerShare] = {}
        p_intercepted_by_slicers = 0.0
        p_lost_at_slicer_gaps = 0.0
        p_missed_slicer_array = 0.0
        power_per_slice: Dict[int, float] = {}

        if self.slicer is not None and self.slicer.enabled and len(self.slicer.slices) > 0:
            # Initialize slice share tracking
            for s in self.slicer.slices:
                slice_shares[s.slice_id] = SlicePowerShare(slice_id=s.slice_id)

            # Determine bounding box of entire slicer array
            all_x_min = min(s.center_x - s.width / 2.0 for s in self.slicer.slices if s.enabled)
            all_x_max = max(s.center_x + s.width / 2.0 for s in self.slicer.slices if s.enabled)
            all_y_min = min(s.center_y - s.height / 2.0 for s in self.slicer.slices if s.enabled)
            all_y_max = max(s.center_y + s.height / 2.0 for s in self.slicer.slices if s.enabled)

            act_idx = np.where(bundle.active_mask)[0]
            x_act = bundle.x[act_idx]
            y_act = bundle.y[act_idx]
            p_act = bundle.power[act_idx]

            in_bbox = (x_act >= all_x_min) & (x_act <= all_x_max) & (y_act >= all_y_min) & (y_act <= all_y_max)
            outside_bbox = ~in_bbox
            p_missed_slicer_array = float(np.sum(p_act[outside_bbox]))

            # Trace through slicer array
            self.slicer.trace(bundle)
            active_p = bundle.active_power
            total_lost_at_slicer = p_on_image_plane - active_p

            p_lost_at_slicer_gaps = max(0.0, total_lost_at_slicer - p_missed_slicer_array)
            p_intercepted_by_slicers = p_on_image_plane - p_missed_slicer_array

            for s in self.slicer.slices:
                ch_mask = (bundle.channel_id == s.slice_id) & bundle.active_mask
                ch_p = float(np.sum(bundle.power[ch_mask]))
                power_per_slice[s.slice_id] = ch_p
                slice_shares[s.slice_id].p_incident = ch_p
                slice_shares[s.slice_id].p_reflected = ch_p

            self.stages.append(
                OpticalStage(
                    name=self.slicer.name,
                    z=self.slicer.z,
                    active_power=active_p,
                    clipped_power=tot_power - active_p,
                    total_power=tot_power,
                    n_active_rays=int(np.sum(bundle.active_mask)),
                )
            )
        else:
            p_intercepted_by_slicers = p_on_image_plane
            p_lost_at_slicer_gaps = 0.0
            p_missed_slicer_array = 0.0
            bundle.channel_id[bundle.active_mask] = 0
            slice_shares[0] = SlicePowerShare(
                slice_id=0,
                p_incident=p_on_image_plane,
                p_reflected=p_on_image_plane,
            )

        # 5. Non-Sequential or Sequential Post-Slicer Tracing
        if self.is_non_sequential_post_slicer and (self.pupil_relay is not None or self.final_lens_3d is not None):
            coupling_res = self._trace_non_sequential_post_slicer(
                bundle,
                tot_power,
                p_after_aperture=p_after_aperture,
                p_on_image_plane=p_on_image_plane,
                p_intercepted_by_slicers=p_intercepted_by_slicers,
                p_lost_at_slicer_gaps=p_lost_at_slicer_gaps,
                p_missed_slicer_array=p_missed_slicer_array,
                slice_shares=slice_shares,
            )
        else:
            coupling_res = self._trace_sequential_post_slicer(
                bundle,
                tot_power,
                p_after_aperture=p_after_aperture,
                p_on_image_plane=p_on_image_plane,
                p_intercepted_by_slicers=p_intercepted_by_slicers,
                p_lost_at_slicer_gaps=p_lost_at_slicer_gaps,
                p_missed_slicer_array=p_missed_slicer_array,
                slice_shares=slice_shares,
            )

        # 6. Calculate Channel Statistics
        per_channel_stats: Dict[int, Dict[str, float]] = {}
        unique_channels = np.unique(bundle.channel_id)
        for ch in unique_channels:
            if ch < 0 and len(unique_channels) > 1:
                continue
            ch_mask = bundle.channel_id == ch
            ch_power = float(np.sum(bundle.power[ch_mask]))
            ch_accepted = float(np.sum(bundle.power[ch_mask & bundle.accepted_mask]))
            ch_spatial = float(
                np.sum(
                    bundle.power[
                        ch_mask
                        & (
                            (bundle.status == RayStatus.ACCEPTED_BY_FIBER)
                            | (bundle.status == RayStatus.REJECTED_BY_NA)
                        )
                    ]
                )
            )
            ch_na = float(
                np.sum(
                    bundle.power[
                        ch_mask
                        & (
                            (bundle.status == RayStatus.ACCEPTED_BY_FIBER)
                            | (bundle.status == RayStatus.REJECTED_BY_POSITION)
                        )
                    ]
                )
            )
            per_channel_stats[int(ch)] = {
                "intercepted_power": ch_power,
                "accepted_power": ch_accepted,
                "spatial_accepted_power": ch_spatial,
                "na_accepted_power": ch_na,
                "coupling_efficiency": ch_accepted / max(ch_power, 1e-12),
            }

        # 7. Optical Loss Budget Table
        loss_budget = []
        for i, st in enumerate(self.stages):
            prev_p = self.stages[i - 1].active_power if i > 0 else st.total_power
            stage_loss = max(prev_p - st.active_power, 0.0)
            stage_trans = st.active_power / prev_p if prev_p > 0 else 0.0
            cum_trans = st.active_power / tot_power if tot_power > 0 else 0.0
            loss_budget.append({
                "Stage": st.name,
                "z (mm)": st.z,
                "Surviving Power": st.active_power,
                "Power Lost": stage_loss,
                "Stage Transmission": stage_trans,
                "Cumulative Transmission": cum_trans,
                "Active Rays": st.n_active_rays,
            })

        p_plane = coupling_res.power_reaching_fiber_plane
        p_accepted = coupling_res.accepted_power
        p_spatial = p_accepted + float(
            np.sum(bundle.power[bundle.status == RayStatus.REJECTED_BY_NA])
        )
        p_na = p_accepted + float(
            np.sum(bundle.power[bundle.status == RayStatus.REJECTED_BY_POSITION])
        )

        frac_core = p_spatial / p_plane if p_plane > 0 else 0.0
        frac_na = p_na / p_plane if p_plane > 0 else 0.0
        frac_both = p_accepted / p_plane if p_plane > 0 else 0.0

        # Fiber face angular metrics
        fiber_active_mask = (bundle.status == RayStatus.ACCEPTED_BY_FIBER) | (bundle.status == RayStatus.REJECTED_BY_NA) | (bundle.status == RayStatus.REJECTED_BY_POSITION) | (bundle.status == RayStatus.REJECTED_BY_BOTH)
        if hasattr(coupling_res, "incidence_angle_rad") and len(coupling_res.incidence_angle_rad) > 0:
            ang_metrics = compute_angular_metrics(
                coupling_res.incidence_angle_rad,
                weights=bundle.power[bundle.active_mask] if np.any(bundle.active_mask) else None,
            )
        else:
            ang_metrics = AngularMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        metrics = SystemMetrics(
            launched_rays=bundle.n_rays,
            launched_power=tot_power,
            aperture_transmitted_power=self.stages[1].active_power if len(self.stages) > 1 else tot_power,
            slicer_intercepted_power=p_intercepted_by_slicers,
            slicer_gap_loss_power=p_lost_at_slicer_gaps,
            power_per_slice=power_per_slice,
            power_reaching_coupling_optic=coupling_res.power_reaching_coupling_optic,
            power_reaching_fiber_plane=p_plane,
            fiber_accepted_power=p_accepted,
            fiber_rejected_position_power=p_plane - p_spatial,
            fiber_rejected_na_power=p_plane - p_na,
            fiber_rejected_both_power=float(np.sum(bundle.power[bundle.status == RayStatus.REJECTED_BY_BOTH])),
            geometric_coupling_efficiency=coupling_res.geometric_coupling_efficiency,
            efficiency_rel_coupling_optic=coupling_res.efficiency_rel_coupling_optic,
            efficiency_rel_fiber_plane=coupling_res.efficiency_rel_fiber_plane,
            fraction_inside_core=frac_core,
            fraction_inside_na=frac_na,
            fraction_satisfying_both=frac_both,
            spot_rms_radius=coupling_res.spot_rms_radius,
            encircled_80_diameter=coupling_res.encircled_80_diameter,
            mean_ray_angle_deg=coupling_res.mean_ray_angle_deg,
            max_ray_angle_deg=coupling_res.max_ray_angle_deg,
            per_channel_stats=per_channel_stats,
            loss_budget_table=loss_budget,
            pre_slicer_spot_metrics=pre_slicer_metrics,
            fiber_angular_metrics=ang_metrics,
        )

        self.last_bundle = bundle
        self.last_coupling_result = coupling_res
        self.last_metrics = metrics
        return bundle, coupling_res, metrics

    def _trace_non_sequential_post_slicer(
        self,
        bundle: RayBundle,
        tot_power: float,
        p_after_aperture: float = 0.0,
        p_on_image_plane: float = 0.0,
        p_intercepted_by_slicers: float = 0.0,
        p_lost_at_slicer_gaps: float = 0.0,
        p_missed_slicer_array: float = 0.0,
        slice_shares: Optional[Dict[int, SlicePowerShare]] = None,
        max_bounces: int = 5,
    ) -> FiberCouplingResult:
        """
        Non-sequential 3D ray tracing post-slicer:
        Finds the nearest positive surface intersection across all candidate mirrors
        simultaneously, separating correct pupil, cross-talk, missed pupils, and condenser clipping.
        """
        if slice_shares is None:
            slice_shares = {}

        p_on_correct_pupil = 0.0
        p_on_wrong_pupil = 0.0
        p_missed_all_pupils = 0.0
        p_on_condenser = 0.0
        p_missed_condenser = 0.0
        p_blocked_by_other_optics = 0.0

        # Step 1: Trace to Pupil Mirrors with True Nearest-Surface Solver
        if self.pupil_relay is not None and self.pupil_relay.enabled and self.pupil_relay.mirrors:
            active_idx = np.where(bundle.active_mask)[0]
            if len(active_idx) > 0:
                n_active = len(active_idx)
                min_t = np.full(n_active, np.inf, dtype=np.float64)
                best_mirror_idx = np.full(n_active, -1, dtype=np.int32)
                best_hit_pos = np.zeros((n_active, 3), dtype=np.float64)
                best_local_uv = np.zeros((n_active, 2), dtype=np.float64)

                r_curr = bundle.r[active_idx].copy()
                k_curr = bundle.k[active_idx].copy()

                for m_idx, m in enumerate(self.pupil_relay.mirrors):
                    if not m.enabled:
                        continue
                    t, hit_pos, local_uv, in_bounds = m.intersect(r_curr, k_curr)
                    valid_hit = in_bounds & (t > 1e-4) & (t < min_t)
                    if np.any(valid_hit):
                        min_t[valid_hit] = t[valid_hit]
                        best_mirror_idx[valid_hit] = m_idx
                        best_hit_pos[valid_hit] = hit_pos[valid_hit]
                        best_local_uv[valid_hit] = local_uv[valid_hit]

                has_hit = best_mirror_idx >= 0
                for m_idx, m in enumerate(self.pupil_relay.mirrors):
                    if not m.enabled:
                        continue
                    m_rays_mask = has_hit & (best_mirror_idx == m_idx)
                    if np.any(m_rays_mask):
                        sub_global_idx = active_idx[m_rays_mask]
                        hit_pts = best_hit_pos[m_rays_mask]
                        local_pts = best_local_uv[m_rays_mask]

                        # Check whether this hit matches the ray's assigned slicer channel
                        is_correct = (bundle.channel_id[sub_global_idx] == m.channel_id)
                        correct_sub = sub_global_idx[is_correct]
                        wrong_sub = sub_global_idx[~is_correct]

                        p_corr = float(np.sum(bundle.power[correct_sub]))
                        p_wrong = float(np.sum(bundle.power[wrong_sub]))

                        p_on_correct_pupil += p_corr
                        p_on_wrong_pupil += p_wrong
                        self.cross_channel_hits += len(wrong_sub)

                        if m.channel_id in slice_shares:
                            slice_shares[m.channel_id].p_correct_pupil += p_corr
                        for w_idx in wrong_sub:
                            ch_w = bundle.channel_id[w_idx]
                            if ch_w in slice_shares:
                                slice_shares[ch_w].p_wrong_pupil += float(bundle.power[w_idx])

                        if self.wrong_pupil_policy == "reject_as_stray":
                            if len(wrong_sub) > 0:
                                bundle.status[wrong_sub] = RayStatus.CLIPPED
                            if len(correct_sub) > 0:
                                m.interact(bundle, correct_sub, hit_pts[is_correct], local_pts[is_correct])
                        else:
                            # propagate_physically
                            m.interact(bundle, sub_global_idx, hit_pts, local_pts)

                # Rays that missed all pupil mirrors are marked CLIPPED
                missed_mask = ~has_hit
                if np.any(missed_mask):
                    missed_global_idx = active_idx[missed_mask]
                    p_missed_all_pupils += float(np.sum(bundle.power[missed_global_idx]))
                    bundle.status[missed_global_idx] = RayStatus.CLIPPED

            bundle.record_snapshot("Pupil Mirror Relays")
            self.stages.append(
                OpticalStage(
                    name="Pupil Mirrors Relay",
                    z=self.pupil_relay.mirrors[0].z if self.pupil_relay.mirrors else 0.0,
                    active_power=bundle.active_power,
                    clipped_power=tot_power - bundle.active_power,
                    total_power=tot_power,
                    n_active_rays=int(np.sum(bundle.active_mask)),
                )
            )
        else:
            p_on_correct_pupil = p_intercepted_by_slicers - p_lost_at_slicer_gaps

        # Step 2: Trace to Final 3D Condenser Coupling Lens
        power_reaching_coupling = bundle.active_power
        if self.final_lens_3d is not None and self.final_lens_3d.is_active:
            active_idx = np.where(bundle.active_mask)[0]
            if len(active_idx) > 0:
                t, hit_pos, local_uv, in_bounds = self.final_lens_3d.intersect(
                    bundle.r[active_idx],
                    bundle.k[active_idx],
                )
                lens_hits = in_bounds & (t > 1e-4)
                if np.any(lens_hits):
                    sub_idx = active_idx[lens_hits]
                    p_on_condenser = float(np.sum(bundle.power[sub_idx]))
                    for s_idx in sub_idx:
                        ch = bundle.channel_id[s_idx]
                        if ch in slice_shares:
                            slice_shares[ch].p_condenser += float(bundle.power[s_idx])

                    self.final_lens_3d.interact(bundle, sub_idx, hit_pos[lens_hits], local_uv[lens_hits])

                lens_missed = ~lens_hits
                if np.any(lens_missed):
                    missed_lens_idx = active_idx[lens_missed]
                    p_missed_condenser = float(np.sum(bundle.power[missed_lens_idx]))
                    bundle.status[missed_lens_idx] = RayStatus.CLIPPED

            bundle.record_snapshot("Final Coupling Lens")
            self.stages.append(
                OpticalStage(
                    name=self.final_lens_3d.name,
                    z=float(self.final_lens_3d.center[2]),
                    active_power=bundle.active_power,
                    clipped_power=tot_power - bundle.active_power,
                    total_power=tot_power,
                    n_active_rays=int(np.sum(bundle.active_mask)),
                )
            )
        else:
            p_on_condenser = bundle.active_power

        # Step 3: Trace to 3D Fiber
        # Capture mask of rays arriving at the fiber plane BEFORE evaluate_coupling updates statuses
        arriving_at_fiber_mask = bundle.active_mask.copy()
        coupling_res = self.fiber.evaluate_coupling(
            bundle,
            power_reaching_coupling_optic=power_reaching_coupling,
        )

        # Record per-slice power reaching fiber and accepted
        for ch, share in slice_shares.items():
            ch_fib_mask = (bundle.channel_id == ch) & arriving_at_fiber_mask
            share.p_fiber = float(np.sum(bundle.power[ch_fib_mask]))
            share.p_accepted = float(np.sum(bundle.power[(bundle.channel_id == ch) & (bundle.status == RayStatus.ACCEPTED_BY_FIBER)]))
            share.p_core = float(
                np.sum(
                    bundle.power[
                        (bundle.channel_id == ch)
                        & (
                            (bundle.status == RayStatus.ACCEPTED_BY_FIBER)
                            | (bundle.status == RayStatus.REJECTED_BY_NA)
                        )
                    ]
                )
            )
            share.p_na = float(
                np.sum(
                    bundle.power[
                        (bundle.channel_id == ch)
                        & (
                            (bundle.status == RayStatus.ACCEPTED_BY_FIBER)
                            | (bundle.status == RayStatus.REJECTED_BY_POSITION)
                        )
                    ]
                )
            )

        self.stages.append(
            OpticalStage(
                name="Fiber Plane",
                z=float(self.fiber.position[2]),
                active_power=coupling_res.power_reaching_fiber_plane,
                clipped_power=tot_power - coupling_res.power_reaching_fiber_plane,
                total_power=tot_power,
                n_active_rays=coupling_res.n_rays_reaching_plane,
            )
        )
        self.stages.append(
            OpticalStage(
                name="Fiber Coupled (Accepted)",
                z=float(self.fiber.position[2]),
                active_power=coupling_res.accepted_power,
                clipped_power=tot_power - coupling_res.accepted_power,
                total_power=tot_power,
                n_active_rays=coupling_res.n_accepted,
            )
        )

        # Assemble comprehensive 16-stage power accounting
        accounting = PowerAccounting(
            p_launch=tot_power,
            p_after_aperture=p_after_aperture,
            p_on_image_plane=p_on_image_plane,
            p_intercepted_by_slicers=p_intercepted_by_slicers,
            p_lost_at_slicer_gaps=p_lost_at_slicer_gaps,
            p_missed_slicer_array=p_missed_slicer_array,
            p_on_correct_pupil=p_on_correct_pupil,
            p_on_wrong_pupil=p_on_wrong_pupil,
            p_missed_all_pupils=p_missed_all_pupils,
            p_blocked_by_other_optics=p_blocked_by_other_optics,
            p_on_condenser=p_on_condenser,
            p_missed_condenser=p_missed_condenser,
            p_at_fiber_plane=coupling_res.power_reaching_fiber_plane,
            p_inside_core=coupling_res.power_inside_core,
            p_inside_na=coupling_res.power_inside_na,
            p_inside_core_and_na=coupling_res.power_inside_core_and_na,
            wrong_pupil_policy=self.wrong_pupil_policy,
            slice_shares=slice_shares,
        )
        coupling_res.power_accounting = accounting
        self.last_power_accounting = accounting

        return coupling_res

    def _trace_sequential_post_slicer(
        self,
        bundle: RayBundle,
        tot_power: float,
        p_after_aperture: float = 0.0,
        p_on_image_plane: float = 0.0,
        p_intercepted_by_slicers: float = 0.0,
        p_lost_at_slicer_gaps: float = 0.0,
        p_missed_slicer_array: float = 0.0,
        slice_shares: Optional[Dict[int, SlicePowerShare]] = None,
    ) -> FiberCouplingResult:
        """Sequential collinear post-slicer trace for baseline Presets 1 & 2."""
        if slice_shares is None:
            slice_shares = {}

        power_reaching_coupling = bundle.active_power
        p_on_condenser = power_reaching_coupling
        for elem in self.coupling_optics:
            elem.trace(bundle)
            active_p = bundle.active_power
            p_on_condenser = active_p
            self.stages.append(
                OpticalStage(
                    name=elem.name,
                    z=elem.z,
                    active_power=active_p,
                    clipped_power=tot_power - active_p,
                    total_power=tot_power,
                    n_active_rays=int(np.sum(bundle.active_mask)),
                )
            )

        # Step 3: Trace to Fiber
        arriving_at_fiber_mask = bundle.active_mask.copy()
        coupling_res = self.fiber.evaluate_coupling(
            bundle,
            power_reaching_coupling_optic=power_reaching_coupling,
        )

        # Record per-slice power reaching fiber and accepted
        for ch, share in slice_shares.items():
            ch_fib_mask = (bundle.channel_id == ch) & arriving_at_fiber_mask
            share.p_fiber = float(np.sum(bundle.power[ch_fib_mask]))
            share.p_accepted = float(np.sum(bundle.power[(bundle.channel_id == ch) & (bundle.status == RayStatus.ACCEPTED_BY_FIBER)]))
            share.p_core = float(
                np.sum(
                    bundle.power[
                        (bundle.channel_id == ch)
                        & (
                            (bundle.status == RayStatus.ACCEPTED_BY_FIBER)
                            | (bundle.status == RayStatus.REJECTED_BY_NA)
                        )
                    ]
                )
            )
            share.p_na = float(
                np.sum(
                    bundle.power[
                        (bundle.channel_id == ch)
                        & (
                            (bundle.status == RayStatus.ACCEPTED_BY_FIBER)
                            | (bundle.status == RayStatus.REJECTED_BY_POSITION)
                        )
                    ]
                )
            )

        self.stages.append(
            OpticalStage(
                name="Fiber Plane",
                z=float(self.fiber.position[2]),
                active_power=coupling_res.power_reaching_fiber_plane,
                clipped_power=tot_power - coupling_res.power_reaching_fiber_plane,
                total_power=tot_power,
                n_active_rays=coupling_res.n_rays_reaching_plane,
            )
        )
        self.stages.append(
            OpticalStage(
                name="Fiber Coupled (Accepted)",
                z=float(self.fiber.position[2]),
                active_power=coupling_res.accepted_power,
                clipped_power=tot_power - coupling_res.accepted_power,
                total_power=tot_power,
                n_active_rays=coupling_res.n_accepted,
            )
        )

        p_refl = p_intercepted_by_slicers - p_lost_at_slicer_gaps
        accounting = PowerAccounting(
            p_launch=tot_power,
            p_after_aperture=p_after_aperture,
            p_on_image_plane=p_on_image_plane,
            p_intercepted_by_slicers=p_intercepted_by_slicers,
            p_lost_at_slicer_gaps=p_lost_at_slicer_gaps,
            p_missed_slicer_array=p_missed_slicer_array,
            p_on_correct_pupil=p_refl,
            p_on_wrong_pupil=0.0,
            p_missed_all_pupils=0.0,
            p_blocked_by_other_optics=0.0,
            p_on_condenser=p_on_condenser,
            p_missed_condenser=0.0,
            p_at_fiber_plane=coupling_res.power_reaching_fiber_plane,
            p_inside_core=coupling_res.power_inside_core,
            p_inside_na=coupling_res.power_inside_na,
            p_inside_core_and_na=coupling_res.power_inside_core_and_na,
            wrong_pupil_policy=self.wrong_pupil_policy,
            slice_shares=slice_shares,
        )
        coupling_res.power_accounting = accounting
        self.last_power_accounting = accounting

        return coupling_res
