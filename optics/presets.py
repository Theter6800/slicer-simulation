"""
Laboratory hardware presets and experimental configurations.
Includes the branched non-sequential architecture:
Slicer -> Pupil Mirrors -> Common Final Lens -> Fiber.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from .elements import ThinLens, CircularAperture, ThinLens3D
from .slicer import SlicerArray, SliceMirror
from .pupil import PupilRelaySystem, PupilMirror
from .fiber import Fiber
from .system import OpticalSystem


def create_old_lens_system(
    aperture_diameter: float = 20.0,
    z_aperture: float = 30.0,
    z_l1: float = 60.0,        # f = 100 mm
    z_l2: float = 140.0,       # f = 35 mm
    z_l3: float = 210.0,       # f = 75 mm
    z_fiber: float = 285.0,    # fiber face
    fiber_core_diameter: float = 1.0,
    fiber_na: float = 0.22,
) -> OpticalSystem:
    """
    PRESET 1: "Old lens-only system"
    LED -> aperture -> L1(f=100) -> L2(f=35) -> L3(f=75) -> Fiber
    NOTE: All distances between elements are placeholders / user input required.
    """
    aperture = CircularAperture(
        name="Aperture (placeholder / user input required)",
        z=z_aperture,
        diameter=aperture_diameter,
    )
    lens1 = ThinLens(
        name="Lens L1 f=100mm (placeholder / user input required)",
        z=z_l1,
        focal_length=100.0,
        diameter=25.4,
    )
    lens2 = ThinLens(
        name="Lens L2 f=35mm (placeholder / user input required)",
        z=z_l2,
        focal_length=35.0,
        diameter=25.4,
    )
    lens3 = ThinLens(
        name="Lens L3 f=75mm (placeholder / user input required)",
        z=z_l3,
        focal_length=75.0,
        diameter=25.4,
    )
    fiber = Fiber(
        core_diameter=fiber_core_diameter,
        na=fiber_na,
        position=(0.0, 0.0, z_fiber),
        axis=(0.0, 0.0, 1.0),
    )

    return OpticalSystem(
        name="Preset 1: Old Lens-Only System",
        fore_optics=[aperture, lens1, lens2, lens3],
        slicer=None,
        pupil_relay=None,
        coupling_optics=[],
        fiber=fiber,
        is_non_sequential_post_slicer=False,
    )


def create_slicer_condenser_system(
    aperture_diameter: float = 20.0,
    z_aperture: float = 30.0,
    z_fore1: float = 60.0,     # f = 100 mm fore-optic
    z_fore2: float = 140.0,    # f = 35 mm fore-optic
    z_slicer: float = 190.0,   # Slicer at enlarged input focal plane
    slice_size: float = 10.0,  # ~10 mm square mirrors
    slice_gap: float = 0.5,    # 0.5 mm inter-slice gap
    z_condenser: float = 230.0,# f = 75 mm common condenser
    z_fiber: float = 305.0,    # fiber face
    fiber_core_diameter: float = 1.0,
    fiber_na: float = 0.22,
    active_slices: int = 4,    # number of enabled slices (1 to 4)
) -> OpticalSystem:
    """
    PRESET 2: "Slicer + common condenser"
    LED -> aperture -> fore-optics(f=100, f=35) -> enlarged input image -> 2x2 slicer -> f75 condenser -> Fiber
    NOTE: Distances are placeholders / user input required.
    """
    aperture = CircularAperture(
        name="Aperture (placeholder / user input required)",
        z=z_aperture,
        diameter=aperture_diameter,
    )
    fore1 = ThinLens(
        name="Fore-Optic L1 f=100mm (placeholder / user input required)",
        z=z_fore1,
        focal_length=100.0,
        diameter=25.4,
    )
    fore2 = ThinLens(
        name="Fore-Optic L2 f=35mm (placeholder / user input required)",
        z=z_fore2,
        focal_length=35.0,
        diameter=25.4,
    )

    slicer = SlicerArray(
        name="2x2 Image Slicer Array (~10mm mirrors)",
        z=z_slicer,
        layout="2x2",
        slice_width=slice_size,
        slice_height=slice_size,
        gap_x=slice_gap,
        gap_y=slice_gap,
    )
    for i, s in enumerate(slicer.slices):
        s.enabled = (i < active_slices)

    condenser = ThinLens(
        name="Common Condenser L3 f=75mm (placeholder / user input required)",
        z=z_condenser,
        focal_length=75.0,
        diameter=50.0,
    )

    fiber = Fiber(
        core_diameter=fiber_core_diameter,
        na=fiber_na,
        position=(0.0, 0.0, z_fiber),
        axis=(0.0, 0.0, 1.0),
    )

    return OpticalSystem(
        name="Preset 2: Slicer + Common Condenser",
        fore_optics=[aperture, fore1, fore2],
        slicer=slicer,
        pupil_relay=None,
        coupling_optics=[condenser],
        fiber=fiber,
        is_non_sequential_post_slicer=False,
    )


from .pupil import PupilRelaySystem, PupilMirror, generate_one_sided_pupil_positions
from .geometry3d import normalize, reflection_bisector


def create_branched_slicer_system(
    n_channels: int = 2,       # Arbitrary discrete channels: 1, 2, 3, 4, 5, 6, ...
    pupil_layout_side: str = "lower",  # "lower", "upper", "left", "right"
    pupil_aim_mode: str = "parallel",  # "parallel" (optimal for condenser focus) or "target_center"
    aperture_diameter: float = 20.0,
    z_aperture: float = 30.0,
    fore_lens_focal_length: Optional[float] = 150.0,  # Objective lens f; None falls back to z_fore1/z_fore2
    fore_lens_z: Optional[float] = None,              # Defaults to z_slicer - fore_lens_focal_length
    z_fore1: float = 60.0,     # Legacy fallback f = 100 mm fore-optic
    z_fore2: float = 140.0,    # Legacy fallback f = 35 mm fore-optic
    z_slicer: float = 190.0,   # Slicer at enlarged input focal plane (strictly on image plane)
    slice_size: float = 10.0,
    slice_width: Optional[float] = None,
    slice_height: Optional[float] = None,
    slice_gap: float = 0.5,
    slicer_positions: Optional[List[Tuple[float, float]]] = None,
    pupil_positions: Optional[List[np.ndarray | Tuple[float, float, float]]] = None,
    pupil_distance_z: float = 60.0,  # Distance in z from slicer to pupil mirrors
    pupil_transverse_offset: float = 25.0,  # Transverse offset to chosen side
    pupil_spacing: float = 14.0,     # Spacing between pupil mirrors
    pupil_mirror_size: float = 12.0,
    pupil_mirror_height: Optional[float] = None,
    pupil_focal_length: Optional[float] = 75.0,  # powered pupil mirror focal length (None for flat)
    condenser_distance_z: float = 80.0,  # Distance in z from pupils to condenser lens
    condenser_center: Optional[Tuple[float, float, float] | np.ndarray] = None,
    condenser_focal_length: float = 75.0,
    condenser_diameter: float = 50.0,
    fiber_core_diameter: float = 1.0,
    fiber_na: float = 0.22,
    fiber_distance: Optional[float] = None,
) -> OpticalSystem:
    """
    PRESET 3: "Asymmetric One-Sided Branched Slicer → Pupil Relay → Common Relay Axis → Condenser → Fiber"

    Pre-slicer region: Sequential on nominal z-axis.
    - An objective fore-optics lens (f = fore_lens_focal_length) forms a broad image
      of the source directly on the image plane at z = z_slicer = z_fore + f.
    Post-slicer region: Asymmetric, one-sided 3D branched geometry for arbitrary N channels.
    - All slicers reflect toward the SAME chosen side (lower, upper, left, right).
    - Each slice S_i is actively aimed at its assigned pupil mirror P_i.
    - Pupil mirrors redirect beams toward a NEW COMMON RELAY AXIS.
    - Common condenser lens is placed on the new relay axis and converges into fiber.
    """
    aperture = CircularAperture(
        name="Aperture",
        z=z_aperture,
        diameter=aperture_diameter,
    )
    if fore_lens_focal_length is not None:
        z_fore = fore_lens_z if fore_lens_z is not None else (z_slicer - fore_lens_focal_length)
        fore_lens = ThinLens(
            name=f"Fore-Optic Objective Lens f={fore_lens_focal_length:.0f}mm",
            z=z_fore,
            focal_length=fore_lens_focal_length,
            diameter=25.4,
        )
        fore_optics_list = [aperture, fore_lens]
    else:
        fore1 = ThinLens(
            name="Fore-Optic L1 f=100mm",
            z=z_fore1,
            focal_length=100.0,
            diameter=25.4,
        )
        fore2 = ThinLens(
            name="Fore-Optic L2 f=35mm",
            z=z_fore2,
            focal_length=35.0,
            diameter=25.4,
        )
        fore_optics_list = [aperture, fore1, fore2]

    actual_n = len(slicer_positions) if slicer_positions is not None else n_channels
    sw = slice_width if slice_width is not None else slice_size
    sh = slice_height if slice_height is not None else slice_size
    layout_str = "1x2" if actual_n == 2 else ("2x2" if actual_n == 4 and slicer_positions is None else f"{actual_n}-channel")

    slicer = SlicerArray(
        name="Asymmetric Image Slicer Array",
        z=z_slicer,
        layout=layout_str,
        slice_width=sw,
        slice_height=sh,
        gap_x=slice_gap,
        gap_y=slice_gap,
        n_slices=actual_n,
        custom_positions=slicer_positions,
    )

    # 1. Determine Pupil Mirror positions on ONE chosen side
    z_pupil = z_slicer + pupil_distance_z
    if pupil_positions is None:
        pupil_positions = generate_one_sided_pupil_positions(
            side=pupil_layout_side,
            n_channels=actual_n,
            z_pupil=z_pupil,
            transverse_offset=pupil_transverse_offset,
            spacing=pupil_spacing,
        )
    else:
        pupil_positions = [np.asarray(p, dtype=np.float64).reshape(3) for p in pupil_positions]

    # 2. Slicers aim towards the pupils analytically
    for i, s in enumerate(slicer.slices):
        if i < len(pupil_positions):
            s.aim_at_pupil(pupil_positions[i], k_in=np.array([0.0, 0.0, 1.0]))

    # 3. Define NEW COMMON RELAY AXIS
    p_centroid = np.mean(pupil_positions, axis=0)
    z_condenser = z_pupil + condenser_distance_z

    if condenser_center is None:
        side_clean = pupil_layout_side.lower().strip()
        offset_frac = 0.8 * pupil_transverse_offset
        if side_clean == "lower":
            c_lens = np.array([0.0, -offset_frac, z_condenser], dtype=np.float64)
        elif side_clean == "upper":
            c_lens = np.array([0.0, offset_frac, z_condenser], dtype=np.float64)
        elif side_clean == "left":
            c_lens = np.array([-offset_frac, 0.0, z_condenser], dtype=np.float64)
        elif side_clean == "right":
            c_lens = np.array([offset_frac, 0.0, z_condenser], dtype=np.float64)
        else:
            c_lens = np.array([0.0, -offset_frac, z_condenser], dtype=np.float64)
    else:
        c_lens = np.asarray(condenser_center, dtype=np.float64).reshape(3)

    # Unit direction along the new relay axis
    a_relay = normalize(c_lens - p_centroid)

    # Condenser lens on the new relay axis
    condenser_lens = ThinLens3D(
        name=f"Common Condenser Lens f={condenser_focal_length:.0f}mm",
        center=c_lens,
        normal=a_relay,
        focal_length=condenser_focal_length,
        diameter=condenser_diameter,
    )

    # Fiber placed along the new relay axis at the condenser focal distance
    d_fib = fiber_distance if fiber_distance is not None else condenser_focal_length
    fiber_center = c_lens + d_fib * a_relay
    fiber = Fiber(
        core_diameter=fiber_core_diameter,
        na=fiber_na,
        position=fiber_center,
        axis=a_relay,
    )

    # 4. Create 3D Pupil Mirrors and aim them toward new relay axis
    pupil_relay = PupilRelaySystem(name="3D One-Sided Pupil Relay")
    pm_h = pupil_mirror_height if pupil_mirror_height is not None else pupil_mirror_size
    for i, p_pos in enumerate(pupil_positions):
        s = slicer.slices[i]
        k_slicer_to_pupil = normalize(p_pos - s.center)
        pm = PupilMirror(
            channel_id=s.slice_id,
            name=f"Pupil Mirror P{s.slice_id + 1}",
            center=p_pos,
            normal=np.array([0.0, 0.0, -1.0]),
            width=pupil_mirror_size,
            height=pm_h,
            focal_length=pupil_focal_length,
            enabled=True,
        )
        if pupil_aim_mode == "parallel":
            n_p = reflection_bisector(k_slicer_to_pupil, a_relay)
            pm.set_pose(p_pos, n_p)
        else:
            pm.aim_towards(k_slicer_to_pupil, c_lens)
        pupil_relay.add_mirror(pm)

    opt_system = OpticalSystem(
        name=f"Preset 3: One-Sided Slicer ({n_channels}ch, {pupil_layout_side}) → Pupil Relay → New Axis Lens → Fiber",
        fore_optics=fore_optics_list,
        slicer=slicer,
        pupil_relay=pupil_relay,
        final_lens_3d=condenser_lens,
        coupling_optics=[],
        fiber=fiber,
        is_non_sequential_post_slicer=True,
    )
    # Attach helper attributes for visualization and diagnostics
    opt_system.relay_axis_origin = p_centroid
    opt_system.relay_axis_direction = a_relay
    opt_system.pupil_layout_side = pupil_layout_side
    opt_system.pupil_aim_mode = pupil_aim_mode

    return opt_system


def create_paper_ifu_system(active_slices: int = 4) -> OpticalSystem:
    """Alias for branched slicer system in 4-channel mode."""
    return create_branched_slicer_system(n_channels=4, pupil_layout_side="lower")


def create_first_default_experiment(active_slices: int = 2) -> OpticalSystem:
    """First Default Experiment: 2-channel one-sided initial test geometry."""
    return create_branched_slicer_system(n_channels=min(max(active_slices, 2), 4), pupil_layout_side="lower")
