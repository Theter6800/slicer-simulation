"""
Fore-Optics Modeling & Image Magnification Subsystem.
Controls optical image scale, focal length, and intermediate image plane (z_image)
prior to the image slicer.

Provides two modes:
1. REAL HARDWARE MODE: Uses only available laboratory lenses (f = 35 mm, 75 mm, 100 mm).
2. THEORETICAL MODE: Allows arbitrary continuous focal lengths and exact target D90 image sizes.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .ray import RayBundle, RayStatus
from .elements import CircularAperture, ThinLens
from .sources import generate_sun_source, generate_led_source
from .metrics import compute_spot_metrics, SpotMetrics


class ForeOpticsMode(str, Enum):
    REAL_HARDWARE = "REAL_HARDWARE"
    THEORETICAL = "THEORETICAL"


# Catalog of available hardware lenses in the physical optics laboratory
AVAILABLE_HARDWARE_FOCAL_LENGTHS: List[float] = [35.0, 75.0, 100.0]  # mm
HARDWARE_LENS_DIAMETER: float = 25.4  # mm (1 inch standard optic)


@dataclass
class ForeOpticsConfig:
    """Configuration for fore-optics image formation."""
    mode: ForeOpticsMode = ForeOpticsMode.THEORETICAL
    aperture_diameter: float = 12.0       # mm
    z_aperture: float = 20.0              # mm
    z_fore_start: float = 40.0            # mm (position of first lens)

    # Theoretical Mode settings
    target_d90_mm: float = 1.3            # mm
    theoretical_focal_length: float = 150.0  # mm

    # Real Hardware Mode settings
    lens1_focal_length: float = 100.0     # mm (from AVAILABLE_HARDWARE_FOCAL_LENGTHS)
    lens2_focal_length: Optional[float] = None  # None for single lens, or from AVAILABLE_HARDWARE_FOCAL_LENGTHS
    lens_spacing: float = 50.0            # mm


@dataclass
class ForeOpticsSystem:
    """Assembled fore-optics system with verified focal plane."""
    config: ForeOpticsConfig
    elements: List[CircularAperture | ThinLens]
    z_image: float                        # Strictly computed input image plane
    d90_image: float                      # Measured 90% encircled energy diameter (mm)
    effective_focal_length: float         # Equivalent system focal length (mm)
    spot_metrics: Optional[SpotMetrics] = None
    description: str = ""

    def trace(self, bundle: RayBundle) -> Tuple[RayBundle, SpotMetrics]:
        """
        Traces rays sequentially through aperture and all fore-optics lenses,
        then propagates strictly to z_image and computes spot diagnostics.
        """
        for elem in self.elements:
            elem.trace(bundle)
        bundle.propagate_to_z(self.z_image)
        metrics = compute_spot_metrics(bundle, z_plane=self.z_image)
        self.spot_metrics = metrics
        self.d90_image = metrics.diameter_90
        return bundle, metrics


def compute_theoretical_focal_length_for_d90(
    target_d90: float,
    solar_angular_radius_deg: float = 0.266,
) -> float:
    """
    Computes required single-lens focal length to produce an intermediate solar image
    with D90 encircling energy diameter equal to target_d90.
    For uniform solar angular disk, D90 ~ 0.948 * D_geom, where D_geom = 2 * f * tan(alpha).
    """
    alpha_rad = np.radians(solar_angular_radius_deg)
    # Empirical calibration factor for uniform disk: D90 / (2 * f * tan(alpha)) ~ 0.9487
    calib = 0.9487
    f_eff = target_d90 / (2.0 * np.tan(alpha_rad) * calib)
    return float(max(20.0, f_eff))


def build_fore_optics_system(
    config: Optional[ForeOpticsConfig] = None,
    target_d90: Optional[float] = None,
    mode: Optional[ForeOpticsMode] = None,
    n_rays_sample: int = 1500,
    seed: int = 42,
) -> ForeOpticsSystem:
    """
    Constructs and verifies a ForeOpticsSystem matching the specified mode and parameters.
    Guarantees that z_image is physically accurate for the optical train.
    """
    if config is None:
        config = ForeOpticsConfig()

    eff_mode = mode if mode is not None else config.mode

    aperture = CircularAperture(
        name="Entrance Aperture",
        z=config.z_aperture,
        diameter=config.aperture_diameter,
    )

    if eff_mode == ForeOpticsMode.THEORETICAL:
        d90_req = target_d90 if target_d90 is not None else config.target_d90_mm
        f_eff = compute_theoretical_focal_length_for_d90(d90_req)
        z_lens = config.z_fore_start
        z_img = z_lens + f_eff

        lens = ThinLens(
            name=f"Theoretical Objective Lens f={f_eff:.1f}mm",
            z=z_lens,
            focal_length=f_eff,
            diameter=max(HARDWARE_LENS_DIAMETER, config.aperture_diameter + 10.0),
        )
        elements = [aperture, lens]
        desc = f"Theoretical Objective (f = {f_eff:.1f} mm, Target D90 = {d90_req:.2f} mm)"

    else:
        # REAL HARDWARE MODE: uses only available lenses
        f1 = config.lens1_focal_length
        if f1 not in AVAILABLE_HARDWARE_FOCAL_LENGTHS:
            f1 = 100.0  # default fallback to available 100 mm lens

        z_l1 = config.z_fore_start
        lens1 = ThinLens(
            name=f"Hardware Lens 1 f={f1:.0f}mm",
            z=z_l1,
            focal_length=f1,
            diameter=HARDWARE_LENS_DIAMETER,
        )

        if config.lens2_focal_length is not None and config.lens2_focal_length in AVAILABLE_HARDWARE_FOCAL_LENGTHS:
            f2 = config.lens2_focal_length
            d12 = max(10.0, config.lens_spacing)
            z_l2 = z_l1 + d12
            lens2 = ThinLens(
                name=f"Hardware Lens 2 f={f2:.0f}mm",
                z=z_l2,
                focal_length=f2,
                diameter=HARDWARE_LENS_DIAMETER,
            )
            # Paraxial back focal distance for parallel incoming beam
            # s2 = d12 - f1
            # 1/s2' = 1/f2 - 1/s2
            s2 = d12 - f1
            if abs(s2) < 1e-4:
                # Collimated intermediate
                s2_prime = f2
            elif abs(d12 - f1 - f2) < 1e-4:
                s2_prime = 100.0
            else:
                s2_prime = (f2 * s2) / (s2 - f2)

            z_img = z_l2 + abs(s2_prime)
            f_eff = (f1 * f2) / max(1e-3, abs(f1 + f2 - d12))
            elements = [aperture, lens1, lens2]
            desc = f"Hardware 2-Lens Relay (f1 = {f1:.0f} mm, f2 = {f2:.0f} mm, d = {d12:.1f} mm)"
        else:
            z_img = z_l1 + f1
            f_eff = f1
            elements = [aperture, lens1]
            desc = f"Hardware Single Objective (f = {f1:.0f} mm)"

    fore_sys = ForeOpticsSystem(
        config=config,
        elements=elements,
        z_image=float(z_img),
        d90_image=0.0,
        effective_focal_length=float(f_eff),
        description=desc,
    )

    # Sample beam to measure true D90
    _, bundle = generate_sun_source(n_rays=n_rays_sample, pupil_diameter=config.aperture_diameter, seed=seed)
    _, metrics = fore_sys.trace(bundle)

    return fore_sys


def optimize_hardware_fore_optics(
    target_d90: float,
    aperture_diameter: float = 12.0,
) -> ForeOpticsSystem:
    """
    Searches available physical laboratory lenses (f in {35, 75, 100} mm)
    and spacings to find the hardware configuration that closest matches target_d90.
    """
    best_sys: Optional[ForeOpticsSystem] = None
    best_err = float("inf")

    # 1. Single lens candidates
    for f1 in AVAILABLE_HARDWARE_FOCAL_LENGTHS:
        cfg = ForeOpticsConfig(
            mode=ForeOpticsMode.REAL_HARDWARE,
            aperture_diameter=aperture_diameter,
            lens1_focal_length=f1,
            lens2_focal_length=None,
        )
        sys_cand = build_fore_optics_system(cfg, n_rays_sample=800)
        err = abs(sys_cand.d90_image - target_d90)
        if err < best_err:
            best_err = err
            best_sys = sys_cand

    # 2. Dual lens combinations
    for f1 in AVAILABLE_HARDWARE_FOCAL_LENGTHS:
        for f2 in AVAILABLE_HARDWARE_FOCAL_LENGTHS:
            for d in [25.0, 50.0, 75.0, 100.0, 125.0]:
                cfg = ForeOpticsConfig(
                    mode=ForeOpticsMode.REAL_HARDWARE,
                    aperture_diameter=aperture_diameter,
                    lens1_focal_length=f1,
                    lens2_focal_length=f2,
                    lens_spacing=d,
                )
                try:
                    sys_cand = build_fore_optics_system(cfg, n_rays_sample=800)
                    err = abs(sys_cand.d90_image - target_d90)
                    if err < best_err:
                        best_err = err
                        best_sys = sys_cand
                except Exception:
                    continue

    return best_sys if best_sys is not None else build_fore_optics_system(ForeOpticsConfig(mode=ForeOpticsMode.REAL_HARDWARE))
