"""
Ray source generators for Lab LED and Sun models.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple
import numpy as np
from .ray import RayBundle, RayStatus


class SourceMode(str, Enum):
    LED = "LED"
    SUN = "SUN"


class LEDSourceType(str, Enum):
    POINT = "point"
    UNIFORM_DISK = "uniform_disk"
    LAMBERTIAN = "lambertian"


def generate_led_source(
    n_rays: int = 10000,
    source_type: LEDSourceType | str = LEDSourceType.UNIFORM_DISK,
    source_diameter: float = 1.0,  # mm
    source_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # x, y, z in mm
    aperture_diameter: float = 25.0,  # mm
    aperture_pos: Tuple[float, float, float] = (0.0, 0.0, 50.0),  # x, y, z in mm
    angular_half_angle_deg: float = 30.0,  # emission half-angle if free emission
    seed: int = 42,
) -> Tuple[RayBundle, RayBundle]:
    """
    Generate rays from an LED source and propagate to the circular entrance aperture.
    
    Returns:
        (launched_bundle, at_aperture_bundle)
        where launched_bundle is at source plane, and at_aperture_bundle is clipped at aperture.
    """
    rng = np.random.default_rng(seed)
    source_type_str = str(source_type).lower()
    if isinstance(source_type, LEDSourceType):
        source_type_str = source_type.value

    x_s, y_s, z_s = source_pos
    x_ap, y_ap, z_ap = aperture_pos
    dist_z = z_ap - z_s
    if dist_z <= 0:
        raise ValueError(f"Aperture z ({z_ap}) must be downstream of source z ({z_s})")

    # 1. Spatial distribution of emission points r0
    r0 = np.zeros((n_rays, 3), dtype=np.float64)
    r0[:, 0] = x_s
    r0[:, 1] = y_s
    r0[:, 2] = z_s

    if source_type_str in ("uniform_disk", "lambertian") and source_diameter > 0:
        # Uniform sampling on a disk of radius R = source_diameter / 2
        rho = (source_diameter / 2.0) * np.sqrt(rng.uniform(0.0, 1.0, n_rays))
        phi = rng.uniform(0.0, 2.0 * np.pi, n_rays)
        r0[:, 0] += rho * np.cos(phi)
        r0[:, 1] += rho * np.sin(phi)

    # 2. Angular distribution
    # We aim rays towards the entrance aperture plane to efficiently trace rays,
    # weighting them according to the selected emission distribution.
    # To sample thoroughly, target an aperture region covering slightly beyond the aperture.
    target_radius = (aperture_diameter / 2.0) * 1.15  # 15% margin to capture clipping
    target_rho = target_radius * np.sqrt(rng.uniform(0.0, 1.0, n_rays))
    target_phi = rng.uniform(0.0, 2.0 * np.pi, n_rays)
    target_x = x_ap + target_rho * np.cos(target_phi)
    target_y = y_ap + target_rho * np.sin(target_phi)
    target_z = np.full(n_rays, z_ap)

    # Compute ray vectors towards target points
    dx = target_x - r0[:, 0]
    dy = target_y - r0[:, 1]
    dz = target_z - r0[:, 2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    kx = dx / dist
    ky = dy / dist
    kz = dz / dist
    k = np.column_stack([kx, ky, kz])

    # Power weighting
    cos_theta = kz  # angle with respect to z-axis (nominal bench axis)
    if source_type_str == "lambertian":
        # Lambertian source emits with intensity proportional to cos(theta)
        weights = np.maximum(cos_theta, 0.0)
    else:
        # Uniform emission within targeted solid angle
        weights = np.ones(n_rays, dtype=np.float64)

    # Normalize total launched power to 1.0
    power = weights / np.sum(weights)

    launched_bundle = RayBundle(r=r0, k=k, power=power)
    launched_bundle.record_snapshot("Source (LED)")

    # Propagate to aperture plane at z = z_ap
    at_aperture_bundle = launched_bundle.clone()
    at_aperture_bundle.propagate_to_z(z_ap)

    # Clip rays outside aperture
    r_ap_sq = (at_aperture_bundle.x - x_ap)**2 + (at_aperture_bundle.y - y_ap)**2
    ap_radius = aperture_diameter / 2.0
    clipped = r_ap_sq > (ap_radius**2)
    at_aperture_bundle.status[clipped] = RayStatus.CLIPPED
    at_aperture_bundle.record_snapshot("Entrance Aperture")

    return launched_bundle, at_aperture_bundle


def generate_sun_source(
    n_rays: int = 10000,
    pupil_diameter: float = 25.0,  # mm (entrance aperture)
    pupil_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # x, y, z in mm
    solar_angular_radius_deg: float = 0.266,  # degrees (standard solar disk half-angle)
    seed: int = 42,
) -> Tuple[RayBundle, RayBundle]:
    """
    Generate rays from the Sun modeled as an extended angular disk.
    
    Rays are generated uniformly over:
    1. Entrance pupil position (circular pupil of diameter pupil_diameter at pupil_pos)
    2. Solar angular disk (half-angle solar_angular_radius_deg)
    
    Returns:
        (source_bundle, at_pupil_bundle)
    """
    rng = np.random.default_rng(seed)
    x_p, y_p, z_p = pupil_pos
    alpha_sun_rad = np.radians(solar_angular_radius_deg)

    # 1. Uniform spatial sampling across circular entrance pupil
    r_pupil = (pupil_diameter / 2.0) * np.sqrt(rng.uniform(0.0, 1.0, n_rays))
    phi_pupil = rng.uniform(0.0, 2.0 * np.pi, n_rays)
    x = x_p + r_pupil * np.cos(phi_pupil)
    y = y_p + r_pupil * np.sin(phi_pupil)
    z = np.full(n_rays, z_p)
    r = np.column_stack([x, y, z])

    # 2. Uniform angular distribution within solar disk
    # For a flat projection or small angles: sin(theta) ~ theta
    # Sampling uniform on disk: theta = alpha_sun * sqrt(u), phi in [0, 2pi)
    theta = alpha_sun_rad * np.sqrt(rng.uniform(0.0, 1.0, n_rays))
    phi_sun = rng.uniform(0.0, 2.0 * np.pi, n_rays)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    kx = sin_theta * np.cos(phi_sun)
    ky = sin_theta * np.sin(phi_sun)
    kz = cos_theta
    k = np.column_stack([kx, ky, kz])

    power = np.full(n_rays, 1.0 / n_rays, dtype=np.float64)

    pupil_bundle = RayBundle(r=r, k=k, power=power)
    pupil_bundle.record_snapshot("Entrance Pupil (Sun)")

    return pupil_bundle.clone(), pupil_bundle
