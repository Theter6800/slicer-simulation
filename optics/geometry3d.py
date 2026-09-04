"""
3D Geometric transformation utilities, coordinate systems, and vector reflection optics.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a 1D vector or an array of row vectors (N, 3)."""
    v_arr = np.asarray(v, dtype=np.float64)
    if v_arr.ndim == 1:
        norm = np.linalg.norm(v_arr)
        return v_arr / norm if norm > 1e-14 else v_arr
    else:
        norms = np.linalg.norm(v_arr, axis=-1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        return v_arr / norms


def build_orthonormal_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a unit normal vector w (pointing along surface normal),
    construct an orthonormal right-handed basis (u, v, w).
    """
    w = normalize(normal)
    # Choose a helper vector not collinear with w
    # If w is primarily along z, use x-axis as helper; otherwise use z-axis
    if np.abs(w[2]) > 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    u = np.cross(helper, w)
    u = normalize(u)
    v = np.cross(w, u)
    v = normalize(v)
    return u, v, w


def reflect_vector(k_in: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Specular reflection of unit direction vector(s) k_in off normal n.
    k_out = k_in - 2 * (k_in . n) * n
    """
    n = normalize(normal)
    if k_in.ndim == 1:
        k_dot_n = np.dot(k_in, n)
        k_out = k_in - 2.0 * k_dot_n * n
        return normalize(k_out)
    else:
        k_dot_n = np.sum(k_in * n, axis=-1, keepdims=True)
        k_out = k_in - 2.0 * k_dot_n * n
        return normalize(k_out)


def reflection_bisector(k_in: np.ndarray, k_out: np.ndarray) -> np.ndarray:
    """
    Calculate the surface normal required to reflect k_in into k_out.
    Normal points against the incoming beam (k_in . n < 0).
    n = (k_in - k_out) / ||k_in - k_out||
    """
    k_in_u = normalize(k_in)
    k_out_u = normalize(k_out)
    diff = k_in_u - k_out_u
    norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-12:
        # Transmission / no deflection
        n = -k_in_u
    else:
        n = diff / norm_diff

    # Ensure normal opposes incoming ray
    if np.dot(k_in_u, n) > 0:
        n = -n
    return normalize(n)


def global_to_local(point: np.ndarray, origin: np.ndarray, basis: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Transform global coordinates to local coordinates (u, v, w) relative to origin and orthonormal basis.
    """
    u, v, w = basis
    rel = point - origin
    if rel.ndim == 1:
        return np.array([np.dot(rel, u), np.dot(rel, v), np.dot(rel, w)], dtype=np.float64)
    else:
        loc_u = np.dot(rel, u)
        loc_v = np.dot(rel, v)
        loc_w = np.dot(rel, w)
        return np.column_stack([loc_u, loc_v, loc_w])


def local_to_global(local_coords: np.ndarray, origin: np.ndarray, basis: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Transform local coordinates (u, v, w) back to global 3D coordinates.
    """
    u, v, w = basis
    if local_coords.ndim == 1:
        return origin + local_coords[0] * u + local_coords[1] * v + local_coords[2] * w
    else:
        return origin + (
            local_coords[:, 0:1] * u +
            local_coords[:, 1:2] * v +
            local_coords[:, 2:3] * w
        )
