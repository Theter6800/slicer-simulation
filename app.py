"""
Interactive Geometric-Optics Simulator for Solar/LED-to-Fiber Coupling.
Featuring 3D Branched Non-Sequential Image Slicer IFU Architecture (Ellen Lee, JATIS 2026).
"""

from __future__ import annotations
import json
import os
import io
import subprocess
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st

from optics.ray import RayBundle, RayStatus
from optics.sources import (
    SourceMode,
    LEDSourceType,
    generate_led_source,
    generate_sun_source,
)
from optics.elements import (
    CircularAperture,
    ThinLens,
    PlaneMirror,
    ThinLens3D,
    PlaneMirror3D,
    SphericalMirror3D,
)
from optics.geometry3d import (
    normalize,
    build_orthonormal_basis,
    reflect_vector,
    reflection_bisector,
)
from optics.slicer import SlicerArray, SliceMirror
from optics.pupil import PupilRelaySystem, PupilMirror, PupilAnalysis
from optics.fiber import Fiber, FiberCouplingResult
from optics.system import OpticalSystem
from optics.metrics import (
    compute_spot_metrics,
    scan_beam_size_vs_z,
    find_z_for_target_footprint,
    compute_etendue_check,
)
from optics.presets import (
    create_old_lens_system,
    create_slicer_condenser_system,
    create_branched_slicer_system,
    create_paper_ifu_system,
    create_first_default_experiment,
)
from scipy.optimize import minimize
from optics.optimization import (
    aim_slicer_mirror,
    optimize_slicer_tilts,
    optimize_fiber_position,
    run_1d_sweep,
    run_2d_sweep,
)
from optics.design_optimizer import (
    OptimizationConfig,
    SingleNOptimizationResult,
    MultiNStudyResult,
    SlicerPupilOptimizer,
)
from optics.power_accounting import PowerAccounting, compute_etendue
from optics.validation import (
    run_all_validations,
    ValidationSuiteReport,
    ValidationCaseResult,
    validate_multi_start_optimization,
)

# Set Plotly default font family to Glacial Indifference
try:
    pio.templates[pio.templates.default].layout.font.family = "Glacial Indifference, League Spartan, sans-serif"
    if "plotly_white" in pio.templates:
        pio.templates["plotly_white"].layout.font.family = "Glacial Indifference, League Spartan, sans-serif"
except Exception:
    pass

# Page configuration
st.set_page_config(
    page_title="Solar/LED-to-Fiber Image Slicer Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Glacial Indifference Typography Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.cdnfonts.com/css/glacial-indifference-2');
    @import url('https://fonts.googleapis.com/css2?family=League+Spartan:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], [class*="st-"], .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, label, button, input, select, textarea, div, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-family: 'Glacial Indifference', 'League Spartan', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Glacial Indifference', 'League Spartan', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-family: 'Glacial Indifference', sans-serif !important;
        font-weight: 700 !important;
    }

    button {
        font-family: 'Glacial Indifference', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Scientific Disclaimer Banner
st.warning(
    "**Scientific Modeling Scope:** This simulator currently uses geometric optics and idealized thin lenses. "
    "It is intended for architecture selection and laboratory planning, not final optical tolerancing.\n\n"
    "**Assumptions & Approximations:** Initially ignores diffraction, Fresnel reflection losses, optical coatings, "
    "wavelength dispersion, real lens aberrations, surface roughness, fiber focal ratio degradation (FRD), "
    "and mechanical tolerances."
)

st.title("Solar / LED-to-Optical-Fiber Image Slicer Simulator")
st.caption(
    "3D Branched Non-Sequential IFU Architecture: Coupling extended source light into a 1.0 mm, NA 0.22 fiber "
    "(Ellen Lee, JATIS 2026)."
)

# Simulator Operation Mode Selector
top_mode_c1, top_mode_c2 = st.columns([3, 1])
with top_mode_c1:
    top_mode = st.radio(
        "Simulator Mode",
        ["Architecture Design Optimizer", "Interactive Bench Alignment"],
        index=0 if st.session_state.get("app_mode", "Architecture Design Optimizer") == "Architecture Design Optimizer" else 1,
        horizontal=True,
        help="Select between automated multi-N global design optimization and manual optical bench alignment.",
    )
    if top_mode != st.session_state.get("app_mode", "Architecture Design Optimizer"):
        st.session_state.app_mode = top_mode
        st.rerun()

# Initialize Session State
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Architecture Design Optimizer"
if "optimizer_study" not in st.session_state:
    st.session_state.optimizer_study = None
if "optimizer_selected_n" not in st.session_state:
    st.session_state.optimizer_selected_n = None
if "n_channels" not in st.session_state:
    st.session_state.n_channels = 2
if "preset_name" not in st.session_state:
    st.session_state.preset_name = "Asymmetric One-Sided Slicer (2-Channel Test)"
if "source_mode" not in st.session_state:
    st.session_state.source_mode = "SUN"
if "fore_focal_opt" not in st.session_state:
    st.session_state.fore_focal_opt = 150.0
if "n_rays" not in st.session_state:
    st.session_state.n_rays = 5000
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42
if "pupil_layout_side" not in st.session_state:
    st.session_state.pupil_layout_side = "lower"
if "pupil_aim_mode" not in st.session_state:
    st.session_state.pupil_aim_mode = "Parallel to Relay Axis (Max Focus)"
if "pupil_dist" not in st.session_state:
    st.session_state.pupil_dist = 40.0
if "pupil_transverse_offset" not in st.session_state:
    st.session_state.pupil_transverse_offset = 20.0
if "condenser_focal_length" not in st.session_state:
    st.session_state.condenser_focal_length = 22.0
if "final_lens_z" not in st.session_state:
    st.session_state.final_lens_z = 290.0
if "pupil_power_enabled" not in st.session_state:
    st.session_state.pupil_power_enabled = True
if "manual_channels" not in st.session_state:
    st.session_state.manual_channels = {}
if "manual_selected_ch" not in st.session_state:
    st.session_state.manual_selected_ch = 0
if "manual_align_mode" not in st.session_state:
    st.session_state.manual_align_mode = "Mode A: Adjust Slicer to Fixed Pupil"
if "manual_fine_thresh" not in st.session_state:
    st.session_state.manual_fine_thresh = 0.1
if "manual_coarse_thresh" not in st.session_state:
    st.session_state.manual_coarse_thresh = 0.5
if "_last_geom_key" not in st.session_state:
    st.session_state._last_geom_key = ""

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("Simulation Controls")

    sb_mode = st.radio(
        "Operation Mode",
        ["Architecture Design Optimizer", "Interactive Bench Alignment"],
        index=0 if st.session_state.app_mode == "Architecture Design Optimizer" else 1,
        key="sidebar_mode_selector",
        help="Select between automated multi-N global design optimization and manual optical bench experimentation.",
    )
    if sb_mode != st.session_state.app_mode:
        st.session_state.app_mode = sb_mode
        st.rerun()

    st.divider()

    curr_n = st.session_state.get("n_channels", 2)
    preset_options = [
        f"Asymmetric One-Sided Slicer ({curr_n}-Channel)",
        "Asymmetric One-Sided Slicer (2-Channel Test)",
        "Asymmetric One-Sided Slicer (4-Channel Full)",
        "Preset 1: Old lens-only system",
        "Preset 2: Slicer + common condenser",
    ]
    dedup_presets = list(dict.fromkeys(preset_options))
    p_idx = dedup_presets.index(st.session_state.preset_name) if st.session_state.preset_name in dedup_presets else 0

    preset_choice = st.selectbox(
        "Hardware Preset / Architecture",
        dedup_presets,
        index=p_idx,
    )
    st.session_state.preset_name = preset_choice

    st.divider()
    st.subheader("Source & Ray Bundle")
    source_mode_choice = st.radio("Source Mode", ["LED", "SUN"], index=0, horizontal=True)
    st.session_state.source_mode = source_mode_choice

    ray_count_options = [1000, 5000, 10000, 25000, 50000]
    n_rays_choice = st.select_slider("Ray Count", options=ray_count_options, value=5000)
    st.session_state.n_rays = n_rays_choice

    seed_choice = st.number_input("Deterministic Random Seed", value=42, step=1)
    st.session_state.random_seed = int(seed_choice)

    st.divider()
    st.subheader("One-Sided Slicer Geometry")
    st.caption("All slicers deflect to ONE side, creating a NEW common relay axis:")

    side_options = ["lower", "upper", "left", "right"]
    current_side_idx = side_options.index(st.session_state.pupil_layout_side) if st.session_state.pupil_layout_side in side_options else 0
    st.session_state.pupil_layout_side = st.selectbox(
        "Pupil Layout Side (One-Sided)",
        side_options,
        index=current_side_idx,
        help="All pupil mirrors lie strictly on this chosen side of the original optical axis",
    )

    aim_options = ["Parallel to Relay Axis (Max Focus)", "Converge to Condenser Center"]
    current_aim_idx = 0 if "Parallel" in st.session_state.pupil_aim_mode else 1
    st.session_state.pupil_aim_mode = st.selectbox(
        "Pupil Redirection Mode",
        aim_options,
        index=current_aim_idx,
        help="Parallel mode aligns chief rays with condenser lens axis, maximizing fiber coupling",
    )

    st.session_state.pupil_transverse_offset = st.number_input(
        "Pupil Transverse Offset (mm)",
        value=float(st.session_state.pupil_transverse_offset),
        step=5.0,
        help="Transverse distance from original optical axis to pupil cluster",
    )

    st.session_state.pupil_dist = st.number_input(
        "Pupil Distance z (mm)",
        value=float(st.session_state.pupil_dist),
        step=5.0,
        help="Axial distance from slicer plane to pupil mirrors",
    )

    st.session_state.pupil_power_enabled = st.checkbox(
        "Enable Powered Pupil Mirrors (f=75mm)",
        value=st.session_state.pupil_power_enabled,
        help="Powered concave mirrors refocus/collimate each channel; flat mirrors only steer",
    )

    st.divider()
    st.caption(
        "Coordinate Conventions:\n"
        "- Pre-slicer: Nominal input axis (+z)\n"
        "- Post-slicer: Asymmetric one-sided relay onto NEW optical axis"
    )

# ==========================================
# OPTICAL FIGURE RENDERING HELPERS
# ==========================================
def render_3d_system_figure(
    system: OpticalSystem,
    traced_bundle: RayBundle,
    max_rays: int = 150,
    isolated_ch: Optional[int] = None,
    show_elements: bool = True,
    show_normals: bool = False,
    title: str = "3D Interactive Ray Trace Scene",
) -> go.Figure:
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # 1. Traced rays
    trajectories = []
    if hasattr(traced_bundle, "trajectories") and traced_bundle.trajectories:
        trajectories = traced_bundle.trajectories
    elif hasattr(traced_bundle, "history") and traced_bundle.history:
        n_p = len(traced_bundle.history)
        n_r = traced_bundle.n_rays if hasattr(traced_bundle, "n_rays") else len(traced_bundle.history[0]["r"])
        for r_i in range(n_r):
            trajectories.append(np.array([traced_bundle.history[p]["r"][r_i] for p in range(n_p)], dtype=np.float64))

    n_total_rays = len(trajectories)
    step = max(1, n_total_rays // max_rays) if n_total_rays > 0 else 1
    for r_i in range(0, n_total_rays, step):
        traj = trajectories[r_i]
        if len(traj) < 2:
            continue
        x_traj = [float(p[0]) for p in traj]
        y_traj = [float(p[1]) for p in traj]
        z_traj = [float(p[2]) for p in traj]

        ch = int(traced_bundle.channel_id[r_i]) if hasattr(traced_bundle, "channel_id") and r_i < len(traced_bundle.channel_id) else -1
        is_faded = (isolated_ch is not None and ch != isolated_ch)
        ray_opacity = 0.08 if is_faded else (0.9 if isolated_ch is not None else 0.6)
        ray_width = 1.0 if is_faded else (3.0 if isolated_ch is not None else 2.0)
        ray_col = colors[ch % len(colors)] if ch >= 0 else "#888888"

        fig.add_trace(
            go.Scatter3d(
                x=x_traj,
                y=y_traj,
                z=z_traj,
                mode="lines",
                line=dict(color=ray_col, width=ray_width),
                opacity=ray_opacity,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 2. Physical Optical Elements
    if show_elements:
        def add_3d_disk(center, normal, radius, name, color="rgba(70, 130, 180, 0.25)"):
            u, v, w = build_orthonormal_basis(normal)
            theta = np.linspace(0, 2 * np.pi, 28)
            x_pts = center[0] + radius * (np.cos(theta) * u[0] + np.sin(theta) * v[0])
            y_pts = center[1] + radius * (np.cos(theta) * u[1] + np.sin(theta) * v[1])
            z_pts = center[2] + radius * (np.cos(theta) * u[2] + np.sin(theta) * v[2])
            fig.add_trace(
                go.Scatter3d(
                    x=x_pts,
                    y=y_pts,
                    z=z_pts,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=name,
                )
            )

        def add_3d_rect(center, normal, width, height, name, color="rgba(255, 140, 0, 0.4)"):
            u, v, w = build_orthonormal_basis(normal)
            hw = width / 2.0
            hh = height / 2.0
            corners = [
                center + hw * u + hh * v,
                center - hw * u + hh * v,
                center - hw * u - hh * v,
                center + hw * u - hh * v,
                center + hw * u + hh * v,
            ]
            fig.add_trace(
                go.Scatter3d(
                    x=[c[0] for c in corners],
                    y=[c[1] for c in corners],
                    z=[c[2] for c in corners],
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=name,
                )
            )

        # Fore-optics lenses
        for elem in system.fore_optics:
            if isinstance(elem, ThinLens):
                add_3d_disk([0, 0, elem.z], [0, 0, 1], elem.radius, elem.name[:16])

        # Slicer Mirrors
        if system.slicer is not None:
            for s in system.slicer.slices:
                if s.enabled:
                    is_active = (isolated_ch is None or s.slice_id == isolated_ch)
                    s_col = "#00D2FF" if (isolated_ch is not None and is_active) else ("#1f77b4" if is_active else "rgba(100, 100, 100, 0.2)")
                    add_3d_rect(s.center, s.normal, s.width, s.height, f"Slice S{s.slice_id + 1}", color=s_col)

        # Pupil Mirrors
        if system.pupil_relay is not None:
            for m in system.pupil_relay.mirrors:
                if m.enabled:
                    is_active = (isolated_ch is None or m.channel_id == isolated_ch)
                    m_col = "#FF9F1C" if (isolated_ch is not None and is_active) else ("#ff7f0e" if is_active else "rgba(100, 100, 100, 0.2)")
                    add_3d_rect(m.center, m.normal, m.width, m.height, m.name, color=m_col)

        # Final Condenser Lens
        if system.final_lens_3d is not None:
            add_3d_disk(
                system.final_lens_3d.center,
                system.final_lens_3d.normal,
                system.final_lens_3d.radius,
                "Final Coupling Lens",
                color="#2ca02c",
            )

        # Fiber Face
        add_3d_disk(
            system.fiber.position,
            system.fiber.axis,
            system.fiber.core_radius,
            "Fiber Core (1mm)",
            color="#d62728",
        )

        # Reference Axes
        z_slicer_end = float(system.slicer.z + 20.0) if system.slicer is not None else 250.0
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, 0.0],
                y=[0.0, 0.0],
                z=[0.0, z_slicer_end],
                mode="lines",
                line=dict(color="#888888", width=3, dash="dash"),
                name="Original Source Axis",
            )
        )

        if hasattr(system, "relay_axis_origin") and hasattr(system, "relay_axis_direction"):
            p_orig = system.relay_axis_origin
            a_rel = system.relay_axis_direction
            pt_start = p_orig - 15.0 * a_rel
            pt_end = system.fiber.position + 30.0 * a_rel
            fig.add_trace(
                go.Scatter3d(
                    x=[pt_start[0], pt_end[0]],
                    y=[pt_start[1], pt_end[1]],
                    z=[pt_start[2], pt_end[2]],
                    mode="lines",
                    line=dict(color="#800080", width=4, dash="dot"),
                    name="New Common Relay Axis",
                )
            )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X (mm)", backgroundcolor="#fcfcfc", gridcolor="#e0e0e0"),
            yaxis=dict(title="Y (mm)", backgroundcolor="#fcfcfc", gridcolor="#e0e0e0"),
            zaxis=dict(title="Z (mm)", backgroundcolor="#fcfcfc", gridcolor="#e0e0e0"),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.8, y=-2.0, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=35),
        template="plotly_white",
        height=580,
    )
    return fig


def render_fiber_spot_figure(coupling_res: FiberCouplingResult, title: str = "Fiber Face Spot Distribution") -> go.Figure:
    fig = go.Figure()
    theta_c = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=0.5 * np.cos(theta_c),
            y=0.5 * np.sin(theta_c),
            mode="lines",
            line=dict(color="black", width=2.5, dash="dash"),
            name="1.0 mm Core Boundary (r=0.5mm)",
        )
    )
    class_defs = [
        (RayStatus.ACCEPTED_BY_FIBER, "Accepted (Spatial & NA OK)", "#00CC96", 5),
        (RayStatus.REJECTED_BY_NA, "Rejected: NA Only (r OK, θ > NA)", "#FFA15A", 4),
        (RayStatus.REJECTED_BY_POSITION, "Rejected: Spatial Only (r > 0.5mm, θ OK)", "#636EFA", 4),
        (RayStatus.REJECTED_BY_BOTH, "Rejected: Both Failed", "#EF553B", 4),
    ]
    for s_code, s_label, s_color, s_size in class_defs:
        c_mask = coupling_res.classifications == s_code
        if np.any(c_mask):
            fig.add_trace(
                go.Scatter(
                    x=coupling_res.hit_x[c_mask],
                    y=coupling_res.hit_y[c_mask],
                    mode="markers",
                    marker=dict(color=s_color, size=s_size, opacity=0.7),
                    name=f"{s_label} ({np.sum(c_mask)})",
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Local u (mm)",
        yaxis_title="Local v (mm)",
        template="plotly_white",
        height=420,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.2, 1.2]),
        yaxis=dict(range=[-1.2, 1.2]),
    )
    return fig


def render_fiber_angular_figure(coupling_res: FiberCouplingResult, title: str = "Angular Acceptance Phase Space") -> go.Figure:
    fig = go.Figure()
    fig.add_vrect(x0=0.0, x1=0.5, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.0, y1=12.71, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="r = 0.5 mm")
    fig.add_hline(y=12.71, line_dash="dash", line_color="red", annotation_text="NA = 0.22 (12.71°)")

    class_defs = [
        (RayStatus.ACCEPTED_BY_FIBER, "Accepted (Spatial & NA OK)", "#00CC96", 5),
        (RayStatus.REJECTED_BY_NA, "Rejected: NA Only (r OK, θ > NA)", "#FFA15A", 4),
        (RayStatus.REJECTED_BY_POSITION, "Rejected: Spatial Only (r > 0.5mm, θ OK)", "#636EFA", 4),
        (RayStatus.REJECTED_BY_BOTH, "Rejected: Both Failed", "#EF553B", 4),
    ]
    for s_code, s_label, s_color, s_size in class_defs:
        c_mask = coupling_res.classifications == s_code
        if np.any(c_mask):
            fig.add_trace(
                go.Scatter(
                    x=coupling_res.r_coords[c_mask],
                    y=coupling_res.theta_angles_deg[c_mask],
                    mode="markers",
                    marker=dict(color=s_color, size=s_size, opacity=0.7),
                    name=s_label,
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Radial Position r (mm)",
        yaxis_title="Ray Angle θ to Fiber Axis (deg)",
        template="plotly_white",
        height=420,
    )
    return fig


def render_loss_waterfall_figure(loss_budget: Dict[str, float], title: str = "Optical Loss Waterfall") -> go.Figure:
    fig = go.Figure()
    stages = list(loss_budget.keys())
    powers = [float(loss_budget[k]) for k in stages]
    cols = ["#1f77b4" if "Accepted" not in s else "#2ca02c" for s in stages]
    fig.add_trace(
        go.Bar(
            x=stages,
            y=powers,
            marker_color=cols,
            text=[f"{p:.3f} W" for p in powers],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Optical Stage / Acceptance Metric",
        yaxis_title="Active Optical Power (W)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def reoptimize_manual_geometry(system: OpticalSystem, manual_channels: dict) -> Tuple[bool, float, float]:
    n = len(manual_channels)
    if n == 0 or system.slicer is None or system.pupil_relay is None:
        return False, 0.0, 0.0

    side = st.session_state.get("pupil_layout_side", "lower")
    p_dist = float(st.session_state.get("pupil_dist", 60.0))
    p_offset = float(st.session_state.get("pupil_transverse_offset", 25.0))

    cfg = OptimizationConfig(
        n_min=n,
        n_max=n,
        pupil_layout_side=side,
        pupil_distance_z=p_dist,
        pupil_transverse_offset=p_offset,
        exploration_rays=400,
        validation_rays=1500,
        max_de_iter=10,
        popsize=6,
        polish=True,
    )
    optimizer = SlicerPupilOptimizer(cfg)

    # 1. Baseline trace
    val_bundle = optimizer.generate_source_bundle(cfg.validation_rays)
    _, curr_coupling, _ = system.trace(val_bundle)
    old_eff = float(curr_coupling.geometric_coupling_efficiency)

    # 2. Extract vector from manual_channels
    x0_list = []
    for i in range(n):
        ch_d = manual_channels[i]
        x0_list.extend([float(ch_d["slicer_pos"][0]), float(ch_d["slicer_pos"][1])])
    for i in range(n):
        ch_d = manual_channels[i]
        x0_list.extend([float(ch_d["pupil_pos"][0]), float(ch_d["pupil_pos"][1]), float(ch_d["pupil_pos"][2])])
    x0 = np.array(x0_list, dtype=np.float64)

    expl_bundle = optimizer.get_pre_slicer_bundle(cfg.exploration_rays)
    spot_rad = optimizer.get_spot_radius(expl_bundle)
    w, h = optimizer.get_slice_dimensions(n, spot_rad)
    source_power = float(optimizer._cached_source_bundle.total_power)

    def obj_func(vec):
        return optimizer.evaluate_merit_function(vec, n, w, h, expl_bundle, source_power)

    # Polish with Nelder-Mead
    res = minimize(obj_func, x0, method="Nelder-Mead", options={"maxiter": 50, "disp": False})
    best_x = res.x if res.fun < obj_func(x0) else x0

    opt_sys = optimizer.build_candidate_system(best_x, n, w, h)
    _, opt_coupling, _ = opt_sys.trace(val_bundle)
    new_eff = float(opt_coupling.geometric_coupling_efficiency)

    # Update manual_channels with refined positions & tilts
    for i, s in enumerate(opt_sys.slicer.slices):
        pm = opt_sys.pupil_relay.mirrors[i]
        manual_channels[i]["slicer_pos"] = [float(s.center_x), float(s.center_y), float(s.z)]
        manual_channels[i]["slicer_tip_x"] = float(s.tip_x_deg)
        manual_channels[i]["slicer_tilt_y"] = float(s.tilt_y_deg)
        manual_channels[i]["pupil_pos"] = [float(pm.center[0]), float(pm.center[1]), float(pm.center[2])]
        manual_channels[i]["pupil_tip_x"] = float(pm.tip_x_deg)
        manual_channels[i]["pupil_tilt_y"] = float(pm.tilt_y_deg)

    return True, old_eff, new_eff


# ==========================================
# BUILD CURRENT OPTICAL SYSTEM
# ==========================================
def get_current_system() -> OpticalSystem:
    preset = st.session_state.preset_name
    p_dist = st.session_state.pupil_dist
    p_focal = 75.0 if st.session_state.pupil_power_enabled else None
    side = st.session_state.pupil_layout_side
    aim_mode = "parallel" if "Parallel" in st.session_state.pupil_aim_mode else "target_center"
    offset = st.session_state.pupil_transverse_offset
    f_cond = st.session_state.condenser_focal_length

    current_geom_key = f"{preset}_{side}_{aim_mode}_{p_dist}_{offset}_{f_cond}_{p_focal}"
    if st.session_state.get("_last_geom_key") != current_geom_key:
        st.session_state["_last_geom_key"] = current_geom_key
        st.session_state["manual_channels"] = {}

    if "2-Channel" in preset:
        sys = create_branched_slicer_system(
            n_channels=2,
            pupil_layout_side=side,
            pupil_aim_mode=aim_mode,
            pupil_distance_z=p_dist,
            pupil_transverse_offset=offset,
            pupil_focal_length=p_focal,
            condenser_focal_length=f_cond,
        )
    elif "4-Channel" in preset:
        sys = create_branched_slicer_system(
            n_channels=4,
            pupil_layout_side=side,
            pupil_aim_mode=aim_mode,
            pupil_distance_z=p_dist,
            pupil_transverse_offset=offset,
            pupil_focal_length=p_focal,
            condenser_focal_length=f_cond,
        )
    elif "Asymmetric" in preset or "Channel" in preset:
        sys = create_branched_slicer_system(
            n_channels=int(st.session_state.get("n_channels", 2)),
            pupil_layout_side=side,
            pupil_aim_mode=aim_mode,
            pupil_distance_z=p_dist,
            pupil_transverse_offset=offset,
            pupil_focal_length=p_focal,
            condenser_focal_length=f_cond,
        )
    elif preset == "Preset 1: Old lens-only system":
        return create_old_lens_system()
    else:
        return create_slicer_condenser_system(active_slices=4)

    # Sync and apply manual channel states
    if sys.slicer is not None:
        if "manual_channels" not in st.session_state or not isinstance(st.session_state.manual_channels, dict):
            st.session_state.manual_channels = {}
        for s in sys.slicer.slices:
            ch_id = s.slice_id
            pm = sys.pupil_relay.mirrors[ch_id] if (sys.pupil_relay and ch_id < len(sys.pupil_relay.mirrors)) else None
            if ch_id not in st.session_state.manual_channels:
                st.session_state.manual_channels[ch_id] = {
                    "slicer_pos": [float(s.center_x), float(s.center_y), float(s.z)],
                    "slicer_tip_x": float(s.tip_x_deg),
                    "slicer_tilt_y": float(s.tilt_y_deg),
                    "slicer_rot_z": float(s.rot_z_deg),
                    "slicer_locked": False,
                    "pupil_pos": [float(pm.center[0]), float(pm.center[1]), float(pm.center[2])] if pm else [0.0, 0.0, 0.0],
                    "pupil_tip_x": float(pm.tip_x_deg) if pm else 0.0,
                    "pupil_tilt_y": float(pm.tilt_y_deg) if pm else 0.0,
                    "pupil_rot_z": float(pm.rot_z_deg) if pm else 0.0,
                    "pupil_locked": False,
                    "pupil_focal": float(pm.focal_length) if (pm and pm.focal_length) else None,
                }
            else:
                ch_data = st.session_state.manual_channels[ch_id]
                s.center_x = ch_data["slicer_pos"][0]
                s.center_y = ch_data["slicer_pos"][1]
                s.z = ch_data["slicer_pos"][2]
                s.set_tilt(ch_data["slicer_tip_x"], ch_data["slicer_tilt_y"], ch_data.get("slicer_rot_z", 0.0))
                if pm:
                    pm.set_position(ch_data["pupil_pos"][0], ch_data["pupil_pos"][1], ch_data["pupil_pos"][2])
                    pm.set_angles(ch_data["pupil_tip_x"], ch_data["pupil_tilt_y"], ch_data.get("pupil_rot_z", 0.0))
                    if ch_data.get("pupil_focal") is not None:
                        pm.focal_length = float(ch_data["pupil_focal"])
                        pm.is_powered = (pm.focal_length > 0.0)
                        pm.radius_of_curvature = 2.0 * pm.focal_length
    return sys


@st.cache_data(show_spinner=False)
def get_initial_bundle(source_mode: str, n_rays: int, seed: int) -> RayBundle:
    if source_mode == "SUN":
        _, bundle = generate_sun_source(
            n_rays=n_rays,
            pupil_diameter=20.0,
            solar_angular_radius_deg=0.266,
            seed=seed,
        )
    else:
        _, bundle = generate_led_source(
            n_rays=n_rays,
            source_type=LEDSourceType.UNIFORM_DISK,
            source_diameter=1.5,
            aperture_diameter=20.0,
            aperture_pos=(0.0, 0.0, 30.0),
            seed=seed,
        )
    return bundle


# ==========================================
# ARCHITECTURE DESIGN OPTIMIZER MODE VIEW
# ==========================================
if st.session_state.app_mode == "Architecture Design Optimizer":
    st.subheader("Automated Architecture Design Optimizer")
    st.markdown(
        "**Objective:** Maximize physical optical power coupling $\\eta = P_{\\text{accepted}} / P_{\\text{launched}}$ "
        "into a multimode fiber (core diameter $\\le 1.0\\text{ mm}$, radius $\\le 0.5\\text{ mm}$, $\\text{NA} \\le 0.22$, external medium air $n_{\\text{ext}}=1.0$).\n\n"
        "A ray is coupled if and only if **both** physical criteria are satisfied:\n"
        "1. **Spatial Core:** $\\sqrt{x_{\\text{fiber}}^2 + y_{\\text{fiber}}^2} \\le 0.5\\text{ mm}$\n"
        "2. **Angular Numerical Aperture:** $n_{\\text{ext}} \\cdot \\sin(\\theta_{\\text{fiber}}) \\le 0.22$ ($\\theta \\le 12.71^\\circ$ in air)\n\n"
        "This optimizer systematically evaluates candidate discrete slice counts $N_{\\text{slicer}} \\in [N_{\\text{min}} \\dots N_{\\text{max}}]$ "
        "separately without assuming more slicers is better. Slicer mirrors are strictly placed at the input focal plane ($z_s = 190.0\\text{ mm}$), "
        "pupil mirrors are strictly placed on the chosen side of the optical axis, and all mirror orientations are derived analytically "
        "using 3D vector reflection bisectors at every evaluation."
    )

    with st.expander("Optimization Configuration & Study Controls", expanded=True):
        opt_c1, opt_c2, opt_c3 = st.columns(3)
        with opt_c1:
            src_opts = ["SUN", "LED"]
            cur_src = st.session_state.get("source_mode", "SUN")
            src_idx = src_opts.index(cur_src) if cur_src in src_opts else 0
            source_mode_opt = st.selectbox(
                "Optical Source Mode",
                src_opts,
                index=src_idx,
                help="Sun: angular radius 0.266 deg. LED: uniform disk 1.5mm diameter.",
            )

            n_range = st.slider(
                "Candidate Slicer Channels (N)",
                min_value=1,
                max_value=6,
                value=(1, 4),
                help="Discrete range of slicer channels to evaluate. Each N is optimized globally.",
            )
            n_min, n_max = n_range[0], n_range[1]

            side_opts = ["lower", "upper", "left", "right"]
            cur_side = st.session_state.get("pupil_layout_side", "lower")
            s_idx = side_opts.index(cur_side) if cur_side in side_opts else 0
            pupil_side_opt = st.selectbox(
                "Pupil Layout Side (One-Sided)",
                side_opts,
                index=s_idx,
                help="All pupil mirrors are placed strictly on this chosen side of the optical axis.",
            )

        with opt_c2:
            fore_focal_opt = st.number_input(
                "Fore-Optic Objective Focal Length f_fore (mm)",
                value=float(st.session_state.get("fore_focal_opt", 150.0)),
                step=10.0,
                min_value=50.0,
                max_value=300.0,
                help="Objective lens focal length. Forms broad real image at image plane z_slicer = z_fore + f_fore.",
            )
            if source_mode_opt == "SUN":
                d_sun_calc = 2.0 * fore_focal_opt * np.tan(np.radians(0.266))
                st.caption(f"Broad Solar Disk Diameter on Slicer: {d_sun_calc:.2f} mm")
            else:
                st.caption("LED Source: Uniform disk image at slicer plane")

            condenser_focal_opt = st.number_input(
                "Condenser Lens Focal Length (mm)",
                value=float(st.session_state.get("condenser_focal_length", 22.0)),
                step=1.0,
                min_value=15.0,
                max_value=60.0,
                help="Common condenser lens focal length converging channels into fiber.",
            )

        with opt_c3:
            pupil_dist_opt = st.number_input(
                "Pupil Mirror Distance L (mm)",
                value=float(st.session_state.get("pupil_dist", 40.0)),
                step=5.0,
                help="Axial distance from slicer plane to pupil mirrors.",
            )
            pupil_offset_opt = st.number_input(
                "Pupil Transverse Offset (mm)",
                value=float(st.session_state.get("pupil_transverse_offset", 20.0)),
                step=5.0,
                help="Transverse distance from optical axis to pupil cluster.",
            )
            opt_intensity = st.selectbox(
                "Search Intensity & Ray Budget",
                [
                    "Fast Search (300 exploration rays, 1500 validation rays, 10 DE iters)",
                    "Standard Search (500 exploration rays, 3000 validation rays, 20 DE iters)",
                    "Deep Search (1000 exploration rays, 5000 validation rays, 30 DE iters)",
                ],
                index=0,
                help="Differential Evolution global search budget followed by Nelder-Mead polishing.",
            )
            opt_seed = st.number_input("Deterministic Random Seed", value=int(st.session_state.get("random_seed", 42)), step=1)

        run_opt_btn = st.button("Run Architecture Optimization Study", type="primary", use_container_width=True)

    if run_opt_btn:
        if "Fast" in opt_intensity:
            expl_r, val_r, max_it, pop = 300, 1500, 10, 6
        elif "Standard" in opt_intensity:
            expl_r, val_r, max_it, pop = 500, 3000, 20, 8
        else:
            expl_r, val_r, max_it, pop = 1000, 5000, 30, 10

        cfg = OptimizationConfig(
            n_min=n_min,
            n_max=n_max,
            source_mode=source_mode_opt,
            fore_focal_length=fore_focal_opt,
            condenser_focal_length=condenser_focal_opt,
            pupil_layout_side=pupil_side_opt,
            pupil_distance_z=pupil_dist_opt,
            pupil_transverse_offset=pupil_offset_opt,
            exploration_rays=expl_r,
            validation_rays=val_r,
            max_de_iter=max_it,
            popsize=pop,
            polish=True,
            random_seed=int(opt_seed),
        )
        optimizer = SlicerPupilOptimizer(cfg)

        prog_bar = st.progress(0.0)
        status_box = st.empty()

        def prog_callback(curr_n, frac, msg):
            span = max(1, n_max - n_min + 1)
            tot_f = (curr_n - n_min + frac) / span
            prog_bar.progress(min(1.0, max(0.0, tot_f)))
            status_box.markdown(f"**Optimization in Progress (N = {curr_n})**: {msg}")

        study_res = optimizer.run_multi_n_study(n_min, n_max, progress_callback=prog_callback)
        prog_bar.progress(1.0)
        status_box.success(f"Architecture study complete! Winning configuration: N* = {study_res.winning_n} channels.")
        st.session_state.optimizer_study = study_res
        st.session_state.optimizer_selected_n = study_res.winning_n
        st.session_state.source_mode = source_mode_opt
        st.session_state.fore_focal_opt = fore_focal_opt
        st.session_state.condenser_focal_length = condenser_focal_opt
        st.session_state.pupil_layout_side = pupil_side_opt
        st.session_state.pupil_dist = pupil_dist_opt
        st.session_state.pupil_transverse_offset = pupil_offset_opt

    if st.session_state.optimizer_study is not None:
        study = st.session_state.optimizer_study
        winner_n = study.winning_n
        winner_res = study.results[winner_n]
        pa_win = winner_res.power_accounting

        st.divider()
        st.markdown("### 1. Optical Architecture Analysis & Validation Gate")

        if getattr(study, "winner_declared", False):
            st.success(study.physical_rationale)
            w1, w2, w3, w4, w5, w6 = st.columns(6)
            with w1:
                st.metric("Architecture Winner", f"N* = {winner_n} Slicers", help="All 7 physical validation tests passed")
            with w2:
                st.metric("eta_total (abs)", f"{winner_res.coupling_efficiency * 100.0:.2f}%", help="P_inside_core_AND_NA / P_launch")
            with w3:
                st.metric("eta_core_launch", f"{pa_win.eta_core_launch * 100.0:.2f}%" if pa_win else "-", help="P_inside_core / P_launch")
            with w4:
                st.metric("eta_NA_launch", f"{pa_win.eta_na_launch * 100.0:.2f}%" if pa_win else "-", help="P_inside_NA / P_launch")
            with w5:
                st.metric("eta_core_conditional", f"{pa_win.eta_core_conditional * 100.0:.1f}%" if pa_win else "-", help="P_inside_core / P_at_fiber_plane")
            with w6:
                st.metric("eta_NA_conditional", f"{pa_win.eta_na_conditional * 100.0:.1f}%" if pa_win else "-", help="P_inside_NA / P_at_fiber_plane")
        else:
            st.warning(
                "DO NOT PRESENT N=1 AS A PHYSICAL OPTIMUM YET.\n\n"
                "Architecture Winner DEFERRED: Candidate rankings are being evaluated under the 7-case physical validation suite. "
                "No candidate architecture is declared optimal until all physical criteria are verified."
            )
            w1, w2, w3, w4, w5, w6 = st.columns(6)
            with w1:
                st.metric("Candidate (Validation Pending)", f"N = {winner_n} Slicers", help="Winner declaration deferred until validation tests complete")
            with w2:
                st.metric("eta_total (abs)", f"{winner_res.coupling_efficiency * 100.0:.2f}%", help="P_inside_core_AND_NA / P_launch")
            with w3:
                st.metric("eta_core_launch", f"{pa_win.eta_core_launch * 100.0:.2f}%" if pa_win else "-", help="P_inside_core / P_launch")
            with w4:
                st.metric("eta_NA_launch", f"{pa_win.eta_na_launch * 100.0:.2f}%" if pa_win else "-", help="P_inside_NA / P_launch")
            with w5:
                st.metric("eta_core_conditional", f"{pa_win.eta_core_conditional * 100.0:.1f}%" if pa_win else "-", help="P_inside_core / P_at_fiber_plane")
            with w6:
                st.metric("eta_NA_conditional", f"{pa_win.eta_na_conditional * 100.0:.1f}%" if pa_win else "-", help="P_inside_NA / P_at_fiber_plane")

        st.markdown("### 2. Multi-N Architecture Comparison Table")
        st.caption("Stage powers and explicit efficiency metrics relative to P_launch and conditional on fiber plane arrival:")
        st.dataframe(study.comparison_table, use_container_width=True)

        st.markdown("### 3. Multi-Start Optimization Statistics (10 Random Restarts per N)")
        st.caption("Distribution of coupling efficiency and clipping across 10 random initial geometries:")
        if hasattr(study, "multi_start_table") and not study.multi_start_table.empty:
            st.dataframe(study.multi_start_table, use_container_width=True)

        st.markdown("### 4. Transfer Optimum to Laboratory Bench")
        c_load1, c_load2 = st.columns([3, 1])
        with c_load1:
            st.write("Transfer this winning geometry into the Interactive Bench Alignment workbench for manual alignment inspection and experimental adjustments.")
        with c_load2:
            if st.button(f"Load Optimum (N = {winner_n}) into Manual Bench", type="primary", use_container_width=True):
                st.session_state.n_channels = winner_n
                st.session_state.preset_name = f"Asymmetric One-Sided Slicer ({winner_n}-Channel)"
                st.session_state.manual_channels = {}
                for s in winner_res.system.slicer.slices:
                    ch_id = s.slice_id
                    pm = winner_res.system.pupil_relay.mirrors[ch_id] if (winner_res.system.pupil_relay and ch_id < len(winner_res.system.pupil_relay.mirrors)) else None
                    st.session_state.manual_channels[ch_id] = {
                        "slicer_pos": [float(s.center_x), float(s.center_y), float(s.z)],
                        "slicer_tip_x": float(s.tip_x_deg),
                        "slicer_tilt_y": float(s.tilt_y_deg),
                        "slicer_rot_z": float(s.rot_z_deg),
                        "slicer_locked": False,
                        "pupil_pos": [float(pm.center[0]), float(pm.center[1]), float(pm.center[2])] if pm else [0.0, 0.0, 0.0],
                        "pupil_tip_x": float(pm.tip_x_deg) if pm else 0.0,
                        "pupil_tilt_y": float(pm.tilt_y_deg) if pm else 0.0,
                        "pupil_rot_z": float(pm.rot_z_deg) if pm else 0.0,
                        "pupil_locked": False,
                        "pupil_focal": float(pm.focal_length) if (pm and pm.focal_length) else None,
                    }
                st.session_state.app_mode = "Interactive Bench Alignment"
                st.rerun()

        st.divider()
        st.markdown("### 5. Candidate Architecture Inspection & Power Accounting")
        ch_list = list(study.results.keys())
        default_sel_idx = ch_list.index(st.session_state.get("optimizer_selected_n", winner_n)) if st.session_state.get("optimizer_selected_n", winner_n) in ch_list else 0
        selected_insp_n = st.radio(
            "Select Architecture to Inspect",
            ch_list,
            index=default_sel_idx,
            format_func=lambda n: f"N = {n} Slicers ({'WINNER - ' if (getattr(study, 'winner_declared', False) and n == winner_n) else ''}Efficiency: {study.results[n].coupling_efficiency*100.0:.2f}%)",
            horizontal=True,
        )
        st.session_state.optimizer_selected_n = selected_insp_n
        inspected_res = study.results[selected_insp_n]
        pa_insp = inspected_res.power_accounting

        # Stage-by-Stage Power Accounting Table relative to P_launch
        st.markdown("#### Stage-by-Stage Power Accounting Pipeline (Relative to P_launch)")
        if pa_insp is not None:
            p_launch = pa_insp.p_launch
            stages_data = [
                {"Stage Description": "P_launch (Total Optical Power Launched)", "Power (W)": f"{pa_insp.p_launch:.4f}", "Fraction of P_launch": "100.00%"},
                {"Stage Description": "P_after_aperture (Transmitted through Aperture)", "Power (W)": f"{pa_insp.p_after_aperture:.4f}", "Fraction of P_launch": f"{pa_insp.p_after_aperture/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_on_slicers (Incident at Slicer Image Plane)", "Power (W)": f"{pa_insp.p_on_slicers:.4f}", "Fraction of P_launch": f"{pa_insp.p_on_slicers/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_after_slicer_gaps (Reflected inside Active Slices)", "Power (W)": f"{pa_insp.p_after_slicer_gaps:.4f}", "Fraction of P_launch": f"{pa_insp.p_after_slicer_gaps/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_on_correct_pupil (Struck Assigned Pupil Mirror)", "Power (W)": f"{pa_insp.p_on_correct_pupil:.4f}", "Fraction of P_launch": f"{pa_insp.p_on_correct_pupil/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_on_wrong_pupil (Cross-Channel Stray Hit)", "Power (W)": f"{pa_insp.p_on_wrong_pupil:.4f}", "Fraction of P_launch": f"{pa_insp.p_on_wrong_pupil/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_missed_pupil (Missed All Pupil Mirrors)", "Power (W)": f"{pa_insp.p_missed_pupil:.4f}", "Fraction of P_launch": f"{pa_insp.p_missed_pupil/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_on_final_lens (Transmitted through Condenser Lens)", "Power (W)": f"{pa_insp.p_on_final_lens:.4f}", "Fraction of P_launch": f"{pa_insp.p_on_final_lens/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_at_fiber_plane (Arriving at Fiber Tip Plane)", "Power (W)": f"{pa_insp.p_at_fiber_plane:.4f}", "Fraction of P_launch": f"{pa_insp.p_at_fiber_plane/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_inside_core (Spatial Condition: r <= 0.5 mm)", "Power (W)": f"{pa_insp.p_inside_core:.4f}", "Fraction of P_launch": f"{pa_insp.p_inside_core/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_inside_NA (Angular Condition: sin θ <= 0.22)", "Power (W)": f"{pa_insp.p_inside_na:.4f}", "Fraction of P_launch": f"{pa_insp.p_inside_na/p_launch*100.0:.2f}%"},
                {"Stage Description": "P_inside_core_AND_NA (Final Valid Coupled Power)", "Power (W)": f"{pa_insp.p_inside_core_and_na:.4f}", "Fraction of P_launch": f"{pa_insp.p_inside_core_and_na/p_launch*100.0:.2f}%"},
            ]
            st.dataframe(pd.DataFrame(stages_data), use_container_width=True)

            # Explicit 6 efficiency metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("eta_total (abs)", f"{pa_insp.eta_total * 100.0:.2f}%", help="P_inside_core_AND_NA / P_launch")
            with m2:
                st.metric("eta_core_launch", f"{pa_insp.eta_core_launch * 100.0:.2f}%", help="P_inside_core / P_launch")
            with m3:
                st.metric("eta_NA_launch", f"{pa_insp.eta_na_launch * 100.0:.2f}%", help="P_inside_NA / P_launch")
            with m4:
                st.metric("eta_core_conditional", f"{pa_insp.eta_core_conditional * 100.0:.1f}%", help="P_inside_core / P_at_fiber_plane")
            with m5:
                st.metric("eta_NA_conditional", f"{pa_insp.eta_na_conditional * 100.0:.1f}%", help="P_inside_NA / P_at_fiber_plane")
            with m6:
                st.metric("eta_coupling_conditional", f"{pa_insp.eta_coupling_conditional * 100.0:.1f}%", help="P_inside_core_AND_NA / P_at_fiber_plane")

        st.markdown("#### 3D Interactive Ray Trace Scene")
        fig_opt_3d = render_3d_system_figure(
            inspected_res.system,
            inspected_res.traced_bundle,
            max_rays=200,
            title=f"3D Ray Trace Scene (N = {selected_insp_n} Channels)",
        )
        st.plotly_chart(fig_opt_3d, use_container_width=True)

        st.markdown("### 6. Physical Component Specifications")
        tab_spec_s, tab_spec_p, tab_spec_rf = st.tabs([
            "Slicer Mirrors (Input Image Plane)",
            "Pupil Relay Mirrors (One-Sided Cluster)",
            "Recombination Lens & Fiber",
        ])
        with tab_spec_s:
            st.dataframe(inspected_res.slicer_table, use_container_width=True)
        with tab_spec_p:
            st.dataframe(inspected_res.pupil_table, use_container_width=True)
        with tab_spec_rf:
            rf_dict = {
                "Component / Subsystem": [
                    "Input Image Plane z",
                    "Pupil Relay Mirror Distance L",
                    "Condenser Lens Center",
                    "Condenser Lens Focal Length",
                    "Multimode Fiber Center",
                    "Fiber Axis Direction",
                    "Fiber Core Diameter / Radius",
                    "Fiber Numerical Aperture (NA)",
                    "Acceptance Half-Angle in Air",
                ],
                "Specification": [
                    f"{inspected_res.system.slicer.z:.1f} mm",
                    f"{inspected_res.system.pupil_relay.mirrors[0].center[2] - inspected_res.system.slicer.z:.1f} mm",
                    f"({inspected_res.system.final_lens_3d.center[0]:.2f}, {inspected_res.system.final_lens_3d.center[1]:.2f}, {inspected_res.system.final_lens_3d.center[2]:.2f}) mm",
                    f"f = {inspected_res.system.final_lens_3d.focal_length:.1f} mm",
                    f"({inspected_res.system.fiber.position[0]:.2f}, {inspected_res.system.fiber.position[1]:.2f}, {inspected_res.system.fiber.position[2]:.2f}) mm",
                    f"({inspected_res.system.fiber.axis[0]:.3f}, {inspected_res.system.fiber.axis[1]:.3f}, {inspected_res.system.fiber.axis[2]:.3f})",
                    "1.0 mm (r = 0.5 mm)",
                    "NA = 0.22",
                    "θ_max = 12.71°",
                ],
            }
            st.dataframe(pd.DataFrame(rf_dict), use_container_width=True)

        st.markdown("### 6. Fiber Coupling Diagnostics")
        diag_c1, diag_c2 = st.columns(2)
        with diag_c1:
            fig_sp = render_fiber_spot_figure(inspected_res.coupling_result, title=f"Fiber Face Spot Distribution (N = {selected_insp_n})")
            st.plotly_chart(fig_sp, use_container_width=True)
        with diag_c2:
            fig_ag = render_fiber_angular_figure(inspected_res.coupling_result, title=f"Angular Acceptance Phase Space (N = {selected_insp_n})")
            st.plotly_chart(fig_ag, use_container_width=True)

        st.markdown("### 7. Optical Power Loss Budget")
        fig_water = render_loss_waterfall_figure(inspected_res.loss_budget, title=f"Power Loss Waterfall (N = {selected_insp_n})")
        st.plotly_chart(fig_water, use_container_width=True)

    # Stop execution when in optimizer mode so manual trace and 10 tabs are skipped!
    st.stop()


# ==========================================
# INTERACTIVE BENCH ALIGNMENT MODE VIEW
# ==========================================
current_system = get_current_system()
source_bundle = get_initial_bundle(
    st.session_state.source_mode,
    st.session_state.n_rays,
    st.session_state.random_seed,
)

# Execute End-to-End Trace
traced_bundle, coupling_res, metrics = current_system.trace(source_bundle)

# ==========================================
# TOP KPI DASHBOARD
# ==========================================
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(
        label="Coupling Efficiency",
        value=f"{coupling_res.geometric_coupling_efficiency * 100.0:.2f}%",
        help="Accepted ray power / Total launched source power",
    )
with kpi2:
    st.metric(
        label="Fiber Plane Power",
        value=f"{coupling_res.power_reaching_fiber_plane * 100.0:.1f}%",
        help="Power reaching the 3D fiber face plane",
    )
with kpi3:
    st.metric(
        label="Inside Core (r <= 0.5mm)",
        value=f"{metrics.fraction_inside_core * 100.0:.1f}%",
        help="Fraction of fiber plane power inside 1.0 mm core",
    )
with kpi4:
    st.metric(
        label="Inside NA (θ <= 12.7°)",
        value=f"{metrics.fraction_inside_na * 100.0:.1f}%",
        help="Fraction of fiber plane power inside NA = 0.22",
    )
with kpi5:
    st.metric(
        label="Cross-Channel Stray",
        value=f"{current_system.cross_channel_hits}",
        help="Rays from channel i that struck pupil mirror j (i != j)",
    )

st.write("")

# ==========================================
# 10 APPLICATION TABS
# ==========================================
tab_setup, tab_trace, tab_slicer, tab_manual_align, tab_pupil, tab_fiber, tab_compare, tab_opt, tab_sweeps, tab_val = st.tabs([
    "1. System Setup",
    "2. 3D Ray Trace Scene",
    "3. Slicer Plane",
    "4. Manual Channel Alignment",
    "5. Pupil Plane & Auto-Tools",
    "6. Fiber Coupling",
    "7. Compare Architectures",
    "8. Optimization",
    "9. Parameter Sweeps",
    "10. Validation & Étendue",
])

# --------------------------------------------------
# TAB 1: SYSTEM SETUP
# --------------------------------------------------
with tab_setup:
    st.subheader("System Component Parameters & 3D Poses")
    st.info(
        "**Pre-Slicer:** Sequential optics along nominal z optical axis.\n"
        "**Post-Slicer:** 3D Branched non-sequential scene with dedicated pupil mirror poses and common final lens."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Light Source & Fore-Optics")
        st.text_input("Source Mode", value=f"{st.session_state.source_mode} Mode (Seed: {st.session_state.random_seed})", disabled=True)
        aperture_diam = st.number_input("Entrance Aperture Diameter (mm)", value=20.0, step=1.0)
        l1_f = st.number_input("Lens 1 Focal Length (mm)", value=100.0, step=5.0)
        l1_z = st.number_input("Lens 1 z-position (mm) [placeholder]", value=60.0, step=5.0)
        l2_f = st.number_input("Lens 2 Focal Length (mm)", value=35.0, step=5.0)
        l2_z = st.number_input("Lens 2 z-position (mm) [placeholder]", value=140.0, step=5.0)

    with c2:
        st.markdown("#### Slicer & 3D Post-Slicer Elements")
        slicer_z = st.number_input("Slicer Array z (mm)", value=190.0, step=5.0)
        p_dist = st.number_input("Pupil Mirror Distance L (mm)", value=float(st.session_state.pupil_dist), step=5.0)
        final_f = st.number_input("Common Final Lens Focal Length (mm)", value=75.0, step=5.0)
        final_z = st.number_input("Common Final Lens z (mm)", value=310.0, step=5.0)
        fib_z = st.number_input("Fiber Face z (mm)", value=385.0, step=5.0)

    st.divider()
    st.markdown("#### Save / Export Configuration")
    config_dict = {
        "preset_name": st.session_state.preset_name,
        "source_mode": st.session_state.source_mode,
        "n_rays": st.session_state.n_rays,
        "pupil_distance_mm": p_dist,
        "pupil_power_enabled": st.session_state.pupil_power_enabled,
        "final_lens": {"f_mm": final_f, "z_mm": final_z},
        "fiber": {"core_diam_mm": 1.0, "na": 0.22, "z_mm": fib_z},
    }
    st.download_button(
        label="Export Configuration to JSON",
        data=json.dumps(config_dict, indent=2),
        file_name="branched_slicer_simulation.json",
        mime="application/json",
    )


# --------------------------------------------------
# TAB 2: 3D RAY TRACE SCENE (PRIMARY VIEW)
# --------------------------------------------------
with tab_trace:
    st.subheader("3D Interactive Optical Scene & Projections")
    st.markdown(
        "Renders the complete 3D branched geometry: Pre-slicer train, Slicer reflection branches, "
        "Pupil mirrors, Common final coupling lens, and Multimode fiber."
    )

    t_ctrl1, t_ctrl2, t_ctrl3 = st.columns([1, 1, 2])
    with t_ctrl1:
        view_mode = st.selectbox(
            "Visualization Mode",
            ["3D Interactive Scene", "X-Z Projection (Horizontal)", "Y-Z Projection (Vertical)", "X-Y Top View"],
            index=0,
        )
        n_slices_avail = len(current_system.slicer.slices) if current_system.slicer else 0
        focus_opts = ["Show All Channels"] + [f"Channel {i+1} Only" for i in range(n_slices_avail)]
        focus_choice = st.selectbox("Channel Focus / Isolation", focus_opts, index=0)
        isolated_ch = int(focus_choice.split()[1]) - 1 if focus_choice != "Show All Channels" else None

    with t_ctrl2:
        ray_display_mode = st.radio(
            "Ray Bundle Selection",
            ["Show chief rays only", "Show full ray bundles"],
            index=1,
            horizontal=True,
        )
    with t_ctrl3:
        n_render_rays = st.slider("Rays to Render", 10, 300, 60, step=10)
        show_elements = st.checkbox("Render 3D Optical Element Geometries", value=True)
        show_normals = st.checkbox("Show Surface Normal Vectors", value=False)

    # 3D Plotly Figure
    fig_scene = go.Figure()

    # Determine rays to plot
    if ray_display_mode == "Show chief rays only":
        # Select one ray per channel
        ray_indices = []
        for ch in np.unique(traced_bundle.channel_id):
            if ch >= 0:
                match = np.where((traced_bundle.channel_id == ch) & traced_bundle.active_mask)[0]
                if len(match) > 0:
                    ray_indices.append(match[0])
        if not ray_indices:
            ray_indices = list(range(min(4, traced_bundle.n_rays)))
    else:
        ray_indices = list(range(min(n_render_rays, traced_bundle.n_rays)))

    # Trace trajectories from history
    n_planes = len(traced_bundle.history)
    colors = px.colors.qualitative.Plotly

    for r_i in ray_indices:
        z_traj = [traced_bundle.history[p]["r"][r_i, 2] for p in range(n_planes)]
        x_traj = [traced_bundle.history[p]["r"][r_i, 0] for p in range(n_planes)]
        y_traj = [traced_bundle.history[p]["r"][r_i, 1] for p in range(n_planes)]
        ch = traced_bundle.channel_id[r_i]
        is_faded = (isolated_ch is not None and ch != isolated_ch)
        ray_opacity = 0.08 if is_faded else (0.9 if isolated_ch is not None else 0.6)
        ray_width = 1.0 if is_faded else (3.0 if isolated_ch is not None else 2.0)
        ray_col = colors[ch % len(colors)] if ch >= 0 else "#888888"

        if view_mode == "3D Interactive Scene":
            fig_scene.add_trace(
                go.Scatter3d(
                    x=x_traj,
                    y=y_traj,
                    z=z_traj,
                    mode="lines",
                    line=dict(color=ray_col, width=ray_width),
                    opacity=ray_opacity,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        elif view_mode == "X-Z Projection (Horizontal)":
            fig_scene.add_trace(
                go.Scatter(
                    x=z_traj,
                    y=x_traj,
                    mode="lines",
                    line=dict(color=ray_col, width=1.2),
                    opacity=0.6,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        elif view_mode == "Y-Z Projection (Vertical)":
            fig_scene.add_trace(
                go.Scatter(
                    x=z_traj,
                    y=y_traj,
                    mode="lines",
                    line=dict(color=ray_col, width=1.2),
                    opacity=0.6,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:  # X-Y Top View
            fig_scene.add_trace(
                go.Scatter(
                    x=x_traj,
                    y=y_traj,
                    mode="lines",
                    line=dict(color=ray_col, width=1.2),
                    opacity=0.6,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Render Physical Optical Elements in 3D
    if show_elements and view_mode == "3D Interactive Scene":
        # Helper to draw circular disks
        def add_3d_disk(center, normal, radius, name, color="rgba(70, 130, 180, 0.25)"):
            u, v, w = build_orthonormal_basis(normal)
            theta = np.linspace(0, 2 * np.pi, 28)
            x_pts = center[0] + radius * (np.cos(theta) * u[0] + np.sin(theta) * v[0])
            y_pts = center[1] + radius * (np.cos(theta) * u[1] + np.sin(theta) * v[1])
            z_pts = center[2] + radius * (np.cos(theta) * u[2] + np.sin(theta) * v[2])
            fig_scene.add_trace(
                go.Scatter3d(
                    x=x_pts,
                    y=y_pts,
                    z=z_pts,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=name,
                )
            )

        # Helper to draw oriented rectangles
        def add_3d_rect(center, normal, width, height, name, color="rgba(255, 140, 0, 0.4)"):
            u, v, w = build_orthonormal_basis(normal)
            hw, hh = width / 2.0, height / 2.0
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)]
            x_pts = [center[0] + cu * u[0] + cv * v[0] for cu, cv in corners]
            y_pts = [center[1] + cu * u[1] + cv * v[1] for cu, cv in corners]
            z_pts = [center[2] + cu * u[2] + cv * v[2] for cu, cv in corners]
            fig_scene.add_trace(
                go.Scatter3d(
                    x=x_pts,
                    y=y_pts,
                    z=z_pts,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=name,
                )
            )

        # 1. Fore-optics lenses
        for elem in current_system.fore_optics:
            if isinstance(elem, ThinLens):
                add_3d_disk([0, 0, elem.z], [0, 0, 1], elem.radius, elem.name[:16])

        # 2. Slicer Mirrors
        if current_system.slicer is not None:
            for s in current_system.slicer.slices:
                if s.enabled:
                    is_active = (isolated_ch is None or s.slice_id == isolated_ch)
                    s_col = "#00D2FF" if (isolated_ch is not None and is_active) else ("#1f77b4" if is_active else "rgba(100, 100, 100, 0.2)")
                    add_3d_rect(s.center, s.normal, s.width, s.height, f"Slice S{s.slice_id + 1}", color=s_col)

        # 3. Pupil Mirrors
        if current_system.pupil_relay is not None:
            for m in current_system.pupil_relay.mirrors:
                if m.enabled:
                    is_active = (isolated_ch is None or m.channel_id == isolated_ch)
                    m_col = "#FF9F1C" if (isolated_ch is not None and is_active) else ("#ff7f0e" if is_active else "rgba(100, 100, 100, 0.2)")
                    add_3d_rect(m.center, m.normal, m.width, m.height, m.name, color=m_col)

        # 4. Common Final Lens
        if current_system.final_lens_3d is not None:
            add_3d_disk(
                current_system.final_lens_3d.center,
                current_system.final_lens_3d.normal,
                current_system.final_lens_3d.radius,
                "Final Coupling Lens",
                color="#2ca02c",
            )

        # 5. Fiber Face
        add_3d_disk(
            current_system.fiber.position,
            current_system.fiber.axis,
            current_system.fiber.core_radius,
            "Fiber Core (1mm)",
            color="#d62728",
        )

        # 6. Reference Axes: Original Source Axis & New Common Relay Axis
        z_slicer_end = float(current_system.slicer.z + 20.0) if current_system.slicer is not None else 250.0
        fig_scene.add_trace(
            go.Scatter3d(
                x=[0.0, 0.0],
                y=[0.0, 0.0],
                z=[0.0, z_slicer_end],
                mode="lines",
                line=dict(color="#888888", width=3, dash="dash"),
                name="Original Source Axis",
            )
        )

        if hasattr(current_system, "relay_axis_origin") and hasattr(current_system, "relay_axis_direction"):
            p_orig = current_system.relay_axis_origin
            a_rel = current_system.relay_axis_direction
            pt_start = p_orig - 15.0 * a_rel
            pt_end = current_system.fiber.position + 30.0 * a_rel
            fig_scene.add_trace(
                go.Scatter3d(
                    x=[pt_start[0], pt_end[0]],
                    y=[pt_start[1], pt_end[1]],
                    z=[pt_start[2], pt_end[2]],
                    mode="lines",
                    line=dict(color="#800080", width=4, dash="dot"),
                    name="New Common Relay Axis",
                )
            )

        # Optional Surface Normals
        if show_normals:
            if current_system.slicer is not None:
                for s in current_system.slicer.slices:
                    n_end = s.center + 15.0 * s.normal
                    fig_scene.add_trace(
                        go.Scatter3d(
                            x=[s.center[0], n_end[0]],
                            y=[s.center[1], n_end[1]],
                            z=[s.center[2], n_end[2]],
                            mode="lines",
                            line=dict(color="blue", width=2),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
            if current_system.pupil_relay is not None:
                for m in current_system.pupil_relay.mirrors:
                    n_end = m.center + 15.0 * m.normal
                    fig_scene.add_trace(
                        go.Scatter3d(
                            x=[m.center[0], n_end[0]],
                            y=[m.center[1], n_end[1]],
                            z=[m.center[2], n_end[2]],
                            mode="lines",
                            line=dict(color="orange", width=2),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

    # Layout styling
    if view_mode == "3D Interactive Scene":
        fig_scene.update_layout(
            title="3D Optical Bench Scene View",
            scene=dict(
                xaxis_title="Transverse x (mm)",
                yaxis_title="Transverse y (mm)",
                zaxis_title="Optical Axis z (mm)",
                aspectmode="data",
            ),
            template="plotly_white",
            height=680,
        )
    else:
        x_label = "z (mm)" if "Z" in view_mode else "x (mm)"
        y_label = "x (mm)" if "X-Z" in view_mode else ("y (mm)" if "Y-Z" in view_mode else "y (mm)")
        fig_scene.update_layout(
            title=f"2D Projection View: {view_mode}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            height=540,
        )

    st.plotly_chart(fig_scene, use_container_width=True)


# --------------------------------------------------
# TAB 3: SLICER PLANE
# --------------------------------------------------
with tab_slicer:
    st.subheader("Input Focal Plane & Slice Partitioning")
    st.markdown(
        "At the input focal plane, rectangular mirrors section the extended field into discrete channels, "
        "imparting unique 3D reflection angles to steer each channel along its own branch."
    )

    s_c1, s_c2 = st.columns([1, 1])
    with s_c1:
        st.markdown("#### Slicer Mirror Footprints & Incident Rays")
        fig_s_plane = go.Figure()
        if current_system.slicer is not None:
            for s in current_system.slicer.slices:
                hw, hh = s.width / 2.0, s.height / 2.0
                corners = [
                    (s.center_x - hw, s.center_y - hh),
                    (s.center_x + hw, s.center_y - hh),
                    (s.center_x + hw, s.center_y + hh),
                    (s.center_x - hw, s.center_y + hh),
                    (s.center_x - hw, s.center_y - hh),
                ]
                fig_s_plane.add_trace(
                    go.Scatter(
                        x=[c[0] for c in corners],
                        y=[c[1] for c in corners],
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(31, 119, 180, 0.15)",
                        line=dict(color="#1f77b4", width=2),
                        name=f"Slice S{s.slice_id + 1} (Tip={s.tip_x_deg:.1f}°, Tilt={s.tilt_y_deg:.1f}°)",
                    )
                )

            inc_snap = next((h for h in traced_bundle.history if "Incident" in h["label"]), None)
            if inc_snap is not None:
                sub_n = min(1500, len(inc_snap["r"]))
                fig_s_plane.add_trace(
                    go.Scatter(
                        x=inc_snap["r"][:sub_n, 0],
                        y=inc_snap["r"][:sub_n, 1],
                        mode="markers",
                        marker=dict(size=3, color="orange", opacity=0.5),
                        name="Incident Rays",
                    )
                )

        fig_s_plane.update_layout(
            title="Slicer Plane Active Tiles",
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            template="plotly_white",
            height=440,
            xaxis=dict(scaleanchor="y", scaleratio=1),
        )
        st.plotly_chart(fig_s_plane, use_container_width=True)

    with s_c2:
        st.markdown("#### Scan Input Beam Size vs z")
        st.caption("Locate input focal plane or desired footprint dimension:")
        scan_z_min = st.number_input("Scan z Min (mm)", value=120.0, step=10.0)
        scan_z_max = st.number_input("Scan z Max (mm)", value=240.0, step=10.0)
        target_fp = st.number_input("Target Footprint Width (mm)", value=20.0, step=1.0)

        if st.button("Run Beam Scan vs z", use_container_width=True):
            scan_data = scan_beam_size_vs_z(source_bundle, scan_z_min, scan_z_max, n_points=40)
            best_fp_z = find_z_for_target_footprint(source_bundle, target_fp, scan_z_min, scan_z_max)

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=scan_data["z"], y=scan_data["rms_radius"], mode="lines", name="RMS Radius (mm)"))
            fig_sc.add_trace(go.Scatter(x=scan_data["z"], y=scan_data["bbox_width"], mode="lines", name="BBox Width (mm)"))
            fig_sc.add_vline(x=best_fp_z, line_dash="dash", line_color="green", annotation_text=f"Target at z={best_fp_z:.1f}")
            fig_sc.update_layout(
                title="Transverse Size vs z",
                xaxis_title="z (mm)",
                yaxis_title="Size (mm)",
                template="plotly_white",
                height=340,
            )
            st.plotly_chart(fig_sc, use_container_width=True)
            st.success(f"Best z for target {target_fp:.1f} mm footprint: **{best_fp_z:.1f} mm**")


# --------------------------------------------------
# TAB 4: MANUAL CHANNEL ALIGNMENT WORKBENCH
# --------------------------------------------------
with tab_manual_align:
    st.subheader("Manual Slicer ↔ Pupil Alignment Workbench")
    st.markdown(
        "Directly align each slicer mirror with its assigned pupil mirror as on an optical bench. "
        "Select a channel to isolate its beam path, adjust slicer or pupil poses with multi-scale precision "
        "(down to 0.001°), and inspect real-time 2D face hit coordinates, reflection diagnostics, and condenser lens coupling."
    )

    bench_reopt_c1, bench_reopt_c2 = st.columns([3, 1])
    with bench_reopt_c1:
        st.caption("Numerical Bench Refinement: Polish current manual slicer and pupil mirror poses using local optimization to maximize fiber coupling.")
    with bench_reopt_c2:
        if st.button("Re-Optimize from Current Geometry", key="btn_reopt_manual_tab", use_container_width=True):
            with st.spinner("Polishing manual bench geometry..."):
                ok, old_e, new_e = reoptimize_manual_geometry(current_system, st.session_state.manual_channels)
                if ok:
                    st.success(f"Bench refinement complete! Coupling efficiency: {old_e*100.0:.2f}% -> {new_e*100.0:.2f}%.")
                    st.rerun()
                else:
                    st.error("Failed to polish current geometry.")

    if current_system.slicer is None or current_system.pupil_relay is None or len(current_system.slicer.slices) == 0:
        st.info("Manual Channel Alignment requires a branched image slicer preset (2-Channel or 4-Channel).")
    else:
        n_slices = len(current_system.slicer.slices)
        ch_options = [f"Channel {i+1} (Slice S{i+1} → Pupil P{i+1})" for i in range(n_slices)]

        # Top Control Bar: Channel Selection, Alignment Mode, and Locks
        top_c1, top_c2, top_c3 = st.columns([2, 2, 2])
        with top_c1:
            if "manual_selected_ch" not in st.session_state or st.session_state.manual_selected_ch >= n_slices:
                st.session_state.manual_selected_ch = 0
            
            chosen_ch_str = st.radio(
                "Active Alignment Channel",
                ch_options,
                index=st.session_state.manual_selected_ch,
                horizontal=True,
            )
            selected_ch_idx = ch_options.index(chosen_ch_str)
            st.session_state.manual_selected_ch = selected_ch_idx

        with top_c2:
            align_mode = st.radio(
                "Bench Alignment Mode",
                [
                    "Mode A: Adjust Slicer to Fixed Pupil",
                    "Mode B: Adjust Pupil to Slicer Ray",
                ],
                index=0 if "Mode A" in st.session_state.manual_align_mode else 1,
                help="Mode A fixes the pupil mirror and adjusts slicer tilt to hit the pupil center. Mode B fixes the slicer and translates/tilts the pupil mirror to catch and redirect the ray.",
            )
            st.session_state.manual_align_mode = align_mode

        with top_c3:
            st.markdown("**Channel Lock Controls**")
            ch_data = st.session_state.manual_channels.get(selected_ch_idx, {})
            lock_slicer = st.checkbox(
                f"Lock Slicer S{selected_ch_idx+1} (Prevent changes)",
                value=ch_data.get("slicer_locked", False),
                key=f"lock_slicer_{selected_ch_idx}",
            )
            lock_pupil = st.checkbox(
                f"Lock Pupil P{selected_ch_idx+1} (Prevent changes)",
                value=ch_data.get("pupil_locked", False),
                key=f"lock_pupil_{selected_ch_idx}",
            )
            st.session_state.manual_channels[selected_ch_idx]["slicer_locked"] = lock_slicer
            st.session_state.manual_channels[selected_ch_idx]["pupil_locked"] = lock_pupil

        st.divider()

        # Active elements for chosen channel
        curr_slice = current_system.slicer.slices[selected_ch_idx]
        curr_pupil = current_system.pupil_relay.mirrors[selected_ch_idx]

        # Chief ray trace from slicer center
        k_in_nom = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        k_ref_s = curr_slice.get_reflected_chief_ray(k_in_nom)
        r_s_center = curr_slice.center

        # Stage 1: Intersect with Pupil Mirror
        pupil_u, pupil_v, pupil_miss, pupil_in_bounds, pupil_hit_pos = curr_pupil.get_local_hit(
            r_in=r_s_center,
            k_in=k_ref_s,
        )

        # Stage 2: Pupil reflection toward Condenser Lens
        k_pupil_ref = reflect_vector(k_ref_s, curr_pupil.normal)
        cond_u, cond_v, cond_miss, cond_in_bounds, cond_hit_pos = (0.0, 0.0, 0.0, True, np.zeros(3))
        if current_system.final_lens_3d is not None:
            cond_u, cond_v, cond_miss, cond_in_bounds, cond_hit_pos = current_system.final_lens_3d.get_local_hit(
                r_in=pupil_hit_pos,
                k_in=k_pupil_ref,
            )

        # Slicer Diagnostics
        diag = curr_slice.get_reflection_diagnostics(k_in=k_in_nom, target_pos=curr_pupil.center)
        ideal_tip, ideal_tilt = curr_slice.get_ideal_angles_for_target(curr_pupil.center, k_in=k_in_nom)
        delta_tip = curr_slice.tip_x_deg - ideal_tip
        delta_tilt = curr_slice.tilt_y_deg - ideal_tilt

        # ----------------------------------------------------
        # Helper Callbacks for Dual Widget Synchronization
        # ----------------------------------------------------
        def make_sync_callbacks(ch_idx, category, param_name, min_v=-35.0, max_v=35.0):
            num_k = f"num_{category}_{param_name}_{ch_idx}"
            slide_k = f"slide_{category}_{param_name}_{ch_idx}"
            
            def from_num():
                val = float(st.session_state[num_k])
                st.session_state.manual_channels[ch_idx][param_name] = val
                st.session_state[slide_k] = float(np.clip(val, min_v, max_v))

            def from_slide():
                val = float(st.session_state[slide_k])
                st.session_state.manual_channels[ch_idx][param_name] = val
                st.session_state[num_k] = val

            return from_num, from_slide

        def step_angle(ch_idx, category, param_name, delta, min_v=-35.0, max_v=35.0):
            cur = float(st.session_state.manual_channels[ch_idx][param_name])
            new_val = round(cur + delta, 5)
            st.session_state.manual_channels[ch_idx][param_name] = new_val
            num_k = f"num_{category}_{param_name}_{ch_idx}"
            slide_k = f"slide_{category}_{param_name}_{ch_idx}"
            st.session_state[num_k] = new_val
            st.session_state[slide_k] = float(np.clip(new_val, min_v, max_v))

        def step_pos(ch_idx, category, coord_idx, delta):
            param = f"{category}_pos"
            st.session_state.manual_channels[ch_idx][param][coord_idx] = round(
                st.session_state.manual_channels[ch_idx][param][coord_idx] + delta, 3
            )
            axis_name = ["x", "y", "z"][coord_idx]
            w_k = f"pos_{category}_{axis_name}_{ch_idx}"
            st.session_state[w_k] = st.session_state.manual_channels[ch_idx][param][coord_idx]

        # ----------------------------------------------------
        # Two-Column Layout: Controls (Left) & Diagnostics (Right)
        # ----------------------------------------------------
        col_ctrl, col_diag = st.columns([1, 1])

        with col_ctrl:
            st.markdown(f"### Manual Bench Controls: Channel {selected_ch_idx+1}")

            # SECTION 1: SLICER CONTROLS
            st.markdown(f"#### Slicer Mirror S{selected_ch_idx+1} Orientation & Piston")
            if lock_slicer:
                st.info(f"Slicer S{selected_ch_idx+1} is LOCKED. Unlock above to modify parameters.")

            # Slicer Tip X
            st.markdown("**Tip X (Rotation about X axis - steers along Y)**")
            st.caption("Quick step increments (deg):")
            step_cols1 = st.columns(8)
            steps = [-1.0, -0.1, -0.01, -0.001, +0.001, +0.01, +0.1, +1.0]
            step_labels = ["-1°", "-0.1°", "-0.01°", "-0.001°", "+0.001°", "+0.01°", "+0.1°", "+1°"]
            for s_col, s_v, s_l in zip(step_cols1, steps, step_labels):
                with s_col:
                    st.button(
                        s_l,
                        key=f"btn_stip_{s_l}_{selected_ch_idx}",
                        on_click=step_angle,
                        args=(selected_ch_idx, "slicer", "slicer_tip_x", s_v),
                        disabled=lock_slicer,
                        use_container_width=True,
                    )
            
            num_stip_k = f"num_slicer_slicer_tip_x_{selected_ch_idx}"
            slide_stip_k = f"slide_slicer_slicer_tip_x_{selected_ch_idx}"
            if num_stip_k not in st.session_state:
                st.session_state[num_stip_k] = float(ch_data["slicer_tip_x"])
            if slide_stip_k not in st.session_state:
                st.session_state[slide_stip_k] = float(np.clip(ch_data["slicer_tip_x"], -35.0, 35.0))
            
            stip_fn_num, stip_fn_slide = make_sync_callbacks(selected_ch_idx, "slicer", "slicer_tip_x")
            
            sc_num1, sc_sl1 = st.columns([1, 2])
            with sc_num1:
                st.number_input(
                    "Tip X (deg)",
                    value=float(st.session_state[num_stip_k]),
                    step=0.001,
                    format="%.4f",
                    key=num_stip_k,
                    disabled=lock_slicer,
                    on_change=stip_fn_num,
                )
            with sc_sl1:
                st.slider(
                    "Tip X Slider",
                    min_value=-35.0,
                    max_value=35.0,
                    value=float(st.session_state[slide_stip_k]),
                    step=0.01,
                    key=slide_stip_k,
                    disabled=lock_slicer,
                    on_change=stip_fn_slide,
                    label_visibility="collapsed",
                )

            # Slicer Tilt Y
            st.markdown("**Tilt Y (Rotation about Y axis - steers along X)**")
            st.caption("Quick step increments (deg):")
            step_cols2 = st.columns(8)
            for s_col, s_v, s_l in zip(step_cols2, steps, step_labels):
                with s_col:
                    st.button(
                        s_l,
                        key=f"btn_stilt_{s_l}_{selected_ch_idx}",
                        on_click=step_angle,
                        args=(selected_ch_idx, "slicer", "slicer_tilt_y", s_v),
                        disabled=lock_slicer,
                        use_container_width=True,
                    )

            num_stilt_k = f"num_slicer_slicer_tilt_y_{selected_ch_idx}"
            slide_stilt_k = f"slide_slicer_slicer_tilt_y_{selected_ch_idx}"
            if num_stilt_k not in st.session_state:
                st.session_state[num_stilt_k] = float(ch_data["slicer_tilt_y"])
            if slide_stilt_k not in st.session_state:
                st.session_state[slide_stilt_k] = float(np.clip(ch_data["slicer_tilt_y"], -35.0, 35.0))

            stilt_fn_num, stilt_fn_slide = make_sync_callbacks(selected_ch_idx, "slicer", "slicer_tilt_y")

            sc_num2, sc_sl2 = st.columns([1, 2])
            with sc_num2:
                st.number_input(
                    "Tilt Y (deg)",
                    value=float(st.session_state[num_stilt_k]),
                    step=0.001,
                    format="%.4f",
                    key=num_stilt_k,
                    disabled=lock_slicer,
                    on_change=stilt_fn_num,
                )
            with sc_sl2:
                st.slider(
                    "Tilt Y Slider",
                    min_value=-35.0,
                    max_value=35.0,
                    value=float(st.session_state[slide_stilt_k]),
                    step=0.01,
                    key=slide_stilt_k,
                    disabled=lock_slicer,
                    on_change=stilt_fn_slide,
                    label_visibility="collapsed",
                )

            # Slicer Translation Controls
            with st.expander(f"Slicer Mirror S{selected_ch_idx+1} Translation (X, Y, Z)", expanded=False):
                sp_x_k = f"pos_slicer_x_{selected_ch_idx}"
                sp_y_k = f"pos_slicer_y_{selected_ch_idx}"
                sp_z_k = f"pos_slicer_z_{selected_ch_idx}"
                if sp_x_k not in st.session_state:
                    st.session_state[sp_x_k] = float(ch_data["slicer_pos"][0])
                if sp_y_k not in st.session_state:
                    st.session_state[sp_y_k] = float(ch_data["slicer_pos"][1])
                if sp_z_k not in st.session_state:
                    st.session_state[sp_z_k] = float(ch_data["slicer_pos"][2])

                c_sx, c_sy, c_sz = st.columns(3)
                with c_sx:
                    st.number_input("Center X (mm)", value=float(st.session_state[sp_x_k]), step=0.05, format="%.3f", key=sp_x_k, disabled=lock_slicer, on_change=lambda: st.session_state.manual_channels[selected_ch_idx]["slicer_pos"].__setitem__(0, st.session_state[sp_x_k]))
                with c_sy:
                    st.number_input("Center Y (mm)", value=float(st.session_state[sp_y_k]), step=0.05, format="%.3f", key=sp_y_k, disabled=lock_slicer, on_change=lambda: st.session_state.manual_channels[selected_ch_idx]["slicer_pos"].__setitem__(1, st.session_state[sp_y_k]))
                with c_sz:
                    st.number_input("Piston Z (mm)", value=float(st.session_state[sp_z_k]), step=0.1, format="%.3f", key=sp_z_k, disabled=lock_slicer, on_change=lambda: st.session_state.manual_channels[selected_ch_idx]["slicer_pos"].__setitem__(2, st.session_state[sp_z_k]))

            st.divider()

            # SECTION 2: PUPIL CONTROLS
            st.markdown(f"#### Pupil Mirror P{selected_ch_idx+1} Position & Orientation")
            if lock_pupil:
                st.info(f"Pupil P{selected_ch_idx+1} is LOCKED. Unlock above to modify parameters.")

            # Pupil Tip X
            st.markdown("**Pupil Tip X (Rotation about X axis)**")
            st.caption("Quick step increments (deg):")
            step_cols_pt = st.columns(8)
            for s_col, s_v, s_l in zip(step_cols_pt, steps, step_labels):
                with s_col:
                    st.button(
                        s_l,
                        key=f"btn_ptip_{s_l}_{selected_ch_idx}",
                        on_click=step_angle,
                        args=(selected_ch_idx, "pupil", "pupil_tip_x", s_v),
                        disabled=lock_pupil,
                        use_container_width=True,
                    )

            num_ptip_k = f"num_pupil_pupil_tip_x_{selected_ch_idx}"
            slide_ptip_k = f"slide_pupil_pupil_tip_x_{selected_ch_idx}"
            if num_ptip_k not in st.session_state:
                st.session_state[num_ptip_k] = float(ch_data["pupil_tip_x"])
            if slide_ptip_k not in st.session_state:
                st.session_state[slide_ptip_k] = float(np.clip(ch_data["pupil_tip_x"], -35.0, 35.0))

            ptip_fn_num, ptip_fn_slide = make_sync_callbacks(selected_ch_idx, "pupil", "pupil_tip_x")

            pc_num1, pc_sl1 = st.columns([1, 2])
            with pc_num1:
                st.number_input(
                    "Pupil Tip X (deg)",
                    value=float(st.session_state[num_ptip_k]),
                    step=0.001,
                    format="%.4f",
                    key=num_ptip_k,
                    disabled=lock_pupil,
                    on_change=ptip_fn_num,
                )
            with pc_sl1:
                st.slider(
                    "Pupil Tip X Slider",
                    min_value=-35.0,
                    max_value=35.0,
                    value=float(st.session_state[slide_ptip_k]),
                    step=0.01,
                    key=slide_ptip_k,
                    disabled=lock_pupil,
                    on_change=ptip_fn_slide,
                    label_visibility="collapsed",
                )

            # Pupil Tilt Y
            st.markdown("**Pupil Tilt Y (Rotation about Y axis)**")
            st.caption("Quick step increments (deg):")
            step_cols_ptilt = st.columns(8)
            for s_col, s_v, s_l in zip(step_cols_ptilt, steps, step_labels):
                with s_col:
                    st.button(
                        s_l,
                        key=f"btn_ptilt_{s_l}_{selected_ch_idx}",
                        on_click=step_angle,
                        args=(selected_ch_idx, "pupil", "pupil_tilt_y", s_v),
                        disabled=lock_pupil,
                        use_container_width=True,
                    )

            num_ptilt_k = f"num_pupil_pupil_tilt_y_{selected_ch_idx}"
            slide_ptilt_k = f"slide_pupil_pupil_tilt_y_{selected_ch_idx}"
            if num_ptilt_k not in st.session_state:
                st.session_state[num_ptilt_k] = float(ch_data["pupil_tilt_y"])
            if slide_ptilt_k not in st.session_state:
                st.session_state[slide_ptilt_k] = float(np.clip(ch_data["pupil_tilt_y"], -35.0, 35.0))

            ptilt_fn_num, ptilt_fn_slide = make_sync_callbacks(selected_ch_idx, "pupil", "pupil_tilt_y")

            pc_num2, pc_sl2 = st.columns([1, 2])
            with pc_num2:
                st.number_input(
                    "Pupil Tilt Y (deg)",
                    value=float(st.session_state[num_ptilt_k]),
                    step=0.001,
                    format="%.4f",
                    key=num_ptilt_k,
                    disabled=lock_pupil,
                    on_change=ptilt_fn_num,
                )
            with pc_sl2:
                st.slider(
                    "Pupil Tilt Y Slider",
                    min_value=-35.0,
                    max_value=35.0,
                    value=float(st.session_state[slide_ptilt_k]),
                    step=0.01,
                    key=slide_ptilt_k,
                    disabled=lock_pupil,
                    on_change=ptilt_fn_slide,
                    label_visibility="collapsed",
                )

            # Pupil Translation (X, Y, Z)
            with st.expander(f"Pupil Mirror P{selected_ch_idx+1} Translation (X, Y, Z)", expanded=(align_mode.startswith("Mode B"))):
                st.caption("Quick translation steps (mm):")
                pos_steps = [-5.0, -1.0, -0.1, +0.1, +1.0, +5.0]
                pos_step_labels = ["-5mm", "-1mm", "-0.1mm", "+0.1mm", "+1mm", "+5mm"]
                
                # Step buttons for Transverse Y
                st.caption("Transverse Y Translation:")
                p_y_cols = st.columns(6)
                for py_c, pv, pl in zip(p_y_cols, pos_steps, pos_step_labels):
                    with py_c:
                        st.button(
                            pl,
                            key=f"btn_py_{pl}_{selected_ch_idx}",
                            on_click=step_pos,
                            args=(selected_ch_idx, "pupil", 1, pv),
                            disabled=lock_pupil,
                            use_container_width=True,
                        )

                pp_x_k = f"pos_pupil_x_{selected_ch_idx}"
                pp_y_k = f"pos_pupil_y_{selected_ch_idx}"
                pp_z_k = f"pos_pupil_z_{selected_ch_idx}"
                if pp_x_k not in st.session_state:
                    st.session_state[pp_x_k] = float(ch_data["pupil_pos"][0])
                if pp_y_k not in st.session_state:
                    st.session_state[pp_y_k] = float(ch_data["pupil_pos"][1])
                if pp_z_k not in st.session_state:
                    st.session_state[pp_z_k] = float(ch_data["pupil_pos"][2])

                c_px, c_py, c_pz = st.columns(3)
                with c_px:
                    st.number_input("Center X (mm)", value=float(st.session_state[pp_x_k]), step=0.1, format="%.3f", key=pp_x_k, disabled=lock_pupil, on_change=lambda: st.session_state.manual_channels[selected_ch_idx]["pupil_pos"].__setitem__(0, st.session_state[pp_x_k]))
                with c_py:
                    st.number_input("Center Y (mm)", value=float(st.session_state[pp_y_k]), step=0.1, format="%.3f", key=pp_y_k, disabled=lock_pupil, on_change=lambda: st.session_state.manual_channels[selected_ch_idx]["pupil_pos"].__setitem__(1, st.session_state[pp_y_k]))
                with c_pz:
                    st.number_input("Center Z (mm)", value=float(st.session_state[pp_z_k]), step=0.5, format="%.3f", key=pp_z_k, disabled=lock_pupil, on_change=lambda: st.session_state.manual_channels[selected_ch_idx]["pupil_pos"].__setitem__(2, st.session_state[pp_z_k]))

        with col_diag:
            st.markdown(f"### Visual Targets & Diagnostics: Channel {selected_ch_idx+1}")

            # 1. PUPIL FACE 2D TARGET DIAGRAM
            st.markdown(f"#### Pupil Mirror P{selected_ch_idx+1} Face Target Diagram")
            
            # Threshold settings
            with st.expander("Alignment Acceptance Thresholds", expanded=False):
                th_c1, th_c2 = st.columns(2)
                with th_c1:
                    fine_th = st.number_input("Fine Target Threshold (mm)", value=float(st.session_state.manual_fine_thresh), step=0.02, format="%.3f")
                    st.session_state.manual_fine_thresh = fine_th
                with th_c2:
                    coarse_th = st.number_input("Coarse Limit Threshold (mm)", value=float(st.session_state.manual_coarse_thresh), step=0.05, format="%.3f")
                    st.session_state.manual_coarse_thresh = coarse_th

            fine_th = float(st.session_state.manual_fine_thresh)
            coarse_th = float(st.session_state.manual_coarse_thresh)

            if pupil_miss < fine_th:
                badge_html = f'<div style="background-color:#28a745; color:white; padding:6px 14px; border-radius:4px; font-weight:700; display:inline-block; margin-bottom:8px;">ALIGNED: Radial Miss = {pupil_miss:.4f} mm (&lt; {fine_th:.3f} mm)</div>'
            elif pupil_miss < coarse_th:
                badge_html = f'<div style="background-color:#ffc107; color:black; padding:6px 14px; border-radius:4px; font-weight:700; display:inline-block; margin-bottom:8px;">MODERATE ALIGNMENT: Radial Miss = {pupil_miss:.4f} mm (&lt; {coarse_th:.3f} mm)</div>'
            else:
                badge_html = f'<div style="background-color:#dc3545; color:white; padding:6px 14px; border-radius:4px; font-weight:700; display:inline-block; margin-bottom:8px;">MISALIGNED: Radial Miss = {pupil_miss:.4f} mm (&gt;= {coarse_th:.3f} mm)</div>'
            
            st.markdown(badge_html, unsafe_allow_html=True)

            # Target 2D Plotly Figure
            pw = curr_pupil.width
            ph = curr_pupil.height
            hw = pw / 2.0
            hh = ph / 2.0
            
            fig_pupil_face = go.Figure()
            # Mirror Boundary
            fig_pupil_face.add_shape(
                type="rect",
                x0=-hw, y0=-hh, x1=hw, y1=hh,
                line=dict(color="#2B3A42", width=2.5),
                fillcolor="rgba(240, 244, 248, 0.5)",
            )
            # Center Crosshairs
            fig_pupil_face.add_shape(type="line", x0=-hw, y0=0, x1=hw, y1=0, line=dict(color="#888888", width=1, dash="dot"))
            fig_pupil_face.add_shape(type="line", x0=0, y0=-hh, x1=0, y1=hh, line=dict(color="#888888", width=1, dash="dot"))
            
            # Tolerance Rings
            th_angle = np.linspace(0, 2*np.pi, 100)
            fig_pupil_face.add_trace(go.Scatter(
                x=fine_th * np.cos(th_angle),
                y=fine_th * np.sin(th_angle),
                mode="lines",
                line=dict(color="#28a745", width=1.5, dash="dash"),
                name=f"Fine Goal (r={fine_th}mm)",
            ))
            fig_pupil_face.add_trace(go.Scatter(
                x=coarse_th * np.cos(th_angle),
                y=coarse_th * np.sin(th_angle),
                mode="lines",
                line=dict(color="#ffc107", width=1.5, dash="dash"),
                name=f"Coarse Limit (r={coarse_th}mm)",
            ))

            # Target Center Point (0, 0)
            fig_pupil_face.add_trace(go.Scatter(
                x=[0.0], y=[0.0],
                mode="markers",
                marker=dict(symbol="cross", size=12, color="#000000"),
                name="Target Center (0, 0)",
            ))

            # Hit Vector Line
            fig_pupil_face.add_trace(go.Scatter(
                x=[0.0, pupil_u],
                y=[0.0, pupil_v],
                mode="lines",
                line=dict(color="#EF553B", width=2.5),
                name=f"Error Vector ({pupil_miss:.3f} mm)",
            ))

            # Actual Hit Point
            hit_color = "#28a745" if pupil_miss < fine_th else ("#FFA15A" if pupil_miss < coarse_th else "#EF553B")
            fig_pupil_face.add_trace(go.Scatter(
                x=[pupil_u], y=[pupil_v],
                mode="markers+text",
                marker=dict(symbol="circle", size=14, color=hit_color, line=dict(color="#000000", width=1.5)),
                text=[f"Hit: u={pupil_u:.3f}, v={pupil_v:.3f}"],
                textposition="top right",
                name="Chief Ray Hit",
            ))

            axis_limit = max(hw, hh) * 1.15
            fig_pupil_face.update_layout(
                xaxis=dict(range=[-axis_limit, axis_limit], title="Local u (mm)", scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[-axis_limit, axis_limit], title="Local v (mm)"),
                template="plotly_white",
                height=360,
                margin=dict(l=40, r=20, t=30, b=30),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_pupil_face, use_container_width=True)

            # 2. SECONDARY STAGE: CONDENSER LENS COUPLING
            st.markdown("#### Stage 2: Common Condenser Lens Coupling")
            if current_system.final_lens_3d is not None:
                lens_rad = current_system.final_lens_3d.radius
                if cond_in_bounds:
                    cond_badge = f'<div style="background-color:#28a745; color:white; padding:4px 10px; border-radius:4px; font-weight:700; display:inline-block;">IN APERTURE: Hit offset = {cond_miss:.2f} mm (Clear Radius = {lens_rad:.1f} mm)</div>'
                else:
                    cond_badge = f'<div style="background-color:#dc3545; color:white; padding:4px 10px; border-radius:4px; font-weight:700; display:inline-block;">OUT OF APERTURE: Hit offset = {cond_miss:.2f} mm &gt; {lens_rad:.1f} mm</div>'
                st.markdown(cond_badge, unsafe_allow_html=True)
                st.caption(f"Condenser Local Coords: u = {cond_u:.2f} mm, v = {cond_v:.2f} mm")

            # 3. SLICER REFLECTION DIAGNOSTICS & LAW OF REFLECTION
            st.markdown("#### Slicer Reflection Diagnostics")
            diag_cols = st.columns(3)
            with diag_cols[0]:
                st.metric("Incidence θi", f"{diag['theta_i_deg']:.3f}°")
            with diag_cols[1]:
                st.metric("Reflection θr", f"{diag['theta_r_deg']:.3f}°")
            with diag_cols[2]:
                st.metric("Pointing Error", f"{diag['pointing_error_deg']:.3f}°")

            st.caption(
                f"Law of Reflection: θi == θr verified: **{diag['verified']}**\n\n"
                f"- Incident k_in: `[{diag['k_in'][0]:.4f}, {diag['k_in'][1]:.4f}, {diag['k_in'][2]:.4f}]`\n"
                f"- Mirror normal n: `[{diag['normal'][0]:.4f}, {diag['normal'][1]:.4f}, {diag['normal'][2]:.4f}]`\n"
                f"- Reflected k_ref: `[{diag['k_ref'][0]:.4f}, {diag['k_ref'][1]:.4f}, {diag['k_ref'][2]:.4f}]`"
            )

            # 4. RECOMMENDATION / ASSIST BOX
            st.markdown("#### Ideal Alignment Recommendation")
            st.markdown(
                f"Analytically computed angles to hit Pupil P{selected_ch_idx+1} center:\n"
                f"- **Recommended Tip X:** `{ideal_tip:.4f}°` (Δ = `{delta_tip:+.4f}°`)\n"
                f"- **Recommended Tilt Y:** `{ideal_tilt:.4f}°` (Δ = `{delta_tilt:+.4f}°`)"
            )
            
            def apply_ideal_angles(ch_i, t_x, t_y):
                st.session_state.manual_channels[ch_i]["slicer_tip_x"] = round(t_x, 5)
                st.session_state.manual_channels[ch_i]["slicer_tilt_y"] = round(t_y, 5)
                w_tip = f"num_slicer_slicer_tip_x_{ch_i}"
                w_tilt = f"num_slicer_slicer_tilt_y_{ch_i}"
                s_tip = f"slide_slicer_slicer_tip_x_{ch_i}"
                s_tilt = f"slide_slicer_slicer_tilt_y_{ch_i}"
                st.session_state[w_tip] = round(t_x, 5)
                st.session_state[w_tilt] = round(t_y, 5)
                st.session_state[s_tip] = float(np.clip(t_x, -35.0, 35.0))
                st.session_state[s_tilt] = float(np.clip(t_y, -35.0, 35.0))

            st.button(
                f"Apply Recommended Values to Slicer S{selected_ch_idx+1}",
                on_click=apply_ideal_angles,
                args=(selected_ch_idx, ideal_tip, ideal_tilt),
                disabled=lock_slicer,
                use_container_width=True,
            )

        st.divider()

        # ----------------------------------------------------
        # 3D ISOLATED CHANNEL VISUALIZATION
        # ----------------------------------------------------
        st.markdown(f"### 3D Isolated Beam Path: Channel {selected_ch_idx+1}")
        st.caption("Highlights the selected slicer and assigned pupil mirror in full brightness. All unselected channels are faded.")

        fig_iso = go.Figure()

        # Helper to draw 3D rectangle
        def draw_rect_3d(fig, center, normal, width, height, name, color, opacity=1.0, width_line=3):
            u, v, w = build_orthonormal_basis(normal)
            hw, hh = width / 2.0, height / 2.0
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)]
            x_pts = [center[0] + cu * u[0] + cv * v[0] for cu, cv in corners]
            y_pts = [center[1] + cu * u[1] + cv * v[1] for cu, cv in corners]
            z_pts = [center[2] + cu * u[2] + cv * v[2] for cu, cv in corners]
            fig.add_trace(go.Scatter3d(
                x=x_pts, y=y_pts, z=z_pts,
                mode="lines",
                line=dict(color=color, width=width_line),
                opacity=opacity,
                name=name,
            ))

        # Draw all slicer mirrors
        for s in current_system.slicer.slices:
            is_active = (s.slice_id == selected_ch_idx)
            s_col = "#00D2FF" if is_active else "rgba(100, 100, 100, 0.18)"
            s_op = 1.0 if is_active else 0.2
            s_w = 4 if is_active else 1
            draw_rect_3d(fig_iso, s.center, s.normal, s.width, s.height, f"Slice S{s.slice_id+1}", s_col, opacity=s_op, width_line=s_w)

        # Draw all pupil mirrors
        for m in current_system.pupil_relay.mirrors:
            is_active = (m.channel_id == selected_ch_idx)
            m_col = "#FF9F1C" if is_active else "rgba(100, 100, 100, 0.18)"
            m_op = 1.0 if is_active else 0.2
            m_w = 4 if is_active else 1
            draw_rect_3d(fig_iso, m.center, m.normal, m.width, m.height, f"Pupil P{m.channel_id+1}", m_col, opacity=m_op, width_line=m_w)

        # Draw Reflected Chief Ray for active channel
        p_hit = pupil_hit_pos
        c_hit = cond_hit_pos if current_system.final_lens_3d is not None else curr_pupil.center + 50.0 * k_pupil_ref
        
        # Slicer to Pupil ray
        fig_iso.add_trace(go.Scatter3d(
            x=[curr_slice.center[0], p_hit[0]],
            y=[curr_slice.center[1], p_hit[1]],
            z=[curr_slice.center[2], p_hit[2]],
            mode="lines+markers",
            line=dict(color="#FF3864", width=5),
            marker=dict(size=4),
            name="Slicer Chief Ray",
        ))

        # Pupil to Condenser ray
        fig_iso.add_trace(go.Scatter3d(
            x=[p_hit[0], c_hit[0]],
            y=[p_hit[1], c_hit[1]],
            z=[p_hit[2], c_hit[2]],
            mode="lines+markers",
            line=dict(color="#2BD9FE", width=5),
            marker=dict(size=4),
            name="Pupil Relayed Ray",
        ))

        # Hit point marker on Pupil face
        fig_iso.add_trace(go.Scatter3d(
            x=[p_hit[0]], y=[p_hit[1]], z=[p_hit[2]],
            mode="markers",
            marker=dict(size=8, color="#FFFF00", symbol="diamond"),
            name="Pupil Hit Point",
        ))

        # Target center of Pupil mirror
        fig_iso.add_trace(go.Scatter3d(
            x=[curr_pupil.center[0]], y=[curr_pupil.center[1]], z=[curr_pupil.center[2]],
            mode="markers",
            marker=dict(size=8, color="#00FF00", symbol="cross"),
            name="Pupil Center Target",
        ))

        # Condenser Lens
        if current_system.final_lens_3d is not None:
            u_l, v_l, _ = build_orthonormal_basis(current_system.final_lens_3d.normal)
            th_l = np.linspace(0, 2*np.pi, 30)
            c_l = current_system.final_lens_3d.center
            r_l = current_system.final_lens_3d.radius
            fig_iso.add_trace(go.Scatter3d(
                x=c_l[0] + r_l * (np.cos(th_l)*u_l[0] + np.sin(th_l)*v_l[0]),
                y=c_l[1] + r_l * (np.cos(th_l)*u_l[1] + np.sin(th_l)*v_l[1]),
                z=c_l[2] + r_l * (np.cos(th_l)*u_l[2] + np.sin(th_l)*v_l[2]),
                mode="lines",
                line=dict(color="#28a745", width=3),
                name="Condenser Lens",
            ))

        fig_iso.update_layout(
            template="plotly_white",
            height=480,
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=20, b=20),
        )
        st.plotly_chart(fig_iso, use_container_width=True)

        st.divider()

        # ----------------------------------------------------
        # ALL-CHANNELS SUMMARY TABLE & JSON EXPORT
        # ----------------------------------------------------
        st.markdown("### All-Channels Bench Alignment Summary")
        summary_rows = []
        full_export_data = {}

        for ch_idx, s in enumerate(current_system.slicer.slices):
            pm = current_system.pupil_relay.mirrors[ch_idx] if (current_system.pupil_relay and ch_idx < len(current_system.pupil_relay.mirrors)) else None
            if pm is None:
                continue
            
            k_s_ref = s.get_reflected_chief_ray(k_in_nom)
            u_p, v_p, miss_p, in_b_p, hit_p = pm.get_local_hit(s.center, k_s_ref)
            d_p = s.get_reflection_diagnostics(k_in=k_in_nom, target_pos=pm.center)
            
            # Condenser hit
            k_p_ref = reflect_vector(k_s_ref, pm.normal)
            miss_cond = 0.0
            if current_system.final_lens_3d is not None:
                _, _, miss_cond, in_b_c, _ = current_system.final_lens_3d.get_local_hit(hit_p, k_p_ref)

            status_str = "Aligned" if miss_p < fine_th else ("Moderate" if miss_p < coarse_th else "Misaligned")

            summary_rows.append({
                "Channel": f"Channel {ch_idx+1}",
                "Slicer": f"S{ch_idx+1}",
                "Tip X (deg)": f"{s.tip_x_deg:.4f}",
                "Tilt Y (deg)": f"{s.tilt_y_deg:.4f}",
                "Pupil Hit (u, v) mm": f"({u_p:.3f}, {v_p:.3f})",
                "Pupil Miss (mm)": f"{miss_p:.4f}",
                "In Aperture": "Yes" if in_b_p else "No",
                "Pointing Error (deg)": f"{d_p['pointing_error_deg']:.3f}",
                "Condenser Miss (mm)": f"{miss_cond:.2f}",
                "Status": status_str,
            })

            full_export_data[f"channel_{ch_idx+1}"] = {
                "slicer": {
                    "slice_id": ch_idx,
                    "position": [float(s.center_x), float(s.center_y), float(s.z)],
                    "tip_x_deg": float(s.tip_x_deg),
                    "tilt_y_deg": float(s.tilt_y_deg),
                    "rot_z_deg": float(s.rot_z_deg),
                },
                "pupil": {
                    "channel_id": ch_idx,
                    "position": [float(pm.center[0]), float(pm.center[1]), float(pm.center[2])],
                    "tip_x_deg": float(pm.tip_x_deg),
                    "tilt_y_deg": float(pm.tilt_y_deg),
                    "rot_z_deg": float(pm.rot_z_deg),
                    "focal_length": pm.focal_length,
                },
                "metrics": {
                    "pupil_miss_mm": float(miss_p),
                    "in_pupil_aperture": bool(in_b_p),
                    "slicer_pointing_error_deg": float(d_p["pointing_error_deg"]),
                    "condenser_miss_mm": float(miss_cond),
                    "status": status_str,
                }
            }

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.download_button(
            label="Download Bench Alignment Configuration (JSON)",
            data=json.dumps(full_export_data, indent=2),
            file_name="slicer_pupil_bench_alignment.json",
            mime="application/json",
        )


# --------------------------------------------------
# TAB 5: PUPIL PLANE & AUTO-ALIGNMENT TOOLS
# --------------------------------------------------
with tab_pupil:
    st.subheader("Pupil Mirrors: 3D Positions & Alignment Tools")
    st.markdown(
        "Each slicer channel reflects in a unique 3D direction toward its dedicated pupil mirror. "
        "Use the tools below to automatically place and aim all pupil mirrors analytically."
    )

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("#### Auto-Placement along Slicer Beams")
        st.caption("Computes Pi = Si + L * k_ref_i for each channel:")

        p_dist_input = st.number_input("Pupil Distance L along beam (mm)", value=float(st.session_state.pupil_dist), step=5.0)
        if st.button("Auto Place Pupil Mirrors", use_container_width=True):
            if current_system.slicer is not None and current_system.pupil_relay is not None:
                current_system.pupil_relay.auto_place_mirrors(current_system.slicer, distance=p_dist_input)
                st.session_state.pupil_dist = p_dist_input
                st.success("Pupil mirrors placed exactly along reflected chief rays!")

    with p_col2:
        st.markdown("#### Aim All Pupil Mirrors at Final Lens")
        st.caption("Analytically calculates bisector normals directing each channel to the final lens:")

        target_fl_z = st.number_input("Target Lens z (mm)", value=float(st.session_state.final_lens_z), step=5.0)
        if st.button("Aim All Pupil Mirrors at Final Relay", use_container_width=True):
            if current_system.slicer is not None and current_system.pupil_relay is not None:
                current_system.pupil_relay.aim_all_at_target(
                    current_system.slicer,
                    target_point=np.array([0.0, 0.0, target_fl_z]),
                )
                st.success("All pupil mirrors aimed analytically at common final lens center!")

    st.divider()
    st.markdown("#### Pupil Mirror Geometry Table")
    if current_system.pupil_relay is not None and current_system.pupil_relay.mirrors:
        pup_data = []
        for m in current_system.pupil_relay.mirrors:
            pup_data.append({
                "Channel": f"P{m.channel_id + 1}",
                "Center X (mm)": f"{m.center[0]:.2f}",
                "Center Y (mm)": f"{m.center[1]:.2f}",
                "Center Z (mm)": f"{m.center[2]:.2f}",
                "Normal Vector": f"[{m.normal[0]:.3f}, {m.normal[1]:.3f}, {m.normal[2]:.3f}]",
                "Width x Height": f"{m.width:.1f} x {m.height:.1f} mm",
                "Powered Mirror": f"f = {m.focal_length:.1f} mm" if m.is_powered else "Flat",
            })
        st.dataframe(pd.DataFrame(pup_data), use_container_width=True)


# --------------------------------------------------
# TAB 5: FIBER COUPLING & SPATIAL/ANGULAR ACCEPTANCE
# --------------------------------------------------
with tab_fiber:
    st.subheader("Multimode Fiber Coupling Performance")
    st.markdown(
        "Ray acceptance is evaluated relative to the **3D fiber axis**:\n"
        "1. **Spatial Condition:** Distance from ray hit to fiber center $\\le 0.5\\text{ mm}$ (1.0 mm core)\n"
        "2. **Angular Condition:** Incidence angle $\\sin(\\theta) \\le \\text{NA} = 0.22$ ($\\theta \\le 12.71^\\circ$ in air)"
    )

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("#### Fiber Face Spot Diagram")
        fig_ff = go.Figure()

        theta_c = np.linspace(0, 2 * np.pi, 200)
        fig_ff.add_trace(
            go.Scatter(
                x=0.5 * np.cos(theta_c),
                y=0.5 * np.sin(theta_c),
                mode="lines",
                line=dict(color="black", width=2.5, dash="dash"),
                name="1.0 mm Core Boundary (r=0.5mm)",
            )
        )

        class_defs = [
            (RayStatus.ACCEPTED_BY_FIBER, "Accepted (Spatial & NA OK)", "#00CC96", 5),
            (RayStatus.REJECTED_BY_NA, "Rejected: NA Only (r OK, θ > NA)", "#FFA15A", 4),
            (RayStatus.REJECTED_BY_POSITION, "Rejected: Spatial Only (r > 0.5mm, θ OK)", "#636EFA", 4),
            (RayStatus.REJECTED_BY_BOTH, "Rejected: Both Failed", "#EF553B", 4),
        ]

        for s_code, s_label, s_color, s_size in class_defs:
            c_mask = coupling_res.classifications == s_code
            if np.any(c_mask):
                fig_ff.add_trace(
                    go.Scatter(
                        x=coupling_res.hit_x[c_mask],
                        y=coupling_res.hit_y[c_mask],
                        mode="markers",
                        marker=dict(color=s_color, size=s_size, opacity=0.7),
                        name=f"{s_label} ({np.sum(c_mask)})",
                    )
                )

        fig_ff.update_layout(
            title="Fiber Face Spot Distribution",
            xaxis_title="Local u (mm)",
            yaxis_title="Local v (mm)",
            template="plotly_white",
            height=440,
            xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.2, 1.2]),
            yaxis=dict(range=[-1.2, 1.2]),
        )
        st.plotly_chart(fig_ff, use_container_width=True)

    with f_col2:
        st.markdown("#### Angular Acceptance Phase Space (θ vs r)")
        fig_as = go.Figure()
        fig_as.add_vrect(x0=0.0, x1=0.5, fillcolor="green", opacity=0.08, line_width=0)
        fig_as.add_hrect(y0=0.0, y1=12.71, fillcolor="green", opacity=0.08, line_width=0)
        fig_as.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="r = 0.5 mm")
        fig_as.add_hline(y=12.71, line_dash="dash", line_color="red", annotation_text="NA = 0.22 (12.71°)")

        for s_code, s_label, s_color, s_size in class_defs:
            c_mask = coupling_res.classifications == s_code
            if np.any(c_mask):
                fig_as.add_trace(
                    go.Scatter(
                        x=coupling_res.r_coords[c_mask],
                        y=coupling_res.theta_angles_deg[c_mask],
                        mode="markers",
                        marker=dict(color=s_color, size=s_size, opacity=0.7),
                        name=s_label,
                    )
                )

        fig_as.update_layout(
            title="Phase Space Acceptance Plot (Angle vs Radius)",
            xaxis_title="Radial Position r (mm)",
            yaxis_title="Ray Angle θ to Fiber Axis (deg)",
            template="plotly_white",
            height=440,
        )
        st.plotly_chart(fig_as, use_container_width=True)

    st.divider()
    l_c1, l_c2 = st.columns([1, 1])
    with l_c1:
        st.markdown("#### Optical Loss Budget")
        st.dataframe(
            pd.DataFrame(metrics.loss_budget_table).style.format({
                "z (mm)": "{:.1f}",
                "Surviving Power": "{:.3f}",
                "Power Lost": "{:.3f}",
                "Stage Transmission": "{:.1%}",
                "Cumulative Transmission": "{:.1%}",
            }),
            use_container_width=True,
        )

    with l_c2:
        st.markdown("#### Power Coupled per Field Channel")
        if metrics.per_channel_stats:
            ch_rows = [
                {
                    "Channel": f"Channel S{ch_id + 1}",
                    "Intercepted Power": st_c["intercepted_power"],
                    "Accepted Power": st_c["accepted_power"],
                }
                for ch_id, st_c in metrics.per_channel_stats.items()
            ]
            fig_b = px.bar(
                pd.DataFrame(ch_rows),
                x="Channel",
                y=["Intercepted Power", "Accepted Power"],
                barmode="group",
                title="Channel Power Distribution",
                template="plotly_white",
            )
            st.plotly_chart(fig_b, use_container_width=True)


# --------------------------------------------------
# TAB 6: COMPARE ARCHITECTURES
# --------------------------------------------------
with tab_compare:
    st.subheader("Comparative Benchmark: Lens-Only vs Slicer Architectures")
    st.markdown(
        "Traces the identical source ray bundle across all candidate architectures to compare "
        "power coupling into the 1.0 mm multimode fiber."
    )

    if st.button("Run Comparative Benchmark", use_container_width=True):
        sys_old = create_old_lens_system()
        sys_cond = create_slicer_condenser_system(active_slices=4)
        sys_b2 = create_branched_slicer_system(
            n_channels=2,
            pupil_layout_side=st.session_state.pupil_layout_side,
            pupil_focal_length=75.0,
        )
        sys_b4 = create_branched_slicer_system(
            n_channels=4,
            pupil_layout_side=st.session_state.pupil_layout_side,
            pupil_focal_length=75.0,
        )

        _, res_old, m_old = sys_old.trace(source_bundle)
        _, res_cond, m_cond = sys_cond.trace(source_bundle)
        _, res_b2, m_b2 = sys_b2.trace(source_bundle)
        _, res_b4, m_b4 = sys_b4.trace(source_bundle)

        df_bench = pd.DataFrame([
            {
                "Architecture": "A) Old Lens-Only System",
                "Power at Fiber Plane": f"{res_old.power_reaching_fiber_plane * 100.0:.1f}%",
                "Spatial Acceptance": f"{m_old.fraction_inside_core * 100.0:.1f}%",
                "NA Acceptance": f"{m_old.fraction_inside_na * 100.0:.1f}%",
                "Coupling Efficiency": f"{res_old.geometric_coupling_efficiency * 100.0:.2f}%",
                "Spot RMS (mm)": f"{res_old.spot_rms_radius:.3f}",
            },
            {
                "Architecture": "B) Slicer + Common Condenser",
                "Power at Fiber Plane": f"{res_cond.power_reaching_fiber_plane * 100.0:.1f}%",
                "Spatial Acceptance": f"{m_cond.fraction_inside_core * 100.0:.1f}%",
                "NA Acceptance": f"{m_cond.fraction_inside_na * 100.0:.1f}%",
                "Coupling Efficiency": f"{res_cond.geometric_coupling_efficiency * 100.0:.2f}%",
                "Spot RMS (mm)": f"{res_cond.spot_rms_radius:.3f}",
            },
            {
                "Architecture": "C) Asymmetric One-Sided Slicer (2-Channel Test)",
                "Power at Fiber Plane": f"{res_b2.power_reaching_fiber_plane * 100.0:.1f}%",
                "Spatial Acceptance": f"{m_b2.fraction_inside_core * 100.0:.1f}%",
                "NA Acceptance": f"{m_b2.fraction_inside_na * 100.0:.1f}%",
                "Coupling Efficiency": f"{res_b2.geometric_coupling_efficiency * 100.0:.2f}%",
                "Spot RMS (mm)": f"{res_b2.spot_rms_radius:.3f}",
            },
            {
                "Architecture": "D) Asymmetric One-Sided Slicer (4-Channel Full)",
                "Power at Fiber Plane": f"{res_b4.power_reaching_fiber_plane * 100.0:.1f}%",
                "Spatial Acceptance": f"{m_b4.fraction_inside_core * 100.0:.1f}%",
                "NA Acceptance": f"{m_b4.fraction_inside_na * 100.0:.1f}%",
                "Coupling Efficiency": f"{res_b4.geometric_coupling_efficiency * 100.0:.2f}%",
                "Spot RMS (mm)": f"{res_b4.spot_rms_radius:.3f}",
            },
        ])
        st.dataframe(df_bench, use_container_width=True)


# --------------------------------------------------
# TAB 7: OPTIMIZATION & ALIGNMENT
# --------------------------------------------------
with tab_opt:
    st.subheader("Auto-Alignment & Numerical Optimization")
    st.markdown("Optimize mirror tilts and fiber focus to maximize optical weight coupled into the fiber.")

    st.markdown("#### One-Sided Relay Alignment Routines")
    r_c1, r_c2 = st.columns(2)
    with r_c1:
        if st.button("Aim All Slicers at Assigned Pupil Mirrors", use_container_width=True):
            if current_system.slicer is not None and current_system.pupil_relay is not None:
                pupil_pts = [m.center for m in current_system.pupil_relay.mirrors]
                current_system.slicer.aim_all_at_pupils(pupil_pts)
                st.success("All slicers actively aimed at assigned pupil mirrors!")
    with r_c2:
        if st.button("Redirect All Pupil Channels to Common Relay Axis", use_container_width=True):
            if current_system.slicer is not None and current_system.pupil_relay is not None:
                mode = "parallel" if "Parallel" in st.session_state.pupil_aim_mode else "target_center"
                c_lens = current_system.final_lens_3d.center if current_system.final_lens_3d is not None else np.array([0, -20, 330])
                a_rel = getattr(current_system, "relay_axis_direction", None)
                current_system.pupil_relay.redirect_all_to_relay_axis(
                    current_system.slicer,
                    condenser_center=c_lens,
                    relay_axis_direction=a_rel,
                    mode=mode,
                )
                st.success(f"All pupil mirrors aligned ({mode} mode) along the new common relay axis!")

    st.divider()
    o_c1, o_c2 = st.columns(2)
    with o_c1:
        st.markdown("#### Aim Individual Slicer Slice")
        slice_aim_id = st.selectbox("Select Slice ID", [0, 1, 2, 3], index=0)
        target_pt = st.number_input("Target z for slice aim (mm)", value=float(current_system.final_lens_3d.center[2] if current_system.final_lens_3d is not None else 330.0), step=5.0)

        if st.button("Aim Slice", use_container_width=True):
            if current_system.slicer is not None:
                tip, tilt = aim_slicer_mirror(current_system.slicer, slice_aim_id, (0.0, 0.0, target_pt))
                st.success(f"Slice {slice_aim_id} aimed! Tip={tip:.2f}°, Tilt={tilt:.2f}°")

    with o_c2:
        st.markdown("#### Optimize Fiber Position (Focus)")
        if st.button("Optimize Fiber Position (Focus)", use_container_width=True):
            with st.spinner("Optimizing fiber position..."):
                opt_res = optimize_fiber_position(current_system, source_bundle, maxiter=30)
                st.success(
                    f"Optimal Fiber Position: x={opt_res['optimal_x']:.2f}, y={opt_res['optimal_y']:.2f}, "
                    f"z={opt_res['optimal_z']:.1f} mm. Coupling: **{opt_res['optimized_efficiency'] * 100.0:.2f}%**"
                )

    st.divider()
    st.markdown("#### Automated Architecture Design Optimizer")
    st.write("Perform global differential evolution search across discrete N slicer channels to find the mathematically optimal architecture.")
    if st.button("Open Architecture Design Optimizer", key="btn_open_arch_opt_tab", type="primary", use_container_width=True):
        st.session_state.app_mode = "Architecture Design Optimizer"
        st.rerun()

    st.markdown("#### Re-Optimize Current Manual Geometry")
    st.write("Perform local simplex polishing starting from the exact slicer and pupil mirror poses currently configured on the manual bench.")
    if st.button("Re-Optimize Current Manual Geometry", key="btn_reopt_tab_opt", use_container_width=True):
        with st.spinner("Polishing current bench geometry..."):
            ok, old_e, new_e = reoptimize_manual_geometry(current_system, st.session_state.manual_channels)
            if ok:
                st.success(f"Bench refinement complete! Coupling efficiency: {old_e*100.0:.2f}% -> {new_e*100.0:.2f}%.")
                st.rerun()
            else:
                st.error("Failed to polish current geometry.")


# --------------------------------------------------
# TAB 8: PARAMETER SWEEPS
# --------------------------------------------------
with tab_sweeps:
    st.subheader("Parametric Sensitivity Sweeps")
    sweep_sel = st.selectbox(
        "Select Parameter to Sweep",
        ["Pupil Distance L (mm)", "Fiber Defocus z (mm)", "Fiber NA", "Fiber Core Diameter (mm)"],
    )

    if sweep_sel == "Pupil Distance L (mm)":
        dist_vals = np.linspace(40.0, 100.0, 13)
        if st.button("Run Sweep vs Pupil Distance L", use_container_width=True):
            def set_L(sys: OpticalSystem, val: float):
                if sys.slicer is not None and sys.pupil_relay is not None:
                    sys.pupil_relay.auto_place_mirrors(sys.slicer, distance=val)
                    sys.pupil_relay.aim_all_at_target(sys.slicer, np.array([0.0, 0.0, 310.0]))
            sw = run_1d_sweep(current_system, source_bundle, "Pupil Distance L (mm)", dist_vals, set_L)
            fig_sw = px.line(
                x=sw["values"],
                y=sw["efficiency"] * 100.0,
                labels={"x": "Pupil Distance L (mm)", "y": "Coupling Efficiency (%)"},
                title="Coupling Efficiency vs Pupil Distance L",
                template="plotly_white",
            )
            st.plotly_chart(fig_sw, use_container_width=True)

    elif sweep_sel == "Fiber Defocus z (mm)":
        fz_vals = np.linspace(350.0, 420.0, 15)
        if st.button("Run Sweep vs Fiber z", use_container_width=True):
            def set_fz(sys: OpticalSystem, val: float):
                sys.fiber.position[2] = val
            sw = run_1d_sweep(current_system, source_bundle, "Fiber z (mm)", fz_vals, set_fz)
            fig_sw = px.line(
                x=sw["values"],
                y=sw["efficiency"] * 100.0,
                labels={"x": "Fiber z (mm)", "y": "Coupling Efficiency (%)"},
                title="Coupling Efficiency vs Fiber Defocus",
                template="plotly_white",
            )
            st.plotly_chart(fig_sw, use_container_width=True)

    elif sweep_sel == "Fiber NA":
        na_vals = np.linspace(0.12, 0.35, 12)
        if st.button("Run Sweep vs Fiber NA", use_container_width=True):
            def set_na(sys: OpticalSystem, val: float):
                sys.fiber.na = val
            sw = run_1d_sweep(current_system, source_bundle, "Fiber NA", na_vals, set_na)
            fig_sw = px.line(
                x=sw["values"],
                y=sw["efficiency"] * 100.0,
                labels={"x": "Fiber NA", "y": "Coupling Efficiency (%)"},
                title="Coupling Efficiency vs Fiber NA",
                template="plotly_white",
            )
            st.plotly_chart(fig_sw, use_container_width=True)

    else:
        d_vals = np.linspace(0.4, 2.0, 13)
        if st.button("Run Sweep vs Fiber Core Diameter", use_container_width=True):
            def set_cd(sys: OpticalSystem, val: float):
                sys.fiber.core_diameter = val
                sys.fiber.core_radius = val / 2.0
            sw = run_1d_sweep(current_system, source_bundle, "Core Diameter (mm)", d_vals, set_cd)
            fig_sw = px.line(
                x=sw["values"],
                y=sw["efficiency"] * 100.0,
                labels={"x": "Core Diameter (mm)", "y": "Coupling Efficiency (%)"},
                title="Coupling Efficiency vs Fiber Core Diameter",
                template="plotly_white",
            )
            st.plotly_chart(fig_sw, use_container_width=True)


# --------------------------------------------------
# TAB 9: VALIDATION & ÉTENDUE
# --------------------------------------------------
with tab_val:
    st.subheader("Physics Verification, Étendue Check & Test Runner")
    
    st.markdown("#### 7-Case Physical Validation Suite")
    st.caption("Verifies spatial collapse (2 mm displacement), angular collapse (> 12.7°), ideal oversized aperture lossless transmission, zero-gap scaling, etendue conservation, and 10-restart multi-start stability.")

    if st.button("Execute 7-Case Physical Validation Suite", type="primary", use_container_width=True):
        with st.spinner("Executing 7 physical validation cases..."):
            v_report = run_all_validations()
            st.session_state.validation_report = v_report

    if "validation_report" in st.session_state and st.session_state.validation_report is not None:
        rep: ValidationSuiteReport = st.session_state.validation_report
        if rep.all_passed:
            st.success("ALL 7 PHYSICAL VALIDATION CASES PASSED RIGOROUSLY!")
        else:
            st.error("ONE OR MORE PHYSICAL VALIDATION TESTS FAILED:")
        st.markdown(rep.summary_markdown)

    st.divider()
    val_c1, val_c2 = st.columns(2)

    with val_c1:
        st.markdown("#### Étendue Conservation & Concentration Check")
        g_src, g_fib, eta_max, is_allowed = compute_etendue(
            pupil_diameter=st.session_state.get("aperture_diameter", 12.0),
            solar_angular_radius_deg=0.266,
            fiber_core_diameter=st.session_state.get("fiber_core_diameter", 1.0),
            fiber_na=st.session_state.get("fiber_na", 0.22),
        )
        st.metric(label="Fiber Étendue (mm²·sr)", value=f"{g_fib:.5f}")
        st.metric(label="Source Étendue (mm²·sr)", value=f"{g_src:.5f}")
        st.metric(label="Thermodynamic Upper Bound (eta_max)", value=f"{eta_max * 100.0:.2f}%")
        if is_allowed:
            st.info("Passive concentration physically permitted: G_source <= G_fiber. Maximum possible transmission is 100%.")
        else:
            st.warning(f"ATTENTION: G_source > G_fiber. By the second law of thermodynamics, maximum passive transmission is bounded at {eta_max*100.0:.2f}%.")

    with val_c2:
        st.markdown("#### Automated Pytest Test Runner")
        st.caption("Executes all 38 automated pytest unit tests (branched geometry, energy conservation, manual alignment, optimizer, and validation suite):")
        if st.button("Run Automated Test Suite (38 Tests)", use_container_width=True):
            with st.spinner("Running 38 unit tests..."):
                proc = subprocess.run(["./.venv/bin/pytest", "-v", "tests/"], capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success("ALL 38 AUTOMATED UNIT TESTS PASSED!")
                else:
                    st.error("Test failures detected:")
                st.code(proc.stdout, language="bash")

    st.divider()
    st.markdown("#### Data Export")
    ex_c1, ex_c2 = st.columns(2)
    with ex_c1:
        ray_df = pd.DataFrame({
            "u_mm": coupling_res.hit_x[:1000],
            "v_mm": coupling_res.hit_y[:1000],
            "r_mm": coupling_res.r_coords[:1000],
            "theta_deg": coupling_res.theta_angles_deg[:1000],
            "power": coupling_res.powers[:1000],
            "status": coupling_res.classifications[:1000],
        })
        st.download_button(
            "Download Fiber Ray Hits CSV",
            data=ray_df.to_csv(index=False),
            file_name="fiber_ray_hits.csv",
            mime="text/csv",
        )
    with ex_c2:
        st.download_button(
            "Download Loss Budget CSV",
            data=pd.DataFrame(metrics.loss_budget_table).to_csv(index=False),
            file_name="loss_budget.csv",
            mime="text/csv",
        )
