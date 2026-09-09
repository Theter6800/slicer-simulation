"""
Solar and LED to Optical Fiber Geometric Optics Simulator.
Inspired by Ellen Lee's image slicer IFU principles.
"""

from .ray import RayBundle, RayStatus
from .sources import SourceMode, generate_led_source, generate_sun_source
from .elements import CircularAperture, ThinLens, PlaneMirror, ThinLens3D, Surface3D
from .fiber import Fiber, FiberCouplingResult
from .slicer import SliceMirror, SlicerArray
from .pupil import PupilAnalysis, PupilMirror, PupilRelaySystem, generate_one_sided_pupil_positions
from .system import OpticalSystem, OpticalStage, OpticalGeometry
from .metrics import SystemMetrics, compute_spot_metrics, compute_etendue_check
from .presets import (
    create_old_lens_system,
    create_slicer_condenser_system,
    create_branched_slicer_system,
    create_paper_ifu_system,
    create_first_default_experiment,
)
from .power_accounting import PowerAccounting, compute_etendue
from .validation import run_all_validations, ValidationSuiteReport, ValidationCaseResult
from .design_optimizer import (
    SlicerPupilOptimizer,
    OptimizationConfig,
    SingleNOptimizationResult,
    MultiNStudyResult,
    OptimizationVariablesSummary,
    HierarchicalSearchResult,
    get_optimization_variables_summary,
    run_hierarchical_architecture_search,
    export_architecture_design_to_dict,
    export_architecture_design_to_json,
    export_architecture_design_to_csv,
    compute_detailed_loss_breakdown,
    run_magnification_slice_sweep,
)
from .fore_optics import (
    ForeOpticsMode,
    ForeOpticsConfig,
    ForeOpticsSystem,
    build_fore_optics_system,
    AVAILABLE_HARDWARE_FOCAL_LENGTHS,
    compute_theoretical_focal_length_for_d90,
    optimize_hardware_fore_optics,
)
from .sensitivity import (
    ToleranceSensitivityEngine,
    SensitivityCurve,
    SensitivityReport,
)

__all__ = [
    "RayBundle",
    "RayStatus",
    "SourceMode",
    "generate_led_source",
    "generate_sun_source",
    "CircularAperture",
    "ThinLens",
    "PlaneMirror",
    "ThinLens3D",
    "Surface3D",
    "Fiber",
    "FiberCouplingResult",
    "SliceMirror",
    "SlicerArray",
    "PupilAnalysis",
    "PupilMirror",
    "PupilRelaySystem",
    "generate_one_sided_pupil_positions",
    "OpticalSystem",
    "OpticalStage",
    "SystemMetrics",
    "compute_spot_metrics",
    "compute_etendue_check",
    "create_old_lens_system",
    "create_slicer_condenser_system",
    "create_branched_slicer_system",
    "create_paper_ifu_system",
    "create_first_default_experiment",
    "SlicerPupilOptimizer",
    "OptimizationConfig",
    "SingleNOptimizationResult",
    "MultiNStudyResult",
    "OptimizationVariablesSummary",
    "HierarchicalSearchResult",
    "get_optimization_variables_summary",
    "run_hierarchical_architecture_search",
    "export_architecture_design_to_dict",
    "export_architecture_design_to_json",
    "export_architecture_design_to_csv",
    "compute_detailed_loss_breakdown",
    "run_magnification_slice_sweep",
    "PowerAccounting",
    "SlicePowerShare",
    "compute_etendue",
    "AngularMetrics",
    "compute_angular_metrics",
    "classify_fiber_phase_space",
    "run_all_validations",
    "ValidationSuiteReport",
    "ValidationCaseResult",
]
