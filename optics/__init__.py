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
from .system import OpticalSystem, OpticalStage
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
    "PowerAccounting",
    "compute_etendue",
    "run_all_validations",
    "ValidationSuiteReport",
    "ValidationCaseResult",
]
