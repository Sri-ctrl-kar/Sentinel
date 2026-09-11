"""Reasoning layer: from recorded events to assessed risk.

This is where Sentinel stops describing and starts reasoning. It consumes the
M0.2 temporal event memory and produces structured, explainable assessments of
developing situations.

Layering rule (enforced by ``tests/test_memory_layering.py``)
------------------------------------------------------------
Nothing in this package may import :mod:`app.perception`, torch, Ultralytics,
OpenCV or numpy. The engine reasons about `Event` objects; it neither knows nor
cares what produced them.
"""

from .config import RiskConfig
from .explainer import Explainer, format_incident
from .kinematics import ImageMotion, MotionEstimator, closest_approach
from .models.risk import (
    FactorScore,
    RiskAssessment,
    RiskReport,
    severity_for,
)
from .risk_engine import RiskEngine

__all__ = [
    "RiskConfig",
    "RiskEngine",
    "Explainer",
    "format_incident",
    "ImageMotion",
    "MotionEstimator",
    "closest_approach",
    "FactorScore",
    "RiskAssessment",
    "RiskReport",
    "severity_for",
]
