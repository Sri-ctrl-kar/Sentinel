"""Independently testable risk factors.

Each factor answers one narrow question about a candidate situation and
returns a :class:`~app.reasoning.models.risk.FactorScore` in ``[0, 1]`` with
the measurements it used. The engine does the weighting and aggregation; a
factor never knows the final score, which is what makes each one testable on
its own.
"""

from .base import RiskCandidate, RiskContext, RiskFactor, MetaRiskFactor
from .persistence import EscalationFactor, PersistenceFactor
from .proximity import ClosingSpeedFactor, ProximityFactor
from .trajectory import TrajectoryFactor
from .zone import ZoneFactor

#: The four base factors, in report order. Their weights sum to 100.
BASE_FACTORS = (ProximityFactor, ClosingSpeedFactor, TrajectoryFactor, ZoneFactor)

__all__ = [
    "RiskCandidate",
    "RiskContext",
    "RiskFactor",
    "MetaRiskFactor",
    "ProximityFactor",
    "ClosingSpeedFactor",
    "TrajectoryFactor",
    "ZoneFactor",
    "PersistenceFactor",
    "EscalationFactor",
    "BASE_FACTORS",
]
