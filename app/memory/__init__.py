"""Temporal event memory.

Sentinel's perception layer is stateless in the long run: it knows what is on
screen *now*. This package holds what happened *over time* — a persistent,
chronological, queryable record of every entity's history.

Layering rule (enforced by :mod:`tests.test_memory_layering`)
------------------------------------------------------------
This package imports :mod:`app.events.schema` and :mod:`app.spatial` and
nothing else from the project. It must never import :mod:`app.perception`, a
model library, or a video library. A temporal memory that knows what a YOLO
tensor looks like cannot be reused when the detector changes — and the
reasoning layer that lands in a later milestone will sit on top of this
package, not on top of perception.
"""

from .query import EventFilter
from .temporal import EntityState, EntityTimeline, TemporalEventMemory

__all__ = [
    "EventFilter",
    "EntityState",
    "EntityTimeline",
    "TemporalEventMemory",
]
