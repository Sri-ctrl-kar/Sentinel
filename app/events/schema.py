from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Event:
    timestamp: float
    entity_id: str
    action: str
    attributes: Dict[str, Any]
    position: Optional[List[float]] = None
