"""In-memory analysis registry (M0.9).

Single process, single machine, no database — that is the whole design. An
analysis is a video someone uploaded plus whatever the pipeline has produced
from it so far, held in a dict behind a lock. Nothing here persists across a
restart, and nothing here needs to for a demo.

The store owns two things the rest of the API should not have to think about:
the temporary directory uploaded videos live in, and the state machine
``queued -> running -> complete | failed``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import PipelineConfig

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"

STATUSES = (STATUS_QUEUED, STATUS_RUNNING, STATUS_COMPLETE, STATUS_FAILED)

#: Terminal states: an analysis in one of these will never change again.
TERMINAL_STATUSES = (STATUS_COMPLETE, STATUS_FAILED)


@dataclass
class Progress:
    """How far through the clip the runner has got."""

    frames_processed: int = 0
    total_frames: Optional[int] = None

    @property
    def percent(self) -> Optional[float]:
        if not self.total_frames:
            return None
        return round(100.0 * min(self.frames_processed / self.total_frames, 1.0), 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "total_frames": self.total_frames,
            "percent": self.percent,
        }


@dataclass
class AnalysisRecord:
    """One uploaded video and everything derived from it.

    Deliberately a plain container: the runner fills it in, the routes read it,
    and the validated core objects (``PipelineResult``, ``TemporalEventMemory``,
    ``RiskReport``) are stored as themselves rather than pre-serialised. The
    route layer decides what JSON to make of them.
    """

    analysis_id: str
    video_path: str
    original_filename: str
    config: PipelineConfig
    status: str = STATUS_QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: Progress = field(default_factory=Progress)

    #: Options that shape reasoning rather than perception, kept separate so a
    #: risk re-run can use different ones without re-running inference.
    risk_step: float = 0.5
    operating_zones: List[str] = field(default_factory=list)

    # --- produced by the runner -----------------------------------------
    result: Any = None            # app.pipeline.PipelineResult
    frames: List[Any] = field(default_factory=list)   # runner.FrameRecord
    video_metadata: Optional[Dict[str, Any]] = None
    device: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    # --- computed on demand, cached -------------------------------------
    risk_reports: Optional[List[Any]] = None          # List[RiskReport]

    @property
    def is_complete(self) -> bool:
        return self.status == STATUS_COMPLETE

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 4)

    @property
    def temporal(self) -> Any:
        return getattr(self.result, "temporal", None) if self.result else None

    def mark_running(self) -> None:
        self.status = STATUS_RUNNING
        self.started_at = time.time()

    def mark_complete(self) -> None:
        self.status = STATUS_COMPLETE
        self.finished_at = time.time()

    def mark_failed(self, exc: BaseException) -> None:
        """Record a failure as data, not as a lost traceback.

        The message is the exception's own text; no internal path or stack is
        exposed, because a demo API's error field is read by a browser.
        """
        self.status = STATUS_FAILED
        self.finished_at = time.time()
        self.error = {"type": type(exc).__name__, "message": str(exc)}


class AnalysisStore:
    """Thread-safe registry of analyses, plus the upload directory.

    One lock around a dict is the right amount of machinery here: the work
    itself happens outside the lock (the runner mutates its own record), and
    the only shared mutation is adding and removing entries.
    """

    def __init__(self, upload_dir: Optional[str] = None) -> None:
        self._records: Dict[str, AnalysisRecord] = {}
        self._lock = threading.Lock()
        self._owns_dir = upload_dir is None
        self.upload_dir = upload_dir or tempfile.mkdtemp(prefix="sentinel-api-")
        os.makedirs(self.upload_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def new_id(self) -> str:
        return uuid.uuid4().hex

    def video_path_for(self, analysis_id: str, filename: str) -> str:
        """Where an upload is stored. The extension is kept; the name is not.

        Using the analysis id as the filename means a caller cannot influence
        the path at all — no traversal, no collision, no surprise.
        """
        suffix = os.path.splitext(filename or "")[1].lower()
        return os.path.join(self.upload_dir, f"{analysis_id}{suffix}")

    def add(self, record: AnalysisRecord) -> AnalysisRecord:
        with self._lock:
            self._records[record.analysis_id] = record
        return record

    def get(self, analysis_id: str) -> Optional[AnalysisRecord]:
        with self._lock:
            return self._records.get(analysis_id)

    def list(self) -> List[AnalysisRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.created_at)

    def delete(self, analysis_id: str) -> bool:
        """Forget an analysis and remove its uploaded video."""
        with self._lock:
            record = self._records.pop(analysis_id, None)
        if record is None:
            return False
        try:
            if os.path.isfile(record.video_path):
                os.remove(record.video_path)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        return True

    def clear(self) -> None:
        for record in self.list():
            self.delete(record.analysis_id)

    def shutdown(self) -> None:
        """Drop everything, including the temporary directory we created."""
        self.clear()
        if self._owns_dir and os.path.isdir(self.upload_dir):
            shutil.rmtree(self.upload_dir, ignore_errors=True)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
