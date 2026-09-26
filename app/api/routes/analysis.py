"""Analysis routes: upload, status, events, risk, frame timeline, video, device.

Every route is an adapter. It validates input, asks a validated core object for
an answer, and shapes that answer into a Pydantic model. No route computes
anything about perception, tracking, events or risk.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse

from ...accel import (
    DeviceUnavailable,
    available_devices,
    probe_environment,
    resolve_device,
)
from ...config import PipelineConfig
from ...reasoning import RiskConfig, RiskEngine
from ...spatial import ZoneSet
from .. import runner
from ..schemas import (
    AnalysisCreated,
    AnalysisListResponse,
    AnalysisStatusResponse,
    DeviceInfo,
    DeviceResponse,
    EventsResponse,
    RiskResponse,
    TimelineResponse,
)
from ..store import AnalysisRecord, AnalysisStore

router = APIRouter(prefix="/api", tags=["analysis"])

#: Containers OpenCV can actually open. Checked before anything is written to
#: disk, so an unsupported upload costs nothing.
ALLOWED_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

#: Upload ceiling. A demo API should refuse a 2 GB file politely rather than
#: fill a temporary directory with it.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def get_store(request: Request) -> AnalysisStore:
    return request.app.state.store


def require_record(request: Request, analysis_id: str) -> AnalysisRecord:
    record = get_store(request).get(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown analysis {analysis_id}")
    return record


def require_complete(record: AnalysisRecord) -> AnalysisRecord:
    """Refuse to answer from a half-finished run, and say which state it is in."""
    if record.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=(
                f"analysis {record.analysis_id} is {record.status}; "
                "results are available when it is complete"
            ),
        )
    return record


# ---------------------------------------------------------------------------
# Serialisation helpers — each one reads a core object's own to_dict()
# ---------------------------------------------------------------------------
def device_info(payload: Dict[str, Any]) -> DeviceInfo:
    return DeviceInfo(**payload)


def spec_to_info(spec: Any, requested: Optional[str] = None) -> DeviceInfo:
    payload = spec.to_dict()
    payload["semantic_device"] = spec.kind
    payload["requested"] = requested
    return DeviceInfo(**payload)


def status_response(record: AnalysisRecord) -> AnalysisStatusResponse:
    pipeline = None
    if record.is_complete and record.result is not None:
        metadata = record.result.metadata
        temporal = record.temporal
        pipeline = {
            "frames_processed": record.result.frames_processed,
            "processing_seconds": round(record.result.elapsed_seconds, 3),
            "processing_fps": round(record.result.fps, 2),
            "detections_total": record.result.detections_total,
            "track_reports_total": record.result.track_reports_total,
            "track_ids_created": record.result.track_ids_created,
            "track_frames": dict(record.result.track_frames),
            "event_count": len(record.result.events),
            "entities": temporal.entities() if temporal else [],
            "detector": metadata.get("detector", {}) or {},
            "tracker": metadata.get("tracker"),
            "coordinate_space": "image_pixels",
            "zones": [
                z.get("name")
                for z in ((metadata.get("zones") or {}).get("zones") or [])
                if isinstance(z, dict) and z.get("name")
            ],
        }

    return AnalysisStatusResponse(
        analysis_id=record.analysis_id,
        status=record.status,
        original_filename=record.original_filename,
        created_at=record.created_at,
        elapsed_seconds=record.elapsed_seconds,
        progress=record.progress.to_dict(),
        device=device_info(record.device) if record.device else None,
        video=record.video_metadata,
        pipeline=pipeline,
        error=record.error,
        video_url=f"/api/analyze/{record.analysis_id}/video",
    )


def risk_reports_for(record: AnalysisRecord) -> List[Any]:
    """Assess the clip with the existing engine, once, then cache it.

    Uses ``RiskEngine.assess_timeline`` exactly as ``app/main.py`` does. No
    threshold, weight or scoring decision is made here.
    """
    if record.risk_reports is not None:
        return record.risk_reports
    temporal = record.temporal
    if temporal is None:
        record.risk_reports = []
        return record.risk_reports

    engine = RiskEngine(
        RiskConfig(
            operating_zones=list(record.operating_zones),
            zones=record.config.zones,
        )
    )
    record.risk_reports = engine.assess_timeline(temporal, step=record.risk_step)
    return record.risk_reports


def worst_report(reports: List[Any]) -> Optional[Any]:
    reports = [r for r in reports if r.assessments]
    return max(reports, key=lambda r: r.max_score) if reports else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=AnalysisCreated, status_code=202)
async def create_analysis(
    request: Request,
    background: BackgroundTasks,
    video: UploadFile = File(..., description="A video file to analyse"),
    detector: str = Form("yolo"),
    weights: Optional[str] = Form(None),
    device: str = Form("auto"),
    imgsz: Optional[int] = Form(None),
    confidence: Optional[float] = Form(None),
    classes: Optional[str] = Form(None, description="Comma-separated class names"),
    stride: Optional[int] = Form(None),
    max_frames: Optional[int] = Form(None),
    sample_interval: Optional[float] = Form(None),
    zones: Optional[str] = Form(None, description="'NAME=X1,Y1,X2,Y2', ';'-separated"),
    operating_zones: Optional[str] = Form(None),
    risk_step: float = Form(0.5),
) -> AnalysisCreated:
    """Accept a video, store it, and start the existing pipeline on it.

    Returns ``202`` with an ``analysis_id`` immediately; the run happens in a
    background task. Every option is an existing :class:`PipelineConfig` field
    — this route introduces no new knob and no new default.
    """
    store = get_store(request)

    suffix = os.path.splitext(video.filename or "")[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported video type {suffix or '(none)'}; expected one of "
                f"{', '.join(ALLOWED_SUFFIXES)}"
            ),
        )

    payload = await video.read()
    if not payload:
        raise HTTPException(status_code=400, detail="the uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit",
        )

    analysis_id = store.new_id()
    video_path = store.video_path_for(analysis_id, video.filename or "upload.mp4")
    with open(video_path, "wb") as handle:
        handle.write(payload)

    try:
        config = _config_from_form(
            video_path=video_path,
            detector=detector,
            weights=weights,
            device=device,
            imgsz=imgsz,
            confidence=confidence,
            classes=classes,
            stride=stride,
            max_frames=max_frames,
            sample_interval=sample_interval,
            zones=zones,
        )
        selected = resolve_device(config.device)
    except DeviceUnavailable as exc:
        os.remove(video_path)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        os.remove(video_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = store.add(
        AnalysisRecord(
            analysis_id=analysis_id,
            video_path=video_path,
            original_filename=video.filename or "upload.mp4",
            config=config,
            risk_step=risk_step,
            operating_zones=_split(operating_zones),
        )
    )
    record.device = runner.device_payload(config)
    background.add_task(runner.run_analysis, record)

    return AnalysisCreated(
        analysis_id=analysis_id,
        status=record.status,
        original_filename=record.original_filename,
        device=spec_to_info(selected, requested=config.device),
    )


def _split(value: Optional[str], separator: str = ",") -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(separator) if part.strip()]


def _config_from_form(
    *,
    video_path: str,
    detector: str,
    weights: Optional[str],
    device: str,
    imgsz: Optional[int],
    confidence: Optional[float],
    classes: Optional[str],
    stride: Optional[int],
    max_frames: Optional[int],
    sample_interval: Optional[float],
    zones: Optional[str],
) -> PipelineConfig:
    """Build a PipelineConfig, leaving every unset field at its validated default.

    Zone parsing reuses ``app.main.parse_inline_zone`` rather than repeating the
    format, so the CLI and the API cannot disagree about what a zone is.
    """
    from ...main import parse_inline_zone

    config = PipelineConfig(video_path=video_path, detector=detector, device=device)
    if weights:
        config.weights = weights
    if imgsz is not None:
        config.imgsz = imgsz
    if confidence is not None:
        config.confidence = confidence
    if classes is not None:
        parsed = _split(classes)
        config.classes = parsed or None
    if stride is not None:
        config.stride = stride
    if max_frames is not None:
        config.max_frames = max_frames
    if sample_interval is not None:
        config.sample_interval = sample_interval

    zone_specs = _split(zones, separator=";")
    if zone_specs:
        config.zones = ZoneSet([parse_inline_zone(spec) for spec in zone_specs])

    # The API never writes the CLI's output files; a caller reads JSON from the
    # endpoints instead.
    config.output_path = ""
    config.memory_path = None
    return config


@router.get("/analyze", response_model=AnalysisListResponse)
def list_analyses(request: Request) -> AnalysisListResponse:
    records = get_store(request).list()
    return AnalysisListResponse(
        count=len(records),
        analyses=[status_response(r) for r in records],
    )


@router.get("/analyze/{analysis_id}", response_model=AnalysisStatusResponse)
def get_analysis(request: Request, analysis_id: str) -> AnalysisStatusResponse:
    return status_response(require_record(request, analysis_id))


@router.delete("/analyze/{analysis_id}", status_code=204)
def delete_analysis(request: Request, analysis_id: str) -> None:
    require_record(request, analysis_id)
    get_store(request).delete(analysis_id)


@router.get("/analyze/{analysis_id}/events", response_model=EventsResponse)
def get_events(
    request: Request,
    analysis_id: str,
    entity_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    since: Optional[float] = Query(None),
    until: Optional[float] = Query(None),
) -> EventsResponse:
    """The temporal event stream, straight from ``TemporalEventMemory``."""
    record = require_complete(require_record(request, analysis_id))
    temporal = record.temporal
    if temporal is None:
        return EventsResponse(
            analysis_id=analysis_id, count=0, coordinate_space="image_pixels", events=[]
        )

    criteria: Dict[str, Any] = {}
    if action:
        criteria["actions"] = [action]
    if entity_id:
        events = temporal.entity_history(entity_id, **criteria)
    elif since is not None or until is not None:
        first, last = temporal.span
        events = temporal.between(
            since if since is not None else (first or 0.0),
            until if until is not None else (last or 0.0),
            **criteria,
        )
    else:
        events = temporal.stream(**criteria)

    counts: Dict[str, int] = {}
    for event in events:
        counts[event.action] = counts.get(event.action, 0) + 1

    return EventsResponse(
        analysis_id=analysis_id,
        count=len(events),
        coordinate_space="image_pixels",
        action_counts=counts,
        events=[e.to_dict() for e in events],
        entities=[s.to_dict() for s in temporal.states().values()],
    )


@router.get("/analyze/{analysis_id}/risk", response_model=RiskResponse)
def get_risk(
    request: Request,
    analysis_id: str,
    include_timeline: bool = Query(True),
) -> RiskResponse:
    """The risk timeline produced by ``RiskEngine.assess_timeline``."""
    record = require_complete(require_record(request, analysis_id))
    reports = risk_reports_for(record)
    worst = worst_report(reports)
    interpretation = "0-100 ordinal risk score; NOT a calibrated probability"

    return RiskResponse(
        analysis_id=analysis_id,
        risk_step=record.risk_step,
        coordinate_space=worst.coordinate_space if worst else "image_pixels",
        score_interpretation=interpretation,
        max_risk_score=round(worst.max_score, 2) if worst else 0.0,
        max_severity=worst.severity if worst else "normal",
        report_count=len(reports),
        worst=worst.to_dict() if worst else None,
        timeline=[r.to_dict() for r in reports] if include_timeline else [],
    )


@router.get("/analyze/{analysis_id}/timeline", response_model=TimelineResponse)
def get_timeline(
    request: Request,
    analysis_id: str,
    limit: Optional[int] = Query(None, gt=0),
    offset: int = Query(0, ge=0),
    with_risk: bool = Query(True),
) -> TimelineResponse:
    """Frame-by-frame tracks, events and risk — what the overlay draws from."""
    record = require_complete(require_record(request, analysis_id))
    reports = risk_reports_for(record) if with_risk else []
    rows = runner.build_timeline(record, reports, limit=limit, offset=offset)

    return TimelineResponse(
        analysis_id=analysis_id,
        coordinate_space="image_pixels",
        frame_count=len(record.frames),
        returned=len(rows),
        offset=offset,
        video=record.video_metadata,
        frames=rows,
    )


@router.get("/analyze/{analysis_id}/video")
def get_video(request: Request, analysis_id: str) -> FileResponse:
    """Serve the uploaded video back, byte for byte.

    The server never draws, annotates, re-encodes or modifies it: the frontend
    plays the original and draws boxes from the timeline coordinates.
    """
    record = require_record(request, analysis_id)
    if not os.path.isfile(record.video_path):
        raise HTTPException(status_code=404, detail="the uploaded video is no longer on disk")
    # `inline`, not the FileResponse default of `attachment`: a browser refuses
    # to play a media element whose source is served as a download, which is
    # exactly what the frontend needs this endpoint for. The bytes are
    # unchanged either way.
    return FileResponse(
        record.video_path,
        filename=record.original_filename,
        content_disposition_type="inline",
    )


@router.get("/device", response_model=DeviceResponse, tags=["device"])
def get_device(request: Request, requested: str = Query("auto")) -> DeviceResponse:
    """What hardware this service would use, in Sentinel's semantic terms."""
    try:
        selected = resolve_device(requested)
    except DeviceUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return DeviceResponse(
        selected=spec_to_info(selected, requested=requested),
        available=[spec_to_info(spec) for spec in available_devices()],
        environment=probe_environment().to_dict(),
    )
