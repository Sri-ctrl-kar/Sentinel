"""The M0.9 HTTP contract.

Deterministic and offline: the detector is ``MockDetector``, the clip is the
synthetic one the suite renders itself, and the incident reasoner is the
deterministic mock. No GPU, no model weights, no downloads, no network.

FastAPI is an optional dependency, so the whole module skips when it is absent —
the rest of the suite keeps passing on a machine that never installed it.
"""

from __future__ import annotations

import io
import json

import pytest

fastapi = pytest.importorskip(
    "fastapi", reason="the API is optional; pip install -r requirements-api.txt"
)

from fastapi.testclient import TestClient  # noqa: E402

from app.api.schemas import (  # noqa: E402
    EventPayload,
    SituationLifecyclePayload,
    ExplanationPayload,
    GroundingPayload,
    IncidentEvidencePayload,
    LifecyclePointPayload,
    RiskAssessmentPayload,
    RiskReportPayload,
    TrackPayload,
)
from app.intelligence.lifecycle import INCIDENT_STATES  # noqa: E402
from app.api.server import ApiSettings, create_app  # noqa: E402
from app.api.store import AnalysisStore  # noqa: E402

# `device: cpu` is part of what makes this suite deterministic, like the mock
# detector beside it. Without it the upload asks for `auto`, which correctly
# prefers whatever accelerator the machine has — Metal on a Mac, ROCm on the
# benchmark box — and a contract test would then assert the hardware it happens
# to run on. Production device selection is not touched: `auto` still means
# auto, and a test that wants to check `auto` asks for it explicitly below.
MOCK_RUN = {
    "detector": "mock",
    "sample_interval": "0.1",
    "risk_step": "0.2",
    "device": "cpu",
}


@pytest.fixture
def store(tmp_path):
    store = AnalysisStore(upload_dir=str(tmp_path / "uploads"))
    yield store
    store.clear()


@pytest.fixture
def client(store):
    """A client whose background tasks run before the response is returned.

    Starlette's TestClient runs background tasks synchronously, which makes
    every assertion here deterministic: no polling, no sleeps, no flakes. In
    production under uvicorn the response is sent first and the run happens
    after, which is what the 202 and the status endpoint are for.
    """
    with TestClient(create_app(store=store)) as client:
        yield client


@pytest.fixture
def video_bytes(demo_video):
    with open(demo_video, "rb") as handle:
        return handle.read()


def upload(client, video_bytes, filename="clip.mp4", content_type="video/mp4", **form):
    data = dict(MOCK_RUN)
    for key, value in form.items():
        if value is None:
            # Drop the field entirely, so a test can exercise the API's own
            # default rather than a value this helper chose for determinism.
            data.pop(key, None)
        else:
            data[key] = str(value)
    return client.post(
        "/api/analyze",
        files={"video": (filename, io.BytesIO(video_bytes), content_type)},
        data=data,
    )


@pytest.fixture
def analysis_id(client, video_bytes):
    response = upload(client, video_bytes)
    assert response.status_code == 202, response.text
    return response.json()["analysis_id"]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
def test_health_reports_the_service(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "sentinel-api"
    assert body["api_version"] == "0.9"


def test_the_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Sentinel API"
    paths = schema["paths"]
    for route in (
        "/api/analyze",
        "/api/analyze/{analysis_id}",
        "/api/analyze/{analysis_id}/events",
        "/api/analyze/{analysis_id}/risk",
        "/api/analyze/{analysis_id}/timeline",
        "/api/analyze/{analysis_id}/incident",
        "/api/analyze/{analysis_id}/video",
        "/api/device",
        "/api/health",
    ):
        assert route in paths, f"missing route {route}"


def test_the_description_keeps_the_ordinal_score_disclaimer(client):
    schema = client.get("/openapi.json").json()
    assert "not probabilities" in schema["info"]["description"]


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
def test_an_upload_creates_an_analysis(client, video_bytes):
    response = upload(client, video_bytes)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] in ("queued", "running", "complete")
    assert body["original_filename"] == "clip.mp4"
    assert body["device"]["kind"] == "cpu"
    assert len(body["analysis_id"]) == 32


def test_a_missing_file_is_rejected(client):
    assert client.post("/api/analyze", data=MOCK_RUN).status_code == 422


def test_an_unsupported_extension_is_rejected(client, video_bytes):
    response = upload(client, video_bytes, filename="clip.txt")
    assert response.status_code == 415
    assert "unsupported video type" in response.json()["detail"]


def test_an_empty_upload_is_rejected(client):
    response = client.post(
        "/api/analyze",
        files={"video": ("clip.mp4", io.BytesIO(b""), "video/mp4")},
        data=MOCK_RUN,
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_a_rejected_upload_is_not_stored(client, store, video_bytes):
    upload(client, video_bytes, filename="clip.exe")
    assert len(store) == 0


def test_an_unavailable_device_is_refused_before_running(client, video_bytes):
    from app.accel import gpu_kind

    if gpu_kind() == "rocm":  # pragma: no cover - only on an AMD host
        pytest.skip("this machine has an AMD GPU")
    response = upload(client, video_bytes, device="rocm")
    assert response.status_code == 409
    assert "AMD ROCm" in response.json()["detail"]


def test_a_bad_zone_specification_is_a_client_error(client, video_bytes):
    response = upload(client, video_bytes, zones="malformed-zone")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def test_status_reports_a_completed_run(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}").json()
    assert body["status"] == "complete"
    assert body["error"] is None
    assert body["progress"]["frames_processed"] > 0
    assert body["progress"]["percent"] is not None
    assert body["video"]["width"] > 0 and body["video"]["fps"] > 0
    assert body["video_url"] == f"/api/analyze/{analysis_id}/video"

    pipeline = body["pipeline"]
    assert pipeline["frames_processed"] == body["progress"]["frames_processed"]
    assert pipeline["detections_total"] > 0
    assert pipeline["track_ids_created"] > 0
    assert pipeline["event_count"] > 0
    assert pipeline["coordinate_space"] == "image_pixels"
    assert pipeline["detector"]["backend"] == "mock"
    assert pipeline["tracker"] == "byte_iou"
    assert pipeline["entities"]


def test_status_for_an_unknown_analysis_is_404(client):
    assert client.get("/api/analyze/nope").status_code == 404


def test_analyses_can_be_listed_and_deleted(client, analysis_id):
    listing = client.get("/api/analyze").json()
    assert listing["count"] == 1
    assert listing["analyses"][0]["analysis_id"] == analysis_id

    assert client.delete(f"/api/analyze/{analysis_id}").status_code == 204
    assert client.get(f"/api/analyze/{analysis_id}").status_code == 404
    assert client.get("/api/analyze").json()["count"] == 0


def test_the_four_statuses_are_the_documented_ones():
    from app.api.store import STATUSES

    assert STATUSES == ("queued", "running", "complete", "failed")


def test_a_queued_analysis_reports_queued_before_it_runs(client, store, video_bytes):
    """The state machine, observed without racing the runner."""
    from app.api.routes.analysis import status_response
    from app.api.store import AnalysisRecord
    from app.config import PipelineConfig

    record = store.add(
        AnalysisRecord(
            analysis_id="pending",
            video_path="clip.mp4",
            original_filename="clip.mp4",
            config=PipelineConfig(video_path="clip.mp4", detector="mock"),
        )
    )
    body = client.get("/api/analyze/pending").json()
    assert body["status"] == "queued"
    assert body["pipeline"] is None
    assert body["elapsed_seconds"] is None

    record.mark_running()
    assert client.get("/api/analyze/pending").json()["status"] == "running"
    assert status_response(record).progress.frames_processed == 0


def test_results_are_refused_while_an_analysis_is_incomplete(client, store):
    from app.api.store import AnalysisRecord
    from app.config import PipelineConfig

    store.add(
        AnalysisRecord(
            analysis_id="pending",
            video_path="clip.mp4",
            original_filename="clip.mp4",
            config=PipelineConfig(video_path="clip.mp4", detector="mock"),
        )
    )
    for suffix in ("events", "risk", "timeline", "incident"):
        response = client.get(f"/api/analyze/pending/{suffix}")
        assert response.status_code == 409, suffix
        assert "queued" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
def test_an_unreadable_video_ends_as_failed(client):
    response = client.post(
        "/api/analyze",
        files={"video": ("broken.mp4", io.BytesIO(b"not a video at all"), "video/mp4")},
        data=MOCK_RUN,
    )
    assert response.status_code == 202
    body = client.get(f"/api/analyze/{response.json()['analysis_id']}").json()
    assert body["status"] == "failed"
    assert body["error"]["type"]
    assert body["error"]["message"]
    assert body["pipeline"] is None


def test_a_failed_analysis_still_answers_its_status(client):
    response = client.post(
        "/api/analyze",
        files={"video": ("broken.mp4", io.BytesIO(b"garbage"), "video/mp4")},
        data=MOCK_RUN,
    )
    analysis_id = response.json()["analysis_id"]
    assert client.get(f"/api/analyze/{analysis_id}").status_code == 200
    assert client.get(f"/api/analyze/{analysis_id}/events").status_code == 409


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
def test_events_serialise_with_the_core_field_names(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/events").json()
    assert body["count"] == len(body["events"])
    assert body["coordinate_space"] == "image_pixels"
    assert body["action_counts"]

    event = body["events"][0]
    for field in ("timestamp", "entity_id", "action", "coordinate_space", "attributes"):
        assert field in event
    assert EventPayload(**event)


def test_events_carry_boxes_for_the_frontend(client, analysis_id):
    events = client.get(f"/api/analyze/{analysis_id}/events").json()["events"]
    boxed = [e for e in events if e.get("bbox")]
    assert boxed
    assert len(boxed[0]["bbox"]) == 4


def test_events_include_folded_entity_states(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/events").json()
    assert body["entities"]
    state = body["entities"][0]
    for field in ("entity_id", "first_seen", "last_seen", "present", "event_count"):
        assert field in state


def test_events_can_be_filtered_by_entity_and_action(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/events").json()
    entity = body["events"][0]["entity_id"]

    per_entity = client.get(
        f"/api/analyze/{analysis_id}/events", params={"entity_id": entity}
    ).json()
    assert per_entity["count"] > 0
    assert {e["entity_id"] for e in per_entity["events"]} == {entity}

    appeared = client.get(
        f"/api/analyze/{analysis_id}/events", params={"action": "appeared"}
    ).json()
    assert {e["action"] for e in appeared["events"]} == {"appeared"}


def test_events_can_be_windowed_by_time(client, analysis_id):
    body = client.get(
        f"/api/analyze/{analysis_id}/events", params={"since": 0.0, "until": 0.5}
    ).json()
    assert all(0.0 <= e["timestamp"] <= 0.5 for e in body["events"])


def test_the_api_event_payload_declares_every_core_event_key(client, analysis_id):
    """Drift guard: if Event.to_dict() grows a key, this test says so."""
    events = client.get(f"/api/analyze/{analysis_id}/events").json()["events"]
    declared = set(EventPayload.model_fields)
    for event in events:
        assert set(event) <= declared, set(event) - declared


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
def test_risk_serialises_the_engines_own_timeline(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/risk").json()
    assert body["report_count"] == len(body["timeline"])
    assert body["risk_step"] == 0.2
    assert body["coordinate_space"] == "image_pixels"
    assert RiskResponseIsSane(body)


def RiskResponseIsSane(body) -> bool:
    assert 0.0 <= body["max_risk_score"] <= 100.0
    assert body["max_severity"] in ("normal", "low", "medium", "high", "critical")
    return True


def test_risk_carries_the_ordinal_score_disclaimer(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/risk").json()
    assert "NOT a calibrated probability" in body["score_interpretation"]
    if body["worst"]:
        assert "NOT a calibrated probability" in body["worst"]["score_interpretation"]


def without_nulls(value):
    """Drop null-valued keys, recursively.

    The core's ``to_dict()`` omits unset optional fields; the API declares them
    and returns ``null`` so a client sees a stable key set. Comparing without
    nulls checks the *values* agree, which is the thing that matters.
    """
    if isinstance(value, dict):
        return {k: without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [without_nulls(v) for v in value]
    return value


def test_the_worst_report_matches_the_engine(client, analysis_id, store):
    from app.reasoning import RiskEngine

    body = client.get(f"/api/analyze/{analysis_id}/risk").json()
    record = store.get(analysis_id)
    reports = RiskEngine().assess_timeline(record.temporal, step=0.2)
    expected = max((r for r in reports if r.assessments), key=lambda r: r.max_score)
    assert without_nulls(body["worst"]) == without_nulls(expected.to_dict())
    assert body["max_risk_score"] == round(expected.max_score, 2)


def test_the_api_declares_optional_fields_the_core_omits(client, analysis_id, store):
    """A documented, deliberate difference: stable keys instead of absent ones.

    ``TimeToRisk.to_dict()`` leaves ``reason`` out when there is none; the API
    returns ``"reason": null``. Values never differ — only presence does — so a
    frontend can read one key set for every response.
    """
    from app.reasoning import RiskEngine

    record = store.get(analysis_id)
    reports = RiskEngine().assess_timeline(record.temporal, step=0.2)
    expected = max((r for r in reports if r.assessments), key=lambda r: r.max_score)
    core = expected.to_dict()["assessments"][0]["time_to_risk"]

    served = client.get(f"/api/analyze/{analysis_id}/risk").json()
    api = served["worst"]["assessments"][0]["time_to_risk"]

    assert "reason" in api
    assert set(core) <= set(api)
    for key, value in core.items():
        assert api[key] == value


def test_risk_assessments_carry_factors_and_time_to_risk(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/risk").json()
    assessment = body["worst"]["assessments"][0]
    assert RiskAssessmentPayload(**assessment)
    assert assessment["contributing_factors"]
    factor = assessment["contributing_factors"][0]
    for field in ("name", "score", "weight", "contribution", "rationale"):
        assert field in factor
    assert "recommended_intervention" in assessment


def test_the_risk_timeline_can_be_omitted(client, analysis_id):
    body = client.get(
        f"/api/analyze/{analysis_id}/risk", params={"include_timeline": False}
    ).json()
    assert body["timeline"] == []
    assert body["report_count"] > 0


def test_the_api_risk_payload_declares_every_core_risk_key(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/risk").json()
    report_fields = set(RiskReportPayload.model_fields)
    assert set(body["worst"]) <= report_fields
    assessment_fields = set(RiskAssessmentPayload.model_fields)
    for assessment in body["worst"]["assessments"]:
        assert set(assessment) <= assessment_fields


# ---------------------------------------------------------------------------
# Frame timeline / track serialisation
# ---------------------------------------------------------------------------
def test_the_timeline_serialises_per_frame_tracks(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/timeline").json()
    assert body["frame_count"] == body["returned"] > 0
    assert body["coordinate_space"] == "image_pixels"
    assert body["video"]["width"] > 0

    populated = [f for f in body["frames"] if f["tracks"]]
    assert populated
    track = populated[0]["tracks"][0]
    assert TrackPayload(**track)
    assert track["track_id"] >= 1
    assert track["class_name"]
    assert 0.0 <= track["confidence"] <= 1.0
    assert len(track["bbox"]) == 4
    assert len(track["position"]) == 2


def test_the_timeline_row_carries_everything_requirement_8_asks_for(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/timeline").json()
    row = next(f for f in body["frames"] if f["tracks"])
    assert "frame_index" in row and "timestamp" in row
    track = row["tracks"][0]
    assert {"track_id", "class_name", "bbox", "confidence"} <= set(track)
    assert "events" in row
    assert "risk" in row


def test_the_timeline_attaches_risk_where_it_exists(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/timeline").json()
    with_risk = [f for f in body["frames"] if f["risk"]]
    assert with_risk
    risk = with_risk[0]["risk"]
    assert risk["severity"] in ("normal", "low", "medium", "high", "critical")
    assert risk["coordinate_space"] == "image_pixels"


def test_the_timeline_paginates(client, analysis_id):
    full = client.get(f"/api/analyze/{analysis_id}/timeline").json()
    page = client.get(
        f"/api/analyze/{analysis_id}/timeline", params={"limit": 4, "offset": 3}
    ).json()
    assert page["returned"] == 4
    assert page["offset"] == 3
    assert page["frames"][0] == full["frames"][3]


def test_the_timeline_is_denser_than_the_event_stream(client, analysis_id):
    """Requirement 9, from the outside: the overlay channel is frame-rate dense."""
    timeline = client.get(f"/api/analyze/{analysis_id}/timeline").json()
    events = client.get(f"/api/analyze/{analysis_id}/events").json()
    frames_with_tracks = sum(1 for f in timeline["frames"] if f["tracks"])
    assert frames_with_tracks > events["count"] / 2


def test_risk_can_be_left_out_of_the_timeline(client, analysis_id):
    body = client.get(
        f"/api/analyze/{analysis_id}/timeline", params={"with_risk": False}
    ).json()
    assert all(f["risk"] is None for f in body["frames"])


# ---------------------------------------------------------------------------
# Video passthrough
# ---------------------------------------------------------------------------
def test_the_original_video_is_served_back_unmodified(client, analysis_id, video_bytes):
    response = client.get(f"/api/analyze/{analysis_id}/video")
    assert response.status_code == 200
    assert response.content == video_bytes, "the server must not re-encode the upload"


def test_the_video_is_served_inline_so_a_browser_can_play_it(client, analysis_id):
    """`attachment` makes a browser download the file instead of playing it.

    The frontend renders the original video in a media element, so the
    disposition is part of the contract, not a detail.
    """
    response = client.get(f"/api/analyze/{analysis_id}/video")
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("inline")
    assert "attachment" not in disposition
    assert response.headers["content-type"] == "video/mp4"


def test_the_video_route_supports_range_requests(client, analysis_id):
    """Seeking in the player asks for byte ranges."""
    assert client.get(f"/api/analyze/{analysis_id}/video").headers["accept-ranges"] == "bytes"


def test_the_video_route_404s_for_an_unknown_analysis(client):
    assert client.get("/api/analyze/nope/video").status_code == 404


# ---------------------------------------------------------------------------
# Incident: evidence -> explanation -> grounding
# ---------------------------------------------------------------------------
def test_the_incident_endpoint_returns_the_full_chain(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/incident").json()
    assert body["incident_found"] is True
    assert body["lifecycle_state"] in (
        "observed",
        "developing",
        "imminent",
        "current",
        "resolved",
    )
    assert IncidentEvidencePayload(**body["evidence"])
    assert ExplanationPayload(**body["explanation"])
    assert GroundingPayload(**body["grounding"])


def test_the_incident_reports_the_lifecycle_the_situation_passed_through(
    client, analysis_id
):
    """The clip's states over time, not only the state at its worst moment.

    The worst moment is the one the score peaks at, which is systematically the
    moment the situation is already unsafe — so a response carrying only that
    state can almost never show ``developing`` or ``imminent``, however clearly
    the engine predicted them earlier.
    """
    body = client.get(f"/api/analyze/{analysis_id}/incident").json()
    history = [LifecyclePointPayload(**point) for point in body["lifecycle_history"]]

    assert history, "a found incident always passed through at least one state"
    assert [p.timestamp for p in history] == sorted(p.timestamp for p in history)
    assert {p.state for p in history} <= set(INCIDENT_STATES)

    # The scalar the rest of the response describes is the entry at the worst
    # moment — the history adds to it and never contradicts it.
    worst_timestamp = body["evidence"]["timestamp"]
    at_worst = [p for p in history if p.timestamp == pytest.approx(worst_timestamp)]
    assert len(at_worst) == 1
    assert at_worst[0].state == body["lifecycle_state"]

    # This clip is a converging approach: it is predicted before it happens.
    assert "imminent" in {p.state for p in history}
    assert history[0].timestamp < worst_timestamp


def test_every_situation_in_the_clip_gets_its_own_lifecycle(client, analysis_id, store):
    """A viewer scrubbing a long clip meets more than the worst situation.

    The incident endpoint describes one moment; the video has many. Every
    assessed entity pair is reported with its own lifecycle so the UI can
    follow whichever situation is live at the playhead without inventing a
    state for it.
    """
    from app.api.routes.analysis import risk_reports_for
    from app.intelligence.lifecycle import situation_key

    record = store.get(analysis_id)
    reports = risk_reports_for(record)
    expected_keys = {
        situation_key(a) for r in reports for a in r.assessments
    }

    body = client.get(f"/api/analyze/{analysis_id}/incident").json()
    situations = [SituationLifecyclePayload(**s) for s in body["situations"]]

    assert situations, "a found incident means at least one situation exists"
    assert {tuple(sorted(s.entity_ids)) for s in situations} == expected_keys
    for situation in situations:
        assert situation.history, "a situation with no history is not a situation"
        stamps = [p.timestamp for p in situation.history]
        assert stamps == sorted(stamps)
        assert len(stamps) == len(set(stamps))
        assert {p.state for p in situation.history} <= set(INCIDENT_STATES)


def test_the_worst_situation_appears_in_both_places_identically(client, analysis_id):
    """`lifecycle_history` is the worst situation's entry, not a separate walk."""
    body = client.get(f"/api/analyze/{analysis_id}/incident").json()
    worst_ids = sorted(e["entity_id"] for e in body["evidence"]["entities"])
    matching = [
        s for s in body["situations"] if sorted(s["entity_ids"]) == worst_ids
    ]
    assert len(matching) == 1
    assert matching[0]["history"] == body["lifecycle_history"]


def test_situations_are_derived_by_the_m07_tracker_per_pair(client, analysis_id, store):
    """One tracker, many situations: it keys on the pair, so they stay apart."""
    from app.api.routes.analysis import risk_reports_for
    from app.api.routes.incident import situation_lifecycles
    from app.intelligence.lifecycle import LifecycleTracker, situation_key

    record = store.get(analysis_id)
    reports = risk_reports_for(record)

    tracker = LifecycleTracker()
    expected: dict = {}
    for report in reports:
        for assessment in report.assessments:
            expected.setdefault(situation_key(assessment), []).append(
                tracker.update(assessment)
            )

    produced = {
        tuple(sorted(s["entity_ids"])): [p["state"] for p in s["history"]]
        for s in situation_lifecycles(reports)
    }
    assert produced == expected


def test_a_situation_is_absent_where_the_engine_assessed_nothing(client, analysis_id, store):
    """No history point may exist for a step that produced no assessment."""
    from app.api.routes.analysis import risk_reports_for
    from app.api.routes.incident import situation_lifecycles
    from app.intelligence.lifecycle import situation_key

    record = store.get(analysis_id)
    reports = risk_reports_for(record)
    assessed = {
        (situation_key(a), round(a.timestamp, 4))
        for r in reports
        for a in r.assessments
    }

    for situation in situation_lifecycles(reports):
        key = tuple(sorted(situation["entity_ids"]))
        for point in situation["history"]:
            assert (key, point["timestamp"]) in assessed


def test_the_lifecycle_history_follows_one_situation_only(client, analysis_id, store):
    """Blending entity pairs would report a progression that never happened."""
    from app.api.routes.incident import lifecycle_history
    from app.api.routes.analysis import risk_reports_for, worst_report
    from app.intelligence.lifecycle import situation_key

    record = store.get(analysis_id)
    reports = risk_reports_for(record)
    worst = worst_report(reports)
    key = situation_key(worst.top)

    history = lifecycle_history(reports, worst.top)
    stamps = [p["timestamp"] for p in history]

    # Every step whose own assessments contain this pair is represented once,
    # and no step contributes twice.
    expected = [
        round(r.timestamp, 4)
        for r in reports
        if any(situation_key(a) == key for a in r.assessments)
    ]
    assert stamps == expected
    assert len(stamps) == len(set(stamps))


def test_the_lifecycle_history_is_derived_by_the_m07_tracker(client, analysis_id, store):
    """No second derivation: the route's output must equal the tracker's."""
    from app.api.routes.incident import lifecycle_history
    from app.api.routes.analysis import risk_reports_for, worst_report
    from app.intelligence.lifecycle import LifecycleTracker, situation_key

    record = store.get(analysis_id)
    reports = risk_reports_for(record)
    worst = worst_report(reports)
    key = situation_key(worst.top)

    tracker = LifecycleTracker()
    expected = []
    for report in reports:
        for candidate in report.assessments:
            if situation_key(candidate) == key:
                expected.append(tracker.update(candidate))
                break

    assert [p["state"] for p in lifecycle_history(reports, worst.top)] == expected


def test_the_incident_evidence_comes_from_the_validated_builder(client, analysis_id, store):
    from app.intelligence import evidence_from_assessment
    from app.reasoning import RiskEngine

    record = store.get(analysis_id)
    reports = [r for r in RiskEngine().assess_timeline(record.temporal, step=0.2) if r.assessments]
    worst = max(reports, key=lambda r: r.max_score)
    expected = evidence_from_assessment(worst.top, events=record.temporal.events)

    body = client.get(f"/api/analyze/{analysis_id}/incident").json()
    assert body["evidence"] == expected.to_dict()


def test_incident_evidence_states_its_units_and_space(client, analysis_id):
    evidence = client.get(f"/api/analyze/{analysis_id}/incident").json()["evidence"]
    assert evidence["coordinate_space"] == "image_pixels"
    assert evidence["distance_unit"] == "px"
    assert evidence["speed_unit"] == "px/s"
    assert "NOT a calibrated probability" in evidence["score_interpretation"]


def test_incident_quantities_carry_units(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/incident").json()
    assert body["quantities"]
    for quantity in body["quantities"]:
        assert quantity["unit"] in ("px", "px/s", "m", "m/s")
        assert quantity["coordinate_space"] == "image_pixels"


def test_the_default_reasoner_is_the_deterministic_mock(client, analysis_id):
    body = client.get(f"/api/analyze/{analysis_id}/incident").json()
    assert body["explanation"]["provider"] == "mock"
    assert body["explanation"]["is_language_model"] is False
    assert "deterministic template" in body["reasoner"]


def test_the_explanation_can_be_skipped(client, analysis_id):
    body = client.get(
        f"/api/analyze/{analysis_id}/incident", params={"explain": False}
    ).json()
    assert body["evidence"] is not None
    assert body["explanation"] is None
    assert body["grounding"] is None


def test_the_grounding_report_is_serialised(client, analysis_id):
    grounding = client.get(f"/api/analyze/{analysis_id}/incident").json()["grounding"]
    assert grounding["ok"] is True
    assert grounding["violations"] == []
    assert grounding["incident_id"]


def test_grounding_violations_serialise_when_they_occur(client, analysis_id, monkeypatch):
    """A failing explanation must reach the client as structured violations."""
    from app.intelligence.schema import IncidentExplanation
    from app.intelligence.providers import mock as mock_provider

    def hallucinate(self, evidence):
        return IncidentExplanation.from_model_output(
            {
                "summary": "Three workers were struck in the warehouse aisle.",
                "severity_explanation": "There is an 87% probability of injury.",
                "evidence_points": ["they were 17.25 metres apart"],
                "predicted_outcome": "The collision occurred.",
                "recommended_action": "The forklift was stopped automatically.",
                "urgency": "immediate",
                "uncertainty": "None.",
                "coordinate_space": evidence.coordinate_space,
            },
            incident_id=evidence.incident_id,
            provider="mock",
            model="test",
        )

    monkeypatch.setattr(mock_provider.MockIncidentReasoner, "reason", hallucinate)
    body = client.get(f"/api/analyze/{analysis_id}/incident").json()
    grounding = body["grounding"]
    assert grounding["ok"] is False
    codes = {v["code"] for v in grounding["violations"]}
    assert {"invented_location", "probability_language", "claimed_intervention"} <= codes
    for violation in grounding["violations"]:
        assert GroundingViolationHasShape(violation)


def GroundingViolationHasShape(violation) -> bool:
    assert set(violation) == {"code", "detail", "excerpt"}
    return True


def test_malformed_reasoner_output_is_a_502(client, analysis_id, monkeypatch):
    from app.intelligence.providers import mock as mock_provider
    from app.intelligence.schema import ExplanationSchemaError

    def broken(self, evidence):
        raise ExplanationSchemaError("missing field(s): urgency")

    monkeypatch.setattr(mock_provider.MockIncidentReasoner, "reason", broken)
    response = client.get(f"/api/analyze/{analysis_id}/incident")
    assert response.status_code == 502
    assert "schema validation" in response.json()["detail"]


def test_an_unknown_reasoner_is_a_client_error(client, analysis_id):
    response = client.get(
        f"/api/analyze/{analysis_id}/incident", params={"reasoner": "gpt-9000"}
    )
    assert response.status_code == 400
    assert "unknown reasoner" in response.json()["detail"]


def test_a_provider_without_credentials_is_a_503(client, analysis_id):
    from app.accel import probe_environment  # noqa: F401 - keeps import order stable
    from app.intelligence.settings import detect_credential_source
    import os

    if detect_credential_source(dict(os.environ)) is not None:  # pragma: no cover
        pytest.skip("a credential is present; the provider would really be built")
    response = client.get(
        f"/api/analyze/{analysis_id}/incident", params={"reasoner": "anthropic"}
    )
    assert response.status_code == 503
    assert "mock" in response.json()["detail"]


def test_no_incident_is_reported_honestly(client, store, video_bytes, monkeypatch):
    """A clip with no assessable situation says so instead of inventing one."""
    from app.api.routes import incident as incident_routes

    # Patched where it is *used*: incident.py imported the name, so replacing it
    # on analysis.py would not be seen.
    monkeypatch.setattr(incident_routes, "worst_report", lambda reports: None)
    response = upload(client, video_bytes)
    analysis_id = response.json()["analysis_id"]
    body = client.get(f"/api/analyze/{analysis_id}/incident").json()
    assert body["incident_found"] is False
    assert body["evidence"] is None
    assert "no assessment" in body["note"]


# ---------------------------------------------------------------------------
# Device metadata (requirement 12)
# ---------------------------------------------------------------------------
def test_the_device_endpoint_reports_the_semantic_device(client):
    body = client.get("/api/device").json()
    selected = body["selected"]
    assert selected["kind"] in ("cpu", "rocm", "cuda", "mps")
    assert selected["semantic_device"] == selected["kind"]
    assert selected["label"]
    assert body["available"]
    assert body["environment"]["os"]
    assert "hip" in body["environment"]


def test_the_device_endpoint_never_calls_rocm_cuda(client, monkeypatch):
    import sys

    sys.path.insert(0, "tests")
    from test_accel_device import fake_torch

    from app.accel import device as accel

    monkeypatch.setattr(
        accel,
        "_torch",
        lambda: fake_torch(cuda_available=True, hip="6.2", names=("MI300X",)),
    )
    selected = client.get("/api/device", params={"requested": "rocm"}).json()["selected"]
    assert selected["kind"] == "rocm"
    assert selected["semantic_device"] == "rocm"
    assert selected["label"] == "AMD ROCm / HIP"
    assert selected["torch_device"] == "cuda"  # the runtime string, named as such
    assert "NVIDIA" not in json.dumps(selected)


def test_requesting_an_unavailable_device_is_409(client):
    from app.accel import gpu_kind

    if gpu_kind() == "rocm":  # pragma: no cover
        pytest.skip("this machine has an AMD GPU")
    assert client.get("/api/device", params={"requested": "rocm"}).status_code == 409


def test_the_analysis_reports_the_device_it_ran_on(client, analysis_id):
    """This run asked for the CPU, so the CPU is what it must report."""
    device = client.get(f"/api/analyze/{analysis_id}").json()["device"]
    assert device["kind"] == "cpu"
    assert device["semantic_device"] == "cpu"
    assert device["requested"] == "cpu"


def test_an_upload_without_a_device_asks_for_auto_and_says_so(client, video_bytes):
    """The default still reaches the pipeline, and the run reports what it got.

    Deliberately no assertion that the result is the CPU: `auto` resolves to
    this machine's accelerator, and pinning that here would make the suite fail
    on exactly the hardware Sentinel is built to use.
    """
    body = upload(client, video_bytes, device=None).json()
    device = client.get(f"/api/analyze/{body['analysis_id']}").json()["device"]
    assert device["requested"] == "auto"
    assert device["kind"] in ("cpu", "rocm", "cuda", "mps")
    assert device["semantic_device"] == device["kind"]


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
def test_cors_is_enabled_for_the_frontend_origin(store):
    settings = ApiSettings(cors_origins=["http://localhost:5173"])
    with TestClient(create_app(settings=settings, store=store)) as client:
        response = client.get(
            "/api/health", headers={"Origin": "http://localhost:5173"}
        )
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_a_cors_preflight_is_answered(store):
    settings = ApiSettings(cors_origins=["http://localhost:5173"])
    with TestClient(create_app(settings=settings, store=store)) as client:
        response = client.options(
            "/api/analyze",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-methods" in response.headers


def test_cors_can_be_disabled(store):
    with TestClient(create_app(settings=ApiSettings(cors_origins=[]), store=store)) as client:
        response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert "access-control-allow-origin" not in response.headers


def test_cors_origins_come_from_the_environment():
    settings = ApiSettings.from_env(
        {"SENTINEL_API_CORS_ORIGINS": "https://a.example, https://b.example"}
    )
    assert settings.cors_origins == ["https://a.example", "https://b.example"]
    assert settings.cors_enabled

    assert ApiSettings.from_env({"SENTINEL_API_CORS_ORIGINS": ""}).cors_enabled is False
    assert ApiSettings.from_env({}).cors_origins[0].startswith("http://localhost")


def test_settings_describe_themselves_without_secrets():
    described = ApiSettings.from_env({}).describe()
    assert "cors=on" in described
    for marker in ("key", "token", "secret", "password"):
        assert marker not in described.lower()


# ---------------------------------------------------------------------------
# A situation that resolves and escalates again
# ---------------------------------------------------------------------------
def _lifecycle_sequence():
    """Reports for one pair that goes imminent → resolved → imminent → current.

    Built directly rather than detected from a clip: a resolve-and-re-escalate
    sequence is exactly the shape that is hard to find on demand in real
    footage, and these are the states the frontend trail has to render in the
    order they occurred.
    """
    from app.reasoning.models.risk import RiskAssessment, RiskReport, TimeToRisk
    from app.reasoning.prediction import (
        CURRENTLY_UNSAFE_PROXIMITY,
        NO_PREDICTED_CONFLICT,
        PREDICTED_TRAJECTORY_CONFLICT,
    )

    def imminent(timestamp, score):
        return RiskAssessment(
            risk_score=score, severity="medium",
            incident_type="PERSON_VEHICLE_COLLISION_RISK",
            involved_entity_ids=["person_1", "car_3"], timestamp=timestamp,
            prediction_outcome=PREDICTED_TRAJECTORY_CONFLICT,
            time_to_risk=TimeToRisk(status=TimeToRisk.STATUS_PREDICTED, seconds=0.4),
        )

    def quiet(timestamp):
        return RiskAssessment(
            risk_score=2.0, severity="normal",
            incident_type="CLOSE_PROXIMITY",
            involved_entity_ids=["person_1", "car_3"], timestamp=timestamp,
            prediction_outcome=NO_PREDICTED_CONFLICT,
        )

    def unsafe(timestamp):
        return RiskAssessment(
            risk_score=70.0, severity="high",
            incident_type="PERSON_VEHICLE_COLLISION_RISK",
            involved_entity_ids=["person_1", "car_3"], timestamp=timestamp,
            prediction_outcome=CURRENTLY_UNSAFE_PROXIMITY,
            time_to_risk=TimeToRisk(status=TimeToRisk.STATUS_ALREADY_UNSAFE, seconds=0.0),
        )

    assessments = [
        imminent(10.0, 40.0), imminent(10.5, 41.0),   # imminent, held
        quiet(12.0), quiet(12.5),                      # resolved, held
        imminent(14.0, 44.0),                          # imminent again
        unsafe(16.0),                                  # current
    ]
    return [RiskReport(timestamp=a.timestamp, assessments=[a]) for a in assessments]


def test_a_situation_that_resolves_and_escalates_again_keeps_every_step():
    """The history is a record, not a set of states.

    A trail that kept only the first time each state appeared could not place
    the second `imminent`, and would print `resolved` — first seen at 12.0 s —
    above an `imminent` first seen at 10.0 s. The backend therefore reports one
    point per assessment, in the order the engine produced them, repeats and
    all.
    """
    from app.api.routes.incident import situation_lifecycles

    reports = _lifecycle_sequence()
    situations = situation_lifecycles(reports)
    assert len(situations) == 1
    history = situations[0]["history"]

    # `resolved` needs the previous state to have been active, so the second
    # quiet assessment is `observed` — the tracker's own rule, not a guess.
    assert [p["state"] for p in history] == [
        "imminent", "imminent", "resolved", "observed", "imminent", "current",
    ]
    stamps = [p["timestamp"] for p in history]
    assert stamps == [10.0, 10.5, 12.0, 12.5, 14.0, 16.0]
    assert stamps == sorted(stamps)


def test_the_re_escalation_timestamp_is_the_engine_timestamp():
    """No point is moved or invented: each one is its assessment's own moment."""
    from app.api.routes.incident import situation_lifecycles

    reports = _lifecycle_sequence()
    history = situation_lifecycles(reports)[0]["history"]
    assessed = [a.timestamp for r in reports for a in r.assessments]
    assert [p["timestamp"] for p in history] == assessed
    assert [p["risk_score"] for p in history] == [
        round(a.risk_score, 2) for r in reports for a in r.assessments
    ]


def test_both_lifecycle_views_agree_on_a_re_escalation():
    """`lifecycle_history` and `situations` walk the same sequence."""
    from app.api.routes.incident import lifecycle_history, situation_lifecycles

    reports = _lifecycle_sequence()
    worst = max(
        (a for r in reports for a in r.assessments), key=lambda a: a.risk_score
    )
    assert lifecycle_history(reports, worst) == situation_lifecycles(reports)[0]["history"]
