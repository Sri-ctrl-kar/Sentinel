#!/usr/bin/env python3
"""Verify Sentinel's Anthropic incident reasoner against the real SDK.

Two modes, one code path:

``--loopback`` (default)
    Drives the genuine ``anthropic`` client against a local HTTP server that
    stands in for ``api.anthropic.com``. No credentials, no internet. This
    proves what the SDK actually puts on the wire from our keyword arguments —
    a hand-written stub cannot, because a stub agrees with whatever you assert
    about it — and replays a recorded response through the SDK's own parsing.

``--live``
    Makes exactly **one** real API call, if a credential is present. Nothing
    else changes: the same evidence, the same schema, the same grounding gate.

Either way the script re-runs the deterministic risk engine before and after
the call and diffs the result, because the claim this whole layer rests on is
that the model has no authority over it.

No API key is ever printed. The script reports which *source* a credential came
from, never its value.

Usage::

    python scripts/verify_anthropic_provider.py            # loopback
    python scripts/verify_anthropic_provider.py --live     # one real call
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.intelligence import (  # noqa: E402
    check_grounding,
    evidence_from_assessment,
)
from app.intelligence.grounding import (  # noqa: E402
    CLAIMED_INTERVENTION,
    ENTITY_COUNT_INFLATION,
    INVENTED_LOCATION,
    INVENTED_NUMBER,
    PAST_TENSE_CLAIM,
    PROBABILITY_LANGUAGE,
    UNACKNOWLEDGED_LIMITATION,
    UNIT_MISMATCH,
    UNKNOWN_ENTITY,
)
from app.intelligence.settings import ReasonerSettings  # noqa: E402
from app.reasoning import RiskEngine  # noqa: E402
from app.scenarios import load_world  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "anthropic_message_response.json")
RULE = "-" * 72

#: Each property M0.7 must hold, and the grounding check that decides it.
PROPERTY_CHECKS = (
    ("no invented numeric measurements", (INVENTED_NUMBER,)),
    ("metres stay metres / pixels stay pixels", (UNIT_MISMATCH,)),
    ("predicted risk described as predicted", (PAST_TENSE_CLAIM,)),
    ("risk score not described as a probability", (PROBABILITY_LANGUAGE,)),
    ("entity IDs stay grounded", (UNKNOWN_ENTITY, ENTITY_COUNT_INFLATION)),
    ("no invented place", (INVENTED_LOCATION,)),
    ("no intervention claimed", (CLAIMED_INTERVENTION,)),
    ("thin evidence acknowledged", (UNACKNOWLEDGED_LIMITATION,)),
)


# ---------------------------------------------------------------------------
# The loopback stand-in for the API
# ---------------------------------------------------------------------------
class _Recorder(BaseHTTPRequestHandler):
    captured: Dict[str, Any] = {}
    body: Dict[str, Any] = {}

    def do_POST(self) -> None:  # noqa: N802 - http.server's spelling
        length = int(self.headers.get("content-length", 0))
        _Recorder.captured = {
            "path": self.path,
            "request": json.loads(self.rfile.read(length)),
            "anthropic_version": self.headers.get("anthropic-version"),
        }
        payload = json.dumps(_Recorder.body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        pass


def loopback_client(response_body: Dict[str, Any]):
    """A real SDK client pointed at a local server that records the request."""
    import anthropic

    _Recorder.body = response_body
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    client = anthropic.Anthropic(
        api_key="loopback-placeholder-not-a-credential",
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        max_retries=0,
    )
    return client, server


# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="make ONE real API call instead of using the loopback server",
    )
    parser.add_argument("--scenario", default="D")
    args = parser.parse_args(argv)

    try:
        import anthropic
    except ImportError:
        print("LIVE_PROVIDER_TEST = NOT_RUN")
        print("REASON = the 'anthropic' package is not installed")
        return 1

    print(RULE)
    print("  SENTINEL M0.7 — ANTHROPIC PROVIDER VERIFICATION")
    print(RULE)
    print(f"anthropic SDK   : {anthropic.__version__}")

    settings = ReasonerSettings.from_env()
    print(f"settings        : {settings.describe()}")

    # --- deterministic half, before anything AI-shaped happens -------------
    scenario = load_world(args.scenario)
    engine = RiskEngine(scenario.config, calibration=scenario.calibration)
    before = engine.assess(scenario.memory, at=scenario.at).to_dict()
    assessment = engine.assess(scenario.memory, at=scenario.at).top
    evidence = evidence_from_assessment(assessment, events=scenario.memory.events)
    evidence_before = evidence.to_json()

    print(f"scenario        : {args.scenario} — {scenario.description}")
    print(
        f"deterministic   : {assessment.incident_type} "
        f"score {assessment.risk_score:.1f} severity {assessment.severity} "
        f"space {assessment.coordinate_space}"
    )

    # --- the call ---------------------------------------------------------
    from app.intelligence.providers.anthropic_claude import AnthropicIncidentReasoner

    server = None
    if args.live:
        if not settings.has_credentials:
            print()
            print("LIVE_PROVIDER_TEST = NOT_RUN")
            print("REASON = missing credentials")
            return 2
        print(f"mode            : LIVE — one call to {settings.model}")
        reasoner = AnthropicIncidentReasoner(settings=settings)
    else:
        with open(FIXTURE, "r", encoding="utf-8") as handle:
            body = json.load(handle)
        client, server = loopback_client(body)
        print("mode            : LOOPBACK — real SDK, local server, no network")
        reasoner = AnthropicIncidentReasoner(client=client, settings=settings)

    try:
        explanation = reasoner.reason(evidence)
    finally:
        if server is not None:
            server.shutdown()

    print()
    if not args.live:
        captured = _Recorder.captured
        print("--- what the SDK put on the wire " + "-" * 39)
        print(f"  endpoint          : POST {captured['path']}")
        print(f"  anthropic-version : {captured['anthropic_version']}")
        request = captured["request"]
        print(f"  model             : {request['model']}")
        print(f"  max_tokens        : {request['max_tokens']}")
        print(f"  thinking          : {json.dumps(request.get('thinking'))}")
        fmt = request["output_config"]["format"]
        print(f"  output_config     : format.type={fmt['type']} "
              f"additionalProperties={fmt['schema']['additionalProperties']} "
              f"required={len(fmt['schema']['required'])} fields")
        print(f"  system            : {len(request['system'])} chars")
        print(f"  user              : {len(request['messages'][0]['content'])} chars "
              "(the evidence, and nothing else)")
        print()

    # --- the response -----------------------------------------------------
    print("--- provider metadata reported by the API " + "-" * 30)
    print(f"  provider          : {explanation.provider}")
    print(f"  model (served)    : {explanation.model}")
    print(f"  is_language_model : {explanation.is_language_model}")
    print()

    print("--- schema validation " + "-" * 50)
    print("  PASS — output parsed and validated as an IncidentExplanation")
    print()

    report = check_grounding(explanation, evidence)
    print("--- grounding gate " + "-" * 53)
    print(f"  {'PASS' if report.ok else 'FAIL'} — {report.summary()}")
    for violation in report.violations:
        print(f"    ! {violation}")
    print()

    print("--- required properties " + "-" * 48)
    codes = set(report.codes)
    for label, offending in PROPERTY_CHECKS:
        hit = codes & set(offending)
        print(f"  [{'FAIL' if hit else 'PASS'}] {label}")
    print()

    print("--- explanation " + "-" * 56)
    print(explanation.to_json())
    print()

    # --- deterministic half, after ---------------------------------------
    after = engine.assess(scenario.memory, at=scenario.at).to_dict()
    unchanged = after == before and evidence.to_json() == evidence_before
    print("--- deterministic risk unchanged " + "-" * 39)
    print(f"  {'YES' if unchanged else 'NO'} — risk report and evidence byte-identical "
          "before and after the provider call")
    print(f"  max risk score    : {before['max_risk_score']} -> {after['max_risk_score']}")
    print(RULE)

    if args.live:
        print("LIVE_PROVIDER_TEST = PASS" if report.ok else "LIVE_PROVIDER_TEST = FAIL")
    else:
        print("LIVE_PROVIDER_TEST = NOT_RUN")
        print("REASON = loopback mode (pass --live with credentials for a real call)")
    return 0 if (report.ok and unchanged) else 3


if __name__ == "__main__":
    raise SystemExit(main())
