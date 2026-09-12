"""Mechanical grounding checks on generated explanations.

A prompt is a request; this module is the enforcement. Every explanation is
compared against the evidence it was supposed to describe, and specific,
named failures are reported:

* a number that is not in the evidence,
* a pixel measurement relabelled as metres (or vice versa),
* a risk score restated as a probability or a percentage,
* a predicted conflict written as something that has happened,
* an entity, a place or an object that does not exist,
* a claim that Sentinel did something — it never does,
* silence about a limitation the evidence explicitly recorded.

The checks are intentionally blunt and textual. They can be stricter than a
careful human reader would be; that is the right trade for a safety system,
because a false alarm here costs a rewrite while a miss costs a fabricated
incident report. Every violation names the offending text so a human can
judge it.

No model runtime is imported. These checks run in CI with no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from ..spatial import GROUND_PLANE_METERS
from .evidence import IncidentEvidence
from .schema import IncidentExplanation

# --- violation codes -------------------------------------------------------
INVENTED_NUMBER = "invented_number"
UNIT_MISMATCH = "unit_mismatch"
PROBABILITY_LANGUAGE = "probability_language"
PAST_TENSE_CLAIM = "past_tense_claim"
MISSING_PREDICTION_FRAMING = "missing_prediction_framing"
UNKNOWN_ENTITY = "unknown_entity"
ENTITY_COUNT_INFLATION = "entity_count_inflation"
INVENTED_LOCATION = "invented_location"
CLAIMED_INTERVENTION = "claimed_intervention"
UNACKNOWLEDGED_LIMITATION = "unacknowledged_limitation"
COORDINATE_SPACE_MISMATCH = "coordinate_space_mismatch"

#: How far a stated number may sit from an evidence value and still count as
#: the same number. Covers the rounding a writer naturally applies.
NUMBER_TOLERANCE = 0.05

#: Numbers that are always allowed because they describe the scale itself,
#: not a measurement: the bounds of the ordinal risk score.
SCALE_CONSTANTS = (0.0, 100.0)

#: Signed decimals, not preceded by a word character or a dot — so digits
#: inside identifiers ("person_1") and version-like strings are not read as
#: measurements. The sign is part of the number: a closing speed of -4.0 is
#: not the same fact as 4.0.
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
# Identifier-shaped tokens: underscore-joined names (``person_1``,
# ``closing_speed``) and word-plus-number tokens (``forklift3``,
# ``vehicle-4``). Ordinary hyphenated English ("forward-looking") is not an
# identifier and is deliberately not matched.
_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z]+_[A-Za-z0-9_]+\b|\b[A-Za-z]{2,}-?\d+\b"
)

#: A percentage is always a probability claim: there is nothing in the
#: evidence a "%" could legitimately refer to.
_PERCENT_RE = re.compile(r"\d\s?%", re.IGNORECASE)

#: These words are violations *unless negated* — "the risk score is not a
#: probability" is exactly the disclaimer Sentinel wants, and forbidding the
#: word outright would forbid saying so.
_PROBABILITY_WORD_RE = re.compile(
    r"\bpercent\b|\bpercentage\b|\bprobability\b|\bprobabilities\b"
    r"|\blikelihood\b|\bchance\b|\bchances\b|\bodds\b",
    re.IGNORECASE,
)

_IMPERIAL_OR_METRIC_RE = re.compile(
    r"\bmet(?:er|re)s?\b|\bmet(?:er|re)s per second\b|\bm/s\b|\bkm/h\b|\bkph\b"
    r"|\bmph\b|\bfeet\b|\bfoot\b|\bft\b|\byards?\b|\binches\b|\bmiles?\b"
    r"|\d\s*m\b",
    re.IGNORECASE,
)
_PIXEL_RE = re.compile(r"\bpixels?\b|\bpx\b", re.IGNORECASE)

_OCCURRENCE_RE = re.compile(
    r"\bcollided\b|\bcollision (?:has )?occurred\b|\bhas collided\b|\bstruck\b"
    r"|\bhit the\b|\bwas hit\b|\bwere hit\b|\bcrashed\b|\bimpact occurred\b"
    r"|\bwas injured\b|\bwere injured\b|\bhas already happened\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|yet to|has not|have not)\b[^.;]*$", re.IGNORECASE
)

_INTERVENTION_RE = re.compile(
    r"\bhas been (?:stopped|halted|notified|alerted|warned|evacuated)\b"
    r"|\bwas (?:stopped|halted|notified|alerted|warned|evacuated|shut down)\b"
    r"|\bwere (?:stopped|halted|notified|alerted|warned|evacuated)\b"
    r"|\bsentinel (?:stopped|halted|alerted|notified|sounded|triggered|shut)\b"
    r"|\bwe (?:stopped|halted|alerted|notified)\b"
    r"|\bthe (?:alarm|siren|horn) (?:was |has been )?sounded\b"
    r"|\bbrakes were applied\b|\bemergency stop was\b",
    re.IGNORECASE,
)

_FORWARD_MARKERS = (
    "predict",
    "will ",
    "would ",
    "expected",
    "forecast",
    "projected",
    "on track",
    "if current",
    "if the current",
    "is set to",
    "about to",
)

_LIMITATION_MARKERS = (
    "insufficient",
    "not enough",
    "too few",
    "limited",
    "unavailable",
    "could not",
    "cannot",
    "no prediction",
    "unable",
)

#: Words that name a place. Sentinel's evidence contains no places, only zone
#: names — so any of these that is not part of a zone name was invented.
_LOCATION_WORDS = (
    "warehouse",
    "aisle",
    "dock",
    "loading bay",
    "street",
    "road",
    "highway",
    "intersection",
    "crosswalk",
    "sidewalk",
    "pavement",
    "factory",
    "plant",
    "building",
    "site",
    "yard",
    "corridor",
    "ramp",
    "parking",
    "garage",
    "shop floor",
    "warehouse floor",
    "depot",
    "terminal",
    "platform",
    "junction",
)

_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}
_COUNT_NOUNS = {
    "entities": None,
    "objects": None,
    "tracks": None,
    "people": "person",
    "persons": "person",
    "workers": "person",
    "pedestrians": "person",
    "vehicles": "vehicle",
    "forklifts": "forklift",
    "trucks": "truck",
    "cars": "car",
}
_COUNT_RE = re.compile(
    r"\b(one|two|three|four|five|six|\d+)\s+(" + "|".join(_COUNT_NOUNS) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GroundingViolation:
    """One specific way an explanation departed from its evidence."""

    code: str
    detail: str
    excerpt: str = ""

    def __str__(self) -> str:
        suffix = f" — {self.excerpt!r}" if self.excerpt else ""
        return f"[{self.code}] {self.detail}{suffix}"

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "excerpt": self.excerpt}


@dataclass(frozen=True)
class GroundingReport:
    """The verdict on one explanation."""

    incident_id: str
    violations: List[GroundingViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> List[str]:
        return [v.code for v in self.violations]

    def has(self, code: str) -> bool:
        return code in self.codes

    def summary(self) -> str:
        if self.ok:
            return "grounded: every claim traces to the supplied evidence"
        return f"{len(self.violations)} grounding violation(s): " + ", ".join(
            sorted(set(self.codes))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
        }


class GroundingError(RuntimeError):
    """Raised when an explanation fails its grounding check."""

    def __init__(self, report: GroundingReport) -> None:
        super().__init__(report.summary())
        self.report = report


def check_grounding(
    explanation: IncidentExplanation, evidence: IncidentEvidence
) -> GroundingReport:
    """Compare an explanation against the evidence it claims to describe."""
    text = explanation.full_text()
    violations: List[GroundingViolation] = []

    violations.extend(_check_coordinate_space(explanation, evidence))
    violations.extend(_check_units(text, evidence))
    violations.extend(_check_numbers(text, evidence))
    violations.extend(_check_probability(text))
    violations.extend(_check_tense(text, evidence))
    violations.extend(_check_entities(text, evidence))
    violations.extend(_check_counts(text, evidence))
    violations.extend(_check_locations(text, evidence))
    violations.extend(_check_intervention(text))
    violations.extend(_check_limitations(text, evidence))

    return GroundingReport(incident_id=evidence.incident_id, violations=violations)


def assert_grounded(
    explanation: IncidentExplanation, evidence: IncidentEvidence
) -> GroundingReport:
    """Check, and raise :class:`GroundingError` if anything is ungrounded."""
    report = check_grounding(explanation, evidence)
    if not report.ok:
        raise GroundingError(report)
    return report


# ---------------------------------------------------------------------------
def _check_coordinate_space(
    explanation: IncidentExplanation, evidence: IncidentEvidence
) -> List[GroundingViolation]:
    if explanation.coordinate_space != evidence.coordinate_space:
        return [
            GroundingViolation(
                COORDINATE_SPACE_MISMATCH,
                "explanation claims coordinate space "
                f"{explanation.coordinate_space!r}; the evidence is "
                f"{evidence.coordinate_space!r}",
            )
        ]
    return []


def _check_units(text: str, evidence: IncidentEvidence) -> List[GroundingViolation]:
    if evidence.coordinate_space == GROUND_PLANE_METERS:
        match = _PIXEL_RE.search(text)
        if match:
            return [
                GroundingViolation(
                    UNIT_MISMATCH,
                    "calibrated ground-plane evidence described in pixels",
                    _excerpt(text, match),
                )
            ]
        return []

    match = _IMPERIAL_OR_METRIC_RE.search(text)
    if match:
        return [
            GroundingViolation(
                UNIT_MISMATCH,
                "image-pixel evidence described as a physical distance; "
                "no calibration was applied so no measurement is in metres",
                _excerpt(text, match),
            )
        ]
    return []


def _check_numbers(text: str, evidence: IncidentEvidence) -> List[GroundingViolation]:
    allowed = _allowed_numbers(evidence)
    scrubbed = _strip_known_identifiers(text, evidence)
    violations: List[GroundingViolation] = []
    for match in _NUMBER_RE.finditer(scrubbed):
        value = float(match.group())
        if _is_allowed_number(value, allowed):
            continue
        violations.append(
            GroundingViolation(
                INVENTED_NUMBER,
                f"{match.group()} does not appear in the evidence",
                _excerpt(scrubbed, match),
            )
        )
    return violations


def _allowed_numbers(evidence: IncidentEvidence) -> List[float]:
    values = [float(v) for v in evidence.numeric_facts()]
    values.extend(SCALE_CONSTANTS)
    return values


def _is_allowed_number(value: float, allowed: Iterable[float]) -> bool:
    for candidate in allowed:
        if abs(value - candidate) <= NUMBER_TOLERANCE + 1e-9:
            return True
        for places in (0, 1, 2, 3):
            if abs(round(candidate, places) - value) <= 1e-9:
                return True
    return False


def _known_identifiers(evidence: IncidentEvidence) -> Set[str]:
    """Strings from the evidence that legitimately contain digits or joins."""
    known: Set[str] = {
        evidence.incident_id,
        evidence.incident_type,
        evidence.incident_state,
        evidence.severity,
        evidence.coordinate_space,
        evidence.distance_unit,
        evidence.speed_unit,
    }
    known.update(evidence.entity_ids)
    known.update(f.name for f in evidence.factors)
    for entity in evidence.entities:
        known.update(entity.zones)
        if entity.class_name:
            known.add(entity.class_name)
    if evidence.prediction is not None:
        known.add(evidence.prediction.outcome)
        if evidence.prediction.unavailable_reason:
            known.add(evidence.prediction.unavailable_reason)
    if evidence.time_to_risk is not None:
        known.add(evidence.time_to_risk.status)
        if evidence.time_to_risk.reason:
            known.add(evidence.time_to_risk.reason)
    known.update(evidence.triggered_event_ids)
    known.update(evidence.triggered_event_actions)
    # Field names from the evidence itself: an explanation that quotes
    # ``closing_speed`` is quoting Sentinel, not inventing an entity.
    known.update(_nested_keys(evidence.to_dict()))
    return {k for k in known if k}


def _nested_keys(payload: Any) -> Set[str]:
    keys: Set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key))
            keys |= _nested_keys(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            keys |= _nested_keys(item)
    return keys


def _strip_known_identifiers(text: str, evidence: IncidentEvidence) -> str:
    """Remove evidence identifiers so their digits are not read as claims."""
    scrubbed = text
    for identifier in sorted(_known_identifiers(evidence), key=len, reverse=True):
        if len(identifier) < 3:
            # Never strip "m" or "px" — they are units, and removing them
            # everywhere would chew through ordinary words.
            continue
        scrubbed = re.sub(
            rf"\b{re.escape(identifier)}\b", " ", scrubbed, flags=re.IGNORECASE
        )
    return scrubbed


def _check_probability(text: str) -> List[GroundingViolation]:
    violations: List[GroundingViolation] = []
    for match in _PERCENT_RE.finditer(text):
        violations.append(
            GroundingViolation(
                PROBABILITY_LANGUAGE,
                "states a percentage; the risk score is an ordinal 0-100 "
                "engineering signal and no evidence value is a percentage",
                _excerpt(text, match),
            )
        )
    for match in _PROBABILITY_WORD_RE.finditer(text):
        if _is_negated(text, match.start()):
            continue
        violations.append(
            GroundingViolation(
                PROBABILITY_LANGUAGE,
                "risk is stated as a probability, likelihood or chance; the "
                "risk score is an ordinal engineering signal",
                _excerpt(text, match),
            )
        )
    return violations


def _check_tense(text: str, evidence: IncidentEvidence) -> List[GroundingViolation]:
    unsafe_now = bool(
        evidence.time_to_risk is not None and evidence.time_to_risk.is_already_unsafe
    )
    violations: List[GroundingViolation] = []

    if not unsafe_now:
        for match in _OCCURRENCE_RE.finditer(text):
            if _is_negated(text, match.start()):
                continue
            violations.append(
                GroundingViolation(
                    PAST_TENSE_CLAIM,
                    "describes a conflict as having occurred; the evidence "
                    "reports a prediction, not an event",
                    _excerpt(text, match),
                )
            )

    if evidence.has_usable_prediction and not unsafe_now:
        lowered = text.lower()
        if not any(marker in lowered for marker in _FORWARD_MARKERS):
            violations.append(
                MISSING_FRAMING_VIOLATION,
            )
    return violations


MISSING_FRAMING_VIOLATION = GroundingViolation(
    MISSING_PREDICTION_FRAMING,
    "the evidence is a forward-looking prediction but the explanation never "
    "frames it as one",
)


def _is_negated(text: str, index: int) -> bool:
    window = text[max(0, index - 30) : index]
    return bool(_NEGATION_RE.search(window))


def _check_entities(text: str, evidence: IncidentEvidence) -> List[GroundingViolation]:
    known = {k.lower() for k in _known_identifiers(evidence)}
    known.update({"image_pixels", "ground_plane_meters", "time_to_risk", "m/s"})
    violations: List[GroundingViolation] = []
    seen: Set[str] = set()
    for match in _IDENTIFIER_RE.finditer(text):
        token = match.group()
        lowered = token.lower()
        if lowered in known or lowered in seen:
            continue
        seen.add(lowered)
        violations.append(
            GroundingViolation(
                UNKNOWN_ENTITY,
                f"{token!r} is not an entity, zone or term in the evidence",
                _excerpt(text, match),
            )
        )
    return violations


def _check_counts(text: str, evidence: IncidentEvidence) -> List[GroundingViolation]:
    violations: List[GroundingViolation] = []
    classes = [(e.class_name or "").lower() for e in evidence.entities]
    for match in _COUNT_RE.finditer(text):
        stated_raw, noun = match.group(1).lower(), match.group(2).lower()
        stated = _COUNT_WORDS.get(stated_raw)
        if stated is None:
            try:
                stated = int(stated_raw)
            except ValueError:  # pragma: no cover - regex guarantees digits
                continue
        wanted = _COUNT_NOUNS[noun]
        if wanted is None:
            actual = len(evidence.entities)
        elif wanted == "vehicle":
            actual = sum(1 for c in classes if c and c != "person")
        else:
            actual = sum(1 for c in classes if c == wanted)
        if stated > actual:
            violations.append(
                GroundingViolation(
                    ENTITY_COUNT_INFLATION,
                    f"claims {stated} {noun}; the evidence lists {actual}",
                    _excerpt(text, match),
                )
            )
    return violations


def _check_locations(text: str, evidence: IncidentEvidence) -> List[GroundingViolation]:
    allowed = _zone_vocabulary(evidence)
    lowered = text.lower()
    violations: List[GroundingViolation] = []
    for word in _LOCATION_WORDS:
        if word in allowed:
            continue
        match = re.search(rf"\b{re.escape(word)}\b", lowered)
        if match:
            violations.append(
                GroundingViolation(
                    INVENTED_LOCATION,
                    f"names a location ({word!r}) that the evidence does not "
                    "contain; Sentinel evidence has no site or place names",
                    _excerpt(text, match),
                )
            )
    return violations


def _zone_vocabulary(evidence: IncidentEvidence) -> Set[str]:
    words: Set[str] = set()
    for entity in evidence.entities:
        for zone in entity.zones:
            lowered = zone.lower()
            words.add(lowered)
            words.update(re.split(r"[_\-\s]+", lowered))
    return {w for w in words if w}


def _check_intervention(text: str) -> List[GroundingViolation]:
    match = _INTERVENTION_RE.search(text)
    if match:
        return [
            GroundingViolation(
                CLAIMED_INTERVENTION,
                "claims an intervention was carried out; Sentinel observes and "
                "reports, it never acts",
                _excerpt(text, match),
            )
        ]
    return []


def _check_limitations(
    text: str, evidence: IncidentEvidence
) -> List[GroundingViolation]:
    needs_caveat = evidence.insufficient_history or (
        evidence.prediction is not None
        and evidence.prediction.unavailable_reason is not None
    )
    if not needs_caveat:
        return []
    lowered = text.lower()
    if any(marker in lowered for marker in _LIMITATION_MARKERS):
        return []
    reason = (
        evidence.prediction.unavailable_reason if evidence.prediction else "unknown"
    )
    return [
        GroundingViolation(
            UNACKNOWLEDGED_LIMITATION,
            f"the evidence records that no prediction was possible ({reason}) "
            "but the explanation does not say so",
        )
    ]


def _excerpt(text: str, match: "re.Match[str]", width: int = 48) -> str:
    start = max(0, match.start() - width // 2)
    end = min(len(text), match.end() + width // 2)
    return text[start:end].strip().replace("\n", " ")
