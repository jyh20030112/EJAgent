"""Evidence and deterministic verification contracts for host integrations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.evaluation import EvaluationCriterion, EvaluationPlan
from ejagent.contracts.json import JsonValue, freeze_json_value, thaw_json_value
from ejagent.kernel.trajectory import CheckpointSignal


def fingerprint(value: JsonValue) -> str:
    payload = json.dumps(thaw_json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class EvidenceUnavailable(Exception):
    """Expected temporary inability to observe or verify evidence."""


class EvaluationProtocolError(ValueError):
    """An integration broke the evaluator contract; this is not task failure."""


class EvaluationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"

    @property
    def verdict(self) -> bool | None:
        if self is EvaluationStatus.PASS:
            return True
        if self is EvaluationStatus.FAIL:
            return False
        return None


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    """One bounded immutable observation, separate from its location and revision.

    revision must change whenever the source changes, even if it later returns to
    the same value. identity depends on semantic content, not capture timestamps.
    """

    revision: str
    value: JsonValue
    location: str
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        for name in ("revision", "location"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EvaluationProtocolError(f"{name} must be non-empty text")
        object.__setattr__(self, "value", freeze_json_value(self.value))
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise EvaluationProtocolError("observation time must be timezone-aware")

    @property
    def identity(self) -> str:
        return fingerprint(self.value)


class EvidenceSource(Protocol):
    """Host-owned source. Reads are read-only; revision checks bracket all reads."""

    async def revision(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> str: ...

    async def read(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> EvidenceSnapshot: ...

    def close_run(self, run_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CheckResult:
    """A verifier's short, inspectable conclusion using declared evidence keys."""

    status: EvaluationStatus
    rationale: str
    evidence_keys: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, EvaluationStatus):
            raise EvaluationProtocolError("status must be EvaluationStatus")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise EvaluationProtocolError("rationale must be non-empty text")
        for name in ("evidence_keys", "missing_evidence"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise EvaluationProtocolError(f"{name} must contain non-empty text")
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class ItemEvaluation:
    criterion_id: str
    method: str
    status: EvaluationStatus
    rationale: str
    evidence_refs: tuple[str, ...]
    evidence_versions: Mapping[str, str]
    missing_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_versions", MappingProxyType(dict(self.evidence_versions))
        )
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "missing_evidence", tuple(self.missing_evidence))


@dataclass(frozen=True, slots=True)
class EvaluationCost:
    """Per-checkpoint evaluator work, separate from Actor actions and requests."""

    source_reads: int = 0
    revision_checks: int = 0
    verifier_calls: int = 0
    cache_hits: int = 0
    elapsed_ms: int = 0
    model_requests: int = 0
    model_input_tokens: int = 0
    model_output_tokens: int = 0
    model_unreported_requests: int = 0


@dataclass(frozen=True, slots=True)
class JudgeAttempt:
    """One admitted judge request; includes format recovery, never raw output."""

    criterion_id: str
    request_index: int
    retry_index: int
    format_steered: bool
    outcome: str
    detail: str
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    run_id: str
    checkpoint_id: str
    plan: EvaluationPlan | None
    requirements: tuple[ItemEvaluation, ...]
    constraints: tuple[ItemEvaluation, ...]
    evidence: Mapping[str, EvidenceSnapshot]
    diagnostics: Mapping[str, str]
    invalidated_refs: tuple[str, ...]
    new_evidence: tuple[str, ...]
    fact_capture_complete: bool
    cost: EvaluationCost
    judge_attempts: tuple[JudgeAttempt, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )
        for name in (
            "requirements",
            "constraints",
            "invalidated_refs",
            "new_evidence",
            "judge_attempts",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def report_ref(self) -> str:
        return f"evaluation:{self.checkpoint_id}"

    def evidence_ref(self, key: str) -> str:
        return f"evidence:{self.run_id}:{key}:{self.evidence[key].identity}"


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    """A rule must depend only on its declared evidence and immutable criterion.

    signal and previous_report supply diagnostics/association; dependencies on
    completion text must declare the reserved `$completion` evidence key.
    """

    criterion: EvaluationCriterion
    evidence: Mapping[str, EvidenceSnapshot]
    signal: CheckpointSignal
    previous_report: EvaluationReport | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


Verifier = Callable[[VerificationRequest, CancellationToken], Awaitable[CheckResult]]
ReportSink = Callable[[EvaluationReport], None]
