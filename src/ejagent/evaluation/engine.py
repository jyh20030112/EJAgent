"""Run-scoped evidence collection, validation, caching, and invalidation."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import TypeVar

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.evaluation import EvaluationCriterion, EvaluationPlan
from ejagent.contracts.json import thaw_json_value
from ejagent.evaluation.judge import JudgeUsage, ModelJudge
from ejagent.evaluation.types import (
    CheckResult,
    EvaluationCost,
    EvaluationProtocolError,
    EvaluationReport,
    EvaluationStatus,
    EvidenceSnapshot,
    EvidenceSource,
    EvidenceUnavailable,
    ItemEvaluation,
    ReportSink,
    VerificationRequest,
    Verifier,
    fingerprint,
)
from ejagent.kernel.trajectory import CheckpointSignal, CheckpointTrigger

T = TypeVar("T")


@dataclass
class _Work:
    started: float = field(default_factory=monotonic)
    judge_before: JudgeUsage = field(default_factory=JudgeUsage)
    reads: int = 0
    revision_checks: int = 0
    calls: int = 0
    hits: int = 0


@dataclass
class _RunState:
    plan: EvaluationPlan | None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    previous: EvaluationReport | None = None
    work: _Work = field(default_factory=_Work)
    seen: set[str] = field(default_factory=set)
    cache: dict[str, tuple[tuple[tuple[str, str, str], ...], CheckResult]] = field(
        default_factory=dict
    )


class GoalEvaluator:
    """Evaluate fixed criteria using deterministic checks and an optional model judge."""

    def __init__(
        self,
        *,
        sources: Mapping[str, EvidenceSource],
        verifiers: Mapping[str, Verifier],
        timeout_seconds: float = 2.0,
        max_concurrency: int = 4,
        max_evidence_bytes: int = 1_048_576,
        report_sink: ReportSink | None = None,
        semantic_judge: ModelJudge | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        for value in (max_concurrency, max_evidence_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("evaluation bounds must be positive integers")
        if "$completion" in sources:
            raise ValueError("$completion is a reserved evidence source")
        self._sources = dict(sources)
        self._verifiers = dict(verifiers)
        self._timeout = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_evidence_bytes = max_evidence_bytes
        self._sink = report_sink
        self._judge = semantic_judge
        self._runs: dict[str, _RunState] = {}

    @property
    def resources(self) -> tuple[object, ...]:
        return (self._judge.model,) if self._judge is not None else ()

    @property
    def active_run_ids(self) -> tuple[str, ...]:
        return tuple(self._runs)

    def latest_report(self, run_id: str) -> EvaluationReport | None:
        state = self._runs.get(run_id)
        return state.previous if state else None

    async def evaluate(
        self,
        checkpoint_id: str,
        signal: CheckpointSignal,
        *,
        cancellation: CancellationToken,
    ) -> EvaluationReport:
        if not isinstance(signal, CheckpointSignal):
            raise TypeError("signal must be CheckpointSignal")
        cancellation.raise_if_cancelled()
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise EvaluationProtocolError("checkpoint_id must be non-empty text")
        state = self._runs.setdefault(signal.run_id, _RunState(signal.evaluation_plan))
        await cancellation.run(state.lock.acquire())
        try:
            if state.plan != signal.evaluation_plan:
                raise EvaluationProtocolError(
                    "evaluation plan cannot change within a Run"
                )
            state.work = _Work(
                judge_before=self._judge.usage(signal.run_id)
                if self._judge
                else JudgeUsage()
            )
            try:
                report = await cancellation.run(
                    self._evaluate(checkpoint_id, signal, state, cancellation)
                )
            except BaseException as exc:
                interrupted = self._interrupted_report(
                    checkpoint_id, signal, state, type(exc).__name__
                )
                state.previous = interrupted
                if self._sink is not None:
                    try:
                        self._sink(interrupted)
                    except Exception:
                        pass  # Preserve the original cancellation/protocol error.
                error: BaseException = exc
                while isinstance(error, ExceptionGroup) and len(error.exceptions) == 1:
                    error = error.exceptions[0]
                raise error from None
            if self._sink is not None:
                self._sink(report)
            state.previous = report
            return report
        finally:
            state.lock.release()

    async def _bounded(
        self, operation: Callable[[], Awaitable[T]], cancellation: CancellationToken
    ) -> T:
        async def invoke() -> T:
            async with self._semaphore:
                async with asyncio.timeout(self._timeout):
                    return await operation()

        return await cancellation.run(invoke())

    async def _evaluate(
        self,
        checkpoint_id: str,
        signal: CheckpointSignal,
        state: _RunState,
        cancellation: CancellationToken,
    ) -> EvaluationReport:
        work = state.work
        plan = state.plan
        if plan is None:
            return EvaluationReport(
                signal.run_id,
                checkpoint_id,
                None,
                (),
                (),
                {},
                {},
                (),
                (),
                False,
                EvaluationCost(),
            )
        criteria = (*plan.requirements, *plan.constraints)
        active_ids = {
            item.criterion_id
            for item in criteria
            if not item.completion_only
            or signal.trigger is CheckpointTrigger.COMPLETION_PROPOSED
        }
        keys = tuple(
            dict.fromkeys(
                key
                for item in criteria
                if item.criterion_id in active_ids
                for key in item.evidence_keys
            )
        )
        snapshots: dict[str, EvidenceSnapshot] = {}
        diagnostics: dict[str, str] = {}
        before: dict[str, str] = {}
        after: dict[str, str] = {}

        async def revision(key: str, target: dict[str, str]) -> None:
            if key == "$completion":
                return
            source = self._sources.get(key)
            if source is None:
                diagnostics[key] = "evidence source is not configured"
                return
            work.revision_checks += 1
            try:
                value = await self._bounded(
                    lambda: source.revision(signal, cancellation=cancellation),
                    cancellation,
                )
                if not isinstance(value, str) or not value.strip():
                    raise EvaluationProtocolError(
                        "source revision must be non-empty text"
                    )
                target[key] = value
            except (EvidenceUnavailable, OSError, TimeoutError) as exc:
                diagnostics[key] = f"{type(exc).__name__}: {exc}".strip()[:512]

        async def read(key: str) -> None:
            if key in diagnostics:
                return
            if key == "$completion":
                candidate = signal.completion_candidate
                if candidate is None or candidate.truncated:
                    diagnostics[key] = "complete proposed final response is unavailable"
                    return
                snapshots[key] = EvidenceSnapshot(
                    fingerprint(candidate.text),
                    candidate.text,
                    f"candidate:{checkpoint_id}",
                )
                return
            source = self._sources[key]
            work.reads += 1
            try:
                value = await self._bounded(
                    lambda: source.read(signal, cancellation=cancellation), cancellation
                )
                if not isinstance(value, EvidenceSnapshot):
                    raise EvaluationProtocolError("source must return EvidenceSnapshot")
                # Size limits apply to host sources as well as built-in file reads.
                if (
                    len(json.dumps(thaw_json_value(value.value)).encode())
                    > self._max_evidence_bytes
                ):
                    raise EvidenceUnavailable("evidence exceeds configured size bound")
                snapshots[key] = value
            except (EvidenceUnavailable, OSError, TimeoutError) as exc:
                diagnostics[key] = f"{type(exc).__name__}: {exc}".strip()[:512]

        # Three barriers bracket the entire evidence set, not just individual reads.
        async with asyncio.TaskGroup() as group:
            for key in keys:
                group.create_task(revision(key, before))
        async with asyncio.TaskGroup() as group:
            for key in keys:
                group.create_task(read(key))
        async with asyncio.TaskGroup() as group:
            for key in keys:
                group.create_task(revision(key, after))
        raced = any(
            key != "$completion"
            and key not in diagnostics
            and (before[key] != after[key] or snapshots[key].revision != after[key])
            for key in snapshots
        )
        if raced:
            for key in keys:
                diagnostics[key] = (
                    "environment changed during capture; no consistent snapshot"
                )
        for key in diagnostics:
            snapshots.pop(key, None)

        def ref(key: str) -> str:
            return f"evidence:{signal.run_id}:{key}:{snapshots[key].identity}"

        async def verify(item: EvaluationCriterion) -> ItemEvaluation:
            if item.criterion_id not in active_ids:
                return ItemEvaluation(
                    item.criterion_id,
                    item.method,
                    EvaluationStatus.UNKNOWN,
                    "scheduled for completion audit",
                    (),
                    {},
                    ("proposed completion",),
                )
            missing = tuple(key for key in item.evidence_keys if key not in snapshots)
            dependencies = tuple(
                (key, snapshots[key].revision, snapshots[key].identity)
                for key in item.evidence_keys
                if key in snapshots
            )
            cached = state.cache.get(item.criterion_id)
            if missing:
                result = CheckResult(
                    EvaluationStatus.UNKNOWN,
                    "required evidence is unavailable",
                    missing_evidence=missing,
                )
                state.cache.pop(item.criterion_id, None)
            elif cached is not None and cached[0] == dependencies:
                result = cached[1]
                work.hits += 1
            else:
                request = VerificationRequest(
                    item,
                    {key: snapshots[key] for key in item.evidence_keys},
                    signal,
                    state.previous,
                )

                async def rule(method: str) -> CheckResult:
                    verifier = self._verifiers.get(method)
                    if verifier is None:
                        return CheckResult(
                            EvaluationStatus.UNKNOWN,
                            f"verification method {method!r} is not configured",
                            missing_evidence=(f"method:{method}",),
                        )
                    work.calls += 1
                    try:
                        value = await self._bounded(
                            lambda: verifier(request, cancellation), cancellation
                        )
                        self._validate_result(value, item)
                        return value
                    except (EvidenceUnavailable, OSError, TimeoutError) as exc:
                        return CheckResult(
                            EvaluationStatus.UNKNOWN,
                            f"{type(exc).__name__}: {exc}"[:512],
                            missing_evidence=(f"verification:{item.criterion_id}",),
                        )

                if item.semantic:
                    guard = await rule(item.guard_method) if item.guard_method else None
                    if guard is not None and guard.status is not EvaluationStatus.PASS:
                        result = guard
                    elif self._judge is None:
                        result = CheckResult(
                            EvaluationStatus.UNKNOWN,
                            "semantic judge is not configured",
                            missing_evidence=("semantic judge",),
                        )
                    else:
                        result = await self._judge.evaluate(request, cancellation)
                else:
                    result = await rule(item.method)
                self._validate_result(result, item)
                if result.status in (EvaluationStatus.PASS, EvaluationStatus.FAIL):
                    state.cache[item.criterion_id] = (dependencies, result)
                else:
                    state.cache.pop(item.criterion_id, None)
            return ItemEvaluation(
                item.criterion_id,
                item.method,
                result.status,
                result.rationale,
                tuple(ref(key) for key in result.evidence_keys),
                {
                    key: snapshots[key].revision
                    for key in item.evidence_keys
                    if key in snapshots
                },
                result.missing_evidence,
            )

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(verify(item)) for item in criteria]
        results = tuple(task.result() for task in tasks)
        # Slow verifiers must not certify an environment changed during validation.
        final_revisions: dict[str, str] = {}
        async with asyncio.TaskGroup() as group:
            for key in snapshots:
                group.create_task(revision(key, final_revisions))
        changed_during_verification = any(
            key != "$completion"
            and (key in diagnostics or final_revisions.get(key) != evidence.revision)
            for key, evidence in snapshots.items()
        )
        if changed_during_verification:
            diagnostics.update(
                {
                    key: "environment changed or became unavailable during verification"
                    for key in keys
                }
            )
            snapshots.clear()
            state.cache.clear()
            results = tuple(
                ItemEvaluation(
                    item.criterion_id,
                    item.method,
                    EvaluationStatus.UNKNOWN,
                    "verification snapshot is no longer current",
                    (),
                    {},
                    item.evidence_keys,
                )
                for item in criteria
            )
        invalidated: list[str] = []
        if state.previous is not None:
            for key, old in state.previous.evidence.items():
                current = snapshots.get(key)
                if current is None or (current.revision, current.identity) != (
                    old.revision,
                    old.identity,
                ):
                    invalidated.append(state.previous.evidence_ref(key))
        new = tuple(ref(key) for key in snapshots if ref(key) not in state.seen)
        state.seen.update(new)
        complete = (
            bool(snapshots)
            and not diagnostics
            and all(
                item.status.verdict is not None
                for item in results
                if item.criterion_id in active_ids
            )
        )
        return EvaluationReport(
            signal.run_id,
            checkpoint_id,
            plan,
            results[: len(plan.requirements)],
            results[len(plan.requirements) :],
            snapshots,
            diagnostics,
            tuple(invalidated),
            new,
            complete,
            self._cost(state, signal),
            judge_attempts=self._judge.attempts(signal.run_id)[
                work.judge_before.requests :
            ]
            if self._judge
            else (),
        )

    def _cost(self, state: _RunState, signal: CheckpointSignal) -> EvaluationCost:
        work = state.work
        after = self._judge.usage(signal.run_id) if self._judge else JudgeUsage()
        before = work.judge_before
        return EvaluationCost(
            work.reads,
            work.revision_checks,
            work.calls,
            work.hits,
            int((monotonic() - work.started) * 1000),
            model_requests=after.requests - before.requests,
            model_input_tokens=after.input_tokens - before.input_tokens,
            model_output_tokens=after.output_tokens - before.output_tokens,
            model_unreported_requests=after.unreported_requests
            - before.unreported_requests,
        )

    def _interrupted_report(
        self,
        checkpoint_id: str,
        signal: CheckpointSignal,
        state: _RunState,
        reason: str,
    ) -> EvaluationReport:
        plan = state.plan
        requirements = plan.requirements if plan else ()
        constraints = plan.constraints if plan else ()

        def unknown(item: EvaluationCriterion) -> ItemEvaluation:
            return ItemEvaluation(
                item.criterion_id,
                item.method,
                EvaluationStatus.UNKNOWN,
                f"evaluation interrupted: {reason}",
                (),
                {},
                item.evidence_keys,
            )

        previous = state.previous
        return EvaluationReport(
            signal.run_id,
            checkpoint_id,
            plan,
            tuple(unknown(item) for item in requirements),
            tuple(unknown(item) for item in constraints),
            {},
            {
                key: f"evaluation interrupted: {reason}"
                for item in (*requirements, *constraints)
                for key in item.evidence_keys
            },
            tuple(previous.evidence_ref(key) for key in previous.evidence)
            if previous
            else (),
            (),
            False,
            self._cost(state, signal),
            judge_attempts=self._judge.attempts(signal.run_id)[
                state.work.judge_before.requests :
            ]
            if self._judge
            else (),
        )

    @staticmethod
    def _validate_result(result: CheckResult, item: EvaluationCriterion) -> None:
        if not isinstance(result, CheckResult):
            raise EvaluationProtocolError("verifier must return CheckResult")
        if set(result.evidence_keys) - set(item.evidence_keys):
            raise EvaluationProtocolError("verifier cited undeclared evidence")
        if result.status is not EvaluationStatus.UNKNOWN and not result.evidence_keys:
            raise EvaluationProtocolError(
                "known or conflicting verdicts require evidence"
            )
        if result.status.verdict is not None and result.missing_evidence:
            raise EvaluationProtocolError(
                "known verdict cannot also require missing evidence"
            )

    def close_run(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state is not None and state.lock.locked():
            raise RuntimeError("cannot close an evaluation while capture is active")
        self._runs.pop(run_id, None)
        errors: list[Exception] = []
        for source in {
            id(source): source for source in self._sources.values()
        }.values():
            try:
                source.close_run(run_id)
            except Exception as exc:
                errors.append(exc)
        if self._judge is not None:
            try:
                self._judge.close_run(run_id)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("evidence source cleanup failed", errors)
