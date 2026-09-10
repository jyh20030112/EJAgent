"""Adapter that keeps the internal trajectory API out of host evaluator setup."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from ejagent._trajectory.context import (
    TrajectoryContextBuffer,
    TrajectoryContextEvent,
    TrajectoryContextEventKind,
    TrajectoryContextPipeline,
)
from ejagent._trajectory.online import (
    CheckpointEvaluation,
    CheckpointEvaluationRequest,
    OnlineTrajectoryMonitor,
    TrajectoryUpdate,
)
from ejagent._trajectory.shadow import EnvironmentFact, FactValidity
from ejagent.contracts.context import ContextPipeline
from ejagent.contracts.control import CancellationToken
from ejagent.contracts.evaluation import EvaluationPlan
from ejagent.contracts.json import JsonObject
from ejagent.evaluation.engine import GoalEvaluator
from ejagent.evaluation.types import EvaluationReport, fingerprint
from ejagent.kernel.trajectory import CheckpointSignal


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    checkpoint_id: str
    verdict: str
    completion_allowed: bool | None
    report_ref: str
    completion_feedback: str | None = None


class _TrajectoryEvaluator:
    def __init__(self, evaluator: GoalEvaluator) -> None:
        self.evaluator = evaluator

    async def evaluate(
        self, request: CheckpointEvaluationRequest, *, cancellation: CancellationToken
    ) -> CheckpointEvaluation:
        report = await self.evaluator.evaluate(
            request.checkpoint_id, request.signal, cancellation=cancellation
        )
        plan = report.plan
        assert plan is not None
        criteria = (*plan.requirements, *plan.constraints)
        keys = tuple(
            dict.fromkeys(key for item in criteria for key in item.evidence_keys)
        )
        facts: list[EnvironmentFact] = []
        for key in keys:
            evidence = report.evidence.get(key)
            if (
                evidence is None
                and report.fact_capture_complete
                and key not in report.diagnostics
            ):
                continue
            ref = (
                report.evidence_ref(key)
                if evidence
                else f"unavailable:{report.run_id}:{key}"
            )
            facts.append(
                EnvironmentFact(
                    fact_id=f"{ref}:v:{fingerprint(evidence.revision) if evidence else 'unknown'}",
                    subject=key,
                    predicate="observed_value",
                    value=evidence.value if evidence else None,
                    scope=tuple(
                        item.criterion_id
                        for item in criteria
                        if key in item.evidence_keys
                    ),
                    source=f"evaluation-source:{key}",
                    observed_at=evidence.observed_at if evidence else datetime.now(UTC),
                    checkpoint_id=request.checkpoint_id,
                    evidence_ref=ref,
                    freshness="current-capture",
                    authority="host-evidence-source",
                    validity=FactValidity.CURRENT if evidence else FactValidity.UNKNOWN,
                    validity_reason=None
                    if evidence
                    else report.diagnostics.get(key, "scheduled for completion audit"),
                )
            )
        if request.previous_checkpoint is not None:
            for fact in request.previous_checkpoint.current_facts:
                if fact.evidence_ref in report.invalidated_refs:
                    facts.append(
                        replace(
                            fact,
                            checkpoint_id=request.checkpoint_id,
                            validity=FactValidity.INVALIDATED,
                            invalidated_at_checkpoint=request.checkpoint_id,
                            validity_reason="source revision changed or evidence became unavailable",
                        )
                    )
        values: JsonObject = {key: item.value for key, item in report.evidence.items()}
        requirements = {
            item.criterion_id: item.status.verdict for item in report.requirements
        }
        constraints = {
            item.criterion_id: item.status.verdict for item in report.constraints
        }
        return CheckpointEvaluation(
            projection_version=f"goal-evaluator.v1:{plan.version}",
            state_fingerprint=fingerprint(
                {
                    "plan": plan.version,
                    "goal": plan.goal,
                    "facts": values,
                    "requirements": requirements,
                    "constraints": constraints,
                }
            ),
            environment_facts=values,
            requirements=requirements,
            constraints=constraints,
            new_evidence=report.new_evidence,
            facts=tuple(facts),
            fact_capture_complete=report.fact_capture_complete,
        )


class EvaluationMonitor:
    """Supply as Harness trajectory; context_pipeline() exposes evaluator feedback."""

    def __init__(
        self,
        evaluator: GoalEvaluator,
        *,
        update_sink: Callable[[TrajectoryUpdate], None] | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._update_sink = update_sink
        self._buffer = TrajectoryContextBuffer()
        self._monitor = OnlineTrajectoryMonitor(
            _TrajectoryEvaluator(evaluator), update_sink=self._publish
        )
        self._skipped: dict[str, int] = {}

    def validate_plan(self, plan: EvaluationPlan) -> None:
        self._evaluator.validate_plan(plan)

    @property
    def resources(self) -> tuple[object, ...]:
        return self._evaluator.resources

    def context_pipeline(
        self, *, base: ContextPipeline | None = None
    ) -> TrajectoryContextPipeline:
        return TrajectoryContextPipeline(source=self._buffer, base=base)

    async def capture(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> EvaluationReceipt:
        if signal.evaluation_plan is None:
            index = self._skipped.get(signal.run_id, 0)
            report = await self._evaluator.evaluate(
                f"{signal.run_id}:cp{index}", signal, cancellation=cancellation
            )
            self._skipped[signal.run_id] = index + 1
            return EvaluationReceipt(
                report.checkpoint_id, "not_evaluated", None, report.report_ref
            )
        update = await self._monitor.capture(signal, cancellation=cancellation)
        latest = self._evaluator.latest_report(signal.run_id)
        assert latest is not None
        return EvaluationReceipt(
            update.checkpoint_id,
            update.verdict,
            update.completion_allowed,
            latest.report_ref,
            self._completion_feedback(latest)
            if update.completion_allowed is False
            else None,
        )

    @staticmethod
    def _completion_feedback(report: EvaluationReport) -> str:
        unmet = [
            item
            for item in (*report.requirements, *report.constraints)
            if item.status.verdict is not True
        ]
        selected: list[dict[str, object]] = []

        def encode() -> str:
            return json.dumps(
                {
                    "instruction": "Completion has not been accepted. Address unmet items or gather missing evidence within this Run. Consult the evaluation plan and full report for any omitted items.",
                    "unmet_items": selected,
                    "omitted_items": len(unmet) - len(selected),
                },
                ensure_ascii=False,
            )

        for item in unmet:
            selected.append(
                {
                    "id": item.criterion_id,
                    "status": item.status.value,
                    "reason": item.rationale[:512],
                    "missing_evidence": [
                        value[:128] for value in item.missing_evidence[:8]
                    ],
                }
            )
            if len(encode()) > 16_384:
                selected.pop()
                break
        return encode()

    def _publish(self, update: TrajectoryUpdate) -> None:
        report = self._evaluator.latest_report(update.signal.run_id)
        assert report is not None and report.plan is not None
        frame = update.to_context_frame(
            goal=report.plan.goal, next_turn=update.signal.turn + 1
        )
        frame = replace(
            frame,
            current_plan=update.signal.execution_plan,
            event=replace(
                frame.event,
                invalidated_fact_ids=tuple(
                    item.fact_id for item in update.checkpoint.invalidated_facts
                ),
            ),
        )
        if not report.fact_capture_complete:
            missing = tuple(
                dict.fromkeys(
                    (
                        *report.diagnostics,
                        *(
                            key
                            for item in (*report.requirements, *report.constraints)
                            for key in item.missing_evidence
                        ),
                    )
                )
            )
            frame = replace(
                frame,
                event=TrajectoryContextEvent(
                    event_id=f"{report.checkpoint_id}:unavailable",
                    kind=TrajectoryContextEventKind.EVALUATION_UNAVAILABLE,
                    affected_items=tuple(
                        item.criterion_id
                        for item in (*report.requirements, *report.constraints)
                        if item.status.verdict is None
                    ),
                    missing_evidence=missing,
                    invalidated_fact_ids=tuple(
                        item.fact_id for item in update.checkpoint.invalidated_facts
                    ),
                ),
            )
        # Only the next decision frame is transiently retained for this Run.
        self._buffer.close_run(update.signal.run_id)
        self._buffer.publish(frame)
        if self._update_sink is not None:
            self._update_sink(replace(update, context_event=frame.event))

    def close_run(self, run_id: str) -> None:
        try:
            self._monitor.close_run(run_id)
        finally:
            self._buffer.close_run(run_id)
            self._skipped.pop(run_id, None)
            self._evaluator.close_run(run_id)
