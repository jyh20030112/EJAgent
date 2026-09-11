from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ejagent._trajectory.shadow import (
    EnvironmentFact,
    FactValidity,
    ProgressSnapshot,
    TrajectoryCheckpoint,
)
from ejagent.context.pipeline import IdentityContextPipeline
from ejagent.contracts.context import (
    ContextPipeline,
    ContextProtocolError,
    ContextRequest,
    ContextView,
)
from ejagent.contracts.control import CancellationToken
from ejagent.contracts.json import JsonValue, thaw_json_value
from ejagent.contracts.lifecycle import ManagedResource
from ejagent.contracts.messages import TransientInstruction


class TrajectoryContextEventKind(StrEnum):
    """Decision-boundary events eligible for model Context projection."""

    FACTS_UPDATED = "facts_updated"
    PROGRESS_EVALUATED = "progress_evaluated"
    CYCLE_SUSPECTED = "cycle_suspected"
    CYCLE_CONFIRMED = "cycle_confirmed"
    CONSTRAINT_VIOLATED = "constraint_violated"
    EXTERNAL_STATE_CHANGED = "external_state_changed"
    COMPLETION_AUDIT_FAILED = "completion_audit_failed"
    EVALUATION_UNAVAILABLE = "evaluation_unavailable"


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _text_tuple(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    result = tuple(values)
    for item in result:
        _required_text(item, f"{label} item")
    return result


@dataclass(frozen=True, slots=True)
class TrajectoryContextEvent:
    """Evaluator-owned reason to reconsider the next model decision."""

    event_id: str
    kind: TrajectoryContextEventKind
    evidence_refs: tuple[str, ...] = ()
    affected_items: tuple[str, ...] = ()
    causal_actions: tuple[str, ...] = ()
    invalidated_fact_ids: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.event_id, "event_id")
        if not isinstance(self.kind, TrajectoryContextEventKind):
            raise TypeError("event kind must be a TrajectoryContextEventKind")
        for name in (
            "evidence_refs",
            "affected_items",
            "causal_actions",
            "invalidated_fact_ids",
            "missing_evidence",
        ):
            object.__setattr__(self, name, _text_tuple(getattr(self, name), name))
        if (
            self.kind is TrajectoryContextEventKind.CYCLE_CONFIRMED
            and not self.causal_actions
        ):
            raise ValueError("CycleConfirmed requires causal_actions")
        if (
            self.kind is TrajectoryContextEventKind.CONSTRAINT_VIOLATED
            and not self.affected_items
        ):
            raise ValueError("ConstraintViolated requires affected_items")
        if (
            self.kind is TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED
            and not self.invalidated_fact_ids
        ):
            raise ValueError("ExternalStateChanged requires invalidated_fact_ids")
        if self.kind is TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED and not (
            self.affected_items or self.missing_evidence
        ):
            raise ValueError(
                "CompletionAuditFailed requires unmet items or missing Evidence"
            )
        if self.kind is TrajectoryContextEventKind.EVALUATION_UNAVAILABLE and not (
            self.affected_items or self.missing_evidence
        ):
            raise ValueError(
                "EvaluationUnavailable requires unknown items or missing evidence"
            )


@dataclass(frozen=True, slots=True)
class TrajectoryContextFrame:
    """Current State and Assessment supplied at one model decision boundary."""

    run_id: str
    turn: int
    goal: str
    checkpoint: TrajectoryCheckpoint
    progress: ProgressSnapshot
    event: TrajectoryContextEvent
    current_plan: str | None = None
    refuted_hypotheses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.run_id, "frame run_id")
        if isinstance(self.turn, bool) or not isinstance(self.turn, int):
            raise TypeError("frame turn must be an integer")
        if self.turn <= 0:
            raise ValueError("frame turn must be greater than zero")
        _required_text(self.goal, "frame goal")
        if not isinstance(self.checkpoint, TrajectoryCheckpoint):
            raise TypeError("frame checkpoint must be a TrajectoryCheckpoint")
        if not isinstance(self.progress, ProgressSnapshot):
            raise TypeError("frame progress must be a ProgressSnapshot")
        if self.progress.checkpoint_id != self.checkpoint.checkpoint_id:
            raise ValueError(
                "frame Progress and Checkpoint must describe the same State"
            )
        if not isinstance(self.event, TrajectoryContextEvent):
            raise TypeError("frame event must be a TrajectoryContextEvent")
        if self.current_plan is not None:
            _required_text(self.current_plan, "frame current_plan")
        object.__setattr__(
            self,
            "refuted_hypotheses",
            _text_tuple(self.refuted_hypotheses, "refuted_hypotheses"),
        )


@dataclass(frozen=True, slots=True)
class ProjectedTrajectoryContext:
    """Disposable instruction plus inspectable projection metadata."""

    instruction: TransientInstruction
    event: TrajectoryContextEventKind | None
    checkpoint_id: str | None


class TrajectoryContextProjector:
    """Project checkpoint state independently of optional intervention feedback."""

    SCHEMA = "ejagent.trajectory-context.v2"
    _ALERT_EVENTS = frozenset(
        {
            TrajectoryContextEventKind.CYCLE_CONFIRMED,
            TrajectoryContextEventKind.CONSTRAINT_VIOLATED,
            TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED,
            TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED,
            TrajectoryContextEventKind.EVALUATION_UNAVAILABLE,
        }
    )

    def project(self, frame: TrajectoryContextFrame) -> ProjectedTrajectoryContext:
        """Always project observed state; only actionable events add feedback."""
        if not isinstance(frame, TrajectoryContextFrame):
            raise TypeError("frame must be a TrajectoryContextFrame")
        if frame.event.kind is TrajectoryContextEventKind.EVALUATION_UNAVAILABLE:
            return self._project_unavailable(frame)
        self._validate_facts(frame.checkpoint)
        self._validate_event(frame)
        event = frame.event.kind if frame.event.kind in self._ALERT_EVENTS else None
        payload = {
            "schema": self.SCHEMA,
            "run_id": frame.run_id,
            "for_turn": frame.turn,
            "state_status": "available",
            "goal_anchor": frame.goal,
            "checkpoint": frame.checkpoint.checkpoint_id,
            "current_facts": [
                self._current_fact_payload(item)
                for item in frame.checkpoint.current_facts
            ],
            "invalidated_facts": [
                self._invalidated_fact_payload(item)
                for item in frame.checkpoint.invalidated_facts
            ],
            "requirements": dict(frame.checkpoint.requirements),
            "constraints": dict(frame.checkpoint.constraints),
            "revisable_plan": frame.current_plan,
            "refuted_hypotheses": list(frame.refuted_hypotheses),
            "progress": self._progress_payload(frame.progress),
            "missing_evidence": list(frame.event.missing_evidence),
            "feedback": self._feedback(frame) if event is not None else None,
        }
        return self._render(payload, event, frame.checkpoint.checkpoint_id)

    def _feedback(self, frame: TrajectoryContextFrame) -> dict[str, object]:
        return {
            "event": frame.event.kind.value,
            "event_id": frame.event.event_id,
            "event_evidence_refs": list(frame.event.evidence_refs),
            "affected_items": list(frame.event.affected_items),
            "recent_causal_actions": list(frame.event.causal_actions),
            "instruction": self._instruction(frame.event.kind),
        }

    @staticmethod
    def _render(
        payload: dict[str, object],
        event: TrajectoryContextEventKind | None,
        checkpoint_id: str | None,
    ) -> ProjectedTrajectoryContext:
        return ProjectedTrajectoryContext(
            TransientInstruction(
                json.dumps(
                    {"trajectory_context": payload},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                f"trajectory:{event.value}"
                if event is not None
                else "trajectory:state",
            ),
            event,
            checkpoint_id,
        )

    def project_missing(self, request: ContextRequest) -> ProjectedTrajectoryContext:
        """Do not replay an older observation as a current decision's assessment."""
        event = TrajectoryContextEventKind.EVALUATION_UNAVAILABLE
        return self._render(
            {
                "schema": self.SCHEMA,
                "run_id": request.run_id,
                "for_turn": request.turn,
                "state_status": "unavailable",
                "checkpoint": None,
                "missing_evidence": [
                    "No checkpoint observation is available for this decision"
                ],
                "feedback": {
                    "event": event.value,
                    "instruction": (
                        "No checkpoint observation is available for this decision. "
                        "Earlier context does not establish current facts or "
                        "completion status; obtain fresh verification before "
                        "claiming completion."
                    ),
                },
            },
            event,
            None,
        )

    def _project_unavailable(
        self, frame: TrajectoryContextFrame
    ) -> ProjectedTrajectoryContext:
        if frame.checkpoint.fact_capture_complete:
            raise ValueError("unavailable feedback requires incomplete evaluation")
        self._validate_event(frame)
        verdicts = {**frame.checkpoint.requirements, **frame.checkpoint.constraints}
        if any(
            name not in verdicts or verdicts[name] is not None
            for name in frame.event.affected_items
        ):
            raise ValueError("unavailable affected items must have unknown verdicts")
        return self._render(
            {
                "schema": self.SCHEMA,
                "run_id": frame.run_id,
                "for_turn": frame.turn,
                "state_status": "unavailable",
                "goal_anchor": frame.goal,
                "checkpoint": frame.checkpoint.checkpoint_id,
                "missing_evidence": list(frame.event.missing_evidence),
                "invalidated_facts": [
                    self._invalidated_fact_payload(item)
                    for item in frame.checkpoint.invalidated_facts
                ],
                "feedback": self._feedback(frame),
            },
            frame.event.kind,
            frame.checkpoint.checkpoint_id,
        )

    @staticmethod
    def _validate_facts(checkpoint: TrajectoryCheckpoint) -> None:
        if not checkpoint.fact_capture_complete:
            raise ValueError("trajectory Context requires complete Fact capture")
        uncertain = tuple(
            item
            for item in checkpoint.facts
            if item.validity in (FactValidity.STALE, FactValidity.UNKNOWN)
        )
        if uncertain:
            fact_ids = ", ".join(item.fact_id for item in uncertain)
            raise ValueError(
                f"trajectory Context cannot present stale or unknown Facts: {fact_ids}"
            )
        if not checkpoint.current_facts:
            raise ValueError("trajectory Context requires at least one current Fact")

    @staticmethod
    def _validate_event(frame: TrajectoryContextFrame) -> None:
        event = frame.event
        checkpoint = frame.checkpoint
        invalidated = {item.fact_id for item in checkpoint.invalidated_facts}
        missing_invalidations = set(event.invalidated_fact_ids) - invalidated
        if missing_invalidations:
            missing = ", ".join(sorted(missing_invalidations))
            raise ValueError(
                f"event references Facts not explicitly invalidated: {missing}"
            )
        if event.kind is TrajectoryContextEventKind.CONSTRAINT_VIOLATED:
            inconsistent = tuple(
                name
                for name in event.affected_items
                if checkpoint.constraints.get(name) is not False
            )
            if inconsistent:
                names = ", ".join(inconsistent)
                raise ValueError(
                    f"ConstraintViolated items must have false verdicts: {names}"
                )
        if event.kind is TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED:
            verdicts = {**checkpoint.requirements, **checkpoint.constraints}
            inconsistent = tuple(
                name
                for name in event.affected_items
                if name not in verdicts or verdicts[name] is True
            )
            if inconsistent:
                names = ", ".join(inconsistent)
                raise ValueError(
                    f"CompletionAuditFailed items must be unmet or unknown: {names}"
                )

    @staticmethod
    def _current_fact_payload(fact: EnvironmentFact) -> dict[str, object]:
        return {
            "fact_id": fact.fact_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "value": thaw_json_value(fact.value),
            "scope": list(fact.scope),
            "source": fact.source,
            "observed_at": fact.observed_at.isoformat(),
            "checkpoint": fact.checkpoint_id,
            "evidence_ref": fact.evidence_ref,
            "freshness": fact.freshness,
            "authority": fact.authority,
        }

    @staticmethod
    def _invalidated_fact_payload(fact: EnvironmentFact) -> dict[str, object]:
        return {
            "fact_id": fact.fact_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "invalidated_at_checkpoint": fact.invalidated_at_checkpoint,
            "reason": fact.validity_reason,
            "evidence_ref": fact.evidence_ref,
        }

    @staticmethod
    def _progress_payload(progress: ProgressSnapshot) -> dict[str, object]:
        return {
            "current_requirement_coverage": progress.current_requirement_coverage,
            "best_requirement_coverage": progress.best_requirement_coverage,
            "requirement_coverage_delta": progress.requirement_coverage_delta,
            "task_progress_delta": progress.task_progress_delta,
            "status": progress.status.value,
            "gained_requirements": list(progress.gained_requirements),
            "regressed_requirements": list(progress.regressed_requirements),
            "violated_constraints": list(progress.violated_constraints),
            "unresolved_constraints": list(progress.unresolved_constraints),
            "newly_violated_constraints": list(progress.newly_violated_constraints),
            "recovered_constraints": list(progress.recovered_constraints),
            "new_evidence": list(progress.new_evidence),
            "actor_actions_since_previous": progress.actor_actions_since_previous,
            "cost_since_previous": {
                "actor_actions": progress.cost_since_previous.actor_actions,
                "model_requests": progress.cost_since_previous.model_requests,
                "total_tokens": progress.cost_since_previous.total_tokens,
                "elapsed_ms": progress.cost_since_previous.elapsed_ms,
            },
        }

    @staticmethod
    def _instruction(kind: TrajectoryContextEventKind) -> str:
        instructions = {
            TrajectoryContextEventKind.EVALUATION_UNAVAILABLE: (
                "Evaluation is incomplete or conflicting. Gather the missing "
                "evidence and recheck affected items. Discard invalidated "
                "conclusions; no current facts or completion score are asserted here."
            ),
            TrajectoryContextEventKind.CYCLE_CONFIRMED: (
                "Replan from the Goal and current Facts. Do not repeat the exhausted "
                "Action path unless new Evidence changes the decision."
            ),
            TrajectoryContextEventKind.CONSTRAINT_VIOLATED: (
                "Recover the violated Constraint before claiming completion or making "
                "dependent progress."
            ),
            TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED: (
                "Discard beliefs based on invalidated Facts and re-evaluate the Plan "
                "against the refreshed State."
            ),
            TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED: (
                "Continue this Run. Do not claim completion; gather the missing Evidence "
                "or satisfy the unmet Goal items first."
            ),
        }
        return instructions[kind]


TrajectoryContextSource = Callable[[ContextRequest], TrajectoryContextFrame | None]


class TrajectoryContextPipeline(ContextPipeline):
    """Opt-in adapter that appends one trajectory projection after base Context."""

    def __init__(
        self,
        *,
        source: TrajectoryContextSource,
        base: ContextPipeline | None = None,
        projector: TrajectoryContextProjector | None = None,
    ) -> None:
        if not callable(source):
            raise TypeError("source must be callable")
        self._source = source
        self._base = base or IdentityContextPipeline()
        self._projector = projector or TrajectoryContextProjector()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        if isinstance(self._base, ManagedResource):
            await self._base.start()
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        try:
            if isinstance(self._base, ManagedResource):
                await self._base.shutdown()
        finally:
            self._started = False

    async def build(
        self,
        request: ContextRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextView:
        if not self._started:
            raise ContextProtocolError("TrajectoryContextPipeline is not started")
        cancellation.raise_if_cancelled()
        view = await cancellation.run(
            self._base.build(request, cancellation=cancellation)
        )
        if not isinstance(view, ContextView):
            raise ContextProtocolError(
                "wrapped ContextPipeline.build() must return ContextView"
            )
        try:
            frame = self._source(request)
            if frame is not None and not isinstance(frame, TrajectoryContextFrame):
                raise TypeError("source must return TrajectoryContextFrame or None")
            if frame is not None and (
                frame.run_id != request.run_id or frame.turn != request.turn
            ):
                raise ValueError(
                    "trajectory frame must match ContextRequest run and turn"
                )
            projection = (
                self._projector.project_missing(request)
                if frame is None
                else self._projector.project(frame)
            )
        except (TypeError, ValueError) as exc:
            raise ContextProtocolError(str(exc)) from exc
        metadata: dict[str, JsonValue] = {
            **view.metadata,
            "trajectory_context_visible": True,
            "trajectory_feedback_visible": projection.event is not None,
        }
        if frame is not None:
            metadata["trajectory_event"] = frame.event.kind.value
            metadata["trajectory_checkpoint"] = frame.checkpoint.checkpoint_id
        return ContextView(
            run_id=view.run_id,
            source_revision=view.source_revision,
            turn=view.turn,
            messages=(*view.messages, projection.instruction),
            metadata=metadata,
        )


class TrajectoryContextBuffer:
    """Mutable host adapter exposing updates at exact decision boundaries."""

    def __init__(self) -> None:
        self._frames: dict[tuple[str, int], TrajectoryContextFrame] = {}

    def publish(self, frame: TrajectoryContextFrame) -> None:
        if not isinstance(frame, TrajectoryContextFrame):
            raise TypeError("frame must be a TrajectoryContextFrame")
        self._frames[(frame.run_id, frame.turn)] = frame

    def __call__(self, request: ContextRequest) -> TrajectoryContextFrame | None:
        if not isinstance(request, ContextRequest):
            raise TypeError("request must be a ContextRequest")
        return self._frames.get((request.run_id, request.turn))

    def close_run(self, run_id: str) -> tuple[TrajectoryContextFrame, ...]:
        _required_text(run_id, "run_id")
        keys = tuple(key for key in self._frames if key[0] == run_id)
        frames = tuple(self._frames.pop(key) for key in sorted(keys))
        return frames


def immutable_frames(
    frames: Mapping[tuple[str, int], TrajectoryContextFrame],
) -> TrajectoryContextSource:
    """Build a deterministic in-memory source for experiments and hosts."""

    frozen = MappingProxyType(dict(frames))
    return lambda request: frozen.get((request.run_id, request.turn))
