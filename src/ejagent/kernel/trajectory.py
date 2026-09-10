"""Stable Runtime seam for optional online trajectory observation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.evaluation import (
    CompletionCandidate,
    EvaluationPlan,
    ToolObservation,
)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must not be negative")
    return value


def _optional_non_negative_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _non_negative_integer(value, label)


class CheckpointTrigger(StrEnum):
    """Runtime boundary at which fresh environment evaluation is requested."""

    BASELINE = "baseline"
    TOOL_BATCH_COMPLETED = "tool_batch_completed"
    VERIFICATION_COMPLETED = "verification_completed"
    EXTERNAL_CHANGE = "external_change"
    COMPLETION_PROPOSED = "completion_proposed"


@dataclass(frozen=True, slots=True)
class TrajectoryCost:
    """Cumulative measurable resource cost at one Checkpoint."""

    actor_actions: int = 0
    model_requests: int = 0
    total_tokens: int | None = None
    elapsed_ms: int | None = None

    def __post_init__(self) -> None:
        _non_negative_integer(self.actor_actions, "cost actor_actions")
        _non_negative_integer(self.model_requests, "cost model_requests")
        _optional_non_negative_integer(self.total_tokens, "cost total_tokens")
        _optional_non_negative_integer(self.elapsed_ms, "cost elapsed_ms")


@dataclass(frozen=True, slots=True)
class CausalAction:
    """One Action attributed to the State transition being checkpointed."""

    action_id: str
    signature: str

    def __post_init__(self) -> None:
        _required_text(self.action_id, "causal action_id")
        _required_text(self.signature, "causal action signature")


@dataclass(frozen=True, slots=True)
class CheckpointSignal:
    """Runtime-owned identity, cause, and cost for one Checkpoint capture."""

    run_id: str
    trigger: CheckpointTrigger
    turn: int
    cumulative_cost: TrajectoryCost
    causal_actions: tuple[CausalAction, ...] = ()
    causal_batch_id: str | None = None
    causally_complete: bool = True
    unattributed_action_ids: tuple[str, ...] = ()
    causal_exclusion_reason: str | None = None
    evaluation_plan: EvaluationPlan | None = None
    task: str | None = None
    completion_candidate: CompletionCandidate | None = None
    tool_observations: tuple[ToolObservation, ...] = ()
    observations_complete: bool = True
    execution_plan: str | None = None

    def __post_init__(self) -> None:
        if self.execution_plan is not None:
            _required_text(self.execution_plan, "execution_plan")
        if self.evaluation_plan is not None and not isinstance(
            self.evaluation_plan, EvaluationPlan
        ):
            raise TypeError("evaluation_plan must be EvaluationPlan or None")
        if self.task is not None and not isinstance(self.task, str):
            raise TypeError("task must be text or None")
        if self.completion_candidate is not None:
            if not isinstance(self.completion_candidate, CompletionCandidate):
                raise TypeError("completion_candidate must be CompletionCandidate")
            if self.trigger is not CheckpointTrigger.COMPLETION_PROPOSED:
                raise ValueError("completion candidate requires completion trigger")
        observations = tuple(self.tool_observations)
        if len(observations) > 128 or not all(
            isinstance(item, ToolObservation) for item in observations
        ):
            raise ValueError(
                "tool_observations must contain at most 128 ToolObservation values"
            )
        if not isinstance(self.observations_complete, bool):
            raise TypeError("observations_complete must be boolean")
        object.__setattr__(self, "tool_observations", observations)
        _required_text(self.run_id, "signal run_id")
        if not isinstance(self.trigger, CheckpointTrigger):
            raise TypeError("signal trigger must be a CheckpointTrigger")
        _non_negative_integer(self.turn, "signal turn")
        if not isinstance(self.cumulative_cost, TrajectoryCost):
            raise TypeError("signal cumulative_cost must be a TrajectoryCost")
        actions = tuple(self.causal_actions)
        if not all(isinstance(item, CausalAction) for item in actions):
            raise TypeError("causal_actions must contain CausalAction values")
        action_ids = [item.action_id for item in actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("causal_actions must have unique action_id values")
        object.__setattr__(self, "causal_actions", actions)
        if self.trigger is CheckpointTrigger.BASELINE:
            if self.turn != 0:
                raise ValueError("baseline signal must use turn zero")
            if actions or self.cumulative_cost.actor_actions:
                raise ValueError("baseline signal cannot contain actor Actions")
        elif self.trigger is CheckpointTrigger.TOOL_BATCH_COMPLETED:
            if not actions:
                raise ValueError("tool batch signal must contain causal Actions")
            if self.causal_batch_id is None:
                raise ValueError("tool batch signal must identify its causal batch")
        elif actions:
            raise ValueError("only a tool batch signal may contain causal Actions")
        if self.causal_batch_id is not None:
            _required_text(self.causal_batch_id, "causal_batch_id")
        unattributed = tuple(self.unattributed_action_ids)
        for item in unattributed:
            _required_text(item, "unattributed_action_ids item")
        object.__setattr__(self, "unattributed_action_ids", unattributed)
        if not isinstance(self.causally_complete, bool):
            raise TypeError("causally_complete must be a bool")
        if self.causally_complete:
            if unattributed or self.causal_exclusion_reason is not None:
                raise ValueError(
                    "causally complete signal cannot carry causal exclusions"
                )
        else:
            if self.causal_exclusion_reason is None:
                raise ValueError(
                    "causally incomplete signal must have an exclusion reason"
                )
            _required_text(self.causal_exclusion_reason, "causal_exclusion_reason")


class TrajectoryCaptureResult(Protocol):
    """Minimal receipt Runtime records without knowing trajectory internals."""

    @property
    def checkpoint_id(self) -> str:
        """Return the host-assigned Checkpoint identity."""

    @property
    def verdict(self) -> str:
        """Return the observation-only assessment verdict."""

    @property
    def completion_allowed(self) -> bool | None:
        """Return completion audit advice, when this was a completion capture."""


class TrajectoryMonitor(Protocol):
    """Optional host implementation invoked at Runtime-safe boundaries."""

    async def capture(
        self,
        signal: CheckpointSignal,
        *,
        cancellation: CancellationToken,
    ) -> TrajectoryCaptureResult:
        """Capture current environment truth for one semantic boundary."""

    def close_run(self, run_id: str) -> object:
        """Release all monitor-owned state for a finished or aborted Run."""
