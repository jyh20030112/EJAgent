"""Task definitions and revisable execution plans, separate from acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.evaluation import EvaluationCriterion, EvaluationPlan
from ejagent.contracts.json import JsonObject
from ejagent.contracts.messages import ConversationMessage, is_conversation_message
from ejagent.contracts.model import ModelUsage
from ejagent.contracts.tools import ToolDefinition


def bounded_text(value: object, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{label} must be non-empty text of at most {maximum} characters"
        )
    return value


class StepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class PlanStep:
    step_id: str
    description: str
    requirement_ids: tuple[str, ...]
    status: StepStatus = StepStatus.PENDING

    def __post_init__(self) -> None:
        bounded_text(self.step_id, "step_id", 128)
        bounded_text(self.description, "step description")
        if not isinstance(self.status, StepStatus):
            raise TypeError("step status must be StepStatus")
        ids = tuple(self.requirement_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("step requirement_ids must be non-empty and unique")
        for value in ids:
            bounded_text(value, "requirement_id", 128)
        object.__setattr__(self, "requirement_ids", ids)

    def to_dict(self) -> JsonObject:
        return {
            "id": self.step_id,
            "description": self.description,
            "requirement_ids": self.requirement_ids,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    version: int
    steps: tuple[PlanStep, ...]
    reason: str
    based_on_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ValueError("plan version must be a positive integer")
        bounded_text(self.reason, "plan reason")
        steps = tuple(self.steps)
        if not 1 <= len(steps) <= 64 or not all(isinstance(s, PlanStep) for s in steps):
            raise ValueError("plan requires 1-64 PlanStep values")
        if len({s.step_id for s in steps}) != len(steps):
            raise ValueError("step IDs must be unique")
        if sum(s.status is StepStatus.IN_PROGRESS for s in steps) > 1:
            raise ValueError("at most one step may be in progress")
        if self.based_on_checkpoint is not None:
            bounded_text(self.based_on_checkpoint, "checkpoint")
        object.__setattr__(self, "steps", steps)

    def validate_against(self, acceptance: EvaluationPlan) -> None:
        required = {item.criterion_id for item in acceptance.requirements}
        known = required | {item.criterion_id for item in acceptance.constraints}
        covered = {key for step in self.steps for key in step.requirement_ids}
        if not required <= covered or not covered <= known:
            raise ValueError(
                "execution steps must cover all requirements and only reference known criteria"
            )

    def to_dict(self) -> JsonObject:
        return {
            "version": self.version,
            "steps": tuple(s.to_dict() for s in self.steps),
            "reason": self.reason,
            "based_on_checkpoint": self.based_on_checkpoint,
        }


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    task: str
    evaluation_plan: EvaluationPlan
    execution_plan: ExecutionPlan
    unknowns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        bounded_text(self.task, "task", 16_384)
        if not isinstance(self.evaluation_plan, EvaluationPlan):
            raise TypeError("evaluation_plan must be EvaluationPlan")
        if not isinstance(self.execution_plan, ExecutionPlan):
            raise TypeError("execution_plan must be ExecutionPlan")
        self.execution_plan.validate_against(self.evaluation_plan)
        unknowns = tuple(self.unknowns)
        if len(unknowns) > 32:
            raise ValueError("at most 32 unknowns are allowed")
        for value in unknowns:
            bounded_text(value, "unknown")
        object.__setattr__(self, "unknowns", unknowns)

    def to_dict(self) -> JsonObject:
        def criterion(item: EvaluationCriterion) -> JsonObject:
            return {
                "id": item.criterion_id,
                "description": item.description,
                "method": item.method,
                "evidence_keys": item.evidence_keys,
                "semantic": item.semantic,
                "guard_method": item.guard_method,
                "completion_only": item.completion_only,
            }

        plan = self.evaluation_plan
        return {
            "task": self.task,
            "goal": plan.goal,
            "acceptance_version": plan.version,
            "requirements": tuple(criterion(x) for x in plan.requirements),
            "constraints": tuple(criterion(x) for x in plan.constraints),
            "unknowns": self.unknowns,
            "execution_plan": self.execution_plan.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    run_id: str
    query: str
    messages: tuple[ConversationMessage, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()

    def __post_init__(self) -> None:
        bounded_text(self.run_id, "planning run_id")
        bounded_text(self.query, "query", 65_536)
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        if not all(is_conversation_message(m) for m in self.messages):
            raise TypeError("planning history must contain conversation messages")
        if not all(isinstance(t, ToolDefinition) for t in self.tools):
            raise TypeError("planning tools must contain ToolDefinition values")


@dataclass(frozen=True, slots=True)
class PlanningResult:
    definition: TaskDefinition
    requests: int = 0
    usage: ModelUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.definition, TaskDefinition):
            raise TypeError("planning result requires a TaskDefinition")
        if type(self.requests) is not int or self.requests < 0:
            raise ValueError("planning requests must be a non-negative integer")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("planning usage must be ModelUsage or None")


class PlanningError(RuntimeError):
    """Preparation failed; no execution or completion may be inferred."""

    def __init__(
        self, message: str, *, requests: int = 0, usage: ModelUsage | None = None
    ) -> None:
        super().__init__(message)
        self.requests = requests
        self.usage = usage


class TaskPlanner(Protocol):
    async def plan(
        self, request: PlanningRequest, *, cancellation: CancellationToken
    ) -> PlanningResult: ...
