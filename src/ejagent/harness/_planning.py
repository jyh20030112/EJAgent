"""Run-local execution-plan state shared by tool, context, and checkpoint adapters."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime

from ejagent.context.pipeline import IdentityContextPipeline
from ejagent.contracts.context import ContextPipeline, ContextRequest, ContextView
from ejagent.contracts.control import CancellationToken
from ejagent.contracts.json import JsonObject, thaw_json_value
from ejagent.contracts.messages import ToolCall, TransientInstruction
from ejagent.contracts.planning import ExecutionPlan, TaskDefinition
from ejagent.contracts.runs import AuditRecord, RunOutcome
from ejagent.contracts.tools import ToolDefinition, ToolExecutionResult, ToolExecutor
from ejagent.kernel.trajectory import (
    CheckpointSignal,
    TrajectoryCaptureResult,
    TrajectoryMonitor,
)
from ejagent.planning.model import object_fields, parse_steps

_UPDATE_PLAN = ToolDefinition(
    "update_plan",
    "Revise execution steps after checkpoint feedback. Supply the current version and latest checkpoint. "
    "Step completion is an actor claim, not acceptance evidence. Cannot change task, goal, or acceptance criteria.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ("expected_version", "based_on_checkpoint", "reason", "steps"),
        "properties": {
            "expected_version": {"type": "integer", "minimum": 1},
            "based_on_checkpoint": {"type": "string"},
            "reason": {"type": "string", "minLength": 1, "maxLength": 4096},
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 64,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ("id", "description", "requirement_ids", "status"),
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "requirement_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "status": {
                            "type": "string",
                            "enum": ("pending", "in_progress", "completed", "blocked"),
                        },
                    },
                },
            },
        },
    },
)


class _PlanningSession:
    def __init__(
        self,
        *,
        run_id: str,
        definition: TaskDefinition,
        tools: ToolExecutor,
        context: ContextPipeline | None,
        trajectory: TrajectoryMonitor,
        clock: Callable[[], datetime],
    ) -> None:
        if any(t.name == _UPDATE_PLAN.name for t in tools.definitions):
            raise ValueError("update_plan is reserved when task planning is enabled")
        self.run_id = run_id
        self.definition = definition
        self.plan = definition.execution_plan
        self.tools = tools
        self.context = context or IdentityContextPipeline()
        self.trajectory = trajectory
        self.clock = clock
        self.checkpoint: str | None = None
        self.records: list[AuditRecord] = []
        self.record("task_planned", definition.to_dict())

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return (*self.tools.definitions, _UPDATE_PLAN)

    def record(self, kind: str, payload: JsonObject) -> None:
        self.records.append(
            AuditRecord(self.run_id, len(self.records) + 1, kind, self.clock(), payload)
        )

    async def execute(
        self, call: ToolCall, *, cancellation: CancellationToken
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        if call.name != _UPDATE_PLAN.name:
            return await self.tools.execute(call, cancellation=cancellation)
        try:
            if len(json.dumps(thaw_json_value(call.arguments)).encode()) > 65_536:
                raise ValueError("plan update exceeds size bound")
            value = object_fields(
                call.arguments,
                {"expected_version", "based_on_checkpoint", "reason", "steps"},
            )
            expected = value["expected_version"]
            if type(expected) is not int or expected != self.plan.version:
                raise ValueError(
                    f"stale plan version; current version is {self.plan.version}"
                )
            if (
                self.checkpoint is None
                or value["based_on_checkpoint"] != self.checkpoint
            ):
                raise ValueError(
                    "plan update must cite the latest successful checkpoint"
                )
            plan = ExecutionPlan(
                self.plan.version + 1,
                parse_steps(value["steps"]),
                value["reason"],
                self.checkpoint,
            )
            plan.validate_against(self.definition.evaluation_plan)
            # No await between compare and replace: concurrent proposals use CAS.
            self.plan = plan
            self.record(
                "execution_plan_updated", {"call_id": call.id, **plan.to_dict()}
            )
            return ToolExecutionResult(plan.to_dict())
        except (TypeError, ValueError, KeyError) as exc:
            self.record(
                "execution_plan_rejected",
                {
                    "call_id": call.id,
                    "reason": str(exc),
                    "current_version": self.plan.version,
                },
            )
            return ToolExecutionResult(
                {
                    "current_version": self.plan.version,
                    "latest_checkpoint": self.checkpoint,
                },
                error=str(exc),
            )

    async def build(
        self, request: ContextRequest, *, cancellation: CancellationToken
    ) -> ContextView:
        view = await self.context.build(request, cancellation=cancellation)
        task = dict(self.definition.to_dict())
        task["execution_plan"] = self.plan.to_dict()
        task["latest_checkpoint"] = self.checkpoint
        task["instruction"] = (
            "Use checkpoint evidence to reconsider execution steps before acting. "
            "Call update_plan when steps or their status change, with the current version and latest checkpoint. "
            "This is the authoritative execution plan; older plans in conversation are historical. "
            "Completed steps are your claims, not verified requirements. "
            "Task goals and acceptance conditions are fixed for this Run. "
            "If they need to change, report the mismatch rather than weaken them."
        )
        return replace(
            view,
            messages=(
                *view.messages,
                TransientInstruction(
                    json.dumps(
                        {"task_context": thaw_json_value(task)}, ensure_ascii=False
                    ),
                    "planning",
                ),
            ),
            metadata={**view.metadata, "execution_plan_version": self.plan.version},
        )

    async def capture(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> TrajectoryCaptureResult:
        self.checkpoint = None
        receipt = await self.trajectory.capture(
            replace(
                signal,
                execution_plan=json.dumps(
                    thaw_json_value(self.plan.to_dict()), ensure_ascii=False
                ),
            ),
            cancellation=cancellation,
        )
        if not isinstance(receipt.checkpoint_id, str) or not receipt.checkpoint_id:
            raise ValueError("monitor did not return a checkpoint ID")
        self.checkpoint = receipt.checkpoint_id
        return receipt

    def close_run(self, run_id: str) -> None:
        self.checkpoint = None
        self.trajectory.close_run(run_id)


def append_planning_audit(
    outcome: RunOutcome, records: Sequence[AuditRecord]
) -> RunOutcome:
    if not records:
        return outcome
    # Timestamp order preserves when preparation/updates happened. Stable sorting
    # keeps per-source order even when a host clock deliberately returns equal times.
    ordered = sorted((*records, *outcome.audit_records), key=lambda r: r.occurred_at)
    return replace(
        outcome,
        audit_records=tuple(replace(r, sequence=i) for i, r in enumerate(ordered, 1)),
    )
