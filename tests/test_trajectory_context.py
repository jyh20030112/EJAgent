from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime

from ejagent._trajectory import (
    EnvironmentFact,
    FactValidity,
    ProgressSnapshot,
    ProgressStatus,
    TrajectoryCheckpoint,
    TrajectoryContextBuffer,
    TrajectoryContextEvent,
    TrajectoryContextEventKind,
    TrajectoryContextFrame,
    TrajectoryContextPipeline,
    TrajectoryCost,
)
from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    CancellationToken,
    ContextProtocolError,
    ContextRequest,
    ContextView,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    RunIntent,
    RunLimits,
    RunSpec,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutor,
    TransientInstruction,
    UserMessage,
)
from ejagent.contracts.json import JsonValue
from ejagent.kernel import RuntimeKernel

NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)


def _fact(
    fact_id: str,
    value: JsonValue,
    *,
    validity: FactValidity = FactValidity.CURRENT,
) -> EnvironmentFact:
    return EnvironmentFact(
        fact_id=fact_id,
        subject="deployment/router",
        predicate="active_color",
        value=value,
        scope=("R-route",),
        source="deployment-api",
        observed_at=NOW,
        checkpoint_id="cp2" if validity is FactValidity.CURRENT else "cp1",
        evidence_ref=f"deployment://evidence/{fact_id}",
        freshness="valid until the next deployment generation",
        authority="active routing color only",
        validity=validity,
        invalidated_at_checkpoint=(
            "cp2" if validity is FactValidity.INVALIDATED else None
        ),
        validity_reason=(
            "deployment generation changed"
            if validity is FactValidity.INVALIDATED
            else None
        ),
    )


def _checkpoint(*, complete: bool = True) -> TrajectoryCheckpoint:
    return TrajectoryCheckpoint(
        checkpoint_id="cp2",
        projection_version="deployment-v1",
        state_fingerprint="controller-only-fingerprint",
        environment_facts={"active_color": "blue"},
        requirements={"R-route": False, "R-health": True},
        constraints={"C-availability": True},
        new_evidence=("health probe remained green",),
        actor_action_count=4,
        causal_action_signatures=("set-route:blue",),
        facts=(
            _fact("route-blue", "blue"),
            _fact("route-green-old", "green", validity=FactValidity.INVALIDATED),
        ),
        fact_capture_complete=complete,
    )


def _progress() -> ProgressSnapshot:
    return ProgressSnapshot(
        checkpoint_id="cp2",
        requirements={"R-route": False, "R-health": True},
        constraints={"C-availability": True},
        current_requirement_coverage=0.5,
        best_requirement_coverage=0.5,
        requirement_coverage_delta=0.0,
        task_progress_delta=0.0,
        status=ProgressStatus.REGRESSED,
        gained_requirements=(),
        regressed_requirements=("R-route",),
        violated_constraints=(),
        unresolved_constraints=(),
        newly_violated_constraints=(),
        recovered_constraints=(),
        new_evidence=("health probe remained green",),
        actor_actions_since_previous=1,
        cost_since_previous=TrajectoryCost(actor_actions=1),
    )


def _frame(
    kind: TrajectoryContextEventKind,
    *,
    turn: int = 2,
    complete: bool = True,
) -> TrajectoryContextFrame:
    event_arguments: dict[str, tuple[str, ...]] = {}
    checkpoint = _checkpoint(complete=complete)
    progress = _progress()
    if kind is TrajectoryContextEventKind.CYCLE_CONFIRMED:
        event_arguments = {
            "causal_actions": ("set-route:green", "set-route:blue"),
            "evidence_refs": ("checkpoint://cp0-cp4",),
        }
    elif kind is TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED:
        event_arguments = {
            "invalidated_fact_ids": ("route-green-old",),
            "evidence_refs": ("deployment://generation/2",),
        }
    elif kind is TrajectoryContextEventKind.CONSTRAINT_VIOLATED:
        event_arguments = {
            "affected_items": ("C-availability",),
            "evidence_refs": ("probe://availability/2",),
        }
        checkpoint = replace(checkpoint, constraints={"C-availability": False})
        progress = replace(progress, constraints={"C-availability": False})
    elif kind is TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED:
        event_arguments = {
            "affected_items": ("R-route",),
            "missing_evidence": ("production route verification",),
        }
    return TrajectoryContextFrame(
        run_id="context-run",
        turn=turn,
        goal="Route production traffic to blue while preserving availability.",
        checkpoint=checkpoint,
        progress=progress,
        event=TrajectoryContextEvent(
            event_id=f"event-{turn}",
            kind=kind,
            **event_arguments,
        ),
        current_plan="Toggle the active route and verify health.",
        refuted_hypotheses=("Changing standby color updates active traffic",),
    )


def _request(turn: int = 2) -> ContextRequest:
    return ContextRequest(
        run_id="context-run",
        source_revision=0,
        turn=turn,
        committed_messages=(SystemMessage("stable"),),
        pending_messages=(UserMessage("deploy"),),
    )


class TrajectoryContextPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def _build(
        self,
        frame: TrajectoryContextFrame | None,
    ) -> ContextView:
        pipeline = TrajectoryContextPipeline(source=lambda request: frame)
        await pipeline.start()
        try:
            return await pipeline.build(
                _request(),
                cancellation=CancellationSource().token,
            )
        finally:
            await pipeline.shutdown()

    async def test_cycle_suspicion_keeps_state_but_remains_controller_only(
        self,
    ) -> None:
        view = await self._build(_frame(TrajectoryContextEventKind.CYCLE_SUSPECTED))

        self.assertTrue(view.metadata["trajectory_context_visible"])
        self.assertFalse(view.metadata["trajectory_feedback_visible"])
        self.assertEqual(view.metadata["trajectory_event"], "cycle_suspected")
        instruction = view.messages[-1]
        self.assertIsInstance(instruction, TransientInstruction)
        self.assertEqual(instruction.source, "trajectory:state")
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(payload["schema"], "ejagent.trajectory-context.v2")
        self.assertEqual(payload["state_status"], "available")
        self.assertIsNone(payload["feedback"])
        self.assertEqual(payload["checkpoint"], "cp2")
        self.assertEqual(payload["for_turn"], 2)
        self.assertEqual(payload["requirements"], {"R-route": False, "R-health": True})
        self.assertEqual(payload["progress"]["current_requirement_coverage"], 0.5)
        self.assertEqual(payload["current_facts"][0]["observed_at"], NOW.isoformat())
        self.assertEqual(payload["invalidated_facts"][0]["fact_id"], "route-green-old")
        self.assertNotIn("cycle_suspected", instruction.content)
        self.assertNotIn("Replan", instruction.content)

    async def test_normal_observations_provide_state_without_intervention(self) -> None:
        for kind in (
            TrajectoryContextEventKind.FACTS_UPDATED,
            TrajectoryContextEventKind.PROGRESS_EVALUATED,
        ):
            with self.subTest(kind=kind):
                view = await self._build(_frame(kind))
                state = json.loads(view.messages[-1].content)["trajectory_context"]
                self.assertEqual(state["current_facts"][0]["value"], "blue")
                self.assertIsNone(state["feedback"])
                self.assertFalse(view.metadata["trajectory_feedback_visible"])

    async def test_missing_observation_does_not_invent_state_or_success(self) -> None:
        view = await self._build(None)
        state = json.loads(view.messages[-1].content)["trajectory_context"]
        self.assertEqual(state["state_status"], "unavailable")
        self.assertIsNone(state["checkpoint"])
        self.assertTrue(state["missing_evidence"])
        self.assertEqual(state["feedback"]["event"], "evaluation_unavailable")
        for name in ("current_facts", "requirements", "constraints", "progress"):
            self.assertNotIn(name, state)

    async def test_snapshot_keeps_identity_and_cannot_leak_into_other_decisions(
        self,
    ) -> None:
        buffer = TrajectoryContextBuffer()
        frame = _frame(TrajectoryContextEventKind.CYCLE_SUSPECTED)
        buffer.publish(frame)
        pipeline = TrajectoryContextPipeline(source=buffer)
        await pipeline.start()
        try:
            first = await pipeline.build(_request(), cancellation=CancellationToken())
            again = await pipeline.build(_request(), cancellation=CancellationToken())
            self.assertEqual(first, again)
            for request in (_request(3), replace(_request(), run_id="other-run")):
                view = await pipeline.build(request, cancellation=CancellationToken())
                state = json.loads(view.messages[-1].content)["trajectory_context"]
                self.assertEqual(state["state_status"], "unavailable")
                self.assertIsNone(state["checkpoint"])
                self.assertNotIn("current_facts", state)
            buffer.close_run("context-run")
            closed = await pipeline.build(_request(), cancellation=CancellationToken())
            self.assertNotIn(
                "current_facts",
                json.loads(closed.messages[-1].content)["trajectory_context"],
            )
        finally:
            await pipeline.shutdown()

    async def test_confirmed_cycle_projects_current_truth_and_provenance(self) -> None:
        view = await self._build(_frame(TrajectoryContextEventKind.CYCLE_CONFIRMED))

        instruction = view.messages[-1]
        self.assertIsInstance(instruction, TransientInstruction)
        assert isinstance(instruction, TransientInstruction)
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(payload["feedback"]["event"], "cycle_confirmed")
        self.assertEqual(
            payload["goal_anchor"],
            _frame(TrajectoryContextEventKind.CYCLE_CONFIRMED).goal,
        )
        self.assertEqual(payload["current_facts"][0]["value"], "blue")
        self.assertEqual(
            payload["current_facts"][0]["evidence_ref"],
            "deployment://evidence/route-blue",
        )
        self.assertEqual(payload["invalidated_facts"][0]["fact_id"], "route-green-old")
        self.assertNotIn("controller-only-fingerprint", instruction.content)
        self.assertNotIn('"value":"green"', instruction.content)
        self.assertIn("Replan from the Goal", payload["feedback"]["instruction"])

    async def test_nested_fact_values_are_projected_as_json(self) -> None:
        value: JsonValue = {"routes": [{"color": "blue", "healthy": True}]}
        frame = _frame(TrajectoryContextEventKind.FACTS_UPDATED)
        frame = replace(
            frame,
            checkpoint=replace(frame.checkpoint, facts=(_fact("routes", value),)),
        )

        view = await self._build(frame)

        instruction = view.messages[-1]
        assert isinstance(instruction, TransientInstruction)
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(payload["current_facts"][0]["value"], value)

    async def test_external_change_marks_historical_fact_as_invalidated(self) -> None:
        view = await self._build(
            _frame(TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED)
        )

        instruction = view.messages[-1]
        assert isinstance(instruction, TransientInstruction)
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(
            payload["invalidated_facts"],
            [
                {
                    "evidence_ref": "deployment://evidence/route-green-old",
                    "fact_id": "route-green-old",
                    "invalidated_at_checkpoint": "cp2",
                    "predicate": "active_color",
                    "reason": "deployment generation changed",
                    "subject": "deployment/router",
                }
            ],
        )
        self.assertIn("Discard beliefs", payload["feedback"]["instruction"])

    async def test_failed_completion_audit_explicitly_continues_same_run(self) -> None:
        view = await self._build(
            _frame(TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED)
        )

        instruction = view.messages[-1]
        assert isinstance(instruction, TransientInstruction)
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(payload["feedback"]["affected_items"], ["R-route"])
        self.assertEqual(payload["missing_evidence"], ["production route verification"])
        self.assertIn("Continue this Run", payload["feedback"]["instruction"])

    async def test_constraint_violation_projects_a_recovery_boundary(self) -> None:
        view = await self._build(_frame(TrajectoryContextEventKind.CONSTRAINT_VIOLATED))

        instruction = view.messages[-1]
        assert isinstance(instruction, TransientInstruction)
        payload = json.loads(instruction.content)["trajectory_context"]
        self.assertEqual(payload["constraints"], {"C-availability": False})
        self.assertEqual(payload["feedback"]["affected_items"], ["C-availability"])
        self.assertIn(
            "Recover the violated Constraint", payload["feedback"]["instruction"]
        )

    async def test_incomplete_fact_capture_fails_closed(self) -> None:
        with self.assertRaisesRegex(ContextProtocolError, "complete Fact capture"):
            await self._build(
                _frame(TrajectoryContextEventKind.CYCLE_CONFIRMED, complete=False)
            )


class _TwoTurnModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelResponseCompleted(
                AssistantMessage(tool_calls=(ToolCall("inspect-1", "inspect", {}),))
            )
        else:
            yield ModelResponseCompleted(AssistantMessage(content="replanned"))


class _InspectTool(ToolExecutor):
    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                "inspect",
                "Inspect current deployment State.",
                {"type": "object", "additionalProperties": False},
            ),
        )

    async def execute(
        self,
        call: ToolCall,
        *,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        return ToolExecutionResult({"active_color": "blue"})


class TrajectoryContextRuntimeTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_is_visible_only_at_the_next_model_decision(self) -> None:
        model = _TwoTurnModel()
        frame = _frame(TrajectoryContextEventKind.CYCLE_CONFIRMED)
        pipeline = TrajectoryContextPipeline(
            source=lambda request: frame if request.turn == 2 else None
        )
        kernel = RuntimeKernel(model=model, tools=_InspectTool(), context=pipeline)
        await pipeline.start()
        try:
            outcome = await kernel.run(
                RunSpec(
                    run_id="context-run",
                    base_revision=0,
                    intent=RunIntent.TASK,
                    task="deploy",
                    messages=(SystemMessage("stable"), UserMessage("deploy")),
                    limits=RunLimits(max_turns=2),
                    configuration_revision="trajectory-context-test",
                )
            )
        finally:
            await pipeline.shutdown()

        self.assertTrue(outcome.result.succeeded)
        self.assertEqual(len(model.requests), 2)
        missing = next(
            message
            for message in model.requests[0].messages
            if isinstance(message, TransientInstruction)
        )
        first_state = json.loads(missing.content)["trajectory_context"]
        self.assertEqual(first_state["state_status"], "unavailable")
        self.assertIsNone(first_state["checkpoint"])
        self.assertNotIn("cycle_confirmed", missing.content)
        projected = [
            message
            for message in model.requests[1].messages
            if isinstance(message, TransientInstruction)
        ]
        self.assertEqual(len(projected), 1)
        self.assertIn('"event":"cycle_confirmed"', projected[0].content)


if __name__ == "__main__":
    unittest.main()
