from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from streamlit.testing.v1 import AppTest

from ejagent.contracts import (
    AssistantMessage,
    CancellationToken,
    ControlStatus,
    ModelRequest,
    ModelStreamEvent,
    RunStatus,
    TransientInstruction,
)
from ejagent.evaluation import EvaluationStatus
from ejagent.harness import HarnessStatus
from examples.streamlit_runtime import (
    COMPLETION_DEMO_TASK,
    TRAJECTORY_DEMO_TASK,
    DemoValidationModel,
    RuntimeConfig,
    RuntimeSnapshot,
    StreamlitRuntimeController,
)


class _RecordingDemoModel(DemoValidationModel):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[ModelRequest] = []

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        async for event in super().stream(request, cancellation=cancellation):
            yield event


def _wait_for(
    controller: StreamlitRuntimeController,
    predicate: Callable[[RuntimeSnapshot], bool],
    *,
    timeout: float = 3.0,
) -> RuntimeSnapshot:
    deadline = time.monotonic() + timeout
    snapshot = controller.snapshot()
    while not predicate(snapshot):
        if time.monotonic() >= deadline:
            raise AssertionError(f"runtime condition timed out: {snapshot!r}")
        time.sleep(0.01)
        snapshot = controller.snapshot()
    return snapshot


class StreamlitRuntimeControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store_root = Path(temporary.name)
        self.controllers: list[StreamlitRuntimeController] = []

    def tearDown(self) -> None:
        for controller in reversed(self.controllers):
            controller.close()

    def controller(
        self,
        *,
        delay: float = 0.05,
        trajectory_enabled: bool = True,
        model: DemoValidationModel | None = None,
        max_turns: int = 20,
    ) -> StreamlitRuntimeController:
        demo = model if model is not None else DemoValidationModel()
        controller = StreamlitRuntimeController(
            RuntimeConfig(
                agent_id="streamlit-test",
                store_root=self.store_root,
                probe_delay_seconds=delay,
                trajectory_enabled=trajectory_enabled,
                max_turns=max_turns,
            ),
            model_factory=lambda: demo,
        )
        self.controllers.append(controller)
        return controller

    def test_parallel_run_is_committed_and_restored_from_jsonl(self) -> None:
        controller = self.controller()

        controller.start_run("validate parallel execution")
        snapshot = _wait_for(controller, lambda item: item.revision == 1)

        self.assertEqual(snapshot.status, HarnessStatus.READY)
        self.assertEqual(len(snapshot.audits), 1)
        self.assertEqual(snapshot.audits[0].result.status, RunStatus.COMPLETED)
        self.assertEqual(len(snapshot.probes), 2)
        first, second = snapshot.probes
        assert first.finished_at is not None
        assert second.finished_at is not None
        self.assertLess(
            max(first.started_at, second.started_at),
            min(first.finished_at, second.finished_at),
        )
        committed_messages = snapshot.messages

        controller.close()
        restored = self.controller()
        restored_snapshot = restored.snapshot()

        self.assertEqual(restored_snapshot.revision, 1)
        self.assertEqual(restored_snapshot.messages, committed_messages)
        self.assertEqual(len(restored_snapshot.audits), 1)
        self.assertEqual(restored_snapshot.trajectory_updates, ())
        self.assertEqual(restored_snapshot.trajectory_contexts, ())

    def test_trajectory_feedback_reaches_model_without_entering_conversation(
        self,
    ) -> None:
        model = _RecordingDemoModel()
        controller = self.controller(model=model)
        controller.start_run("validate parallel execution")
        snapshot = _wait_for(controller, lambda item: item.revision == 1)

        self.assertEqual(
            [
                u.assessment.progress[-1].current_requirement_coverage
                for u in snapshot.trajectory_updates
            ],
            [0.0, 1.0, 1.0],
        )
        self.assertTrue(snapshot.trajectory_updates[-1].completion_allowed)
        self.assertEqual([item.turn for item in snapshot.trajectory_contexts], [1, 2])
        for request, delivery in zip(
            model.requests, snapshot.trajectory_contexts, strict=True
        ):
            self.assertIn(delivery.instruction, request.messages)
        self.assertFalse(
            any(
                isinstance(message, TransientInstruction)
                for message in snapshot.messages
            )
        )
        self.assertFalse(
            any(
                isinstance(message, AssistantMessage)
                and "trajectory_context" in (message.content or "")
                for message in snapshot.messages
            )
        )

    def test_cycle_feedback_changes_demo_actions_and_recovers_parallelism(self) -> None:
        model = _RecordingDemoModel()
        controller = self.controller(delay=0.01, model=model)
        controller.start_run(TRAJECTORY_DEMO_TASK)
        snapshot = _wait_for(controller, lambda item: item.revision == 1)

        verdicts = [update.verdict for update in snapshot.trajectory_updates]
        self.assertIn("cycle_suspected", verdicts)
        self.assertIn("non_progress_cycle", verdicts)
        deliveries = snapshot.trajectory_contexts
        self.assertNotIn(
            "trajectory:cycle_suspected",
            [item.instruction.source for item in deliveries],
        )
        confirmed = next(
            item
            for item in deliveries
            if item.instruction.source == "trajectory:cycle_confirmed"
        )
        self.assertIn(
            confirmed.instruction, model.requests[confirmed.turn - 1].messages
        )
        batches = [
            message.tool_calls
            for message in snapshot.messages
            if isinstance(message, AssistantMessage) and message.tool_calls
        ]
        self.assertEqual([len(batch) for batch in batches], [1, 1, 1, 1, 1, 1, 2])
        assert snapshot.last_result is not None
        self.assertEqual(snapshot.last_result.turns, 8)
        self.assertTrue(snapshot.trajectory_updates[-1].completion_allowed)
        self.assertTrue(
            snapshot.trajectory_updates[-1].checkpoint.requirements["probes_overlapped"]
        )

    def test_disabled_feedback_preserves_original_context(self) -> None:
        model = _RecordingDemoModel()
        controller = self.controller(trajectory_enabled=False, model=model)
        controller.start_run("validate parallel execution")
        snapshot = _wait_for(controller, lambda item: item.revision == 1)

        self.assertEqual(snapshot.trajectory_updates, ())
        self.assertEqual(snapshot.trajectory_contexts, ())
        self.assertFalse(
            any(
                isinstance(message, TransientInstruction)
                and message.source.startswith("trajectory:")
                for request in model.requests
                for message in request.messages
            )
        )
        self.assertFalse(
            any(
                record.kind.startswith("trajectory_")
                for record in snapshot.audits[0].records
            )
        )

    def test_recovery_requires_feedback_and_respects_turn_limit(self) -> None:
        controller = self.controller(delay=0.01, trajectory_enabled=False, max_turns=8)
        controller.start_run(TRAJECTORY_DEMO_TASK)
        snapshot = _wait_for(
            controller,
            lambda item: item.status is HarnessStatus.READY and bool(item.audits),
        )
        self.assertEqual(len(snapshot.probes), 8)
        self.assertNotEqual(snapshot.audits[0].result.status, RunStatus.COMPLETED)

    def test_cancel_stops_tools_without_advancing_revision(self) -> None:
        controller = self.controller(delay=0.5)
        controller.start_run("cancel this validation")
        _wait_for(
            controller,
            lambda item: sum(probe.finished_at is None for probe in item.probes) == 2,
        )

        self.assertTrue(controller.cancel("test cancellation"))
        snapshot = _wait_for(
            controller,
            lambda item: item.status is HarnessStatus.READY and len(item.audits) == 1,
        )

        self.assertEqual(snapshot.revision, 0)
        self.assertIsNone(snapshot.last_result)
        self.assertIsNotNone(snapshot.latest_outcome)
        assert snapshot.latest_outcome is not None
        self.assertEqual(snapshot.latest_outcome.result.status, RunStatus.CANCELLED)
        self.assertTrue(all(probe.cancelled for probe in snapshot.probes))
        cancelled_run_id = snapshot.trajectory_updates[0].signal.run_id
        controller.start_run("validate after cancellation")
        recovered = _wait_for(controller, lambda item: item.revision == 1)
        self.assertNotEqual(
            recovered.trajectory_updates[0].signal.run_id, cancelled_run_id
        )
        self.assertEqual(
            recovered.trajectory_updates[0]
            .assessment.progress[0]
            .current_requirement_coverage,
            0.0,
        )
        self.assertTrue(recovered.trajectory_updates[-1].completion_allowed)
        self.assertTrue(
            all(
                item.run_id != cancelled_run_id
                for item in recovered.trajectory_contexts
            )
        )

    def test_steering_and_follow_up_are_admitted_during_active_run(self) -> None:
        controller = self.controller(delay=0.1)
        controller.start_run("initial validation")
        _wait_for(
            controller,
            lambda item: sum(probe.finished_at is None for probe in item.probes) == 2,
        )

        steering = controller.steer("mention the accepted steering")
        follow_up = controller.follow_up("run the queued validation")

        self.assertEqual(steering.status, ControlStatus.ACCEPTED)
        self.assertEqual(follow_up.status, ControlStatus.ACCEPTED)
        snapshot = _wait_for(
            controller,
            lambda item: (
                item.revision == 2
                and item.status is HarnessStatus.READY
                and item.pending_follow_ups == 0
            ),
        )
        self.assertEqual(len(snapshot.audits), 2)
        self.assertEqual(len(snapshot.probes), 4)
        self.assertEqual(
            snapshot.trajectory_updates[0]
            .assessment.progress[0]
            .current_requirement_coverage,
            0.0,
        )
        self.assertEqual(
            {item.run_id for item in snapshot.trajectory_contexts},
            {snapshot.trajectory_updates[0].signal.run_id},
        )
        self.assertTrue(
            any(
                isinstance(message, AssistantMessage)
                and message.content is not None
                and "Applied steering" in message.content
                for message in snapshot.messages
            )
        )

    def test_formal_evaluator_semantic_review_and_enforcement_share_real_feedback(
        self,
    ) -> None:
        controller = StreamlitRuntimeController(
            RuntimeConfig(
                agent_id="formal-evaluation",
                store_root=self.store_root,
                probe_delay_seconds=0.01,
                semantic_review=True,
                completion_enforced=True,
            )
        )
        self.controllers.append(controller)
        controller.start_run(TRAJECTORY_DEMO_TASK)
        snapshot = _wait_for(controller, lambda item: item.revision == 1, timeout=5)
        self.assertEqual(snapshot.last_result.status, RunStatus.COMPLETED)
        report = snapshot.evaluation_reports[-1]
        self.assertEqual(len(report.requirements), 4)
        self.assertTrue(
            all(item.status is EvaluationStatus.PASS for item in report.requirements)
        )
        self.assertEqual(
            sum(item.cost.model_requests for item in snapshot.evaluation_reports), 1
        )
        self.assertEqual(
            sum(
                item.cost.model_input_tokens + item.cost.model_output_tokens
                for item in snapshot.evaluation_reports
            ),
            60,
        )
        self.assertTrue(
            any(
                item.instruction.source == "trajectory:cycle_confirmed"
                for item in snapshot.trajectory_contexts
            )
        )
        self.assertTrue(snapshot.trajectory_updates[-1].completion_allowed)
        logs = list((self.store_root / "evaluations").glob("*.jsonl"))
        self.assertEqual(len(logs), 1)
        stored = [json.loads(line) for line in logs[0].read_text().splitlines()]
        self.assertEqual(stored[-1]["report_ref"], report.report_ref)
        self.assertEqual(stored[-1]["requirements"][-1]["status"], "pass")

    def test_completion_recovery_demo_rejects_then_finishes_same_run(self) -> None:
        controller = StreamlitRuntimeController(
            RuntimeConfig(
                agent_id="completion-demo",
                store_root=self.store_root,
                probe_delay_seconds=0.01,
                semantic_review=True,
                completion_enforced=True,
            )
        )
        self.controllers.append(controller)
        controller.start_run(COMPLETION_DEMO_TASK)
        snapshot = _wait_for(controller, lambda item: item.revision == 1, timeout=5)
        self.assertEqual(snapshot.last_result.turns, 3)
        rejected = [
            record
            for record in snapshot.audits[-1].records
            if record.kind == "completion_rejected"
        ]
        self.assertEqual(len(rejected), 1)
        self.assertNotIn(
            AssistantMessage("Parallel validation is complete."), snapshot.messages
        )
        self.assertTrue(
            any(
                item.instruction.source == "completion_audit"
                for item in snapshot.trajectory_contexts
            )
        )
        self.assertEqual(
            sum(report.cost.model_requests for report in snapshot.evaluation_reports), 1
        )


class StreamlitAppSmokeTests(unittest.TestCase):
    def test_app_semantic_controls_display_formal_reports_and_separate_costs(
        self,
    ) -> None:
        app_path = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as root:
            app = AppTest.from_file(app_path, default_timeout=5).run()
            app.text_input[1].set_value(root).run()
            app.slider[0].set_value(0.25).run()
            next(
                item
                for item in app.checkbox
                if item.label == "Semantic completion review"
            ).check().run()
            next(
                item
                for item in app.checkbox
                if item.label == "Require completion approval"
            ).check().run()
            next(item for item in app.button if item.label == "Start").click().run()
            controller = app.session_state["ejagent_runtime_controller"]
            try:
                next(
                    item
                    for item in app.button
                    if item.label == "Run parallel validation"
                ).click().run()
                _wait_for(controller, lambda item: item.revision == 1, timeout=5)
                app.run()
                self.assertFalse(app.exception)
                metrics = {item.label: item.value for item in app.metric}
                self.assertEqual(metrics["Judge model requests"], "1")
                self.assertEqual(metrics["Judge reported tokens"], "60")
                self.assertEqual(metrics["Actor model requests"], "2")
                self.assertTrue(
                    any(item.value == "Evaluation details" for item in app.subheader)
                )
                self.assertTrue(
                    any(
                        item.label == "Evidence and diagnostics"
                        for item in app.expander
                    )
                )
            finally:
                controller.close()

    def test_app_entrypoint_imports_from_outside_repository(self) -> None:
        app_path = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as working_directory:
            completed = subprocess.run(
                (sys.executable, str(app_path)),
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_app_renders_before_runtime_is_started(self) -> None:
        app_path = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
        app = AppTest.from_file(app_path).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "EJAgent Harness Validation")
        self.assertTrue(any("Start the runtime" in item.value for item in app.info))

    def test_app_starts_and_stops_demo_runtime(self) -> None:
        app_path = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as root:
            app = AppTest.from_file(app_path, default_timeout=5).run()
            app.text_input[1].set_value(root).run()
            app.slider[0].set_value(0.25).run()
            app.button[0].click().run()
            controller = app.session_state["ejagent_runtime_controller"]
            try:
                self.assertFalse(app.exception)
                self.assertTrue(
                    any(
                        metric.label == "Harness" and metric.value == "ready"
                        for metric in app.metric
                    )
                )
                run = next(
                    button
                    for button in app.button
                    if button.label == "Run parallel validation"
                )
                run.click().run()
                self.assertFalse(app.exception)
                _wait_for(controller, lambda item: item.revision == 1, timeout=5)
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(
                    any(
                        metric.label == "Requirement coverage"
                        and metric.value == "100%"
                        for metric in app.metric
                    )
                )
                self.assertTrue(any(item.label == "Trajectory" for item in app.tabs))
                recover = next(
                    button
                    for button in app.button
                    if button.label == "Run trajectory recovery"
                )
                recover.click().run()
                self.assertFalse(app.exception)
                _wait_for(controller, lambda item: item.revision == 2, timeout=5)
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(
                    any(
                        "trajectory:cycle_confirmed" in item.label
                        for item in app.expander
                    )
                )
                stop = next(button for button in app.button if button.label == "Stop")
                stop.click().run()
                self.assertFalse(app.exception)
            finally:
                controller.close()


class DynamicPlanningControllerTests(unittest.TestCase):
    def test_query_bound_plan_replaces_fixed_three_probe_acceptance(self) -> None:
        from ejagent.contracts import ToolCall
        from tests.test_task_planning import CapturedModel, revise

        payload = {
            "task": "Run only probe B",
            "goal": "Probe B completes",
            "criteria": [
                {
                    "id": "b_only",
                    "capability": "probe_b_completed",
                    "description": "Probe B completes in this Run",
                }
            ],
            "steps": [
                {
                    "id": "run_b",
                    "description": "Run probe B",
                    "requirement_ids": ["b_only"],
                    "status": "pending",
                }
            ],
            "unknowns": [],
            "unsupported": [],
        }
        actor = CapturedModel(
            [
                revise,
                AssistantMessage(tool_calls=(ToolCall("b", "parallel_probe_b", {}),)),
                AssistantMessage("Probe B completed"),
            ]
        )
        planning_model = CapturedModel([AssistantMessage(json.dumps(payload))])
        with tempfile.TemporaryDirectory() as directory:
            controller = StreamlitRuntimeController(
                RuntimeConfig(
                    store_root=Path(directory),
                    probe_delay_seconds=0.01,
                    dynamic_planning=True,
                    completion_enforced=True,
                ),
                model_factory=lambda: actor,
                planner_factory=lambda: planning_model,
            )
            try:
                controller.start_run("Run only probe B")
                snapshot = _wait_for(controller, lambda s: s.latest_outcome is not None)
                self.assertTrue(snapshot.latest_outcome.result.succeeded)
                self.assertEqual(
                    [
                        x.criterion_id
                        for x in snapshot.task_definition.evaluation_plan.requirements
                    ],
                    ["b_only"],
                )
                self.assertEqual(snapshot.execution_plan.version, 2)
                self.assertEqual(
                    [p.tool_name for p in snapshot.probes], ["parallel_probe_b"]
                )
                self.assertEqual(len(planning_model.requests), 1)
                self.assertIn(
                    "Run only probe B", planning_model.requests[0].messages[1].content
                )
            finally:
                controller.close()
        self.assertEqual(planning_model.stops, 1)

    def test_planning_requires_explicit_model_and_trajectory(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeConfig(dynamic_planning=True, trajectory_enabled=False)
        with self.assertRaisesRegex(ValueError, "planner_factory"):
            StreamlitRuntimeController(RuntimeConfig(dynamic_planning=True))
