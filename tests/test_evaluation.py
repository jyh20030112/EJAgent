from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from test_harness_controls import NoTools, ScriptedModel
from test_runtime_kernel import RecordingTools

from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    CancellationToken,
    CompletionCandidate,
    ContextRequest,
    JsonValue,
    RunStatus,
    ToolCall,
    ToolControl,
    ToolDefinition,
    ToolExecutionResult,
    TransientInstruction,
)
from ejagent.evaluation import (
    CheckResult,
    EvaluationCriterion,
    EvaluationMonitor,
    EvaluationPlan,
    EvaluationProtocolError,
    EvaluationReport,
    EvaluationStatus,
    EvidenceSnapshot,
    EvidenceUnavailable,
    FileEvidenceSource,
    GoalEvaluator,
    JsonlEvaluationJournal,
    ProbeEvidenceSource,
    VerificationRequest,
    boolean_field,
    file_exists,
    json_fields,
)
from ejagent.harness import AgentHarness
from ejagent.kernel.trajectory import (
    CausalAction,
    CheckpointSignal,
    CheckpointTrigger,
    TrajectoryCost,
)
from ejagent.storage import JsonlSessionStore


def plan(
    *, keys: tuple[str, ...] = ("state",), method: str = "ready", version: str = "v1"
) -> EvaluationPlan:
    return EvaluationPlan(
        "Deliver a verified result",
        version,
        (EvaluationCriterion("ready", "Result is ready", method, keys),),
    )


def signal(
    evaluation_plan: EvaluationPlan | None,
    turn: int = 0,
    *,
    run_id: str = "run",
    completion: CompletionCandidate | None = None,
) -> CheckpointSignal:
    trigger = (
        CheckpointTrigger.BASELINE
        if turn == 0
        else CheckpointTrigger.VERIFICATION_COMPLETED
    )
    if completion is not None:
        trigger = CheckpointTrigger.COMPLETION_PROPOSED
    return CheckpointSignal(
        run_id,
        trigger,
        turn,
        TrajectoryCost(),
        evaluation_plan=evaluation_plan,
        task="task",
        completion_candidate=completion,
    )


class MutableSource:
    def __init__(self, value: dict[str, JsonValue] | None = None) -> None:
        self.value = {"ready": True} if value is None else value
        self.version = 0
        self.unavailable = False
        self.slow = False
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.closed = []

    async def revision(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> str:
        cancellation.raise_if_cancelled()
        return str(self.version)

    async def read(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> EvidenceSnapshot:
        if self.unavailable:
            raise EvidenceUnavailable("test source is offline")
        if self.slow:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.stopped.set()
        return EvidenceSnapshot(str(self.version), self.value, f"capture:{signal.turn}")

    def close_run(self, run_id: str) -> None:
        self.closed.append(run_id)


class TestGoalEvaluator(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.token = CancellationSource().token

    async def evaluate(
        self,
        evaluator: GoalEvaluator,
        evaluation_plan: EvaluationPlan | None,
        turn: int = 0,
        **kwargs: Any,
    ) -> EvaluationReport:
        return await evaluator.evaluate(
            f"run:cp{turn}",
            signal(evaluation_plan, turn, **kwargs),
            cancellation=self.token,
        )

    async def test_no_plan_has_no_verdicts_and_does_not_read_sources(self) -> None:
        source = MutableSource()
        source.unavailable = True
        evaluator = GoalEvaluator(sources={"state": source}, verifiers={})
        report = await self.evaluate(evaluator, None)
        self.assertIsNone(report.plan)
        self.assertEqual(report.requirements, ())
        self.assertEqual(report.cost.source_reads, 0)
        evaluator.close_run("run")
        self.assertEqual(evaluator.active_run_ids, ())

    async def test_only_changed_dependencies_are_reverified(self) -> None:
        sources = {"a": MutableSource(), "b": MutableSource()}
        criteria = tuple(
            EvaluationCriterion(key, key, "ready", (key,)) for key in sources
        )
        task_plan = EvaluationPlan("both ready", "v1", criteria)
        evaluator = GoalEvaluator(
            sources=sources, verifiers={"ready": boolean_field("ready")}
        )
        first = await self.evaluate(evaluator, task_plan)
        sources["a"].value["ready"] = False
        sources["a"].version += 1
        second = await self.evaluate(evaluator, task_plan, 1)
        self.assertEqual(second.cost.verifier_calls, 1)
        self.assertEqual(second.cost.cache_hits, 1)
        self.assertEqual(first.evidence["a"].value["ready"], True)
        self.assertEqual(
            [item.status for item in second.requirements],
            [EvaluationStatus.FAIL, EvaluationStatus.PASS],
        )

    async def test_read_concurrency_is_bounded(self) -> None:
        active = maximum = 0

        class ConcurrentSource(MutableSource):
            async def read(
                self, signal: CheckpointSignal, *, cancellation: CancellationToken
            ) -> EvidenceSnapshot:
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                try:
                    await asyncio.sleep(0.002)
                    return await super().read(signal, cancellation=cancellation)
                finally:
                    active -= 1

        sources = {f"source-{index}": ConcurrentSource() for index in range(6)}
        evaluator = GoalEvaluator(
            sources=sources,
            verifiers={"ready": boolean_field("ready")},
            max_concurrency=2,
        )
        report = await self.evaluate(evaluator, plan(keys=tuple(sources)))
        self.assertTrue(report.fact_capture_complete)
        self.assertEqual(maximum, 2)
        self.assertEqual(active, 0)

    async def test_change_while_verifier_runs_prevents_stale_pass(self) -> None:
        source = MutableSource()

        async def verify(
            request: VerificationRequest, cancellation: CancellationToken
        ) -> CheckResult:
            # Simulate a concurrent external writer during a slow verification.
            source.version += 1
            source.value = {"ready": False}
            return CheckResult(
                EvaluationStatus.PASS, "old snapshot was ready", ("state",)
            )

        evaluator = GoalEvaluator(
            sources={"state": source}, verifiers={"ready": verify}
        )
        report = await self.evaluate(evaluator, plan())
        self.assertFalse(report.fact_capture_complete)
        self.assertEqual(report.evidence, {})
        self.assertEqual(report.requirements[0].status, EvaluationStatus.UNKNOWN)

    async def test_plan_is_immutable_and_cannot_change_within_run(self) -> None:
        requirements = [EvaluationCriterion("r", "ready", "ready", ("state",))]
        original = EvaluationPlan("goal", "v1", requirements)
        requirements.clear()
        self.assertEqual(len(original.requirements), 1)
        evaluator = GoalEvaluator(sources={}, verifiers={})
        await self.evaluate(evaluator, original)
        with self.assertRaisesRegex(EvaluationProtocolError, "cannot change"):
            await self.evaluate(evaluator, replace(original, goal="different"), 1)
        with self.assertRaisesRegex(EvaluationProtocolError, "cannot change"):
            await self.evaluate(evaluator, None, 2)

    async def test_file_absence_bad_structure_and_changed_pass(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.json"
            evaluator = GoalEvaluator(
                sources={"file": FileEvidenceSource(path)},
                verifiers={"exists": file_exists, "shape": json_fields("answer")},
            )
            task_plan = EvaluationPlan(
                "Write result",
                "v1",
                (
                    EvaluationCriterion("exists", "file exists", "exists", ("file",)),
                    EvaluationCriterion("shape", "answer field", "shape", ("file",)),
                ),
            )
            absent = await self.evaluate(evaluator, task_plan)
            self.assertEqual(
                [item.status for item in absent.requirements],
                [EvaluationStatus.FAIL] * 2,
            )
            path.write_text('{"answer":42}')
            passed = await self.evaluate(evaluator, task_plan, 1)
            self.assertTrue(
                all(
                    item.status is EvaluationStatus.PASS for item in passed.requirements
                )
            )
            path.write_text("{}")
            changed = await self.evaluate(evaluator, task_plan, 2)
            self.assertEqual(changed.requirements[1].status, EvaluationStatus.FAIL)
            self.assertIn(passed.evidence_ref("file"), changed.invalidated_refs)
            path.write_text("broken JSON")
            malformed = await self.evaluate(evaluator, task_plan, 3)
            self.assertEqual(malformed.requirements[1].status, EvaluationStatus.FAIL)
            path.write_bytes(b"\xff")
            corrupt = await self.evaluate(evaluator, task_plan, 4)
            self.assertTrue(
                all(
                    item.status is EvaluationStatus.UNKNOWN
                    for item in corrupt.requirements
                )
            )
            self.assertFalse(corrupt.fact_capture_complete)

    async def test_bounded_file_reads_and_nonregular_artifacts_are_unknown(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result"
            path.write_text("too long")
            for target in (path, Path(root)):
                evaluator = GoalEvaluator(
                    sources={"file": FileEvidenceSource(target, max_bytes=2)},
                    verifiers={"exists": file_exists},
                )
                report = await self.evaluate(
                    evaluator, plan(keys=("file",), method="exists")
                )
                self.assertEqual(
                    report.requirements[0].status, EvaluationStatus.UNKNOWN
                )

    async def test_probe_domain_shares_engine_and_is_run_scoped(self) -> None:
        probes = ProbeEvidenceSource()
        task_plan = EvaluationPlan(
            "Run probes together",
            "v1",
            tuple(
                EvaluationCriterion(name, name, name, ("probes",))
                for name in ("probe_a", "probe_b", "overlapped")
            ),
        )
        evaluator = GoalEvaluator(
            sources={"probes": probes},
            verifiers={
                name: boolean_field(name)
                for name in ("probe_a", "probe_b", "overlapped")
            },
        )
        baseline = await self.evaluate(evaluator, task_plan)
        self.assertTrue(
            all(item.status is EvaluationStatus.FAIL for item in baseline.requirements)
        )
        probes.record("run", "probe_a", started_at=0, finished_at=2)
        probes.record("run", "probe_b", started_at=1, finished_at=3)
        result = await self.evaluate(evaluator, task_plan, 1)
        self.assertTrue(
            all(item.status is EvaluationStatus.PASS for item in result.requirements)
        )
        evaluator.close_run("run")
        second = await self.evaluate(evaluator, task_plan, run_id="next")
        self.assertTrue(
            all(item.status is EvaluationStatus.FAIL for item in second.requirements)
        )

    async def test_repeated_semantic_evidence_is_deduplicated_and_checks_cached(
        self,
    ) -> None:
        source = MutableSource()
        evaluator = GoalEvaluator(
            sources={"state": source}, verifiers={"ready": boolean_field("ready")}
        )
        first = await self.evaluate(evaluator, plan())
        second = await self.evaluate(evaluator, plan(), 1)
        self.assertEqual(first.evidence_ref("state"), second.evidence_ref("state"))
        self.assertEqual(second.new_evidence, ())
        self.assertEqual(second.cost.cache_hits, 1)
        source.version += 1
        third = await self.evaluate(evaluator, plan(), 2)
        self.assertEqual(third.new_evidence, ())
        self.assertEqual(third.cost.verifier_calls, 1)
        self.assertIn(first.evidence_ref("state"), third.invalidated_refs)

    async def test_new_evidence_can_reduce_uncertainty_without_new_pass(self) -> None:
        source = MutableSource({"ready": False})
        source.unavailable = True
        evaluator = GoalEvaluator(
            sources={"state": source}, verifiers={"ready": boolean_field("ready")}
        )
        await self.evaluate(evaluator, plan())
        source.unavailable = False
        recovered = await self.evaluate(evaluator, plan(), 1)
        self.assertEqual(recovered.requirements[0].status, EvaluationStatus.FAIL)
        self.assertEqual(len(recovered.new_evidence), 1)

    async def test_source_timeout_invalidates_pass_then_recovers(self) -> None:
        source = MutableSource()
        evaluator = GoalEvaluator(
            sources={"state": source},
            verifiers={"ready": boolean_field("ready")},
            timeout_seconds=0.02,
        )
        first = await self.evaluate(evaluator, plan())
        source.slow = True
        failed_read = await self.evaluate(evaluator, plan(), 1)
        self.assertTrue(source.stopped.is_set())
        self.assertEqual(failed_read.requirements[0].status, EvaluationStatus.UNKNOWN)
        self.assertIn(first.evidence_ref("state"), failed_read.invalidated_refs)
        source.slow = False
        recovered = await self.evaluate(evaluator, plan(), 2)
        self.assertEqual(recovered.requirements[0].status, EvaluationStatus.PASS)
        self.assertEqual(recovered.cost.verifier_calls, 1)

    async def test_verifier_timeout_is_not_cached(self) -> None:
        attempts = 0

        async def check(
            request: VerificationRequest, cancellation: CancellationToken
        ) -> CheckResult:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.Event().wait()
            return CheckResult(EvaluationStatus.PASS, "ready", ("state",))

        evaluator = GoalEvaluator(
            sources={"state": MutableSource()},
            verifiers={"ready": check},
            timeout_seconds=0.02,
        )
        first = await self.evaluate(evaluator, plan())
        self.assertEqual(first.requirements[0].status, EvaluationStatus.UNKNOWN)
        self.assertFalse(first.fact_capture_complete)
        second = await self.evaluate(evaluator, plan(), 1)
        self.assertEqual(second.requirements[0].status, EvaluationStatus.PASS)
        self.assertEqual(attempts, 2)

    async def test_conflicting_sources_do_not_become_pass_or_fail(self) -> None:
        evaluator = GoalEvaluator(
            sources={
                "a": MutableSource({"ready": True}),
                "b": MutableSource({"ready": False}),
            },
            verifiers={"ready": boolean_field("ready")},
        )
        report = await self.evaluate(evaluator, plan(keys=("a", "b")))
        self.assertEqual(report.requirements[0].status, EvaluationStatus.CONFLICT)
        self.assertIsNone(report.requirements[0].status.verdict)
        self.assertEqual(len(report.requirements[0].evidence_refs), 2)
        self.assertFalse(report.fact_capture_complete)

    async def test_environment_race_invalidates_whole_capture(self) -> None:
        class RacingSource(MutableSource):
            async def revision(
                self, signal: CheckpointSignal, *, cancellation: CancellationToken
            ) -> str:
                self.version += 1
                return str(self.version)

        evaluator = GoalEvaluator(
            sources={"a": RacingSource(), "b": MutableSource()},
            verifiers={"ready": boolean_field("ready")},
        )
        report = await self.evaluate(evaluator, plan(keys=("a", "b")))
        self.assertEqual(report.evidence, {})
        self.assertEqual(report.requirements[0].status, EvaluationStatus.UNKNOWN)

    async def test_completion_candidate_is_an_explicit_dependency(self) -> None:
        candidates = []

        async def check(
            request: VerificationRequest, cancellation: CancellationToken
        ) -> CheckResult:
            candidates.append(
                (
                    request.evidence["$completion"].value,
                    request.previous_report,
                    request.signal.evaluation_plan.artifact_refs,
                )
            )
            return CheckResult(
                EvaluationStatus.PASS, "contains expected response", ("$completion",)
            )

        evaluator = GoalEvaluator(sources={}, verifiers={"answer": check})
        task_plan = replace(
            plan(keys=("$completion",), method="answer"),
            artifact_refs=("artifact:result",),
        )
        first = await self.evaluate(evaluator, task_plan)
        self.assertEqual(first.requirements[0].status, EvaluationStatus.UNKNOWN)
        second = await self.evaluate(
            evaluator, task_plan, 1, completion=CompletionCandidate("done")
        )
        self.assertEqual(second.requirements[0].status, EvaluationStatus.PASS)
        self.assertEqual(candidates[0][0], "done")
        self.assertIs(candidates[0][1], first)
        self.assertEqual(candidates[0][2], ("artifact:result",))
        third = await self.evaluate(
            evaluator, task_plan, 2, completion=CompletionCandidate("partial", True)
        )
        self.assertEqual(third.requirements[0].status, EvaluationStatus.UNKNOWN)

    async def test_invalid_verifier_reference_is_a_protocol_error(self) -> None:
        async def check(
            request: VerificationRequest, cancellation: CancellationToken
        ) -> CheckResult:
            return CheckResult(EvaluationStatus.PASS, "unsupported", ("undeclared",))

        evaluator = GoalEvaluator(
            sources={"state": MutableSource()}, verifiers={"ready": check}
        )
        with self.assertRaises(EvaluationProtocolError):
            await self.evaluate(evaluator, plan())
        evaluator.close_run("run")
        self.assertEqual(evaluator.active_run_ids, ())

    async def test_reports_and_evidence_remain_resolvable_after_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            journal = JsonlEvaluationJournal(Path(root) / "evaluation.jsonl")
            evaluator = GoalEvaluator(
                sources={"state": MutableSource()},
                verifiers={"ready": boolean_field("ready")},
                report_sink=journal,
            )
            report = await self.evaluate(evaluator, plan())
            evaluator.close_run("run")
            stored = json.loads(journal.path.read_text())
            self.assertEqual(stored["report_ref"], report.report_ref)
            self.assertEqual(
                stored["evidence"]["state"]["evidence_ref"],
                stored["requirements"][0]["evidence_refs"][0],
            )
            self.assertEqual(stored["evidence"]["state"]["value"], {"ready": True})
            self.assertEqual(
                stored["requirements"][0]["evidence_versions"], {"state": "0"}
            )


class TestEvaluationHarness(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.token = CancellationSource().token

    async def projected(
        self, monitor: EvaluationMonitor, run_id: str = "run", turn: int = 1
    ) -> list[dict[str, Any]]:
        pipeline = monitor.context_pipeline()
        await pipeline.start()
        try:
            view = await pipeline.build(
                ContextRequest(run_id, 0, turn, ()), cancellation=self.token
            )
        finally:
            await pipeline.shutdown()
        return [
            json.loads(item.content)["trajectory_context"]
            for item in view.messages
            if isinstance(item, TransientInstruction)
        ]

    async def test_unknown_feedback_recovers_and_old_facts_are_not_projected(
        self,
    ) -> None:
        source = MutableSource()
        evaluator = GoalEvaluator(
            sources={"state": source}, verifiers={"ready": boolean_field("ready")}
        )
        monitor = EvaluationMonitor(evaluator)
        await monitor.capture(signal(plan()), cancellation=self.token)
        source.unavailable = True
        receipt = await monitor.capture(signal(plan(), 1), cancellation=self.token)
        self.assertEqual(receipt.verdict, "insufficient_evidence")
        partial = (await self.projected(monitor, turn=2))[0]
        self.assertEqual(partial["feedback"]["event"], "evaluation_unavailable")
        self.assertNotIn("current_facts", partial)
        self.assertNotIn("progress", partial)
        self.assertEqual(partial["missing_evidence"], ["state"])
        self.assertEqual(len(partial["invalidated_facts"]), 1)
        source.unavailable = False
        receipt = await monitor.capture(
            signal(plan(), 2, completion=CompletionCandidate("done")),
            cancellation=self.token,
        )
        self.assertTrue(receipt.completion_allowed)
        monitor.close_run("run")
        closed_state = (await self.projected(monitor, turn=3))[0]
        self.assertEqual(closed_state["state_status"], "unavailable")
        self.assertIsNone(closed_state["checkpoint"])
        self.assertNotIn("progress", closed_state)
        self.assertEqual(evaluator.active_run_ids, ())

    async def test_duplicate_reads_still_allow_cycle_detection(self) -> None:
        source = MutableSource({"ready": False})
        evaluator = GoalEvaluator(
            sources={"state": source}, verifiers={"ready": boolean_field("ready")}
        )
        monitor = EvaluationMonitor(evaluator)
        await monitor.capture(signal(plan()), cancellation=self.token)
        for turn in range(1, 5):
            # Re-observation and source revision change are not semantic novelty.
            source.version += 1
            current = CheckpointSignal(
                "run",
                CheckpointTrigger.TOOL_BATCH_COMPLETED,
                turn,
                TrajectoryCost(actor_actions=turn),
                causal_actions=(CausalAction(f"call-{turn}", "read-state"),),
                causal_batch_id=f"batch-{turn}",
                evaluation_plan=plan(),
            )
            receipt = await monitor.capture(current, cancellation=self.token)
        self.assertEqual(receipt.verdict, "non_progress_cycle")
        frame = (await self.projected(monitor, turn=5))[0]
        self.assertEqual(frame["feedback"]["event"], "cycle_confirmed")
        monitor.close_run("run")

    async def test_constraint_regression_uses_existing_progress_calculation(
        self,
    ) -> None:
        source = MutableSource({"ready": False, "safe": True})
        task_plan = replace(
            plan(),
            constraints=(
                EvaluationCriterion("safe", "preserve safety", "safe", ("state",)),
            ),
        )
        evaluator = GoalEvaluator(
            sources={"state": source},
            verifiers={"ready": boolean_field("ready"), "safe": boolean_field("safe")},
        )
        monitor = EvaluationMonitor(evaluator)
        await monitor.capture(signal(task_plan), cancellation=self.token)
        source.value = {"ready": True, "safe": False}
        source.version += 1
        await monitor.capture(signal(task_plan, 1), cancellation=self.token)
        frame = (await self.projected(monitor, turn=2))[0]
        self.assertEqual(frame["feedback"]["event"], "constraint_violated")
        self.assertIsNone(frame["progress"]["task_progress_delta"])
        self.assertEqual(frame["progress"]["newly_violated_constraints"], ["safe"])

    async def test_failed_completion_is_advisory_and_report_refs_reach_audit(
        self,
    ) -> None:
        reports = []
        evaluator = GoalEvaluator(
            sources={"state": MutableSource({"ready": False})},
            verifiers={"ready": boolean_field("ready")},
            report_sink=reports.append,
        )
        monitor = EvaluationMonitor(evaluator)
        model = ScriptedModel([AssistantMessage("done")])
        async with AgentHarness(
            agent_id="test",
            model=model,
            tools=NoTools(),
            trajectory=monitor,
            context=monitor.context_pipeline(),
        ) as harness:
            outcome = await harness.run("task", evaluation_plan=plan())
            self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
            self.assertEqual(outcome.result.usage.request_count, 1)
            checkpoints = [
                item
                for item in outcome.audit_records
                if item.kind == "trajectory_checkpointed"
            ]
            self.assertFalse(checkpoints[-1].payload["completion_allowed"])
            self.assertEqual(
                checkpoints[-1].payload["evaluation_report_ref"], reports[-1].report_ref
            )
            self.assertFalse(
                any(isinstance(item, TransientInstruction) for item in harness.messages)
            )
        self.assertEqual(evaluator.active_run_ids, ())

    async def test_follow_up_and_continue_have_independent_plans(self) -> None:
        reports = []
        evaluator = GoalEvaluator(
            sources={"state": MutableSource()},
            verifiers={"ready": boolean_field("ready")},
            report_sink=reports.append,
        )
        monitor = EvaluationMonitor(evaluator)
        model = ScriptedModel(
            [
                AssistantMessage("first"),
                AssistantMessage("second"),
                AssistantMessage("third"),
                AssistantMessage("fourth"),
            ],
            block_first=True,
        )
        async with AgentHarness(
            agent_id="test",
            model=model,
            tools=NoTools(),
            trajectory=monitor,
            context=monitor.context_pipeline(),
        ) as harness:
            task = asyncio.create_task(harness.run("one", evaluation_plan=plan()))
            await model.started.wait()
            second = harness.follow_up("two", evaluation_plan=plan(version="v2"))
            third = harness.follow_up("three")
            model.release.set()
            await task
            await second.wait()
            await third.wait()
            await harness.continue_run()
        versions = {}
        for report in reports:
            versions[report.run_id] = report.plan.version if report.plan else None
        self.assertEqual(list(versions.values()), ["v1", "v2", None, None])
        self.assertEqual(evaluator.active_run_ids, ())
        for request in model.requests[2:]:
            states = [
                json.loads(item.content)["trajectory_context"]
                for item in request.messages
                if isinstance(item, TransientInstruction)
            ]
            self.assertEqual(len(states), 1)
            self.assertEqual(states[0]["state_status"], "unavailable")
            self.assertIsNone(states[0]["checkpoint"])
            self.assertNotIn("requirements", states[0])
            self.assertNotIn("current_facts", states[0])

    async def test_cancellation_during_capture_cleans_sources_and_frames(self) -> None:
        source = MutableSource()
        source.slow = True
        evaluator = GoalEvaluator(
            sources={"state": source},
            verifiers={"ready": boolean_field("ready")},
            timeout_seconds=60,
        )
        monitor = EvaluationMonitor(evaluator)
        model = ScriptedModel([])
        async with AgentHarness(
            agent_id="test",
            model=model,
            tools=NoTools(),
            trajectory=monitor,
            context=monitor.context_pipeline(),
            run_id_factory=lambda: "run",
        ) as harness:
            task = asyncio.create_task(harness.run("task", evaluation_plan=plan()))
            await source.started.wait()
            harness.cancel("stop")
            outcome = await task
            self.assertEqual(outcome.result.status, RunStatus.CANCELLED)
        self.assertTrue(source.stopped.is_set())
        self.assertEqual(source.closed, ["run"])
        self.assertEqual(evaluator.active_run_ids, ())
        state = (await self.projected(monitor))[0]
        self.assertEqual(state["state_status"], "unavailable")
        self.assertIsNone(state["checkpoint"])
        self.assertNotIn("current_facts", state)

    async def test_tool_batch_observations_and_completion_text_reach_verifier(
        self,
    ) -> None:
        observations = []

        class RecordingSource(MutableSource):
            async def read(
                self, signal: CheckpointSignal, *, cancellation: CancellationToken
            ) -> EvidenceSnapshot:
                observations.append(signal)
                return await super().read(signal, cancellation=cancellation)

        evaluator = GoalEvaluator(
            sources={"state": RecordingSource()},
            verifiers={"ready": boolean_field("ready")},
        )
        monitor = EvaluationMonitor(evaluator)
        model = ScriptedModel(
            [
                AssistantMessage(tool_calls=(ToolCall("call-1", "read"),)),
                AssistantMessage("final text"),
            ]
        )
        tools = RecordingTools(
            (ToolDefinition("read"),),
            results={"read": ToolExecutionResult({"answer": 42})},
        )
        async with AgentHarness(
            agent_id="test", model=model, tools=tools, trajectory=monitor
        ) as harness:
            outcome = await harness.run("task", evaluation_plan=plan())
        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        batch = observations[1]
        self.assertEqual(batch.tool_observations[0].call_id, "call-1")
        self.assertTrue(batch.observations_complete)
        self.assertEqual(batch.task, "task")
        self.assertEqual(observations[-1].completion_candidate.text, "final text")

    async def test_tool_completion_also_checks_all_criteria(self) -> None:
        reports = []
        evaluator = GoalEvaluator(
            sources={"state": MutableSource({"ready": False})},
            verifiers={"ready": boolean_field("ready")},
            report_sink=reports.append,
        )
        monitor = EvaluationMonitor(evaluator)
        model = ScriptedModel(
            [AssistantMessage(tool_calls=(ToolCall("finish-1", "finish"),))]
        )
        tools = RecordingTools(
            (ToolDefinition("finish"),),
            results={
                "finish-1": ToolExecutionResult(
                    {}, control=ToolControl.COMPLETE, output="done"
                )
            },
        )
        async with AgentHarness(
            agent_id="test", model=model, tools=tools, trajectory=monitor
        ) as harness:
            outcome = await harness.run("task", evaluation_plan=plan())
        captures = [
            item
            for item in outcome.audit_records
            if item.kind == "trajectory_checkpointed"
        ]
        self.assertEqual(len(captures), 3)
        self.assertEqual(captures[-1].payload["trigger"], "completion_proposed")
        self.assertFalse(captures[-1].payload["completion_allowed"])
        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)

    async def test_kernel_bounds_observations_and_candidate_without_changing_output(
        self,
    ) -> None:
        observed = []

        class ObservingSource(MutableSource):
            async def read(
                self, signal: CheckpointSignal, *, cancellation: CancellationToken
            ) -> EvidenceSnapshot:
                observed.append(signal)
                return await super().read(signal, cancellation=cancellation)

        output = "x" * 65_537
        model = ScriptedModel(
            [
                AssistantMessage(
                    tool_calls=tuple(
                        ToolCall(f"call-{i}", "read", {"index": i}) for i in range(129)
                    )
                ),
                AssistantMessage(output),
            ]
        )
        evaluator = GoalEvaluator(
            sources={"state": ObservingSource()},
            verifiers={"ready": boolean_field("ready")},
        )
        monitor = EvaluationMonitor(evaluator)
        async with AgentHarness(
            agent_id="test",
            model=model,
            tools=RecordingTools((ToolDefinition("read"),)),
            trajectory=monitor,
        ) as harness:
            outcome = await harness.run("task", evaluation_plan=plan())
        self.assertEqual(outcome.result.output, output)
        self.assertEqual(len(observed[1].tool_observations), 128)
        self.assertFalse(observed[1].observations_complete)
        self.assertEqual(len(observed[-1].completion_candidate.text), 65_536)
        self.assertTrue(observed[-1].completion_candidate.truncated)

    async def test_temporary_failure_does_not_disable_monitor_for_rest_of_run(
        self,
    ) -> None:
        class RecoveringSource(MutableSource):
            async def read(
                self, signal: CheckpointSignal, *, cancellation: CancellationToken
            ) -> EvidenceSnapshot:
                if signal.turn == 0:
                    raise EvidenceUnavailable("temporarily offline")
                return await super().read(signal, cancellation=cancellation)

        evaluator = GoalEvaluator(
            sources={"state": RecoveringSource()},
            verifiers={"ready": boolean_field("ready")},
        )
        monitor = EvaluationMonitor(evaluator)
        model = ScriptedModel(
            [
                AssistantMessage(tool_calls=(ToolCall("read-1", "read"),)),
                AssistantMessage("done"),
            ]
        )
        async with AgentHarness(
            agent_id="test",
            model=model,
            tools=RecordingTools((ToolDefinition("read"),)),
            trajectory=monitor,
            context=monitor.context_pipeline(),
        ) as harness:
            outcome = await harness.run("task", evaluation_plan=plan())
        self.assertFalse(
            any(
                item.kind == "trajectory_capture_failed"
                for item in outcome.audit_records
            )
        )
        first_context = [
            item
            for item in model.requests[0].messages
            if isinstance(item, TransientInstruction)
        ][0]
        next_context = [
            item
            for item in model.requests[1].messages
            if isinstance(item, TransientInstruction)
        ][0]
        self.assertEqual(first_context.source, "trajectory:evaluation_unavailable")
        self.assertIn(
            "current_facts", json.loads(next_context.content)["trajectory_context"]
        )
        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)

    async def test_session_restore_does_not_restore_run_plan_or_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            reports = []
            evaluator = GoalEvaluator(
                sources={"state": MutableSource()},
                verifiers={"ready": boolean_field("ready")},
                report_sink=reports.append,
            )
            monitor = EvaluationMonitor(evaluator)
            async with AgentHarness(
                agent_id="test",
                model=ScriptedModel([AssistantMessage("first")]),
                tools=NoTools(),
                trajectory=monitor,
                context=monitor.context_pipeline(),
                store=JsonlSessionStore(root),
            ) as first:
                await first.run("task", evaluation_plan=plan())
            model = ScriptedModel([AssistantMessage("continued")])
            async with AgentHarness(
                agent_id="test",
                model=model,
                tools=NoTools(),
                trajectory=monitor,
                context=monitor.context_pipeline(),
                store=JsonlSessionStore(root),
            ) as restored:
                self.assertEqual(restored.revision, 1)
                outcome = await restored.continue_run()
            self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
            self.assertIsNone(reports[-1].plan)
            states = [
                json.loads(item.content)["trajectory_context"]
                for item in model.requests[0].messages
                if isinstance(item, TransientInstruction)
            ]
            self.assertEqual(len(states), 1)
            self.assertEqual(states[0]["state_status"], "unavailable")
            self.assertIsNone(states[0]["checkpoint"])
            self.assertNotIn("requirements", states[0])
            self.assertNotIn("current_facts", states[0])

    async def test_protocol_failure_is_audited_and_still_cleans_run_state(self) -> None:
        async def broken(
            request: VerificationRequest, cancellation: CancellationToken
        ) -> CheckResult:
            return CheckResult(
                EvaluationStatus.PASS, "invalid evidence reference", ("wrong",)
            )

        source = MutableSource()
        evaluator = GoalEvaluator(
            sources={"state": source}, verifiers={"ready": broken}
        )
        monitor = EvaluationMonitor(evaluator)
        async with AgentHarness(
            agent_id="test",
            model=ScriptedModel([AssistantMessage("done")]),
            tools=NoTools(),
            trajectory=monitor,
        ) as harness:
            outcome = await harness.run("task", evaluation_plan=plan())
        failures = [
            item
            for item in outcome.audit_records
            if item.kind == "trajectory_capture_failed"
        ]
        self.assertEqual(failures[0].payload["error_type"], "EvaluationProtocolError")
        self.assertEqual(evaluator.active_run_ids, ())
        self.assertEqual(len(source.closed), 1)
