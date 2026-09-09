from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from test_evaluation import MutableSource, signal
from test_harness_controls import NoTools, ScriptedModel
from test_semantic_evaluation import JudgeModel, semantic_plan

from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    ModelRequest,
    RunCancelledError,
    RunStatus,
    TransientInstruction,
    UserMessage,
)
from ejagent.evaluation import (
    CompletionMode,
    CompletionPolicy,
    EvaluationMonitor,
    EvaluationStatus,
    GoalEvaluator,
    JsonlEvaluationJournal,
    JudgeLimits,
    ModelJudge,
    VerificationRequest,
)
from ejagent.harness import AgentHarness


class TestJudgeOutput(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.source = MutableSource()
        self.cancel = CancellationSource()

    def evaluator(self, model: JudgeModel, **kwargs: Any) -> GoalEvaluator:
        self.judge = ModelJudge(model, **kwargs)
        return GoalEvaluator(
            sources={"state": self.source}, verifiers={}, semantic_judge=self.judge
        )

    async def evaluate(
        self, evaluator: GoalEvaluator, index: int = 0, run: str = "run"
    ) -> Any:
        return await evaluator.evaluate(
            f"{run}:cp{index}",
            signal(semantic_plan(), index, run_id=run),
            cancellation=self.cancel.token,
        )

    async def test_fenced_verdicts_are_validated_without_an_extra_request(self) -> None:
        for status in ("pass", "fail", "unknown", "conflict"):
            for fence in ("```json", "```", "~~~JSON", "````json"):
                with self.subTest(status=status, fence=fence):
                    closing = fence.rstrip("jsonJSON")
                    model = JudgeModel(
                        lambda item, fence=fence, status=status, closing=closing: (
                            " \n"
                            + fence
                            + "\r\n"
                            + json.dumps({**item, "status": status})
                            + "\r\n"
                            + closing
                            + "\n"
                        )
                    )
                    report = await self.evaluate(self.evaluator(model))
                    self.assertEqual(report.requirements[0].status.value, status)
                    self.assertEqual(report.cost.model_requests, 1)
                    self.assertEqual(
                        report.judge_attempts[0].outcome, "recovered_markdown"
                    )
                    self.assertEqual(report.judge_attempts[0].retry_index, 0)

    async def test_format_steer_survives_cache_then_clears_after_plain_json(
        self,
    ) -> None:
        model = JudgeModel(lambda item: "```json\n" + json.dumps(item) + "\n```")
        evaluator = self.evaluator(model)
        await self.evaluate(evaluator)
        cached = await self.evaluate(evaluator, 1)
        self.assertEqual(cached.cost.cache_hits, 1)
        self.assertEqual(cached.judge_attempts, ())
        model.transform = None
        self.source.version += 1
        report = await self.evaluate(evaluator, 2)
        request = model.requests[-1]
        correction = request.messages[1]
        self.assertIsInstance(correction, TransientInstruction)
        assert isinstance(correction, TransientInstruction)
        self.assertEqual(correction.source, "judge:output_format")
        self.assertIn("Do not use backticks", correction.content)
        self.assertTrue(report.judge_attempts[0].format_steered)
        self.assertEqual(report.judge_attempts[0].request_index, 2)
        self.source.version += 1
        await self.evaluate(evaluator, 3)
        self.assertEqual(len(model.requests[-1].messages), 2)
        self.assertEqual(len(self.judge.attempts("run")), 3)
        evaluator.close_run("run")
        self.assertEqual(self.judge.attempts("run"), ())

    async def test_format_steer_is_isolated_across_runs(self) -> None:
        model = JudgeModel(lambda item: "```json\n" + json.dumps(item) + "\n```")
        evaluator = self.evaluator(model)
        await self.evaluate(evaluator)
        model.transform = None
        await self.evaluate(evaluator, run="other")
        self.assertEqual(len(model.requests[-1].messages), 2)
        evaluator.close_run("run")
        await self.evaluate(evaluator)
        self.assertEqual(len(model.requests[-1].messages), 2)

    async def test_invalid_output_retries_same_evidence_inside_judge(self) -> None:
        model = JudgeModel()
        model.transform = lambda item: "not JSON" if len(model.requests) == 1 else item
        report = await self.evaluate(self.evaluator(model))
        self.assertEqual(report.requirements[0].status, EvaluationStatus.PASS)
        self.assertEqual(report.cost.model_requests, 2)
        self.assertEqual(report.cost.model_input_tokens, 40)
        self.assertEqual(report.cost.model_output_tokens, 20)
        self.assertEqual(model.requests[0].messages[-1], model.requests[1].messages[-1])
        self.assertEqual(
            [a.outcome for a in report.judge_attempts], ["invalid_json", "valid"]
        )
        self.assertEqual([a.retry_index for a in report.judge_attempts], [0, 1])
        self.assertTrue(report.judge_attempts[1].format_steered)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "reports.jsonl"
            JsonlEvaluationJournal(path)(report)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["judge_attempts"][1]["outcome"], "valid")
            self.assertEqual(saved["cost"]["model_requests"], 2)

    async def test_fences_do_not_bypass_field_or_reference_validation(self) -> None:
        for changes in (
            {"status": "maybe"},
            {"criterion_id": "wrong"},
            {"evidence_refs": ["invented"]},
            {"extra": "pass"},
        ):
            with self.subTest(changes=changes):
                model = JudgeModel(
                    lambda item, changes=changes: (
                        "```json\n" + json.dumps({**item, **changes}) + "\n```"
                    )
                )
                report = await self.evaluate(self.evaluator(model))
                self.assertEqual(
                    report.requirements[0].status, EvaluationStatus.UNKNOWN
                )
                self.assertEqual(report.cost.model_requests, 2)
                self.assertTrue(
                    all(a.outcome == "invalid_schema" for a in report.judge_attempts)
                )

    async def test_ambiguous_or_partial_json_is_not_repaired(self) -> None:
        for wrap in (
            lambda s: "Explanation\n```json\n" + s + "\n```",
            lambda s: "```json\n" + s + "\n```\nMore text",
            lambda s: "```json\n" + s + "\n```\n```json\n" + s + "\n```",
            lambda s: "```json\n" + s,
            lambda s: "```json\n" + s + "\n~~~",
            lambda s: "```python\n" + s + "\n```",
            lambda s: "```json\n" + s[:-1] + ',"status":"fail"}\n```',
            lambda s: "```",
        ):
            with self.subTest(wrap=wrap):
                report = await self.evaluate(
                    self.evaluator(
                        JudgeModel(lambda item, wrap=wrap: wrap(json.dumps(item)))
                    )
                )
                self.assertEqual(
                    report.requirements[0].status, EvaluationStatus.UNKNOWN
                )
                self.assertEqual(report.cost.model_requests, 2)

    async def test_known_fail_and_semantic_unknown_are_not_retried(self) -> None:
        for status in ("fail", "unknown", "conflict"):
            model = JudgeModel(lambda item, status=status: {**item, "status": status})
            report = await self.evaluate(self.evaluator(model))
            self.assertEqual(report.requirements[0].status.value, status)
            self.assertEqual(report.cost.model_requests, 1)

    async def test_retries_respect_request_and_token_budgets(self) -> None:
        for limits in (
            JudgeLimits(max_requests=1),
            JudgeLimits(max_tokens=30),
            JudgeLimits(max_format_retries=0),
        ):
            with self.subTest(limits=limits):
                model = JudgeModel(lambda item: "bad JSON")
                report = await self.evaluate(self.evaluator(model, limits=limits))
                self.assertEqual(len(model.requests), 1)
                self.assertEqual(report.cost.model_requests, 1)
                self.assertEqual(
                    report.requirements[0].status, EvaluationStatus.UNKNOWN
                )

    async def test_format_steer_counts_toward_prompt_limit(self) -> None:
        model = JudgeModel(lambda item: "```json\n" + json.dumps(item) + "\n```")
        evaluator = self.evaluator(model)
        await self.evaluate(evaluator)
        size = sum(len(m.content.encode()) for m in model.requests[0].messages)
        self.judge.limits = replace(self.judge.limits, max_prompt_bytes=size + 1)
        self.source.version += 1
        report = await self.evaluate(evaluator, 1)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(report.judge_attempts, ())
        self.assertIn("prompt byte limit", report.requirements[0].rationale)

    async def test_truncated_and_refused_outputs_are_not_accepted_or_retried(
        self,
    ) -> None:
        for reason, outcome in (
            ("length", "truncated"),
            ("max_tokens", "truncated"),
            ("content_filter", "refused"),
            ("refusal", "refused"),
        ):

            class FinishedModel(JudgeModel):
                async def stream(self, request: Any, *, cancellation: Any) -> Any:
                    async for event in super().stream(
                        request, cancellation=cancellation
                    ):
                        yield replace(event, finish_reason=self.reason)

            model = FinishedModel()
            model.reason = reason
            report = await self.evaluate(self.evaluator(model))
            self.assertEqual(report.requirements[0].status, EvaluationStatus.UNKNOWN)
            self.assertEqual(report.cost.model_requests, 1)
            self.assertEqual(report.judge_attempts[0].outcome, outcome)
            self.assertEqual(report.judge_attempts[0].finish_reason, reason)

    async def test_retries_share_one_timeout(self) -> None:
        class SlowModel(JudgeModel):
            async def stream(self, request: Any, *, cancellation: Any) -> Any:
                await asyncio.sleep(0.03)
                async for event in super().stream(request, cancellation=cancellation):
                    yield event

        model = SlowModel(lambda item: "bad JSON")
        report = await self.evaluate(
            self.evaluator(model, limits=JudgeLimits(timeout_seconds=0.05))
        )
        self.assertEqual(report.requirements[0].status, EvaluationStatus.UNKNOWN)
        self.assertEqual(report.cost.model_requests, 2)
        self.assertEqual(report.cost.model_unreported_requests, 1)
        self.assertEqual(report.judge_attempts[-1].outcome, "interrupted")

    async def test_cancellation_during_retry_preserves_attempt_costs(self) -> None:
        model = JudgeModel(lambda item: "bad JSON")
        evaluator = self.evaluator(model)
        reports = []
        evaluator = GoalEvaluator(
            sources={"state": self.source},
            verifiers={},
            semantic_judge=self.judge,
            report_sink=reports.append,
        )

        second_started = asyncio.Event()

        def first(item: dict[str, Any]) -> object:
            if len(model.requests) == 2:
                model.block = True
                second_started.set()
            return "bad JSON"

        model.transform = first
        pending = asyncio.create_task(self.evaluate(evaluator))
        await second_started.wait()
        self.cancel.cancel()
        with self.assertRaises(RunCancelledError):
            await pending
        self.assertTrue(model.closed_request.is_set())
        self.assertEqual(reports[-1].cost.model_requests, 2)
        self.assertEqual(reports[-1].cost.model_unreported_requests, 1)
        self.assertEqual(reports[-1].judge_attempts[-1].outcome, "interrupted")
        evaluator.close_run("run")

    async def test_evidence_change_during_format_retry_invalidates_pass(self) -> None:
        model = JudgeModel()

        def change(item: dict[str, Any]) -> object:
            if len(model.requests) == 1:
                return "bad JSON"
            self.source.version += 1
            return item

        model.transform = change
        report = await self.evaluate(self.evaluator(model))
        self.assertEqual(report.requirements[0].status, EvaluationStatus.UNKNOWN)
        self.assertEqual(report.evidence, {})
        self.assertEqual(report.cost.model_requests, 2)
        self.assertEqual(report.judge_attempts[-1].outcome, "valid")

    async def test_custom_response_format_and_explicit_prompt_only_mode(self) -> None:
        for value in (
            None,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "verdict",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
        ):
            model = JudgeModel()
            await self.evaluate(self.evaluator(model, response_format=value))
            self.assertEqual(model.requests[0].response_format, value)

    def test_request_response_metadata_validation_and_deep_freezing(self) -> None:
        schema = {"type": "object", "required": ["status"]}
        options = {"type": "json_schema", "json_schema": {"schema": schema}}
        request = ModelRequest((UserMessage("JSON"),), response_format=options)
        schema["required"].append("injected")
        assert request.response_format is not None
        self.assertEqual(
            request.response_format["json_schema"]["schema"]["required"], ("status",)
        )
        for value in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                JudgeLimits(max_format_retries=value)

    async def test_internal_format_retry_does_not_reject_actor_completion(self) -> None:
        model = JudgeModel()
        model.transform = lambda item: "bad JSON" if len(model.requests) == 1 else item
        evaluator = self.evaluator(model)
        monitor = EvaluationMonitor(evaluator)
        actor = ScriptedModel([AssistantMessage("verified")])
        async with AgentHarness(
            agent_id="test",
            model=actor,
            tools=NoTools(),
            trajectory=monitor,
            context=monitor.context_pipeline(),
            completion_policy=CompletionPolicy(CompletionMode.ENFORCE, max_retries=0),
        ) as harness:
            outcome = await harness.run(
                "task", evaluation_plan=semantic_plan(completion_only=True)
            )
            self.assertEqual(harness.revision, 1)
        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(outcome.result.turns, 1)
        self.assertEqual(len(model.requests), 2)
        self.assertFalse(
            any(r.kind == "completion_rejected" for r in outcome.audit_records)
        )
        self.assertFalse(
            any(
                isinstance(m, TransientInstruction)
                and m.source == "judge:output_format"
                for request in actor.requests
                for m in request.messages
            )
        )

    async def test_standalone_cancellation_interrupts_stream_and_semaphore_wait(
        self,
    ) -> None:
        model = JudgeModel(block=True)
        judge = ModelJudge(model, limits=JudgeLimits(max_concurrency=1))
        plan = semantic_plan()
        first_signal = signal(plan)
        evidence = await self.source.read(first_signal, cancellation=self.cancel.token)
        request = VerificationRequest(
            plan.requirements[0], {"state": evidence}, first_signal, None
        )
        first = asyncio.create_task(judge.evaluate(request, self.cancel.token))
        await asyncio.wait_for(model.started_request.wait(), 1)
        second_cancel = CancellationSource()
        second = asyncio.create_task(
            judge.evaluate(
                replace(request, signal=signal(plan, run_id="second")),
                second_cancel.token,
            )
        )
        try:
            await asyncio.sleep(0.01)
            second_cancel.cancel()
            with self.assertRaises(RunCancelledError):
                await asyncio.wait_for(second, 0.5)
            self.assertEqual(len(model.requests), 1)
            self.assertEqual(judge.attempts("second"), ())
        finally:
            self.cancel.cancel()
            second_cancel.cancel()
            await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(model.closed_request.is_set())
        self.assertEqual(judge.attempts("run")[-1].outcome, "interrupted")
        judge.close_run("run")
        judge.close_run("second")
