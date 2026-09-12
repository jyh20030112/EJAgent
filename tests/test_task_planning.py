from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path

from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    CancellationToken,
    CompletionMode,
    CompletionPolicy,
    EvaluationCriterion,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelUsage,
    RunCancelledError,
    RunStatus,
    StepStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolResultMessage,
    TransientInstruction,
    UserMessage,
)
from ejagent.evaluation import (
    EvaluationMonitor,
    FileEvidenceSource,
    GoalEvaluator,
    file_exists,
)
from ejagent.harness import AgentHarness
from ejagent.planning import (
    ModelTaskPlanner,
    PlannerLimits,
    PlanningError,
    PlanningRequest,
    VerificationCapability,
)
from ejagent.storage import JsonlSessionStore
from ejagent.tools import FunctionTool, FunctionToolExecutor


def proposal(task: str = "Create the requested artifact") -> dict[str, object]:
    return {
        "task": task,
        "goal": "The requested artifact exists",
        "criteria": [
            {
                "id": "artifact",
                "capability": "artifact_exists",
                "description": "Deliver the requested artifact",
            }
        ],
        "steps": [
            {
                "id": "write",
                "description": "Write the artifact",
                "requirement_ids": ["artifact"],
                "status": "pending",
            }
        ],
        "unknowns": [],
        "unsupported": [],
    }


def capability() -> VerificationCapability:
    return VerificationCapability(
        "artifact_exists",
        EvaluationCriterion(
            "template", "The host-configured artifact exists", "exists", ("artifact",)
        ),
        required=True,
    )


class CapturedModel:
    def __init__(
        self,
        replies: list[AssistantMessage | Callable[[ModelRequest], AssistantMessage]],
        *,
        finish_reason: str = "stop",
    ) -> None:
        self.replies = list(replies)
        self.requests: list[ModelRequest] = []
        self.finish_reason = finish_reason
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1

    async def shutdown(self) -> None:
        self.stops += 1

    async def stream(
        self, request: ModelRequest, *, cancellation: CancellationToken
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        reply = self.replies.pop(0)
        yield ModelResponseCompleted(
            reply(request) if callable(reply) else reply,
            ModelUsage(10, 5, 15),
            self.finish_reason,
        )


def planner(value: dict[str, object] | None = None) -> ModelTaskPlanner:
    return ModelTaskPlanner(
        CapturedModel([AssistantMessage(json.dumps(value or proposal()))]),
        capabilities=(capability(),),
    )


def task_context(request: ModelRequest) -> dict:
    return next(
        json.loads(m.content)["task_context"]
        for m in request.messages
        if isinstance(m, TransientInstruction) and m.source == "planning"
    )


def revise(
    request: ModelRequest,
    *,
    version_delta: int = 0,
    status: str = "in_progress",
    checkpoint: str | None = None,
    extra: bool = False,
) -> AssistantMessage:
    context = task_context(request)
    plan = context["execution_plan"]
    steps = plan["steps"]
    steps[0]["status"] = status
    steps[0]["description"] = "Recover after evidence feedback"
    args = {
        "expected_version": plan["version"] + version_delta,
        "based_on_checkpoint": checkpoint or context["latest_checkpoint"],
        "reason": "The latest checkpoint requires more work",
        "steps": steps,
    }
    if extra:
        args["goal"] = "weakened goal"
    return AssistantMessage(
        tool_calls=(ToolCall(f"update-{len(request.messages)}", "update_plan", args),)
    )


class TestModelTaskPlanner(unittest.IsolatedAsyncioTestCase):
    async def test_dynamic_query_and_registered_binding(self) -> None:
        adapter = planner(proposal("Create release notes"))
        result = await adapter.plan(
            PlanningRequest(
                "run",
                "Write release notes",
                (
                    UserMessage("Use Markdown"),
                    ToolResultMessage(
                        "prior", "read_file", {"fact": "prior tool observation"}
                    ),
                ),
            ),
            cancellation=CancellationToken(),
        )
        self.assertEqual(result.definition.task, "Create release notes")
        self.assertEqual(
            result.definition.evaluation_plan.requirements[0].method, "exists"
        )
        self.assertEqual(result.definition.execution_plan.version, 1)
        self.assertEqual(result.usage.total_tokens, 15)
        request = adapter.model.requests[0]
        self.assertEqual(request.response_format, {"type": "json_object"})
        self.assertEqual(request.tools, ())
        self.assertIn("Write release notes", request.messages[1].content)
        self.assertIn("Use Markdown", request.messages[1].content)
        self.assertIn("prior tool observation", request.messages[1].content)

    async def test_rejects_unknown_capabilities_and_unsupported_goals(self) -> None:
        cases = []
        unknown = proposal()
        unknown["criteria"][0]["capability"] = "invented_shell_verifier"
        cases.append(unknown)
        unsupported = proposal()
        unsupported["unsupported"] = ["Cannot verify production deployment"]
        cases.append(unsupported)
        weakened = proposal()
        weakened["steps"][0]["requirement_ids"] = ["invented"]
        cases.append(weakened)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(PlanningError):
                await planner(value).plan(
                    PlanningRequest("r", "query"), cancellation=CancellationToken()
                )

    async def test_required_capability_cannot_be_omitted(self) -> None:
        adapter = ModelTaskPlanner(
            CapturedModel([AssistantMessage(json.dumps(proposal()))]),
            capabilities=(
                capability(),
                VerificationCapability(
                    "other",
                    EvaluationCriterion("x", "Host condition", "exists", ("artifact",)),
                    required=True,
                ),
            ),
        )
        with self.assertRaisesRegex(PlanningError, "required host capability"):
            await adapter.plan(
                PlanningRequest("r", "q"), cancellation=CancellationToken()
            )

    async def test_bounded_json_and_finish_reason(self) -> None:
        for text, reason in [
            (json.dumps(proposal()), "length"),
            ('{"task":1,"task":2}', "stop"),
            ("prefix " + json.dumps(proposal()), "stop"),
        ]:
            adapter = ModelTaskPlanner(
                CapturedModel([AssistantMessage(text)], finish_reason=reason),
                capabilities=(capability(),),
                limits=PlannerLimits(max_format_retries=0),
            )
            with self.assertRaises(PlanningError):
                await adapter.plan(
                    PlanningRequest("r", "q"), cancellation=CancellationToken()
                )
        adapter = ModelTaskPlanner(
            CapturedModel(
                [AssistantMessage("```json\n" + json.dumps(proposal()) + "\n```")]
            ),
            capabilities=(capability(),),
            response_format=None,
        )
        await adapter.plan(PlanningRequest("r", "q"), cancellation=CancellationToken())
        self.assertIsNone(adapter.model.requests[0].response_format)

    async def test_prompt_bound_prevents_model_call(self) -> None:
        model = CapturedModel([])
        adapter = ModelTaskPlanner(
            model,
            capabilities=(capability(),),
            limits=PlannerLimits(max_prompt_bytes=10),
        )
        with self.assertRaises(PlanningError):
            await adapter.plan(
                PlanningRequest("r", "q"), cancellation=CancellationToken()
            )
        self.assertEqual(model.requests, [])

    async def test_format_retries_validate_nested_fields_and_account_for_all_calls(
        self,
    ) -> None:
        invalid = proposal()
        invalid["steps"][0]["status"] = "completed"
        invalid["steps"][0]["description"] = 123
        texts = ["not JSON", json.dumps(invalid), json.dumps(proposal())]
        model = CapturedModel([AssistantMessage(text) for text in texts])
        adapter = ModelTaskPlanner(
            model,
            capabilities=(capability(),),
            limits=PlannerLimits(max_format_retries=2),
        )
        result = await adapter.plan(
            PlanningRequest("run", "Original goal"), cancellation=CancellationToken()
        )
        self.assertEqual(result.requests, 3)
        self.assertEqual(result.usage.total_tokens, 45)
        self.assertEqual(result.usage.input_tokens, 30)
        for request in model.requests:
            self.assertEqual(request.messages[0], model.requests[0].messages[0])
            self.assertEqual(request.messages[-1], model.requests[0].messages[-1])
            self.assertEqual(request.tools, ())
            self.assertEqual(request.response_format, {"type": "json_object"})
        error = json.loads(model.requests[2].messages[2].content)[
            "structured_output_error"
        ]
        self.assertIn("steps.0.status", error["errors"])
        self.assertIn("steps.0.description", error["errors"])
        self.assertEqual(error["previous_response_excerpt"], texts[1])
        self.assertEqual(model.requests[2].messages[1].source, "planner:output_format")

    async def test_fenced_output_is_accepted_and_only_reminder_reaches_next_query(
        self,
    ) -> None:
        text = json.dumps(proposal("private previous task"))
        model = CapturedModel(
            [
                AssistantMessage("```json\n" + text + "\n```"),
                AssistantMessage(json.dumps(proposal())),
                AssistantMessage(json.dumps(proposal())),
            ]
        )
        adapter = ModelTaskPlanner(model, capabilities=(capability(),))
        for index in range(3):
            result = await adapter.plan(
                PlanningRequest(str(index), "New query"),
                cancellation=CancellationToken(),
            )
            self.assertEqual(result.requests, 1)
        self.assertEqual(len(model.requests[0].messages), 2)
        self.assertEqual(len(model.requests[1].messages), 3)
        self.assertIn("Do not use backticks", model.requests[1].messages[1].content)
        self.assertNotIn("private previous task", str(model.requests[1].messages))
        self.assertEqual(len(model.requests[2].messages), 2)

    async def test_retry_budget_exhaustion_never_returns_a_plan(self) -> None:
        for limits, calls in (
            (PlannerLimits(max_format_retries=0), 1),
            (PlannerLimits(max_format_retries=2), 3),
            (PlannerLimits(max_tokens=15), 1),
            (PlannerLimits(max_tokens=10), 1),
        ):
            with self.subTest(limits=limits):
                model = CapturedModel([AssistantMessage("bad JSON")] * 3)
                adapter = ModelTaskPlanner(
                    model, capabilities=(capability(),), limits=limits
                )
                with self.assertRaises(PlanningError) as caught:
                    await adapter.plan(
                        PlanningRequest("r", "q"), cancellation=CancellationToken()
                    )
                self.assertEqual(caught.exception.requests, calls)
                self.assertEqual(caught.exception.usage.total_tokens, calls * 15)
                self.assertEqual(len(model.requests), calls)

    async def test_retry_context_counts_toward_prompt_limit(self) -> None:
        model = CapturedModel([AssistantMessage(json.dumps(proposal()))])
        adapter = ModelTaskPlanner(model, capabilities=(capability(),))
        await adapter.plan(PlanningRequest("r", "q"), cancellation=CancellationToken())
        size = sum(len(m.content.encode()) for m in model.requests[0].messages)
        adapter.limits = replace(adapter.limits, max_prompt_bytes=size + 1)
        model.replies = [AssistantMessage("invalid JSON")]
        with self.assertRaisesRegex(PlanningError, "context exceeds") as caught:
            await adapter.plan(
                PlanningRequest("r", "q"), cancellation=CancellationToken()
            )
        self.assertEqual(caught.exception.requests, 1)
        self.assertEqual(len(model.requests), 2)

    async def test_missing_usage_prevents_unaccounted_retries(self) -> None:
        class NoUsage(CapturedModel):
            async def stream(self, request, *, cancellation):
                async for event in super().stream(request, cancellation=cancellation):
                    yield replace(event, usage=None)

        model = NoUsage([AssistantMessage("bad JSON")])
        adapter = ModelTaskPlanner(model, capabilities=(capability(),))
        with self.assertRaisesRegex(PlanningError, "usage is unavailable") as caught:
            await adapter.plan(
                PlanningRequest("r", "q"), cancellation=CancellationToken()
            )
        self.assertEqual(caught.exception.requests, 1)
        self.assertIsNone(caught.exception.usage)

    async def test_retries_share_timeout_and_close_cancelled_stream(self) -> None:
        class BlocksRetry(CapturedModel):
            def __init__(self):
                super().__init__([AssistantMessage("bad JSON")])
                self.started = asyncio.Event()
                self.closed = asyncio.Event()

            async def stream(self, request, *, cancellation):
                if not self.requests:
                    async for event in super().stream(
                        request, cancellation=cancellation
                    ):
                        yield event
                else:
                    self.started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.closed.set()

        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                model = BlocksRetry()
                token = CancellationSource()
                adapter = ModelTaskPlanner(
                    model,
                    capabilities=(capability(),),
                    limits=PlannerLimits(timeout_seconds=0.1),
                )
                task = asyncio.create_task(
                    adapter.plan(PlanningRequest("r", "q"), cancellation=token.token)
                )
                await model.started.wait()
                if cancel:
                    token.cancel()
                with self.assertRaises(RunCancelledError if cancel else PlanningError):
                    await task
                self.assertTrue(model.closed.is_set())


class TestPlannedHarness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "artifact.md"
        self.reports = []
        self.evaluator = GoalEvaluator(
            sources={"artifact": FileEvidenceSource(self.path)},
            verifiers={"exists": file_exists},
            report_sink=self.reports.append,
        )
        self.monitor = EvaluationMonitor(self.evaluator)

        async def write(call, cancellation):
            self.path.write_text("delivered")
            return ToolExecutionResult({"written": True})

        self.tools = FunctionToolExecutor(
            (FunctionTool(ToolDefinition("write_artifact"), write),)
        )

    def harness(self, actor: CapturedModel, **kwargs) -> AgentHarness:
        return AgentHarness(
            agent_id="planned",
            model=actor,
            tools=self.tools,
            planner=kwargs.pop("planner", planner()),
            trajectory=self.monitor,
            context=self.monitor.context_pipeline(),
            completion_policy=CompletionPolicy(CompletionMode.ENFORCE),
            run_id_factory=lambda: "run",
            **kwargs,
        )

    async def test_feedback_replan_enforcement_and_durable_audit(self) -> None:
        actor = CapturedModel(
            [
                lambda r: revise(r, status="completed"),
                AssistantMessage("Finished, all requirements passed"),
                lambda r: revise(r),
                AssistantMessage(tool_calls=(ToolCall("write", "write_artifact", {}),)),
                lambda r: revise(r, status="completed"),
                AssistantMessage("The artifact has been created"),
            ]
        )
        store = JsonlSessionStore(self.root / "sessions")
        planner_model = CapturedModel(
            [
                AssistantMessage("private malformed planner response"),
                AssistantMessage(json.dumps(proposal())),
            ]
        )
        harness = self.harness(
            actor,
            store=store,
            planner=ModelTaskPlanner(planner_model, capabilities=(capability(),)),
        )
        async with harness:
            outcome = await harness.run("Create the artifact")
        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        preparation = next(
            r for r in outcome.audit_records if r.kind == "planning_usage"
        )
        self.assertEqual(preparation.payload["model_requests"], 2)
        self.assertEqual(preparation.payload["usage"]["total_tokens"], 30)
        self.assertIn("planner:output_format", str(planner_model.requests[1].messages))
        for value in (
            str(actor.requests),
            str(harness.messages),
            str(outcome.audit_records),
        ):
            self.assertNotIn("planner:output_format", value)
            self.assertNotIn("private malformed planner response", value)
        self.assertEqual(harness.last_execution_plan.version, 4)
        self.assertEqual(
            harness.last_execution_plan.steps[0].status, StepStatus.COMPLETED
        )
        rejected = [r for r in outcome.audit_records if r.kind == "completion_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertTrue(
            any(
                isinstance(m, TransientInstruction) and m.source == "completion_audit"
                for m in actor.requests[2].messages
            )
        )
        self.assertNotIn(
            "Finished, all requirements passed",
            [getattr(m, "content", None) for m in harness.messages],
        )
        context = task_context(actor.requests[2])
        self.assertEqual(context["execution_plan"]["version"], 2)
        self.assertEqual(context["goal"], "The requested artifact exists")
        trajectory = next(
            json.loads(m.content)["trajectory_context"]
            for m in actor.requests[2].messages
            if isinstance(m, TransientInstruction)
            and m.source.startswith("trajectory:")
        )
        self.assertEqual(json.loads(trajectory["revisable_plan"])["version"], 2)
        self.assertFalse(trajectory["requirements"]["artifact"])
        self.assertEqual(
            [r.sequence for r in outcome.audit_records],
            list(range(1, len(outcome.audit_records) + 1)),
        )
        self.assertEqual(
            len(
                [r for r in outcome.audit_records if r.kind == "execution_plan_updated"]
            ),
            3,
        )
        self.assertEqual(self.evaluator.active_run_ids, ())
        persisted = "\n".join(
            p.read_text() for p in (self.root / "sessions").rglob("*.jsonl")
        )
        self.assertIn("execution_plan_updated", persisted)
        self.assertIn("task_planned", persisted)
        self.assertNotIn("planner:output_format", persisted)
        self.assertNotIn("private malformed planner response", persisted)
        self.assertEqual(actor.starts, 1)
        self.assertEqual(actor.stops, 1)

    async def test_exhausted_output_recovery_never_starts_actor_or_commits_task(
        self,
    ) -> None:
        actor = CapturedModel([])
        planner_model = CapturedModel([AssistantMessage("bad JSON")] * 2)
        async with self.harness(
            actor,
            planner=ModelTaskPlanner(planner_model, capabilities=(capability(),)),
        ) as harness:
            outcome = await harness.run("Create the artifact")
        self.assertFalse(outcome.result.succeeded)
        self.assertEqual(actor.requests, [])
        self.assertFalse(self.path.exists())
        self.assertEqual(harness.revision, 0)
        self.assertIsNone(harness.last_task_definition)
        preparation = next(
            r for r in outcome.audit_records if r.kind == "planning_usage"
        )
        self.assertEqual(preparation.payload["model_requests"], 2)
        self.assertEqual(preparation.payload["usage"]["total_tokens"], 30)

    async def test_invalid_updates_leave_plan_unchanged(self) -> None:
        actor = CapturedModel(
            [
                lambda r: revise(r, version_delta=1),
                lambda r: revise(r, checkpoint="other-run:cp0"),
                lambda r: revise(r, extra=True),
                AssistantMessage(tool_calls=(ToolCall("write", "write_artifact", {}),)),
                AssistantMessage("Done"),
            ]
        )
        async with self.harness(actor) as harness:
            outcome = await harness.run("Create the artifact")
        self.assertTrue(outcome.result.succeeded)
        self.assertEqual(harness.last_execution_plan.version, 1)
        self.assertEqual(
            len(
                [
                    r
                    for r in outcome.audit_records
                    if r.kind == "execution_plan_rejected"
                ]
            ),
            3,
        )

    async def test_planning_failure_has_no_actor_actions_or_conversation_commit(
        self,
    ) -> None:
        unsupported = proposal()
        unsupported["unsupported"] = ["No suitable evidence source"]
        actor = CapturedModel([])
        async with self.harness(actor, planner=planner(unsupported)) as harness:
            outcome = await harness.run("Deploy a service")
        self.assertEqual(outcome.result.status, RunStatus.FAILED)
        self.assertEqual(outcome.result.turns, 0)
        self.assertEqual(actor.requests, [])
        self.assertEqual(harness.revision, 0)
        self.assertEqual(outcome.failure.phase.value, "preparation")
        self.assertIn("planning_failed", [r.kind for r in outcome.audit_records])

    async def test_unbound_verifier_rejected_before_execution(self) -> None:
        adapter = ModelTaskPlanner(
            CapturedModel([AssistantMessage(json.dumps(proposal()))]),
            capabilities=(
                VerificationCapability(
                    "artifact_exists",
                    EvaluationCriterion(
                        "x", "Check artifact", "not_registered", ("artifact",)
                    ),
                ),
            ),
        )
        actor = CapturedModel([])
        async with self.harness(actor, planner=adapter) as harness:
            result = await harness.run("query")
        self.assertFalse(result.result.succeeded)
        self.assertIn("not_registered", result.failure.message)
        self.assertEqual(actor.requests, [])

    async def test_cancel_during_planning_is_cooperative(self) -> None:
        started = asyncio.Event()

        class BlockingPlanner:
            async def plan(self, request, *, cancellation):
                started.set()
                await asyncio.Event().wait()

        actor = CapturedModel([])
        async with self.harness(actor, planner=BlockingPlanner()) as harness:
            running = asyncio.create_task(harness.run("query"))
            await started.wait()
            self.assertTrue(harness.cancel("stop preparation"))
            outcome = await asyncio.wait_for(running, 1)
        self.assertEqual(outcome.result.status, RunStatus.CANCELLED)
        self.assertEqual(actor.requests, [])

    async def test_two_updates_against_one_version_commit_only_once(self) -> None:
        def competing(request):
            first = revise(request).tool_calls[0]
            return AssistantMessage(
                tool_calls=(
                    ToolCall("plan-a", "update_plan", first.arguments),
                    ToolCall("plan-b", "update_plan", first.arguments),
                )
            )

        actor = CapturedModel(
            [
                competing,
                AssistantMessage(tool_calls=(ToolCall("write", "write_artifact", {}),)),
                AssistantMessage("Done"),
            ]
        )
        async with self.harness(actor) as harness:
            result = await harness.run("Create artifact")
        self.assertTrue(result.result.succeeded)
        self.assertEqual(harness.last_execution_plan.version, 2)
        self.assertEqual(
            len(
                [r for r in result.audit_records if r.kind == "execution_plan_updated"]
            ),
            1,
        )
        self.assertEqual(
            len(
                [r for r in result.audit_records if r.kind == "execution_plan_rejected"]
            ),
            1,
        )

    async def test_revision_cannot_remove_acceptance_coverage(self) -> None:
        def drop_requirement(request):
            context = task_context(request)
            return AssistantMessage(
                tool_calls=(
                    ToolCall(
                        "drop",
                        "update_plan",
                        {
                            "expected_version": 1,
                            "based_on_checkpoint": context["latest_checkpoint"],
                            "reason": "Skip a hard requirement",
                            "steps": [
                                {
                                    "id": "skip",
                                    "description": "Do something else",
                                    "requirement_ids": ["other"],
                                    "status": "completed",
                                }
                            ],
                        },
                    ),
                )
            )

        actor = CapturedModel(
            [
                drop_requirement,
                AssistantMessage(tool_calls=(ToolCall("write", "write_artifact", {}),)),
                AssistantMessage("Done"),
            ]
        )
        async with self.harness(actor) as harness:
            result = await harness.run("Create artifact")
        self.assertTrue(result.result.succeeded)
        self.assertEqual(harness.last_execution_plan.version, 1)
        record = next(
            r for r in result.audit_records if r.kind == "execution_plan_rejected"
        )
        self.assertIn("cover all requirements", record.payload["reason"])

    async def test_reported_planner_cost_survives_binding_failure(self) -> None:
        adapter = ModelTaskPlanner(
            CapturedModel([AssistantMessage(json.dumps(proposal()))]),
            capabilities=(
                VerificationCapability(
                    "artifact_exists",
                    EvaluationCriterion(
                        "x", "Missing method", "missing", ("artifact",)
                    ),
                ),
            ),
        )
        async with self.harness(CapturedModel([]), planner=adapter) as harness:
            result = await harness.run("query")
        usage = next(r for r in result.audit_records if r.kind == "planning_usage")
        self.assertEqual(usage.payload["usage"]["total_tokens"], 15)
        self.assertEqual(usage.payload["model_requests"], 1)

    async def test_new_queries_get_fresh_plans_and_no_old_steps(self) -> None:
        model = CapturedModel(
            [
                AssistantMessage(json.dumps(proposal("First task"))),
                AssistantMessage(json.dumps(proposal("Second task"))),
            ]
        )
        actor = CapturedModel(
            [
                revise,
                AssistantMessage(tool_calls=(ToolCall("write", "write_artifact", {}),)),
                AssistantMessage("Done"),
                AssistantMessage("Already exists"),
            ]
        )
        run_ids = iter(("first", "second"))
        harness = AgentHarness(
            agent_id="fresh-plans",
            model=actor,
            tools=self.tools,
            planner=ModelTaskPlanner(model, capabilities=(capability(),)),
            trajectory=self.monitor,
            context=self.monitor.context_pipeline(),
            completion_policy=CompletionPolicy(CompletionMode.ENFORCE),
            run_id_factory=lambda: next(run_ids),
        )
        async with harness:
            await harness.run("Create first artifact")
            self.assertEqual(harness.last_execution_plan.version, 2)
            result = await harness.run("Check it again")
        self.assertTrue(result.result.succeeded)
        self.assertEqual(harness.last_task_definition.task, "Second task")
        self.assertEqual(harness.last_execution_plan.version, 1)
        context = task_context(actor.requests[-1])
        self.assertEqual(context["task"], "Second task")
        self.assertEqual(context["execution_plan"]["version"], 1)
        self.assertEqual(context["latest_checkpoint"], "second:cp0")
