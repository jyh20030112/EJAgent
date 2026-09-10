from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from ejagent.contracts import (
    AssistantMessage,
    CancellationToken,
    CompletionMode,
    CompletionPolicy,
    EvaluationCriterion,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelUsage,
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
        completed = proposal()
        completed["steps"][0]["status"] = "completed"
        cases.append(completed)
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
        harness = self.harness(actor, store=store)
        async with harness:
            outcome = await harness.run("Create the artifact")
        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
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
        self.assertEqual(actor.starts, 1)
        self.assertEqual(actor.stops, 1)

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
