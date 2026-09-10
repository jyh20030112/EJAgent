"""Add a real Python feature in an isolated workspace with inspectable contexts.

Default: configured LLM for both planning and execution. --demo uses deterministic
model responses but still edits files, runs Python tests, and enforces completion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from ejagent.contracts import (
    AssistantMessage,
    CancellationToken,
    CompletionMode,
    CompletionPolicy,
    ContextMessage,
    EvaluationCriterion,
    JsonObject,
    ManagedResource,
    ModelPort,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelUsage,
    RunLimits,
    RunOutcome,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolResultMessage,
    TransientInstruction,
    UserMessage,
    thaw_json_value,
)
from ejagent.evaluation import (
    CommandEvidenceSource,
    EvaluationMonitor,
    GoalEvaluator,
    JsonlEvaluationJournal,
    WorkspaceEvidenceSource,
    command_succeeded,
)
from ejagent.harness import AgentHarness
from ejagent.planning import ModelTaskPlanner, VerificationCapability
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.storage import JsonlSessionStore
from ejagent.tools import FunctionTool, FunctionToolExecutor

QUERY = (
    "Add overlap_seconds(a_start, a_end, b_start, b_end) to intervals.py. "
    "It must return the positive overlap duration, or 0 for disjoint, touching, "
    "or empty intervals. Preserve overlapped(). Pass the provided regression tests."
)
INITIAL_CODE = """def overlapped(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)
"""
FIXED_CODE = (
    INITIAL_CODE
    + """

def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))
"""
)
REGRESSION_TESTS = """import unittest
import intervals

class TestIntervals(unittest.TestCase):
    def test_existing_boolean_semantics(self):
        self.assertTrue(intervals.overlapped(0, 5, 2, 8))
        self.assertFalse(intervals.overlapped(0, 2, 2, 8))
        self.assertFalse(intervals.overlapped(0, 1, 2, 8))

    def test_positive_duration(self):
        self.assertEqual(intervals.overlap_seconds(0, 5, 2, 8), 3)
        self.assertEqual(intervals.overlap_seconds(1, 5, 2, 3), 1)

    def test_no_overlap(self):
        self.assertEqual(intervals.overlap_seconds(0, 1, 2, 8), 0)
        self.assertEqual(intervals.overlap_seconds(0, 2, 2, 8), 0)
        self.assertEqual(intervals.overlap_seconds(2, 2, 0, 8), 0)
"""


def _message_payload(message: ContextMessage) -> dict[str, object]:
    payload: dict[str, object] = {"type": type(message).__name__}
    if isinstance(message, ToolResultMessage):
        payload.update(
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            result=thaw_json_value(message.result),
            is_error=message.is_error,
        )
    else:
        payload["content"] = message.content
    if isinstance(message, TransientInstruction):
        payload["source"] = message.source
    if isinstance(message, AssistantMessage):
        payload["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": thaw_json_value(c.arguments)}
            for c in message.tool_calls
        ]
    return payload


class RecordingModel:
    def __init__(self, model: ModelPort, path: Path) -> None:
        self.model = model
        self.path = path
        self.requests: list[ModelRequest] = []

    async def start(self) -> None:
        if isinstance(self.model, ManagedResource):
            await self.model.start()

    async def shutdown(self) -> None:
        if isinstance(self.model, ManagedResource):
            await self.model.shutdown()

    async def stream(
        self, request: ModelRequest, *, cancellation: CancellationToken
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "request": len(self.requests),
                        "messages": [_message_payload(m) for m in request.messages],
                        "tools": [
                            {
                                "name": t.name,
                                "description": t.description,
                                "input_schema": thaw_json_value(t.input_schema),
                            }
                            for t in request.tools
                        ],
                        "response_format": thaw_json_value(request.response_format),
                        "max_output_tokens": request.max_output_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        async for event in self.model.stream(request, cancellation=cancellation):
            yield event


class DemoFeaturePlanner:
    """Fixed fixture strategy for offline tests, not a general-purpose planner."""

    async def stream(
        self, request: ModelRequest, *, cancellation: CancellationToken
    ) -> AsyncIterator[ModelStreamEvent]:
        message = request.messages[-1]
        assert isinstance(message, UserMessage)
        query = json.loads(message.content)["query"]
        value = {
            "task": query,
            "goal": "Add correct overlap duration while preserving the boolean function",
            "criteria": [
                {
                    "id": "regression",
                    "capability": "regression_tests",
                    "description": "The provided interval tests pass",
                }
            ],
            "steps": [
                {
                    "id": "implement",
                    "description": "Implement the feature and verify against tests",
                    "requirement_ids": ["regression"],
                    "status": "pending",
                }
            ],
            "unknowns": [],
            "unsupported": [],
        }
        yield ModelResponseCompleted(
            AssistantMessage(json.dumps(value)), ModelUsage(10, 10, 20), "stop"
        )


class DemoFeatureActor:
    """Deliberately miss a boundary case, then repair after real test feedback."""

    def __init__(self) -> None:
        self.turn = 0

    async def stream(
        self, request: ModelRequest, *, cancellation: CancellationToken
    ) -> AsyncIterator[ModelStreamEvent]:
        self.turn += 1
        context = next(
            json.loads(m.content)["task_context"]
            for m in request.messages
            if isinstance(m, TransientInstruction) and m.source == "planning"
        )
        if self.turn in (1, 4, 6):
            plan = context["execution_plan"]
            plan["steps"][0]["status"] = (
                "completed" if self.turn == 6 else "in_progress"
            )
            plan["steps"][0]["description"] = (
                "Clamp non-overlap duration to zero and rerun tests"
                if self.turn == 4
                else plan["steps"][0]["description"]
            )
            reply = AssistantMessage(
                tool_calls=(
                    ToolCall(
                        f"plan-{self.turn}",
                        "update_plan",
                        {
                            "expected_version": plan["version"],
                            "based_on_checkpoint": context["latest_checkpoint"],
                            "reason": "Boundary-case failure requires a correction"
                            if self.turn == 4
                            else "Advance the execution plan using current evidence",
                            "steps": plan["steps"],
                        },
                    ),
                )
            )
        elif self.turn in (2, 5):
            code = (
                FIXED_CODE
                if self.turn == 5
                else INITIAL_CODE
                + "\ndef overlap_seconds(a_start, a_end, b_start, b_end):\n    return min(a_end, b_end) - max(a_start, b_start)\n"
            )
            reply = AssistantMessage(
                tool_calls=(
                    ToolCall(f"write-{self.turn}", "write_feature", {"content": code}),
                )
            )
        else:
            reply = AssistantMessage(
                "Added overlap_seconds; interval regression tests pass."
            )
        yield ModelResponseCompleted(reply, ModelUsage(10, 10, 20), "stop")


async def run_feature(
    root: Path, *, actor: ModelPort, planner_model: ModelPort, query: str = QUERY
) -> RunOutcome:
    """Use a new directory so rerunning never overwrites an existing work product."""
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
    source = root / "intervals.py"
    tests = root / "test_intervals.py"
    for path, content in ((source, INITIAL_CODE), (tests, REGRESSION_TESTS)):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    workspace = WorkspaceEvidenceSource(root, (source.name, tests.name))
    command = CommandEvidenceSource(
        workspace,
        (
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            ".",
            "-p",
            "test_intervals.py",
            "-v",
        ),
    )

    async def read(
        call: ToolCall, cancellation: CancellationToken
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        return ToolExecutionResult(
            {source.name: source.read_text(), tests.name: tests.read_text()}
        )

    async def write(
        call: ToolCall, cancellation: CancellationToken
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        text = call.arguments.get("content")
        if (
            set(call.arguments) != {"content"}
            or not isinstance(text, str)
            or len(text.encode()) > 32_768
        ):
            return ToolExecutionResult(
                {}, error="content must be a string of at most 32768 bytes"
            )
        source.write_text(text, encoding="utf-8")
        return ToolExecutionResult(
            {
                "written": source.name,
                "verification": "The checkpoint will run host tests against the new file version.",
            }
        )

    tools = FunctionToolExecutor(
        (
            FunctionTool(
                ToolDefinition(
                    "read_workspace",
                    "Read source and host regression tests",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                ),
                read,
            ),
            FunctionTool(
                ToolDefinition(
                    "write_feature",
                    "Replace intervals.py; host tests are not writable through this tool",
                    {
                        "type": "object",
                        "required": ("content",),
                        "properties": {"content": {"type": "string"}},
                        "additionalProperties": False,
                    },
                ),
                write,
            ),
        )
    )
    monitor = EvaluationMonitor(
        GoalEvaluator(
            sources={"tests": command},
            verifiers={"command": command_succeeded},
            timeout_seconds=35,
            report_sink=JsonlEvaluationJournal(root / "evaluations.jsonl"),
        )
    )
    recorded_actor = RecordingModel(actor, root / "actor-contexts.jsonl")
    recorded_planner = RecordingModel(planner_model, root / "planner-contexts.jsonl")
    planner = ModelTaskPlanner(
        recorded_planner,
        capabilities=(
            VerificationCapability(
                "regression_tests",
                EvaluationCriterion(
                    "template",
                    "The provided interval tests pass; this does not verify goals beyond their scope",
                    "command",
                    ("tests",),
                ),
                required=True,
            ),
        ),
        environment={
            "files": {source.name: INITIAL_CODE, tests.name: REGRESSION_TESTS},
            "scope": "Only interval overlap duration and existing overlapped compatibility can be verified here.",
        },
    )
    async with AgentHarness(
        agent_id="planned-feature",
        model=recorded_actor,
        tools=tools,
        planner=planner,
        trajectory=monitor,
        context=monitor.context_pipeline(),
        completion_policy=CompletionPolicy(CompletionMode.ENFORCE, max_retries=3),
        initial_messages=(
            SystemMessage(
                "Implement the user's feature in the supplied workspace. Review the current plan and checkpoint before each action. Update plan status through update_plan. Tests run automatically at checkpoints. Only claim completion when host tests pass."
            ),
        ),
        store=JsonlSessionStore(root / "sessions"),
        limits=RunLimits(max_turns=16),
    ) as harness:
        outcome = await harness.run(query)
        summary: JsonObject = {
            "status": outcome.result.status.value,
            "output": outcome.result.output,
            "failure": outcome.failure.message if outcome.failure else None,
            "actor_requests": len(recorded_actor.requests),
            "planner_requests": len(recorded_planner.requests),
            "final_plan": harness.last_execution_plan.to_dict()
            if harness.last_execution_plan
            else None,
            "task": harness.last_task_definition.to_dict()
            if harness.last_task_definition
            else None,
        }
    (root / "result.json").write_text(
        json.dumps(thaw_json_value(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Feature execution contexts",
        "",
        f"Result: {outcome.result.status.value}",
        "",
        "Full request messages and tool schemas are in actor-contexts.jsonl and planner-contexts.jsonl.",
    ]
    for index, request in enumerate(recorded_actor.requests, 1):
        lines.extend(("", f"## Actor request {index}"))
        for message in request.messages:
            if isinstance(message, TransientInstruction):
                lines.extend(
                    (
                        "",
                        f"### {message.source}",
                        "",
                        "```json",
                        json.dumps(
                            json.loads(message.content), ensure_ascii=False, indent=2
                        ),
                        "```",
                    )
                )
    (root / "contexts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use scripted model replies with real file edits and tests",
    )
    parser.add_argument("--query", default=QUERY)
    parser.add_argument("--output", type=Path, help="New output/workspace directory")
    args = parser.parse_args()
    if args.output is None:
        parent = Path(".ejagent-sessions")
        parent.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="planned-feature-", dir=parent)).resolve()
    else:
        root = args.output.resolve()
    actor: ModelPort
    planner_model: ModelPort
    if args.demo:
        actor, planner_model = DemoFeatureActor(), DemoFeaturePlanner()
    else:
        config = ModelConfig.from_env()
        actor, planner_model = OpenAIModelPort(config), OpenAIModelPort(config)
    result = asyncio.run(
        run_feature(root, actor=actor, planner_model=planner_model, query=args.query)
    )
    print(f"{result.result.status.value}: {root / 'contexts.md'}", flush=True)
    if result.failure:
        print(result.failure.message, flush=True)
    if not result.result.succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
