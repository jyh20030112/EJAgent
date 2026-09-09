from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ejagent.context import IdentityContextPipeline, SkillsContextPipeline
from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    ContextRequest,
    ContextSummary,
    FailureCode,
    ModelCallError,
    ModelProtocolError,
    ModelRequest,
    ModelResponseCompleted,
    ModelTextDelta,
    ModelThinkingDelta,
    RunStatus,
    SystemMessage,
    ToolCall,
    ToolControl,
    ToolDefinition,
    ToolExecutionError,
    ToolExecutionResult,
    ToolProtocolError,
    ToolResultMessage,
    TransientInstruction,
    UserMessage,
)
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import (
    CompositeToolExecutor,
    FunctionTool,
    FunctionToolExecutor,
    McpToolExecutor,
)


def _delta(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    tool_calls: Sequence[Any] = (),
    reasoning: str | None = None,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                ),
            )
        ],
        usage=None,
    )


def _tool_fragment(
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeOpenAIStream:
    def __init__(self, chunks: Sequence[Any]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> FakeOpenAIStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.closed = True


class FakeCompletions:
    def __init__(
        self,
        responses: Sequence[FakeOpenAIStream | Exception],
    ) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeOpenAIStream:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeOpenAIClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _config() -> ModelConfig:
    return ModelConfig(
        model="test-model",
        api_key="test-key",
        base_url="https://model.invalid/v1",
    )


class OpenAIModelPortTests(unittest.IsolatedAsyncioTestCase):
    async def test_serializes_typed_request_and_normalizes_stream(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=7,
            completion_tokens=3,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=2,
                cache_write_tokens=None,
            ),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
        )
        stream = FakeOpenAIStream(
            [
                _delta(content="答", reasoning="想"),
                _delta(content="案", finish_reason="stop"),
                SimpleNamespace(choices=[], usage=usage),
            ]
        )
        completions = FakeCompletions([stream])
        port = OpenAIModelPort(
            _config(),
            client=FakeOpenAIClient(completions),  # type: ignore[arg-type]
        )
        await port.start()
        request = ModelRequest(
            max_output_tokens=128,
            messages=(
                SystemMessage("system"),
                UserMessage("你好"),
                AssistantMessage(
                    tool_calls=(ToolCall("call-0", "lookup", {"q": "值"}),)
                ),
                ToolResultMessage("call-0", "lookup", {"found": True}),
                ContextSummary(1, 2, "summary", "compact-v1"),
                TransientInstruction("focus", "steering"),
            ),
            tools=(
                ToolDefinition(
                    name="lookup",
                    description="Lookup a value.",
                    input_schema={"type": "object"},
                ),
            ),
        )

        events = [
            event
            async for event in port.stream(
                request,
                cancellation=CancellationSource().token,
            )
        ]

        self.assertEqual(events[0], ModelTextDelta("答"))
        self.assertEqual(events[1], ModelThinkingDelta("想"))
        self.assertEqual(events[2], ModelTextDelta("案"))
        completed = events[3]
        self.assertIsInstance(completed, ModelResponseCompleted)
        assert isinstance(completed, ModelResponseCompleted)
        self.assertEqual(completed.message, AssistantMessage("答案"))
        self.assertEqual(completed.usage.input_tokens, 7)  # type: ignore[union-attr]
        sent = completions.requests[0]
        self.assertEqual(sent["max_completion_tokens"], 128)
        self.assertEqual(sent["messages"][1], {"role": "user", "content": "你好"})
        self.assertEqual(
            sent["messages"][2]["tool_calls"][0]["function"]["arguments"],
            '{"q":"值"}',
        )
        self.assertEqual(sent["messages"][3]["content"], '{"found":true}')
        self.assertIn("Derived summary", sent["messages"][4]["content"])
        self.assertIn("Transient instruction", sent["messages"][5]["content"])
        self.assertEqual(
            sent["tools"][0]["function"]["parameters"],
            {"type": "object"},
        )
        self.assertTrue(stream.closed)

    async def test_reassembles_and_decodes_streamed_tool_call(self) -> None:
        stream = FakeOpenAIStream(
            [
                _delta(
                    tool_calls=(
                        _tool_fragment(
                            call_id="call-1",
                            name="weather",
                            arguments='{"city":',
                        ),
                    )
                ),
                _delta(
                    finish_reason="tool_calls",
                    tool_calls=(_tool_fragment(arguments='"杭州"}'),),
                ),
            ]
        )
        port = OpenAIModelPort(
            _config(),
            client=FakeOpenAIClient(FakeCompletions([stream])),  # type: ignore[arg-type]
        )
        await port.start()

        events = [
            event
            async for event in port.stream(
                ModelRequest(messages=(UserMessage("weather"),)),
                cancellation=CancellationSource().token,
            )
        ]

        self.assertEqual(
            events,
            [
                ModelResponseCompleted(
                    AssistantMessage(
                        tool_calls=(ToolCall("call-1", "weather", {"city": "杭州"}),)
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

    async def test_normalizes_expected_provider_failure(self) -> None:
        port = OpenAIModelPort(
            _config(),
            client=FakeOpenAIClient(  # type: ignore[arg-type]
                FakeCompletions([TimeoutError("late")])
            ),
        )
        await port.start()

        with self.assertRaises(ModelCallError) as raised:
            async for _ in port.stream(
                ModelRequest(messages=(UserMessage("hello"),)),
                cancellation=CancellationSource().token,
            ):
                pass

        self.assertEqual(raised.exception.code, FailureCode.TIMEOUT)
        self.assertTrue(raised.exception.retryable)

    async def test_rejects_invalid_tool_argument_json_as_protocol_error(self) -> None:
        stream = FakeOpenAIStream(
            [
                _delta(
                    finish_reason="tool_calls",
                    tool_calls=(
                        _tool_fragment(
                            call_id="call-1",
                            name="lookup",
                            arguments="not-json",
                        ),
                    ),
                )
            ]
        )
        port = OpenAIModelPort(
            _config(),
            client=FakeOpenAIClient(FakeCompletions([stream])),  # type: ignore[arg-type]
        )
        await port.start()

        with self.assertRaises(ModelProtocolError):
            async for _ in port.stream(
                ModelRequest(messages=(UserMessage("lookup"),)),
                cancellation=CancellationSource().token,
            ):
                pass


class FakeMcpManager:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def startup(self) -> None:
        self.events.append("start")

    async def shutdown(self) -> None:
        self.events.append("shutdown")

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "docs__search",
                    "description": "Search docs.",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ]

    async def call_tool(self, tool_name: str, args: dict[str, object]) -> str:
        self.calls.append((tool_name, args))
        return "found"


class ToolExecutorAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_function_executor_dispatches_normalized_result(self) -> None:
        async def add(
            call: ToolCall,
            cancellation: Any,
        ) -> ToolExecutionResult:
            cancellation.raise_if_cancelled()
            return ToolExecutionResult(
                {"value": call.arguments["left"] + call.arguments["right"]},
                control=ToolControl.COMPLETE,
                output="42",
            )

        executor = FunctionToolExecutor(
            (
                FunctionTool(
                    ToolDefinition(
                        "add",
                        input_schema={"type": "object"},
                    ),
                    add,
                ),
            )
        )

        result = await executor.execute(
            ToolCall("call-1", "add", {"left": 19, "right": 23}),
            cancellation=CancellationSource().token,
        )

        self.assertEqual(result.result["value"], 42)  # type: ignore[index]
        self.assertEqual(result.control, ToolControl.COMPLETE)
        with self.assertRaises(ToolExecutionError):
            await executor.execute(
                ToolCall("call-2", "missing"),
                cancellation=CancellationSource().token,
            )

    async def test_function_executor_rejects_invalid_return_type(self) -> None:
        async def invalid(call: ToolCall, cancellation: Any) -> Any:
            return {"not": "normalized"}

        executor = FunctionToolExecutor(
            (FunctionTool(ToolDefinition("invalid"), invalid),)
        )
        with self.assertRaises(ToolProtocolError):
            await executor.execute(
                ToolCall("call-1", "invalid"),
                cancellation=CancellationSource().token,
            )

    async def test_mcp_executor_owns_lifecycle_and_normalizes_metadata(self) -> None:
        manager = FakeMcpManager()
        executor = McpToolExecutor(manager=manager)

        await executor.start()
        result = await executor.execute(
            ToolCall("call-1", "docs__search", {"q": "kernel"}),
            cancellation=CancellationSource().token,
        )
        await executor.shutdown()

        self.assertEqual(result, ToolExecutionResult("found"))
        self.assertEqual(manager.calls, [("docs__search", {"q": "kernel"})])
        self.assertEqual(manager.events, ["start", "shutdown"])
        self.assertEqual(executor.definitions, ())

    async def test_composite_refreshes_dynamic_routes(self) -> None:
        manager = FakeMcpManager()
        mcp = McpToolExecutor(manager=manager)
        local = FunctionToolExecutor()
        composite = CompositeToolExecutor((local, mcp))

        self.assertEqual(composite.definitions, ())
        await composite.start()
        self.assertEqual(
            tuple(item.name for item in composite.definitions),
            ("docs__search",),
        )
        await composite.shutdown()
        self.assertEqual(composite.definitions, ())


class SkillsContextPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        skill = self.root / "release_notes"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\nname: release_notes\ndescription: Write releases.\n---\n"
            "# Instructions\nUse concise bullets.\n",
            encoding="utf-8",
        )

    async def test_injects_index_and_explicit_skill_only_into_context(self) -> None:
        pipeline = SkillsContextPipeline(
            self.root,
            base=IdentityContextPipeline(),
        )
        await pipeline.start()
        request = ContextRequest(
            run_id="run-1",
            source_revision=0,
            turn=1,
            committed_messages=(SystemMessage("system"),),
            pending_messages=(UserMessage("$release_notes draft this"),),
        )

        view = await pipeline.build(
            request,
            cancellation=CancellationSource().token,
        )
        await pipeline.shutdown()

        self.assertEqual(request.transient_instructions, ())
        additions = [
            item for item in view.messages if isinstance(item, TransientInstruction)
        ]
        self.assertEqual(
            tuple(item.source for item in additions),
            ("skills:index", "skills:release_notes"),
        )
        self.assertIn("Use concise bullets", additions[1].content)
        self.assertEqual(view.metadata["selected_skill"], "release_notes")


class CoreAdapterIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_runs_openai_tool_round_trip(self) -> None:
        first = FakeOpenAIStream(
            [
                _delta(
                    finish_reason="tool_calls",
                    tool_calls=(
                        _tool_fragment(
                            call_id="call-add",
                            name="add",
                            arguments='{"left":20,"right":22}',
                        ),
                    ),
                )
            ]
        )
        second = FakeOpenAIStream([_delta(content="42", finish_reason="stop")])
        completions = FakeCompletions([first, second])
        model = OpenAIModelPort(
            _config(),
            client=FakeOpenAIClient(completions),  # type: ignore[arg-type]
        )

        async def add(call: ToolCall, cancellation: Any) -> ToolExecutionResult:
            return ToolExecutionResult(
                {
                    "value": call.arguments["left"] + call.arguments["right"],
                }
            )

        tools = FunctionToolExecutor((FunctionTool(ToolDefinition("add"), add),))
        harness = AgentHarness(
            agent_id="calculator",
            model=model,
            tools=tools,
            initial_messages=(SystemMessage("Use tools."),),
            run_id_factory=lambda: "run-1",
        )

        async with harness:
            outcome = await harness.run("add 20 and 22")

        self.assertEqual(outcome.result.status, RunStatus.COMPLETED)
        self.assertEqual(outcome.result.output, "42")
        self.assertEqual(harness.revision, 1)
        self.assertEqual(len(completions.requests), 2)
        self.assertEqual(
            completions.requests[1]["messages"][-1],
            {
                "role": "tool",
                "tool_call_id": "call-add",
                "content": '{"value":42}',
            },
        )
