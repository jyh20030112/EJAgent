from __future__ import annotations

import unittest
from typing import Any

from test_anthropic_provider import (
    FakeAnthropicClient,
    FakeAnthropicMessages,
    _anthropic_config,
    _text_events,
)
from test_core_adapters import (
    FakeCompletions,
    FakeOpenAIClient,
    FakeOpenAIStream,
    _config,
    _delta,
)

from ejagent.contracts import (
    AssistantMessage,
    CancellationSource,
    ModelCallError,
    ModelRequest,
    ModelResponseCompleted,
    UserMessage,
)
from ejagent.providers import AnthropicModelPort, OpenAIModelPort


class TestModelOutputFormat(unittest.IsolatedAsyncioTestCase):
    async def test_openai_passes_response_format_without_mutating_request(self) -> None:
        formats: list[Any] = [
            None,
            {"type": "json_object"},
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "verdict",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        for value in formats:
            with self.subTest(value=value):
                completions = FakeCompletions(
                    [FakeOpenAIStream([_delta(content="{}", finish_reason="stop")])]
                )
                port = OpenAIModelPort(_config(), client=FakeOpenAIClient(completions))
                await port.start()
                request = ModelRequest(
                    (UserMessage("Return JSON"),), response_format=value
                )
                events = [
                    event
                    async for event in port.stream(
                        request, cancellation=CancellationSource().token
                    )
                ]
                self.assertEqual(events[-1].finish_reason, "stop")
                if value is None:
                    self.assertNotIn("response_format", completions.requests[0])
                else:
                    sent = completions.requests[0]["response_format"]
                    self.assertEqual(sent, value)
                    sent["type"] = "text"
                    self.assertEqual(request.response_format["type"], value["type"])
                await port.shutdown()

    async def test_openai_preserves_truncation_and_filter_finish_reasons(self) -> None:
        for reason in ("length", "content_filter"):
            completions = FakeCompletions(
                [FakeOpenAIStream([_delta(content="{", finish_reason=reason)])]
            )
            port = OpenAIModelPort(_config(), client=FakeOpenAIClient(completions))
            await port.start()
            events = [
                event
                async for event in port.stream(
                    ModelRequest((UserMessage("JSON"),)),
                    cancellation=CancellationSource().token,
                )
            ]
            self.assertEqual(events[-1].finish_reason, reason)
            await port.shutdown()

    async def test_anthropic_preserves_max_tokens_finish_reason(self) -> None:
        events = _text_events("{")
        for event in events:
            if event["type"] == "message_delta":
                event["delta"]["stop_reason"] = "max_tokens"
        port = AnthropicModelPort(
            _anthropic_config(),
            client=FakeAnthropicClient(FakeAnthropicMessages([events])),
        )
        await port.start()
        responses = [
            event
            async for event in port.stream(
                ModelRequest((UserMessage("JSON"),)),
                cancellation=CancellationSource().token,
            )
        ]
        self.assertEqual(responses[-1].finish_reason, "max_tokens")
        await port.shutdown()

    async def test_anthropic_rejects_unsupported_passthrough_before_request(
        self,
    ) -> None:
        messages = FakeAnthropicMessages([])
        port = AnthropicModelPort(
            _anthropic_config(), client=FakeAnthropicClient(messages)
        )
        await port.start()
        with self.assertRaisesRegex(ModelCallError, "response_format"):
            async for _ in port.stream(
                ModelRequest(
                    (UserMessage("JSON"),), response_format={"type": "json_object"}
                ),
                cancellation=CancellationSource().token,
            ):
                pass
        self.assertEqual(messages.requests, [])
        await port.shutdown()

    def test_finish_reason_and_response_format_validation(self) -> None:
        for reason in ("", " ", 1):
            with self.assertRaises(ValueError):
                ModelResponseCompleted(AssistantMessage("{}"), finish_reason=reason)
        for value in ([], "json", {"schema": float("nan")}):
            with self.assertRaises((TypeError, ValueError)):
                ModelRequest((UserMessage("JSON"),), response_format=value)
