from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from ejagent.contracts.control import CancellationToken, RunCancelledError
from ejagent.contracts.json import JsonValue, thaw_json_value
from ejagent.contracts.messages import (
    AssistantMessage,
    ContextMessage,
    ContextSummary,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    TransientInstruction,
    UserMessage,
)
from ejagent.contracts.model import (
    ModelCallError,
    ModelPort,
    ModelProtocolError,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelThinkingDelta,
    ModelUsage,
)
from ejagent.contracts.runs import FailureCode
from ejagent.contracts.tools import ToolDefinition
from ejagent.providers.anthropic_config import AnthropicConfig

_CONTEXT_OVERFLOW_PHRASES = (
    "prompt is too long",
    "request too large",
    "too many tokens",
    "context window",
)


@dataclass(slots=True)
class _StreamingToolUse:
    id: str
    name: str
    initial_input: Any
    fragments: list[str] = field(default_factory=list)


class AnthropicModelPort(ModelPort):
    """Translate typed Core requests to Anthropic's content-block protocol."""

    def __init__(
        self,
        config: AnthropicConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not isinstance(config, AnthropicConfig):
            raise TypeError("config must be an AnthropicConfig")
        self.config = config
        self._client = client
        self._owns_client = client is None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        if self._client is None:
            try:
                anthropic = importlib.import_module("anthropic")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Anthropic support requires `pip install ejagent-core[anthropic]`"
                ) from exc
            options: dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
                "max_retries": 0,
            }
            if self.config.base_url is not None:
                options["base_url"] = self.config.base_url
            self._client = anthropic.AsyncAnthropic(**options)
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = None
        self._started = False

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        if not self._started:
            raise RuntimeError("AnthropicModelPort is not started")
        client = self._client
        if client is None:
            raise RuntimeError("Anthropic client is not initialized")
        if request.response_format is not None:
            raise ModelCallError(
                FailureCode.PROVIDER_ERROR,
                "AnthropicModelPort does not support response_format; configure "
                "ModelJudge(response_format=None) for prompt-based JSON output",
            )

        system, messages = _request_messages(request.messages)
        if not messages:
            raise ModelProtocolError(
                "Anthropic requests require at least one user or assistant message"
            )
        options: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": min(self.config.max_tokens, request.max_output_tokens)
            if request.max_output_tokens is not None
            else self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
        }
        if system is not None:
            options["system"] = system
        if request.tools:
            options["tools"] = [_tool_to_anthropic(tool) for tool in request.tools]

        manager: Any | None = None
        entered = False
        content_parts: list[str] = []
        tool_uses: dict[int, _StreamingToolUse] = {}
        input_tokens: int | None = None
        output_tokens: int | None = None
        cache_read_tokens: int | None = None
        cache_write_tokens: int | None = None
        stop_reason: str | None = None
        has_message_stop = False
        try:
            manager = client.messages.stream(**options)
            response = await cancellation.run(manager.__aenter__())
            entered = True
            iterator = response.__aiter__()
            while True:
                cancellation.raise_if_cancelled()
                try:
                    event = await cancellation.run(anext(iterator))
                except StopAsyncIteration:
                    break

                event_type = _field(event, "type")
                if event_type == "message_start":
                    usage = _field(_field(event, "message"), "usage")
                    if usage is not None:
                        input_tokens = _usage_integer(usage, "input_tokens", 0)
                        cache_read_tokens = _usage_integer(
                            usage, "cache_read_input_tokens", None
                        )
                        cache_write_tokens = _usage_integer(
                            usage, "cache_creation_input_tokens", None
                        )
                        initial_output = _usage_integer(usage, "output_tokens", None)
                        if initial_output is not None:
                            output_tokens = initial_output
                    continue
                if event_type == "content_block_start":
                    block = _field(event, "content_block")
                    block_type = _field(block, "type")
                    if block_type == "tool_use":
                        index = _event_index(event)
                        call_id = _field(block, "id")
                        name = _field(block, "name")
                        if not isinstance(call_id, str) or not call_id:
                            raise ModelProtocolError(
                                "Anthropic stream returned a tool use without an id"
                            )
                        if not isinstance(name, str) or not name:
                            raise ModelProtocolError(
                                "Anthropic stream returned a tool use without a name"
                            )
                        tool_uses[index] = _StreamingToolUse(
                            call_id,
                            name,
                            _field(block, "input"),
                        )
                    continue
                if event_type == "content_block_delta":
                    delta = _field(event, "delta")
                    delta_type = _field(delta, "type")
                    if delta_type == "text_delta":
                        text = _field(delta, "text")
                        if isinstance(text, str) and text:
                            content_parts.append(text)
                            yield ModelTextDelta(text)
                    elif delta_type == "thinking_delta":
                        thinking = _field(delta, "thinking")
                        if isinstance(thinking, str) and thinking:
                            yield ModelThinkingDelta(thinking)
                    elif delta_type == "input_json_delta":
                        index = _event_index(event)
                        tool_use = tool_uses.get(index)
                        if tool_use is None:
                            raise ModelProtocolError(
                                "Anthropic tool input delta preceded its start event"
                            )
                        fragment = _field(delta, "partial_json")
                        if not isinstance(fragment, str):
                            raise ModelProtocolError(
                                "Anthropic tool input delta was not a string"
                            )
                        tool_use.fragments.append(fragment)
                    continue
                if event_type == "message_delta":
                    delta = _field(event, "delta")
                    raw_reason = _field(delta, "stop_reason")
                    if raw_reason is not None:
                        stop_reason = str(raw_reason)
                    usage = _field(event, "usage")
                    if usage is not None:
                        updated_output = _usage_integer(usage, "output_tokens", None)
                        if updated_output is not None:
                            output_tokens = updated_output
                    continue
                if event_type == "message_stop":
                    has_message_stop = True
                    continue
                if event_type == "error":
                    error = _field(event, "error")
                    raise RuntimeError(str(_field(error, "message") or error))
        except (asyncio.CancelledError, RunCancelledError):
            raise
        except ModelProtocolError:
            raise
        except Exception as exc:
            raise _model_call_error(exc, operation="Anthropic message stream") from exc
        finally:
            if entered and manager is not None:
                with suppress(Exception):
                    await manager.__aexit__(None, None, None)

        if not has_message_stop or stop_reason is None:
            raise ModelProtocolError(
                "Anthropic stream ended without message_stop and a stop_reason"
            )
        calls = tuple(
            _completed_tool_use(tool_uses[index]) for index in sorted(tool_uses)
        )
        try:
            message = AssistantMessage(
                content="".join(content_parts) or None,
                tool_calls=calls,
            )
        except (TypeError, ValueError) as exc:
            raise ModelProtocolError(
                f"invalid Anthropic assistant message: {exc}"
            ) from exc

        usage = _normalize_usage(
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
        )
        yield ModelResponseCompleted(
            message=message, usage=usage, finish_reason=stop_reason
        )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _event_index(event: Any) -> int:
    index = _field(event, "index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ModelProtocolError("Anthropic stream returned an invalid block index")
    return index


def _request_messages(
    messages: tuple[ContextMessage, ...],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    provider_messages: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(message.content)
        elif isinstance(message, ContextSummary):
            system_parts.append(
                f"[Derived summary: revisions {message.source_revision_start}-"
                f"{message.source_revision_end}; {message.compactor_id}]\n"
                f"{message.content}"
            )
        elif isinstance(message, TransientInstruction):
            system_parts.append(
                f"[Transient instruction: {message.source}]\n{message.content}"
            )
        elif isinstance(message, UserMessage):
            provider_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": message.content}],
                }
            )
        elif isinstance(message, AssistantMessage):
            provider_messages.append(_assistant_to_anthropic(message))
        elif isinstance(message, ToolResultMessage):
            block = _tool_result_to_anthropic(message)
            if _can_append_tool_result(provider_messages):
                provider_messages[-1]["content"].append(block)
            else:
                provider_messages.append({"role": "user", "content": [block]})
        else:
            raise TypeError(f"unsupported ContextMessage {type(message).__name__}")
    return "\n\n".join(system_parts) or None, provider_messages


def _assistant_to_anthropic(message: AssistantMessage) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if message.content:
        content.append({"type": "text", "text": message.content})
    content.extend(
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": thaw_json_value(call.arguments),
        }
        for call in message.tool_calls
    )
    return {"role": "assistant", "content": content}


def _tool_result_to_anthropic(message: ToolResultMessage) -> dict[str, Any]:
    result = thaw_json_value(message.result)
    content = (
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": content,
        "is_error": message.is_error,
    }


def _can_append_tool_result(messages: list[dict[str, Any]]) -> bool:
    if not messages or messages[-1]["role"] != "user":
        return False
    content = messages[-1]["content"]
    return bool(content) and all(
        block.get("type") == "tool_result" for block in content
    )


def _tool_to_anthropic(definition: ToolDefinition) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": definition.name,
        "input_schema": thaw_json_value(definition.input_schema),
    }
    if definition.description is not None:
        value["description"] = definition.description
    return value


def _completed_tool_use(tool_use: _StreamingToolUse) -> ToolCall:
    if tool_use.fragments:
        try:
            arguments: JsonValue = json.loads("".join(tool_use.fragments) or "{}")
        except json.JSONDecodeError as exc:
            raise ModelProtocolError(
                f"Anthropic tool {tool_use.name!r} returned invalid input JSON"
            ) from exc
    else:
        arguments = tool_use.initial_input
    if not isinstance(arguments, Mapping):
        raise ModelProtocolError(
            f"Anthropic tool {tool_use.name!r} input must be an object"
        )
    try:
        return ToolCall(tool_use.id, tool_use.name, arguments)
    except (TypeError, ValueError) as exc:
        raise ModelProtocolError(f"invalid Anthropic tool use: {exc}") from exc


def _usage_integer(usage: Any, name: str, default: int | None) -> int | None:
    raw = _field(usage, name)
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ModelProtocolError(f"Anthropic usage {name} must be non-negative")
    return raw


def _normalize_usage(
    uncached_input: int | None,
    output: int | None,
    cache_read: int | None,
    cache_write: int | None,
) -> ModelUsage | None:
    if uncached_input is None and output is None:
        return None
    input_tokens = (uncached_input or 0) + (cache_read or 0) + (cache_write or 0)
    output_tokens = output or 0
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def _model_call_error(exc: Exception, *, operation: str) -> ModelCallError:
    message = f"{operation} failed: {exc}"
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name == "AuthenticationError" or status in {401, 403}:
        return ModelCallError(FailureCode.AUTHENTICATION, message)
    if name == "RateLimitError" or status == 429:
        return ModelCallError(FailureCode.RATE_LIMIT, message, retryable=True)
    if (
        name == "APITimeoutError"
        or isinstance(exc, TimeoutError)
        or status
        in {
            408,
            504,
        }
    ):
        return ModelCallError(FailureCode.TIMEOUT, message, retryable=True)
    if _is_context_overflow(exc):
        return ModelCallError(FailureCode.CONTEXT_OVERFLOW, message)
    return ModelCallError(
        FailureCode.PROVIDER_ERROR,
        message,
        retryable=status in {500, 502, 503, 529},
    )


def _is_context_overflow(exc: Exception) -> bool:
    values = [str(exc)]
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        error = body.get("error", body)
        if isinstance(error, Mapping):
            values.extend(
                str(value)
                for key in ("type", "message")
                if (value := error.get(key)) is not None
            )
    text = " ".join(values).lower()
    return any(phrase in text for phrase in _CONTEXT_OVERFLOW_PHRASES)
