from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

from openai import (
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from ejagent.contracts.control import CancellationToken, RunCancelledError
from ejagent.contracts.json import thaw_json_value
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
from ejagent.providers.config import ModelConfig

_CONTEXT_OVERFLOW_CODES = {
    "context_length_error",
    "context_length_exceeded",
    "context_window_exceeded",
    "input_too_long",
    "prompt_too_long",
}
_CONTEXT_OVERFLOW_PHRASES = (
    "context length exceeded",
    "context window exceeded",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
)


@dataclass(slots=True)
class _StreamingToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


class OpenAIModelPort(ModelPort):
    """Translate typed Core requests to OpenAI-compatible chat completions."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self.config = config
        self._client = client
        self._owns_client = client is None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=0,
            )
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
            raise RuntimeError("OpenAIModelPort is not started")
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI client is not initialized")

        options: dict[str, Any] = {
            "model": self.config.model,
            "messages": cast(
                Any, [_message_to_openai(item) for item in request.messages]
            ),
            "temperature": self.config.temperature,
            "tools": cast(Any, [_tool_to_openai(item) for item in request.tools])
            or None,
            "stream": True,
        }
        if self.config.include_usage:
            options["stream_options"] = {"include_usage": True}
        if request.max_output_tokens is not None:
            options["max_completion_tokens"] = request.max_output_tokens
        if request.response_format is not None:
            options["response_format"] = thaw_json_value(request.response_format)

        response: Any | None = None
        content_parts: list[str] = []
        tool_calls: dict[int, _StreamingToolCall] = {}
        usage: ModelUsage | None = None
        finish_reason: str | None = None
        try:
            response = await cancellation.run(client.chat.completions.create(**options))
            if response is None:
                raise ModelProtocolError("OpenAI stream request returned no response")
            iterator = response.__aiter__()
            while True:
                cancellation.raise_if_cancelled()
                try:
                    chunk = await cancellation.run(anext(iterator))
                except StopAsyncIteration:
                    break

                raw_usage = getattr(chunk, "usage", None)
                if raw_usage is not None:
                    try:
                        usage = _normalize_usage(raw_usage)
                    except (TypeError, ValueError) as exc:
                        raise ModelProtocolError(
                            f"OpenAI stream returned invalid usage: {exc}"
                        ) from exc
                choices = getattr(chunk, "choices", None) or ()
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None) is not None:
                    finish_reason = str(choice.finish_reason)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if content:
                    text = str(content)
                    content_parts.append(text)
                    yield ModelTextDelta(text)
                for field in (
                    "reasoning_content",
                    "reasoning",
                    "reasoning_text",
                ):
                    reasoning = getattr(delta, field, None)
                    if isinstance(reasoning, str) and reasoning:
                        yield ModelThinkingDelta(reasoning)
                        break
                for position, partial in enumerate(
                    getattr(delta, "tool_calls", None) or ()
                ):
                    index = getattr(partial, "index", None)
                    if index is None:
                        index = position
                    if (
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or index < 0
                    ):
                        raise ModelProtocolError(
                            "OpenAI stream returned an invalid tool call index"
                        )
                    call = tool_calls.setdefault(index, _StreamingToolCall())
                    call_id = getattr(partial, "id", None)
                    if call_id:
                        call.id = str(call_id)
                    function = getattr(partial, "function", None)
                    if function is None:
                        continue
                    name = getattr(function, "name", None)
                    if name:
                        call.name += str(name)
                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        call.arguments += str(arguments)
        except (asyncio.CancelledError, RunCancelledError):
            raise
        except ModelProtocolError:
            raise
        except Exception as exc:
            raise _model_call_error(exc, operation="chat completion stream") from exc
        finally:
            if response is not None:
                close = getattr(response, "close", None) or getattr(
                    response,
                    "aclose",
                    None,
                )
                if close is not None:
                    with suppress(Exception):
                        result = close()
                        if inspect.isawaitable(result):
                            await result

        if finish_reason is None:
            raise ModelProtocolError("OpenAI stream ended without a finish_reason")
        normalized_calls = tuple(
            _completed_tool_call(tool_calls[index]) for index in sorted(tool_calls)
        )
        try:
            message = AssistantMessage(
                content="".join(content_parts) or None,
                tool_calls=normalized_calls,
            )
        except (TypeError, ValueError) as exc:
            raise ModelProtocolError(
                f"invalid OpenAI assistant message: {exc}"
            ) from exc
        yield ModelResponseCompleted(
            message=message, usage=usage, finish_reason=finish_reason
        )


def _message_to_openai(message: ContextMessage) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        value: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            value["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            thaw_json_value(call.arguments),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        return value
    if isinstance(message, ToolResultMessage):
        result = thaw_json_value(message.result)
        content = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        )
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": content,
        }
    if isinstance(message, ContextSummary):
        return {
            "role": "system",
            "content": (
                f"[Derived summary: revisions {message.source_revision_start}-"
                f"{message.source_revision_end}; {message.compactor_id}]\n"
                f"{message.content}"
            ),
        }
    if isinstance(message, TransientInstruction):
        return {
            "role": "system",
            "content": f"[Transient instruction: {message.source}]\n{message.content}",
        }
    raise TypeError(f"unsupported ContextMessage {type(message).__name__}")


def _tool_to_openai(definition: ToolDefinition) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": definition.name,
        "parameters": thaw_json_value(definition.input_schema),
    }
    if definition.description is not None:
        function["description"] = definition.description
    return {"type": "function", "function": function}


def _completed_tool_call(call: _StreamingToolCall) -> ToolCall:
    if not call.id or not call.name:
        raise ModelProtocolError("OpenAI stream returned an incomplete tool call")
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as exc:
        raise ModelProtocolError(
            f"OpenAI tool {call.name!r} returned invalid argument JSON"
        ) from exc
    if not isinstance(arguments, Mapping):
        raise ModelProtocolError(
            f"OpenAI tool {call.name!r} arguments must decode to an object"
        )
    try:
        return ToolCall(id=call.id, name=call.name, arguments=arguments)
    except (TypeError, ValueError) as exc:
        raise ModelProtocolError(f"invalid OpenAI tool call: {exc}") from exc


def _usage_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _normalize_usage(raw_usage: Any) -> ModelUsage:
    input_tokens = int(_usage_field(raw_usage, "prompt_tokens") or 0)
    output_tokens = int(_usage_field(raw_usage, "completion_tokens") or 0)
    prompt_details = _usage_field(raw_usage, "prompt_tokens_details")
    completion_details = _usage_field(raw_usage, "completion_tokens_details")
    cache_read = _usage_field(prompt_details, "cached_tokens")
    cache_write = _usage_field(prompt_details, "cache_write_tokens")
    reasoning = _usage_field(completion_details, "reasoning_tokens")
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_tokens=int(cache_read) if cache_read is not None else None,
        cache_write_tokens=int(cache_write) if cache_write is not None else None,
        reasoning_tokens=int(reasoning) if reasoning is not None else None,
    )


def _model_call_error(exc: Exception, *, operation: str) -> ModelCallError:
    message = f"{operation} failed: {exc}"
    if isinstance(exc, AuthenticationError):
        return ModelCallError(FailureCode.AUTHENTICATION, message)
    if isinstance(exc, RateLimitError):
        return ModelCallError(FailureCode.RATE_LIMIT, message, retryable=True)
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return ModelCallError(FailureCode.TIMEOUT, message, retryable=True)
    if _is_context_overflow(exc):
        return ModelCallError(FailureCode.CONTEXT_OVERFLOW, message)
    return ModelCallError(FailureCode.PROVIDER_ERROR, message)


def _is_context_overflow(exc: Exception) -> bool:
    values: list[str] = []
    for candidate in (
        getattr(exc, "code", None),
        getattr(exc, "type", None),
        getattr(exc, "body", None),
    ):
        if isinstance(candidate, str):
            values.append(candidate)
        elif isinstance(candidate, Mapping):
            for key in ("code", "type", "message"):
                value = candidate.get(key)
                if isinstance(value, str):
                    values.append(value)
    values.append(str(exc))
    normalized = {value.strip().lower() for value in values}
    if normalized & _CONTEXT_OVERFLOW_CODES:
        return True
    text = " ".join(normalized)
    return any(phrase in text for phrase in _CONTEXT_OVERFLOW_PHRASES)
