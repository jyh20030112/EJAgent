from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.json import JsonObject, freeze_json_object
from ejagent.contracts.messages import (
    AssistantMessage,
    ContextMessage,
    is_context_message,
)
from ejagent.contracts.runs import FailureCode
from ejagent.contracts.tools import ToolDefinition


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Provider-neutral token usage for one completed model response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        values = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
        for name, value in values.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        if (
            self.cache_read_tokens is not None
            and self.cache_read_tokens > self.input_tokens
        ):
            raise ValueError("cache_read_tokens must not exceed input_tokens")
        if (
            self.cache_write_tokens is not None
            and self.cache_write_tokens > self.input_tokens
        ):
            raise ValueError("cache_write_tokens must not exceed input_tokens")
        if (
            self.reasoning_tokens is not None
            and self.reasoning_tokens > self.output_tokens
        ):
            raise ValueError("reasoning_tokens must not exceed output_tokens")

    def to_dict(self) -> dict[str, int | None]:
        """Return a detached JSON-compatible representation."""

        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Provider-neutral request produced for one Kernel turn."""

    messages: tuple[ContextMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    max_output_tokens: int | None = None
    response_format: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.response_format is not None:
            object.__setattr__(
                self,
                "response_format",
                freeze_json_object(self.response_format, label="response_format"),
            )
        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer or None")
        object.__setattr__(self, "messages", tuple(self.messages))
        if not all(is_context_message(message) for message in self.messages):
            raise TypeError("model messages must contain ContextMessage values")
        object.__setattr__(self, "tools", tuple(self.tools))
        if not all(isinstance(tool, ToolDefinition) for tool in self.tools):
            raise TypeError("model tools must contain ToolDefinition values")


@dataclass(frozen=True, slots=True)
class ModelTextDelta:
    """One Provider-neutral piece of assistant text."""

    delta: str

    def __post_init__(self) -> None:
        if not isinstance(self.delta, str) or not self.delta:
            raise ValueError("model text delta must not be empty")


@dataclass(frozen=True, slots=True)
class ModelThinkingDelta:
    """One Provider-neutral piece of provisional model reasoning."""

    delta: str

    def __post_init__(self) -> None:
        if not isinstance(self.delta, str) or not self.delta:
            raise ValueError("model thinking delta must not be empty")


@dataclass(frozen=True, slots=True)
class ModelResponseCompleted:
    """Terminal stream event containing one normalized assistant response."""

    message: AssistantMessage
    usage: ModelUsage | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, AssistantMessage):
            raise TypeError("completed model message must be AssistantMessage")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("completed model usage must be ModelUsage or None")
        if self.finish_reason is not None and (
            not isinstance(self.finish_reason, str) or not self.finish_reason.strip()
        ):
            raise ValueError("finish_reason must be non-empty text or None")


ModelStreamEvent: TypeAlias = (
    ModelTextDelta | ModelThinkingDelta | ModelResponseCompleted
)


class ModelCallError(RuntimeError):
    """Expected operational failure reported by a ModelPort."""

    def __init__(
        self,
        code: FailureCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        if not isinstance(code, FailureCode):
            raise TypeError("Model error code must be a FailureCode")
        if not message.strip():
            raise ValueError("Model error message must not be empty")
        if not isinstance(retryable, bool):
            raise TypeError("Model error retryable must be a boolean")
        self.code = code
        self.retryable = retryable
        super().__init__(message)


class ModelProtocolError(RuntimeError):
    """A ModelPort violated the stable Kernel stream protocol."""


class ModelPort(Protocol):
    """Ready Provider boundary consumed by a Runtime Kernel."""

    def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream one normalized model response without managing resources."""
