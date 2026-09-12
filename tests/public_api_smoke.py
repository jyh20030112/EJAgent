"""Smoke-test the installed package without relying on the source tree."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
from importlib.metadata import version
from importlib.resources import files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-no-mcp", action="store_true")
    args = parser.parse_args()

    if args.expect_no_mcp and importlib.util.find_spec("fastmcp") is not None:
        raise AssertionError("fastmcp must not be installed by the core package")
    if args.expect_no_mcp and importlib.util.find_spec("anthropic") is not None:
        raise AssertionError("anthropic must not be installed by the core package")
    if importlib.util.find_spec("simagentplg") is not None:
        raise AssertionError("the former import package must not be installed")

    import ejagent
    import ejagent.evaluation

    for name in ejagent.evaluation.__all__:
        if not hasattr(ejagent.evaluation, name):
            raise AssertionError(f"evaluation export does not resolve: {name}")

    missing_attributes = [
        name for name in ejagent.__all__ if not hasattr(ejagent, name)
    ]
    if missing_attributes:
        raise AssertionError(
            f"public exports do not resolve: {', '.join(missing_attributes)}"
        )
    required_exports = {
        "AgentHarness",
        "AnthropicConfig",
        "AnthropicModelPort",
        "RuntimeKernel",
        "OpenAIModelPort",
        "ModelConfig",
        "FunctionToolExecutor",
        "CompositeToolExecutor",
        "McpToolExecutor",
        "IdentityContextPipeline",
        "DerivedCompactionPipeline",
        "SkillsContextPipeline",
        "JsonlSessionStore",
        "SkillCatalog",
    }
    if missing := required_exports.difference(ejagent.__all__):
        raise AssertionError(
            f"required public exports are missing: {', '.join(sorted(missing))}"
        )
    forbidden_exports = {
        "BaseAgent",
        "AgentOrchestrator",
        "ModelAdapter",
        "OpenAIModelAdapter",
        "BaseHandler",
        "ToolMiddleware",
        "SessionStorage",
    }
    if present := forbidden_exports.intersection(ejagent.__all__):
        raise AssertionError(
            f"legacy exports remain public: {', '.join(sorted(present))}"
        )

    if not files("ejagent").joinpath("py.typed").is_file():
        raise AssertionError("installed package is missing the py.typed marker")
    if version("ejagent-core") != "0.7.0":
        raise AssertionError("installed distribution has the wrong version")

    if args.expect_no_mcp:

        async def check_missing_mcp_message() -> None:
            executor = ejagent.McpToolExecutor("unused.json")
            try:
                await executor.start()
            except RuntimeError as exc:
                if "ejagent-core[mcp]" not in str(exc):
                    raise AssertionError(
                        "missing MCP dependencies produced no install guidance"
                    ) from exc
            else:
                raise AssertionError("MCP startup unexpectedly succeeded")

        asyncio.run(check_missing_mcp_message())

        async def check_missing_anthropic_message() -> None:
            model = ejagent.AnthropicModelPort(
                ejagent.AnthropicConfig("unused", "unused")
            )
            try:
                await model.start()
            except RuntimeError as exc:
                if "ejagent-core[anthropic]" not in str(exc):
                    raise AssertionError(
                        "missing Anthropic dependency produced no install guidance"
                    ) from exc
            else:
                raise AssertionError("Anthropic startup unexpectedly succeeded")

        asyncio.run(check_missing_anthropic_message())


if __name__ == "__main__":
    main()
