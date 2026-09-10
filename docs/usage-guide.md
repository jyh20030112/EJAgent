# EJAgent Harness 使用指南

本文介绍如何通过 `AgentHarness` 组合上下文、工具、状态、控制与轨迹反馈。
项目定位见 [Harness 概览](harness-overview.md)；类的职责、数据所有权和内部调用链
见[类、配置与执行链路](core-classes-and-runtime-flow.md)。发行包名仍为 `ejagent-core`。

## 1. 安装与环境

EJAgent Core 要求 Python 3.12+。

```bash
# 基础包，包含 OpenAI-compatible Provider
pip install ejagent-core

# Anthropic Messages adapter
pip install 'ejagent-core[anthropic]'

# MCP adapter
pip install 'ejagent-core[mcp]'

# 同时安装两者
pip install 'ejagent-core[anthropic,mcp]'
```

源码开发环境：

```bash
uv sync --locked --all-extras --group dev
```

OpenAI-compatible 配置写入 `.env`：

```env
MODEL_API_KEY=sk-xxxxxxxx
MODEL_URL=https://api.example.com/v1
CHAT_MODEL=your-model
LLM_TIMEOUT=60
LLM_TEMPERATURE=0.7
LLM_INCLUDE_USAGE=true
```

Anthropic 配置：

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
ANTHROPIC_MODEL=your-claude-model
ANTHROPIC_MAX_TOKENS=4096
ANTHROPIC_TIMEOUT=60
ANTHROPIC_TEMPERATURE=0.7
# ANTHROPIC_BASE_URL=https://api.anthropic.com
```

不要提交 `.env`、Provider key 或包含秘密的 Session journal。

## 2. 创建并运行一个 AgentHarness

```python
import asyncio

from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor


async def main() -> None:
    harness = AgentHarness(
        agent_id="assistant",
        model=OpenAIModelPort(ModelConfig.from_env()),
        tools=FunctionToolExecutor(),
        initial_messages=(SystemMessage("请准确、简洁地回答。"),),
    )

    async with harness:
        first = await harness.run("记住我的项目代号是 CORE-2048。")
        second = await harness.run("项目代号是什么？")
        print(first.result.output)
        print(second.result.output)


asyncio.run(main())
```

推荐始终使用 `async with`。它会启动 Provider、MCP、Context 等资源，从 Store 恢复
状态，并在退出时取消未完成工作、等待 observers、反向关闭资源。
无法使用上下文管理器时，可手动 `await harness.start()`，并在 `finally` 中
`await harness.shutdown()`。

Harness 的常用只读状态：

```python
print(harness.status)
print(harness.agent_id)
print(harness.revision)
print(harness.messages)
print(harness.last_result)
print(harness.snapshot)
```

只有成功提交的 Run 才会出现在 `messages` 并推进 `revision`。

## 3. 读取 RunOutcome

`run()` 和 `continue_run()` 都返回 `RunOutcome`：

```python
from ejagent.contracts import RunStatus

outcome = await harness.run("完成任务")

if outcome.result.status is RunStatus.COMPLETED:
    print(outcome.result.output)
else:
    print(outcome.result.stop_reason)
    if outcome.failure is not None:
        print(outcome.failure.phase)
        print(outcome.failure.code)
        print(outcome.failure.message)
        print(outcome.failure.retryable)

print(outcome.result.turns)
print(outcome.result.usage.to_dict())
print(outcome.delta.messages)
print(outcome.audit_records)
```

`delta` 是本次 Run 建议提交的消息。失败或取消时它可能包含部分过程，但默认不会进入
Conversation；这些过程仍会进入 Store 的 Audit。

## 4. 配置 Provider

### OpenAI-compatible

使用环境变量：

```python
from ejagent.providers import ModelConfig, OpenAIModelPort

model = OpenAIModelPort(ModelConfig.from_env())
```

直接配置：

```python
model = OpenAIModelPort(
    ModelConfig(
        model="your-model",
        api_key="...",
        base_url="https://api.example.com/v1",
        timeout=60,
        temperature=0.2,
        include_usage=True,
    )
)
```

该 adapter 使用 Chat Completions streaming。`include_usage=True` 对 token budget 很
重要；如果设置了 Run token budget，而某次请求没有 usage，Kernel 会停止继续请求。

### Anthropic Messages

```python
from ejagent.providers import AnthropicConfig, AnthropicModelPort

model = AnthropicModelPort(AnthropicConfig.from_env())
```

或直接配置：

```python
model = AnthropicModelPort(
    AnthropicConfig(
        model="your-claude-model",
        api_key="...",
        max_tokens=4096,
        timeout=60,
        temperature=0.2,
    )
)
```

OpenAI 和 Anthropic adapter 都可通过 `client=` 注入已有异步 SDK client。此时
Harness 不会关闭该 client，调用方负责其生命周期。

## 5. 配置 Run 限制和元数据

```python
from ejagent.contracts import RunLimits

defaults = RunLimits(
    max_turns=12,
    max_tokens=20_000,
    max_repeated_tool_calls=3,
)

harness = AgentHarness(
    agent_id="limited-agent",
    model=model,
    tools=FunctionToolExecutor(),
    limits=defaults,
    configuration_revision="prod-2026-08-01",
)

outcome = await harness.run(
    "执行任务",
    limits=RunLimits(max_turns=4, max_tokens=5_000),
    metadata={"tenant": "demo", "request_source": "web"},
)
```

单次 `limits` 覆盖 Harness 默认值。`metadata` 必须是 JSON-compatible 对象，会传入
Context 并出现在相关 Audit 中。`configuration_revision` 是审计标签，不会动态加载配置。

同一模型响应中的多个工具调用始终并发执行。所有调用完成后，结果按模型给出的顺序
进入 Conversation；当前没有串行模式或并发数量开关。

## 6. 使用 Function Tool

### 定义与注册

```python
from ejagent.contracts import (
    CancellationToken,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
)
from ejagent.tools import FunctionTool, FunctionToolExecutor

ADD = ToolDefinition(
    name="add",
    description="Add two numbers.",
    input_schema={
        "type": "object",
        "properties": {
            "left": {"type": "number"},
            "right": {"type": "number"},
        },
        "required": ["left", "right"],
    },
)


async def add(
    call: ToolCall,
    cancellation: CancellationToken,
) -> ToolExecutionResult:
    cancellation.raise_if_cancelled()
    left = call.arguments.get("left")
    right = call.arguments.get("right")
    if (
        isinstance(left, bool)
        or not isinstance(left, (int, float))
        or isinstance(right, bool)
        or not isinstance(right, (int, float))
    ):
        return ToolExecutionResult(
            {"status": "error", "error": "arguments must be numbers"},
            error="arguments must be numbers",
        )
    return ToolExecutionResult({"status": "success", "value": left + right})


tools = FunctionToolExecutor((FunctionTool(ADD, add),))
```

JSON Schema 会发给模型，但 `FunctionToolExecutor` 不自动验证参数，函数必须自行验证。
工具名必须唯一，并满足 1–64 位字母、数字、下划线或短横线。

### 控制 Run 是否继续

```python
from ejagent.contracts import ToolControl, ToolExecutionResult

# 结果返回模型，继续下一 turn
ToolExecutionResult({"value": 42}, control=ToolControl.CONTINUE)

# 直接成功结束 Run；output 成为 RunResult.output
ToolExecutionResult(
    {"value": 42},
    control=ToolControl.COMPLETE,
    output="42",
)

# 拒绝或取消，均不提交 Conversation
ToolExecutionResult({"reason": "denied"}, control=ToolControl.REJECT)
ToolExecutionResult({"reason": "stopped"}, control=ToolControl.CANCEL)
```

业务错误通常返回带 `error=` 的 `ToolExecutionResult`，让模型看到
`ToolResultMessage(is_error=True)`；工具基础设施不可用时才抛 `ToolExecutionError`。

## 7. 组合多个 ToolExecutor

```python
from ejagent.tools import CompositeToolExecutor, McpToolExecutor

local_tools = FunctionToolExecutor((FunctionTool(ADD, add),))
mcp_tools = McpToolExecutor("mcp_config.json")
tools = CompositeToolExecutor((local_tools, mcp_tools))

harness = AgentHarness(
    agent_id="combined-tools",
    model=model,
    tools=tools,
)
```

Composite 会管理子 executor 的生命周期并按工具名路由。任意重复名称都会失败。
不要再把相同子 executor 放入 Harness 的 `resources`，Harness 虽会对直接对象去重，
但无法识别 Composite 内部重复所有权。

## 8. 使用 MCP Tools

安装 MCP extra，然后创建配置：

```json
{
  "playwright": {
    "command": "npx",
    "args": ["@playwright/mcp@latest", "--headless"]
  }
}
```

```python
from ejagent.tools import McpToolExecutor

tools = McpToolExecutor("mcp_config.json")
```

MCP 工具在 Core 中使用 `service__tool` 名称。单个 MCP service 启动失败会被记录并
跳过；启动后可检查 `tools.definitions`。

测试或自定义传输可以注入满足 `McpManager` Protocol 的 `manager=`；`config_path` 和
`manager` 必须且只能提供一个。

## 9. Context Pipeline

### Identity 投影

默认行为等价于：

```python
from ejagent.context import IdentityContextPipeline

context = IdentityContextPipeline()
```

它按 committed messages、当前 Run pending messages、transient instructions 的顺序
组装请求，不改写内容。

### 派生压缩

Core 提供压缩流程，但不绑定摘要模型；你需要实现 `ContextCompactor`：

```python
from ejagent.context import DerivedCompactionPipeline
from ejagent.contracts import (
    CancellationToken,
    ContextCompactionOutput,
    ContextCompactionRequest,
)


class MyCompactor:
    async def compact(
        self,
        request: ContextCompactionRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextCompactionOutput:
        cancellation.raise_if_cancelled()
        # 实际项目可在这里调用单独的摘要模型。
        summary = f"Earlier conversation contains {len(request.messages)} messages."
        return ContextCompactionOutput(summary, "my-compactor-v1")


context = DerivedCompactionPipeline(
    MyCompactor(),
    minimum_messages=20,
)
```

它保留 Conversation 开头连续的 SystemMessage，将达到阈值的其余已提交历史替换为
一次性的 `ContextSummary`。每次超过阈值的模型调用都会重新请求 compactor；若成本
敏感，应在自定义 compactor 中实现缓存。

### Skills

目录结构：

```text
my_skills/
└── release_notes/
    ├── SKILL.md
    ├── template.md              # 可选
    └── examples/sample.md       # 可选
```

`SKILL.md` 可含 YAML frontmatter：

```markdown
---
name: release_notes
description: Convert changes into user-facing release notes.
---

# Instructions

Write concise release notes.
```

```python
from ejagent.context import SkillsContextPipeline

context = SkillsContextPipeline("my_skills")
harness = AgentHarness(
    agent_id="skill-agent",
    model=model,
    tools=FunctionToolExecutor(),
    context=context,
)

async with harness:
    outcome = await harness.run("$release_notes 为这些变更写发布说明……")
```

Pipeline 总是投影精简 Skill 索引；只有任务显式包含 `$release_notes` 或
`skill:release_notes` 时才注入完整内容。Skill 内容不会提交到 Conversation。

组合 Skills 和压缩：

```python
context = SkillsContextPipeline(
    "my_skills",
    base=DerivedCompactionPipeline(MyCompactor(), minimum_messages=20),
)
```

直接使用目录能力：

```python
from ejagent.skills import SkillCatalog

catalog = SkillCatalog("my_skills")
await catalog.discover()
for skill in catalog.skills:
    print(skill.name, skill.description, skill.skill_md)

selected = catalog.get("release_notes")
index_text = catalog.build_index_content()
full_text = catalog.build_skill_context_content(selected.name)
```

`discover()` 每个 Catalog 实例只扫描一次；需要刷新文件时重建 Catalog。

## 10. 临时 Session

省略 `store` 时，Harness 在自身生命周期内维护 Conversation 和 revision。这种模式
不会持久化 Audit，也不能由另一个 Harness 恢复，适合无状态请求和短期任务。

## 11. JSONL 持久化与 Audit

```python
from ejagent.storage import JsonlSessionStore

store = JsonlSessionStore(
    ".ejagent-sessions",
    lock_timeout=10.0,
)
```

不同进程只要使用相同目录和 `agent_id`，Harness 启动时就会恢复最新 Conversation。
每个 Agent 的文件名由 agent ID 的 SHA-256 生成。Store 使用 append-only JSONL、CAS、
幂等 run ID、POSIX 跨进程锁、fsync 和崩溃尾部恢复。

读取审计：

```python
audits = await store.load_audit("same-agent")
for audit in audits:
    print(audit.run_id, audit.committed, audit.result.status)
    for record in audit.records:
        print(record.sequence, record.kind, record.payload)
```

## 12. CONTINUE、取消、Steering 和 Follow-up

以下控制方法只能作用于活跃 Run。异步应用可用自己的事件通知；最小示例可先等待
Harness 进入 `RUNNING`：

```python
import asyncio

from ejagent.harness import HarnessStatus


async def wait_until_running(run_task: asyncio.Task[object]) -> bool:
    while not run_task.done() and harness.status is not HarnessStatus.RUNNING:
        await asyncio.sleep(0)
    return harness.status is HarnessStatus.RUNNING
```

可在构造 Harness 时用 `steering_capacity=` 和 `follow_up_capacity=` 设置两个
队列的正整数容量，默认均为 16。

### 从当前 Conversation 继续

```python
outcome = await harness.continue_run()
```

它创建一个新的 Run，但不追加 UserMessage。它不是暂停恢复，也不会恢复未提交的失败
Run。

### 取消

```python
run_task = asyncio.create_task(harness.run("执行一个较长任务"))

# 实际应用通常在 UI、超时处理或其他协程中触发
if await wait_until_running(run_task):
    changed = harness.cancel("user requested stop")
outcome = await run_task
```

取消是协作式的；Provider、Tool 和 Context adapter 应正确使用传入的
`CancellationToken`。第一次取消返回 `True`，重复取消或没有活跃 Run 返回 `False`。

### Steering

```python
run_task = asyncio.create_task(harness.run("先分析，再调用工具"))

# 必须在 Run 仍为 RUNNING 时由另一个协程调用
if await wait_until_running(run_task):
    receipt = harness.steer("不要修改文件，只做只读分析")
    print(receipt.status, receipt.accepted)

outcome = await run_task
```

Steering 只在下一次模型调用前的安全点生效。若当前 turn 已经过最后一个安全点，它会
返回 `TOO_LATE` 或在 Run 结束时记录 `steering_discarded`。它不会进入 Conversation。

### Follow-up

```python
run_task = asyncio.create_task(harness.run("完成第一项任务"))

# 稍后由已经确认 Run 活跃的协程提交
if await wait_until_running(run_task):
    handle = harness.follow_up("完成后再总结风险")
else:
    handle = None
first = await run_task

if handle is not None and handle.accepted:
    second = await handle.wait()
    print(second.result.output)
else:
    print(handle.receipt.status)
```

Follow-up 是独立 TASK Run，按 FIFO 排队并基于此前成功提交的 Conversation 执行。
空闲时调用会返回 `NOT_RUNNING`；队列满返回 `QUEUE_FULL`。对未接收 handle 调用
`wait()` 会抛 `FollowUpRejectedError`。

应用需要根据自己的 UI、网络事件或工具进度协调调用时机；Core 当前没有单独的
“Run 已到达某个安全点”公共事件接口。

## 13. RunObserver 与额外资源

Observer 适合日志、指标和 tracing：

```python
from ejagent.contracts import RunAudit


class PrintObserver:
    async def observe(self, audit: RunAudit) -> None:
        print(audit.run_id, audit.committed, audit.result.status)


harness = AgentHarness(
    agent_id="observed-agent",
    model=model,
    tools=FunctionToolExecutor(),
    observers=(PrintObserver(),),
)
```

Observer 在 Store 决策后异步运行。它的延迟或失败不会改变 Run，因此不要用它承担
“必须成功”才能算完成的业务写入。

额外资源只需实现 `start()` 和 `shutdown()`：

```python
class DatabaseConnection:
    async def start(self) -> None:
        ...

    async def shutdown(self) -> None:
        ...


harness = AgentHarness(
    agent_id="resource-agent",
    model=model,
    tools=FunctionToolExecutor(),
    resources=(DatabaseConnection(),),
)
```

资源按声明顺序启动、反序关闭；启动中途失败会回滚已启动资源。

## 14. 轨迹评估与上下文反馈

`AgentHarness(trajectory=monitor, context=pipeline, ...)` 可以把在线监控和模型反馈
接入同一个 Harness。`TrajectoryMonitor` 是 `ejagent.kernel` 中的协议；内置的
`OnlineTrajectoryMonitor`、`CheckpointEvaluator` 和 `TrajectoryContextPipeline`
仍位于内部包 `ejagent._trajectory`，尚未作为稳定顶层 API 发布。

应用需要提供 evaluator，从实际环境采集事实、验证需求和约束。监控器在检查点计算
进度与循环判断；通过 `update_sink` 将结果发布到 `TrajectoryContextBuffer`，再由
`TrajectoryContextPipeline` 在指定 Run、指定轮次追加临时反馈。只配置监控器时会有
审计回执，但不会自动增加模型上下文。Run 结束时，应通过关闭回调释放暂存帧和宿主状态。

仓库中的交互示例提供完整接线：

```bash
uv sync --locked --all-extras --group dev
uv run streamlit run examples/streamlit_app.py
```

应用默认开启 **Trajectory feedback**。选择 Demo 模式，在 **Controls** 点击
**Run parallel validation** 查看正常进度，或点击 **Run trajectory recovery** 观察
串行重复如何在循环反馈后转为并发。恢复演示至少需要 8 轮和 160 个 Demo Token。
**Trajectory** 页签展示检查点和实际加入模型 Context 的临时指令。

该示例仅验证本次 Run 中 A 完成、B 完成及 A/B 重叠三个需求，不能替代其他应用的
任务评估。完成审核建议也不等于执行控制：`completion_allowed=False` 目前不会让
Kernel 自动拒绝结束或再执行一轮。详细边界见[轨迹集成](trajectory-runtime-readiness.md)。

## 15. 高级嵌入：直接使用执行内核

已有应用自行承担 Harness 职责时，可以直接使用 `RuntimeKernel` 执行单个 Run：

```python
import asyncio
from uuid import uuid4

from ejagent.contracts import RunIntent, RunSpec, SystemMessage
from ejagent.kernel import RuntimeKernel
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor


async def main() -> None:
    model = OpenAIModelPort(ModelConfig.from_env())
    tools = FunctionToolExecutor()
    await model.start()
    try:
        kernel = RuntimeKernel(model=model, tools=tools)
        spec = RunSpec(
            run_id=str(uuid4()),
            base_revision=0,
            intent=RunIntent.TASK,
            task="完成一次独立任务",
            messages=(SystemMessage("请准确回答。"),),
        )
        outcome = await kernel.run(spec)
        print(outcome.result.output)
    finally:
        await model.shutdown()


asyncio.run(main())
```

Kernel 不启动资源、不读取或写入 Store，也不会提交 `outcome.delta`。调用方必须自行
负责生命周期、并发串行化、CAS 和恢复。绝大多数应用不应使用这条路径。

## 16. 自定义 Adapter

所有 seam 都是结构化 Protocol，不要求继承基类：

- `ModelPort`：把 Provider request/stream 转成 Core model events。
- `ToolExecutor`：暴露稳定 definitions 并执行 ToolCall。
- `ContextPipeline`：从 ContextRequest 构建一次性 ContextView。
- `ContextCompactor`：生成摘要文本，不修改 Conversation。
- `SessionStore`：实现 load、幂等 compare-and-commit。
- `AuditReader`：可选的 Audit 查询能力。
- `RunObserver`：提交后的旁路观察。
- `ManagedResource`：由 Harness 管理 start/shutdown。
- `TrajectoryMonitor`：通过 Kernel 定义的边界观察执行；宿主负责领域评估与反馈接线。

预期运行失败应使用对应的 `ModelCallError`、`ToolExecutionError`、
`ContextBuildError` 或 `SessionStoreError`。返回错误类型、缺失终止事件等实现 bug 应
使用或触发 `*ProtocolError`，不要伪装成普通业务失败。

## 17. 当前实现边界和易错点

- 一个 `AgentHarness` 只代表一个逻辑 Agent，不管理多 Agent。
- 不支持真正暂停并序列化一个正在执行的 Run；`continue_run()` 是新 Run。
- 多个 Run 会被 Harness 串行化；同一模型响应中的 tool calls 始终并发执行。
- `RunPolicy` 尚未成为可注入实现；限制和终止逻辑目前在 Kernel 内。
- 轨迹评估和 Context 反馈已可接入；完成门控支持同一 Run 内的有限重试。
- [动态任务规划](task-planning.md)支持根据 query 生成验收计划，执行者通过 `update_plan`
  修订执行步骤；自动拒绝普通动作和强制重新规划尚未提供。
- Harness 会消费 Provider streaming，但当前只把 delta 写入 Audit，不向 `run()` 调用方
  实时推送。调用方只能在 Run 完成后读取 outcome。
- `DerivedCompactionPipeline` 只生成派生视图，不减少 durable journal 大小。
- 省略 Store 时状态不能跨 Harness 恢复；`JsonlSessionStore` 的跨进程文件锁要求 POSIX。
- MCP 单个 service 失败会被跳过；不要把“Harness 成功启动”等同于“所有 MCP 服务
  都可用”。

## 18. 验证

```bash
uv run ruff check src tests examples benchmarks
uv run ruff format --check src tests examples benchmarks
uv run mypy
uv run python -m unittest discover -s tests -p 'test*.py' -q
uv build
```
