<div align="center">
  <img src="assets/ejagent-mascot.svg" width="180" alt="从蛋中孵化的 EJAgent 吉祥物">
  <h1>EJAgent Core</h1>
  <p><em>孵化属于你的 Agent —— 围绕上下文、工具、状态、控制与证据反馈构建的 Python Agent Harness。</em></p>
  <p>
    <a href="https://github.com/jyh20030112/EJAgent/stargazers"><img src="https://img.shields.io/github/stars/jyh20030112/EJAgent" alt="GitHub Stars"></a>
    <a href="https://pypi.org/project/ejagent-core/"><img src="https://img.shields.io/pypi/v/ejagent-core" alt="PyPI 版本"></a>
    <a href="https://github.com/jyh20030112/EJAgent/actions/workflows/ci.yml"><img src="https://github.com/jyh20030112/EJAgent/actions/workflows/ci.yml/badge.svg" alt="CI 状态"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="MIT License"></a>
    <br>
    <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
    <img src="https://img.shields.io/badge/asyncio-native-4B8BBE" alt="原生 asyncio">
    <img src="https://img.shields.io/badge/MCP-ready-7C3AED" alt="支持 MCP">
  </p>
  <p><a href="README.md">English</a> · <strong>简体中文</strong></p>
</div>

---

EJAgent Core 是一个 Python **Agent Harness**，围绕 Agent 的决策组织上下文、工具、
状态、控制与评估。它支持 Agent 跨任务保留工作状态、从已提交记录恢复、接受用户干预，
并在选择下一步行动时使用经过验证的环境反馈。

它可以用于构建智能助手、编程 Agent、研究 Agent 和面向具体任务的应用。模型 Provider、
工具能力、上下文策略、存储和评估器可以按应用领域组合。

## 可以用它做什么

- **高度定制的 Agent**：模型 Provider、工具后端、上下文策略、存储层和观察器均可
  独立替换。
- **有状态的智能助手**：在多个任务间保留类型化对话历史，并从最近一次已提交状态
  继续运行。
- **可持久恢复的 Agent**：将 Session 写入追加式 Journal，进程重启后可以恢复。
- **能够使用工具的 Agent**：接入 Python 函数、组合多个工具执行器，或通过统一接口
  连接 MCP 服务。
- **可控的 Agent**：取消当前任务、干预下一次模型调用、排队后续任务，并限制 turn
  数或 token 用量。
- **上下文感知的 Agent**：注入本地 Skill、为长对话生成摘要，或者实现自己的上下文
  策略。
- **可观察的系统**：记录结构化结果、故障、token 用量、模型事件和工具活动，同时让
  观察逻辑与执行过程解耦。
- **具备轨迹反馈的 Agent**：接入宿主评估器，判断需求满足情况、约束和重复的状态／
  动作模式，再将相关反馈加入下一次模型上下文。
- **不绑定 Provider 的应用**：使用 OpenAI-compatible Endpoint、Anthropic，或为其他
  模型 API 实现适配器。

## 为什么选择 EJAgent Core

Harness 负责组织模型能看到什么、能执行什么、哪些状态跨任务保留，以及执行证据如何
影响后续决策。EJAgent 将已接受的 Conversation、执行 Audit 和临时模型 Context 分开，
让反馈和摘要可以更新，同时保持已提交历史的完整性。

`AgentHarness` 是应用入口，拥有生命周期、已接受状态、控制和提交协调职责。
`RuntimeKernel` 是其中执行单次 Run 的内核；上下文管道、工具、Provider 和可选轨迹
评估共同组成 Harness 的能力。项目职责与当前实现边界见
[Agent Harness 概览](docs/harness-overview.md)。

## 安装

EJAgent Core 需要 Python 3.12 或更高版本。

```bash
uv add ejagent-core
```

按需添加可选集成：

```bash
uv add 'ejagent-core[anthropic]'  # Anthropic
uv add 'ejagent-core[mcp]'        # MCP
```

## 快速开始

配置 OpenAI-compatible Endpoint：

```env
MODEL_API_KEY=sk-xxxxxxxx
MODEL_URL=https://api.example.com/v1
CHAT_MODEL=your-model
```

创建一个有状态 Agent：

```python
from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor

model = OpenAIModelPort(ModelConfig.from_env())
harness = AgentHarness(
    agent_id="assistant",
    model=model,
    tools=FunctionToolExecutor(),
    initial_messages=(SystemMessage("请准确回答。"),),
)

async with harness:
    await harness.run("记住我的项目是 EJAgent。")
    answer = await harness.run("我的项目是什么？")
    print(answer.result.output)
```

不改变调用方式，就能为同一个 Agent 增加更多能力：

```python
from ejagent.context import SkillsContextPipeline
from ejagent.storage import JsonlSessionStore
from ejagent.tools import McpToolExecutor

harness = AgentHarness(
    agent_id="assistant",
    model=model,
    tools=McpToolExecutor("mcp_config.json"),
    context=SkillsContextPipeline("skills"),
    store=JsonlSessionStore(".ejagent-sessions"),
)
```

## 评估任务结果

`ejagent.evaluation` 支持为每个 Run 绑定不可变的 `EvaluationPlan`，读取带版本的证据，
并评估验收项。`GoalEvaluator` 统一编排确定性 `Verifier` 与可选的 `ModelJudge`，
后者处理显式声明的语义验收项。语义项可以配置 `guard_method`：确定性前置检查返回
`pass` 后才调用 LLM。两条路径生成统一的评估报告，确定性检查不发起模型请求。

库负责证据采集、验证编排、结果校验和报告，内置文件、工作区、命令和探针证据源。
宿主负责配置证据源、注册验证能力，并按需提供业务专用规则。`EvaluationMonitor`
将报告接入轨迹分析。缺少证据时返回未知，证据变化后使旧结论失效。

显式配置 `CompletionPolicy(CompletionMode.ENFORCE)` 可拦截未通过的完成请求，
并在当前 Run 内重试。默认保持观察模式。
接入方法、自定义检查、评估日志
及无需凭证的 `examples/evaluate_artifact.py` 示例见[评估模块指南](docs/evaluation.md)。

## 每轮决策都能看到状态

通过 `EvaluationMonitor.context_pipeline()` 配置轨迹上下文管道后，每次模型决策都会
收到 checkpoint 状态，包括当前事实、需求与约束的验收结果，以及进展。疑似循环仅隐藏
警告，状态仍然可见；确认循环、完成验收失败等需要行动的事件才附加可选的 `feedback`。

观察缺失或不完整时，明确标记状态不可用，不沿用旧的成功结论或覆盖率。上下文投影本身
不增加评估或 Planner 调用，临时指令也不写入已提交的对话历史。字段结构与迁移说明见
[v2 轨迹上下文文档](docs/trajectory-context-projection.md)。

## 动态任务与执行计划

配置 `AgentHarness(planner=ModelTaskPlanner(...))` 后，每次 query 都可以动态生成
任务、目标、验收项和初始执行步骤。模型只能引用宿主已注册的验证能力；执行者可以
根据 checkpoint 反馈调用 `update_plan`。Harness 校验版本并记录修订，Run 内的验收
条件保持固定，步骤标记完成不会替代证据验收。

```bash
uv run python examples/planned_feature.py         # 正式 LLM
uv run python examples/planned_feature.py --demo  # 脚本模型，真实文件修改和测试
```

示例在独立工作目录中添加 Python 功能，保留实际模型上下文、计划版本、测试证据及
审计记录。Streamlit 的 Provider 模式可启用 **Dynamic task planning**，按 query 选择
探针验收项。接入方式与能力边界见[动态任务规划指南](docs/task-planning.md)。

## 在 Streamlit 中体验 Harness

仓库提供一个用于体验 Harness 行为的交互式应用。确定性 Demo 模式不需要凭据，
也可以切换到前文配置的 OpenAI-compatible Endpoint。

```bash
uv sync --locked --extra streamlit
uv run streamlit run examples/streamlit_app.py
```

页面可以检查 JSONL 恢复、工具并行耗时、取消、Steering、FIFO Follow-up、Run
限制、Revision、Usage 和持久化 Audit。点击 **Start** 时会锁定本次运行配置；如需
修改，请先停止当前会话。

应用默认开启 **Trajectory feedback**，可以在启动前关闭。宿主评估器根据本次 Run
的真实工具记录检查三个需求：A 完成、B 完成、已完成的 A/B 执行时间存在重叠。
这个覆盖率只衡量探针验证，不评价任意聊天任务。**Trajectory** 页签展示检查点进度、
循环判断、完成建议，以及实际加入模型上下文的临时指令。详细信息仅保留当前应用会话
的最近一次 Run；重启后可在独立评估日志中查到完整报告，Audit 保留关联回执。

选择 Demo 模式，点击 **Controls → Run trajectory recovery**，可以复现反馈过程：
模型先交替串行调用探针，分析器确认循环后，模型根据上下文反馈改为同时调用两个探针。
请至少允许 8 轮和 160 个 Demo Token。原有并发验证仍可使用；真实 Provider 模式也接入
同一套正式评估器和上下文管道。启用 **Semantic completion review** 可单独审核最终总结，
启用 **Require completion approval** 则会在当前 Run 内限次重试。演示模式使用确定性的
Judge 替身；真实 Provider 模式会发起独立模型请求。界面展示逐项理由、证据版本、缺失项
以及 Actor／Judge 成本，完整报告保存在会话目录的 `evaluations/` 下。开启完成拦截后，
**Run completion recovery** 可演示同一 Run 内先拒绝未经验证的声明、再完成探针验证的
流程，Demo 模式共需 3 轮。

## 每个边界都可以定制

| 想要替换的部分       | 扩展接口           |
| -------------------- | ------------------ |
| 模型 Provider 或协议 | `ModelPort`        |
| 本地或远程工具后端   | `ToolExecutor`     |
| 上下文选择与投影策略 | `ContextPipeline`  |
| 长历史摘要方式       | `ContextCompactor` |
| Session 持久化       | `SessionStore`     |
| 日志、追踪或指标     | `RunObserver`      |
| 任务与初始计划生成   | `TaskPlanner`      |
| 评估证据采集         | `EvidenceSource`   |
| 确定性验收规则       | `Verifier`         |
| 在线轨迹观察         | `ejagent.kernel` 中的 `TrajectoryMonitor` |

这些接口保持精简并且不绑定 Provider，通过 `AgentHarness` 组合应用需要的能力。
底层在线监控器和轨迹上下文适配器位于内部包 `ejagent._trajectory`。应用可以通过
公开的 `ejagent.evaluation` 模块配置证据来源、确定性检查和 `ModelJudge` 语义评估。

## 内置能力

- OpenAI-compatible 与 Anthropic 流式模型适配器
- Python 函数工具、组合工具执行器和 MCP 工具
- 本地 Skill 发现与显式 Skill 激活
- 不改写对话历史的派生上下文摘要
- 进程内临时 Session 与持久化 JSONL Session
- 工具并发执行，同时保持 Conversation 顺序确定
- 协作式取消、实时 Steering 和 FIFO Follow-up
- 结构化 Audit 与统一的 token 用量统计
- 基于 Revision、支持幂等和跨进程文件锁的 Session 提交
- 确定性与可选 LLM 验收评估，统一生成报告
- 可选的在线轨迹评估，每轮提供状态并按需附加反馈

当前原生用户输入仅支持文本：`run(task: str)`、`follow_up(task: str)` 和
`UserMessage.content` 均接收字符串。尚未实现图片或图文混合消息；图片 URL 或 Base64
字符串会作为文本传递，不会自动成为模型的视觉输入。

当前每个 `AgentHarness` 管理一个逻辑 Agent，尚未实现多 Agent 协调和任意时刻的
mid-Run 暂停／恢复。已支持执行者根据反馈提出执行计划更新；自动拒绝动作和强制
重新规划仍属于后续策略工作。完成
审核拦截已作为独立的显式策略提供，默认仍仅观察。

## 文档

- [Agent Harness 概览](docs/harness-overview.md)：项目定位、职责、反馈链路和当前能力。
- [全功能使用指南](docs/usage-guide.md)：安装、配置和全部内置能力的使用方式。
- [Harness 类与执行链路](docs/core-classes-and-runtime-flow.md)：内部模型与完整 Run
  生命周期。
- [Agent Harness 架构](docs/runtime-kernel-harness-design.md)：规范性的架构边界与
  不变量。
- [轨迹集成](docs/trajectory-runtime-readiness.md)：在线观察、上下文反馈和执行策略边界。

## 开发

```bash
uv sync --locked --all-extras --group dev
uv run ruff check src tests examples benchmarks
uv run ruff format --check src tests examples benchmarks
uv run mypy
uv run python -m unittest discover -s tests -p 'test*.py' -q
uv build
```
