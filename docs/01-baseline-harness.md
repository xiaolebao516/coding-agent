# 01. Baseline Harness 架构梳理与面试题单（M1–M2.5）

本文档对应 Issue #7（Baseline 固化阶段），用于快速回顾 `coding-agent` 在引入论文优化前的核心工程底座，并整理阶段一高频面试追问与答题思路。

---

## 1. 当前 Baseline 定位

在 `v0.2-baseline` 阶段，`coding-agent` 是一个**不依赖 LangGraph / AutoGen 等重型黑盒框架、从零手写的轻量级 Coding Agent Harness**：

- **工具配置**：单工具 `bash`（`BashTool`）
- **上下文策略**：全量历史消息（无压缩、无折叠）
- **运行环境**：隔离的 Docker 容器（`DockerRuntime`）+ 独立 SWE-bench 工作区（`workspace.py`）
- **模型通道**：统一 `Model` Protocol，已适配 DeepSeek、Gemini、Groq、OpenRouter、Magpie 与通用 OpenAI-compatible 接口

固化这个 Baseline 的目的很明确：**后续每引入一项论文优化（M3 ACI 工具、M4 上下文压缩与 Prefix Cache 权衡），都以它为严格对照组做消融实验。**

---

## 2. 核心模块与数据流（5 分钟捡回代码）

```text
CLI (cli.py)
  ├─ 参数校验 & Fail-Fast（缺少 API Key / 非法预算直接退出，不启动容器）
  ├─ ToolRegistry (tools/registry.py) ──> 单一 Schema 来源 (specs())
  ├─ Model Adapter (deepseek.py / gemini.py / openai_compatible.py ...)
  ├─ DockerRuntime (runtime/docker.py) ──> 隔离沙箱执行
  └─ Agent.run() (agent.py) + TrajectoryRecorder (trajectory.py)
```

### 2.1 `contracts.py`：核心数据结构与终止语义
- `Task` / `ToolCall` / `ToolResult` / `Usage`（含 `input_tokens`、`cache_read_tokens`、`output_tokens`、`cost_usd`）
- `ModelFinishReason`：把模型返回显式分成 4 类：
  - `FINISHED`：正常结束或正常工具调用
  - `OUTPUT_TRUNCATED`：输出达到上限被截断（如 DeepSeek `length` / Gemini `MAX_TOKENS`）
  - `PROVIDER_STOPPED`：厂商拒绝或异常中止（如 `content_filter`、`SAFETY`、空 `candidates`）
  - `INVALID_ACTION`：模型格式错误（非法 JSON 参数、一次返回多个并行工具调用、Gemini `MALFORMED_FUNCTION_CALL`）
- `StopReason`：任务级终止原因，覆盖 `MODEL_FINISHED`、`OUTPUT_TRUNCATED`、`PROVIDER_STOPPED`、`MAX_STEPS`、`BUDGET_EXHAUSTED`、`BUDGET_UNKNOWN`、`TOKEN_LIMIT_EXHAUSTED`、`TOKEN_USAGE_UNKNOWN`、`MODEL_ERROR`、`RUNTIME_ERROR`。

### 2.2 `agent.py`：主循环与容错优先级
每一轮 `while steps < self.max_steps` 严格遵守以下检查顺序：

1. **预算前置熔断**：若配置了 USD `budget`，先检查 `_budget_stop_reason`；如果累计 `cost_usd` 变为 `None`，采用 **Fail-Closed** 策略返回 `BUDGET_UNKNOWN`，避免失控扣费。
2. **调用模型并累计账单**：无论响应是否合法，只要厂商返回了 `usage`，**先合并计费再校验格式**（防止格式错误白烧钱却没记账）。
3. **不可恢复终止优先**：
   - 先判 `OUTPUT_TRUNCATED`（截断的工具 JSON 绝不能执行）。
   - 再判 `PROVIDER_STOPPED`（安全拦截/服务中止无法靠提示词自愈）。
4. **Run-level Token 软上限检查（`max_total_tokens`）**：
   - 在已累计本轮 usage 之后、派发下一轮动作之前检查。
   - 若开启上限但 `input_tokens` 或 `output_tokens` 缺失，Fail-Closed 返回 `TOKEN_USAGE_UNKNOWN`。
   - **细节**：如果本轮模型已经给出最终文本回答（`tool_call is None` 且非 `INVALID_ACTION`），即使刚过软上限也不作废该回答；但若还要继续调工具或重试，则立即以 `TOKEN_LIMIT_EXHAUSTED` 终止。
   - 注意：`cache_read_tokens` 是 `input_tokens` 的子集，累计总 Token 时不重复计算。
5. **有界自愈（Bounded Recovery）**：
   - `INVALID_ACTION`：将诊断信息包装成反馈消息追加进 `history`，**`steps += 1`** 后进入下一轮（保证重试受 `max_steps` 和 `max_total_tokens` 双重约束）。
   - 未知工具（`tool_registry.get(...) is None`）：构造带 `error="unknown tool: ..."` 的 `ToolResult` 回传模型，同样计入步数。
6. **沙箱执行与异常隔离**：调用 `tool.execute(...)`，若底层 runtime 抛出非预期异常，返回 `RUNTIME_ERROR`。

### 2.3 `runtime/docker.py`：Docker 沙箱隔离策略
- 容器启动参数：`--network none`（断网）、`--memory 2g`、`--cpus 2`、`--pids-limit 256`（防 Fork 炸弹）、`-v workspace:/workspace`。
- 输出保护：单次命令 stdout/stderr 截断上限 `MAX_OUTPUT_CHARS = 100_000`。
- 超时处理：`subprocess.TimeoutExpired` 时主动调用 `self.close()` 强杀容器，防止僵尸进程残留，并返回 `timed_out=True`。

### 2.4 多模型适配中的关键工程细节
- **Gemini 轮次约束**：Gemini 要求 `user` / `model` 交替对话；当工具结果（以 `user` 角色传入）紧接 `INVALID_ACTION` 的反馈消息时，`_merge_consecutive_user_turns` 会合并相邻 `user` 轮次，避免 API 报 400。
- **DeepSeek / OpenAI-compatible 解析**：先提取 `usage`，再检查 `finish_reason` 与工具参数 JSON 合法性，确保异常响应不丢账单。

---

## 3. Baseline 的已知局限（为什么需要 M3 与 M4 演进）

1. **工具层过于原始**：仅有 `bash`，模型容易 `cat` 大文件撑爆上下文，或用 `sed`/重定向破坏 Python 缩进（将在 **M3 #8** 引入 SWE-agent ACI 受控工具 + AST 语法护栏解决）。
2. **上下文线性膨胀**：历史工具输出原封不动保留到结束，既烧钱又引发长轨迹指令衰减；而简单粗暴地每步截断又会打碎 Prefix Cache（将在 **M4 #9** 引入可插拔 Middleware + 周期性 Observation Masking 解决）。
3. **缺少端到端评测数据**：目前只有离线单测，缺少真实仓库 Bug 题的对照实验数据（将在 **M5 #10** 补齐）。

---

## 4. 阶段一面试高频题单（自问自答）

1. **为什么不用 LangGraph / DeerFlow，而是自己手写 Agent Loop？**
   - 答题要点：本项目定位是可控、可做消融实验的轻量级 Coding Agent Harness。手写 Loop 只有约 200 行，能精确控制「计费先于校验」「Fail-Closed 预算/Token 熔断」「截断与格式错误的差异化处理」以及后续的 Middleware 钩子，避免框架黑盒带来的隐式状态流转与调试成本。
2. **模型返回非法 JSON 工具参数时，为什么不直接抛异常终止，也不无限重试？**
   - 答题要点：区分 `PROVIDER_STOPPED`（安全策略拦截、服务异常，重试无意义，直接终止）与 `INVALID_ACTION`（模型偶发格式错、并行调多工具，可自愈）。将其转成诊断反馈写回对话，同时**保留已发生的账单并计入 `steps`**，受 `max_steps` 与 `max_total_tokens` 双重收敛。
3. **Token 上限检查为什么放在「模型返回后、工具执行前」？如果最后一轮刚超限但模型已经给出最终答案怎么处理？**
   - 答题要点：Token 是软上限（防止继续发起昂贵的后续轮次和工具执行）。如果模型已经给出最终答案且不再请求工具，作废该结果会白白浪费已支付的推理成本，因此允许该轮正常 `MODEL_FINISHED`；若仍需调工具或重试则拦截终止。
4. **`cache_read_tokens` 怎么参与 Token 统计和计费？**
   - 答题要点：`cache_read_tokens` 是 `input_tokens` 的子集，统计总消耗不能重复叠加；但在计费上缓存命中部分享受大幅折扣（如 1/10），这直接引出后续 M4 中「上下文压缩 vs 保持 Prefix Cache 稳定」的核心权衡。
5. **如果 Provider 没有返回 `usage`（为 `None`），你的预算/Token 管控怎么处理？**
   - 答题要点：采取 **Fail-Closed（故障关闭）** 原则。只要用户显式设置了 `budget` 或 `max_total_tokens`，一旦 `cost_usd` 或 token 统计变成 `None`，立刻以 `BUDGET_UNKNOWN` / `TOKEN_USAGE_UNKNOWN` 停止，宁可保守中断也不放任无上限扣费。
6. **Docker 沙箱做了哪些安全与资源隔离？命令超时怎么防止容器里残留死循环进程？**
   - 答题要点：启动时施加 `--network none`、`--memory 2g`、`--cpus 2`、`--pids-limit 256`；单次 `docker exec` 设 `process_timeout`，超时触发 `TimeoutExpired` 后直接 `docker rm -f` 销毁容器（比单纯杀宿主机 client 进程更彻底，避免容器内孤儿进程继续空转）。
7. **ToolRegistry 的 `specs()` 为什么强调单一数据源（Single Source of Truth）？**
   - 答题要点：每个工具的参数校验用 Pydantic `args_model` 定义，传给 LLM 的 Function Schema 直接从同一份 `ToolRegistry` 派生，保证「告诉模型的参数定义」与「运行时校验逻辑」永远一致，新增 ACI 工具时不会出现 Schema 漂移。
8. **你这个项目和 Claude Code / Codex 比有什么优势？**
   - 答题要点：明确边界——它不是去替代商业通用 CLI 产品，而是面向离线评测与策略实验的轻量可插拔 Harness；核心价值是把模型适配、工具护栏、上下文/Cache 策略、预算熔断和轨迹回放解耦，便于控制变量做定量消融实验，同时在实现中覆盖了与成熟 Harness 同源的核心工程难题。
