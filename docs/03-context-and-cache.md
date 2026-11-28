# 03. 可插拔 Middleware 链：Observation Masking、Prefix Cache 权衡与事件提醒（M4 演进教程与面试题单）

本文档对应 Issue #9（M4 阶段），结合 2025–2026 年三项核心论文与工程实践，讲解如何在不破坏核心 `Agent` 状态机的前提下，引入可插拔上下文工程与缓存感知策略。

---

## 1. 论文与工程背景

1. **JetBrains Research, *The Complexity Trap***（NeurIPS 2025 DL4Code, arXiv:2508.21433）：
   - 在 SWE-agent 脚手架上实证表明：简单的 **Observation Masking**（保留近 $M$ 轮完整工具输出，把更早的冗长 stdout/stderr 替换为占位符，同时完整保留每步的 Reasoning/Action）就能把 Token 成本压降约 50% 且不明显掉点；相比之下，每步调用 LLM 做历史摘要不仅增加延迟和额外调用成本，还容易丢失精确行号与报错细节。
2. ***Cache Engineering for Agents***（arXiv:2610.28577）& **Manus 上下文工程实践**：
   - 大模型服务端的 Prompt Prefix Cache（KV Cache 复用）要求**输入前缀逐字节一致**。缓存命中部分的输入单价通常只有普通输入的 1/10。
   - **关键陷阱**：如果每走一步都把第 $t-M$ 轮的历史 Observation 改写成占位符，等于**每一轮都破坏了历史前缀**，导致 Prefix Cache 命中率归零——Token 数虽然少了，实际账单反而可能更高。
3. ***Building Effective AI Coding Agents for the Terminal***（OPENDEV, arXiv:2603.05344）：
   - 长轨迹中模型容易出现「指令衰减（Instruction Fade-out）」、陷入连续只读探索或反复执行同一条失败命令。论文采用 **Event-driven System Reminder**（尾部事件注入提醒），在不改动头部 System Prompt 前缀的前提下及时纠偏。

---

## 2. 架构设计（`src/coding_agent/middleware.py`）

### 2.1 为什么用 `MiddlewareModel`（装饰器 + 责任链），而不是把压缩逻辑写死在 `agent.py`？
- **开闭原则与零侵入**：`Agent.run()` 内部已经维护了严格的计费、Fail-Closed 熔断、`OUTPUT_TRUNCATED` 与 `INVALID_ACTION` 状态机不变量。通过实现与 `Model` 协议完全一致的 `MiddlewareModel` 包装器，在 `before_model` / `after_model` 钩子中串联任意中间件（类似 DeerFlow 2.0 的中间件链），**既能零风险复用全部 165+ 已有单测，又能对所有 Provider（DeepSeek / Gemini / Groq / OpenRouter）统一生效**。
- **不原地篡改 `history`**：`ContextMaskingMiddleware` 始终在浅拷贝上生成视图，保证 `Agent` 内部原始轨迹与录制日志完整无损。

### 2.2 三种可切换上下文策略（供消融实验对比）
- `full`：全量上下文（Baseline）。
- `per_step_masking`：每步滑动窗口折叠早于近 $M$ 轮的工具输出。上下文最短，但每步触发一次 `prefix_invalidations`。
- `epoch_masking`：引入**分代水位线（Watermark）**，每新增 $K$ 轮可折叠工具输出才推进一次水位线。在两次 Epoch 之间，历史前缀保持严格 Append-only，**兼顾上下文长度控制与中间轮次的 Prefix Cache 命中率**。

### 2.3 `EventReminderMiddleware`（尾部无损纠偏）
- 仅在最近一条 `role == "tool"` 的 Observation 尾部追加提醒，**绝不修改头部 Prompt**。
- 监听 3 类高发异常：连续 $N$ 步只读未修改（`view_file`/`search_repo`）、连续 2 次完全相同的失败调用、剩余步数 $\le 2$。

---

## 3. 阶段三面试高频题单（自问自答）

1. **为什么选择 Observation Masking 而不是用小模型每轮做 LLM Summary？**
   - 答题要点：引用 JetBrains *The Complexity Trap* 结论——在 Coding 任务中，历史工具输出（长堆栈、大段文件）占用了大部分 Token，而模型真正需要保留的是「我做过哪些动作和决策」。Observation Masking 零额外 LLM 调用、零延迟、不引入摘要幻觉；如果需要旧文件内容，模型可随时再次 `view_file`。
2. **做了上下文压缩后，为什么有时候账单反而变贵了？（高频深挖题）**
   - 答题要点：因为破坏了 Prompt Prefix Cache！如果每步都滑动修改中间消息，服务端无法命中前缀缓存，所有历史 Token 都要按全价重算 Prefill。我们的 `epoch_masking` 通过每 $K$ 步批量推进一次水位线，把 $N$ 次前缀失效降为 $\lfloor N/K \rfloor$ 次，并用 `MiddlewareStats` 对比 `char_reduction_ratio`、`prefix_invalidations` 和真实 `cache_hit_rate`。
3. **Event-driven Reminder 为什么追加在最新一条 Tool Observation 尾部，而不是动态修改 System Prompt？**
   - 答题要点：两个原因——① 修改头部 System Prompt 会 100% 打碎整条对话的 Prefix Cache；② 处于对话尾部的提醒离当前生成位置最近（Recency Bias），更能有效对抗长轨迹下的指令衰减。
4. **你的 `MiddlewareModel` 和 DeerFlow 2.0 的中间件链有什么异同？**
   - 答题要点：思想同源（通过 `before_model` / `after_model` 钩子把横切关注点从核心执行循环中解耦）；不同在于本项目保持极简纯 Python Protocol，没有引入 LangGraph 重型状态图依赖，更便于在统一评测预算下做单变量消融实验。
