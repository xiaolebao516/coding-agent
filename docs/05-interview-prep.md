# 05. 面试总题单与简历写法模板（M6）

本文档汇总 `coding-agent` 四个演进阶段的简历表达模板、高频追问索引与「对比成熟 Harness」的标准应答框架，供日常实习面试前集中复习。

---

## 1. 简历项目描述模板（完成本地 12–15 题评测后填入实测数字）

**项目名称**：`coding-agent` —— 轻量级可插拔 Coding Agent Harness 与上下文/缓存消融框架
**技术栈**：Python 3.12、Docker、Pydantic v2、DeepSeek / Gemini / OpenAI-compatible API、GitHub Actions CI

- **核心闭环与容错架构**：不依赖重型黑盒框架，从零实现 Agent Loop、统一多模型协议适配、Docker 隔离沙箱（断网、CPU/内存/PID 限额与超时强杀）及 Trajectory 录制；设计先计费后校验、Fail-Closed 预算/Token 熔断机制，区分不可恢复 `PROVIDER_STOPPED` 与有界自愈 `INVALID_ACTION`，配套 180+ 离线回归单测。
- **ACI 受控工具与语法护栏（SWE-agent 演进）**：基于单一 Schema 数据源 `ToolRegistry` 实现 `view_file`（250 行窗口分页）、`search_repo`（降噪有界检索）与 `edit_file`（唯一匹配替换 + 写入前 Python `ast.parse` 语法拦截），在本地 _[N]_ 题 Python Bug 评测集上相比纯 `bash` 基线将解决率从 _[X%]_ 提升至 _[Y%]_，平均步数降低 _[Z%]_。
- **缓存感知上下文中间件（The Complexity Trap + Prefix Cache 权衡）**：以装饰器/责任链模式抽离 `MiddlewareModel`，对比全量上下文、每步 Observation Masking 与分代水位线 `epoch_masking`，并结合尾部 `EventReminderMiddleware` 防长轨迹衰减；在保护服务端 Prompt Prefix Cache 的前提下，将单任务平均 Input Token 降低 _[A%]_、Prefix Cache 命中率维持在 _[B%]_、单任务平均成本下降 _[C%]_。

---

## 2. 面试高频追问总索引（按考察维度）

### A. 项目定位与架构选型
1. 你的项目和 Claude Code / Codex 比有什么优势？为什么不直接用现成的？ -> 见 [`docs/01-baseline-harness.md`](01-baseline-harness.md) 第 4 节第 8 题。
2. 为什么不用 LangGraph / DeerFlow 而选择手写 Agent Loop + `MiddlewareModel`？ -> 见 [`docs/01-baseline-harness.md`](01-baseline-harness.md) 第 4 节第 1 题 & [`docs/03-context-and-cache.md`](03-context-and-cache.md) 第 3 节第 4 题。

### B. 容错、计费与安全隔离（后端/Agent 工程必问）
3. `PROVIDER_STOPPED`、`OUTPUT_TRUNCATED` 和 `INVALID_ACTION` 的处理优先级和计费原则是什么？ -> 见 [`docs/01-baseline-harness.md`](01-baseline-harness.md) 第 2.2 节。
4. 为什么预算和 Token 上限采用 Fail-Closed？最后一轮刚超软上限但已产出最终答案怎么处理？ -> 见 [`docs/01-baseline-harness.md`](01-baseline-harness.md) 第 4 节第 3、5 题。
5. Docker 沙箱怎么防 Fork 炸弹、死循环进程残留与路径逃逸？ -> 见 [`docs/01-baseline-harness.md`](01-baseline-harness.md) 第 2.3 节 & [`docs/02-aci-tools.md`](02-aci-tools.md) 第 2.1 节。

### C. 工具设计（ACI）与上下文/缓存工程（PE / Agent 深挖核心）
6. 为什么选 Exact Search-and-Replace + AST 语法护栏，而不是整文件重写或 `git apply`？ -> 见 [`docs/02-aci-tools.md`](02-aci-tools.md) 第 3 节。
7. 为什么用 Observation Masking 而不是每步 LLM Summary？ -> 见 [`docs/03-context-and-cache.md`](03-context-and-cache.md) 第 3 节第 1 题。
8. **杀手锏题**：为什么每步做上下文压缩有时会让真实账单不降反升？`epoch_masking` 是如何解决的？ -> 见 [`docs/03-context-and-cache.md`](03-context-and-cache.md) 第 3 节第 2 题。
9. `EventReminderMiddleware` 为什么追加在最新 Observation 尾部而不是改写 System Prompt？ -> 见 [`docs/03-context-and-cache.md`](03-context-and-cache.md) 第 3 节第 3 题。

### D. 评测闭环与 Badcase 迭代
10. 消融实验如何防止不同策略之间的工作区污染？除了 Pass@1 还看了哪些过程指标？ -> 见 [`docs/04-eval-and-badcase.md`](04-eval-and-badcase.md) 第 4 节第 1、2 题。
11. 举 2 个你在本地评测中亲手定位的真实 Badcase（现象 → Trajectory 根因 → 修复前后指标变化）。 -> 本地跑完实验后按 [`docs/04-eval-and-badcase.md`](04-eval-and-badcase.md) 第 3 节补充。

---

## 3. 你接下来的本地执行清单（3 步走）

1. **拉取最新 `main` 并跑通离线单测**：`git checkout main && git pull && pytest -q`。
2. **准备 12–15 道真实 Python Bug 题跑 4 组消融实验**：用 DeepSeek API 跑 `baseline` / `aci` / `aci_per_step` / `aci_epoch`，把 `analyze.py` 输出的真实表格填回 `README.md`。
3. **挑 2–3 个 Badcase 复盘并过一遍 5 篇 `docs/` 题单**：确保上面 11 个问题都能脱离文档流畅口述。
