# coding-agent

一个面向代码修复与策略消融实验的轻量级 **Coding Agent Harness**（Python 3.12，零重型 Agent 框架黑盒依赖）。

项目定位不是替代 Claude Code / Codex 等商业终端产品，而是将 **执行沙箱、多模型协议适配、ACI 受控工具护栏、缓存感知上下文中间件、预算/Token 熔断与独立验证评测** 解耦为可插拔模块，用于在统一预算与隔离 Docker 环境下对前沿 Agent 论文优化做可复现的单变量消融实验。

---

## 核心特性与架构演进

```text
CLI / Eval Runner (cli.py / eval/runner.py)
  ├─ ToolRegistry (Single Source of Truth)
  │    ├─ Baseline: bash
  │    └─ ACI (M3): bash + view_file + search_repo + edit_file (AST 语法护栏)
  ├─ MiddlewareModel (M4 可插拔中间件链, Decorator + Chain of Responsibility)
  │    ├─ ContextMaskingMiddleware: full | per_step_masking | epoch_masking (保护 Prefix Cache)
  │    └─ EventReminderMiddleware: 尾部事件驱动纠偏提醒（只读循环 / 重复失败 / 步数预警）
  ├─ Model Adapters: DeepSeek / Gemini / Groq / OpenRouter / Magpie / OpenAI-compatible
  ├─ Agent Core Loop (agent.py)
  │    ├─ 先计费后校验、Fail-Closed USD Budget & Run-level Token Cap
  │    └─ 区分不可恢复 PROVIDER_STOPPED 与有界自愈 INVALID_ACTION
  └─ DockerRuntime (runtime/docker.py) + Independent Verify Workspace (eval/runner.py)
       ├─ --network none, --memory 2g, --cpus 2, --pids-limit 256, 超时强杀回收
       └─ 独立验证工作区：仅同步非测试文件变更，拦截修改测试导致的假通过（Test Tampering）
```

### 护栏与安全边界的明确区分

- **面向模型的辅助护栏（Ergonomic Guardrail）**：`edit_file` 中的 `ast.parse` 语法检查用于零成本、毫秒级拦截缩进/语法错，减少模型无效步数；它**不是**安全边界（模型在开放 `bash` 时仍可通过 shell 修改文件）。
- **安全与评测隔离边界（Security & Eval Boundary）**：由 `DockerRuntime`（断网、CPU/内存/PID 限额、超时强杀）与 `eval/runner.py` 的**双工作区隔离验证**（Agent 在 `agent_workspace` 运行，验证时仅将非受保护源码变更同步到全新 `verify_workspace` 再运行 `verify_command`，并记录 `tampered_test_files`）共同保证。

### 论文优化与工程落地

1. **受控 ACI 与静态语法护栏（M3）**
   - 参考 Princeton NLP *[SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)*（NeurIPS 2024）。
2. **周期性 Observation Masking 与 Prefix Cache 权衡（M4）**
   - 参考 JetBrains Research *[The Complexity Trap](https://arxiv.org/abs/2508.21433)*（NeurIPS 2025 DL4Code）与 *[Cache Engineering for Agents](https://arxiv.org/html/2610.28577v1)*（2026）。
3. **尾部事件驱动防衰减提醒（M4）**
   - 参考 *[Building Effective AI Coding Agents for the Terminal](https://arxiv.org/abs/2603.05344)*（OPENDEV, 2026）。

---

## 快速开始与离线测试

```bash
pip install -e ".[dev]"
pytest -q
```

---

## 本地消融实验与复现（M5）

详细操作与 Badcase 归因方法见 [`docs/04-eval-and-badcase.md`](docs/04-eval-and-badcase.md)。5 组消融配置严格满足单变量递进：

```bash
export DEEPSEEK_API_KEY="sk-..."

# 1. 在统一预算、双工作区隔离验证与异常断点续跑 (--resume) 下运行 5 组单变量消融
python -m coding_agent.eval.runner \
  --manifest eval/sample_manifest.json \
  --profiles baseline aci aci_per_step aci_epoch aci_epoch_remind \
  --provider deepseek \
  --model deepseek-chat \
  --max-steps 15 \
  --budget 0.5 \
  --output-dir artifacts/eval_runs \
  --resume

# 2. 汇总 .eval.json 并输出 Markdown 消融对比表
python -m coding_agent.eval.analyze --results-dir artifacts/eval_runs
```

### 消融实验结果（本地实测后替换下表）

| Profile | Tasks | Resolved Rate | Runner Errors | Avg Steps | Avg Input Tokens | Prefix Cache Hit | Prefix Invalidations/Task | Avg Cost (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | _待本地实测_ | - | - | - | - | - | - | - |
| `aci` | _待本地实测_ | - | - | - | - | - | - | - |
| `aci_per_step` | _待本地实测_ | - | - | - | - | - | - | - |
| `aci_epoch` | _待本地实测_ | - | - | - | - | - | - | - |
| `aci_epoch_remind` | _待本地实测_ | - | - | - | - | - | - | - |

---

## 演进教程与面试准备文档

- [01. Baseline Harness 架构梳理与面试题单（M1–M2.5）](docs/01-baseline-harness.md)
- [02. ACI 受控工具集与静态语法护栏（M3）](docs/02-aci-tools.md)
- [03. 可插拔 Middleware 链：Observation Masking、Prefix Cache 与事件提醒（M4）](docs/03-context-and-cache.md)
- [04. 本地消融实验操作指南与 Badcase 归因手册（M5）](docs/04-eval-and-badcase.md)
- [05. 面试总题单与简历写法模板（M6）](docs/05-interview-prep.md)
