# 04. 本地消融实验操作指南与 Badcase 归因手册（M5）

本文档对应 Issue #10（M5 阶段），用于指导你在本地用低成本（约 20–30 元 DeepSeek API 额度）跑出 **4 组策略消融对照实验**，并沉淀真实数据与 Badcase 复盘到 `README.md` 和简历中。

---

## 1. 四组消融对照配置（`src/coding_agent/eval/profiles.py`）

| Profile | 工具集 | 上下文策略 | 事件提醒 | 验证假设 |
| :--- | :--- | :--- | :---: | :--- |
| `baseline` | `bash` | `full` | 否 | M2.5 基线对照 |
| `aci` | `aci`（+`view_file`/`search_repo`/`edit_file`） | `full` | 否 | 验证 SWE-agent 受控工具与 AST 护栏对步数和解决率的提升 |
| `aci_per_step` | `aci` | `per_step_masking` | 是 | 验证 *The Complexity Trap* 每步折叠旧 Observation 的降本效果，同时观测 Prefix Cache 命中率下降 |
| `aci_epoch` | `aci` | `epoch_masking`（每 4 步推进水位线） | 是 | 验证分代水位线能否在维持低 Token 消耗的同时恢复 Prefix Cache 命中率、进一步压降真实 USD 成本 |

---

## 2. 本地评测操作步骤

### 第一步：准备 12–15 道真实 Python Bug 题清单
参考 `eval/sample_manifest.json` 编写你自己的 `eval/my_tasks.json`：
- **推荐来源**：从 SWE-bench Lite / Verified（如 `pytest`、`requests`、`sympy`、`scikit-learn` 等中等难度单文件/双文件 Python Bug）中挑 **12–15 道** 已验证 `verify_command` 在初始仓库必挂、打完标准补丁必过的题目。
- **隔离保证**：`runner.py` 每次运行前都会用 `shutil.copytree` 把原始 `workspace` 复制到独立临时目录再挂载进 Docker，因此 `baseline`、`aci`、`aci_per_step`、`aci_epoch` 之间绝对互不污染。

### 第二步：一键运行消融实验
```bash
export DEEPSEEK_API_KEY="sk-..."

python -m coding_agent.eval.runner \
  --manifest eval/my_tasks.json \
  --profiles baseline aci aci_per_step aci_epoch \
  --provider deepseek \
  --model deepseek-chat \
  --max-steps 15 \
  --budget 0.5 \
  --output-dir artifacts/eval_runs
```

### 第三步：汇总导出 Markdown 对比表
跑完后终端会自动打印对比表，也可以随时重新汇总：
```bash
python -m coding_agent.eval.analyze --results-dir artifacts/eval_runs
```
将输出的 Markdown 表格直接贴回 `README.md` 替换占位数据。

---

## 3. Badcase 归因分类模板（面试官最爱追问）

对所有 `resolved: false` 的任务，打开对应的 `*.trajectory.json`，按以下 5 类归因并统计占比，挑出 **2–3 个最有代表性的 Badcase** 写进 `README.md`：

1. **定位失败（Localization Failure）**：Agent 没找到真正出 Bug 的文件/函数，在无关文件里打转。
2. **编辑破坏（Edit/Syntax Failure）**：在 `baseline` 下因 `sed`/引号转义破坏语法；记录切换到 `aci` 后 AST 护栏是否成功拦截并修复。
3. **上下文膨胀 / 缓存抖动（Context Bloat / Cache Thrashing）**：长 `pytest` 堆栈或大文件输出导致后期推理跑偏，或 `per_step_masking` 触发过多 `prefix_invalidations` 导致成本高于预期。
4. **死循环 / 预算耗尽（Loop / Step Exhaustion）**：反复执行相同失败命令触发 `MAX_STEPS` 或 `BUDGET_EXHAUSTED`；记录 `EventReminderMiddleware` 是否打断了循环。
5. **环境 / 测试验证误判（Env / Flaky Test）**：依赖缺失、超时或测试命令与目标版本不匹配。

---

## 4. 阶段四面试高频题单（自问自答）

1. **你是怎么保证消融实验公平、可复现的？**
   - 答题要点：统一题目清单、统一底层模型版本、统一 `max_steps` 与 `budget` 上限、统一 Docker 镜像与断网沙箱；每次 `(profile, task)` 运行前都从只读源仓库 `copytree` 一份全新隔离副本，避免前一组修改污染后一组；最终统一用 `verify_command` 退出码判定 `resolved`，不靠人工肉眼打分。
2. **除了最终通过率（Pass@1），你还盯哪些过程指标？为什么？**
   - 答题要点：盯 4 个过程指标——① 平均步数（反映工具效率与定位收敛速度）；② 平均 Input Token 与字符压缩比（反映上下文控制）；③ `cache_read_tokens / input_tokens` 与 `prefix_invalidations`（反映缓存工程收益，避免出现「Token 降了但账单升了」的伪优化）；④ `INVALID_ACTION` 触发与自愈率（反映模型协议与 Schema 健壮性）。
3. **跑消融实验时有没有遇到「某项优化反而在某些题上变差」的情况？怎么分析的？**
   - 答题要点（结合你本地实测数据作答）：例如在极短任务（3–4 步解决）中，ACI 的多工具 Schema 本身会略微增加首轮 Prompt Token；或者 `per_step_masking` 在频繁回看早期报错堆栈的题目上会迫使模型重新 `view_file`，通过把 `keep_recent_observations` 调优或切换 `epoch_masking` 取得了更好的成本-准确率平衡。
