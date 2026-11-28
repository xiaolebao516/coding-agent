# 02. ACI 受控工具集与静态语法护栏（M3 演进教程与面试题单）

本文档对应 Issue #8（M3 阶段），基于 Princeton NLP 论文 *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*（NeurIPS 2024, arXiv:2405.15793），解释为什么从纯 `bash` 演进到受控 ACI 工具集，以及每个工程细节如何应对大厂面试深挖。

---

## 1. 为什么纯 `bash` 不够好？（演进动机）

在 M2.5 Baseline 中，Agent 只有 `BashTool`。在真实仓库修 Bug 时，纯 `bash` 会高频触发三类典型失败（Badcase）：

1. **上下文瞬间爆炸**：模型执行 `cat large_file.py` 或无限制 `grep -rn`，单步输出数万字符，不仅烧掉大量 Input Token，还会把早期的任务描述和关键报错稀释掉。
2. **静默语法破坏**：模型用 `sed -i`、`awk` 或 `python -c` 拼接字符串改文件，极易出现缩进错位、引号未转义或括号漏配，直到跑 `pytest` 才发现 `SyntaxError`，白白浪费 2–3 轮步数。
3. **修改歧义不可控**：当文件中存在多处相似代码片段时，盲目替换容易改错位置。

---

## 2. 本阶段实现的核心设计（`src/coding_agent/tools/aci.py`）

为了保持「所有代码操作都在 Docker 沙箱内完成、不直接读写宿主机文件系统」的安全边界，3 个新工具全部通过 `Runtime.run_process(["python", "-c", script, json_payload])` 在沙箱内执行，同时用 `build_tool_registry("bash" | "aci")` 保留基线对照开关。

### 2.1 `view_file(path, start_line, end_line)` —— 窗口化带行号阅读器
- **硬上限保护**：单次最多返回 `MAX_VIEW_WINDOW = 250` 行，即使模型传 `end_line=9999` 也会被自动截断，防止单步撑爆上下文。
- **可继续提示**：输出带绝对行号和总行数，并在底部提示剩余行数与下一次调用建议参数，减少模型瞎猜行号。
- **路径越界防护**：`Path.resolve()` 后校验目标路径必须位于 `/workspace` 内，作为 Docker 隔离之外的纵深防御。

### 2.2 `search_repo(pattern, path, max_results, regex)` —— 降噪有界检索
- 自动跳过 `.git`、`__pycache__`、`.venv`、`node_modules` 等噪音目录，跳过二进制文件。
- 限制最大返回条数（默认 30，上限 100）和单行最大字符数（200），输出规范的 `relpath:lineno: snippet` 格式。

### 2.3 `edit_file(path, old_str, new_str)` —— 唯一匹配替换 + Python AST 语法护栏
- **唯一性强制**：`old_str` 出现 0 次直接提示用 `view_file` 核对缩进；出现 >1 次时列出所有匹配行号并拒绝修改，要求提供更多上下文。
- **写入前静态校验（核心论文思想）**：对 `.py` 文件，在真正 `write_text` 落盘前先执行 `ast.parse(updated)`；若触发 `SyntaxError`，**原文件保持未修改**，直接把出错行号、列号、问题代码行和报错信息回传给模型，让模型在下一轮原地自愈。
- **修改后即时回显**：落盘成功后自动回显修改位置前后各 2 行带行号预览，省去模型额外调用一次 `view_file` 确认的步数。

---

## 3. 备选方案权衡（为什么选 Search-and-Replace 而不是整文件重写或 Unified Diff？）

- **否掉整文件重写（Full-file rewrite）**：大文件重写输出 Token 极贵、延迟高，且容易漏写未改动部分（模型偷懒写 `# ... existing code ...`）。
- **否掉原始 Unified Diff（`git apply`）**：弱/中等模型经常算错 hunk 头部行号（`@@ -42,7 +42,8 @@`），导致 `git apply` 频繁报 patch does not apply。
- **选择 Exact Search-and-Replace + AST 护栏**：模型只需要复制精确上下文片段即可唯一锚定，无需心算行号，容错率显著更高；配合 `ast.parse` 护栏兜底语法正确性。

---

## 4. 阶段二面试高频题单（自问自答）

1. **什么是 ACI（Agent-Computer Interface）？它和人类用的 CLI/IDE 有什么区别？**
   - 答题要点：出自 SWE-agent（NeurIPS 2024）。人类看屏幕有视觉滚动和 IDE 实时红线，而 LLM 的输入输出是线性 Token 流。ACI 的核心是把环境反馈改造成「有界、带定位锚点（行号）、出错即阻断并给出可操作修复提示」的结构化接口。
2. **为什么 `edit_file` 不在宿主机 Python 进程里直接 `Path.write_text`，而要包一层 `runtime.run_process(["python", "-c", ...])`？**
   - 答题要点：保证所有文件读写都经过 `Runtime` 抽象边界，在生产模式下完全约束在 Docker 容器内（遵守容器权限与路径映射），在离线测试模式下又能无缝替换成 `LocalWorkspaceRuntime` / `FakeRuntime`，不破坏架构分层。
3. **AST 语法护栏能防住所有错误吗？它的局限是什么？**
   - 答题要点：不能。`ast.parse` 只能拦截语法/缩进错误（`SyntaxError`），防不住运行时逻辑错、未定义变量（`NameError`）或导入路径错误；这些仍需要靠后续在沙箱中运行 `pytest` 回归测试来验证。它的价值在于**零成本、毫秒级挡掉最常见的低级编辑破坏，避免脏文件污染后续推理**。
4. **怎么设计消融实验证明 ACI 真的有效？**
   - 答题要点：固定同一批题目、同一模型、同一 `max_steps` 与预算上限，只切换 `build_tool_registry("bash")` vs `build_tool_registry("aci")`，对比：任务解决率（Resolved Rate）、平均步数、平均 Input Token（观察是否减少了 `cat` 导致的膨胀）以及语法错误重试次数。
