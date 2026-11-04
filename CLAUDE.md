# GoS (Graph of States)

把 [Graph of States](https://github.com/gaorch85/Graph-of-States)（fork 自 commit `5fa52a1`，MIT，ICML 2026，arXiv 2603.21250）接到 harness 上做微服务根因定位。上游的 agents、belief graph、FSM 不变：SRE Commander 维护信念图，调用 Linux_Agent、DatabaseOperator 等专家取证，最多 3 步（`configs/Agent_Configs.yaml` 的 `session_max_steps`）。

## 文件

| 文件 | 作用 |
|---|---|
| `agent_runner.py` | harness 入口：stdin/stdout 协议、trajectory |
| `run.py` | port 库：LLM transport、补丁、stage 和事件埋点、`run_case`、`report_graph_of`、`extract_answer` |
| `telemetry.py` | 专家工具 trace / log / metric（`OpsTelemetry`），取代上游的 shell / log / metric |
| `access_rcabench.py` | DuckDB `Access`，指标、调用边、日志的统计和 flag 规则（docstring 有阈值） |
| `prompts/Micro/`, `prompts/report_graph.py` | 微服务提示词；带传播图的 report-or-refine 提示词 |

上游代码但运行路径不用：`main.py`、`Run/`、`baselines/`、`prompts/baselines/`、`prompts/DiagnosisArena/`、`prompts/FailureDiagnosis-IT/`、`evaluate_*.py`、`requirements.txt`（上游自己的 pin，我们不安装）。相对上游的唯一修改是 `prompts/prompt_generation.py` 第 166 行的引号（兼容 Python 3.11，文本不变）和新增的 port 文件。

## Harness 如何驱动

配置 `gos-rcabench`、`gos-rca100`：cwd `agents/gos`，命令 `agents/_rca_shared/.venv/bin/python agent_runner.py`，一个 case 一个进程。

- stdin：JSON payload。只读 `question`（告警来源）和 `data_dir`；harness 的 system/user/compress 提示词被忽略，GoS 用自己的提示词。
- stdout：只有最后一行 JSON，键为 `output`（CausalGraph 字符串）、`trajectory`、`usage`、`graph_from`（`report` / `extract_answer` / `none`）、`gos`。GoS 自己的 print 全部进 stderr。
- 事件：`$RCA_EVENTS_DIR/events.jsonl`，LLM role 为 `gos:SRE_Commander:<stage>`、`gos:expert:<Expert>:decision|content`、`gos:belief_text`、`gos:extract_answer`、`gos:json_repair`；工具 `gos.trace`、`gos.log`、`gos.metric`。

## 环境

共用 `agents/_rca_shared/.venv`，构建方式见 `../_rca_shared/CLAUDE.md`（`uv pip install --python agents/_rca_shared/.venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r agents/_rca_shared/requirements.txt`）。必须与 `_rca_shared` 同级。`cp .env.example .env` 后填 `OPENAI_API_KEY`，也可用 `_rca_shared/.env` 或 `RCA_ENV_FILE`。

## 数据集

数据层都走 `_rca_shared/cases.py`，由沙箱的表决定数据集。

| 项 | RCABench（`gos-rcabench`） | RCA100（`gos-rca100`） |
|---|---|---|
| 图来源 `GOS_GRAPH_SOURCE` | `extract`：上游 report-or-refine 提示词，事后由 `extract_answer` 映射；节点 state 为空 | `report`（默认）：Commander type-1 回答自带 `graph`，缺失或不可用时回退 `extract_answer` |
| 症状首行 | `[Alert] SLO violation on service X during the incident window ...` | `[Alert] <alert title> on service X during the alert window ...` |
| 单条告警 | `[Alert] violated operation: <endpoint> ...` | `[Alert] alerting operation: <operation> ...` |
| metric 工具的指标名 | RCABench 原名 | `<entity_set>.<metric>`，operation 序列带 `{operation="..."}` |
| 图节点 state | `SERVICE_STATES["rcabench"]`（`rca.yaml`） | `SERVICE_STATES["rca100"]`（`rca100.yaml`） |

`report_graph_of` 清洗模型的图：名字大小写不敏感地匹配服务列表，丢弃未知名字、自环和重复边，把根因和边端点补进节点。

## Smoke（不调用真实 LLM）

```bash
cd agents/gos
../_rca_shared/.venv/bin/python tests/smoke_fake_llm.py [--case <converted dir>] [--keep]
../_rca_shared/.venv/bin/python tests/smoke_fake_llm.py --dataset rca100 [--case <RCA100_CONV>/t001]
```

RCABench 默认 case 为 `${RCABENCH_ROOT:-<repo>/datasets/RCABench-500}/ts0-mysql-container-kill-9t6n24/converted`。RCA100 的转换方式见 `../_rca_shared/CLAUDE.md`。依次验证正常运行、无 graph 回退、`GOS_GRAPH_SOURCE=extract`、3 步上限，成功时最后一行是 `SMOKE OK`。

## 注意

- 到 3 步仍返回 `type: 2` 时，case 以空答案结束（`root_causes: []`），算 agent 失败，不是基础设施失败。
- LLM 传输失败（`_complete` 3 次尝试后）抛异常，进程非零退出，harness 记基础设施失败。`KeyError`、`ValueError` 等 GoS 逻辑错误则结束 case 并打印空结果，错误写在 `gos.error`。
- `run.py` 有几个容错补丁（悬空边跳过、hypothesis 默认 level、回复归一化、一次 JSON 修复），它们只防崩溃，不做分析。
- RCA100 的 APM 计数器（`error_count`、`slow_count`）只在非零时上报，`Access` 会把它们标为 `CHANGED`，点之间的空隙也可能触发 `SERIES_STOPPED`。规则未针对 RCA100 调整。
- `extract` 和 `report` 两种图来源不同，不要跨设置比较结果。
- `utils/public_function.py` 的 `api_key = "Your-API-Key"` 是上游占位符，运行时已被 `run.py` 替换。
- `main.py` 里有上游作者的本地路径（`/data/luoyu6/...`），不在运行路径上。
