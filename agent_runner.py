"""GoS (Graph-of-States) harness adapter.

Usage (cwd = agents/gos):
    ../_rca_shared/.venv/bin/python agent_runner.py < payload.json

stdin : JSON {question, system_prompt, user_prompt, compress_system_prompt,
        compress_user_prompt, data_dir}. GoS uses its own prompts, so only
        ``question`` (alarm source) and ``data_dir`` (RCABench converted/ dir or RCA100
        sandbox, see ``_rca_shared/cases.py``) are read.
stdout: last line = JSON {output, trajectory, usage, graph_from, gos}.
        Everything else (GoS's own prints included) goes to stderr.
        output = the graph of the Commander's final type-1 answer (run.report_graph_of); with
        GOS_GRAPH_SOURCE=extract, or when that graph is missing or unusable, extract_answer's roots +
        path (graph_from says which).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent
if str(AGENTS) not in sys.path:
    sys.path.append(str(AGENTS))

from _rca_shared.events import EV, install_hooks  # noqa: E402
from _rca_shared.llm import load_env  # noqa: E402


class TrajectoryRecorder:
    """Wraps ``Completions.create`` (outermost) and flattens every call into OpenAI-format messages.

    Each call contributes its request messages followed by the assistant reply; every
    message carries ``agent`` = the event-log role active at call time (``gos:*``).
    """

    def __init__(self):
        self.messages: list[dict] = []
        self.calls = 0

    def install(self) -> None:
        from openai.resources.chat import completions as mod

        orig = mod.Completions.create
        rec = self

        def recorded(self_inner, *args, **kwargs):
            resp = orig(self_inner, *args, **kwargs)
            try:
                rec.add(EV.current_role(), kwargs.get("messages") or [], resp)
            except Exception as exc:  # noqa: BLE001  (recording must never break the call)
                print(f"[gos] trajectory record failed: {exc}", file=sys.stderr)
            return resp

        mod.Completions.create = recorded

    @staticmethod
    def _msg(m) -> dict:
        if isinstance(m, dict):
            d = dict(m)
        else:
            d = m.model_dump() if hasattr(m, "model_dump") else {"role": "unknown", "content": str(m)}
        return d

    def add(self, agent: str, request_messages, resp) -> None:
        self.calls += 1
        for m in request_messages:
            d = self._msg(m)
            d["agent"] = agent
            d["call_index"] = self.calls
            self.messages.append(d)
        choice = resp.choices[0].message if getattr(resp, "choices", None) else None
        out = {"role": "assistant", "content": (choice.content if choice else None) or "",
               "agent": agent, "call_index": self.calls}
        if choice is not None and getattr(choice, "tool_calls", None):
            out["tool_calls"] = [tc.model_dump() for tc in choice.tool_calls]
        self.messages.append(out)


def causal_graph(roots: list[str], path: list[list[str]]) -> dict:
    comps: list[str] = []
    for c in list(roots) + [x for hop in path for x in hop]:
        if c not in comps:
            comps.append(c)
    return {"nodes": [{"component": c, "state": []} for c in comps],
            "edges": [{"source": a, "target": b} for a, b in path],
            "root_causes": [{"component": r, "state": []} for r in roots],
            "component_to_service": {}}


def main() -> None:
    load_env(HERE)
    tracker = install_hooks()          # usage tracker + event log on the OpenAI SDK
    recorder = TrajectoryRecorder()
    recorder.install()

    payload = json.loads(sys.stdin.read())
    real_stdout = sys.stdout
    sys.stdout = sys.stderr            # GoS prints a lot; keep stdout for the result line
    try:
        from _rca_shared import cases
        data_dir = Path(payload["data_dir"])
        cases.use(data_dir)                        # RCABench or RCA100, from the sandbox's tables
        cases.set_question(payload.get("question", ""), payload.get("cairn_incident"))
        import run as gos           # noqa: E402  (chdir + GoS patches)

        res = gos.run_case(data_dir, payload.get("question", ""), cairn="cairn_incident" in payload)
        graph = res["graph"] or causal_graph(res["roots"], res["path"])
        result = {
            "output": json.dumps(graph, ensure_ascii=False),
            "trajectory": recorder.messages,
            "usage": tracker.get_usage(),
            "graph_from": res["graph_from"],
            "gos": {k: res[k] for k in ("answer", "report", "graph_from", "report_graph_raw", "roots", "path",
                                         "hypotheses", "n_nodes", "n_edges", "llm_calls",
                                         "tool_calls", "stage_calls", "extract_raw", "error",
                                         "alarm", "elapsed_s")},
        }
    finally:
        sys.stdout = real_stdout
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
