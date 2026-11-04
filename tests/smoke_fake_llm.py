"""Offline smoke test of the GoS harness port (no real LLM call is ever made).

    cd agents/gos
    ../_rca_shared/.venv/bin/python tests/smoke_fake_llm.py [--case <converted dir>] [--keep]
    ../_rca_shared/.venv/bin/python tests/smoke_fake_llm.py --dataset rca100 [--case <RCA100_CONV>/t001]

RCA100: the case runs from a harness-like sandbox (the 11 tables only, random directory name) with
the t001 question (``_rca_shared/tests/rca100_case.py``); the fake names RCA100 services.

1. Access unit check on the real case (prints a few rows of every method).
2. Runs ``agent_runner.py`` end-to-end in child processes whose ``openai...Completions.create`` is
   replaced by a canned fake BEFORE the harness hooks are installed. The fake answers every GoS
   stage with valid GoS JSON and drives trace/log/metric tool calls.
3. Asserts the output contract and the event log.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

GOS = Path(__file__).resolve().parents[1]
AGENTS = GOS.parent
DEFAULT_CASE = str(Path(os.environ.get("RCABENCH_ROOT", AGENTS.parent / "datasets" / "RCABench-500"))
                   / "ts0-mysql-container-kill-9t6n24" / "converted")
QUESTION = """The following API endpoints are experiencing SLO violations:
- HTTP POST http://ts-ui-dashboard:8080/api/v1/users/login
- HTTP GET http://ts-ui-dashboard:8080/api/v1/verifycode/generate
Find the root cause."""
# Services the fake names per dataset: the caller that shows the errors, the root it blames, the entry
# service, a metric of the root, log keywords; and the alarmed service the port must derive.
NAMES = {
    "rcabench": {"svc": "ts-auth-service", "root": "mysql", "entry": "ts-ui-dashboard",
                 "metric": "container.cpu.usage", "keywords": ["mysql", "SQL"], "alarm": "ts-ui-dashboard"},
    "rca100": {"svc": "checkout", "root": "payment", "entry": "frontend",
               "metric": "apm.service.legacy.error", "keywords": ["payment"], "alarm": "checkout"},
}
N = NAMES[os.environ.get("GOS_SMOKE_DATASET", "rcabench")]


# ─────────────────────────────── fake LLM ───────────────────────────────
def _completion(content: str | None):
    from openai.types.chat import ChatCompletion

    msg = {"role": "assistant", "content": content}
    return ChatCompletion.model_validate({
        "id": "fake", "object": "chat.completion", "created": 0, "model": "fake-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    })


def _text(m) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c or ""


def _tool_history(user: str) -> int:
    line = next((ln for ln in user.splitlines() if ln.startswith("Retrieve/tool call history:")), "")
    return line.count('{"tool":')


def fake_create(self, *args, **kwargs):
    return _fake_answer(kwargs)


def _fake_answer(kwargs):
    msgs = kwargs.get("messages") or []
    system = next((_text(m) for m in msgs if (m.get("role") if isinstance(m, dict) else "") == "system"), "")
    users = [_text(m) for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    user = users[-1] if users else ""

    # GoS stages
    if 'extract 1 core "symptom node"' in system:
        return _completion(json.dumps({"symptom_node": "ts-ui-dashboard SLO violation on login/verifycode",
                                       "isolated_evidence": ["Login and verify-code endpoints violate SLO in the incident window"]}))
    if "generate coarse-grained candidate hypotheses" in system:
        ev = re.findall(r"ID: (node-\d+)\.", user)
        edges = [{"src": e, "dst": "h001", "relation": "support"} for e in ev]
        return _completion(json.dumps({"candidates": [
            {"id": "h001", "label": "Database dependency failure", "confidence": 0.7, "why": "auth errors"},
            {"id": "h002", "label": "Gateway overload", "confidence": 0.2, "why": "weak"}], "edges": edges}))
    if "determine whether to call agents" in system:
        return _completion(json.dumps([{"expert_name": "Linux_Agent"}, {"expert_name": "DatabaseOperator"}]))
    if "Focus ONLY on two core judgments" in system:
        n = _tool_history(user)
        need = 2 if "Role: Service Operator" in system else 1
        return _completion(json.dumps({"type": 2, "decision": "tool_call" if n < need else "analyze"}))
    if "Generate content that fully matches the given decision" in system:
        n = _tool_history(user)
        if "'tool_call'" in user:
            if "Role: Service Operator" in system:
                call = ({"tool": "trace", "service": N["svc"]} if n == 0 else
                        {"tool": "log", "service": N["svc"], "keywords": N["keywords"], "limit": 5})
            else:
                call = {"tool": "metric", "service": N["root"], "metric_names": [N["metric"]]}
            return _completion(json.dumps({"type": 2, "decision": "tool_call", "content": json.dumps(call)}))
        return _completion(json.dumps({"type": 2, "decision": "analyze",
                                       "content": "ts-auth-service logs communications link failures to mysql; mysql series stopped."}))
    if 'mandatory top-level fields: "edit", "nodes", "edges"' in system:
        hyp = re.findall(r"'(node-\d+)': \{'type': 'Hypothesis'", user)
        top = hyp[0] if hyp else "node-3"
        return _completion(json.dumps({
            "edit": [{"node_id": top, "confidence": 0.9, "why": "mysql restart and auth SQL errors"}],
            "nodes": [{"id": "ev101", "label": "mysql container metrics stopped", "node_type": "Evidence"}],
            "edges": [{"src": "ev101", "dst": top, "relation": "support"}]}))
    if system.startswith("You need to generate a coherent text summarizing"):
        return _completion(json.dumps({"belief": "Database dependency failure (mysql) is the leading hypothesis."}))
    if "The central agent first determines the action based on the Report_Flag" in system:
        if os.environ.get("GOS_FAKE_NEVER_REPORT") == "1":
            return _completion(json.dumps({"type": 2, "candidates": []}))
        out = {"type": 1, "answer": f"{N['root']} failure",
               "report": f"{N['root']} failed; {N['svc']} lost its calls to it; {N['entry']} broke."}
        if os.environ.get("GOS_FAKE_NO_GRAPH") != "1":
            # upper-case name and an edge to an unknown service: report_graph_of must fix / drop them
            out["graph"] = {"root_causes": [{"component": N["root"].upper(), "state": ["UNAVAILABLE"]}],
                            "edges": [{"source": N["root"], "target": N["svc"]},
                                      {"source": N["svc"], "target": N["entry"]},
                                      {"source": N["root"], "target": "no-such-service"}],
                            "nodes": [{"component": N["svc"], "state": ["HIGH_ERROR_RATE"]},
                                      {"component": N["entry"], "state": "HIGH_LATENCY"}]}
        return _completion(json.dumps(out))
    if "Map the conclusion to the benchmark's output format" in user:
        return _completion(json.dumps({"roots": [N["root"]],
                                       "path": [[N["root"], N["svc"]], [N["svc"], N["entry"]]]}))
    raise AssertionError(f"fake LLM: unrecognised call; system={system[:200]!r} user={user[:200]!r}")


def child() -> None:
    from openai.resources.chat import completions as mod

    mod.Completions.create = fake_create        # BEFORE agent_runner installs the harness hooks
    sys.path.insert(0, str(GOS))
    os.chdir(GOS)
    import agent_runner

    agent_runner.main()


# ─────────────────────────────── checks ───────────────────────────────
def access_check(case: str, question: str) -> None:
    sys.path.insert(0, str(GOS))
    sys.path.insert(1, str(AGENTS))
    from _rca_shared import cases
    from access_rcabench import Access

    cases.use(case)
    cases.set_question(question)
    root, svc, entry = N["root"], N["svc"], N["entry"]
    assert cases.alarm_of(case) == N["alarm"], cases.alarm_of(case)
    a = Access(case)
    obs = a.list_data()
    print(f"[access] list_data: {len(obs)} services; {root} -> {obs.get(root)}")
    for s in (root, svc):
        for r in a.metric_stats(s)[:3]:
            print(f"[access] metric_stats({s}): {r}")
    for r in a.call_stats(caller=entry)[:3]:
        print(f"[access] call_stats(caller={entry}): {r}")
    for r in a.call_stats(callee=svc)[:2]:
        print(f"[access] call_stats(callee={svc}): {r}")
    for r in a.span_samples(svc, errors_only=True, limit=2):
        print(f"[access] span_samples: {r}")
    print(f"[access] onsets: {a.onsets([root, svc])}")
    for r in a.log_samples(svc, limit=2):
        print(f"[access] log_samples: {str(r)[:160]}")
    st = a.log_stats(svc)
    print(f"[access] log_stats: ref={st['warnplus_reference']} inc={st['warnplus_incident']} "
          f"top={[t['template'][:60] for t in st['top_templates'][:2]]}")
    assert obs and a.metric_stats(root) and a.call_stats(caller=entry)


def run_runner(case: str, keep: bool, never_report: bool = False,
               question: str = QUESTION, no_graph: bool = False,
               graph_source: str | None = None) -> tuple[dict, list[dict]]:
    ev_dir = tempfile.mkdtemp(prefix="gos_smoke_")
    env = dict(os.environ, RCA_EVENTS_DIR=ev_dir, OPENAI_API_KEY="fake-key-offline",
               OPENAI_BASE_URL="http://127.0.0.1:9/v1", RCA_MODEL="fake-model", RCA_ENV_FILE="/dev/null")
    if never_report:
        env["GOS_FAKE_NEVER_REPORT"] = "1"
    if no_graph:
        env["GOS_FAKE_NO_GRAPH"] = "1"
    env.pop("GOS_GRAPH_SOURCE", None)
    if graph_source:
        env["GOS_GRAPH_SOURCE"] = graph_source
    payload = {"question": question, "system_prompt": "", "user_prompt": question,
               "compress_system_prompt": "", "compress_user_prompt": "", "data_dir": case}
    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"],
                       input=json.dumps(payload), capture_output=True, text=True, cwd=str(GOS), env=env,
                       timeout=900)
    if p.returncode != 0:
        print(p.stderr[-4000:])
        raise SystemExit(f"child exited {p.returncode}")
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout must hold only the result line, got {len(lines)} lines"
    result = json.loads(lines[-1])
    events = [json.loads(ln) for ln in open(Path(ev_dir) / "events.jsonl")]
    print(f"[run] events dir: {ev_dir}{'' if keep else ' (temp)'}")
    return result, events


def _user_of(ev: dict) -> str:
    msgs = ev["request"]["messages"]
    return next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")


def check_common(result: dict, events: list[dict]) -> None:
    assert set(result) == {"output", "trajectory", "usage", "graph_from", "gos"}, set(result)
    graph = json.loads(result["output"])
    assert set(graph) >= {"nodes", "edges", "root_causes", "component_to_service"}
    assert graph["root_causes"] and graph["root_causes"][0]["component"] == N["root"], graph
    assert graph["edges"][0] == {"source": N["root"], "target": N["svc"]}
    # the graph of the Commander's final type-1 answer, cleaned: known services only, no extraction call
    assert result["graph_from"] == "report" and result["gos"]["graph_from"] == "report"
    assert graph["edges"] == [{"source": N["root"], "target": N["svc"]}, {"source": N["svc"], "target": N["entry"]}]
    assert graph["root_causes"] == [{"component": N["root"], "state": ["UNAVAILABLE"]}], graph
    states = {n["component"]: n["state"] for n in graph["nodes"]}
    assert states == {N["svc"]: ["HIGH_ERROR_RATE"], N["entry"]: ["HIGH_LATENCY"], N["root"]: ["UNAVAILABLE"]}, states
    traj = result["trajectory"]
    assert traj and all("agent" in m and m["role"] in ("system", "user", "assistant", "tool") for m in traj)
    assert result["usage"].get("total_tokens", 0) > 0 or result["usage"], result["usage"]

    kinds = {}
    for e in events:
        kinds[e["event"]] = kinds.get(e["event"], 0) + 1
    llm = [e for e in events if e["event"] == "llm_call"]
    tools = [e for e in events if e["event"] == "tool_call"]
    arts = [e for e in events if e["event"] == "artifact"]
    roles = sorted({e["role"] for e in llm})
    print(f"[run] events: {kinds}; llm roles: {roles}")
    assert llm and tools and arts
    assert all(e["role"] and e["role"].startswith("gos:") for e in llm), "every llm_call needs a gos:* role"
    for need in ("ingest_symptom", "ingest_hypotheses", "plan", "proposal", "report_or_refine", "belief_text"):
        assert any(need in r for r in roles), f"no llm_call for {need}"
    assert "gos:extract_answer" not in roles, "extract_answer runs only when the report has no usable graph"
    # report_or_refine: graph specification and the case's services, no medical illustration
    ror = [e for e in llm if e["role"] == "gos:SRE_Commander:report_or_refine"]
    sys_p = ror[0]["request"]["messages"][0]["content"]
    assert "Graph specifications (for type=1 JSON)" in sys_p and "cause -> effect" in sys_p
    for word in ("fever", "patient", "Patient", "Staphylococcus", "influenza", "WBC", "infection"):
        assert word not in sys_p, word
    u = _user_of(ror[0])
    assert f"The alarmed service is: {N['alarm']}" in u and f"names for the graph): " in u and N["root"] in u
    assert any(r.startswith("gos:expert:Linux_Agent:") for r in roles)
    assert any(r.startswith("gos:expert:DatabaseOperator:") for r in roles)
    # expert calls nest under the run-loop step
    exp = [e for e in llm if e["role"].startswith("gos:expert:")]
    assert all((e.get("parent_id") or "").startswith("gos_step") for e in exp), "expert calls must nest under a step"
    tnames = sorted({t["tool"] for t in tools})
    print(f"[run] tool calls: {len(tools)} {tnames}")
    assert {"gos.trace", "gos.log", "gos.metric"} <= set(tnames)
    assert all(t["args"].get("parsed") and t["result"] for t in tools)
    assert all(t["role"].startswith("gos:expert:") for t in tools)
    anames = {a["name"] for a in arts}
    assert {"belief_graph", "hypotheses", "fsm_state", "final_answer", "step_start", "expert_analysis"} <= anames, anames
    # no metric-name list in the tool prompt
    for e in exp:
        assert "Available metric names" not in _user_of(e)


def check_rca100(events: list[dict]) -> None:
    """RCA100 inputs: alarm and symptom from the RCA100 question, tools over the RCA100 tables."""
    art = {a["name"]: a["data"] for a in events if a["event"] == "artifact"}
    setup = art["gos_setup"]
    assert setup["alarm"] == "checkout", setup["alarm"]
    assert setup["symptom"].startswith("[Alert] checkout 错误次数告警 on service checkout during the alert window "
                                       "2026-04-25 05:18:12 UTC ~ 2026-04-25 05:28:12 UTC."), setup["symptom"]
    assert "[Alert] alerting operation: /oteldemo.CheckoutService/PlaceOrder (component checkout" in setup["symptom"]
    assert "payment" in setup["instances"] and "load-generator" not in setup["instances"]
    tools = {t["tool"]: t["result"] for t in events if t["event"] == "tool_call"}
    assert "apm.service.legacy.error" in tools["gos.metric"], tools["gos.metric"][:300]
    assert "-> payment" in tools["gos.trace"], tools["gos.trace"][:500]
    print(f"[run] RCA100 OK: alarm checkout, symptom from the alert, RCA100 metrics/traces in tool results")


def main() -> None:
    global N
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--dataset", choices=tuple(NAMES), default="rcabench")
    ap.add_argument("--case", default=None, help="RCABench converted/ dir, or RCA100 converted t001 dir")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    if args.child:
        child()
        return
    N = NAMES[args.dataset]
    os.environ["GOS_SMOKE_DATASET"] = args.dataset          # the children's fake answers
    if args.dataset == "rca100":
        sys.path.insert(0, str(AGENTS / "_rca_shared" / "tests"))
        import rca100_case

        case, question = str(rca100_case.sandbox(rca100_case.converted_case(args.case))), rca100_case.QUESTION
    else:
        case, question = args.case or DEFAULT_CASE, QUESTION
    print(f"[{args.dataset}] case dir: {case}")
    access_check(case, question)
    result, events = run_runner(case, args.keep, question=question)
    check_common(result, events)
    if args.dataset == "rca100":
        check_rca100(events)
    g = result["gos"]
    print(f"[run] output={result['output']}")
    print(f"[run] llm_calls={g['llm_calls']} tool_calls={g['tool_calls']} nodes={g['n_nodes']} "
          f"stage_calls={g['stage_calls']} usage={result['usage']}")
    result, events = run_runner(case, args.keep, question=question, no_graph=True)
    assert result["graph_from"] == "extract_answer", result["graph_from"]
    assert any(e["event"] == "llm_call" and e["role"] == "gos:extract_answer" for e in events)
    g = json.loads(result["output"])
    assert [r["component"] for r in g["root_causes"]] == [N["root"]] and g["edges"][0] == {"source": N["root"], "target": N["svc"]}
    print("[no-graph] OK: a type-1 answer without a graph falls back to extract_answer")
    result, events = run_runner(case, args.keep, question=question, graph_source="extract")
    assert result["graph_from"] == "extract_answer", result["graph_from"]
    assert any(e["event"] == "llm_call" and e["role"] == "gos:extract_answer" for e in events)
    ror = [e for e in events if e["event"] == "llm_call" and e["role"] == "gos:SRE_Commander:report_or_refine"]
    assert ror and "Graph specifications (for type=1 JSON)" not in ror[0]["request"]["messages"][0]["content"]
    g = json.loads(result["output"])
    assert [r["component"] for r in g["root_causes"]] == [N["root"]] and all(n["state"] == [] for n in g["nodes"])
    print("[extract] OK: GOS_GRAPH_SOURCE=extract keeps upstream's report prompt and outputs extract_answer's graph")
    result, events = run_runner(case, args.keep, never_report=True, question=question)
    assert json.loads(result["output"])["root_causes"] == []
    assert sum(e["event"] == "artifact" and e.get("name") == "step_start" for e in events) == 3
    print("[step-limit] OK: no final report after 3 steps becomes an empty agent answer")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
