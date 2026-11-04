"""Graph-of-States (GoS, ICML'26) on RCABench and RCA100: the port library.

Driven by ``agent_runner.py`` (harness stdin/stdout contract), one case per process.
GoS's own agents / belief graph / FSM / prompts run unchanged. The adaptation swaps the LLM transport (their
chatanywhere HTTPS client -> the OpenAI-compatible endpoint of ``_rca_shared.llm``),
the telemetry tools (host shell/log/metric -> microservice trace/log/metric,
``telemetry.py`` over ``access_rcabench.py``; RCABench or RCA100 through
``_rca_shared/cases.py``), the domain prompts
(``prompts/Micro/system_prompt.py``). Where the output graph comes from is set by
``GOS_GRAPH_SOURCE``:

- ``report`` (default; config gos-rca100): the Commander's report-type
  answer (type 1) also writes the fault propagation graph (``prompts/report_graph.py``), and
  the answer-extraction call (``extract_answer``) runs only when that graph is missing or
  invalid;
- ``extract`` (config gos-rcabench): upstream's
  report-or-refine prompt, and ``extract_answer`` maps GoS's final answer and report to root
  services + propagation path.

New in this port: harness event logging (role/parent per LLM call, tool calls,
belief-graph artifacts).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

GOS = str(Path(__file__).resolve().parent)
AGENTS = str(Path(GOS).parent)
if GOS not in sys.path:
    sys.path.insert(0, GOS)
if AGENTS not in sys.path:
    sys.path.append(AGENTS)            # for _rca_shared (appended: GoS's own `agents` package must win)
os.chdir(GOS)                          # GoS loads prompts/<dataset>/system_prompt.py relatively

import yaml                                     # noqa: E402
import utils.public_function as pf              # noqa: E402
import agents.central_agent as ca_mod           # noqa: E402
import agents.expert_agent as ea_mod            # noqa: E402
import belief.graph as bg_mod                   # noqa: E402
from belief.graph import BeliefGraph            # noqa: E402
from belief.fsm import BeliefFSM                # noqa: E402
from agents.central_agent import Central_Agent  # noqa: E402
from agents.expert_agent import Expert_Agent    # noqa: E402

from _rca_shared import llm as shared_llm       # noqa: E402
from _rca_shared import cases                   # noqa: E402
from _rca_shared.events import EV               # noqa: E402
from telemetry import OpsTelemetry              # noqa: E402
from prompts import report_graph                # noqa: E402

DOMAIN_CFG = {
    "expert_names": ["Linux_Agent", "NetworkOperator", "DatabaseOperator"],
    "expert_descriptions": {
        "Linux_Agent": "Service operator: investigates a suspected service with the trace tool (its outgoing calls, callers, error spans), then its logs and runtime/resource metrics, to validate or refute the hypothesis.",
        "NetworkOperator": "Network operator: validates network-level hypotheses (partition, packet loss, injected delay between two services) from vanished or timing-out call edges and per-edge latency.",
        "DatabaseOperator": "Database operator: validates database-related hypotheses (a service's database operations slowing/failing; shared-database contention across clients).",
    },
    "head_name": "SRE_Commander",
    "tool_names": ["Telemetry_Tools"],
    "tool_descriptions": {"Telemetry_Tools": "Experts query offline telemetry via three tools: trace (call graph of a service), log (WARN/ERROR lines of a service), metric (reference-vs-incident statistics of a service's metrics)."},
}
HEAD = DOMAIN_CFG["head_name"]

# ── per-case runtime context (GoS is single-threaded; one case per process) ──
CENTRAL_STAGES = ("ingest_symptom", "ingest_hypotheses", "plan", "proposal", "report_or_refine")
EXPERT_STAGES = ("expert_decision", "expert_content")


class _Ctx:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pending: str | None = None      # stage announced by the prompt constructor
        self.expert: str | None = None       # expert whose run() is active
        self.parent: str | None = None       # event parent id (ingest / step / final)
        self.step = 0
        self.llm_calls = 0
        self.stage_calls: dict[str, int] = {}
        self.graph = None
        self.fsm = None
        self.report_text: str | None = None  # last report_or_refine reply
        self.report_dict: dict | None = None  # ... as GoS parsed it


CTX = _Ctx()
_CLIENT = {}


def _client():
    if "c" not in _CLIENT:
        _CLIENT["c"] = shared_llm.make_client()
    return _CLIENT["c"]


def role_of(stage: str) -> str:
    if stage in CENTRAL_STAGES:
        return f"gos:{HEAD}:{stage}"
    if stage in EXPERT_STAGES:
        return f"gos:expert:{CTX.expert or 'unknown'}:{stage.split('_', 1)[1]}"
    return f"gos:{stage}"


def _complete(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, role: str) -> tuple[str, dict]:
    """One chat completion with a retry loop (3 attempts, retry on empty text).

    Crash policy: if the last attempt raised, the exception propagates.
    """
    s = shared_llm.settings()
    msgs = ([{"role": "system", "content": system_prompt.strip()}] if system_prompt and system_prompt.strip() else []) \
        + [{"role": "user", "content": user_prompt}]
    text, usage, last_exc = "", {}, None
    for attempt in range(3):
        try:
            with EV.scope(role=role, parent=CTX.parent):
                kw = dict(model=s["model"], messages=msgs, temperature=temperature, max_tokens=max_tokens)
                if s.get("extra_body"):
                    kw["extra_body"] = s["extra_body"]
                r = _client().chat.completions.create(**kw)
            CTX.llm_calls += 1
            last_exc = None
            text = (r.choices[0].message.content or "").strip()
            usage = r.usage.model_dump() if r.usage else {}
            if text:
                break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            text = ""
            print(f"[gos] LLM attempt {attempt + 1} failed ({role}): {type(exc).__name__}: {exc}", file=sys.stderr)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return text, usage


def _llm(user_prompt, model_path="deepseek", temperature=0.8, max_tokens=4096, system_prompt="",
         return_meta=False, stage=None, **_):
    stage = stage or CTX.pending or "unknown"
    CTX.pending = None
    CTX.stage_calls[stage] = CTX.stage_calls.get(stage, 0) + 1
    role = role_of(stage)
    user = user_prompt.strip()
    text, usage = _complete(system_prompt, user, temperature, max_tokens, role)
    if stage == "report_or_refine":
        CTX.report_text, CTX.report_dict = text, None
    meta = {"finish_reason": "stop", "model": shared_llm.settings()["model"], "temperature": temperature,
            "max_tokens": max_tokens, "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0), "response": text}
    return (text, meta) if return_meta else text


# ---- transport + tool patches (process-wide, identical behaviour for every case) ----
for mod in (pf, ca_mod, ea_mod, bg_mod):
    mod.llm_generate_response = _llm
ea_mod.Telemetry.from_case_resources = classmethod(
    lambda cls, case_id, resources: OpsTelemetry(resources["case_dir"], resources["alarm"], resources.get("events")))

# robustness guard (adaptation-side): the central agent's proposal JSON sometimes
# references a node id that does not exist (LLM slip); GoS raises and the whole case
# dies. Skip the dangling edge instead -- same class of guard as OpsAgent's tag repair.
_orig_add_edge = BeliefGraph.add_edge


def _tolerant_add_edge(self, src, dst, edge_type, attrs=None):
    if src not in self.nodes or dst not in self.nodes:
        print(f"[gos-adapter] dangling edge skipped: {src}->{dst} ({edge_type})")
        return
    return _orig_add_edge(self, src, dst, edge_type, attrs)


BeliefGraph.add_edge = _tolerant_add_edge

# Hypothesis nodes the LLM creates through an edit sometimes omit the "level" attribute,
# which GoS's own backtrack_frontier_chain then dereferences (KeyError: 'level').
# Default a missing level to the first hypothesis layer so the FSM keeps its semantics.
_orig_add_node = BeliefGraph.add_node


def _defaulted_add_node(self, node_type, label, score=1.0, attrs=None):
    attrs = dict(attrs or {})
    if node_type == "Hypothesis":
        attrs.setdefault("level", 1)
    return _orig_add_node(self, node_type, label, score, attrs)


BeliefGraph.add_node = _defaulted_add_node


# GoS dereferences LLM-produced JSON directly (item["expert_name"], hypo["confidence"],
# node attrs["level"]); a reply that deviates from the shape aborts the whole case. These
# shims normalise the reply at the parse boundary and let a malformed step be skipped --
# they add no analysis and change no decision rule.
def _normalise(obj, _depth=0):
    if isinstance(obj, dict):
        d = {k: _normalise(v, _depth + 1) for k, v in obj.items()}
        if ("label" in d or "why" in d or "node_id" in d) and "confidence" not in d:
            d["confidence"] = float(d.get("score", 0.5) or 0.5)
        return d
    if isinstance(obj, list):
        return [_normalise(v, _depth + 1) for v in obj]
    return obj


_parse_before_norm = pf.parse_json_response


def _normalising_parse(raw):
    return _normalise(_parse_before_norm(raw))


for _m in (pf, ca_mod, ea_mod, bg_mod):
    _m.parse_json_response = _normalising_parse

_orig_act = ca_mod.Central_Agent.act


def _tolerant_act(self, plan):
    items = plan if isinstance(plan, list) else [plan]
    clean = []
    for it in items:
        if isinstance(it, str):
            clean.append({"expert_name": it, "instruction": ""})     # bare expert name
        elif isinstance(it, dict) and it.get("expert_name"):
            clean.append(it)
    return _orig_act(self, clean)


ca_mod.Central_Agent.act = _tolerant_act

_orig_backtrack = ca_mod.Central_Agent.backtrack_frontier_chain


def _tolerant_backtrack(self, *a, **kw):
    try:
        return _orig_backtrack(self, *a, **kw)
    except (KeyError, TypeError) as e:      # malformed node bookkeeping: skip this backtrack
        print(f"[adapter] backtrack skipped ({type(e).__name__}: {e})")
        return None


ca_mod.Central_Agent.backtrack_frontier_chain = _tolerant_backtrack

# JSON repair guard (adaptation-side): GoS parses every LLM reply as JSON and raises on
# a malformed one, killing the case. Ask the model once to fix the JSON, then re-parse.
_orig_parse = pf.parse_json_response


def _tolerant_parse(raw):
    try:
        out = _orig_parse(raw)
    except Exception:
        fixed = _llm("Fix the following so that it is valid JSON. Output ONLY the JSON, no commentary:\n\n"
                     + str(raw)[:6000], temperature=0.0, max_tokens=4096, stage="json_repair")
        out = _orig_parse(fixed)
    if raw is not None and raw == CTX.report_text:     # the Commander's decision, as GoS reads it
        CTX.report_dict = out
    return out


for mod in (pf, ca_mod, ea_mod, bg_mod):
    mod.parse_json_response = _tolerant_parse


# ---- stage / event instrumentation (no behavioural change) ----
def _announce(fn, stage_of):
    def wrapped(*a, **kw):
        CTX.pending = stage_of(kw)
        return fn(*a, **kw)
    wrapped.__wrapped__ = fn
    return wrapped


# The report-or-refine prompt with the propagation graph and without upstream's medical examples
# (prompts/report_graph.py); with GOS_GRAPH_SOURCE=extract upstream's constructor stays in use.
GRAPH_SOURCE = os.environ.get("GOS_GRAPH_SOURCE", "report")
if GRAPH_SOURCE not in ("report", "extract"):
    raise SystemExit(f"GOS_GRAPH_SOURCE must be 'report' or 'extract', not {GRAPH_SOURCE!r}")
if GRAPH_SOURCE == "report":
    ca_mod.construct_report_or_refine_prompt = report_graph.construct_report_or_refine_prompt
for _name, _stage in (("construct_ingest_prompt_basic_node", "ingest_symptom"),
                      ("construct_ingest_prompt_L1_Hypo", "ingest_hypotheses"),
                      ("construct_call_expert_prompt", "plan"),
                      ("construct_generate_proposal_prompt", "proposal"),
                      ("construct_report_or_refine_prompt", "report_or_refine")):
    setattr(ca_mod, _name, _announce(getattr(ca_mod, _name), lambda kw, s=_stage: s))
ea_mod.construct_expert_analyze_prompt = _announce(
    ea_mod.construct_expert_analyze_prompt,
    lambda kw: "expert_decision" if kw.get("task_stage") == "decision_making" else "expert_content")

_orig_belief_text = BeliefGraph.generate_belief_text


def _belief_text(self):
    CTX.pending = "belief_text"
    return _orig_belief_text(self)


BeliefGraph.generate_belief_text = _belief_text


def _hypotheses(graph) -> list[dict]:
    return [{"id": nid, "label": n.get("label"), "score": n.get("score"), "level": (n.get("attrs") or {}).get("level")}
            for nid, n in graph.nodes.items() if n.get("type") == "Hypothesis"]


def emit_state(tag: str, **extra) -> None:
    """Belief graph snapshot, hypothesis list and FSM state as artifacts."""
    g, f = CTX.graph, CTX.fsm
    if g is None:
        return
    with EV.scope(role="gos:runtime", parent=CTX.parent):
        EV.artifact("belief_graph", {"tag": tag, "step": CTX.step, "graph": g.to_dict(), "belief": g.belief}, **extra)
        EV.artifact("hypotheses", {"tag": tag, "step": CTX.step, "hypotheses": _hypotheses(g)})
        if f is not None:
            EV.artifact("fsm_state", {"tag": tag, "step": CTX.step, "state": f.state, "history": f.history,
                                      "stage_steps": f._stage_steps, **extra})


_orig_tick = BeliefFSM.tick_step


def _tick(self, k=1):
    CTX.step += 1
    CTX.parent = EV.new_id(f"gos_step{CTX.step}_")
    with EV.scope(role="gos:runtime", parent=CTX.parent):
        EV.artifact("step_start", {"step": CTX.step, "step_id": CTX.parent})
    return _orig_tick(self, k)


BeliefFSM.tick_step = _tick

_orig_transit = BeliefFSM.maybe_transit


def _transit(self, G):
    advanced = _orig_transit(self, G)
    emit_state("after_step", advance=bool(advanced))
    return advanced


BeliefFSM.maybe_transit = _transit

_orig_expert_run = Expert_Agent.run


def _expert_run(self, frontier):
    prev, CTX.expert = CTX.expert, self.name
    try:
        with EV.scope(role=f"gos:expert:{self.name}", parent=CTX.parent):
            out = _orig_expert_run(self, frontier)
            EV.artifact("expert_analysis", {"step": CTX.step, "expert": self.name,
                                            "frontier": frontier, "analysis": out})
        return out
    finally:
        CTX.expert = prev


Expert_Agent.run = _expert_run


def symptom_text(case_dir: Path, alarm: str) -> str:
    """Incident symptom for the belief graph: alarmed service, window, violated operations.
    Alarms come from the harness question (cases.set_question), never from ground truth.
    RCABench: SLO-violated endpoints; RCA100: the fired alert (title, alerting operation)."""
    import duckdb
    from datetime import datetime, timezone

    t0, t1 = cases.incident_window(case_dir)
    f = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")  # noqa: E731
    con = duckdb.connect()
    try:
        alarms = cases._bind(case_dir, con).alarms()
    except Exception:  # noqa: BLE001
        alarms = []
    finally:
        con.close()
    if cases.kind(case_dir) == "rca100":
        title = next((a.get("title") for a in alarms if a.get("title")), "") or "Alert"
        lines = [f"[Alert] {title} on service {alarm} during the alert window {f(t0)} ~ {f(t1)}."]
        what = "alerting operation"
    else:
        lines = [f"[Alert] SLO violation on service {alarm} during the incident window {f(t0)} ~ {f(t1)}."]
        what = "violated operation"
    for a in alarms[:5]:
        op = a.get("operation") or ""
        if op:
            lines.append(f"[Alert] {what}: {op} (component {a.get('service') or alarm}, "
                         f"severity {float(a.get('severity', 0) or 0):.2f})")
    lines.append("Services in the system: " + ", ".join(cases.instances_of(case_dir)) + ".")
    lines.append("Telemetry snapshots (traces/logs/metrics of every service, reference vs incident window) "
                 "are available to the expert agents via tools.")
    return "\n".join(lines)


SERVICE_STATES = {"rca100": ["HEALTHY", "HIGH_ERROR_RATE", "HIGH_LATENCY", "UNAVAILABLE", "UNKNOWN"],
                  "rcabench": ["HEALTHY", "HIGH_ERROR_RATE", "HIGH_LATENCY", "UNAVAILABLE"]}   # rca100.yaml / rca.yaml


def report_graph_of(report: dict | None, instances: list[str]) -> dict | None:
    """The graph of the Commander's final type-1 answer as a CausalGraph, or None when unusable.

    Components are matched to the case's services (case-insensitive); edges and nodes naming anything
    else, self-loops and duplicates are dropped; roots and edge endpoints missing from "nodes" are added.
    None when the answer has no "graph" or no root cause names a service of the case.
    """
    g = report.get("graph") if isinstance(report, dict) and report.get("type") == 1 else None
    if not isinstance(g, dict):
        return None
    by_lower = {s.lower(): s for s in instances}

    def svc(x):
        return by_lower.get(str(x.get("component") if isinstance(x, dict) else x or "").strip().lower())

    def states(x):
        st = x.get("state") if isinstance(x, dict) else None
        st = [st] if isinstance(st, str) else st
        return [str(v) for v in st] if isinstance(st, list) else []

    roots, root_state = [], {}
    for r in g.get("root_causes") or []:
        c = svc(r)
        if c and c not in roots:
            roots.append(c)
            root_state[c] = states(r)
    if not roots:
        return None
    edges = []
    for e in g.get("edges") or []:
        if isinstance(e, dict):
            a, b = svc(e.get("source")), svc(e.get("target"))
            if a and b and a != b and {"source": a, "target": b} not in edges:
                edges.append({"source": a, "target": b})
    node_state = {}
    for n in g.get("nodes") or []:
        c = svc(n)
        if c and c not in node_state:
            node_state[c] = states(n)
    order = list(dict.fromkeys(roots + list(node_state) + [c for e in edges for c in (e["source"], e["target"])]))
    return {"nodes": [{"component": c, "state": node_state.get(c) or root_state.get(c, [])} for c in order],
            "edges": edges,
            "root_causes": [{"component": c, "state": root_state[c] or node_state.get(c, [])} for c in roots],
            "component_to_service": {}}


def extract_answer(answer: str, report: str, graph: BeliefGraph, instances: list[str], alarm: str) -> dict:
    hypos = [f"{n['label']} (confidence {n.get('score', 0):.2f})" for n in graph.nodes.values()
             if n.get("type") == "Hypothesis"]
    prompt = f"""A root-cause investigation of a microservice incident concluded:
Final answer: {answer}
Report: {report}
Hypotheses considered (label, confidence): {hypos}
Alarmed service: {alarm}
Services in the system: {instances}

Map the conclusion to the benchmark's output format. Output ONLY JSON:
{{"roots": ["<service>", ...], "path": [["<cause service>", "<effect service>"], ...]}}
- roots: the root-cause service(s) named in the conclusion, most likely first, service names copied exactly from the list; at least one.
- path: the fault propagation path from each root to the alarmed service as [cause, effect] hops, only hops the report supports; may be empty."""
    text = _llm(prompt, temperature=0.0, max_tokens=800, stage="extract_answer")
    try:
        d = pf.parse_json_response(text)
    except Exception:
        d = {}
    if not isinstance(d, dict):
        d = {}
    inst = set(instances)
    roots = [r for r in (d.get("roots") or []) if isinstance(r, str) and r in inst]
    path = [[p[0], p[1]] for p in (d.get("path") or [])
            if isinstance(p, (list, tuple)) and len(p) == 2 and p[0] in inst and p[1] in inst and p[0] != p[1]]
    return {"roots": roots, "path": path, "raw": text[:500]}


# GoS-logic failures caused by LLM output the shims above do not cover end the case
# without an answer. Transport and
# other exceptions propagate (crash policy).
AGENT_FAILURES = (KeyError, ValueError, TypeError, IndexError, json.JSONDecodeError)


def load_configs() -> tuple[dict, dict]:
    cfg_agent = yaml.safe_load(open(f"{GOS}/configs/Agent_Configs.yaml"))
    cfg_belief = yaml.safe_load(open(f"{GOS}/configs/Belief_Configs.yaml"))
    return cfg_agent, cfg_belief


def run_case(case_dir: Path, question: str) -> dict:
    """Run GoS on one case."""
    t0 = time.time()
    CTX.reset()
    cfg_agent, cfg_belief = load_configs()
    case_dir = Path(case_dir)
    name = case_dir.parent.name if case_dir.name == "converted" else case_dir.name
    alarm = cases.alarm_of(case_dir)
    instances = cases.instances_of(case_dir)
    symptom = symptom_text(case_dir, alarm)
    if GRAPH_SOURCE == "report":
        report_graph.configure(instances, alarm, SERVICE_STATES[cases.kind(case_dir)])
    resources = {"case_dir": case_dir, "alarm": alarm, "events": EV}
    log_dir = os.environ.get("RCA_EVENTS_DIR") or tempfile.mkdtemp(prefix="gos_")
    log_path = str(Path(log_dir) / "gos_case_log.txt")
    with EV.scope(role="gos:runtime"):
        EV.artifact("gos_setup", {"case": name, "alarm": alarm, "instances": instances, "symptom": symptom,
                                  "domain": DOMAIN_CFG, "llm_args": cfg_agent["llm_args"], "belief": cfg_belief})

    G, F = BeliefGraph(), BeliefFSM(thresholds=cfg_belief.get("fsm", {}).get("thresholds", {}))
    CTX.graph, CTX.fsm = G, F
    llm_args = dict(cfg_agent["llm_args"]); llm_args["model_path"] = "deepseek"
    experts = [Expert_Agent(n, graph=G, fsm=F, case_id=name, dataset="Micro", log_path=log_path,
                            cfg_domain=DOMAIN_CFG, llm_args=llm_args, evidence_text=resources,
                            max_retrieval_steps=llm_args["max_retrieval_steps"])
               for n in DOMAIN_CFG["expert_names"]]
    central = Central_Agent(HEAD, graph=G, fsm=F, dataset="Micro", log_path=log_path,
                            case_id=name, case_symptom=symptom, llm_args=llm_args, cfg_domain=DOMAIN_CFG,
                            experts=experts, evidence_text=resources)
    answer = report = None
    error = None
    try:
        CTX.parent = EV.new_id("gos_ingest_")
        central.ingest()
        emit_state("after_ingest")
        answer, report = central.run()
    except AGENT_FAILURES as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[gos] case ended without an answer: {error}\n{traceback.format_exc()}", file=sys.stderr)
    CTX.parent = EV.new_id("gos_final_")
    emit_state("final")
    # GOS_GRAPH_SOURCE=report: the graph of the final type-1 answer, extract_answer only when it is unusable;
    # GOS_GRAPH_SOURCE=extract: extract_answer.
    final_graph = (report_graph_of(CTX.report_dict, instances)
                   if answer is not None and GRAPH_SOURCE == "report" else None)
    raw_graph = CTX.report_dict.get("graph") if isinstance(CTX.report_dict, dict) else None
    ext = {"roots": [], "path": [], "raw": ""}
    graph_from = "report" if final_graph else "none"
    if final_graph is None and (answer is not None or report is not None):
        ext = extract_answer(answer, report, G, instances, alarm)
        graph_from = "extract_answer"
    with EV.scope(role="gos:runtime", parent=CTX.parent):
        EV.artifact("final_answer", {"answer": answer, "report": report, "graph": final_graph,
                                     "graph_from": graph_from, "report_graph_raw": raw_graph,
                                     "extract": ext, "error": error})
    return {"case": name, "alarm": alarm, "instances": instances, "answer": answer, "report": report,
            "graph": final_graph, "graph_from": graph_from,
            "report_graph_raw": raw_graph,
            "roots": ext["roots"], "path": ext["path"], "extract_raw": ext["raw"], "error": error,
            "hypotheses": [(n["label"], n.get("score")) for n in G.nodes.values() if n.get("type") == "Hypothesis"],
            "n_nodes": len(G.nodes), "n_edges": len(G.edges), "llm_calls": CTX.llm_calls,
            "stage_calls": dict(CTX.stage_calls),
            "tool_calls": sum(getattr(e.telemetry, "calls", 0) for e in experts if e.telemetry),
            "elapsed_s": round(time.time() - t0, 1)}
