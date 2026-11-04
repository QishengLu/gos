"""Telemetry tools for GoS expert agents over an RCABench case.

GoS's IT experts get three offline snapshot tools (shell/log/metric). The
microservice analogue: trace (call graph, replaces shell), log, metric. Same
JSON call protocol so GoS's expert loop (_run_micro) runs unchanged.

Data comes from ``access_rcabench.Access`` and ``_rca_shared.cases.instances_of`` (RCABench or
RCA100); the tool prompt lists no metric names; every tool call is logged to the
harness event log (raw request, parsed args, full result).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
from access_rcabench import Access                     # noqa: E402
from _rca_shared.cases import instances_of             # noqa: E402

_ACCESS: dict[str, Access] = {}


def _access(case_dir: Path) -> Access:
    """One Access (DuckDB connection + materialised tables) per case, shared by the experts."""
    key = str(Path(case_dir).resolve())
    if key not in _ACCESS:
        _ACCESS[key] = Access(case_dir)
    return _ACCESS[key]


class OpsTelemetry:
    def __init__(self, case_dir: Path, alarm: str, events=None):
        self.case_dir = Path(case_dir)
        self.acc = _access(case_dir)
        self.instances = instances_of(case_dir)
        self.alarm = alarm
        self.obs = self.acc.list_data()
        self.events = events          # harness EventLog (None -> no logging)
        self.calls = 0

    def build_tool_prompt(self) -> str:
        ex = json.dumps(["<metric>"])
        return f"""
You can access three offline telemetry tools (priority: trace > log > metric). They work on provided snapshots of one incident (reference window = before the fault, incident window = during the fault).
Alarmed service: {self.alarm}. Services in the system: {', '.join(self.instances)}.

Tool call format (always JSON string):
- Trace tool: {{"tool": "trace", "service": "<service>"}}  -> the service's outgoing calls (callee, calls, p95, error rate, reference vs incident, VANISHED flags), incoming callers, and its own error spans.
- Log tool: {{"tool": "log", "service": "<service>", "keywords": ["error","timeout"], "limit": 15}}  -> WARN/ERROR lines of that service in the incident window (keywords narrow).
- Metric tool: {{"tool": "metric", "service": "<service>", "metric_names": {ex}}}  -> reference vs incident statistics and flags (DEVIATED/CHANGED/GONE/SERIES_STOPPED) for up to 3 metrics; omit metric_names to get the flagged metrics of the service.

Constraints:
- Every call must name a service from the list above.
- Only JSON format is accepted.
""".strip()

    def dispatch_tool(self, raw, enable_fuzzy_shell: bool = True) -> str:
        self.calls += 1
        req, tool, svc, result, err = {}, "", "", None, None
        try:
            req, tool, svc, result = self._dispatch(raw)
            return result
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self.events is not None:
                self.events.tool_call(f"gos.{tool or 'unknown'}",
                                      {"raw": raw if isinstance(raw, (str, dict)) else str(raw),
                                       "parsed": req, "service_resolved": svc},
                                      result=result, error=err)

    def _dispatch(self, raw):
        req = raw if isinstance(raw, dict) else {}
        if not req:
            try:
                req = json.loads(str(raw))
            except Exception:
                m = re.search(r"\{.*\}", str(raw), re.DOTALL)
                try:
                    req = json.loads(m.group(0)) if m else {}
                except Exception:
                    req = {}
        tool = str(req.get("tool") or "").lower()
        svc = self._svc(req.get("service") or req.get("instance") or req.get("command") or "")
        if tool == "trace" or tool == "shell":
            return req, tool, svc, self.trace_tool(svc)
        if tool == "log":
            return req, tool, svc, self.log_tool(svc, req.get("keywords") or [], req.get("limit") or 15)
        if tool == "metric":
            return req, tool, svc, self.metric_tool(svc, req.get("metric_names") or req.get("metrics") or [])
        return req, tool, svc, (f"Unsupported tool request: {str(raw)[:200]}. "
                                "Use trace/log/metric with a service name.")

    def _svc(self, name) -> str:
        name = str(name or "").strip()
        if name in self.instances:
            return name
        t = re.sub(r"[^a-z0-9]", "", name.lower())
        for inst in self.instances:
            if t and t == re.sub(r"[^a-z0-9]", "", inst.lower()):
                return inst
        for inst in self.instances:
            if t and t in re.sub(r"[^a-z0-9]", "", inst.lower()):
                return inst
        return name or self.alarm

    @staticmethod
    def _win(d):
        return "-" if not d else f"calls={d['calls']} p95={d['p95_ms']}ms err={d['err_rate']}"

    def trace_tool(self, svc: str) -> str:
        if svc not in self.obs and svc not in self.instances:
            return f"[trace] unknown service {svc!r}"
        out = [f"[trace] outgoing calls of {svc} (reference -> incident):"]
        for r in self.acc.call_stats(caller=svc)[:25]:
            flag = f" {r['flag']}" if r.get("flag") else ""
            out.append(f"  -> {r['callee']}: ref[{self._win(r.get('reference'))}] inc[{self._win(r.get('incident'))}]{flag}")
        out.append(f"[trace] incoming callers of {svc}:")
        for r in self.acc.call_stats(callee=svc)[:15]:
            flag = f" {r['flag']}" if r.get("flag") else ""
            out.append(f"  <- {r['caller']}: ref[{self._win(r.get('reference'))}] inc[{self._win(r.get('incident'))}]{flag}")
        errs = self.acc.span_samples(service=svc, errors_only=True, limit=8)
        if errs:
            out.append(f"[trace] error spans of {svc} in the incident window (worst first):")
            for s in errs:
                out.append(f"  {s['operation']} dur={s['dur_ms']}ms children={s.get('n_children', '?')}")
        else:
            out.append(f"[trace] {svc} has no error spans in the incident window.")
        on = self.acc.onsets([svc]).get(svc, {})
        if on:
            out.append(f"[trace] onsets of {svc}: {json.dumps(on)}")
        return "\n".join(out)

    def log_tool(self, svc: str, keywords, limit) -> str:
        try:
            limit = max(1, min(int(limit), 40))
        except Exception:
            limit = 15
        kws = [k for k in (keywords if isinstance(keywords, list) else str(keywords).split(",")) if str(k).strip()]
        rows = self.acc.log_samples(service=svc, limit=200)
        if kws:
            rows = [r for r in rows if any(str(k).lower() in r["message"].lower() for k in kws)] or rows
        st = self.acc.log_stats(svc)
        head = (f"[log] {svc}: WARN+ lines reference={st['warnplus_reference']} incident={st['warnplus_incident']}; "
                f"top templates: " + "; ".join(t['template'][:80] for t in st['top_templates'][:4]))
        if not rows:
            return head + "\n[log] no matching lines."
        return head + "\n" + "\n".join(f"  {r['time']} {r['level']} {r['message'][:200]}" for r in rows[-limit:])

    def metric_tool(self, svc: str, names) -> str:
        names = [n for n in (names if isinstance(names, list) else str(names).split(",")) if str(n).strip()][:3]
        rows = self.acc.metric_stats(svc, metrics=names or None)
        if not names:
            rows = [r for r in rows if r.get("flag")][:12]
        if not rows:
            return f"[metric] {svc}: no data for {names or 'flagged metrics'}."
        out = [f"[metric] {svc} (reference vs incident):"]
        for r in rows:
            out.append(f"  {r['metric']}: ref_mean={r.get('reference_mean')} inc_mean={r.get('incident_mean')} "
                       f"inc_max={r.get('incident_max')} z={r.get('z')} flag={r.get('flag')} onset={r.get('onset', '')}")
        return "\n".join(out)
