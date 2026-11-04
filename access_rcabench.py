"""RCABench / RCA100 re-implementation of the ``Access`` data layer used by ``telemetry.py``.

GoS's tools read telemetry through an ``Access`` object. This module provides its methods and return
shapes, computed with
DuckDB over one case directory through the canonical views of
``agents/_rca_shared/cases.py`` (``rcabench.py`` for an RCABench ``converted/``
directory, ``rca100.py`` for an RCA100 sandbox; the rules below apply to both).

Windows
    reference = ``normal_*`` tables, incident = ``abnormal_*`` tables.
    Bounds come from ``env.json`` (NORMAL_START/END, ABNORMAL_START/END, epoch s);
    RCA100: the question's alert window (``rca100._read_env``).
    ``loadgenerator`` / ``load-generator`` and empty service names are excluded everywhere.

Traces
    Call edge caller -> callee: a span whose parent span (same trace_id,
    parent_id = span_id) belongs to a different service. The statistics of an
    edge are those of the callee-side child spans: ``calls`` (count),
    ``p95_ms`` (95th percentile of duration), ``err_rate`` (share of spans with
    status_code = Error).
    Edge flags (comma-joined when several apply):
      VANISHED    reference calls > 0, incident calls = 0
      NEW         reference calls = 0, incident calls > 0
      LATENCY_UP  incident p95 >= 2 x reference p95 and +50 ms or more
      ERRORS_UP   incident err_rate - reference err_rate >= 0.10

Metrics (``metrics`` and ``metrics_sum`` tables; both carry per-interval values)
    Points of every series of (service, metric) are pooled per window.
    reference_mean / reference_std (population) / incident_mean / incident_max,
    z = (incident_mean - reference_mean) / reference_std (None when std = 0).
    Scrape interval = (last - first) / (distinct timestamps - 1) of the reference.
    Metric flags (first match wins):
      GONE            points in reference, none in incident
      SERIES_STOPPED  incident points exist but some gap (incident start, the
                      incident timestamps, incident end) exceeds
                      max(3 x scrape interval, 15 s)
      DEVIATED        |z| >= 3
      CHANGED         not DEVIATED and (relative mean shift >= 50 %, or the
                      reference is constant and the incident mean differs, or
                      the metric exists only in the incident window)
      ''              otherwise
    onset: GONE -> last reference point; SERIES_STOPPED -> start of the largest
    gap; DEVIATED/CHANGED -> first incident point with
    |value - reference_mean| > max(3 x reference_std, 0.5 x |reference_mean|).

Logs
    WARN+ line: level in {WARN, WARNING, ERROR, SEVERE, FATAL, CRITICAL}, or an
    empty level (e.g. mysql) with a message matching warn/warning/error/severe/
    fatal/exception as a word.
    Templates: uuid and hex tokens (>= 6 hex chars with a digit) -> <ID>,
    remaining digit runs -> <N>, whitespace squashed.

Timestamps are rendered as UTC ``YYYY-MM-DD HH:MM:SS`` strings.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

_AGENTS = Path(__file__).resolve().parents[1]
if str(_AGENTS) not in sys.path:
    sys.path.insert(0, str(_AGENTS))
from _rca_shared import cases  # noqa: E402

EXCLUDED = tuple(sorted(cases.EXCLUDED_SERVICES))
WARN_LEVELS = ("WARN", "WARNING", "ERROR", "SEVERE", "FATAL", "CRITICAL")
_WARN_MSG_RE = r"(?i)\b(warn|warning|error|severe|fatal|exception)\b"

# thresholds (documented above)
Z_DEVIATED = 3.0
REL_CHANGED = 0.5
EDGE_LAT_RATIO = 2.0
EDGE_LAT_ABS_MS = 50.0
EDGE_ERR_DELTA = 0.10
GAP_INTERVALS = 3.0
GAP_MIN_S = 15.0


def fmt_ts(t: float | None) -> str | None:
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _r(x: Any, nd: int = 4) -> Any:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def template_of(message: str) -> str:
    s = re.sub(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}\b", "<ID>", str(message or ""))
    s = re.sub(r"(?<![0-9A-Za-z])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}(?![0-9A-Za-z])", "<ID>", s)
    s = re.sub(r"\d+", "<N>", s)
    return re.sub(r"\s+", " ", s).strip()


class Access:
    def __init__(self, case_dir: str | Path):
        self.case_dir = Path(case_dir)
        self.con = duckdb.connect()
        self.bind = cases._bind(self.case_dir, self.con)
        env = cases._read_env(self.case_dir)
        self.inc_start, self.inc_end = cases.incident_window(self.case_dir)
        self.ref_start = int(env["NORMAL_START"]) if env.get("NORMAL_START") else None
        self.ref_end = int(env["NORMAL_END"]) if env.get("NORMAL_END") else None
        self._materialize()
        self._metric_cache: dict[str, list[dict]] = {}

    # ── setup ────────────────────────────────────────────────────────────
    def _materialize(self) -> None:
        excl = ", ".join(f"'{s}'" for s in EXCLUDED)
        for win in ("reference", "incident"):
            tv = self.bind.view("traces", win)
            if tv:
                self.con.execute(f"""CREATE TEMP TABLE spans_{win} AS
                    SELECT epoch(ts) AS t, trace_id, span_id, parent_id, service, span_name,
                           duration_ns / 1e6 AS dur_ms, is_error, http_status
                    FROM {tv} WHERE coalesce(service, '') NOT IN ({excl})""")
            else:
                self.con.execute(f"""CREATE TEMP TABLE spans_{win} (t DOUBLE, trace_id VARCHAR,
                    span_id VARCHAR, parent_id VARCHAR, service VARCHAR, span_name VARCHAR,
                    dur_ms DOUBLE, is_error BOOLEAN, http_status INTEGER)""")
            self.con.execute(f"""CREATE TEMP TABLE edges_{win} AS
                SELECT p.service AS caller, c.service AS callee, c.dur_ms, c.is_error, c.t
                FROM spans_{win} c JOIN spans_{win} p
                  ON c.trace_id = p.trace_id AND c.parent_id = p.span_id
                WHERE c.service <> p.service""")
            mv = self.bind.view("metrics", win)
            if mv:
                self.con.execute(f"""CREATE TEMP TABLE metrics_{win} AS
                    SELECT epoch(ts) AS t, metric, value, service FROM {mv}
                    WHERE service IS NOT NULL AND service NOT IN ({excl}) AND value IS NOT NULL
                      AND NOT isnan(value)""")
            else:
                self.con.execute(f"CREATE TEMP TABLE metrics_{win} (t DOUBLE, metric VARCHAR, value DOUBLE, service VARCHAR)")
            lv = self.bind.view("logs", win)
            if lv:
                self.con.execute(f"""CREATE TEMP TABLE logs_{win} AS
                    SELECT epoch(ts) AS t, service, upper(coalesce(level, '')) AS level, message,
                           (upper(coalesce(level, '')) IN ({', '.join(f"'{x}'" for x in WARN_LEVELS)})
                            OR (coalesce(level, '') = '' AND regexp_matches(coalesce(message, ''), '{_WARN_MSG_RE}')))
                           AS warnplus
                    FROM {lv} WHERE coalesce(service, '') NOT IN ({excl})""")
            else:
                self.con.execute(f"CREATE TEMP TABLE logs_{win} (t DOUBLE, service VARCHAR, level VARCHAR, message VARCHAR, warnplus BOOLEAN)")

    def _q(self, sql: str, params: list | None = None) -> list[dict]:
        cur = self.con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── inventory ────────────────────────────────────────────────────────
    def list_data(self) -> dict[str, dict]:
        """{service: {plane: {"reference": rows, "incident": rows}}} for every observed service."""
        out: dict[str, dict] = {}
        for plane, table in (("traces", "spans"), ("logs", "logs"), ("metrics", "metrics")):
            for win in ("reference", "incident"):
                for r in self._q(f"SELECT service, count(*) AS n FROM {table}_{win} "
                                 f"WHERE service IS NOT NULL AND service <> '' GROUP BY 1"):
                    out.setdefault(r["service"], {}).setdefault(plane, {"reference": 0, "incident": 0})[win] = int(r["n"])
        return dict(sorted(out.items()))

    # ── metrics ──────────────────────────────────────────────────────────
    def _all_metric_stats(self, svc: str) -> list[dict]:
        if svc in self._metric_cache:
            return self._metric_cache[svc]
        agg = {}
        for win in ("reference", "incident"):
            for r in self._q(f"""SELECT metric, count(*) AS n, avg(value) AS mean, stddev_pop(value) AS std,
                                        max(value) AS mx, min(t) AS t0, max(t) AS t1, count(DISTINCT t) AS nts
                                 FROM metrics_{win} WHERE service = ? GROUP BY 1""", [svc]):
                agg.setdefault(r["metric"], {})[win] = r
        rows = []
        for metric, d in agg.items():
            ref, inc = d.get("reference"), d.get("incident")
            row = {"metric": metric,
                   "reference_mean": _r(ref["mean"]) if ref else None,
                   "reference_std": _r(ref["std"]) if ref else None,
                   "incident_mean": _r(inc["mean"]) if inc else None,
                   "incident_max": _r(inc["mx"]) if inc else None,
                   "z": None, "flag": "", "onset": None}
            if ref and not inc:
                row["flag"], row["onset"] = "GONE", fmt_ts(ref["t1"])
                rows.append(row)
                continue
            if inc and not ref:
                row["flag"], row["onset"] = "CHANGED", fmt_ts(inc["t0"])
                rows.append(row)
                continue
            rm, rs, im = float(ref["mean"]), float(ref["std"] or 0.0), float(inc["mean"])
            z = (im - rm) / rs if rs > 1e-12 else None
            row["z"] = _r(z, 2)
            interval = (ref["t1"] - ref["t0"]) / (ref["nts"] - 1) if ref["nts"] and ref["nts"] > 1 else 60.0
            gap_th = max(GAP_INTERVALS * interval, GAP_MIN_S)
            gap, gap_at = self._largest_gap(svc, metric)
            rel = abs(im - rm) / max(abs(rm), 1e-9)
            if gap > gap_th:
                row["flag"], row["onset"] = "SERIES_STOPPED", fmt_ts(gap_at)
            elif z is not None and abs(z) >= Z_DEVIATED:
                row["flag"] = "DEVIATED"
            elif abs(im - rm) > 1e-9 and (rel >= REL_CHANGED or rs <= 1e-12):
                row["flag"] = "CHANGED"
            if row["flag"] in ("DEVIATED", "CHANGED"):
                thr = max(3 * rs, 0.5 * abs(rm))
                first = self.con.execute(
                    "SELECT min(t) FROM metrics_incident WHERE service = ? AND metric = ? AND abs(value - ?) > ?",
                    [svc, metric, rm, thr if thr > 0 else 1e-12]).fetchone()[0]
                row["onset"] = fmt_ts(first)
            rows.append(row)
        prio = {"GONE": 0, "SERIES_STOPPED": 1, "DEVIATED": 2, "CHANGED": 3, "": 4}
        rows.sort(key=lambda r: (prio.get(r["flag"], 5), -abs(r["z"] or 0.0), r["metric"]))
        self._metric_cache[svc] = rows
        return rows

    def _largest_gap(self, svc: str, metric: str) -> tuple[float, float | None]:
        ts = [r[0] for r in self.con.execute(
            "SELECT DISTINCT t FROM metrics_incident WHERE service = ? AND metric = ? ORDER BY 1",
            [svc, metric]).fetchall()]
        pts = [float(self.inc_start)] + ts + [float(self.inc_end)]
        best, at = 0.0, None
        for a, b in zip(pts, pts[1:]):
            if b - a > best:
                best, at = b - a, a
        return best, at

    def metric_stats(self, svc: str, metrics: list[str] | None = None) -> list[dict]:
        """Reference-vs-incident statistics per metric of ``svc`` (flagged first).

        ``metrics`` selects metrics by exact name; a name without an exact match
        selects every metric containing it (case-insensitive).
        """
        rows = self._all_metric_stats(svc)
        if not metrics:
            return rows
        names = {r["metric"] for r in rows}
        want: list[str] = []
        for m in metrics:
            m = str(m).strip()
            if m in names:
                want.append(m)
            else:
                want += [n for n in sorted(names) if m.lower() in n.lower()]
        return [r for r in rows if r["metric"] in set(want)]

    # ── traces ───────────────────────────────────────────────────────────
    def _edge_win(self, win: str, where: str, key: str, svc: str) -> dict[str, dict]:
        out = {}
        for r in self._q(f"""SELECT {key} AS k, count(*) AS calls,
                                    quantile_cont(dur_ms, 0.95) AS p95, avg(is_error::INT) AS err
                             FROM edges_{win} WHERE {where} = ? GROUP BY 1""", [svc]):
            out[r["k"]] = {"calls": int(r["calls"]), "p95_ms": _r(r["p95"], 1), "err_rate": _r(r["err"], 3)}
        return out

    def call_stats(self, caller: str | None = None, callee: str | None = None) -> list[dict]:
        """Edges out of ``caller`` (rows keyed ``callee``) or into ``callee`` (rows keyed ``caller``)."""
        if caller is not None:
            where, key, other = "caller", "callee", "callee"
            svc = caller
        elif callee is not None:
            where, key, other = "callee", "caller", "caller"
            svc = callee
        else:
            raise ValueError("call_stats needs caller= or callee=")
        ref = self._edge_win("reference", where, key, svc)
        inc = self._edge_win("incident", where, key, svc)
        rows = []
        for k in sorted(set(ref) | set(inc)):
            a, b = ref.get(k), inc.get(k)
            flags = []
            if a and not b:
                flags.append("VANISHED")
            elif b and not a:
                flags.append("NEW")
            else:
                if (b["p95_ms"] or 0) >= EDGE_LAT_RATIO * (a["p95_ms"] or 0) and \
                        (b["p95_ms"] or 0) - (a["p95_ms"] or 0) >= EDGE_LAT_ABS_MS:
                    flags.append("LATENCY_UP")
                if (b["err_rate"] or 0) - (a["err_rate"] or 0) >= EDGE_ERR_DELTA:
                    flags.append("ERRORS_UP")
            rows.append({other: k, "reference": a, "incident": b, "flag": ",".join(flags)})
        rows.sort(key=lambda r: (r["flag"] == "", -((r["incident"] or r["reference"])["calls"])))
        return rows

    def span_samples(self, service: str, errors_only: bool = False, limit: int = 8) -> list[dict]:
        """Incident-window spans of ``service``, longest first, with their direct child count."""
        cond = "AND s.is_error" if errors_only else ""
        rows = self._q(f"""SELECT s.span_name AS operation, s.dur_ms, s.t, s.http_status,
                                  (SELECT count(*) FROM spans_incident c
                                   WHERE c.trace_id = s.trace_id AND c.parent_id = s.span_id) AS n_children
                           FROM spans_incident s WHERE s.service = ? {cond}
                           ORDER BY s.dur_ms DESC LIMIT ?""", [service, int(limit)])
        return [{"operation": r["operation"], "dur_ms": _r(r["dur_ms"], 1), "n_children": int(r["n_children"]),
                 "time": fmt_ts(r["t"]), "http_status": r["http_status"]} for r in rows]

    def onsets(self, services: list[str]) -> dict[str, dict]:
        """First incident-window anomaly time per signal and service (signals without one are omitted).

        first_error_span   earliest error span
        first_slow_span    earliest span slower than the reference p99 of the same
                           operation (operations with >= 5 reference spans)
        first_warnplus_log earliest WARN+ line, only if the incident has more WARN+
                           lines than the reference window
        first_metric_anomaly earliest onset among flagged metrics (with the metric name)
        """
        out: dict[str, dict] = {}
        for svc in services:
            d: dict[str, Any] = {}
            t = self.con.execute("SELECT min(t) FROM spans_incident WHERE service = ? AND is_error",
                                 [svc]).fetchone()[0]
            if t is not None:
                d["first_error_span"] = fmt_ts(t)
            t = self.con.execute("""
                WITH ref AS (SELECT span_name, quantile_cont(dur_ms, 0.99) AS p99 FROM spans_reference
                             WHERE service = ? GROUP BY 1 HAVING count(*) >= 5)
                SELECT min(s.t) FROM spans_incident s JOIN ref USING (span_name)
                WHERE s.service = ? AND s.dur_ms > ref.p99""", [svc, svc]).fetchone()[0]
            if t is not None:
                d["first_slow_span"] = fmt_ts(t)
            st = self.log_stats(svc, templates=False)
            if st["warnplus_incident"] > st["warnplus_reference"]:
                t = self.con.execute("SELECT min(t) FROM logs_incident WHERE service = ? AND warnplus",
                                     [svc]).fetchone()[0]
                if t is not None:
                    d["first_warnplus_log"] = fmt_ts(t)
            flagged = [r for r in self._all_metric_stats(svc) if r["flag"] and r["onset"]]
            if flagged:
                first = min(flagged, key=lambda r: r["onset"])
                d["first_metric_anomaly"] = f"{first['onset']} ({first['metric']} {first['flag']})"
            if d:
                out[svc] = d
        return out

    # ── logs ─────────────────────────────────────────────────────────────
    def log_samples(self, service: str, limit: int = 200) -> list[dict]:
        """Incident-window WARN+ lines of ``service`` in time order (all levels if it has none)."""
        rows = self._q("""SELECT t, level, message FROM logs_incident WHERE service = ? AND warnplus
                          ORDER BY t LIMIT ?""", [service, int(limit)])
        if not rows:
            rows = self._q("SELECT t, level, message FROM logs_incident WHERE service = ? ORDER BY t LIMIT ?",
                           [service, int(limit)])
        return [{"time": fmt_ts(r["t"]), "level": r["level"] or "-", "message": str(r["message"] or "")}
                for r in rows]

    def log_stats(self, svc: str, templates: bool = True) -> dict:
        ref = self.con.execute("SELECT count(*) FROM logs_reference WHERE service = ? AND warnplus", [svc]).fetchone()[0]
        inc = self.con.execute("SELECT count(*) FROM logs_incident WHERE service = ? AND warnplus", [svc]).fetchone()[0]
        out = {"warnplus_reference": int(ref), "warnplus_incident": int(inc), "top_templates": []}
        if templates and inc:
            msgs = [r[0] for r in self.con.execute(
                "SELECT message FROM logs_incident WHERE service = ? AND warnplus", [svc]).fetchall()]
            cnt = Counter(template_of(m) for m in msgs)
            out["top_templates"] = [{"template": t, "count": n} for t, n in cnt.most_common(10)]
        return out

    def close(self) -> None:
        self.con.close()
