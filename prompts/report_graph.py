"""The SRE Commander's report-or-refine prompt with the fault propagation graph (port-owned).

``run.py`` installs ``construct_report_or_refine_prompt`` below in place of upstream's
(``prompts/prompt_generation.py``, kept in the file and no longer called). The report-type
answer (type 1) is where GoS concludes, so the graph is written there, from the belief, the
belief graph and the frontier the Commander already reads. Differences from upstream's
task prompt:

1. Type 1 carries a third field, "graph" (root causes, cause -> effect edges, node states),
   specified in constraint 4; the Goal, Constraints and Execution Instructions name it where
   they name "answer" and "report", and constraint 5 excludes it from type 2.
2. The illustrations are microservice incidents instead of medical ones: the refinement
   examples in the Goal and Execution Instructions, and the three worked examples (same
   three situations: mandatory report, insufficient frontier, sufficient frontier). The
   example service names occur in neither RCABench nor RCA100.
3. The user prompt ends with the services of the case and the alarmed service, the names
   the graph must use (upstream's user prompt is otherwise unchanged).

Everything else (roles, the type 1 / type 2 decision rules, the answer and report limits,
the candidate specification) is upstream's text.
"""
from __future__ import annotations

import importlib.util
import json
import os

_CASE: dict = {"services": [], "alarm": "", "states": []}


def configure(services: list[str], alarm: str, states: list[str]) -> None:
    """The case's service list, alarmed service and allowed node states (set by run.run_case)."""
    _CASE.update(services=list(services), alarm=alarm, states=list(states))


_EXAMPLE_1 = {
    "type": 1,
    "answer": "billing service failure",
    "report": ("shop-web errors began at 10:02, one minute after orders started receiving 503 responses "
               "from billing. billing logs 'connection pool exhausted' from 10:01, while ledger-db, the "
               "database billing uses, keeps its normal latency and error rate. billing therefore failed "
               "first and on its own; orders failed because its billing calls failed, and shop-web because "
               "its orders calls failed."),
    "graph": {
        "root_causes": [{"component": "billing", "state": ["HIGH_ERROR_RATE"]}],
        "edges": [{"source": "billing", "target": "orders"}, {"source": "orders", "target": "shop-web"}],
        "nodes": [{"component": "billing", "state": ["HIGH_ERROR_RATE"]},
                  {"component": "orders", "state": ["HIGH_ERROR_RATE"]},
                  {"component": "shop-web", "state": ["HIGH_ERROR_RATE"]}],
    },
}
_EXAMPLE_2 = {
    "type": 2,
    "candidates": [
        {"id": "h002", "label": "billing service failure", "confidence": 0.5,
         "why": "Current frontier 'downstream dependency failure of orders' is too general; billing is one of the "
                "two services orders calls and handles every order, so its failure would slow all orders requests"},
        {"id": "h003", "label": "stock service failure", "confidence": 0.35,
         "why": "stock is the other dependency orders calls; no per-dependency data yet rules it out"},
        {"id": "h004", "label": "network delay between orders and its dependencies", "confidence": 0.15,
         "why": "A slow network path would inflate every orders call at once; less likely because only latency, "
                "not errors, would be expected"},
    ],
}
_EXAMPLE_3 = {
    "type": 1,
    "answer": "orders service internal failure",
    "report": ("From 10:05 orders returns errors on 40% of requests and logs a NullPointerException in createOrder "
               "at the same moment; neither existed in the reference window. billing and stock, which orders calls, "
               "keep normal latency, error rate and request volume. The fault therefore starts inside orders, and "
               "shop-web errors follow from its failing orders calls."),
    "graph": {
        "root_causes": [{"component": "orders", "state": ["HIGH_ERROR_RATE"]}],
        "edges": [{"source": "orders", "target": "shop-web"}],
        "nodes": [{"component": "orders", "state": ["HIGH_ERROR_RATE"]},
                  {"component": "shop-web", "state": ["HIGH_ERROR_RATE"]}],
    },
}


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2).replace("\n", "\n  ")


def construct_report_or_refine_prompt(dataset, expert_name, belief, graph_description, Report_Flag, frontier):

    system_prompt_path = os.path.join("prompts", dataset, "system_prompt.py")
    spec = importlib.util.spec_from_file_location(f"{expert_name}_system_prompt", system_prompt_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    system_prompt = getattr(prompt_module, f"{expert_name}_system_prompt")

    states = ", ".join(_CASE["states"])
    task_prompt = f"""
- Goal:
  The central agent first determines the action based on the Report_Flag and the refinement level of the current frontier:
  1. If Report_Flag is True (mandatory report generation): Generate a final answer, a complete reasoning report and the fault propagation graph based on the current belief, existing graph, and frontier;
  2. If Report_Flag is False (non-mandatory):
     - First evaluate if the current frontier has sufficient granularity to answer the core question (no further refinement possible);
     - If sufficient: Generate a final answer, a complete reasoning report and the fault propagation graph;
     - If insufficient: Generate more fine-grained candidate nodes (hypotheses) that refine the current frontier (e.g., if frontier is "downstream dependency failure of orders", generate "billing service failure", "stock service failure", "network delay between orders and its dependencies", etc.).

- Constraints:
  1. Output only standard JSON with no extra content; all output JSON must contain a mandatory top-level field "type" (integer) to identify the output category:
     - Type 1 (Report-type): Mandatory when Report_Flag=True or frontier is sufficiently refined; contains fields "type":1, "answer" (final conclusion, concise text), "report" (complete reasoning chain, detailed text based on belief, graph, and frontier), "graph" (fault propagation graph, see constraint 4);
       - The answer should be within 30 chars, and the report should be within 500 chars;
       - The report shall be a naturally coherent single paragraph that clearly and completely describes the reasoning process.
     - Type 2 (Candidate-type): Only when Report_Flag=False and frontier is insufficiently refined; contains fields "type":2, "candidates" (array of fine-grained hypothesis nodes);
  2. Candidate node specifications (for type=2 JSON):
     - Each candidate must have "id" (format: h+3-digit number, e.g., h001), "label" (fine-grained hypothesis text), "confidence" (0-1 float), "why" (text explaining why this candidate refines the current frontier, linked to belief/graph);
     - Candidates must be more granular than the current frontier and represent the next level of reasoning (no duplicate/irrelevant content);
  3. Final answer (for type=1 JSON) must directly address the core question and align with the frontier's conclusion (if sufficiently refined);
  4. Graph specifications (for type=1 JSON): "graph" is an object with the fields "root_causes", "edges" and "nodes":
     - "root_causes": the root-cause service(s) of the final answer, most likely first, at least one; each {{"component": "<service>", "state": ["<state>"]}};
     - "edges": the fault propagation from the root cause(s) to the alarmed service, one edge per hop that the belief and the existing graph support; each {{"source": "<service>", "target": "<service>"}}, where the source is the service whose failure caused the target's anomaly (cause -> effect, not caller -> callee);
     - "nodes": every service in "root_causes" and "edges", each {{"component": "<service>", "state": ["<state>"]}};
     - Every component, source and target is a service name copied exactly from the list of services in the input (the root cause is reported as a service even when the fault is in one of its pods or on its node); every state is one of: {states};
  5. Ensure no conflicting fields (e.g., type=1 JSON must NOT contain "candidates"; type=2 JSON must NOT contain "answer"/"report"/"graph").

- Execution Instructions:
  1. Parse the Report_Flag to determine if report generation is mandatory;
  2. If Report_Flag = True:
     a. Synthesize the current belief, existing graph, and frontier to form a final answer;
     b. Generate a complete reasoning report explaining how the answer is derived from belief, graph, and frontier;
     c. Write the fault propagation graph of the answer;
     d. Output JSON with "type":1, "answer", "report" and "graph" fields;
  3. If Report_Flag = False:
     a. Evaluate the granularity of the current frontier:
        - Sufficient: Frontier has the finest possible level of detail to answer the core question (e.g., "billing service failure: connection pool exhausted" – no further refinement possible);
        - Insufficient: Frontier is too general and can be broken down into more specific hypotheses (e.g., "downstream dependency failure of orders" – can be refined into a failure of each service orders calls, or of the network between them);
     b. If frontier is sufficient: Output JSON with "type":1, "answer", "report" and "graph" fields;
     c. If frontier is insufficient: Generate fine-grained candidates that refine the frontier, output JSON with "type":2 and "candidates" field;
  4. Ensure JSON output has no syntax errors and strictly complies with the field requirements (no extra/missing fields).

- Examples:
  Example 1 (Report_Flag=True - mandatory report generation, type=1)
  Input:
  - Report_Flag: True
  - current belief: "shop-web error rate rose at 10:02; orders receives 503 from billing since 10:01; billing logs 'connection pool exhausted' since 10:01; ledger-db latency and errors normal"
  - current frontier: "billing service failure (connection pool exhausted)"
  - existing graph: "Nodes include ev001 (shop-web errors from 10:02), ev002 (orders->billing 503 since 10:01), ev003 (billing pool-exhausted errors since 10:01), ev004 (ledger-db normal); edges link ev002/ev003 to h001 (downstream dependency failure of orders), h001 to h002 (billing service failure), ev004 refutes h003 (ledger-db failure)"
  - services: billing, ledger-db, orders, shop-web, stock; alarmed service: shop-web
  Output JSON:
  {_json(_EXAMPLE_1)}

  Example 2 (Report_Flag=False - frontier insufficient, generate candidates, type=2)
  Input:
  - Report_Flag: False
  - current belief: "shop-web latency up since 10:02; orders latency up since 10:02; orders calls billing and stock; no per-dependency data yet"
  - current frontier: "downstream dependency failure of orders"
  - existing graph: "Nodes include ev001 (shop-web latency up), ev002 (orders latency up); edges link ev001/ev002 to h001 (downstream dependency failure of orders), no further sub-hypotheses"
  Output JSON:
  {_json(_EXAMPLE_2)}

  Example 3 (Report_Flag=False - frontier sufficient, generate report, type=1)
  Input:
  - Report_Flag: False
  - current belief: "orders error rate 40% since 10:05 (0% before); orders logs NullPointerException in createOrder since 10:05; billing and stock normal"
  - current frontier: "orders service internal failure (exception in createOrder)"
  - existing graph: "Nodes include ev001 (orders errors 40% since 10:05), ev002 (createOrder NullPointerException), ev003 (billing and stock normal); edges link ev001/ev002 to h001 (orders internal failure), ev003 refutes h002 (downstream dependency failure of orders)"
  - services: billing, ledger-db, orders, shop-web, stock; alarmed service: shop-web
  Output JSON:
  {_json(_EXAMPLE_3)}
"""

    system_prompt = system_prompt + task_prompt
    user_prompt = f"""
The Report_Flag is {Report_Flag}.
Here is the current belief:
{belief}
Here is the descripion of the most possible hypothesis:
{frontier}
Here is the existing graph:
{graph_description}
Here are the services in the system (names for the graph): {", ".join(_CASE["services"])}
The alarmed service is: {_CASE["alarm"]}
"""


    return system_prompt, user_prompt
