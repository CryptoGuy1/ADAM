"""
adam.llm.prompt
===============

The Decision Agent prompt template and structured-output schema.

This module is the single source of truth for Appendix A of the manuscript.
Running ``python -m adam.llm.prompt --latex`` emits the exact ``lstlisting``
blocks to paste into the appendix, so the prompt in the paper cannot drift from
the prompt the code sends.

Design notes
------------
Gemma 3 1B at INT4 is a small model, and the prompt is shaped around three of
its failure modes observed during the deployment:

1. It will happily emit prose around a JSON object. The instruction to return
   only JSON is repeated at both the start and end of the system prompt, since
   the closing instruction is the one nearest the generation point.
2. It drifts toward ``ANOMALY`` when a numeric threshold appears in context.
   The prompt therefore states the threshold's role explicitly - screening, not
   classification - which is the same distinction Section 3.2.1 draws.
3. It pads ``contributing_factors`` with restatements of the reading. The field
   is capped at three entries and exemplified.

The 256-token cap (Section 3.4.2) is tight for seven fields, so ``reasoning``
is explicitly bounded to two sentences; without that bound the model routinely
truncates mid-object and forces the format-repair retry.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..config import (
    CLASSIFICATION_VALUES,
    DECISION_SCHEMA_FIELDS,
    SEVERITY_LEVELS,
    THRESHOLD_PPM,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Decision Agent of a methane monitoring crew running on an edge node.

Return ONLY a single JSON object. No prose, no markdown fences, no commentary.

Your task is to interpret one candidate methane event using the fused cross-node
estimate, the recent baseline for this location, and similar historical cases
retrieved from the crew's semantic memory. You then recommend an action, which a
separate Coordinator Agent validates against governance policy before anything
executes. You do not execute actions yourself.

The screening threshold of {threshold:.0f} ppm is what caused this event to be
raised for review. It is NOT the classification rule. A reading above it may
still be NORMAL if the baseline is elevated, the cross-node estimate disagrees
with the triggering node, or historical cases at similar concentrations were
benign. A reading near it may be an ANOMALY if it departs sharply from the
recent baseline or corroborating nodes.

Weigh the evidence in this order:
1. Agreement between the triggering node and the fused cross-node estimate. A
   large gap suggests a single-node fault rather than a real release.
2. Departure from the recent baseline window for this location.
3. Precedent from the retrieved historical cases.
4. Absolute concentration.

Emit exactly these seven fields:

  classification         {classifications}
  confidence             number in [0, 1]
  severity               {severities}
  reasoning              at most two sentences citing the specific evidence used
  recommended_action     a short imperative phrase
  contributing_factors   array of at most 3 short strings
  requires_human_review  boolean

Set requires_human_review to true when the evidence conflicts, the cross-node
estimate is dispersed, or confidence is below 0.6.

Return ONLY the JSON object."""


REPAIR_PROMPT = """Your previous response was not a single valid JSON object with the required fields.

Required fields, all seven, nothing else:
{fields}

classification must be one of {classifications}.
severity must be one of {severities}.
confidence must be a number between 0 and 1.
requires_human_review must be true or false.
contributing_factors must be an array of strings.

Re-emit your assessment as ONLY a JSON object. No fences, no prose."""


# ---------------------------------------------------------------------------
# Worked example, used both in the prompt and as the appendix exemplar
# ---------------------------------------------------------------------------

EXAMPLE_OUTPUT: Dict[str, Any] = {
    "classification": "ANOMALY",
    "confidence": 0.82,
    "severity": "HIGH",
    "reasoning": (
        "Three of four nodes agree near 1,350 ppm while node-04 reads 331 ppm "
        "and is flagged as disagreeing, so the fused estimate of 1,130 ppm "
        "understates the corroborated concentration. Two retrieved cases at "
        "comparable readings were confirmed releases."
    ),
    "recommended_action": "Dispatch inspection to node-02 sector and raise alert",
    "contributing_factors": [
        "3 of 4 nodes agree near 1350 ppm",
        "node-04 flagged as disagreeing",
        "2 similar historical releases",
    ],
    "requires_human_review": False,
}


def build_system_prompt(threshold_ppm: float = THRESHOLD_PPM) -> str:
    """Render the system prompt at a given screening threshold."""
    return SYSTEM_PROMPT.format(
        threshold=threshold_ppm,
        classifications=" | ".join(CLASSIFICATION_VALUES),
        severities=" | ".join(SEVERITY_LEVELS),
    )


def build_repair_prompt() -> str:
    """Render the single format-repair instruction. Section 3.4.2."""
    return REPAIR_PROMPT.format(
        fields="\n".join(f"  - {f}" for f in DECISION_SCHEMA_FIELDS),
        classifications=" | ".join(CLASSIFICATION_VALUES),
        severities=" | ".join(SEVERITY_LEVELS),
    )


def _fmt_history(history: Sequence[Dict[str, Any]]) -> str:
    """Render retrieved semantic-memory records, h_past in Equation (3)."""
    if not history:
        return "  (none retrieved)"
    lines = []
    for i, h in enumerate(history, 1):
        lines.append(
            f"  {i}. {h.get('fused_ppm', '?')} ppm on {h.get('date', 'unknown date')}"
            f" -> {h.get('classification', '?')}"
            f" ({h.get('outcome', 'outcome unrecorded')})"
        )
    return "\n".join(lines)


def build_user_prompt(
    trigger_ppm: float,
    trigger_node: str,
    fused_ppm: float,
    node_readings: Sequence[Dict[str, Any]],
    baseline_window: Sequence[float],
    history: Sequence[Dict[str, Any]],
    dispersion_ppm: Optional[float] = None,
    outlier_nodes: Sequence[str] = (),
) -> str:
    """Render the per-event user message.

    Carries the four inputs of Equation (3): the fused estimate m_bar_t, the
    recent temporal context {m_{t-k:t}}, the retrieved history h_past, and the
    triggering reading itself.

    No ground-truth label reaches this function - readings arrive through
    ``SensorReading.redacted()``.
    """
    baseline_mean = (
        sum(baseline_window) / len(baseline_window) if baseline_window else float("nan")
    )
    ratio = fused_ppm / baseline_mean if baseline_mean and baseline_mean > 0 else float("nan")

    node_lines = "\n".join(
        f"  {r.get('node_id', '?')}: {r.get('methane_ppm', '?')} ppm"
        + (f" (weight {1.0/r['error_variance']*1e3:.3f}e-3)"
           if r.get("error_variance") else "")
        for r in node_readings
    )

    parts = [
        "EVENT",
        f"  triggering node:  {trigger_node}",
        f"  triggering value: {trigger_ppm:.1f} ppm",
        "",
        "CROSS-NODE CONTEXT",
        f"  fused estimate:   {fused_ppm:.1f} ppm",
    ]
    if dispersion_ppm is not None:
        parts.append(f"  dispersion:       {dispersion_ppm:.1f} ppm")
    if outlier_nodes:
        parts.append(f"  disagreeing nodes: {', '.join(outlier_nodes)}")
    parts += [
        "  per-node readings:",
        node_lines,
        "",
        "RECENT BASELINE",
        f"  window mean:      {baseline_mean:.1f} ppm over {len(baseline_window)} samples",
        f"  fused / baseline: {ratio:.2f}x",
        "",
        "RETRIEVED HISTORICAL CASES",
        _fmt_history(history),
        "",
        "Return ONLY the JSON object.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON Schema, for documentation and for validating harness fixtures
# ---------------------------------------------------------------------------

OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ADAM Decision Object",
    "type": "object",
    "additionalProperties": False,
    "required": list(DECISION_SCHEMA_FIELDS),
    "properties": {
        "classification": {"enum": list(CLASSIFICATION_VALUES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "severity": {"enum": list(SEVERITY_LEVELS)},
        "reasoning": {"type": "string", "maxLength": 400},
        "recommended_action": {"type": "string", "minLength": 1, "maxLength": 200},
        "contributing_factors": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
        "requires_human_review": {"type": "boolean"},
    },
}


# ---------------------------------------------------------------------------
# Appendix A emitter
# ---------------------------------------------------------------------------

_LATEX_TEMPLATE = r"""% ==========================================================================
%  Appendix A - GENERATED FILE. Do not hand-edit.
%  Produced by: python -m adam.llm.prompt --latex > appendix_a.tex
%  Regenerate whenever adam/llm/prompt.py changes so that the manuscript and
%  the runtime prompt cannot diverge.
% ==========================================================================

The Decision Agent invokes Gemma~3 1B (INT4) locally through Ollama at
temperature $0.1$ with a 256-token response limit (Section~\ref{sec:deployment}).
Listing~\ref{lst:system-prompt} gives the system prompt, Listing~\ref{lst:user-prompt}
the per-event user message rendered for a representative event, and
Listing~\ref{lst:schema} the JSON schema every response must satisfy. On a
schema violation the agent issues the single repair instruction in
Listing~\ref{lst:repair}; if that also fails, or if the model does not respond
within the 30-second decision deadline, the pipeline reverts to deterministic
threshold logic and records \texttt{degraded\_mode=true} in the event trace.

\begin{lstlisting}[style=adamcode,caption={Decision Agent system prompt.},label={lst:system-prompt}]
@@SYSTEM@@
\end{lstlisting}

\begin{lstlisting}[style=adamcode,caption={Per-event user message, rendered for a representative triggered event.},label={lst:user-prompt}]
@@USER@@
\end{lstlisting}

\begin{lstlisting}[style=adamcode,caption={Structured output schema. Responses failing validation trigger one format-repair retry.},label={lst:schema}]
@@SCHEMA@@
\end{lstlisting}

\begin{lstlisting}[style=adamcode,caption={Format-repair instruction, issued at most once per event.},label={lst:repair}]
@@REPAIR@@
\end{lstlisting}

\begin{lstlisting}[style=adamcode,caption={Representative valid response.},label={lst:example}]
@@EXAMPLE@@
\end{lstlisting}
"""


def _representative_user_prompt() -> str:
    """Render the appendix exemplar.

    The fused estimate and dispersion are computed by the same
    ``fuse_readings`` used at runtime rather than hard-coded, so the appendix
    cannot state a value that Equation (2) does not produce from the readings
    printed beside it.
    """
    from ..mechanisms import fuse_readings
    from ..schemas import SensorReading

    # Calibration variances from residuals against the co-located NDIR
    # reference (Section 3.2.2).
    readings = [
        ("node-01", 1298.0, 879.1),
        ("node-02", 1412.0, 1187.0),
        ("node-03", 1355.0, 1125.7),
        ("node-04", 331.0, 1271.5),
    ]
    fusion = fuse_readings(
        [SensorReading(n, 0.0, ppm, error_variance=v) for n, ppm, v in readings],
        outlier_z=1.5,
    )

    return build_user_prompt(
        trigger_ppm=1412.0,
        trigger_node="node-02",
        fused_ppm=fusion.fused_ppm,
        node_readings=[
            {"node_id": n, "methane_ppm": ppm, "error_variance": v}
            for n, ppm, v in readings
        ],
        baseline_window=[318.0, 332.0, 327.0, 340.0, 322.0, 329.0],
        history=[
            {
                "fused_ppm": 1290,
                "date": "2026-03-14",
                "classification": "ANOMALY",
                "outcome": "confirmed release, valve seal",
            },
            {
                "fused_ppm": 1408,
                "date": "2026-03-22",
                "classification": "ANOMALY",
                "outcome": "confirmed release",
            },
            {
                "fused_ppm": 1102,
                "date": "2026-04-02",
                "classification": "NORMAL",
                "outcome": "elevated ambient, no leak found",
            },
        ],
        dispersion_ppm=fusion.dispersion_ppm,
        outlier_nodes=list(fusion.outliers),
    )


def emit_latex() -> str:
    """Render Appendix A as LaTeX listings.

    Token replacement rather than %-formatting or str.format: the template is
    LaTeX, where ``%`` opens a comment and ``{}`` are ubiquitous.
    """
    out = _LATEX_TEMPLATE
    for token, value in (
        ("@@SYSTEM@@", build_system_prompt()),
        ("@@USER@@", _representative_user_prompt()),
        ("@@SCHEMA@@", json.dumps(OUTPUT_JSON_SCHEMA, indent=2)),
        ("@@REPAIR@@", build_repair_prompt()),
        ("@@EXAMPLE@@", json.dumps(EXAMPLE_OUTPUT, indent=2)),
    ):
        out = out.replace(token, value)
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Decision Agent prompt template")
    ap.add_argument("--latex", action="store_true", help="emit Appendix A LaTeX")
    ap.add_argument("--system", action="store_true", help="emit the system prompt")
    ap.add_argument("--user", action="store_true", help="emit a rendered user message")
    ap.add_argument("--schema", action="store_true", help="emit the JSON schema")
    args = ap.parse_args()

    if args.latex:
        print(emit_latex())
    elif args.system:
        print(build_system_prompt())
    elif args.user:
        print(_representative_user_prompt())
    elif args.schema:
        print(json.dumps(OUTPUT_JSON_SCHEMA, indent=2))
    else:
        ap.print_help()
