"""
adam.llm.client
===============

On-device inference through Ollama, plus the failure path Section 3.4.2 and
Section 4.5.2 specify:

    model call -> parse -> [one format-repair retry] -> deterministic fallback

The fallback is the availability mechanism behind Section 4.5.2: terminating
the Ollama process mid-monitoring drops F1 from 0.896 to 0.774 but leaves all 30
crews completing without human intervention, with ``degraded_mode`` written
into the trace so full-reasoning and fallback decisions stay distinguishable in
the audit record.

Confidentiality note
--------------------
``OLLAMA_HOST`` defaults to loopback. Section 4.5.3 reports zero external
egress; :func:`assert_local_endpoint` enforces that invariant at construction
so a misconfigured host cannot silently void the claim.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    LLM_FORMAT_REPAIR_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    THRESHOLD_PPM,
)
from ..schemas import DecisionObject, SchemaViolation
from .prompt import build_repair_prompt, build_system_prompt

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "host.docker.internal"}


class InferenceTimeout(RuntimeError):
    """Raised when generation exceeds the decision deadline."""


class InferenceUnavailable(RuntimeError):
    """Raised when the local model endpoint cannot be reached."""


def assert_local_endpoint(host: str) -> None:
    """Refuse a non-local inference endpoint.

    ADAM's confidentiality claim (Section 4.5.3) is that no sensor context
    leaves the deployment network. Pointing the runtime at a remote Ollama
    would void that silently, so it is rejected loudly instead.
    """
    parsed = urllib.parse.urlparse(host)
    hostname = parsed.hostname or ""
    if hostname in _LOCAL_HOSTS:
        return
    # Private ranges are acceptable: a deployment may run Ollama on a peer node
    # inside the isolated network.
    if hostname.startswith("10.") or hostname.startswith("192.168."):
        return
    if hostname.startswith("172."):
        try:
            second = int(hostname.split(".")[1])
            if 16 <= second <= 31:
                return
        except (IndexError, ValueError):
            pass
    raise InferenceUnavailable(
        f"Refusing non-local inference endpoint {host!r}. ADAM's zero-egress "
        f"claim (Section 4.5.3) requires on-device or in-network inference. "
        f"Use baselines/cloud_only.py for the remote-API comparator."
    )


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Dict[str, Any]:
    """Recover a JSON object from model output.

    Small quantized models wrap objects in fences or prepend a sentence despite
    instruction. Three strategies are tried in order of reliability:
    fenced block, then brace matching that respects string literals, then a
    naive first-to-last brace span.

    Raises
    ------
    SchemaViolation
        When no parseable object is present. The caller converts this into the
        single repair retry.
    """
    if not text or not text.strip():
        raise SchemaViolation("model returned empty output")

    candidates: List[str] = []

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    # Brace matching that ignores braces inside string literals.
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
                start = -1

    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi > lo:
        candidates.append(text[lo : hi + 1])

    for blob in candidates:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise SchemaViolation(
        f"no parseable JSON object in model output (first 200 chars): "
        f"{text[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


def deterministic_fallback(
    fused_ppm: float,
    threshold_ppm: float = THRESHOLD_PPM,
    reason: str = "local model unavailable",
) -> DecisionObject:
    """Threshold-only classification used when reasoning is unavailable.

    This is the behavior measured at F1 = 0.774 in Section 4.5.2 - materially
    below full ADAM, which is the point: the system stays available and says so
    in the trace rather than failing silently or fabricating a judgement.

    Severity is banded off the fused estimate. ``requires_human_review`` is
    always true, since no semantic interpretation stood behind the call.
    """
    is_anomaly = fused_ppm >= threshold_ppm
    ratio = fused_ppm / threshold_ppm if threshold_ppm > 0 else 0.0

    if not is_anomaly:
        severity = "NONE"
    elif ratio >= 5.0:
        severity = "CRITICAL"
    elif ratio >= 2.0:
        severity = "HIGH"
    elif ratio >= 1.5:
        severity = "MODERATE"
    else:
        severity = "LOW"

    obj = DecisionObject(
        classification="ANOMALY" if is_anomaly else "NORMAL",
        confidence=0.50,
        severity=severity,
        reasoning=(
            f"Deterministic fallback: fused estimate {fused_ppm:.1f} ppm "
            f"compared against the {threshold_ppm:.0f} ppm screening threshold. "
            f"No semantic reasoning was applied ({reason})."
        ),
        recommended_action=(
            "Raise alert and request operator review"
            if is_anomaly
            else "Continue monitoring"
        ),
        contributing_factors=["threshold comparison only", reason],
        requires_human_review=True,
    )
    obj.degraded_mode = True
    return obj


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """One Decision Agent inference, with the timing Figure 5 decomposes."""

    decision: DecisionObject
    latency_ms: float
    raw_output: str = ""
    repair_attempted: bool = False
    fell_back: bool = False
    prompt_tokens_est: int = 0


class OllamaClient:
    """Minimal Ollama HTTP client using only the standard library.

    stdlib-only is deliberate: the Raspberry Pi image carries no HTTP client
    beyond what Python ships, and every added dependency is memory the 1B model
    does not get.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        enforce_local: bool = True,
    ) -> None:
        if enforce_local:
            assert_local_endpoint(host)
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -- transport ---------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        url = f"{self.host}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise InferenceUnavailable(
                f"cannot reach Ollama at {url}: {exc}. Is `ollama serve` running "
                f"and has `ollama pull {self.model}` completed?"
            ) from exc
        except TimeoutError as exc:
            raise InferenceTimeout(f"inference exceeded {timeout_s:.1f}s") from exc

    def health(self) -> bool:
        """True when the endpoint responds and the configured model is present."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        base = self.model.split(":")[0]
        return any(n == self.model or n.startswith(base) for n in names)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_s: float,
        temperature: Optional[float] = None,
    ) -> Tuple[str, float]:
        """One raw generation. Returns (text, latency_ms)."""
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens,
            },
        }
        t0 = time.perf_counter()
        resp = self._post("/api/generate", payload, timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return resp.get("response", ""), latency_ms

    # -- the Section 3.4.2 decision path -----------------------------------

    def decide(
        self,
        user_prompt: str,
        fused_ppm: float,
        deadline_s: float,
        threshold_ppm: float = THRESHOLD_PPM,
        repair_retries: int = LLM_FORMAT_REPAIR_RETRIES,
    ) -> InferenceResult:
        """Produce d_t, repairing once and falling back if that fails.

        The deadline is enforced across the whole path, not per call: a repair
        retry that would push the pipeline past C1 is not attempted, since a
        late correct answer is worth less than an on-time degraded one for
        continuous screening.
        """
        system_prompt = build_system_prompt(threshold_ppm)
        started = time.perf_counter()
        raw = ""
        repair_attempted = False

        def elapsed_s() -> float:
            return time.perf_counter() - started

        def remaining_s() -> float:
            return deadline_s - elapsed_s()

        # -- first attempt
        try:
            raw, _ = self.generate(system_prompt, user_prompt, timeout_s=max(remaining_s(), 0.1))
            payload = extract_json(raw)
            decision = DecisionObject.from_model_json(payload)
            return InferenceResult(
                decision=decision,
                latency_ms=elapsed_s() * 1000.0,
                raw_output=raw,
                prompt_tokens_est=len(user_prompt) // 4,
            )
        except (SchemaViolation, ValueError) as exc:
            logger.warning("Decision Agent output off-schema: %s", exc)
        except (InferenceTimeout, InferenceUnavailable) as exc:
            logger.warning("Decision Agent inference failed: %s", exc)
            decision = deterministic_fallback(fused_ppm, threshold_ppm, reason=str(exc)[:80])
            return InferenceResult(
                decision=decision,
                latency_ms=elapsed_s() * 1000.0,
                raw_output=raw,
                fell_back=True,
            )

        # -- single format-repair retry, budget permitting
        for _ in range(max(0, repair_retries)):
            if remaining_s() <= 0.5:
                logger.warning("skipping format repair: decision deadline nearly spent")
                break
            repair_attempted = True
            repair_user = f"{user_prompt}\n\n{build_repair_prompt()}"
            try:
                raw, _ = self.generate(
                    system_prompt, repair_user, timeout_s=max(remaining_s(), 0.1)
                )
                payload = extract_json(raw)
                decision = DecisionObject.from_model_json(payload)
                decision.repair_attempted = True
                return InferenceResult(
                    decision=decision,
                    latency_ms=elapsed_s() * 1000.0,
                    raw_output=raw,
                    repair_attempted=True,
                )
            except (SchemaViolation, ValueError) as exc:
                logger.warning("format repair still off-schema: %s", exc)
            except (InferenceTimeout, InferenceUnavailable) as exc:
                logger.warning("format repair failed: %s", exc)
                break

        # -- deterministic fallback
        decision = deterministic_fallback(
            fused_ppm, threshold_ppm, reason="schema validation failed after repair"
        )
        return InferenceResult(
            decision=decision,
            latency_ms=elapsed_s() * 1000.0,
            raw_output=raw,
            repair_attempted=repair_attempted,
            fell_back=True,
        )


__all__ = [
    "OllamaClient",
    "InferenceResult",
    "InferenceTimeout",
    "InferenceUnavailable",
    "extract_json",
    "deterministic_fallback",
    "assert_local_endpoint",
]
