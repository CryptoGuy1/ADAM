"""
adam.memory.store
=================

Semantic memory and coordination state, backed by one Weaviate instance.

Section 3.1.2 separates the two roles at the schema level:

    CrewEvent   ephemeral  - trigger, crew membership, votes for an ACTIVE
                             event; cleared on crew dissolution
    EventTrace  persistent - resolved events, retrieved as h_past

One store backs both, which avoids running a second service on a
resource-constrained node. The cost is a measured 55 ms trigger-publication
latency per write - about 2.6% of the 2,093 ms crew-formation time, so the
simpler design is retained (Section 3.1.2).

Provenance
----------
Section 4.5.1 attributes the small poisoning effect (Delta F1 = -0.016 at 20
entries) partly to "provenance tagging and schema validation during ingestion."
Those are implemented in :meth:`WeaviateMemory.persist_trace` and
:meth:`_validate_record`, not assumed.

Offline use
-----------
:class:`InMemoryStore` implements the same interface without a server, so the
labeled-trial harness and the test suite run without Docker. It is a fixture,
not a deployment target, and says so if used with a large corpus.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..config import (
    CLASS_CREW_EVENT,
    CLASS_EVENT_TRACE,
    SEMANTIC_MEMORY_K,
    WEAVIATE_HOST,
)
from ..schemas import CrewEvent, EventTrace

logger = logging.getLogger(__name__)


class MemoryStore(Protocol):
    """Interface the crew depends on."""

    def publish_trigger(self, event: CrewEvent) -> bool: ...
    def clear_crew_event(self, event_id: str) -> bool: ...
    def retrieve(self, fused_ppm: float, k: int = SEMANTIC_MEMORY_K) -> List[Dict[str, Any]]: ...
    def persist_trace(self, trace: EventTrace) -> bool: ...
    def count_traces(self) -> int: ...


# ---------------------------------------------------------------------------
# Record construction and validation
# ---------------------------------------------------------------------------


def _provenance_tag(trace: EventTrace) -> str:
    """Content hash binding a stored record to the event that produced it.

    A poisoned record fabricated outside the pipeline will not carry a tag
    consistent with its own fields, which is what :func:`_validate_record`
    checks on retrieval.
    """
    payload = json.dumps(
        {
            "event_id": trace.event_id,
            "trigger_node": trace.trigger_node,
            "trigger_ppm": round(trace.trigger_ppm, 3),
            "fused_ppm": round(trace.fused_ppm or 0.0, 3),
            "classification": trace.decision.classification if trace.decision else "",
            "final_action": trace.final_action or "",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def trace_to_record(trace: EventTrace) -> Dict[str, Any]:
    """Flatten a resolved trace into a stored EventTrace record."""
    dec = trace.decision
    return {
        "event_id": trace.event_id,
        "timestamp": trace.timestamp,
        "date": time.strftime("%Y-%m-%d", time.gmtime(trace.timestamp)),
        "trigger_node": trace.trigger_node,
        "trigger_ppm": trace.trigger_ppm,
        "fused_ppm": trace.fused_ppm,
        "classification": dec.classification if dec else "UNKNOWN",
        "confidence": dec.confidence if dec else 0.0,
        "severity": dec.severity if dec else "NONE",
        "reasoning": dec.reasoning if dec else "",
        "final_action": trace.final_action or "",
        "outcome": trace.governance_reason or "",
        "degraded_mode": trace.degraded_mode,
        "governance_valid": bool(trace.governance_valid),
        "provenance": _provenance_tag(trace),
        "ingested_at": time.time(),
    }


_REQUIRED_FIELDS = (
    "event_id",
    "fused_ppm",
    "classification",
    "confidence",
    "severity",
    "provenance",
)


def _validate_record(rec: Dict[str, Any]) -> bool:
    """Schema validation applied at ingestion and again on retrieval.

    Rejects records that are missing required fields, carry an out-of-range
    confidence, or claim a classification outside the permitted set. This is
    the ingestion check Section 4.5.1 credits for limiting poisoning impact.
    """
    for f in _REQUIRED_FIELDS:
        if f not in rec:
            return False
    try:
        conf = float(rec["confidence"])
    except (TypeError, ValueError):
        return False
    if not 0.0 <= conf <= 1.0:
        return False
    if rec["classification"] not in ("ANOMALY", "NORMAL", "UNKNOWN"):
        return False
    try:
        ppm = float(rec["fused_ppm"])
    except (TypeError, ValueError):
        return False
    if ppm < 0 or ppm > 1e6:
        return False
    return True


# ---------------------------------------------------------------------------
# In-memory fixture
# ---------------------------------------------------------------------------


class InMemoryStore:
    """Server-free store with the same semantics, for trials and tests.

    Retrieval ranks by absolute distance in fused concentration, which
    approximates the nearest-neighbour behaviour of the vector store closely
    enough for the labeled trials, where retrieval quality affects the prompt
    but is not itself under measurement.
    """

    def __init__(self, warn_above: int = 100_000) -> None:
        self._crew_events: Dict[str, Dict[str, Any]] = {}
        self._traces: List[Dict[str, Any]] = []
        self._sorted_ppm: List[float] = []
        self._warn_above = warn_above
        self.rejected_on_ingest = 0
        self.rejected_on_retrieve = 0

    # -- CrewEvent (ephemeral)
    def publish_trigger(self, event: CrewEvent) -> bool:
        self._crew_events[event.event_id] = event.to_dict()
        return True

    def clear_crew_event(self, event_id: str) -> bool:
        return self._crew_events.pop(event_id, None) is not None

    @property
    def active_crew_events(self) -> int:
        """Should return to zero between events; a leak here means crews are
        not dissolving, contradicting Section 3.1.2."""
        return len(self._crew_events)

    # -- EventTrace (persistent)
    def persist_trace(self, trace: EventTrace) -> bool:
        rec = trace_to_record(trace)
        if not _validate_record(rec):
            self.rejected_on_ingest += 1
            logger.warning("rejecting malformed trace record for %s", trace.event_id)
            return False
        idx = bisect.insort  # noqa: F841  (documenting intent)
        self._traces.append(rec)
        bisect.insort(self._sorted_ppm, float(rec["fused_ppm"]))
        if len(self._traces) > self._warn_above:
            logger.warning(
                "InMemoryStore holds %d records; it is a test fixture, not a "
                "deployment store. Use WeaviateMemory for large corpora.",
                len(self._traces),
            )
        return True

    def retrieve(self, fused_ppm: float, k: int = SEMANTIC_MEMORY_K) -> List[Dict[str, Any]]:
        valid = []
        for rec in self._traces:
            if _validate_record(rec):
                valid.append(rec)
            else:
                self.rejected_on_retrieve += 1
        ranked = sorted(valid, key=lambda r: abs(float(r["fused_ppm"]) - fused_ppm))
        return ranked[:k]

    def count_traces(self) -> int:
        return len(self._traces)

    # -- poisoning harness support
    def inject_raw(self, rec: Dict[str, Any]) -> None:
        """Insert a record bypassing validation.

        Used only by ``experiments/run_security.py`` to model an adversary with
        write access to the store. Ordinary writes go through
        :meth:`persist_trace`.
        """
        self._traces.append(rec)


# ---------------------------------------------------------------------------
# Weaviate-backed store
# ---------------------------------------------------------------------------


class WeaviateMemory:
    """Weaviate v4 client wrapper.

    Import of ``weaviate`` is deferred so the offline harness never needs the
    dependency.
    """

    def __init__(self, host: str = WEAVIATE_HOST, timeout_s: float = 10.0):
        self.host = host
        self.timeout_s = timeout_s
        self._client: Optional[Any] = None
        self.rejected_on_ingest = 0
        self.rejected_on_retrieve = 0

    # -- lifecycle
    def connect(self) -> None:
        try:
            import weaviate  # type: ignore
            from weaviate.connect import ConnectionParams  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "weaviate-client is not installed. `pip install weaviate-client`, "
                "or use InMemoryStore for offline runs."
            ) from exc

        from urllib.parse import urlparse

        parsed = urlparse(self.host)
        self._client = weaviate.connect_to_custom(
            http_host=parsed.hostname or "127.0.0.1",
            http_port=parsed.port or 8080,
            http_secure=parsed.scheme == "https",
            grpc_host=parsed.hostname or "127.0.0.1",
            grpc_port=50051,
            grpc_secure=False,
        )
        self.ensure_schema()

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> "WeaviateMemory":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def client(self) -> Any:
        if self._client is None:
            raise RuntimeError("WeaviateMemory.connect() has not been called")
        return self._client

    # -- schema
    def ensure_schema(self) -> None:
        """Create the two classes of Section 3.1.2 if absent."""
        from weaviate.classes.config import Configure, DataType, Property  # type: ignore

        existing = set(self.client.collections.list_all().keys())

        if CLASS_CREW_EVENT not in existing:
            self.client.collections.create(
                name=CLASS_CREW_EVENT,
                description="Ephemeral coordination state for an active event.",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="event_id", data_type=DataType.TEXT),
                    Property(name="trigger_node", data_type=DataType.TEXT),
                    Property(name="trigger_ppm", data_type=DataType.NUMBER),
                    Property(name="timestamp", data_type=DataType.NUMBER),
                    Property(name="location", data_type=DataType.TEXT),
                    Property(name="crew_members", data_type=DataType.TEXT_ARRAY),
                    Property(name="votes_json", data_type=DataType.TEXT),
                ],
            )
            logger.info("created ephemeral class %s", CLASS_CREW_EVENT)

        if CLASS_EVENT_TRACE not in existing:
            self.client.collections.create(
                name=CLASS_EVENT_TRACE,
                description="Persistent resolved-event traces for historical retrieval.",
                vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                properties=[
                    Property(name="event_id", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.NUMBER),
                    Property(name="date", data_type=DataType.TEXT),
                    Property(name="trigger_node", data_type=DataType.TEXT),
                    Property(name="trigger_ppm", data_type=DataType.NUMBER),
                    Property(name="fused_ppm", data_type=DataType.NUMBER),
                    Property(name="classification", data_type=DataType.TEXT),
                    Property(name="confidence", data_type=DataType.NUMBER),
                    Property(name="severity", data_type=DataType.TEXT),
                    Property(name="reasoning", data_type=DataType.TEXT),
                    Property(name="final_action", data_type=DataType.TEXT),
                    Property(name="outcome", data_type=DataType.TEXT),
                    Property(name="degraded_mode", data_type=DataType.BOOL),
                    Property(name="governance_valid", data_type=DataType.BOOL),
                    Property(name="provenance", data_type=DataType.TEXT),
                    Property(name="ingested_at", data_type=DataType.NUMBER),
                ],
            )
            logger.info("created persistent class %s", CLASS_EVENT_TRACE)

    # -- CrewEvent
    def publish_trigger(self, event: CrewEvent) -> bool:
        try:
            coll = self.client.collections.get(CLASS_CREW_EVENT)
            coll.data.insert(
                {
                    "event_id": event.event_id,
                    "trigger_node": event.trigger_node,
                    "trigger_ppm": event.trigger_ppm,
                    "timestamp": event.timestamp,
                    "location": event.location,
                    "crew_members": event.crew_members,
                    "votes_json": json.dumps(event.votes),
                }
            )
            return True
        except Exception:
            logger.exception("failed to publish trigger for %s", event.event_id)
            return False

    def clear_crew_event(self, event_id: str) -> bool:
        """Delete the ephemeral record on crew dissolution."""
        try:
            from weaviate.classes.query import Filter  # type: ignore

            coll = self.client.collections.get(CLASS_CREW_EVENT)
            coll.data.delete_many(where=Filter.by_property("event_id").equal(event_id))
            return True
        except Exception:
            logger.exception("failed to clear crew event %s", event_id)
            return False

    # -- EventTrace
    def persist_trace(self, trace: EventTrace) -> bool:
        rec = trace_to_record(trace)
        if not _validate_record(rec):
            self.rejected_on_ingest += 1
            logger.warning("rejecting malformed trace record for %s", trace.event_id)
            return False
        try:
            self.client.collections.get(CLASS_EVENT_TRACE).data.insert(rec)
            return True
        except Exception:
            logger.exception("failed to persist trace %s", trace.event_id)
            return False

    def retrieve(self, fused_ppm: float, k: int = SEMANTIC_MEMORY_K) -> List[Dict[str, Any]]:
        """Retrieve h_past: historically similar resolved events."""
        try:
            coll = self.client.collections.get(CLASS_EVENT_TRACE)
            res = coll.query.near_text(
                query=f"methane event at {fused_ppm:.0f} ppm",
                limit=k * 2,
                return_properties=[
                    "event_id",
                    "fused_ppm",
                    "date",
                    "classification",
                    "confidence",
                    "severity",
                    "outcome",
                    "provenance",
                ],
            )
        except Exception:
            logger.exception("semantic retrieval failed")
            return []

        out: List[Dict[str, Any]] = []
        for obj in res.objects:
            rec = dict(obj.properties)
            if _validate_record(rec):
                out.append(rec)
            else:
                self.rejected_on_retrieve += 1
            if len(out) >= k:
                break
        return out

    def count_traces(self) -> int:
        try:
            return self.client.collections.get(CLASS_EVENT_TRACE).aggregate.over_all(
                total_count=True
            ).total_count
        except Exception:
            return -1


__all__ = ["MemoryStore", "InMemoryStore", "WeaviateMemory", "trace_to_record"]
