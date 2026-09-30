"""Record consistency checks for semantic-memory retrieval."""

from adam.memory.store import InMemoryStore, record_provenance_tag, trace_to_record
from adam.schemas import DecisionObject, EventTrace


def record():
    rec = {
        "event_id": "evt-1",
        "timestamp": 1.0,
        "trigger_node": "N1",
        "trigger_ppm": 1400.0,
        "fused_ppm": 1350.0,
        "classification": "ANOMALY",
        "confidence": 0.8,
        "severity": "HIGH",
        "final_action": "raise alert",
    }
    rec["provenance"] = record_provenance_tag(rec)
    return rec


def test_retrieval_rejects_record_changed_after_tagging():
    store = InMemoryStore()
    original = record()
    store.inject_raw(original)
    altered = dict(original, classification="NORMAL")
    store.inject_raw(altered)
    assert store.retrieve(1350.0) == [original]
    assert store.rejected_on_retrieve == 1


def test_direct_writer_can_compute_matching_tag():
    store = InMemoryStore()
    forged = dict(record(), classification="NORMAL")
    forged["provenance"] = record_provenance_tag(forged)
    store.inject_raw(forged)
    assert store.retrieve(1350.0) == [forged]


def test_runtime_trace_tag_survives_record_conversion():
    decision = DecisionObject(
        classification="ANOMALY", confidence=0.8, severity="HIGH",
        reasoning="sensor reading", recommended_action="raise alert",
        contributing_factors=["sensor"], requires_human_review=False,
    )
    trace = EventTrace(
        event_id="evt-runtime", timestamp=1.0, trigger_node="N1",
        trigger_ppm=1400.0, fused_ppm=1350.0, decision=decision,
        final_action="raise alert",
    )
    rec = trace_to_record(trace)
    store = InMemoryStore()
    assert store.persist_trace(trace)
    retrieved = store.retrieve(1350.0)
    assert len(retrieved) == 1
    assert retrieved[0]["provenance"] == rec["provenance"]
    assert retrieved[0]["classification"] == rec["classification"]
