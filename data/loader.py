"""
data.loader
===========

Loading D1 (labeled trials) and D2 (deployment traces), plus a simulator that
generates a structurally valid fixture when the real deposit is not to hand.

On the simulator
----------------
:func:`simulate_trials` produces a dataset with the *physics* the paper
describes - MQ-4 drift, cross-sensitivity, an independent reference channel -
and it is calibrated so that a fixed-threshold rule fails in the way Section
4.1 reports. It is **not** a reproduction of the paper's results and must never
be presented as one. Its purposes are:

  * exercising the full pipeline without hardware (CI, review, development)
  * demonstrating what a non-degenerate D1 looks like, against which
    ``data/validate.py`` passes

Every artifact it writes carries ``"source": "simulated"`` in its manifest, and
:func:`load_trials` propagates that flag into the analysis, which refuses to
label simulated output as a paper reproduction.

Reproducing the paper requires the deposited data:
    https://doi.org/10.21227/hyqx-bn32
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from adam.config import (
    SENSOR_ERROR_VARIANCE_RANGE_PPM2,
    EVENTS_PER_TRIAL,
    MQ4_RANGE_PPM,
    N_NODES,
    N_TRIALS,
    SEED,
    THRESHOLD_PPM,
)
from adam.schemas import EventTrace, LabeledEvent, SensorReading

from .validate import (
    DatasetIntegrityError,
    assert_labels_independent,
    check_class_balance,
    check_reference_consistency,
    check_trial_structure,
)

logger = logging.getLogger(__name__)

DEPOSIT_DOI = "https://doi.org/10.21227/hyqx-bn32"


@dataclass
class Dataset:
    """A loaded D1 with its provenance attached."""

    events: List[LabeledEvent]
    source: str  # "deposit" | "simulated"
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_simulated(self) -> bool:
        return self.source == "simulated"

    @property
    def trial_ids(self) -> List[int]:
        return sorted({e.trial_id for e in self.events})

    def trial(self, trial_id: int) -> List[LabeledEvent]:
        return [e for e in self.events if e.trial_id == trial_id]

    def labels(self) -> List[int]:
        return [e.label for e in self.events]

    def primary_ppm(self) -> List[float]:
        return [e.primary.methane_ppm for e in self.events]

    def reference_ppm(self) -> List[float]:
        return [e.reference_ppm for e in self.events]


# ---------------------------------------------------------------------------
# D1 loading
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {"trial_id", "event_index", "timestamp", "label", "reference_ppm"}


def load_trials(
    path: str,
    threshold_ppm: float = THRESHOLD_PPM,
    strict: bool = True,
) -> Dataset:
    """Load D1 from CSV, refusing degenerate labels.

    Expected columns::

        trial_id, event_index, timestamp, label, reference_ppm,
        node_00_ppm, node_00_var, node_01_ppm, node_01_var, ...

    The label-independence guard runs unconditionally. See ``data/validate.py``
    for why it cannot be disabled.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download the deposited datasets from "
            f"{DEPOSIT_DOI}, or generate a fixture with:\n"
            f"    python -m data.loader --simulate --out {path}"
        )

    with open(path, "r", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise DatasetIntegrityError(f"{path} contains no rows")

    missing = _REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise DatasetIntegrityError(f"{path} is missing columns: {sorted(missing)}")

    node_cols = sorted(
        {
            c[: -len("_ppm")]
            for c in rows[0]
            if c.endswith("_ppm") and c != "reference_ppm"
        }
    )
    if not node_cols:
        raise DatasetIntegrityError(
            f"{path} has no per-node <name>_ppm columns; Equation (2) needs "
            f"per-node readings"
        )

    events: List[LabeledEvent] = []
    for row in rows:
        readings: List[SensorReading] = []
        for node in node_cols:
            ppm_raw = row.get(f"{node}_ppm", "")
            if ppm_raw in ("", None):
                continue
            var_raw = row.get(f"{node}_var", "")
            readings.append(
                SensorReading(
                    node_id=node,
                    timestamp=float(row["timestamp"]),
                    methane_ppm=float(ppm_raw),
                    error_variance=float(var_raw) if var_raw else 15.0,
                    reference_ppm=float(row["reference_ppm"]),
                )
            )
        if not readings:
            continue
        events.append(
            LabeledEvent(
                trial_id=int(row["trial_id"]),
                event_index=int(row["event_index"]),
                timestamp=float(row["timestamp"]),
                readings=tuple(readings),
                label=int(row["label"]),
                reference_ppm=float(row["reference_ppm"]),
            )
        )

    manifest_path = os.path.splitext(path)[0] + ".manifest.json"
    manifest: Dict[str, Any] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as fh:
            manifest = json.load(fh)
    source = manifest.get("source", "deposit")

    ds = Dataset(events=events, source=source, manifest=manifest)

    # -- integrity gate
    assert_labels_independent(ds.primary_ppm(), ds.labels(), threshold_ppm)

    warnings: List[str] = []
    sizes = {t: len(ds.trial(t)) for t in ds.trial_ids}
    warnings += check_trial_structure(sizes, N_TRIALS, EVENTS_PER_TRIAL)
    warnings += check_class_balance(ds.labels())
    warnings += check_reference_consistency(ds.reference_ppm(), ds.labels(), threshold_ppm)

    for w in warnings:
        if strict:
            logger.warning("dataset warning: %s", w)
        else:
            logger.debug("dataset warning: %s", w)

    logger.info(
        "loaded %d events across %d trials from %s (source=%s)",
        len(ds.events),
        len(ds.trial_ids),
        path,
        source,
    )
    return ds


# ---------------------------------------------------------------------------
# D2 loading
# ---------------------------------------------------------------------------


def load_deployment_traces(path: str) -> List[EventTrace]:
    """Load D2 from JSON Lines, one serialized EventTrace per line."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download from {DEPOSIT_DOI}, or produce traces "
            f"with `python -m experiments.run_deployment`."
        )
    traces: List[EventTrace] = []
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(EventTrace.from_dict(json.loads(line)))
            except Exception as exc:
                raise DatasetIntegrityError(
                    f"{path}:{lineno} is not a valid EventTrace: {exc}"
                ) from exc
    logger.info("loaded %d deployment traces from %s", len(traces), path)
    return traces


def save_deployment_traces(traces: Sequence[EventTrace], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for t in traces:
            fh.write(t.to_json() + "\n")
    logger.info("wrote %d traces to %s", len(traces), path)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@dataclass
class SimulationParams:
    """Physical parameters of the simulated testbed.

    Defaults produce a dataset in which fixed-threshold detection fails in the
    manner Section 4.1 describes: drift-driven false positives and missed
    detections in the low concentration range, with the reference channel
    unaffected.
    """

    n_trials: int = N_TRIALS
    events_per_trial: int = EVENTS_PER_TRIAL
    n_nodes: int = N_NODES
    threshold_ppm: float = THRESHOLD_PPM

    #: Fraction of events that are true releases.
    anomaly_rate: float = 0.45

    #: Ambient background, ppm.
    background_mean: float = 330.0
    background_sd: float = 45.0

    #: Release concentration distribution, ppm. Centred near the threshold so
    #: that the discrimination problem is non-trivial - a release at 8,000 ppm
    #: needs no learning to detect.
    release_mean: float = 1150.0
    release_sd: float = 520.0

    #: MQ-4 multiplicative drift accumulating within a trial. This is the
    #: non-stationarity a fixed boundary cannot track (Section 5.2).
    drift_per_event: float = 0.0016
    drift_reset_between_trials: bool = True

    #: Multiplicative measurement noise on each MQ-4 channel.
    sensor_noise_cv: float = 0.24

    #: Cross-sensitivity excursions (humidity, VOCs) that lift MQ-4 without any
    #: real methane present - the mechanism behind the static baseline's 0.166 FAR.
    #: These defaults place the fixture's static-threshold baseline near the
    #: F1 = 0.790 / FAR = 0.166 operating point of Table 5, so the fixture
    #: exercises the same discrimination problem. It remains a fixture: matching
    #: the baseline's operating point is not reproducing the paper's results.
    interference_rate: float = 0.21
    interference_gain: float = 2.4

    #: Reference sensor accuracy, Table 4.
    reference_accuracy: float = 0.02

    seed: int = SEED


def simulate_trials(params: Optional[SimulationParams] = None) -> Dataset:
    """Generate a structurally valid D1 fixture.

    NOT a reproduction of the paper's measurements. The returned Dataset is
    tagged ``source="simulated"`` and the analysis layer will refuse to present
    metrics computed over it as paper results.
    """
    p = params or SimulationParams()
    rng = random.Random(p.seed)

    events: List[LabeledEvent] = []
    lo, hi = MQ4_RANGE_PPM
    var_lo, var_hi = SENSOR_ERROR_VARIANCE_RANGE_PPM2

    # Per-node error variance is fixed for the deployment. On the physical
    # testbed it is estimated from residuals of each node's raw readings
    # against the co-located reference over the labeled trials
    # (Section 3.2.2); the fixture draws it from the same range.
    node_var = {
        f"node_{i:02d}": rng.uniform(var_lo, var_hi) for i in range(p.n_nodes)
    }
    node_bias = {n: rng.gauss(0.0, 0.05) for n in node_var}

    t = 0.0
    for trial_id in range(p.n_trials):
        drift = 0.0
        for idx in range(p.events_per_trial):
            t += 1.0
            if p.drift_reset_between_trials:
                drift += p.drift_per_event
            else:
                drift = p.drift_per_event * (trial_id * p.events_per_trial + idx)

            is_release = rng.random() < p.anomaly_rate

            # -- true concentration
            if is_release:
                true_ppm = max(
                    p.background_mean,
                    rng.gauss(p.release_mean, p.release_sd),
                )
            else:
                true_ppm = max(0.0, rng.gauss(p.background_mean, p.background_sd))

            # -- reference channel: accurate, independent of MQ-4 pathologies
            reference_ppm = true_ppm * (1.0 + rng.gauss(0.0, p.reference_accuracy))

            # -- label from the reference channel, never from MQ-4
            label = 1 if is_release and reference_ppm >= p.threshold_ppm else 0

            # -- interference episode: lifts MQ-4 only
            interference = (
                p.interference_gain if rng.random() < p.interference_rate else 1.0
            )

            readings: List[SensorReading] = []
            for node, var in node_var.items():
                observed = (
                    true_ppm
                    * (1.0 + drift)
                    * (1.0 + node_bias[node])
                    * interference
                    * (1.0 + rng.gauss(0.0, p.sensor_noise_cv))
                )
                observed = min(max(observed, lo * 0.5), hi)
                readings.append(
                    SensorReading(
                        node_id=node,
                        timestamp=t,
                        methane_ppm=round(observed, 1),
                        error_variance=round(var, 3),
                        reference_ppm=round(reference_ppm, 1),
                    )
                )

            events.append(
                LabeledEvent(
                    trial_id=trial_id,
                    event_index=idx,
                    timestamp=t,
                    readings=tuple(readings),
                    label=label,
                    reference_ppm=round(reference_ppm, 1),
                )
            )

    ds = Dataset(
        events=events,
        source="simulated",
        manifest={
            "source": "simulated",
            "generator": "data.loader.simulate_trials",
            "params": {k: getattr(p, k) for k in vars(p)},
            "warning": (
                "Synthetic fixture. Structurally valid but NOT a reproduction "
                "of the measurements reported in the manuscript. Reproducing "
                f"the paper requires the deposited data at {DEPOSIT_DOI}."
            ),
        },
    )

    # The fixture must itself pass the guard, or it is no better than the
    # deposit it stands in for.
    assert_labels_independent(ds.primary_ppm(), ds.labels(), p.threshold_ppm)
    return ds


def save_trials(ds: Dataset, path: str) -> None:
    """Write a Dataset to CSV plus a provenance manifest."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    nodes = sorted({r.node_id for e in ds.events for r in e.readings})

    header = ["trial_id", "event_index", "timestamp", "label", "reference_ppm"]
    for n in nodes:
        header += [f"{n}_ppm", f"{n}_var"]

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for e in ds.events:
            by_node = {r.node_id: r for r in e.readings}
            row = [e.trial_id, e.event_index, e.timestamp, e.label, e.reference_ppm]
            for n in nodes:
                r = by_node.get(n)
                row += [r.methane_ppm if r else "", r.error_variance if r else ""]
            w.writerow(row)

    manifest_path = os.path.splitext(path)[0] + ".manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(ds.manifest, fh, indent=2)
    logger.info("wrote %d events to %s (+ manifest)", len(ds.events), path)


def export_from_workbook(workbook_path: str, out_path: str) -> "Dataset":
    """Export D1 from the deposited workbook to the CSV the harnesses read.

    Reads 02_D1_Labeled_Events and writes one row per event with the primary
    node's raw reading, its per-node error variance (recomputed from the
    raw-versus-reference residuals of that node, as in Section 3.2.2), the
    reference reading, and the reference-derived label. Provenance is recorded
    as "deposit" so the analysis can label its outputs accordingly.
    """
    import pandas as pd

    lab = pd.read_excel(workbook_path, sheet_name="02_D1_Labeled_Events")
    resid = pd.to_numeric(lab["Raw_Instantaneous_PPM"], errors="coerce") - \
        pd.to_numeric(lab["Reference_Sensor_PPM"], errors="coerce")
    node_var = resid.groupby(lab["Node_ID"]).var(ddof=1).to_dict()

    events: List[LabeledEvent] = []
    for i, row in lab.iterrows():
        node = str(row["Node_ID"])
        reading = SensorReading(
            node_id=node,
            timestamp=float(pd.Timestamp(row["Timestamp"]).timestamp()),
            methane_ppm=float(row["Raw_Instantaneous_PPM"]),
            error_variance=float(node_var[node]),
            reference_ppm=float(row["Reference_Sensor_PPM"]),
        )
        label = 1 if str(row["Ground_Truth_Label"]).strip().lower() == "anomaly" else 0
        events.append(
            LabeledEvent(
                trial_id=int(row["Trial"]),
                event_index=int(i),
                timestamp=reading.timestamp,
                readings=(reading,),
                label=label,
                reference_ppm=reading.reference_ppm,
            )
        )

    ds = Dataset(
        events=events,
        source="deposit",
        manifest={
            "source": "deposit",
            "workbook": os.path.basename(workbook_path),
            "sheet": "02_D1_Labeled_Events",
            "doi": DEPOSIT_DOI,
        },
    )
    assert_labels_independent(ds.primary_ppm(), ds.labels(), THRESHOLD_PPM)
    save_trials(ds, out_path)
    return ds


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="D1/D2 dataset tooling")
    ap.add_argument("--simulate", action="store_true", help="generate a D1 fixture")
    ap.add_argument("--check", metavar="CSV", help="run integrity checks on a D1 file")
    ap.add_argument(
        "--export", metavar="XLSX",
        help="export D1 from the deposited workbook to the CSV the harnesses read",
    )
    ap.add_argument("--out", default="data/artifacts/d1_simulated.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.export:
        ds = export_from_workbook(args.export, args.out)
        print(f"exported {len(ds.events)} events from {args.export} to {args.out}")
    elif args.simulate:
        ds = simulate_trials(SimulationParams(seed=args.seed))
        save_trials(ds, args.out)
        from .validate import diagnose_labels

        print(diagnose_labels(ds.primary_ppm(), ds.labels()).summary())
    elif args.check:
        from .validate import diagnose_labels

        ds = load_trials(args.check)
        print(f"source: {ds.source}")
        print(diagnose_labels(ds.primary_ppm(), ds.labels()).summary())
    else:
        ap.print_help()
