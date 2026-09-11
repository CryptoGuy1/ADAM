#!/usr/bin/env python3
"""Degraded-conditions harness for the ADAM methane testbed.

Applies sensor drift, measurement noise, and node loss to the raw per-node MQ-4
stream before any system sees it, so every evaluated system consumes an
identical perturbed stream and paired comparisons are exact.

Design constraints this enforces:

  * Injection happens at the raw per-node reading, ahead of fusion and ahead of
    feature assembly, so a perturbation is meaningful for the single-node
    baselines as well as for the crewed configurations.
  * The random stream is derived from (condition, trial) and never from the
    system under test, so all systems in a condition-trial cell receive
    byte-identical input.
  * Every emitted stream is hashed. The manifest records the seed, the applied
    perturbations, and the digest, so a reader can confirm that two systems
    consumed the same bytes rather than taking it on assertion.
  * Ground-truth labels come from the NDIR reference channel and are never
    perturbed. Degrading the sensor must not move the target.

Input
-----
A per-node reading table with one row per node per event:

    event_id, trial, node_id, timestamp, raw_ppm, reference_ppm

The deposited D1 workbook records one reading per event rather than four, so
this harness reads the per-node acquisition file directly. `--selftest`
generates a synthetic table with the same schema for verifying the harness
itself.

Usage
-----
    python3 degraded_harness.py --input per_node_readings.csv --outdir runs/
    python3 degraded_harness.py --selftest
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HARNESS_VERSION = "1.1"
DROPOUT_NODE = "N4"

CONDITIONS = {
    "clean": {},
    "mild_drift": {
        "drift": {"gain": 1.10, "offset_sigma": 0.5, "schedule": "linear_ramp",
                  "onset_frac": 0.0}
    },
    "strong_drift": {
        "drift": {"gain": 1.35, "offset_sigma": 2.0, "schedule": "linear_ramp",
                  "onset_frac": 0.0}
    },
    "noise": {
        "noise": {"gaussian_sigma_frac": 0.60, "impulse_rate": 0.03,
                  "impulse_sigma": 6.0, "independent_per_node": True}
    },
    "one_node_dropout": {
        "dropout": {"node": DROPOUT_NODE, "onset_frac": 0.33,
                    "duration": "to_end_of_trial"}
    },
}

APPLY_ORDER = ("drift", "noise", "dropout")
REQUIRED_COLUMNS = {
    "event_id", "trial", "node_id", "timestamp", "raw_ppm", "reference_ppm"
}


def validate_input(df):
    """Reject ambiguous or malformed acquisition tables before perturbation."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("input contains no readings")
    if df[["event_id", "trial", "node_id", "timestamp"]].isna().any().any():
        raise ValueError("event/trial/node/timestamp identifiers may not be null")
    if df.duplicated(["trial", "event_id", "node_id"]).any():
        raise ValueError("duplicate node reading within a trial/event")
    if (pd.to_numeric(df["raw_ppm"], errors="coerce") < 0).any():
        raise ValueError("raw_ppm may not be negative")
    if pd.to_numeric(df["raw_ppm"], errors="coerce").isna().any():
        raise ValueError("raw_ppm must be numeric")
    if pd.to_numeric(df["reference_ppm"], errors="coerce").isna().any():
        raise ValueError("reference_ppm must be numeric")

    # The co-located NDIR label is event-level and must not vary by node.
    per_event_refs = df.groupby(["trial", "event_id"])["reference_ppm"].nunique(dropna=False)
    if (per_event_refs != 1).any():
        raise ValueError("reference_ppm differs across nodes for the same event")

    if "trigger_node" in df.columns:
        per_event_trigger = df.groupby(["trial", "event_id"])["trigger_node"].nunique(dropna=False)
        if (per_event_trigger != 1).any():
            raise ValueError("trigger_node differs across rows for the same event")
        for (_trial, _event), g in df.groupby(["trial", "event_id"]):
            trigger = str(g["trigger_node"].iloc[0])
            if trigger not in set(g["node_id"].astype(str)):
                raise ValueError(f"trigger_node {trigger} has no reading in event {_event}")
    return True


def condition_seed(condition, trial):
    """Seed derived from the cell, never from the system under test."""
    h = hashlib.sha256(f"{condition}|{trial}".encode()).hexdigest()[:8]
    return int(h, 16), h


def _node_sigma(df):
    """Per-node standard deviation of the clean raw channel, used as the unit
    for drift offset and noise magnitude so perturbations scale with each
    sensor's own variability."""
    return df.groupby("node_id")["raw_ppm"].std(ddof=1).to_dict()


def apply_condition(df, condition, trial, rng, sigma):
    """Return a perturbed copy of one trial's per-node readings.

    Perturbs raw_ppm only. reference_ppm and the derived label are untouched.
    """
    out = df.copy().sort_values(["timestamp", "node_id"]).reset_index(drop=True)
    spec = CONDITIONS[condition]
    n = len(out)

    # Progress through the trial, in [0, 1], used by ramped perturbations.
    t = out.groupby("node_id").cumcount()
    span = t.groupby(out["node_id"]).transform("max").clip(lower=1)
    frac = (t / span).to_numpy()

    out["perturbed_ppm"] = out["raw_ppm"].astype(float).to_numpy()
    out["node_online"] = True

    if "drift" in spec:
        d = spec["drift"]
        ramp = np.clip((frac - d["onset_frac"]) / max(1e-9, 1 - d["onset_frac"]), 0, 1)
        gain = 1.0 + (d["gain"] - 1.0) * ramp
        off = np.array([sigma[nid] for nid in out["node_id"]]) * d["offset_sigma"] * ramp
        out["perturbed_ppm"] = out["perturbed_ppm"] * gain + off

    if "noise" in spec:
        s = spec["noise"]
        sig = np.array([sigma[nid] for nid in out["node_id"]]) * s["gaussian_sigma_frac"]
        out["perturbed_ppm"] = out["perturbed_ppm"] + rng.normal(0.0, 1.0, n) * sig
        hit = rng.random(n) < s["impulse_rate"]
        spike = rng.choice([-1.0, 1.0], n) * s["impulse_sigma"] * sig
        out.loc[hit, "perturbed_ppm"] = out.loc[hit, "perturbed_ppm"] + spike[hit]

    if "dropout" in spec:
        d = spec["dropout"]
        silent = (out["node_id"] == d["node"]) & (frac >= d["onset_frac"])
        out.loc[silent, "node_online"] = False
        out.loc[silent, "perturbed_ppm"] = np.nan

    # A metal-oxide reading cannot go negative.
    out["perturbed_ppm"] = out["perturbed_ppm"].clip(lower=0.0)
    return out


def stream_digest(out):
    """SHA-256 over all agent-facing replay fields for one emitted stream."""
    cols = ["trial", "event_id", "node_id", "timestamp"]
    if "trigger_node" in out.columns:
        cols.append("trigger_node")
    cols += ["perturbed_ppm", "node_online"]
    b = out[cols].round({"perturbed_ppm": 4}).to_csv(index=False).encode()
    return hashlib.sha256(b).hexdigest()


def run(df, outdir):
    validate_input(df)
    os.makedirs(outdir, exist_ok=True)
    sigma = _node_sigma(df)
    manifest, frames = [], []

    for condition in CONDITIONS:
        for trial, g in df.groupby("trial"):
            seed_int, seed_hex = condition_seed(condition, trial)
            rng = np.random.default_rng(seed_int)
            out = apply_condition(g, condition, trial, rng, sigma)
            out.insert(0, "condition", condition)
            digest = stream_digest(out)
            frames.append(out)
            spec = CONDITIONS[condition]
            manifest.append({
                "harness_version": HARNESS_VERSION,
                "condition": condition,
                "trial": int(trial),
                "seed": seed_hex,
                "drift_applied": "drift" in spec,
                "noise_applied": "noise" in spec,
                "dropout_node": spec.get("dropout", {}).get("node"),
                "dropout_onset_frac": spec.get("dropout", {}).get("onset_frac"),
                "n_readings": int(len(out)),
                "n_offline": int((~out["node_online"]).sum()),
                "stream_sha256": digest,
            })

    streams = pd.concat(frames, ignore_index=True)
    streams.to_csv(os.path.join(outdir, "degraded_streams.csv"), index=False)
    man = pd.DataFrame(manifest)
    man.to_csv(os.path.join(outdir, "harness_manifest.csv"), index=False)
    with open(os.path.join(outdir, "harness_spec.json"), "w") as fh:
        json.dump({"harness_version": HARNESS_VERSION,
                   "seed_policy": "sha256(f'{condition}|{trial}')[:8]",
                   "injection_point": "raw per-node MQ-4 reading",
                   "apply_order": list(APPLY_ORDER),
                   "dropout_node": DROPOUT_NODE,
                   "labels_perturbed": False,
                   "conditions": CONDITIONS}, fh, indent=2)
    return streams, man


def selftest():
    """Verify the harness properties that the paper will claim."""
    rng = np.random.default_rng(0)
    rows = []
    for trial in range(1, 4):
        for k in range(50):
            ref = float(rng.uniform(500, 1600))
            for node in ["N1", "N2", "N3", "N4"]:
                rows.append(dict(event_id=f"T{trial:02d}-{k:03d}", trial=trial,
                                 node_id=node, timestamp=k, trigger_node="N1",
                                 raw_ppm=ref + rng.normal(0, 78),
                                 reference_ppm=ref))
    df = pd.DataFrame(rows)
    streams, man = run(df, "/tmp/harness_selftest")

    ok = True

    # 1. Determinism: the same cell reproduces the same digest.
    s2, m2 = run(df, "/tmp/harness_selftest2")
    same = (man["stream_sha256"] == m2["stream_sha256"]).all()
    print(f"  determinism (identical digests on re-run): {same}")
    ok &= bool(same)

    # 2. Independence from the system under test: the digest is a property of
    #    the cell, so any two systems in that cell get identical bytes.
    print(f"  digests unique per condition-trial cell: "
          f"{man['stream_sha256'].nunique() == len(man)}")
    ok &= man["stream_sha256"].nunique() == len(man)

    # 3. Clean is a no-op.
    c = streams[streams.condition == "clean"]
    noop = np.allclose(c["perturbed_ppm"], c["raw_ppm"])
    print(f"  clean condition leaves the stream unchanged: {noop}")
    ok &= bool(noop)

    # 4. Reference/ground-truth channel is copied byte-for-byte.
    merged = streams.merge(
        df[["trial", "event_id", "node_id", "reference_ppm"]],
        on=["trial", "event_id", "node_id"],
        suffixes=("", "_orig"),
        validate="many_to_one",
    )
    lab = np.allclose(merged["reference_ppm"], merged["reference_ppm_orig"])
    print(f"  reference channel untouched: {lab}")
    ok &= bool(lab)

    # 5. Dropout silences exactly N4 from onset onward and does not alter the
    #    surviving N1--N3 values. Fusion code must then exclude the offline row.
    d = streams[streams.condition == "one_node_dropout"]
    off = d[~d.node_online]
    print(f"  dropout affects only {sorted(off.node_id.unique())} "
          f"({len(off)} of {len(d)} readings)")
    ok &= set(off.node_id.unique()) == {DROPOUT_NODE}
    survivors = d[d.node_id != DROPOUT_NODE]
    unchanged = np.allclose(survivors["perturbed_ppm"], survivors["raw_ppm"])
    print(f"  dropout leaves N1--N3 values unchanged: {unchanged}")
    ok &= bool(unchanged)

    # 6. Severity ordering: strong drift displaces more than mild.
    disp = {}
    for cond in ["mild_drift", "strong_drift", "noise"]:
        g = streams[streams.condition == cond]
        disp[cond] = float((g["perturbed_ppm"] - g["raw_ppm"]).abs().mean())
    print(f"  mean |displacement| ppm: " +
          ", ".join(f"{k} {v:.1f}" for k, v in disp.items()))
    ok &= disp["strong_drift"] > disp["mild_drift"]

    print(f"\n  SELFTEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="per-node readings CSV")
    ap.add_argument("--outdir", default="degraded_run")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.input:
        ap.error("--input is required unless --selftest is given")
    df = pd.read_csv(a.input)
    try:
        streams, man = run(df, a.outdir)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"wrote {len(streams)} perturbed readings across "
          f"{len(man)} condition-trial cells to {a.outdir}/")
    print(man.to_string(index=False))
