#!/usr/bin/env python3
"""
verify_chain.py - independently verify the ADAM deployment against the ledger.

The point of this script
------------------------
Every other artifact from the deployment is a file that can be edited after the
fact. Proof-of-Authority transactions cannot: they carry block timestamps, they
are ordered, and they cannot be inserted retroactively. If the deployment ran as
described, the DecisionLogger contract holds a transaction per committed
decision, and their timestamps fall inside the deployment window.

That makes the chain the only independent witness to D2 available now.

What it can establish
---------------------
  * how many decisions were committed
  * exactly when, to block-timestamp precision
  * the per-hour event rate over the run
  * whatever fields the contract's event actually carries (ppm, classification,
    quorum, degraded_mode) for each one

What it cannot establish
------------------------
  * per-stage latencies, CPU, RAM, or bandwidth - never written on-chain
  * the D1 labeled-trial predictions - those are off-chain entirely

So a clean result here anchors the deployment's event count and timing. It does
not speak to Table 5.

Usage
-----
    pip install web3
    export ADAM_CHAIN_RPC=https://fidesf1-rpc.fidesinnova.io
    python verify_chain.py \
        --address 0xYourDecisionLoggerAddress \
        --start "2025-05-04 09:00:00" \
        --end   "2025-05-07 09:00:00" \
        --out   chain_events.csv

Add --address more than once to sweep several contracts.
If you no longer have the address, pass --from-account 0xYourDeployerAddress to
find contracts it created.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from typing import Any, Dict, List, Optional, Tuple


def _connect(rpc: str):
    try:
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware
    except ImportError:
        sys.exit("web3 is not installed.  pip install web3")

    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
    # PoA chains carry extended extraData that the default validator rejects.
    try:
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except Exception:
        pass
    if not w3.is_connected():
        sys.exit(f"cannot reach {rpc}")
    print(f"connected: chain id {w3.eth.chain_id}, head block {w3.eth.block_number:,}")
    return w3


def _block_at_time(w3, target_ts: int, lo: int, hi: int) -> int:
    """Binary search for the first block at or after target_ts."""
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            ts = w3.eth.get_block(mid).timestamp
        except Exception:
            lo = mid + 1
            continue
        if ts < target_ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _find_window(w3, start: dt.datetime, end: dt.datetime) -> Tuple[int, int]:
    head = w3.eth.block_number
    s = _block_at_time(w3, int(start.timestamp()), 0, head)
    e = _block_at_time(w3, int(end.timestamp()), s, head)
    print(f"deployment window maps to blocks {s:,} .. {e:,}  ({e - s:,} blocks)")
    return s, e


def _fetch_logs(w3, address: str, lo: int, hi: int, chunk: int = 5000) -> List[Any]:
    """eth_getLogs in chunks; public RPCs cap the range per call."""
    out: List[Any] = []
    cur = lo
    while cur <= hi:
        top = min(cur + chunk - 1, hi)
        try:
            got = w3.eth.get_logs(
                {
                    "fromBlock": cur,
                    "toBlock": top,
                    "address": w3.to_checksum_address(address),
                }
            )
            out.extend(got)
        except Exception as exc:
            if chunk > 200:
                # Range too wide for this node; halve and retry the same span.
                chunk //= 2
                print(f"  narrowing chunk to {chunk} blocks ({exc})")
                continue
            print(f"  WARNING: blocks {cur}-{top} failed: {exc}")
        cur = top + 1
        if out and len(out) % 100 < 5:
            print(f"  ... {len(out)} logs so far (block {cur:,})")
    return out


def _decode(w3, log: Any, abi: Optional[List[Dict]]) -> Dict[str, Any]:
    """Decode against the ABI when supplied; otherwise report raw topics."""
    rec: Dict[str, Any] = {
        "block": log["blockNumber"],
        "tx_hash": log["transactionHash"].hex(),
        "log_index": log["logIndex"],
        "topic0": log["topics"][0].hex() if log["topics"] else "",
    }
    if abi:
        contract = w3.eth.contract(abi=abi)
        for item in (a for a in abi if a.get("type") == "event"):
            try:
                ev = contract.events[item["name"]]().process_log(log)
                rec["event"] = item["name"]
                for k, v in ev["args"].items():
                    rec[k] = v.hex() if isinstance(v, (bytes, bytearray)) else v
                break
            except Exception:
                continue
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="verify ADAM deployment events on-chain")
    ap.add_argument("--rpc", default="https://fidesf1-rpc.fidesinnova.io")
    ap.add_argument("--address", action="append", required=True,
                    help="contract address; repeat for several")
    ap.add_argument("--abi", help="path to DecisionLogger.json (optional but better)")
    ap.add_argument("--start", default="2025-05-04 09:00:00")
    ap.add_argument("--end", default="2025-05-07 09:00:00")
    ap.add_argument("--expect", type=int, default=459,
                    help="event count the manuscript claims")
    ap.add_argument("--expect-persisted", type=int, default=446,
                    help="successfully persisted count the manuscript claims")
    ap.add_argument("--out", default="chain_events.csv")
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromisoformat(args.end).replace(tzinfo=dt.timezone.utc)
    hours = (end - start).total_seconds() / 3600.0

    abi = None
    if args.abi:
        with open(args.abi) as fh:
            art = json.load(fh)
        abi = art["abi"] if isinstance(art, dict) and "abi" in art else art

    w3 = _connect(args.rpc)
    lo, hi = _find_window(w3, start, end)

    rows: List[Dict[str, Any]] = []
    ts_cache: Dict[int, int] = {}
    for addr in args.address:
        print(f"\nscanning {addr} ...")
        logs = _fetch_logs(w3, addr, lo, hi)
        print(f"  {len(logs)} logs")
        for lg in logs:
            rec = _decode(w3, lg, abi)
            bn = rec["block"]
            if bn not in ts_cache:
                ts_cache[bn] = w3.eth.get_block(bn).timestamp
            rec["timestamp"] = dt.datetime.fromtimestamp(
                ts_cache[bn], dt.timezone.utc
            ).isoformat()
            rec["contract"] = addr
            rows.append(rec)

    if not rows:
        print("\nNo logs found in that window.")
        print("Check: correct contract address? correct RPC? does this node keep")
        print("archive state that far back? A pruned node will return nothing for")
        print("historical ranges even when the transactions exist.")
        return 1

    rows.sort(key=lambda r: (r["block"], r["log_index"]))
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    # ---- summary
    print("\n" + "=" * 62)
    print(f"events on-chain in window : {len(rows)}")
    print(f"manuscript claims         : {args.expect} coordination events")
    print(f"           of which persisted: {args.expect_persisted}")
    print(f"first : {rows[0]['timestamp']}")
    print(f"last  : {rows[-1]['timestamp']}")
    print(f"rate  : {len(rows)/hours:.2f} events/hour over {hours:.0f} h "
          f"(manuscript: {args.expect/hours:.2f}/h)")

    for label, target in (("total", args.expect), ("persisted", args.expect_persisted)):
        delta = len(rows) - target
        verdict = "MATCHES" if abs(delta) <= max(3, 0.02 * target) else "DIFFERS"
        print(f"  vs {label:<10}: {delta:+d}   {verdict}")

    # Per-hour distribution: a genuine 72-hour run is uneven. A flat profile
    # would itself be worth a second look.
    from collections import Counter
    hrs = Counter(r["timestamp"][:13] for r in rows)
    counts = sorted(hrs.values())
    if counts:
        print(f"\nhourly counts: min {counts[0]}, median {counts[len(counts)//2]}, "
              f"max {counts[-1]}, over {len(hrs)} distinct hours")

    print(f"\nwrote {args.out}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
