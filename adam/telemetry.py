"""
adam.telemetry
==============

Per-stage timing and resource sampling.

Figure 5 decomposes T_decision into six stages; Table 6 reports CPU, memory,
bandwidth, and external cost per system. Both come from here, so the numbers in
the paper are a direct readout of the trace rather than a separate measurement
pass.

CPU accounting
--------------
Table 6 reports *peak* CPU during active inference (94.7% for full ADAM).
Constraint C2 governs *sustained* CPU, defined in Section 3.3 as the mean over
the deployment cycle outside active-inference windows, on a rolling five-minute
average. These are different quantities and conflating them is what makes the
94.7% figure look like a C2 violation when it is not. :class:`ResourceSampler`
reports peak; :class:`SustainedCPUMonitor` reports the C2 quantity.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterator, List, Optional, Tuple

from .config import CPU_ROLLING_WINDOW_S, MAX_SUSTAINED_CPU
from .schemas import ResourceCounters, StageLatencies

logger = logging.getLogger(__name__)

try:  # psutil is optional; the harness degrades to /proc parsing without it.
    import psutil  # type: ignore

    _HAVE_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore
    _HAVE_PSUTIL = False


# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------


class StageTimer:
    """Accumulates per-stage wall-clock time for Equation (6).

    Stages may be entered more than once per event (semantic memory is queried
    during aggregation and again on persistence); durations accumulate.
    """

    STAGES = ("T_form", "T_agg", "T_reason", "T_gov", "T_weav", "T_bc")

    def __init__(self) -> None:
        self._elapsed: Dict[str, float] = {s: 0.0 for s in self.STAGES}
        self._order: List[Tuple[str, float, float]] = []

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self._elapsed:
            raise KeyError(f"unknown stage {name!r}; expected one of {self.STAGES}")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            self._elapsed[name] += dt
            self._order.append((name, t0, dt))

    def record(self, name: str, ms: float) -> None:
        """Add a duration measured elsewhere (e.g. inside the LLM client)."""
        if name not in self._elapsed:
            raise KeyError(f"unknown stage {name!r}")
        self._elapsed[name] += ms

    @property
    def total_ms(self) -> float:
        return sum(self._elapsed.values())

    def to_stage_latencies(self) -> StageLatencies:
        return StageLatencies(**self._elapsed)

    def timeline(self) -> List[Tuple[str, float, float]]:
        """(stage, start_perf_counter, duration_ms) in entry order."""
        return list(self._order)


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------


def _read_proc_cpu() -> Optional[Tuple[float, float]]:
    """Return (busy_jiffies, total_jiffies) from /proc/stat, or None."""
    try:
        with open("/proc/stat", "r") as fh:
            parts = fh.readline().split()
    except OSError:
        return None
    if not parts or parts[0] != "cpu":
        return None
    vals = [float(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
    total = sum(vals)
    return total - idle, total


def _read_proc_mem_mb() -> float:
    """Resident set size of this process tree in MB."""
    if _HAVE_PSUTIL:
        try:
            proc = psutil.Process()
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except Exception:
                    pass
            return rss / (1024 * 1024)
        except Exception:
            pass
    try:
        with open("/proc/self/statm", "r") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


def _read_net_bytes() -> int:
    """Cumulative bytes across all interfaces except loopback."""
    if _HAVE_PSUTIL:
        try:
            counters = psutil.net_io_counters(pernic=True)
            return sum(
                c.bytes_sent + c.bytes_recv
                for nic, c in counters.items()
                if not nic.startswith("lo")
            )
        except Exception:
            pass
    total = 0
    try:
        with open("/proc/net/dev", "r") as fh:
            for line in fh.readlines()[2:]:
                nic, _, rest = line.partition(":")
                if nic.strip().startswith("lo"):
                    continue
                fields = rest.split()
                if len(fields) >= 9:
                    total += int(fields[0]) + int(fields[8])
    except OSError:
        pass
    return total


class ResourceSampler:
    """Samples CPU, memory, and network across one event.

    Reports the *peak* CPU observed during the sampled window, which is the
    Table 6 quantity. Sampling runs on a background thread at 10 Hz; on a
    Raspberry Pi 5 this costs well under 1% of a core.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cpu_samples: List[float] = []
        self._mem_samples: List[float] = []
        self._net_start = 0
        self._t_start = 0.0

    def _loop(self) -> None:
        prev = _read_proc_cpu()
        while not self._stop.wait(self.interval_s):
            if _HAVE_PSUTIL:
                try:
                    self._cpu_samples.append(psutil.cpu_percent(interval=None))
                except Exception:
                    pass
            else:
                cur = _read_proc_cpu()
                if prev and cur:
                    d_busy, d_total = cur[0] - prev[0], cur[1] - prev[1]
                    if d_total > 0:
                        self._cpu_samples.append(100.0 * d_busy / d_total)
                prev = cur
            self._mem_samples.append(_read_proc_mem_mb())

    def start(self) -> None:
        self._stop.clear()
        self._cpu_samples.clear()
        self._mem_samples.clear()
        self._net_start = _read_net_bytes()
        self._t_start = time.perf_counter()
        if _HAVE_PSUTIL:
            try:
                psutil.cpu_percent(interval=None)  # prime the differential
            except Exception:
                pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceCounters:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        elapsed = max(time.perf_counter() - self._t_start, 1e-6)
        net_delta = max(_read_net_bytes() - self._net_start, 0)

        return ResourceCounters(
            cpu_peak_pct=max(self._cpu_samples) if self._cpu_samples else 0.0,
            cpu_sustained_pct=(
                sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else 0.0
            ),
            memory_mb=max(self._mem_samples) if self._mem_samples else 0.0,
            bandwidth_kbps=(net_delta / 1024.0) / elapsed,
            external_bytes=0,  # set by the egress monitor, not by NIC totals
            api_cost_usd=0.0,
        )


class SustainedCPUMonitor:
    """Rolling-average CPU for constraint C2.

    Section 3.3 defines sustained utilization as the deployment-cycle mean
    outside active-inference windows, on a rolling five-minute average. Samples
    taken while :meth:`inference_window` is held are excluded, which is exactly
    what makes 94.7% peak and 22% sustained consistent with each other.
    """

    def __init__(self, window_s: float = CPU_ROLLING_WINDOW_S, interval_s: float = 1.0):
        self.window_s = window_s
        self.interval_s = interval_s
        self._samples: Deque[Tuple[float, float]] = deque()
        self._in_inference = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def inference_window(self) -> Iterator[None]:
        """Mark a span as active inference, excluding it from the C2 average."""
        with self._lock:
            self._in_inference = True
        try:
            yield
        finally:
            with self._lock:
                self._in_inference = False

    def _loop(self) -> None:
        prev = _read_proc_cpu()
        while not self._stop.wait(self.interval_s):
            with self._lock:
                skip = self._in_inference
            if _HAVE_PSUTIL:
                try:
                    pct = psutil.cpu_percent(interval=None)
                except Exception:
                    continue
            else:
                cur = _read_proc_cpu()
                if not (prev and cur):
                    prev = cur
                    continue
                d_busy, d_total = cur[0] - prev[0], cur[1] - prev[1]
                prev = cur
                if d_total <= 0:
                    continue
                pct = 100.0 * d_busy / d_total
            if skip:
                continue
            now = time.time()
            with self._lock:
                self._samples.append((now, pct))
                cutoff = now - self.window_s
                while self._samples and self._samples[0][0] < cutoff:
                    self._samples.popleft()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    @property
    def sustained_pct(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(p for _, p in self._samples) / len(self._samples)

    def satisfies_c2(self) -> bool:
        """True when sustained utilization is within the C2 budget."""
        return self.sustained_pct <= MAX_SUSTAINED_CPU * 100.0


# ---------------------------------------------------------------------------
# External egress accounting
# ---------------------------------------------------------------------------


class EgressMonitor:
    """Counts bytes sent to endpoints outside the deployment network.

    Section 4.5.3 reports zero external egress for ADAM against ~43 KB/s for
    Cloud-Only. That is a claim about destinations, not volume, so it is
    accounted at the call site: the only component that increments this counter
    is ``baselines/cloud_only.py``.
    """

    def __init__(self) -> None:
        self._bytes = 0
        self._calls = 0
        self._cost_usd = 0.0
        self._destinations: Dict[str, int] = {}

    def record(self, destination: str, n_bytes: int, cost_usd: float = 0.0) -> None:
        self._bytes += n_bytes
        self._calls += 1
        self._cost_usd += cost_usd
        self._destinations[destination] = self._destinations.get(destination, 0) + n_bytes

    @property
    def total_bytes(self) -> int:
        return self._bytes

    @property
    def total_cost_usd(self) -> float:
        return self._cost_usd

    @property
    def call_count(self) -> int:
        return self._calls

    def summary(self) -> Dict[str, object]:
        return {
            "external_bytes": self._bytes,
            "external_calls": self._calls,
            "cost_usd": round(self._cost_usd, 6),
            "destinations": dict(self._destinations),
        }

    def reset(self) -> None:
        self._bytes = 0
        self._calls = 0
        self._cost_usd = 0.0
        self._destinations.clear()


#: Process-wide egress ledger. Asserted to be zero for ADAM runs.
EGRESS = EgressMonitor()


__all__ = [
    "StageTimer",
    "ResourceSampler",
    "SustainedCPUMonitor",
    "EgressMonitor",
    "EGRESS",
]
