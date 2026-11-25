"""Cut small real-log fixtures out of downloaded Flight Review logs.

Run from the repo root after `flightcause download`:  uv run python tests/make_fixtures.py
Each fixture keeps only the topics the detectors read, inside a time window (seconds since log start).
"""

from pathlib import Path

import numpy as np
from pyulog import ULog

from flightcause.flight import TOPICS

# name: (log id, start s, end s, reason it was picked)
FIXTURES = {
    "crash_prop_failure": ("5e002593-c2b9-4d0f-b413-ea374e97f92d", 0, 25, "prop retention failure: vibration, then impact and flip"),
    "normal_position_flight": (
        "c0fee141-da2a-4b2f-98e3-9c502fc61e48",
        0,
        80,
        "rated good: stabilized, altitude and position modes, no crash",
    ),
}


def crop(src: Path, dst: Path, t0: float, t1: float) -> None:
    ulog = ULog(str(src), TOPICS)
    start = ulog.start_timestamp
    lo, hi = start + int(t0 * 1e6), start + int(t1 * 1e6)
    for d in ulog.data_list:
        ts = d.data["timestamp"]
        # Keep the last sample before the window so slow topics (arming state, land detector) have a value.
        keep = (ts >= lo) & (ts <= hi)
        before = np.flatnonzero(ts < lo)
        if len(before):
            keep[before[-1]] = True
        d.data = {k: v[keep] for k, v in d.data.items()}
    ulog.data_list[:] = [d for d in ulog.data_list if len(d.data["timestamp"])]
    ulog.logged_messages[:] = [m for m in ulog.logged_messages if lo <= m.timestamp <= hi]
    ulog.write_ulog(str(dst))


if __name__ == "__main__":
    out = Path(__file__).parent / "fixtures"
    for name, (log_id, t0, t1, _) in FIXTURES.items():
        crop(Path("data/logs") / f"{log_id}.ulg", out / f"{name}.ulg", t0, t1)
        print(name, (out / f"{name}.ulg").stat().st_size // 1000, "KB")
