"""Thin wrapper over a ULog file: time-aligned numpy series in seconds since log start."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from pyulog import ULog

# Topics the detectors and plots read. Loading only these keeps parsing fast.
TOPICS = [
    "actuator_motors",
    "actuator_outputs",
    "battery_status",
    "cpuload",
    "estimator_status",
    "input_rc",
    "sensor_combined",
    "sensor_gps",
    "vehicle_attitude",
    "vehicle_attitude_setpoint",
    "vehicle_gps_position",
    "vehicle_imu_status",
    "vehicle_land_detected",
    "vehicle_local_position",
    "vehicle_status",
]

# vehicle_status.nav_state values, named as Flight Review names them.
NAV_STATES = {
    0: "Manual",
    1: "Altitude",
    2: "Position",
    3: "Mission",
    4: "Loiter",
    5: "Return to Land",
    6: "Position (Slow)",
    8: "Altitude Cruise",
    10: "Acro",
    12: "Descend",
    13: "Terminate",
    14: "Offboard",
    15: "Stabilized",
    17: "Takeoff",
    18: "Land",
    19: "Follow Target",
    20: "Precision Land",
    21: "Orbit",
}

ARMED = 2  # vehicle_status.arming_state


@dataclass(frozen=True)
class Series:
    t: np.ndarray  # seconds since log start
    v: np.ndarray

    def at(self, t: np.ndarray | float) -> np.ndarray:
        """Zero-order hold: the last sample at or before each time (first sample before start)."""
        i = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1)
        return self.v[i]

    def window(self, t0: float, t1: float) -> Series:
        m = (self.t >= t0) & (self.t <= t1)
        return Series(self.t[m], self.v[m])


@dataclass(frozen=True)
class Message:
    t: float
    level: str
    text: str


class Flight:
    """One parsed log. log_id is the file stem, which is the Flight Review log id."""

    def __init__(self, path: Path, topics: list[str] | None = TOPICS) -> None:
        self.path = path
        self.log_id = path.stem
        self.ulog = ULog(str(path), topics)
        self.t0 = self.ulog.start_timestamp
        self.duration = (self.ulog.last_timestamp - self.t0) / 1e6

    def series(self, topic: str, field: str, instance: int = 0) -> Series | None:
        for d in self.ulog.data_list:
            if d.name == topic and d.multi_id == instance and field in d.data:
                t = (d.data["timestamp"].astype(np.int64) - self.t0) / 1e6
                return Series(t, d.data[field].astype(np.float64))
        return None

    def instances(self, topic: str) -> list[int]:
        return sorted(d.multi_id for d in self.ulog.data_list if d.name == topic)

    def param(self, name: str, default: float) -> float:
        return float(self.ulog.initial_parameters.get(name, default))

    @cached_property
    def version(self) -> int:
        """ver_sw_release as an int, e.g. 0x010d0000 for v1.13. 0 when unknown."""
        try:
            return int(self.ulog.msg_info_dict.get("ver_sw_release", 0))
        except TypeError, ValueError:
            return 0

    @cached_property
    def messages(self) -> list[Message]:
        # A string message with a trailing tab duplicates an event; skip those like Flight Review does.
        return [
            Message((m.timestamp - self.t0) / 1e6, m.log_level_str(), m.message.strip())
            for m in self.ulog.logged_messages
            if m.message and not m.message.endswith("\t")
        ]

    @cached_property
    def armed(self) -> Series | None:
        s = self.series("vehicle_status", "arming_state")
        return None if s is None else Series(s.t, (s.v == ARMED).astype(np.float64))

    def armed_at(self, t: np.ndarray) -> np.ndarray:
        return np.zeros(len(t), bool) if self.armed is None else self.armed.at(t) > 0.5

    def in_air_at(self, t: np.ndarray) -> np.ndarray:
        """Armed and not landed. Falls back to armed when land detection was not logged."""
        landed = self.series("vehicle_land_detected", "landed")
        armed = self.armed_at(t)
        return armed if landed is None else armed & (landed.at(t) < 0.5)


def spans(t: np.ndarray, mask: np.ndarray, min_s: float = 0.0, gap_s: float = 1.0) -> list[tuple[float, float]]:
    """Turn a boolean mask over times into [start, end] intervals, merging gaps shorter than gap_s."""
    if len(t) == 0 or not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    out: list[tuple[float, float]] = []
    for a, b in zip(edges[::2], edges[1::2] - 1):
        start, end = float(t[a]), float(t[b])
        if out and start - out[-1][1] < gap_s:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return [s for s in out if s[1] - s[0] >= min_s]


def clock(t: float) -> str:
    """Seconds since log start as m:ss.s, the format citations use."""
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"
