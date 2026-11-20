"""Deterministic detectors. Each scans one Flight and returns timestamped Findings with evidence.

Thresholds are fixed constants so a finding can always be traced to a rule.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from volatus.flight import NAV_STATES, Flight, Series, clock, spans

Severity = Literal["info", "warning", "critical"]
G = 9.81


@dataclass(frozen=True)
class Finding:
    log_id: str
    kind: str
    severity: Severity
    t_start: float
    t_end: float
    summary: str
    evidence: dict[str, float | str] = field(default_factory=dict)
    # "topic/field" or "topic/field/instance" to plot around this moment.
    signals: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return f"{self.log_id[:8]}-{self.kind}-{self.t_start:.1f}"

    @property
    def when(self) -> str:
        if self.t_end - self.t_start < 0.5:
            return clock(self.t_start)
        return f"{clock(self.t_start)}-{clock(self.t_end)}"


Detector = Callable[[Flight], list[Finding]]


def _r(x: float, nd: int = 2) -> float:
    return round(float(x), nd)


def battery(f: Flight) -> list[Finding]:
    """Low cell voltage, sudden voltage sag under load, and PX4's own battery warnings."""
    v = f.series("battery_status", "voltage_v")
    if v is None:
        return []
    air = f.in_air_at(v.t)
    if not air.any() or v.v[air].max() < 3.0:
        return []
    cells = f.series("battery_status", "cell_count")
    n = int(cells.v.max()) if cells is not None else 0
    if not 3.4 <= v.v[air].max() / max(n, 1) <= 4.45:
        # cell_count is often left at a wrong default; infer it from the in-flight peak voltage instead.
        n = max(1, round(v.v[air].max() / 4.0))
    cell = v.v / n
    sig = ("battery_status/voltage_v", "battery_status/current_a")
    out: list[Finding] = []

    for a, b in spans(v.t, air & (cell < 3.3) & (v.v > 1.0), min_s=1.0, gap_s=30.0):
        w = v.window(a, b)
        out.append(
            Finding(
                f.log_id,
                "battery_low",
                "critical" if w.v.min() / n < 3.1 else "warning",
                a,
                b,
                f"Battery fell to {w.v.min() / n:.2f} V/cell ({w.v.min():.1f} V, {n}S) in flight",
                {"min_cell_v": _r(w.v.min() / n), "min_v": _r(w.v.min()), "cells": n},
                sig,
            )
        )

    # Sag: drop of >=0.25 V/cell below the 5 s rolling max, while in the air.
    if len(v.t) > 2:
        dt = float(np.median(np.diff(v.t))) or 0.1
        k = max(1, int(5.0 / dt))
        roll_max = np.array([v.v[max(0, i - k) : i + 1].max() for i in range(len(v.v))])
        drop = (roll_max - v.v) / n
        for a, b in spans(v.t, air & (drop > 0.25) & (v.v > 1.0), min_s=0.3):
            i = int(np.argmax(np.where((v.t >= a) & (v.t <= b), drop, -1)))
            cur = f.series("battery_status", "current_a")
            amps = float(cur.at(v.t[i])) if cur is not None else float("nan")
            out.append(
                Finding(
                    f.log_id,
                    "battery_sag",
                    "warning",  # sag under load is common in good flights too; it is context, not a verdict
                    a,
                    b,
                    f"Battery voltage sagged {drop[i]:.2f} V/cell ({roll_max[i]:.1f} -> {v.v[i]:.1f} V) within 5 s"
                    + (f" at {amps:.0f} A" if amps > 1 else ""),
                    {"drop_per_cell_v": _r(drop[i]), "from_v": _r(roll_max[i]), "to_v": _r(v.v[i]), "current_a": _r(amps, 1)},
                    sig,
                )
            )

    warn = f.series("battery_status", "warning")
    if warn is not None:
        names = {2: "critical", 3: "emergency", 4: "failed"}
        on = f.armed_at(warn.t) & (warn.v >= 2)
        # First time each level is reached; the level flickers between values near the threshold.
        levels, first = np.unique(warn.v[on].astype(int), return_index=True)
        for t, level in zip(warn.t[on][first], levels):
            name = names.get(level, str(level))
            out.append(
                Finding(
                    f.log_id,
                    "battery_warning",
                    "critical",
                    float(t),
                    float(t),
                    f"PX4 raised battery warning '{name}'",
                    {"warning_level": name},
                    sig,
                )
            )
    return out


ROTORS = {2: 4, 13: 6, 14: 8, 15: 3}  # vehicle_status.system_type (MAV_TYPE) -> motor count


def _motor_outputs(f: Flight) -> tuple[list[Series], float, float, str] | None:
    """Motor commands and their [min, max] range. Prefers normalized actuator_motors (PX4 >= 1.13).

    Mixers put motors on the first outputs, so only the first n channels are read, n from the airframe type.
    """
    st = f.series("vehicle_status", "system_type")
    n = ROTORS.get(int(st.v[-1]), 4) if st is not None else 4
    if f.series("actuator_motors", "control[0]") is not None:
        ms = [s for i in range(n) if (s := f.series("actuator_motors", f"control[{i}]")) is not None]
        return (ms, 0.0, 1.0, "actuator_motors/control") if len(ms) == n else None
    ms = [s for i in range(n) if (s := f.series("actuator_outputs", f"output[{i}]")) is not None]
    # PWM motor channels idle above ~900 us while armed; anything else means this is not a PWM motor layout.
    if len(ms) < n or any(np.percentile(s.v, 95) < 1100 or s.v.max() > 2200 for s in ms):
        return None
    pmax = f.param("PWM_MAIN_MAX", f.param("PWM_MAX", 2000.0))
    pmin = f.param("PWM_MAIN_MIN", f.param("PWM_MIN", 1000.0))
    return ms, pmin, pmax, "actuator_outputs/output"


def motor_saturation(f: Flight) -> list[Finding]:
    """A motor pinned at max output while flying: the controller has run out of authority."""
    got = _motor_outputs(f)
    if got is None:
        return []
    ms, lo, hi, prefix = got
    t = ms[0].t
    vals = np.vstack([np.interp(t, s.t, s.v) for s in ms])
    near_max = vals >= hi - 0.03 * (hi - lo)
    near_min = vals <= lo + 0.05 * (hi - lo)
    air = f.in_air_at(t)
    out: list[Finding] = []
    for a, b in spans(t, air & near_max.any(axis=0), min_s=0.3, gap_s=0.5):
        m = (t >= a) & (t <= b)
        frac = near_max[:, m].mean(axis=1)
        worst = int(np.argmax(frac))
        split = bool((near_max[:, m].any(axis=0) & near_min[:, m].any(axis=0)).any())
        out.append(
            Finding(
                f.log_id,
                "motor_saturation",
                "critical" if b - a >= 2.0 else "warning",
                a,
                b,
                f"Motor {worst + 1} saturated at max output for {b - a:.1f} s"
                + ("; another motor was at minimum at the same time (yaw/attitude authority exhausted or a lost motor)" if split else ""),
                {"motor": worst + 1, "duration_s": _r(b - a, 1), "max_and_min_together": str(split), "n_motors": len(ms)},
                tuple(f"{prefix}[{i}]" for i in range(min(len(ms), 8))),
            )
        )
    return out


def vibration(f: Flight) -> list[Finding]:
    """High IMU vibration (PX4 >= 1.13 metric, bands from Flight Review) and accelerometer clipping."""
    out: list[Finding] = []
    vib = f.series("vehicle_imu_status", "accel_vibration_metric")
    if vib is not None and f.version >= 0x010D0000:
        air = f.in_air_at(vib.t)
        for a, b in spans(vib.t, air & (vib.v > 4.905), min_s=2.0, gap_s=2.0):
            peak = vib.window(a, b).v.max()
            out.append(
                Finding(
                    f.log_id,
                    "vibration",
                    "critical" if peak > 9.81 else "warning",
                    a,
                    b,
                    f"Accel vibration metric reached {peak:.1f} m/s^2 (Flight Review: >4.9 marginal, >9.8 bad)",
                    {"peak_m_s2": _r(peak, 1)},
                    ("vehicle_imu_status/accel_vibration_metric", "sensor_combined/accelerometer_m_s2[2]"),
                )
            )
    else:
        # Older PX4: high-pass the raw accelerometer (subtract a 0.2 s moving mean) and take 1 s RMS.
        z = f.series("sensor_combined", "accelerometer_m_s2[2]")
        if z is not None and len(z.t) > 100:
            dt = float(np.median(np.diff(z.t)))
            k = max(3, int(0.2 / dt))
            hp = z.v - np.convolve(z.v, np.ones(k) / k, mode="same")
            w = max(3, int(1.0 / dt))
            rms = np.sqrt(np.convolve(hp**2, np.ones(w) / w, mode="same"))
            air = f.in_air_at(z.t)
            for a, b in spans(z.t, air & (rms > VIB_RMS_WARN), min_s=2.0, gap_s=2.0):
                peak = float(rms[(z.t >= a) & (z.t <= b)].max())
                out.append(
                    Finding(
                        f.log_id,
                        "vibration",
                        "critical" if peak > 2 * VIB_RMS_WARN else "warning",
                        a,
                        b,
                        f"Z-axis accelerometer vibration {peak:.1f} m/s^2 RMS (high-passed, 1 s windows)",
                        {"peak_rms_m_s2": _r(peak, 1)},
                        ("sensor_combined/accelerometer_m_s2[2]",),
                    )
                )
    for topic, fld in (("vehicle_imu_status", "accel_clipping[2]"), ("sensor_combined", "accelerometer_clipping")):
        c = f.series(topic, fld)
        if c is None:
            continue
        rising = np.r_[False, np.diff(c.v) > 0] if topic == "vehicle_imu_status" else c.v > 0
        air = f.in_air_at(c.t)
        for a, b in spans(c.t, air & rising, min_s=0.0, gap_s=2.0):
            out.append(
                Finding(
                    f.log_id,
                    "accel_clipping",
                    "warning",
                    a,
                    b,
                    "Accelerometer clipped (measurement exceeded sensor range) in flight",
                    {"source": f"{topic}.{fld}"},
                    ("sensor_combined/accelerometer_m_s2[2]",),
                )
            )
        break
    return out


VIB_RMS_WARN = 4.0


def impact(f: Flight) -> list[Finding]:
    """Hard impacts (50 ms mean accel > 5 g while armed) and PX4's free-fall flag.

    Averaging over 50 ms ignores single-sample vibration spikes, which reach 20 g in good flights.
    """
    out: list[Finding] = []
    xs = [s for i in range(3) if (s := f.series("sensor_combined", f"accelerometer_m_s2[{i}]")) is not None]
    if len(xs) == 3:
        t = xs[0].t
        raw = np.sqrt(sum(x.v**2 for x in xs))
        w = max(1, int(0.05 / (float(np.median(np.diff(t))) or 0.004)))
        norm = np.convolve(raw, np.ones(w) / w, "same")
        armed = f.armed_at(t)
        for a, b in spans(t, armed & (norm > 5 * G), gap_s=3.0):
            peak = float(norm[(t >= a) & (t <= b)].max())
            out.append(
                Finding(
                    f.log_id,
                    "impact",
                    "critical",
                    a,
                    b,
                    f"Acceleration of {peak / G:.1f} g (50 ms mean) while armed: impact or crash",
                    {"peak_g": _r(peak / G, 1)},
                    ("sensor_combined/accelerometer_m_s2[2]", "vehicle_local_position/z"),
                )
            )
    ff = f.series("vehicle_land_detected", "freefall")
    if ff is not None:
        for a, b in spans(ff.t, f.armed_at(ff.t) & (ff.v > 0.5)):
            out.append(
                Finding(
                    f.log_id,
                    "freefall",
                    "critical",
                    a,
                    b,
                    "PX4 land detector flagged free fall",
                    {},
                    ("vehicle_local_position/z", "vehicle_local_position/vz"),
                )
            )
    return out


def euler(f: Flight) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    q = [s for i in range(4) if (s := f.series("vehicle_attitude", f"q[{i}]")) is not None]
    if len(q) < 4:
        return None
    w, x, y, z = (s.v for s in q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return q[0].t, np.degrees(roll), np.degrees(pitch)


def _setpoint_euler(f: Flight, t: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    r = f.series("vehicle_attitude_setpoint", "roll_body")
    p = f.series("vehicle_attitude_setpoint", "pitch_body")
    if r is not None and p is not None:
        return np.degrees(r.at(t)), np.degrees(p.at(t))
    q = [s for i in range(4) if (s := f.series("vehicle_attitude_setpoint", f"q_d[{i}]")) is not None]
    if len(q) < 4:
        return None
    w, x, y, z = (s.at(t) for s in q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.degrees(roll), np.degrees(pitch)


def attitude(f: Flight) -> list[Finding]:
    """Attitude not tracking its setpoint (>30 deg for 0.5 s) and extreme tilt (>60 deg: flip)."""
    e = euler(f)
    if e is None:
        return []
    t, roll, pitch = e
    armed = f.armed_at(t)
    air = f.in_air_at(t)
    out: list[Finding] = []
    sig = ("vehicle_attitude/roll_deg", "vehicle_attitude/pitch_deg")
    sp = _setpoint_euler(f, t)
    if sp is not None:
        err = np.hypot(roll - sp[0], pitch - sp[1])
        for a, b in spans(t, air & (err > 30), min_s=0.5, gap_s=0.5):
            peak = float(err[(t >= a) & (t <= b)].max())
            out.append(
                Finding(
                    f.log_id,
                    "attitude_error",
                    "critical" if peak > 60 and b - a >= 1.0 else "warning",
                    a,
                    b,
                    f"Attitude diverged from setpoint by up to {peak:.0f} deg for {b - a:.1f} s",
                    {"peak_error_deg": _r(peak, 0), "duration_s": _r(b - a, 1)},
                    sig,
                )
            )
    tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(roll)) * np.cos(np.radians(pitch)), -1, 1)))
    for a, b in spans(t, armed & (tilt > 60), min_s=0.2):
        peak = float(tilt[(t >= a) & (t <= b)].max())
        out.append(
            Finding(
                f.log_id,
                "flip",
                "critical",
                a,
                b,
                f"Vehicle tilted to {peak:.0f} deg while armed (loss of control or tip-over)",
                {"peak_tilt_deg": _r(peak, 0)},
                sig,
            )
        )
    return out


def gps(f: Flight) -> list[Finding]:
    """GPS fix lost or degraded (fix < 3D, or horizontal accuracy > 5 m) after a good fix, while armed."""
    topic = "vehicle_gps_position" if f.series("vehicle_gps_position", "fix_type") else "sensor_gps"
    fix = f.series(topic, "fix_type")
    eph = f.series(topic, "eph")
    if fix is None or eph is None or not (fix.v >= 3).any():
        return []
    first_good = fix.t[np.argmax(fix.v >= 3)]
    armed = f.armed_at(fix.t) & (fix.t > first_good)
    sats = f.series(topic, "satellites_used")
    out: list[Finding] = []
    for a, b in spans(fix.t, armed & ((fix.v < 3) | (eph.v > 5.0)), min_s=1.0):
        w = (fix.t >= a) & (fix.t <= b)
        lost = bool((fix.v[w] < 3).any())
        evidence: dict[str, float | str] = {"min_fix_type": _r(fix.v[w].min(), 0), "max_eph_m": _r(eph.v[w].max(), 1)}
        if sats is not None:
            evidence["min_satellites"] = _r(sats.v[w].min(), 0)
        out.append(
            Finding(
                f.log_id,
                "gps_loss" if lost else "gps_degraded",
                "critical" if lost else "warning",
                a,
                b,
                ("GPS 3D fix lost" if lost else f"GPS accuracy degraded (eph up to {eph.v[w].max():.1f} m)")
                + f" for {b - a:.1f} s while armed",
                evidence,
                (f"{topic}/fix_type", f"{topic}/eph", f"{topic}/satellites_used"),
            )
        )
    return out


def estimator(f: Flight) -> list[Finding]:
    """EKF innovation test ratios above 1 (measurements rejected) and filter faults, while flying."""
    out: list[Finding] = []
    for name in ("vel_test_ratio", "pos_test_ratio", "hgt_test_ratio", "mag_test_ratio"):
        s = f.series("estimator_status", name)
        if s is None:
            continue
        air = f.in_air_at(s.t)
        for a, b in spans(s.t, air & (s.v > 1.0), min_s=1.0, gap_s=2.0):
            peak = float(s.window(a, b).v.max())
            what = name.split("_")[0]
            out.append(
                Finding(
                    f.log_id,
                    "estimator",
                    "critical" if peak > 2.0 and what in ("vel", "pos", "hgt") else "warning",
                    a,
                    b,
                    f"EKF {what} innovation test ratio {peak:.1f} (>1 means the estimator rejects that sensor)",
                    {"ratio": name, "peak": _r(peak)},
                    (f"estimator_status/{name}",),
                )
            )
    ff = f.series("estimator_status", "filter_fault_flags")
    if ff is not None:
        for a, b in spans(ff.t, f.armed_at(ff.t) & (ff.v > 0)):
            out.append(
                Finding(
                    f.log_id,
                    "estimator",
                    "warning",
                    a,
                    b,
                    "EKF reported internal filter faults",
                    {"filter_fault_flags": int(ff.window(a, b).v.max())},
                    ("estimator_status/filter_fault_flags",),
                )
            )
    return out


def failsafe(f: Flight) -> list[Finding]:
    """Failsafe activation, RC signal loss and data link loss from vehicle_status."""
    out: list[Finding] = []
    nav = f.series("vehicle_status", "nav_state")
    checks: tuple[tuple[str, str, str, Severity], ...] = (
        ("failsafe", "failsafe", "Failsafe triggered", "warning"),
        ("rc_signal_lost", "rc_loss", "RC signal lost", "warning"),
        ("data_link_lost", "datalink_loss", "Ground station data link lost", "warning"),
    )
    for fld, kind, text, sev in checks:
        s = f.series("vehicle_status", fld)
        if s is None:
            continue
        for a, b in spans(s.t, f.in_air_at(s.t) & (s.v > 0.5), gap_s=0.5):
            mode = NAV_STATES.get(int(nav.at(a)), "?") if nav is not None else "?"
            out.append(
                Finding(
                    f.log_id,
                    kind,
                    sev,
                    a,
                    b,
                    f"{text} (mode at onset: {mode})",
                    {"mode": mode},
                    ("vehicle_status/nav_state", f"vehicle_status/{fld}"),
                )
            )
    return out


def segments(f: Flight) -> list[Finding]:
    """One info finding per flight-mode segment while armed, with altitude, speed and battery stats."""
    nav = f.series("vehicle_status", "nav_state")
    if nav is None:
        return []
    armed = f.armed_at(nav.t)
    out: list[Finding] = []
    changes = np.flatnonzero(np.r_[True, np.diff(nav.v) != 0])
    bounds = [*nav.t[changes], f.duration]
    z = f.series("vehicle_local_position", "z")
    vx = f.series("vehicle_local_position", "vx")
    vy = f.series("vehicle_local_position", "vy")
    volt = f.series("battery_status", "voltage_v")
    for j, i in enumerate(changes):
        a, b = float(bounds[j]), float(bounds[j + 1])
        if b - a < 1.0 or not armed[i] and not f.armed_at(np.array([(a + b) / 2]))[0]:
            continue
        mode = NAV_STATES.get(int(nav.v[i]), f"nav_state {int(nav.v[i])}")
        ev: dict[str, float | str] = {"mode": mode, "duration_s": _r(b - a, 1)}
        parts = [f"{mode} mode for {b - a:.0f} s"]
        if z is not None and len(zw := z.window(a, b).v):
            ev["max_alt_m"] = _r(-zw.min(), 1)
            parts.append(f"altitude up to {-zw.min():.1f} m above start")
        if vx is not None and vy is not None and len(vw := vx.window(a, b).v):
            spd = np.hypot(vw, vy.at(vx.window(a, b).t))
            ev["max_speed_m_s"] = _r(spd.max(), 1)
            parts.append(f"max speed {spd.max():.1f} m/s")
        if volt is not None and len(bw := volt.window(a, b).v) and bw.max() > 1:
            ev["battery_v"] = f"{bw[0]:.1f}->{bw[-1]:.1f}"
            parts.append(f"battery {bw[0]:.1f}->{bw[-1]:.1f} V")
        out.append(
            Finding(f.log_id, "segment", "info", a, b, ", ".join(parts), ev, ("vehicle_local_position/z", "vehicle_status/nav_state"))
        )
    return out


def disarm_in_air(f: Flight) -> list[Finding]:
    """Disarm while the land detector still says airborne: kill switch, crash detection or a failsafe."""
    if f.armed is None:
        return []
    off = f.armed.t[np.r_[False, np.diff(f.armed.v) < 0]]
    landed = f.series("vehicle_land_detected", "landed")
    z = f.series("vehicle_local_position", "z")
    out: list[Finding] = []
    for t in off:
        if landed is not None and landed.at(t - 0.2) > 0.5:
            continue
        alt = float(-z.at(t)) if z is not None else float("nan")
        out.append(
            Finding(
                f.log_id,
                "disarm_in_air",
                "critical",
                float(t),
                float(t),
                "Motors disarmed while the vehicle was not detected as landed"
                + (f" ({alt:.1f} m above start)" if np.isfinite(alt) else ""),
                {"alt_m": _r(alt, 1)},
                ("vehicle_local_position/z", "vehicle_status/arming_state"),
            )
        )
    return out


def messages(f: Flight) -> list[Finding]:
    """PX4's own error-level log messages, deduplicated (repeats within 5 s are merged)."""
    out: list[Finding] = []
    last: dict[str, float] = {}
    for m in f.messages:
        if m.level not in ("EMERGENCY", "ALERT", "CRITICAL", "ERROR"):
            continue
        if m.text in last and m.t - last[m.text] < 5.0:
            last[m.text] = m.t
            continue
        last[m.text] = m.t
        out.append(
            Finding(
                f.log_id,
                "px4_message",
                "critical" if m.level in ("EMERGENCY", "ALERT", "CRITICAL") else "warning",
                m.t,
                m.t,
                f"PX4 logged {m.level}: {m.text}",
                {"level": m.level, "text": m.text},
            )
        )
    return out[:40]


def cpu(f: Flight) -> list[Finding]:
    """Flight controller CPU load above 90%."""
    s = f.series("cpuload", "load")
    if s is None:
        return []
    return [
        Finding(
            f.log_id,
            "cpu_overload",
            "warning",
            a,
            b,
            f"CPU load above 90% for {b - a:.0f} s",
            {"peak_load": _r(s.window(a, b).v.max())},
            ("cpuload/load",),
        )
        for a, b in spans(s.t, f.armed_at(s.t) & (s.v > 0.9), min_s=2.0)
    ]


DETECTORS: list[Detector] = [
    battery,
    motor_saturation,
    vibration,
    impact,
    attitude,
    gps,
    estimator,
    failsafe,
    disarm_in_air,
    messages,
    cpu,
    segments,
]


def scan(f: Flight) -> list[Finding]:
    """Run every detector and return findings in time order."""
    return sorted((x for d in DETECTORS for x in d(f)), key=lambda x: (x.t_start, x.kind))
