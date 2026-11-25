"""Detectors on snippets cut from real Flight Review logs (see make_fixtures.py)."""

from conftest import CRASH, NORMAL

from flightcause.detectors import Finding
from flightcause.index import CRASH_KINDS, first_crash


def kinds(fs: list[Finding], severity: str | None = None) -> set[str]:
    return {f.kind for f in fs if severity is None or f.severity == severity}


def test_prop_failure_shows_vibration_before_the_impact(findings: dict[str, list[Finding]]) -> None:
    fs = findings[CRASH]
    crash_t = first_crash(fs)
    assert crash_t is not None and 13.5 < crash_t < 14.5  # 25 g spike at 0:14.0
    vib = [f for f in fs if f.kind == "vibration"]
    assert vib and vib[0].severity == "critical"
    assert vib[0].t_start < crash_t - 5  # the warning sign came well before the crash
    assert float(vib[0].evidence["peak_m_s2"]) > 20


def test_crash_aftermath_is_detected(findings: dict[str, list[Finding]]) -> None:
    fs = findings[CRASH]
    assert {"impact", "flip", "attitude_error"} <= kinds(fs, "critical")
    assert "motor_saturation" in kinds(fs)
    flip = next(f for f in fs if f.kind == "flip")
    assert float(flip.evidence["peak_tilt_deg"]) > 90


def test_battery_warning_reported_once_per_level(findings: dict[str, list[Finding]]) -> None:
    warns = [f for f in findings[CRASH] if f.kind == "battery_warning"]
    levels = [f.evidence["warning_level"] for f in warns]
    assert len(levels) == len(set(levels))


def test_normal_flight_has_no_crash_findings(findings: dict[str, list[Finding]]) -> None:
    fs = findings[NORMAL]
    assert not kinds(fs) & CRASH_KINDS
    assert "critical" not in {f.severity for f in fs}
    assert any(f.kind == "segment" for f in fs)  # mode segments are still indexed


def test_findings_are_time_ordered_with_evidence(findings: dict[str, list[Finding]]) -> None:
    for fs in findings.values():
        assert [f.t_start for f in fs] == sorted(f.t_start for f in fs)
        assert all(f.t_end >= f.t_start for f in fs)
        assert all(f.evidence for f in fs if f.severity == "critical")
