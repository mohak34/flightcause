from pathlib import Path

import pytest

from flightcause.detectors import Finding, scan
from flightcause.flight import Flight
from flightcause.index import Index

FIXTURES = Path(__file__).parent / "fixtures"
CRASH = "crash_prop_failure"
NORMAL = "normal_position_flight"


@pytest.fixture(scope="session")
def findings() -> dict[str, list[Finding]]:
    return {name: scan(Flight(FIXTURES / f"{name}.ulg")) for name in (CRASH, NORMAL)}


@pytest.fixture
def index(tmp_path: Path, findings: dict[str, list[Finding]]) -> Index:
    idx = Index(tmp_path / "test.db")
    idx.add(CRASH, {"rating": "crash_sw_hw"}, findings[CRASH])
    idx.add(NORMAL, {"rating": "good"}, findings[NORMAL])
    return idx
