from conftest import CRASH, NORMAL

from flightcause.index import CRASH_KINDS, Index


def test_question_about_one_log_pulls_the_matching_moment(index: Index) -> None:
    hits = index.search("was vibration a problem?", CRASH)
    top = max(hits, key=lambda h: h.score)
    assert top.finding.kind == "vibration"
    assert all(h.finding.log_id == CRASH for h in hits)


def test_why_crash_returns_crash_and_what_preceded_it(index: Index) -> None:
    hits = index.search("why did this drone crash?", CRASH)
    ks = {h.finding.kind for h in hits}
    assert ks & CRASH_KINDS
    assert "vibration" in ks
    assert all(h.finding.severity != "info" for h in hits)


def test_battery_question_ranks_battery_findings_first(index: Index) -> None:
    hits = sorted(index.search("when did the battery start sagging?", CRASH), key=lambda h: -h.score)
    assert hits[0].finding.kind.startswith("battery")


def test_cross_log_before_crash_keeps_only_precursors_in_crashed_logs(index: Index) -> None:
    hits = index.search("show me all flights with vibration before a crash")
    assert hits
    for h in hits:
        assert h.finding.log_id == CRASH  # the normal flight never crashed
        assert h.before_crash_s is not None and h.before_crash_s > 0
        assert h.finding.kind not in CRASH_KINDS


def test_mode_question_finds_segments_in_the_normal_flight(index: Index) -> None:
    hits = index.search("what flight mode and altitude", NORMAL)
    assert any(h.finding.kind == "segment" for h in hits)
