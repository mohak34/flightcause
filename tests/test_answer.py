from conftest import CRASH

from flightcause.answer import answer
from flightcause.index import Index


def test_answer_keeps_only_citations_of_retrieved_findings(index: Index) -> None:
    hits = index.search("why did this drone crash?", CRASH)
    prompts: list[str] = []

    def fake_llm(system: str, user: str) -> str:
        prompts.append(user)
        return "Vibration started first [F1], then it hit the ground [F2]. Also a motor burned out [F99]."

    result = answer("why did this drone crash?", hits, fake_llm)
    assert result.text is not None
    assert "[F99]" not in result.text
    assert result.dropped_citations == ["F99"]
    assert result.cited == [1, 2]
    # The model sees exactly the retrieved findings, numbered in order, with their timestamps.
    for i, h in enumerate(hits, 1):
        assert f"[F{i}] log {h.finding.log_id[:8]} at {h.finding.when}" in prompts[0]
    assert f"[F{len(hits) + 1}]" not in prompts[0]


def test_without_llm_the_findings_are_still_returned(index: Index) -> None:
    hits = index.search("why did this drone crash?", CRASH)
    result = answer("why did this drone crash?", hits, None)
    assert result.text is None and result.hits == hits
