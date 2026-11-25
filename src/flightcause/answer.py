"""The only LLM step: write an answer from retrieved findings, citing them as [F1], [F2], ...

Any OpenAI-compatible chat endpoint works. Configure with environment variables:
FLIGHTCAUSE_LLM_BASE_URL, FLIGHTCAUSE_LLM_API_KEY, FLIGHTCAUSE_LLM_MODEL.
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from flightcause.index import Hit

# (system prompt, user prompt) -> completion text
LLM = Callable[[str, str], str]

SYSTEM = """You analyze PX4 drone flight logs. You get a question and numbered findings that deterministic
detectors extracted from the logs. Answer using only those findings.
Rules:
- Cite every factual claim with the finding ids in square brackets, e.g. [F2] or [F1][F4].
- Only cite ids that appear in the findings list. Never invent timestamps, values or causes.
- Times are m:ss since log start. Say when one event preceded another.
- If the findings do not answer the question, say so plainly.
- Be short: a few sentences or a short list."""


@dataclass
class Answer:
    question: str
    hits: list[Hit]
    text: str | None = None  # None when no LLM is configured
    cited: list[int] = field(default_factory=list)  # 1-based indexes into hits
    dropped_citations: list[str] = field(default_factory=list)  # ids the model made up, removed from text


def context(hits: list[Hit]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        f = h.finding
        rel = ""
        if h.before_crash_s is not None:
            d = h.before_crash_s
            rel = (
                f"; {d:.1f} s before the first detected crash moment"
                if d > 0.05
                else (f"; {-d:.1f} s after the first detected crash moment" if d < -0.05 else "; at the first detected crash moment")
            )
        ev = ", ".join(f"{k}={v}" for k, v in f.evidence.items())
        lines.append(f"[F{i}] log {f.log_id[:8]} at {f.when} ({f.severity}, {f.kind}{rel}): {f.summary}. Evidence: {ev or 'none'}")
    return "\n".join(lines)


def answer(question: str, hits: list[Hit], llm: LLM | None) -> Answer:
    """Ask the LLM to answer from hits, then keep only citations that point at a retrieved finding."""
    result = Answer(question, hits)
    if llm is None or not hits:
        return result
    raw = llm(SYSTEM, f"Question: {question}\n\nFindings:\n{context(hits)}")
    valid = {f"F{i}" for i in range(1, len(hits) + 1)}

    def check(m: re.Match[str]) -> str:
        if m.group(1) in valid:
            return m.group(0)
        result.dropped_citations.append(m.group(1))
        return ""

    result.text = re.sub(r"\[(F\d+)\]", check, raw).strip()
    result.cited = sorted({int(x[1:]) for x in re.findall(r"\[(F\d+)\]", result.text)})
    return result


def llm_from_env() -> LLM | None:
    """An OpenAI-compatible chat client, or None when FLIGHTCAUSE_LLM_API_KEY is not set."""
    key = os.environ.get("FLIGHTCAUSE_LLM_API_KEY")
    base = os.environ.get("FLIGHTCAUSE_LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.environ.get("FLIGHTCAUSE_LLM_MODEL", "openai/gpt-oss-20b")
    if not key:
        return None

    def call(system: str, user: str) -> str:
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    return call
