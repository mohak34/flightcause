"""Wires detectors, index, plots and the answer step to a data directory.

Layout: <data>/logs/*.ulg, <data>/manifest.json (Flight Review metadata), <data>/flightcause.db, <data>/plots/.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flightcause.answer import LLM, Answer, answer
from flightcause.detectors import Finding, scan
from flightcause.flight import Flight
from flightcause.index import Index
from flightcause.plots import plot_finding

MAX_PLOTS = 6


class Engine:
    def __init__(self, data_dir: Path) -> None:
        self.data = data_dir
        self.logs_dir = data_dir / "logs"
        self.plots_dir = data_dir / "plots"
        self.index = Index(data_dir / "flightcause.db")

    def manifest(self) -> dict[str, dict[str, Any]]:
        p = self.data / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def scan(self, paths: list[Path] | None = None) -> Iterator[tuple[str, int | Exception]]:
        """Run detectors on logs and store the findings. Yields (log_id, n_findings or the error)."""
        meta = self.manifest()
        for path in paths or sorted(self.logs_dir.glob("*.ulg")):
            try:
                findings = scan(Flight(path))
            except Exception as e:  # a corrupt log must not stop a batch scan
                yield path.stem, e
                continue
            self.index.add(path.stem, meta.get(path.stem, {}), findings)
            yield path.stem, len(findings)

    def resolve(self, log_id: str) -> str | None:
        """Full log id from a prefix, as printed in citations."""
        row = self.index.log(log_id)
        return None if row is None else str(row["log_id"])

    def plot(self, finding: Finding, flight: Flight | None = None) -> Path | None:
        path = self.plots_dir / finding.log_id[:8] / f"{finding.id}.png"
        if path.exists():
            return path
        return plot_finding(flight or Flight(self.logs_dir / f"{finding.log_id}.ulg"), finding, path.parent)

    def ask(self, question: str, log_id: str | None, llm: LLM | None, k: int = 12, plots: bool = True) -> tuple[Answer, dict[int, Path]]:
        """Retrieve, answer, and plot the cited findings (or the top retrieved ones without an LLM)."""
        hits = self.index.search(question, log_id, k)
        result = answer(question, hits, llm)
        figures: dict[int, Path] = {}
        if plots:
            wanted = result.cited or [i for i, h in enumerate(hits, 1) if h.finding.severity != "info"]
            flights: dict[str, Flight] = {}
            for i in wanted[:MAX_PLOTS]:
                f = hits[i - 1].finding
                if f.log_id not in flights and (self.logs_dir / f"{f.log_id}.ulg").exists():
                    flights[f.log_id] = Flight(self.logs_dir / f"{f.log_id}.ulg")
                if f.log_id in flights and (p := self.plot(f, flights[f.log_id])) is not None:
                    figures[i] = p
        return result, figures
