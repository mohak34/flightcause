"""Score the detectors against Flight Review pilot ratings: crash_* vs good/great flights."""

from dataclasses import dataclass

from flightcause.index import CRASH_KINDS, Index


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)


def _confusion(labels: dict[str, bool], predicted: set[str]) -> Confusion:
    tp = sum(1 for k, y in labels.items() if y and k in predicted)
    fp = sum(1 for k, y in labels.items() if not y and k in predicted)
    fn = sum(1 for k, y in labels.items() if y and k not in predicted)
    return Confusion(tp, fp, fn, len(labels) - tp - fp - fn)


def report(index: Index) -> str:
    """Markdown tables: log-level crash detection and how often each detector fires per class."""
    labels = {
        r[0]: r[1].startswith("crash")
        for r in index.db.execute("select log_id, rating from logs where rating in ('crash_sw_hw', 'crash_pilot', 'good', 'great')")
    }
    if not labels:
        return "No rated logs indexed. Run `flightcause download` and `flightcause scan` first."
    rows = list(index.db.execute("select log_id, kind, severity from findings where severity != 'info'"))
    crash_pred = {r[0] for r in rows if r[1] in CRASH_KINDS}
    critical_pred = {r[0] for r in rows if r[2] == "critical"}
    n_crash = sum(labels.values())
    n_ok = len(labels) - n_crash

    out = [f"{len(labels)} rated logs: {n_crash} rated as crashes, {n_ok} rated good/great.", ""]
    out += ["| Rule (log flagged if...) | Precision | Recall | Accuracy | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|"]
    for name, pred in (("any crash finding (impact, free fall, flip, disarm in air)", crash_pred), ("any critical finding", critical_pred)):
        c = _confusion(labels, pred)
        out.append(f"| {name} | {c.precision:.2f} | {c.recall:.2f} | {c.accuracy:.2f} | {c.tp} | {c.fp} | {c.fn} | {c.tn} |")

    out += ["", "| Finding kind | Crash logs | Normal logs |", "|---|---|---|"]
    kinds = sorted({r[1] for r in rows})
    for kind in kinds:
        logs = {r[0] for r in rows if r[1] == kind}
        c = sum(1 for k in logs if labels.get(k) is True)
        o = sum(1 for k in logs if labels.get(k) is False)
        out.append(f"| {kind} | {c / n_crash:.0%} ({c}) | {o / n_ok:.0%} ({o}) |" if n_crash and n_ok else f"| {kind} | {c} | {o} |")
    return "\n".join(out)
