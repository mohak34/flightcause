"""SQLite store of logs and findings, with FTS5 (BM25) retrieval over finding text.

Each finding is indexed with its summary plus a fixed keyword list for its kind, so plain
questions ("was vibration a problem?", "why did it crash?") hit the right moments.
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from volatus.detectors import Finding

KEYWORDS = {
    "battery_low": "battery voltage low empty depleted power cell",
    "battery_sag": "battery voltage sag sagging drop brownout power load current",
    "battery_warning": "battery warning low critical power",
    "motor_saturation": "motor saturation saturated thrust esc actuator authority output max",
    "vibration": "vibration vibrations vibe shaking imu accelerometer noise props",
    "accel_clipping": "vibration accelerometer clipping imu range",
    "impact": "crash crashed impact hit ground collision accident",
    "freefall": "crash fall falling freefall dropped drop",
    "flip": "crash flip flipped tilt tumble loss control upside",
    "attitude_error": "attitude tracking error tilt roll pitch control oscillation unstable instability",
    "gps_loss": "gps fix lost satellites position navigation",
    "gps_degraded": "gps accuracy degraded satellites position eph",
    "estimator": "ekf estimator innovation navigation position height velocity heading compass mag",
    "failsafe": "failsafe triggered rtl return emergency",
    "rc_loss": "rc radio signal lost failsafe link remote",
    "datalink_loss": "datalink telemetry link lost ground station gcs",
    "disarm_in_air": "disarm disarmed kill switch motors off air crash",
    "px4_message": "error message log px4",
    "cpu_overload": "cpu load overload processor",
    "segment": "mode flight segment altitude speed timeline",
}
CRASH_KINDS = {"impact", "freefall", "flip", "disarm_in_air"}
SEVERITY_RANK = {"critical": 3, "warning": 2, "info": 1}
STOPWORDS = set(
    """a an and are at be before by did do does drone flight flights for from had has have how i in is it its log logs me
    of on or show that the there this to was were what when where which while why with all any vehicle""".split()
)

SCHEMA = """
create table if not exists logs (
  log_id text primary key, rating text, crashed integer, description text, feedback text,
  duration_s real, mav_type text, log_date text, crash_t real
);
create table if not exists findings (
  id text primary key, log_id text, kind text, severity text, t_start real, t_end real,
  summary text, evidence text, signals text
);
create virtual table if not exists findings_fts using fts5(id unindexed, log_id unindexed, text);
"""


@dataclass(frozen=True)
class Hit:
    finding: Finding
    score: float
    crash_t: float | None  # first detected crash moment in that log, if any

    @property
    def before_crash_s(self) -> float | None:
        return None if self.crash_t is None else self.crash_t - self.finding.t_start


def first_crash(findings: list[Finding]) -> float | None:
    ts = [f.t_start for f in findings if f.kind in CRASH_KINDS]
    return min(ts) if ts else None


class Index:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)

    def add(self, log_id: str, meta: dict[str, Any], findings: list[Finding]) -> None:
        """Replace everything stored for one log."""
        rating = meta.get("rating") or ""
        with self.db:
            self.db.execute("delete from findings where log_id = ?", (log_id,))
            self.db.execute("delete from findings_fts where log_id = ?", (log_id,))
            self.db.execute(
                "insert or replace into logs values (?,?,?,?,?,?,?,?,?)",
                (
                    log_id,
                    rating,
                    int(rating.startswith("crash")),
                    meta.get("description", ""),
                    meta.get("feedback", ""),
                    meta.get("duration_s"),
                    meta.get("mav_type", ""),
                    meta.get("log_date", ""),
                    first_crash(findings),
                ),
            )
            for f in findings:
                self.db.execute(
                    "insert or replace into findings values (?,?,?,?,?,?,?,?,?)",
                    (f.id, log_id, f.kind, f.severity, f.t_start, f.t_end, f.summary, json.dumps(f.evidence), json.dumps(f.signals)),
                )
                text = f"{f.kind.replace('_', ' ')} {f.severity} {f.summary} {KEYWORDS.get(f.kind, '')}"
                self.db.execute("insert into findings_fts values (?,?,?)", (f.id, log_id, text))

    def logs(self) -> list[dict[str, Any]]:
        cur = self.db.execute(
            "select l.*, count(f.id) as n_findings, sum(f.severity = 'critical') as n_critical "
            "from logs l left join findings f on f.log_id = l.log_id group by l.log_id order by l.log_date desc"
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur]

    def log(self, log_id: str) -> dict[str, Any] | None:
        cur = self.db.execute("select * from logs where log_id = ? or log_id like ?", (log_id, f"{log_id}%"))
        row = cur.fetchone()
        return None if row is None else dict(zip([c[0] for c in cur.description], row))

    def findings(self, log_id: str) -> list[Finding]:
        rows = self.db.execute("select * from findings where log_id = ? order by t_start", (log_id,))
        return [_finding(r) for r in rows]

    def search(self, question: str, log_id: str | None = None, k: int = 12) -> list[Hit]:
        """Rank findings for a question. Scoped to one log when log_id is given.

        BM25 relevance is weighted by severity. "before (a|the) crash" keeps only findings that
        precede the first detected crash moment of their log.
        """
        terms = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOPWORDS]
        before_crash = bool(re.search(r"\bbefore (a |the )?(crash|impact)", question.lower()))
        if before_crash:
            terms = [w for w in terms if w not in ("crash", "crashed", "impact")]
        crash_t = {r[0]: r[1] for r in self.db.execute("select log_id, crash_t from logs")}

        relevance: dict[str, float] = {}
        if terms:
            match = " OR ".join(f'"{w}"' for w in terms)
            sql = "select id, bm25(findings_fts) from findings_fts where findings_fts match ?"
            sql += " and log_id = ?" if log_id else ""
            relevance = {r[0]: -float(r[1]) for r in self.db.execute(sql, (match, log_id) if log_id else (match,))}
        if log_id:
            # In one log every notable finding is a candidate; matching ones rank first.
            rows = self.db.execute(
                "select * from findings where log_id = ? and (severity != 'info' or id in (select value from json_each(?)))",
                (log_id, json.dumps(list(relevance))),
            )
        else:
            rows = self.db.execute("select * from findings where id in (select value from json_each(?))", (json.dumps(list(relevance)),))

        hits = []
        for r in rows:
            f = _finding(r)
            sev = SEVERITY_RANK[f.severity]
            hit = Hit(f, relevance.get(f.id, 0.0) * sev + sev, crash_t.get(f.log_id))
            if before_crash and (hit.before_crash_s is None or hit.before_crash_s <= 0 or f.kind in CRASH_KINDS):
                continue
            if hit.before_crash_s is not None and hit.before_crash_s >= -1.0:
                hit = Hit(f, hit.score + 1.0, hit.crash_t)  # what led up to a crash outranks its aftermath
            hits.append(hit)
        hits.sort(key=lambda h: (-h.score, h.finding.t_start))
        if log_id is None:
            # Across logs, keep the best two per log so one noisy log cannot fill the answer.
            per_log: dict[str, int] = {}
            kept = []
            for h in hits:
                per_log[h.finding.log_id] = per_log.get(h.finding.log_id, 0) + 1
                if per_log[h.finding.log_id] <= 2:
                    kept.append(h)
            hits = kept
        return sorted(hits[:k], key=lambda h: (h.finding.log_id, h.finding.t_start))


def _finding(r: tuple[Any, ...]) -> Finding:
    return Finding(r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7]), tuple(json.loads(r[8])))
