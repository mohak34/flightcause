"""Pick a bounded, labeled set of public PX4 logs from Flight Review and download them.

Mirrors the filters of PX4's flight_review/app/download_logs.py, but picks a balanced
crash vs normal sample so the set doubles as an evaluation dataset.
"""

import gzip
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

DBINFO_URL = "https://review.px4.io/dbinfo"
MULTIROTORS = {"Quadrotor", "Hexarotor", "Octorotor"}
CRASH_RATINGS = {"crash_sw_hw", "crash_pilot"}
NORMAL_RATINGS = {"good", "great"}
MAX_BYTES = 80_000_000


def fetch_dbinfo() -> list[dict[str, Any]]:
    """All public log entries. The CDN serves the JSON gzipped without a Content-Encoding header."""
    raw = httpx.get(DBINFO_URL, follow_redirects=True, timeout=300).content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _eligible(e: dict[str, Any], ratings: set[str], min_s: int) -> bool:
    return (
        e["rating"] in ratings
        and e["mav_type"] in MULTIROTORS
        and e.get("source") == "webui"
        and e["log_date"] >= "2019"
        and min_s <= e["duration_s"] <= 1800
        and e["sys_hw"] != "PX4_SITL"
    )


def select(entries: list[dict[str, Any]], n_per_class: int, seed: int = 0) -> list[dict[str, Any]]:
    """Real-hardware multirotor flights since 2019: up to n crashes and n normal flights."""
    rng = random.Random(seed)
    crashes = [e for e in entries if _eligible(e, CRASH_RATINGS, 20)]
    normal = [e for e in entries if _eligible(e, NORMAL_RATINGS, 60)]
    picked = rng.sample(crashes, min(n_per_class, len(crashes)))
    picked += rng.sample(normal, min(n_per_class, len(normal)))
    return picked


def download(entries: list[dict[str, Any]], data_dir: Path, delay_s: float = 2.0) -> None:
    """Download each log to data_dir/logs and record its metadata in data_dir/manifest.json."""
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with httpx.Client(follow_redirects=True, timeout=600) as client:
        for i, e in enumerate(entries, 1):
            log_id = e["log_id"]
            path = logs / f"{log_id}.ulg"
            if log_id in manifest:
                continue
            url = e.get("download_url") or f"https://review.px4.io/download?log={log_id}"
            size = int(client.head(url).headers.get("content-length", 0))
            if size > MAX_BYTES:
                print(f"[{i}/{len(entries)}] skip {log_id}: {size / 1e6:.0f} MB")
                continue
            r = client.get(url)
            if r.status_code != 200:
                print(f"[{i}/{len(entries)}] fail {log_id}: HTTP {r.status_code}")
                continue
            path.write_bytes(r.content)
            manifest[log_id] = {
                k: e.get(k)
                for k in (
                    "rating",
                    "log_date",
                    "description",
                    "feedback",
                    "mav_type",
                    "duration_s",
                    "sys_hw",
                    "ver_sw_release",
                    "error_labels",
                )
            }
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(f"[{i}/{len(entries)}] {log_id} {e['rating']} {len(r.content) / 1e6:.1f} MB")
            time.sleep(delay_s)
