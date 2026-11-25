# volatus

Ask questions about real PX4 drone flight logs: "why did this drone crash?", "when did the battery start sagging?", "show me all flights with motor saturation before a crash". Every answer cites timestamped findings and comes with telemetry plots of those moments.

Deterministic detectors do the analysis. An LLM only writes the final sentences, and only from the findings it was given.

## Setup

```sh
uv sync
uv run volatus download -n 150   # 150 crash + 150 normal multirotor logs from Flight Review (about 2.7 GB)
uv run volatus scan              # run detectors, build data/volatus.db
```

Logs come from [PX4 Flight Review](https://review.px4.io) (public uploads, CC-BY). The download waits 2 s between files and skips files over 80 MB.

The LLM step is optional. Any OpenAI-compatible endpoint works:

```sh
# Groq (default base URL and model)
export VOLATUS_LLM_API_KEY=...
# Gemini
export VOLATUS_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export VOLATUS_LLM_MODEL=gemini-3-flash-preview
```

Without a key, everything except the written answer still works: findings, retrieval, plots, the API.

## Commands

```sh
uv run volatus logs                                   # indexed logs with critical counts
uv run volatus findings 5e002593                      # timeline of one log (--all adds mode segments)
uv run volatus ask "why did this drone crash?" --log 5e002593
uv run volatus ask "show me all flights with motor saturation before a crash"
uv run volatus eval                                   # the tables below
uv run volatus serve --port 8000                      # HTTP API on 127.0.0.1
```

API: `GET /logs`, `GET /logs/{id}`, `GET /search?q=&log_id=&k=`, `POST /ask {question, log_id?, k?}`, `GET /plots/{log}/{file}`.

Tests use two small real-log fixtures in `tests/fixtures` and need no key or network: `uv run pytest`.

## How it works

1. **Read** (`flight.py`): pyulog loads the topics the detectors need into time series (seconds since log start).
2. **Detect** (`detectors.py`): each detector returns findings with a kind, severity, time window, summary and evidence values.
   - Battery: low voltage per cell, sag under load, PX4 battery warnings.
   - Motors: an output pinned at max (and another at min) on the motor channels only.
   - Vibration: the `vehicle_imu_status` metric with Flight Review's bands on PX4 1.13+, else high-passed accel RMS. Also accel clipping.
   - Crash signs: impact (50 ms mean accel above 5 g), free fall, flip (tilt above 60 deg), disarm in the air.
   - Control and navigation: attitude error vs setpoint, GPS fix and accuracy, EKF innovation test ratios and filter faults.
   - Events: failsafe, RC loss, data link loss, PX4 error messages, CPU overload, and one segment per flight mode.
3. **Index** (`index.py`): findings go into SQLite with an FTS5 table. Each row's text is its summary plus fixed keywords for its kind, so "was vibration a problem?" matches vibration findings. Ranking is BM25 times severity. Findings at or before a log's first crash moment outrank the aftermath. "before a crash" keeps only findings that precede it. Across logs, at most 2 hits per log.
4. **Answer** (`answer.py`): the hits go to the model as numbered lines `[F1] log 5e002593 at 0:04.1-0:19.1 (critical, vibration; 9.9 s before the first detected crash moment) ...`. Citations to ids that were not retrieved are stripped from the reply.
5. **Plot** (`plots.py`): one PNG per cited finding, the finding's signals 10 s either side, its window shaded.

## Evaluation

300 multirotor logs from Flight Review, labeled by their uploaders: 150 rated as crashes (`crash_sw_hw`, `crash_pilot`) and 150 rated good or great. A log counts as flagged if the rule fires anywhere in it.

| Rule (log flagged if...) | Precision | Recall | Accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| any crash finding (impact, free fall, flip, disarm in air) | 0.82 | 0.65 | 0.75 | 97 | 22 | 53 | 128 |
| any critical finding | 0.73 | 0.77 | 0.74 | 115 | 43 | 35 | 107 |

How often each detector fires per log:

| Finding kind | Crash logs | Normal logs |
|---|---|---|
| vibration | 63% | 23% |
| attitude_error | 62% | 23% |
| motor_saturation | 53% | 23% |
| impact | 52% | 11% |
| flip | 46% | 3% |
| estimator | 45% | 14% |
| battery_sag | 44% | 42% |
| px4_message | 41% | 34% |
| datalink_loss | 36% | 11% |
| disarm_in_air | 27% | 3% |
| failsafe | 27% | 14% |
| accel_clipping | 20% | 1% |
| battery_low | 19% | 14% |
| rc_loss | 15% | 15% |
| battery_warning | 11% | 5% |
| freefall | 5% | 1% |

Reading the numbers:

- Thresholds were tuned on this same set, so these are in-sample numbers, not a held-out score.
- Ratings are user labels. Some "crash" uploads end before the impact is logged, and some "good" flights include hard landings.
- 53 of 150 crash logs show no crash sign at all. These are the detectors' misses; the per-kind table shows what still fires in them.
- Battery sag, PX4 messages and RC loss fire as often in good flights as in crashes. They are context for an answer, not crash predictors, so they are warnings, not critical.

## Design

`design/dashboard.html` is a clickable mock of the dashboard. Open it in a browser. It shows 92 crash logs, but Ask and Upload are scripted.
