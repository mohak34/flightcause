"""volatus command line: download, scan, logs, findings, ask, eval, serve."""

import argparse
import os
import sys
from pathlib import Path

from volatus.answer import llm_from_env
from volatus.engine import Engine


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="volatus", description="Ask questions about PX4 flight logs.")
    p.add_argument("--data", type=Path, default=Path(os.environ.get("VOLATUS_DATA", "data")), help="data directory (default ./data)")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download a labeled crash/normal sample from Flight Review")
    d.add_argument("-n", type=int, default=150, help="logs per class")
    d.add_argument("--delay", type=float, default=2.0, help="seconds between downloads")
    s = sub.add_parser("scan", help="run detectors and (re)build the index")
    s.add_argument("paths", nargs="*", type=Path, help="specific .ulg files (default: all in data/logs)")
    sub.add_parser("logs", help="list indexed logs")
    f = sub.add_parser("findings", help="timeline of findings for one log")
    f.add_argument("log", help="log id or prefix")
    f.add_argument("--all", action="store_true", help="include info-level mode segments")
    a = sub.add_parser("ask", help="answer a question with cited findings and plots")
    a.add_argument("question")
    a.add_argument("--log", help="restrict to one log (id or prefix)")
    a.add_argument("-k", type=int, default=12, help="findings to retrieve")
    a.add_argument("--no-plots", action="store_true")
    sub.add_parser("eval", help="score detectors against Flight Review crash ratings")
    v = sub.add_parser("serve", help="run the HTTP API")
    v.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    engine = Engine(args.data)

    match args.cmd:
        case "download":
            from volatus.download import download, fetch_dbinfo, select

            download(select(fetch_dbinfo(), args.n), args.data, args.delay)
        case "scan":
            for log_id, result in engine.scan(args.paths or None):
                print(f"{log_id}  {'error: ' + str(result) if isinstance(result, Exception) else str(result) + ' findings'}")
        case "logs":
            for r in engine.index.logs():
                print(f"{r['log_id']}  {r['rating'] or '-':12} {r['n_critical'] or 0:3} critical  {r['description'][:50]}")
        case "findings":
            log_id = engine.resolve(args.log)
            if log_id is None:
                print(f"unknown log {args.log}; run `volatus scan` first", file=sys.stderr)
                return 1
            for x in engine.index.findings(log_id):
                if args.all or x.severity != "info":
                    print(f"{x.when:>15}  {x.severity:8} {x.kind:17} {x.summary}")
        case "ask":
            log_id = None
            if args.log and (log_id := engine.resolve(args.log)) is None:
                print(f"unknown log {args.log}", file=sys.stderr)
                return 1
            llm = llm_from_env()
            result, figures = engine.ask(args.question, log_id, llm, args.k, not args.no_plots)
            if not result.hits:
                print("No matching findings.")
                return 0
            print("Findings:")
            for i, h in enumerate(result.hits, 1):
                x = h.finding
                print(f"[F{i}] {x.log_id[:8]} {x.when:>13} {x.severity:8} {x.kind:17} {x.summary}")
                if i in figures:
                    print(f"      plot: {figures[i]}")
            print()
            if result.text is not None:
                print(result.text)
                if result.dropped_citations:
                    print(f"\n(removed citations to findings that were not retrieved: {', '.join(result.dropped_citations)})")
            else:
                print("No LLM configured (set VOLATUS_LLM_API_KEY). The findings above are the retrieved evidence.")
        case "eval":
            from volatus.evaluate import report

            print(report(engine.index))
        case "serve":
            import uvicorn

            from volatus.api import create_app

            uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port)
    return 0
