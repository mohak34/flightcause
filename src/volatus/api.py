"""HTTP API over the same engine the CLI uses."""

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from volatus.answer import llm_from_env
from volatus.engine import Engine


class AskRequest(BaseModel):
    question: str
    log_id: str | None = None
    k: int = 12


def _finding(h: Any) -> dict[str, Any]:
    f = h.finding
    return {**asdict(f), "id": f.id, "when": f.when, "before_crash_s": h.before_crash_s}


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="volatus")

    def resolve(log_id: str) -> str:
        full = engine.resolve(log_id)
        if full is None:
            raise HTTPException(404, f"unknown log {log_id}")
        return full

    @app.get("/logs")
    def logs() -> list[dict[str, Any]]:
        return engine.index.logs()

    @app.get("/logs/{log_id}")
    def log(log_id: str) -> dict[str, Any]:
        full = resolve(log_id)
        return {
            **(engine.index.log(full) or {}),
            "findings": [{**asdict(f), "id": f.id, "when": f.when} for f in engine.index.findings(full)],
        }

    @app.get("/search")
    def search(q: str, log_id: str | None = None, k: int = 12) -> list[dict[str, Any]]:
        return [_finding(h) for h in engine.index.search(q, resolve(log_id) if log_id else None, k)]

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        result, figures = engine.ask(req.question, resolve(req.log_id) if req.log_id else None, llm_from_env(), req.k)
        return {
            "question": result.question,
            "answer": result.text,
            "llm": result.text is not None,
            "cited": [f"F{i}" for i in result.cited],
            "dropped_citations": result.dropped_citations,
            "findings": [
                {"ref": f"F{i}", **_finding(h), "plot": f"/plots/{figures[i].parent.name}/{figures[i].name}" if i in figures else None}
                for i, h in enumerate(result.hits, 1)
            ],
        }

    @app.get("/plots/{log_prefix}/{name}")
    def plot(log_prefix: str, name: str) -> FileResponse:
        path = (engine.plots_dir / log_prefix / name).resolve()
        if not path.is_relative_to(engine.plots_dir.resolve()) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/png")

    return app
