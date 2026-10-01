"""FastAPI app: analyze API + static frontend."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.combiner import combine
from backend.ml.predict import artifacts_available, load_metrics
from backend.samples import all_samples
from backend.schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger("codeguard")
ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

app = FastAPI(title="CodeGuard AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    ready = artifacts_available()
    return {
        "status": "ok",
        "model_ready": ready,
        "model_status": "Model ready" if ready else "Model not trained",
    }


@app.get("/api/metrics")
def metrics() -> dict:
    data = load_metrics()
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run python -m backend.ml.train",
        )
    return data


@app.get("/api/samples")
def samples() -> list[dict]:
    return all_samples()


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    if body.code is None:
        raise HTTPException(status_code=422, detail="code is required")
    t0 = time.perf_counter()
    try:
        result = combine(body.code, filename=body.filename)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze failed")
        raise HTTPException(status_code=400, detail=f"Could not analyze input: {exc}") from exc
    result["stats"]["analysis_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return AnalyzeResponse.model_validate(result)


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(index_path)


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
