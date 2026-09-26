"""FastAPI service. This is the container the cohere-toolkit talks to.

POST /read   multipart: image=<file>, target_language=<bcp47>   -> DocumentReading JSON
GET  /health
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .backends import make_backend
from .imaging import MAX_PIXELS
from .pipeline import ReadForMe

app = FastAPI(title="ReadForMe", version="0.1.0")


@lru_cache(maxsize=1)
def _engine() -> ReadForMe:
    backend = make_backend(os.environ.get("READFORME_BACKEND"))
    max_pixels = int(os.environ.get("READFORME_MAX_PIXELS", MAX_PIXELS))
    return ReadForMe(backend, max_pixels=max_pixels, explain=os.environ.get("READFORME_EXPLAIN", "0") == "1")


@app.get("/health")
def health():
    return {"ok": True, "backend": os.environ.get("READFORME_BACKEND", "openai")}


@app.post("/read")
async def read(image: UploadFile = File(...), target_language: str = Form("en")):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(415, "image/* expected")
    data = await image.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "image too large")
    try:
        reading, timings = _engine().read(data, target_language=target_language)
    except Exception as e:  # surface model/backend errors to the toolkit tool
        raise HTTPException(502, f"reading failed: {e}") from e
    out = reading.model_dump()
    out["timings"] = timings.__dict__
    return out
