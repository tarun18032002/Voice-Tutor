import asyncio
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

# Ensure Voice-Tutor's src/ is importable regardless of CWD
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv

load_dotenv(_HERE / ".env.local")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

import uvicorn

app = FastAPI(title="Voice Tutor UI")

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

_UI_HTML = Path(__file__).parent / "ui" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_UI_HTML.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/api/token")
async def get_token(room: str = "voice-tutor"):
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured in .env.local")

    try:
        from livekit.api import AccessToken, VideoGrants

        identity = f"student-{uuid.uuid4().hex[:8]}"
        token = (
            AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name("Student")
            .with_grants(VideoGrants(room_join=True, room=room))
        )
        return {
            "token": token.to_jwt(),
            "url": LIVEKIT_URL,
            "room": room,
            "identity": identity,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        def _run_ingestion():
            from src.injestion.injestion_pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            pipeline.ingest(str(tmp_path))

        await asyncio.to_thread(_run_ingestion)
        return {"ok": True, "message": f"Successfully ingested {file.filename}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "livekit_url": LIVEKIT_URL or "(not set)",
        "api_key_set": bool(LIVEKIT_API_KEY),
    }


if __name__ == "__main__":
    uvicorn.run("ui_server:app", host="0.0.0.0", port=8080, reload=True)
