"""
Local bridge server — run this alongside your LiveKit worker.

Usage:
  1. Start your LiveKit worker:   python main.py dev
  2. Start this server:           python server.py
  3. Open http://localhost:8000   in your browser

Requires: pip install fastapi uvicorn python-dotenv livekit-api
"""

import os
import uuid
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# LiveKit token generation
try:
    from livekit.api import AccessToken, VideoGrants
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    print("Warning: livekit-api not installed. Run: pip install livekit-api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tutor-server")

app = FastAPI(title="AI Tutor Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config from your existing .env ──────────────────────────────────────────
LIVEKIT_URL     = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_SECRET  = os.getenv("LIVEKIT_API_SECRET", "secret")
ROOM_NAME       = "tutor-room"

# ── Token endpoint ───────────────────────────────────────────────────────────
@app.get("/api/token")
async def get_token():
    """Generate a LiveKit token for the child to join the tutor room."""
    if not LIVEKIT_AVAILABLE:
        return JSONResponse(
            {"error": "livekit-api package missing. Run: pip install livekit-api"},
            status_code=500,
        )

    participant_identity = f"child-{uuid.uuid4().hex[:8]}"

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_SECRET)
        .with_identity(participant_identity)
        .with_name("Student")
        .with_grants(VideoGrants(room_join=True, room=ROOM_NAME))
        .to_jwt()
    )

    logger.info(f"Token issued for {participant_identity} in room {ROOM_NAME}")
    return {
        "token": token,
        "url": LIVEKIT_URL,
        "room": ROOM_NAME,
        "identity": participant_identity,
    }

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "livekit_url": LIVEKIT_URL,
        "livekit_api_key": LIVEKIT_API_KEY[:6] + "***",
        "room": ROOM_NAME,
    }

# ── Serve the frontend ────────────────────────────────────────────────────────
# The frontend folder sits next to this file.
import pathlib
FRONTEND_DIR = pathlib.Path(__file__).parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
else:
    @app.get("/")
    async def no_frontend():
        return {"message": "Put index.html inside a 'frontend/' folder next to server.py"}


if __name__ == "__main__":
    import uvicorn
    print("\n🚀  AI Tutor server starting…")
    print(f"   Open http://localhost:8000 in your browser\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)