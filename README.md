# Voice-Tutor

Voice-Tutor is a voice-enabled tutoring demo built with LiveKit, FastAPI, and OpenAI-style speech/LLM integrations. It lets a student join a WebRTC room from the browser, speak to an AI tutor, and receive spoken responses plus transcript updates.

## What this project does

- Serves a frontend UI from `ui_server.py`.
- Issues LiveKit access tokens for browser clients.
- Uploads PDF study material and ingests it into the app.
- Runs a LiveKit voice agent backend in `main.py` that performs STT, LLM reasoning, and TTS.
- Sends transcript data to the browser over LiveKit data channels.

## Key components

- `ui_server.py` - FastAPI app that serves `ui/index.html`, provides `/api/token`, `/api/ingest`, and health checks.
- `main.py` - Starts the LiveKit agent worker using `src/query/tutor_agent.py`.
- `src/query/tutor_agent.py` - Agent logic for user query handling, RAG retrieval, and data channel broadcasts.
- `ui/index.html` - Browser front-end that connects to LiveKit, captures microphone audio, and plays back agent audio.

## Prerequisites

- Python 3.14+
- A valid LiveKit deployment with:
  - `LIVEKIT_URL`
  - `LIVEKIT_API_KEY`
  - `LIVEKIT_API_SECRET`

## Setup

1. Create and activate your virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Create a `.env.local` file in the repo root with:

```text
LIVEKIT_URL=https://your-livekit-host
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
```

## Running the app

This project requires two processes:

1. Start the UI server:

```powershell
python ui_server.py
```

2. Start the LiveKit agent backend:

```powershell
python main.py
```

Then open `http://localhost:8080` in your browser.

## Notes

- The frontend uses WebRTC to publish the microphone and receive tutor audio.
- The agent backend must be running for the tutor voice and transcript updates to work.
- PDF ingestion is handled through the UI upload button and the `/api/ingest` endpoint.

## Troubleshooting

- If you only hear your own voice, make sure `main.py` is running and that the agent joins the same LiveKit room.
- If token generation fails, verify `.env.local` and the LiveKit credentials.
