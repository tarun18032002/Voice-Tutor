# Voice Tutor production code

## File placement

```text
project-root/
├── main.py
├── .env.local                 # local only; do not commit
└── agent/
    ├── __init__.py
    ├── config.py
    ├── graph.py
    ├── llm.py
    ├── models.py
    ├── prompts.py
    ├── retriver.py            # spelling kept for compatibility
    ├── test.py                # manual smoke test only
    ├── tools.py
    ├── tutor.py
    └── websearch.py
```

Copy `.env.example` into your deployment configuration. In local development,
you may copy it to `.env.local` and fill the values. Never commit real keys.

## Main changes

1. Production imports `agent.graph`; it no longer imports the duplicated graph
   from `agent.test`.
2. `agent.test` is only a manual smoke test.
3. `tools.py` is the only tool registry, so graph and tutor use the same tools.
4. `prompts.py` is the only tutor system prompt.
5. Qdrant, OpenAI embedding, and Tavily clients are reused per worker process.
6. Model, retrieval, and web calls have bounded timeouts and safe errors.
7. The graph has a graceful model/tool-loop limit.
8. Private and web results are cached per LiveKit room.
9. Likely follow-up questions receive recent fetched context, reducing duplicate
   external lookups without injecting stale context into every new topic.
10. LiveKit uses one `LLMAdapter` and streams both model tokens and optional
    fixed tool-status messages.
11. Preemptive LLM generation remains enabled, while preemptive TTS is off by
    default to reduce overlapping or stale audio.
12. All production values are configurable through environment variables.

## Local smoke test

From the project root:

```bash
python -m agent.test "What is carbon dioxide? Explain it simply."
```

For a general-knowledge-only smoke test, you may temporarily set:

```env
ENABLE_RETRIEVAL=false
ENABLE_WEB_SEARCH=false
```

## Before production

- Keep the dependency versions already tested in your environment pinned.
- Confirm the Qdrant vector dimension matches `EMBEDDING_MODEL`.
- Confirm Qdrant payloads contain the field configured by `QDRANT_TEXT_FIELD`.
- Test interruptions, filler words, long user turns, Qdrant timeout, Tavily
  timeout, and two simultaneous rooms in staging.
- Do not log transcripts, retrieved private documents, secrets, or raw web
  queries unless your privacy policy explicitly allows it.
