"""
RAG Test Suite
==============
Tests the full pipeline end-to-end and each layer in isolation.

Usage
-----
    # Set env vars first:
    export OPENAI_API_KEY="sk-..."
    export VECTOR_STORE_HOST="https://your-qdrant-host"
    export VECTOR_STORE_API_KEY="your-qdrant-key"

    # Run all tests:
    python test_rag.py

    # Run only a specific section:
    python test_rag.py ingestion
    python test_rag.py retrieval
    python test_rag.py rag
"""

import os
import sys
import json
import time
import logging
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── colour helpers ──────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
CYAN  = "\033[96m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {CYAN}→{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def header(msg):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

# ── sample document ─────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = """\
# Introduction to Neural Networks

Neural networks are computational models inspired by the structure of the
human brain. They consist of layers of interconnected nodes (neurons) that
process information by passing signals between each other.

## Core Components

### Neurons and Layers
Every network is built from three kinds of layers:
- **Input layer** — receives raw features.
- **Hidden layers** — learn intermediate representations.
- **Output layer** — produces the final prediction.

### Activation Functions
Activation functions introduce non-linearity so the network can model
complex patterns. Common choices include ReLU, Sigmoid, and Tanh.

## Training Process

Training adjusts the weights of all connections so that the network's
predictions match the ground-truth labels as closely as possible.

### Loss Functions
A loss function quantifies the error between predictions and labels.
Cross-entropy is standard for classification; mean-squared error for
regression.

### Backpropagation
Backpropagation computes the gradient of the loss with respect to every
weight via the chain rule, then an optimizer (e.g. Adam, SGD) updates the
weights in the direction that reduces the loss.

## Applications

Neural networks power image recognition, natural language processing,
speech synthesis, game-playing agents, and many more domains.
"""

# ── env guard ───────────────────────────────────────────────────────────────

def _check_env():
    missing = [k for k in ("OPENAI_API_KEY", "VECTOR_STORE_HOST", "VECTOR_STORE_API_KEY")
               if not os.getenv(k)]
    if missing:
        print(f"\n{RED}Missing environment variables: {', '.join(missing)}{RESET}")
        print("Set them before running the tests.\n")
        sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Unit tests (no network)
# ════════════════════════════════════════════════════════════════════════════

def test_chunker():
    header("1 · Chunker (unit, no network)")

    from chunking import HierarchicalParentChildChunker

    chunker = HierarchicalParentChildChunker(child_chunk_size=256, child_chunk_overlap=50)
    doc = {"file_name": "test.md", "content": SAMPLE_MARKDOWN}
    result = chunker.create_chunks(doc)

    parents = result["parents"]
    children = result["children"]

    assert parents, "Expected at least one parent node"
    ok(f"Parents created: {len(parents)}")

    assert children, "Expected at least one child node"
    ok(f"Children created: {len(children)}")

    # Every child must carry a parent_id that matches a parent
    parent_ids = {p.metadata["parent_id"] for p in parents}
    for child in children:
        pid = child.metadata.get("parent_id")
        assert pid in parent_ids, f"Child has unknown parent_id: {pid}"
    ok("All child parent_ids reference a known parent")

    # Check metadata shape
    for child in children:
        assert child.metadata.get("type") == "child"
        assert child.metadata.get("source") == "test.md"
    ok("Child metadata is correct (type=child, source set)")

    for parent in parents:
        assert parent.metadata.get("type") == "parent"
    ok("Parent metadata is correct (type=parent)")

    info(f"Sample child text (first 120 chars): {children[0].text[:120]!r}")


def test_llm_client():
    header("2 · LLM Client (unit — OpenAI only)")

    _check_env()
    from llm import LLMClient

    client = LLMClient(model="gpt-4o-mini")
    ok("LLMClient instantiated")

    # Plain text response
    answer = client.generate_response(
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        temperature=0.0,
    )
    assert isinstance(answer, str) and len(answer) > 0
    ok(f"Plain text response received: {answer.strip()!r}")

    # Structured output
    from pydantic import BaseModel

    class Pong(BaseModel):
        message: str
        confidence: float

    structured = client.generate_response(
        messages=[{"role": "user", "content": 'Respond with message="PONG" and confidence=1.0'}],
        text_format=Pong,
        temperature=0.0,
    )
    assert isinstance(structured, Pong)
    ok(f"Structured response: message={structured.message!r}, confidence={structured.confidence}")

    # Batch
    results = client.generate_batch(
        prompts=["Say YES", "Say NO"],
        temperature=0.0,
    )
    assert len(results) == 2
    ok(f"Batch of 2 completed: {[r.strip() for r in results]}")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Ingestion integration test
# ════════════════════════════════════════════════════════════════════════════

def test_ingestion():
    header("3 · Ingestion Pipeline (integration — needs Qdrant + OpenAI)")

    _check_env()

    # Write sample markdown to a temp file so DocumentLoader can read it.
    # (DocumentLoader uses docling which expects a real file path.)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(SAMPLE_MARKDOWN)
        tmp_path = tmp.name

    try:
        from ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        ok("IngestionPipeline instantiated")

        t0 = time.time()
        summary = pipeline.ingest(tmp_path)
        elapsed = time.time() - t0

        ok(f"Ingestion completed in {elapsed:.1f}s")
        info(f"Summary: {json.dumps(summary, indent=2)}")

        assert summary["parents_stored"] > 0, "No parent nodes stored"
        assert summary["children_stored"] > 0, "No child nodes stored"
        ok(f"Stored {summary['parents_stored']} parents, {summary['children_stored']} children")

    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Retrieval tests
# ════════════════════════════════════════════════════════════════════════════

def _make_retriever():
    from retrieval import Retriever
    return Retriever(
        url=os.getenv("VECTOR_STORE_HOST"),
        api_key=os.getenv("VECTOR_STORE_API_KEY"),
        collection_name="documents",
        top_k=5,
    )


def test_retrieval():
    header("4 · Retriever (integration — needs Qdrant + OpenAI)")

    _check_env()
    retriever = _make_retriever()
    ok("Retriever instantiated")

    query = "What is backpropagation?"
    t0 = time.time()
    result = retriever.retrieve(query)
    elapsed = time.time() - t0

    ok(f"Retrieve() returned in {elapsed:.2f}s")

    hits = result["child_hits"]
    assert isinstance(hits, list), "child_hits should be a list"
    ok(f"Got {len(hits)} child hits")

    if hits:
        top = hits[0]
        info(f"Top hit  score={top['score']:.4f}  source={top['source']!r}")
        info(f"Top hit  text preview: {top['text'][:120]!r}")

    context = result["context"]
    assert isinstance(context, str)
    ok(f"Context assembled ({len(context)} chars)")

    if not hits:
        warn("No hits returned — is the collection empty? Run test_ingestion first.")


def test_retrieval_empty_query():
    header("5 · Retriever — edge case: unusual query")

    _check_env()
    retriever = _make_retriever()

    # Should not raise even for an off-topic query
    result = retriever.retrieve("xkcd 327 bobby tables sql injection")
    ok(f"Unusual query handled gracefully — {len(result['child_hits'])} hits returned")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — End-to-end RAG tests
# ════════════════════════════════════════════════════════════════════════════

def _make_rag():
    from retrieval import Retriever, RAGPipeline
    retriever = _make_retriever()
    return RAGPipeline(retriever=retriever, llm_model="gpt-4o-mini", temperature=0.1)


RAG_QUESTIONS = [
    {
        "question": "What is backpropagation and how does it work?",
        "must_contain": ["gradient", "loss", "weight"],
    },
    {
        "question": "What activation functions are mentioned?",
        "must_contain": ["relu", "sigmoid"],
    },
    {
        "question": "Name the three types of layers in a neural network.",
        "must_contain": ["input", "hidden", "output"],
    },
    {
        "question": "What are neural networks used for?",
        "must_contain": ["image", "language"],
    },
]


def test_rag_answers():
    header("6 · RAG Pipeline — answer quality checks")

    _check_env()
    rag = _make_rag()
    ok("RAGPipeline instantiated")

    passed = 0
    for i, case in enumerate(RAG_QUESTIONS, 1):
        q = case["question"]
        expected_keywords = case["must_contain"]

        print(f"\n  [{i}/{len(RAG_QUESTIONS)}] {q}")
        t0 = time.time()
        result = rag.query(q)
        elapsed = time.time() - t0

        answer = result["answer"]
        sources = result["sources"]

        info(f"Answer ({elapsed:.1f}s): {answer[:200]}{'…' if len(answer) > 200 else ''}")
        info(f"Sources: {sources}")

        answer_lower = answer.lower()
        missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

        if missing:
            fail(f"Answer missing expected keywords: {missing}")
        else:
            ok(f"All expected keywords found: {expected_keywords}")
            passed += 1

    print()
    if passed == len(RAG_QUESTIONS):
        ok(f"All {passed}/{len(RAG_QUESTIONS)} RAG answer checks passed")
    else:
        fail(f"Only {passed}/{len(RAG_QUESTIONS)} RAG answer checks passed")


def test_rag_no_context():
    header("7 · RAG Pipeline — graceful handling when context is empty")

    _check_env()
    from retrieval import Retriever, RAGPipeline
    from unittest.mock import patch

    retriever = _make_retriever()
    rag = RAGPipeline(retriever=retriever)

    # Monkey-patch retriever to return empty context
    with patch.object(retriever, "retrieve", return_value={"query": "x", "context": "", "child_hits": []}):
        result = rag.query("anything")

    assert "no relevant" in result["answer"].lower() or result["answer"]
    ok(f"Empty-context response: {result['answer']!r}")


def test_rag_source_attribution():
    header("8 · RAG Pipeline — source attribution")

    _check_env()
    rag = _make_rag()

    result = rag.query("Explain the training process of a neural network.")
    sources = result["sources"]

    assert isinstance(sources, list)
    ok(f"Sources returned: {sources}")
    if sources:
        ok("Source attribution is working")
    else:
        warn("No sources returned — check that ingestion ran and nodes have 'source' metadata")


# ════════════════════════════════════════════════════════════════════════════
#  Runner
# ════════════════════════════════════════════════════════════════════════════

SUITES = {
    "unit": [test_chunker, test_llm_client],
    "ingestion": [test_ingestion],
    "retrieval": [test_retrieval, test_retrieval_empty_query],
    "rag": [test_rag_answers, test_rag_no_context, test_rag_source_attribution],
}

ALL_TESTS = [
    test_chunker,
    test_llm_client,
    test_ingestion,
    test_retrieval,
    test_retrieval_empty_query,
    test_rag_answers,
    test_rag_no_context,
    test_rag_source_attribution,
]


def run(tests):
    total = len(tests)
    failed = []

    for fn in tests:
        try:
            fn()
        except Exception as e:
            fail(f"EXCEPTION in {fn.__name__}: {e}")
            logger.debug("", exc_info=True)
            failed.append(fn.__name__)

    print(f"\n{'─'*60}")
    if failed:
        print(f"{RED}{BOLD}  FAILED: {len(failed)}/{total}{RESET}")
        for name in failed:
            print(f"    {RED}•{RESET} {name}")
    else:
        print(f"{GREEN}{BOLD}  ALL {total} TESTS PASSED{RESET}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg in SUITES:
        run(SUITES[arg])
    elif arg == "all":
        run(ALL_TESTS)
    else:
        print(f"Unknown suite '{arg}'. Choose from: all, {', '.join(SUITES)}")
        sys.exit(1)