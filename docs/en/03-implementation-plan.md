# Implementation Plan

## Week 1: retrieval foundation

### Day 1 — project skeleton

- Initialize `uv`, Python 3.12, Ruff, mypy, pytest, and pre-commit.
- Define provider interfaces for chat, embeddings, and vector storage.
- Add settings validation and a `.env.example` without secrets.

### Day 2 — ingestion

- Implement Markdown, text, and PDF loaders.
- Preserve heading/page metadata and compute content hashes.
- Write unit tests for repeated ingestion and document updates.

### Day 3 — retrieval

- Implement vector indexing and top-k retrieval.
- Build the first 30 evaluation questions before tuning.
- Record retrieved IDs and scores in traces.

### Day 4 — grounded generation

- Define a structured answer schema.
- Require citations in the form of retrieved chunk IDs.
- Add a deterministic post-check that rejects nonexistent citations.

### Day 5 — baseline evaluation

- Measure Recall@k, citation precision, latency, and cost.
- Review the ten worst examples and classify failure causes.

## Week 2: Agent behavior and memory

### Day 6 — LangGraph workflow

- Add intent routing, retrieval, generation, validation, and one bounded repair edge.
- Persist checkpoints by `thread_id`.

### Day 7 — memory

- Add namespaced user memory and explicit memory candidate validation.
- Implement list, correct, and delete operations.

### Day 8 — retrieval improvements

- Add metadata filters and, only if baseline evidence supports it, hybrid retrieval or reranking.
- Re-run the unchanged golden dataset.

### Day 9 — API and UI

- Add ingestion, query, document status, and memory-management endpoints.
- Stream graph events without exposing hidden reasoning text.

### Day 10 — hardening

- Add rate limits, timeouts, redaction, integration tests, Docker packaging, and the final evaluation report.

## Definition of done

Code, tests, golden dataset, experiment report, API documentation, architecture decision records, and a reproducible local deployment are committed together.

