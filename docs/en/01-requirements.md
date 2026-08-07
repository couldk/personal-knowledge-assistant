# Requirements

## Problem statement

Users need to search personal technical documents conversationally without sacrificing source traceability or leaking memories between users.

## Core user stories

- As a user, I can ingest Markdown, PDF, and text files.
- I can ask a question and receive an answer with document and chunk citations.
- I am told when available evidence is insufficient.
- I can continue a conversation without resending all prior messages.
- I can save, inspect, correct, and delete personal preferences.
- As an operator, I can trace retrieval, model, and tool activity.

## Functional requirements

1. Document ingestion is incremental and deduplicated by content hash.
2. Every chunk retains source path, page or heading, timestamps, and document ID.
3. Retrieval supports metadata filters and returns scored evidence.
4. The answer schema contains `answer`, `citations`, `confidence`, and `refused`.
5. Short-term conversation state is scoped by `thread_id`.
6. Long-term memory is scoped by `tenant_id` and `user_id`.
7. Memory writes require an explicit policy; raw conversations are not automatically persisted.

## Non-functional requirements

- No answer may cite a chunk that was not retrieved during that run.
- A file update must not leave obsolete chunks active.
- P95 response target for the local corpus: under 8 seconds, excluding first-time ingestion.
- All model and embedding providers must be replaceable through adapters.
- Logs must redact document content and secrets by default.

## Out of scope for version 1

- Autonomous web browsing.
- Image OCR for scanned PDFs.
- Collaborative document editing.
- Training or fine-tuning an embedding model.

## Main risks

- Poor chunking, retrieval misses, misleading similarity scores, citation drift, context overflow, stale memories, and cross-user leakage.

