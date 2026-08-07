# Deployment and Operations

## Environments

- Local: API, PostgreSQL/pgvector, and optional LangSmith tracing.
- Staging: production-like database with synthetic documents.
- Production: managed database, secret manager, TLS, backups, and tenant-aware authentication.

## Configuration

Required settings should include model provider, model name, embedding model and dimensions, database URL, maximum retrieved chunks, timeout, and tracing sampling rate.

Pin embedding model and dimensions with the index version. Changing either requires a new index or migration.

## Observability

Capture:

- Request and trace ID.
- Graph node durations.
- Retrieval IDs and scores, without logging full private text by default.
- Input/output token counts and estimated cost.
- Refusal, validation failure, and memory-write counts.

## Operational procedures

- Back up document metadata, memory, and checkpoints.
- Rebuild vector indexes from source documents; do not treat embeddings as the only copy.
- Run the golden dataset before model, prompt, embedding, chunking, or framework upgrades.
- Sample failed and low-confidence production traces into the offline evaluation set.

## Release gate

No release proceeds when tenant isolation fails, citation precision regresses by more than five percentage points, or the dependency lock file is missing.

