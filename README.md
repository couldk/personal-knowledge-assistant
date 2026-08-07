# Personal Knowledge Assistant

A citation-first personal knowledge assistant over Markdown, PDF, and text documents. This is the first project in the roadmap and focuses on RAG, tool calling, short-term state, and safe long-term memory.

## Primary stack

- LangGraph for orchestration and state.
- LlamaIndex for ingestion, chunking, indexing, and retrieval.
- PostgreSQL + pgvector for production; Chroma may be used for the first local milestone.
- FastAPI for the service boundary.
- LangSmith for traces and evaluation.

## Target duration

Two weeks at 8–12 hours per week.

## Milestones

1. Ingest documents and answer with citations.
2. Route between normal chat and knowledge retrieval.
3. Add user-scoped memory and evidence-based refusal.
4. Evaluate retrieval and grounded answers on a 50-question dataset.

## Suggested commands after implementation

```bash
uv sync
uv run python -m app.cli ingest ./sample_docs
uv run python -m app.cli query "What does the deployment guide require?"
uv run pytest
```

## Documentation

- [中文文档](docs/zh-CN/README.md)
- [English documentation](docs/en/README.md)
