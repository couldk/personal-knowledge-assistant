# Testing and Acceptance

## Test layers

- Unit: parsing, hashes, chunk metadata, citation validation, memory namespaces.
- Integration: ingestion to vector store; query to structured answer.
- Regression: fixed corpus and 50-question golden dataset.
- Security: tenant isolation, malicious document instructions, path validation, and log redaction.

## Evaluation dataset

Include:

- 20 direct fact questions.
- 10 questions requiring two documents.
- 10 unanswerable questions.
- 5 questions containing misleading terms.
- 5 multi-turn questions requiring thread state.

Every example stores expected document IDs, acceptable answer facts, and whether refusal is expected.

## Metrics

- Recall@5: at least 0.85.
- Citation precision: at least 0.90.
- Grounded-answer pass rate: at least 0.85.
- Unanswerable-question refusal rate: at least 0.90.
- Cross-user isolation: 100%.
- Duplicate active chunks after re-ingestion: zero.

## Acceptance scenarios

1. Update a document and verify old chunks are inactive.
2. Ask an unsupported question and receive an evidence-based refusal.
3. Save a preference for user A and verify user B cannot retrieve it.
4. Force the model to emit a fake citation and verify the validator blocks it.
5. Restart the API and continue an existing thread from its checkpoint.

LLM-as-judge may supplement scoring, but deterministic citation checks and periodic human review remain authoritative.

