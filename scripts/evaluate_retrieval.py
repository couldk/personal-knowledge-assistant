import argparse
import asyncio
from pathlib import Path

from personal_knowledge_assistant.application import create_application_services
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.evaluation import (
    RetrievalCaseResult,
    RetrievalEvaluator,
    load_evaluation_cases,
    validate_expected_files,
)
from personal_knowledge_assistant.ingestion import create_default_ingestion_service


def parse_args() -> argparse.Namespace:
    """解析检索评测命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Run the retrieval baseline evaluation.",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        required=True,
        help="Directory containing fixed Markdown evaluation documents.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="JSONL file containing exactly 30 retrieval questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON file that will receive the baseline report.",
    )

    return parser.parse_args()


def print_case_progress(
    position: int,
    total: int,
    result: RetrievalCaseResult,
) -> None:
    """逐题打印不包含私人正文的评测进度。"""

    rank = str(result.first_relevant_rank) if result.first_relevant_rank is not None else "miss"
    print(
        f"[query {position:02d}/{total:02d}] "
        f"{result.case_id} "
        f"hit@1={result.hit_at_1:.0f} "
        f"recall@3={result.recall_at_3:.3f} "
        f"rank={rank} "
        f"latency={result.duration_ms:.2f}ms",
        flush=True,
    )


async def run_evaluation(
    *,
    documents_directory: Path,
    questions_path: Path,
    output_path: Path,
) -> int:
    """建立临时内存索引并运行真实检索基线。"""

    settings = Settings()

    if settings.vector_store_provider != "memory":
        raise RuntimeError("Baseline evaluation requires PKA_VECTOR_STORE_PROVIDER=memory.")

    if settings.embedding_dimension is None:
        raise RuntimeError("Embedding dimension is required for evaluation.")

    document_paths = sorted(documents_directory.glob("*.md"))

    if not document_paths:
        raise RuntimeError(f"No Markdown evaluation documents found: {documents_directory}")

    cases = load_evaluation_cases(
        questions_path,
        expected_count=30,
    )
    validate_expected_files(
        cases,
        {path.name for path in document_paths},
    )

    print("=== Retrieval baseline configuration ===", flush=True)
    print(f"Model: {settings.embedding_model}", flush=True)
    print(f"Dimension: {settings.embedding_dimension}", flush=True)
    print(f"Chunk size: {settings.chunk_size}", flush=True)
    print(f"Chunk overlap: {settings.chunk_overlap}", flush=True)
    print(f"Documents: {len(document_paths)}", flush=True)
    print(f"Questions: {len(cases)}", flush=True)

    services = create_application_services(settings)
    ingestion_service = create_default_ingestion_service()

    print("=== Indexing fixed evaluation documents ===", flush=True)

    for position, path in enumerate(document_paths, start=1):
        import_result = ingestion_service.import_document(path)
        indexing_result = await services.indexing_service.index(import_result)
        print(
            f"[index {position:02d}/{len(document_paths):02d}] "
            f"{path.name} chunks={indexing_result.chunk_count}",
            flush=True,
        )

    print("=== Running 30 retrieval questions ===", flush=True)

    evaluator = RetrievalEvaluator(
        retrieval_service=services.retrieval_service,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        on_case_completed=print_case_progress,
    )
    report = await evaluator.evaluate(cases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== Retrieval baseline completed ===", flush=True)
    print(f"Cases: {report.case_count}", flush=True)
    print(f"Hit@1: {report.hit_at_1:.4f}", flush=True)
    print(f"Recall@3: {report.recall_at_3:.4f}", flush=True)
    print(f"MRR: {report.mean_reciprocal_rank:.4f}", flush=True)
    print(
        f"Average latency: {report.average_duration_ms:.2f}ms",
        flush=True,
    )
    print(f"Report: {output_path.resolve()}", flush=True)

    return 0


def main() -> int:
    """运行命令行入口。"""

    args = parse_args()

    return asyncio.run(
        run_evaluation(
            documents_directory=args.documents,
            questions_path=args.questions,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
