import argparse
import asyncio
from pathlib import Path

from personal_knowledge_assistant.answering import (
    InMemoryAnsweringTracer,
)
from personal_knowledge_assistant.application import (
    create_application_services,
)
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.evaluation import (
    AnswerCaseResult,
    AnsweringEvaluator,
    load_answer_evaluation_cases,
    validate_answer_expected_files,
)
from personal_knowledge_assistant.ingestion import (
    create_default_ingestion_service,
)
from personal_knowledge_assistant.retrieval import (
    InMemoryRetrievalTracer,
)


def non_negative_float(value: str) -> float:
    """解析非负浮点数命令行参数。"""

    try:
        parsed_value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, received: {value}") from exc

    if parsed_value < 0:
        raise argparse.ArgumentTypeError("Value cannot be negative.")

    return parsed_value


def positive_float(value: str) -> float:
    """解析正浮点数命令行参数。"""

    parsed_value = non_negative_float(value)

    if parsed_value == 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")

    return parsed_value


def parse_args(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """解析回答基线评测命令行参数。"""

    parser = argparse.ArgumentParser(
        description=("Run the evidence-grounded answering baseline evaluation."),
    )

    parser.add_argument(
        "--documents",
        type=Path,
        required=True,
        help=("Directory containing the fixed Markdown evaluation documents."),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help=("JSONL file containing exactly 30 answering evaluation questions."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=("JSON file that will receive the answering baseline report."),
    )
    parser.add_argument(
        "--input-cost-per-million",
        type=non_negative_float,
        required=True,
        help=("Chat model input price for one million tokens."),
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=non_negative_float,
        required=True,
        help=("Chat model output price for one million tokens."),
    )
    parser.add_argument(
        "--high-latency-threshold-ms",
        type=positive_float,
        default=5000.0,
        help=("Latency above this value is classified as high latency. Default: 5000."),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Allow an existing baseline report to be overwritten."),
    )

    return parser.parse_args(arguments)


def print_case_progress(
    position: int,
    total: int,
    result: AnswerCaseResult,
) -> None:
    """逐题打印不包含问题或回答正文的进度。"""

    failure_summary = (
        ",".join(reason.value for reason in result.failure_reasons)
        if result.failure_reasons
        else "none"
    )

    print(
        f"[answer {position:02d}/{total:02d}] "
        f"{result.case_id} "
        f"success={result.succeeded} "
        f"retrieval={result.retrieval_recall_at_3:.3f} "
        f"citation_precision="
        f"{result.citation_precision:.3f} "
        f"citation_recall="
        f"{result.citation_recall:.3f} "
        f"term_coverage="
        f"{result.answer_term_coverage:.3f} "
        f"refusal="
        f"{'pass' if result.refusal_correct else 'fail'} "
        f"latency={result.duration_ms:.2f}ms "
        f"tokens={result.total_tokens} "
        f"cost={result.estimated_cost:.8f} "
        f"failures={failure_summary}",
        flush=True,
    )


async def run_evaluation(
    *,
    documents_directory: Path,
    questions_path: Path,
    output_path: Path,
    input_cost_per_million: float,
    output_cost_per_million: float,
    high_latency_threshold_ms: float,
    overwrite: bool = False,
) -> int:
    """建立临时内存索引并执行回答基线评测。"""

    if output_path.exists() and not overwrite:
        raise RuntimeError(
            f"Output report already exists. Use --overwrite to replace it: {output_path}"
        )

    settings = Settings()

    if settings.vector_store_provider != "memory":
        raise RuntimeError(
            "Answering baseline evaluation requires PKA_VECTOR_STORE_PROVIDER=memory."
        )

    if settings.embedding_dimension is None:
        raise RuntimeError("Embedding dimension is required for answering evaluation.")

    document_paths = sorted(documents_directory.glob("*.md"))

    if not document_paths:
        raise RuntimeError(f"No Markdown evaluation documents found: {documents_directory}")

    cases = load_answer_evaluation_cases(
        questions_path,
        expected_count=30,
    )

    validate_answer_expected_files(
        cases,
        {path.name for path in document_paths},
    )

    retrieval_tracer = InMemoryRetrievalTracer()
    answering_tracer = InMemoryAnsweringTracer()

    print(
        "=== Answering baseline configuration ===",
        flush=True,
    )
    print(
        f"Chat model: {settings.chat_model}",
        flush=True,
    )
    print(
        f"Embedding model: {settings.embedding_model}",
        flush=True,
    )
    print(
        f"Embedding dimension: {settings.embedding_dimension}",
        flush=True,
    )
    print(
        f"Chunk size: {settings.chunk_size}",
        flush=True,
    )
    print(
        f"Chunk overlap: {settings.chunk_overlap}",
        flush=True,
    )
    print(
        f"Documents: {len(document_paths)}",
        flush=True,
    )
    print(
        f"Questions: {len(cases)}",
        flush=True,
    )
    print(
        f"Input cost per million tokens: {input_cost_per_million}",
        flush=True,
    )
    print(
        f"Output cost per million tokens: {output_cost_per_million}",
        flush=True,
    )

    services = create_application_services(
        settings,
        retrieval_tracer=retrieval_tracer,
        answering_tracer=answering_tracer,
    )

    ingestion_service = create_default_ingestion_service()

    print(
        "=== Indexing fixed evaluation documents ===",
        flush=True,
    )

    for position, path in enumerate(
        document_paths,
        start=1,
    ):
        import_result = ingestion_service.import_document(path)

        indexing_result = await services.indexing_service.index(import_result)

        print(
            f"[index {position:02d}/"
            f"{len(document_paths):02d}] "
            f"{path.name} "
            f"chunks={indexing_result.chunk_count}",
            flush=True,
        )

    print(
        "=== Running 30 answering questions ===",
        flush=True,
    )

    evaluator = AnsweringEvaluator(
        retrieval_service=(services.retrieval_service),
        answering_service=(services.answering_service),
        retrieval_tracer=retrieval_tracer,
        answering_tracer=answering_tracer,
        chat_model=settings.chat_model,
        embedding_model=(settings.embedding_model),
        input_cost_per_million=(input_cost_per_million),
        output_cost_per_million=(output_cost_per_million),
        high_latency_threshold_ms=(high_latency_threshold_ms),
        on_case_completed=print_case_progress,
    )

    report = await evaluator.evaluate(cases)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "=== Answering baseline completed ===",
        flush=True,
    )
    print(
        f"Cases: {report.case_count}",
        flush=True,
    )
    print(
        f"Success rate: {report.success_rate:.4f}",
        flush=True,
    )
    print(
        f"Retrieval Recall@3: {report.retrieval_recall_at_3:.4f}",
        flush=True,
    )
    print(
        f"Citation precision: {report.citation_precision:.4f}",
        flush=True,
    )
    print(
        f"Citation recall: {report.citation_recall:.4f}",
        flush=True,
    )
    print(
        f"Answer term coverage: {report.answer_term_coverage:.4f}",
        flush=True,
    )
    print(
        f"Refusal accuracy: {report.refusal_accuracy:.4f}",
        flush=True,
    )
    print(
        f"Average latency: {report.average_duration_ms:.2f}ms",
        flush=True,
    )
    print(
        f"Prompt tokens: {report.total_prompt_tokens}",
        flush=True,
    )
    print(
        f"Completion tokens: {report.total_completion_tokens}",
        flush=True,
    )
    print(
        f"Total tokens: {report.total_tokens}",
        flush=True,
    )
    print(
        f"Estimated total cost: {report.estimated_total_cost:.8f}",
        flush=True,
    )
    print(
        "Worst cases: " + ", ".join(result.case_id for result in report.worst_cases),
        flush=True,
    )
    print(
        f"Report: {output_path.resolve()}",
        flush=True,
    )

    return 0


def main() -> int:
    """运行命令行入口。"""

    args = parse_args()

    return asyncio.run(
        run_evaluation(
            documents_directory=args.documents,
            questions_path=args.questions,
            output_path=args.output,
            input_cost_per_million=(args.input_cost_per_million),
            output_cost_per_million=(args.output_cost_per_million),
            high_latency_threshold_ms=(args.high_latency_threshold_ms),
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
