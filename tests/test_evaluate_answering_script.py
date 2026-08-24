import argparse
from pathlib import Path

import pytest
from scripts.evaluate_answering import (
    non_negative_float,
    parse_args,
    positive_float,
    print_case_progress,
)

from personal_knowledge_assistant.evaluation import (
    AnswerCaseResult,
    AnswerFailureReason,
    EvaluationCategory,
)


def _result() -> AnswerCaseResult:
    return AnswerCaseResult(
        case_id="q001",
        category=EvaluationCategory.DIRECT,
        succeeded=True,
        expected_refused=False,
        actual_refused=False,
        retrieved_file_names=["rag.md"],
        retrieved_chunk_ids=["a" * 64],
        cited_file_names=["rag.md"],
        cited_chunk_ids=["a" * 64],
        retrieval_recall_at_3=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        answer_term_coverage=0.75,
        refusal_correct=True,
        duration_ms=1200.0,
        retrieval_duration_ms=100.0,
        answering_duration_ms=1100.0,
        prompt_tokens=200,
        completion_tokens=50,
        total_tokens=250,
        estimated_cost=0.0005,
        failure_reasons=[],
        error_type=None,
    )


def test_parse_args_reads_required_values() -> None:
    args = parse_args(
        [
            "--documents",
            "evaluation/documents",
            "--questions",
            "evaluation/answering_questions.jsonl",
            "--output",
            "evaluation/results/answering.json",
            "--input-cost-per-million",
            "2.0",
            "--output-cost-per-million",
            "8.0",
        ]
    )

    assert args.documents == Path("evaluation/documents")
    assert args.questions == Path("evaluation/answering_questions.jsonl")
    assert args.output == Path("evaluation/results/answering.json")
    assert args.input_cost_per_million == 2.0
    assert args.output_cost_per_million == 8.0
    assert args.high_latency_threshold_ms == 5000.0
    assert args.overwrite is False


def test_parse_args_accepts_overwrite() -> None:
    args = parse_args(
        [
            "--documents",
            "documents",
            "--questions",
            "questions.jsonl",
            "--output",
            "report.json",
            "--input-cost-per-million",
            "0",
            "--output-cost-per-million",
            "0",
            "--overwrite",
        ]
    )

    assert args.overwrite is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0.0),
        ("1.5", 1.5),
    ],
)
def test_non_negative_float_accepts_valid_values(
    value: str,
    expected: float,
) -> None:
    assert non_negative_float(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "-1",
        "not-a-number",
    ],
)
def test_non_negative_float_rejects_invalid_values(
    value: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_float(value)


def test_positive_float_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("0")


def test_print_progress_does_not_expose_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_case_progress(
        1,
        30,
        _result(),
    )

    output = capsys.readouterr().out

    assert "[answer 01/30]" in output
    assert "q001" in output
    assert "retrieval=1.000" in output
    assert "tokens=250" in output
    assert "failures=none" in output

    assert "RAG 如何工作" not in output
    assert "RAG 先检索" not in output


def test_print_progress_includes_failure_reasons(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _result().model_copy(
        update={
            "failure_reasons": [
                AnswerFailureReason.RETRIEVAL_MISS,
                AnswerFailureReason.WRONG_REFUSAL,
            ]
        }
    )

    print_case_progress(
        2,
        30,
        result,
    )

    output = capsys.readouterr().out

    assert "failures=retrieval_miss,wrong_refusal" in output
