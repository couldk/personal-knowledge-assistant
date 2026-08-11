import json
from pathlib import Path

from pydantic import ValidationError

from personal_knowledge_assistant.evaluation.exceptions import (
    EvaluationDataError,
)
from personal_knowledge_assistant.evaluation.models import (
    RetrievalEvaluationCase,
)


def load_evaluation_cases(
    path: Path,
    *,
    expected_count: int = 30,
) -> list[RetrievalEvaluationCase]:
    """读取并校验 JSONL 检索评测问题。"""

    if expected_count <= 0:
        raise ValueError("Expected case count must be positive.")

    if not path.is_file():
        raise EvaluationDataError(f"Evaluation file does not exist: {path}")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvaluationDataError(f"Cannot read evaluation file: {path}") from exc

    cases: list[RetrievalEvaluationCase] = []
    seen_ids: set[str] = set()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
            case = RetrievalEvaluationCase.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise EvaluationDataError(f"Invalid evaluation case at line {line_number}.") from exc

        if case.id in seen_ids:
            raise EvaluationDataError(f"Duplicate evaluation case ID: {case.id}")

        seen_ids.add(case.id)
        cases.append(case)

    if len(cases) != expected_count:
        raise EvaluationDataError(
            f"Expected {expected_count} evaluation cases, received {len(cases)}."
        )

    return cases


def validate_expected_files(
    cases: list[RetrievalEvaluationCase],
    available_file_names: set[str],
) -> None:
    """确保所有黄金文件名都存在于评测语料。"""

    missing = sorted(
        {
            file_name
            for case in cases
            for file_name in case.expected_file_names
            if file_name not in available_file_names
        }
    )

    if missing:
        raise EvaluationDataError("Expected documents are missing: " + ", ".join(missing))
