import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "answering_questions.jsonl"

DOCUMENTS_DIRECTORY = PROJECT_ROOT / "evaluation" / "documents"

ALLOWED_CATEGORIES = {
    "direct",
    "paraphrase",
    "cross_document",
    "multilingual",
    "out_of_scope",
}

EXPECTED_CATEGORY_COUNTS = {
    "direct": 10,
    "paraphrase": 6,
    "cross_document": 5,
    "multilingual": 4,
    "out_of_scope": 5,
}

REQUIRED_KEYS = {
    "id",
    "question",
    "expected_file_names",
    "expected_answer_terms",
    "expected_refused",
    "category",
    "top_k",
}


def _load_payloads() -> list[dict[str, Any]]:
    """读取原始回答评测 JSONL。"""

    payloads: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        QUESTIONS_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Invalid JSON at line {line_number}: {exc}")

        assert isinstance(payload, dict)

        payloads.append(payload)

    return payloads


def test_answering_dataset_has_expected_structure() -> None:
    payloads = _load_payloads()

    assert len(payloads) == 30

    expected_ids = [f"q{index:03d}" for index in range(1, 31)]

    assert [payload["id"] for payload in payloads] == expected_ids

    assert len({payload["id"] for payload in payloads}) == 30

    for payload in payloads:
        assert set(payload) == REQUIRED_KEYS

        assert isinstance(payload["question"], str)
        assert payload["question"].strip()

        assert isinstance(
            payload["expected_file_names"],
            list,
        )
        assert isinstance(
            payload["expected_answer_terms"],
            list,
        )
        assert isinstance(
            payload["expected_refused"],
            bool,
        )

        assert payload["category"] in ALLOWED_CATEGORIES

        assert isinstance(payload["top_k"], int)
        assert 1 <= payload["top_k"] <= 20


def test_answering_dataset_has_expected_distribution() -> None:
    payloads = _load_payloads()

    category_counts = Counter(payload["category"] for payload in payloads)

    assert dict(category_counts) == (EXPECTED_CATEGORY_COUNTS)

    refused_count = sum(payload["expected_refused"] for payload in payloads)

    assert refused_count == 5


def test_answering_dataset_semantics_are_consistent() -> None:
    payloads = _load_payloads()

    available_file_names = {path.name for path in DOCUMENTS_DIRECTORY.glob("*.md")}

    for payload in payloads:
        expected_files = payload["expected_file_names"]
        expected_terms = payload["expected_answer_terms"]
        expected_refused = payload["expected_refused"]
        category = payload["category"]

        assert len(expected_files) == len(set(expected_files))
        assert len(expected_terms) == len(set(expected_terms))

        assert all(isinstance(value, str) and value.strip() for value in expected_files)
        assert all(isinstance(value, str) and value.strip() for value in expected_terms)

        assert set(expected_files) <= (available_file_names)

        if expected_refused:
            assert category == "out_of_scope"
            assert expected_files == []
            assert expected_terms == []
        else:
            assert category != "out_of_scope"
            assert expected_files
            assert expected_terms
