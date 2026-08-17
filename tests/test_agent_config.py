from pathlib import Path

import pytest
from pydantic import ValidationError

from personal_knowledge_assistant.config import Settings


def _clear_agent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """清除可能由终端或IDE注入的Agent环境变量。"""

    monkeypatch.delenv(
        "PKA_AGENT_MAX_RETRIEVAL_ATTEMPTS",
        raising=False,
    )
    monkeypatch.delenv(
        "PKA_AGENT_MIN_RELEVANCE_SCORE",
        raising=False,
    )


def test_agent_settings_have_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """没有.env时使用安全、有限的Agent默认值。"""

    monkeypatch.chdir(tmp_path)
    _clear_agent_environment(monkeypatch)

    settings = Settings.model_validate({})

    assert settings.agent_max_retrieval_attempts == 2
    assert settings.agent_min_relevance_score == 0.35


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("agent_max_retrieval_attempts", 0),
        ("agent_max_retrieval_attempts", 6),
        ("agent_min_relevance_score", -1.1),
        ("agent_min_relevance_score", 1.1),
        ("agent_min_relevance_score", float("nan")),
        ("agent_min_relevance_score", float("inf")),
    ],
)
def test_agent_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field_name: str,
    invalid_value: int | float,
) -> None:
    """Agent配置必须满足次数、范围和有限浮点数约束。"""

    monkeypatch.chdir(tmp_path)
    _clear_agent_environment(monkeypatch)

    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                field_name: invalid_value,
            }
        )


def test_agent_settings_read_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Settings能够读取PKA_前缀的Agent环境变量。"""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "PKA_AGENT_MAX_RETRIEVAL_ATTEMPTS",
        "3",
    )
    monkeypatch.setenv(
        "PKA_AGENT_MIN_RELEVANCE_SCORE",
        "0.45",
    )

    settings = Settings()

    assert settings.agent_max_retrieval_attempts == 3
    assert settings.agent_min_relevance_score == 0.45
