from pathlib import Path

import pytest
from pydantic import ValidationError

from personal_knowledge_assistant.config import Settings


def test_settings_have_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # 切换到不包含 .env 的临时目录，避免本地配置影响默认值测试
    monkeypatch.chdir(tmp_path)

    settings = Settings.model_validate({})

    assert settings.app_env == "local"
    assert settings.retrieval_top_k == 5

    assert settings.chat_api_key is None
    assert settings.chat_base_url == ("https://api.deepseek.com")
    assert settings.chat_timeout_seconds == 60.0
    assert settings.chat_max_retries == 2
    assert settings.chat_max_tokens == 2048
    assert settings.chat_temperature == 0.0

    assert settings.embedding_dimension is None
    assert settings.embedding_base_url == "https://api.siliconflow.cn/v1"

    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64


def test_retrieval_top_k_cannot_be_zero() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"retrieval_top_k": 0})


def test_embedding_dimension_cannot_be_zero() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"embedding_dimension": 0})


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "chunk_size": 128,
                "chunk_overlap": 128,
            }
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("chat_timeout_seconds", 0),
        ("chat_max_retries", -1),
        ("chat_max_tokens", 127),
        ("chat_temperature", -0.1),
        ("chat_temperature", 2.1),
    ],
)
def test_chat_settings_reject_invalid_values(
    field_name: str,
    invalid_value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                field_name: invalid_value,
            }
        )
