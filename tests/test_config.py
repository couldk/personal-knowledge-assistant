import pytest
from pydantic import ValidationError

from personal_knowledge_assistant.config import Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings.model_validate({})

    assert settings.app_env == "local"
    assert settings.retrieval_top_k == 5
    assert settings.chat_api_key is None


def test_retrieval_top_k_cannot_be_zero() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"retrieval_top_k": 0})
