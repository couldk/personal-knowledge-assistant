from hashlib import sha256

import pytest

from personal_knowledge_assistant.agent import (
    create_agent_run_config,
)


def test_agent_run_config_contains_trace_information() -> None:
    config = create_agent_run_config(
        " session-1 ",
    )

    assert config["configurable"] == {
        "thread_id": "session-1",
    }
    assert config["run_name"] == ("knowledge-agent-run")
    assert config["tags"] == [
        "personal-knowledge-assistant",
        "langgraph",
        "day6",
    ]

    metadata = config["metadata"]

    assert metadata["component"] == ("knowledge-agent")
    assert metadata["thread_id_hash"] == sha256(b"session-1").hexdigest()

    assert "session-1" not in metadata.values()


@pytest.mark.parametrize(
    "thread_id",
    [
        "",
        " ",
        "\t",
        "\r\n",
    ],
)
def test_agent_run_config_rejects_empty_thread_id(
    thread_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="thread_id cannot be empty",
    ):
        create_agent_run_config(
            thread_id,
        )
