from __future__ import annotations

import asyncio
import sys
from asyncio import AbstractEventLoop
from collections.abc import (
    Callable,
    Iterator,
    Mapping,
)

import pytest
from pytest import Config, Item, MonkeyPatch

from personal_knowledge_assistant.config import (
    get_settings,
)


@pytest.fixture(autouse=True)
def isolate_test_authentication(
    monkeypatch: MonkeyPatch,
) -> Iterator[None]:
    """避免本地.env中的认证配置影响自动化测试。"""

    monkeypatch.setenv(
        "PKA_AUTH_ENABLED",
        "false",
    )
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


def pytest_asyncio_loop_factories(
    config: Config,
    item: Item,
) -> Mapping[
    str,
    Callable[[], AbstractEventLoop],
]:
    """在Windows上为psycopg提供兼容的Selector事件循环。"""

    del config, item

    if sys.platform == "win32":
        return {
            "selector": asyncio.SelectorEventLoop,
        }

    return {
        "default": asyncio.new_event_loop,
    }
