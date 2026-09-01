from __future__ import annotations

import asyncio
import sys
from asyncio import AbstractEventLoop
from collections.abc import Callable, Mapping

from pytest import Config, Item


def pytest_asyncio_loop_factories(
    config: Config,
    item: Item,
) -> Mapping[str, Callable[[], AbstractEventLoop]]:
    """在 Windows 上为 psycopg 提供兼容的 Selector 事件循环。"""

    del config, item

    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}

    return {"default": asyncio.new_event_loop}
