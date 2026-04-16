"""Shared fixtures for the test suite.

Every test gets its own temporary SQLite database so tests never interfere
with each other or with the real ``app.db``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from backend import database


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path) -> Generator[None, None, None]:
    """Point the database module at a disposable file for each test.

    Overrides ``database.DB_PATH`` before the test runs, calls ``init_db()``
    to create + seed the schema, and restores the original path afterwards.
    """
    original_path = database.DB_PATH
    database.DB_PATH = str(tmp_path / "test.db")
    database.init_db()
    yield
    database.DB_PATH = original_path


@pytest.fixture(params=["asyncio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    """Restrict anyio to the asyncio backend only (trio is not installed)."""
    return request.param
