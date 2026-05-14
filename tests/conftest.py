"""Shared fixtures for the test suite.

Every test gets its own temporary SQLite database so tests never interfere
with each other or with the real ``app.db``.

Pass ``--insecure`` on the pytest command line to run the full test suite
against the deliberately vulnerable ``backend/app_insecure.py`` variant:

    pytest --insecure -v

Security tests that pass on the secure app will fail on the insecure one,
demonstrating exactly which defences each test is measuring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import pytest

from backend import database


# ---------------------------------------------------------------------------
# --insecure flag: swap backend.app for the vulnerable variant
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--insecure",
        action="store_true",
        default=False,
        help=(
            "Run tests against the insecure app variant "
            "(no sanitisation, no access control, minimal system prompt)."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Swap backend.app → backend.app_insecure before any test module is imported."""
    try:
        insecure = config.getoption("--insecure", default=False)
    except ValueError:
        insecure = False
    if insecure:
        import backend
        import backend.app_insecure as _insecure_module
        # Replace in sys.modules so "from backend.app import X" gets the insecure names.
        sys.modules["backend.app"] = _insecure_module
        # Also update the package attribute so mock.patch("backend.app.X") resolves correctly.
        backend.app = _insecure_module


# ---------------------------------------------------------------------------
# Per-test database isolation
# ---------------------------------------------------------------------------


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
