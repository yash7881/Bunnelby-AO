from __future__ import annotations

import pathlib
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.api.app import audit_service, database
from services.api.app.database import Base

# Part 10.2: no automated test may write to the user's real database.
#
# This exists because it already happened twice during Part 10.2: a /chat smoke
# test wrote 4 message rows, and the Phase J audit path wrote 78 tool_runs rows,
# because tool_executor.execute() records an audit row for every capability
# invocation and most tests legitimately call it without caring about auditing.
#
# Relying on each test to remember the patch is the wrong shape. The autouse
# fixture below redirects the write-capable session factories at an isolated
# throwaway database for the whole test session. A test that wants to ASSERT on
# audit rows still patches audit_service.SessionLocal itself; that inner patch
# simply wins.

_REAL_DB = pathlib.Path(database.DATABASE_PATH)


@pytest.fixture(scope="session")
def _isolated_engine():
    with tempfile.TemporaryDirectory() as tmp:
        engine = create_engine(f"sqlite:///{pathlib.Path(tmp).as_posix()}/test-isolation.db")
        Base.metadata.create_all(engine)
        try:
            yield engine
        finally:
            engine.dispose()


@pytest.fixture(autouse=True)
def isolate_database_writes(_isolated_engine, monkeypatch):
    """Point audit writes at a throwaway database for every test."""
    Session = sessionmaker(bind=_isolated_engine)
    monkeypatch.setattr(audit_service, "SessionLocal", Session)
    yield


@pytest.fixture(autouse=True)
def guard_real_database(request):
    """Fail loudly if a test mutated the user's real ao.db.

    Read-only assertions against the live database are allowed (a few tests
    verify migration state), so this compares row counts rather than forbidding
    access outright.
    """
    if not _REAL_DB.exists():
        yield
        return

    import sqlite3

    def counts() -> dict[str, int]:
        con = sqlite3.connect(f"file:{_REAL_DB}?mode=ro", uri=True)
        try:
            tables = [
                row[0]
                for row in con.execute("select name from sqlite_master where type='table'")
            ]
            return {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in tables}
        finally:
            con.close()

    before = counts()
    yield
    after = counts()
    if before != after:
        changed = {k: (before.get(k), after.get(k)) for k in set(before) | set(after) if before.get(k) != after.get(k)}
        pytest.fail(
            f"{request.node.nodeid} mutated the real database {_REAL_DB.name}: {changed}. "
            "Patch SessionLocal (or the owning module's session factory) instead."
        )
