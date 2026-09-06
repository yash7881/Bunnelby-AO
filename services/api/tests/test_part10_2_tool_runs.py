from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from services.api.app import audit_service, brain_agent, tool_execution, tool_executor
from services.api.app.database import Base
from services.api.app.models import Approval, TaskLog, ToolRun, VerificationEvidence
from services.api.app.orchestrator import OrchestratorResult

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MIGRATIONS = REPO_ROOT / "database/migrations/versions"


class MigrationShapeTests(unittest.TestCase):
    def test_migrations_are_sequential_and_additive(self) -> None:
        for revision, down, table in (
            ("0005", "0004", "tool_runs"),
            ("0006", "0005", "verification_evidence"),
        ):
            source = next(MIGRATIONS.glob(f"{revision}_*.py")).read_text(encoding="utf-8")
            self.assertIn(f'revision: str = "{revision}"', source)
            self.assertIn(f'down_revision: Union[str, None] = "{down}"', source)
            self.assertIn(f'op.create_table(\n        "{table}"', source)
            # Additive only: no migration may touch existing data.
            self.assertNotIn("op.drop_table", source.split("def downgrade")[0])
            self.assertNotIn("op.execute", source.split("def downgrade")[0])

    def test_no_migration_drops_or_alters_task_log_or_approvals(self) -> None:
        for path in MIGRATIONS.glob("000[456]_*.py"):
            upgrade = path.read_text(encoding="utf-8").split("def downgrade")[0]
            self.assertNotIn('drop_table("task_log")', upgrade)
            self.assertNotIn('drop_table("approvals")', upgrade)
            self.assertNotIn('alter_column("approvals"', upgrade)

    def test_task_log_model_is_retained_as_history(self) -> None:
        self.assertEqual(TaskLog.__tablename__, "task_log")


class TableSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = create_engine(f"sqlite:///{self.tmp.name}/audit.db")
        Base.metadata.create_all(self.engine)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.engine.dispose)

    def test_tool_runs_has_the_required_columns(self) -> None:
        columns = {c["name"] for c in inspect(self.engine).get_columns("tool_runs")}
        for name in (
            "id", "session_id", "turn_id", "tool_name", "request_json", "request_hash",
            "risk_level", "approval_id", "started_at", "finished_at", "status",
            "error_code", "user_visible_summary",
        ):
            self.assertIn(name, columns)

    def test_verification_evidence_has_the_required_columns(self) -> None:
        columns = {c["name"] for c in inspect(self.engine).get_columns("verification_evidence")}
        for name in (
            "id", "tool_run_id", "approval_id", "verifier_name", "expected_json",
            "observed_json", "verdict", "evidence_text", "created_at",
        ):
            self.assertIn(name, columns)

    def _check_sql(self, model) -> str:  # noqa: ANN001
        from sqlalchemy import CheckConstraint

        return " ".join(
            str(constraint.sqltext)
            for constraint in model.__table__.constraints
            if isinstance(constraint, CheckConstraint)
        )

    def test_status_vocabulary_is_constrained(self) -> None:
        sql = self._check_sql(ToolRun)
        for status in ("success", "failed", "blocked", "requires_approval", "unknown"):
            self.assertIn(status, sql)

    def test_verdict_vocabulary_is_constrained(self) -> None:
        sql = self._check_sql(VerificationEvidence)
        for verdict in ("verified", "failed", "uncertain", "skipped"):
            self.assertIn(verdict, sql)

    def test_an_invalid_status_is_rejected_by_the_database(self) -> None:
        from sqlalchemy.exc import IntegrityError

        Session = sessionmaker(bind=self.engine)
        with Session() as db:
            db.add(
                ToolRun(
                    tool_name="gmail_read",
                    request_json="{}",
                    request_hash="h",
                    risk_level="L0_OBSERVE",
                    requires_approval=False,
                    status="totally-made-up",
                )
            )
            with self.assertRaises(IntegrityError):
                db.commit()


class AuditWritingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = create_engine(f"sqlite:///{self.tmp.name}/audit.db")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        # Never let an audit test touch the real ao.db.
        self.patch = patch.object(audit_service, "SessionLocal", self.Session)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.engine.dispose)

    def _rows(self) -> list[ToolRun]:
        with self.Session() as db:
            return db.query(ToolRun).order_by(ToolRun.id).all()

    def _decision(self, tool: str, **arguments: object) -> brain_agent.BrainDecision:
        return brain_agent.BrainDecision(
            mode="tool", tool=tool, confidence=0.95, arguments=arguments, reason="t"
        )

    def test_a_read_is_recorded(self) -> None:
        result = OrchestratorResult(
            reply="Two emails.", action_type="gmail_summary", memory_content="Two emails."
        )
        with patch.object(tool_execution, "execute_gmail_read", return_value=result):
            tool_executor.execute(
                self._decision("gmail_read"),
                "check my email",
                session_id="sess-a",
                turn_id="turn-a",
            )
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].tool_name, "gmail_read")
        self.assertEqual(rows[0].status, "success")
        self.assertEqual(rows[0].session_id, "sess-a")
        self.assertEqual(rows[0].turn_id, "turn-a")
        self.assertEqual(rows[0].risk_level, "L0_OBSERVE")
        self.assertFalse(rows[0].requires_approval)
        self.assertIsNotNone(rows[0].finished_at)

    def test_an_approval_proposal_is_recorded_as_requires_approval(self) -> None:
        result = OrchestratorResult(
            reply="Drafted.",
            action_type="approval_required",
            memory_content="Drafted.",
            approval={"id": 42, "task_type": "gmail_compose"},
        )
        with patch.object(tool_execution, "execute_gmail_compose", return_value=result):
            tool_executor.execute(
                self._decision("gmail_compose", recipient_hint="rahul"), "email rahul"
            )
        row = self._rows()[0]
        self.assertEqual(row.status, "requires_approval")
        self.assertEqual(row.approval_id, 42)
        self.assertTrue(row.requires_approval)
        self.assertEqual(row.risk_level, "L3_EXTERNAL_WRITE")

    def test_a_failure_is_recorded(self) -> None:
        result = OrchestratorResult(
            reply="Gmail is down.", action_type="error", memory_content="Gmail is down."
        )
        with patch.object(tool_execution, "execute_gmail_read", return_value=result):
            tool_executor.execute(self._decision("gmail_read"), "check my email")
        self.assertEqual(self._rows()[0].status, "failed")

    def test_an_executor_exception_is_recorded_and_contained(self) -> None:
        with patch.object(
            tool_execution, "execute_gmail_read", side_effect=RuntimeError("boom")
        ):
            result = tool_executor.execute(self._decision("gmail_read"), "check my email")
        row = self._rows()[0]
        self.assertEqual(row.status, "failed")
        self.assertEqual(row.error_code, "RuntimeError")
        self.assertEqual(result.action_type, "error")

    def test_invalid_arguments_are_recorded_as_blocked(self) -> None:
        # A REQUIRED field violation (not a droppable optional) is what blocks.
        tool_executor.execute(self._decision("gmail_compose"), "email someone")
        row = self._rows()[0]
        self.assertEqual(row.status, "blocked")
        self.assertEqual(row.error_code, "invalid_arguments")

    def test_an_unsupported_tool_is_recorded_as_blocked(self) -> None:
        tool_executor.execute(self._decision("filesystem_delete"), "delete everything")
        row = self._rows()[0]
        self.assertEqual(row.status, "blocked")
        self.assertEqual(row.error_code, "unsupported_tool")

    def test_request_json_is_sanitized_and_excludes_raw_text(self) -> None:
        secret_body = "my bank password is hunter2 and here is a long private note"
        result = OrchestratorResult(
            reply="Drafted.",
            action_type="approval_required",
            memory_content="Drafted.",
            approval={"id": 1},
        )
        with patch.object(tool_execution, "execute_gmail_compose", return_value=result):
            tool_executor.execute(
                self._decision("gmail_compose", recipient_hint="rahul", body=secret_body),
                f"send rahul this: {secret_body}",
            )
        stored = json.loads(self._rows()[0].request_json)
        self.assertNotIn("raw_message", stored)
        self.assertIn("raw_message_chars", stored)
        self.assertNotIn(secret_body, self._rows()[0].request_json)
        self.assertIn("sha256:", stored["body"])

    def test_request_hash_is_stable_and_recorded(self) -> None:
        result = OrchestratorResult(
            reply="ok", action_type="gmail_summary", memory_content="ok"
        )
        with patch.object(tool_execution, "execute_gmail_read", return_value=result):
            for _ in range(2):
                tool_executor.execute(self._decision("gmail_read"), "check my email")
        rows = self._rows()
        self.assertEqual(rows[0].request_hash, rows[1].request_hash)
        self.assertTrue(rows[0].request_hash)

    def test_audit_failure_never_breaks_the_turn(self) -> None:
        result = OrchestratorResult(
            reply="Two emails.", action_type="gmail_summary", memory_content="Two emails."
        )
        with patch.object(
            audit_service, "SessionLocal", side_effect=RuntimeError("db gone")
        ), patch.object(tool_execution, "execute_gmail_read", return_value=result):
            outcome = tool_executor.execute(self._decision("gmail_read"), "check my email")
        self.assertEqual(outcome.reply, "Two emails.")

    def test_every_registered_capability_produces_a_row(self) -> None:
        cases = {
            "gmail_read": ({}, "execute_gmail_read"),
            "gmail_compose": ({"recipient_hint": "r"}, "execute_gmail_compose"),
            "gmail_reply": ({}, "execute_gmail_reply"),
            "calendar_read": ({}, "execute_calendar_read"),
            "calendar_create": ({"title": "T"}, "execute_calendar_create"),
            "cross_tool_read": ({}, "execute_cross_tool_read"),
        }
        approval = {"id": 1}
        for tool, (arguments, executor_name) in cases.items():
            stub = OrchestratorResult(
                reply="done",
                action_type="approval_required" if tool in {"gmail_compose", "gmail_reply", "calendar_create"} else "task_complete",
                memory_content="done",
                approval=approval if tool in {"gmail_compose", "gmail_reply", "calendar_create"} else None,
            )
            with patch.object(tool_execution, executor_name, return_value=stub):
                tool_executor.execute(self._decision(tool, **arguments), "do it")
        self.assertEqual(
            {row.tool_name for row in self._rows()}, set(cases), "every capability must be audited"
        )


class VerificationEvidenceWritingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = create_engine(f"sqlite:///{self.tmp.name}/audit.db")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.patch = patch.object(audit_service, "SessionLocal", self.Session)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.engine.dispose)

    def test_a_verdict_is_persisted_with_its_evidence(self) -> None:
        row_id = audit_service.record_verification(
            verifier_name="GmailSendVerifier",
            verdict="verified",
            tool_run_id=5,
            approval_id=7,
            expected={"recipient": "a@b.com"},
            observed={"recipient": "a@b.com", "message_id": "m1"},
            evidence_text="matched recipient and subject in Sent",
        )
        self.assertIsNotNone(row_id)
        with self.Session() as db:
            row = db.query(VerificationEvidence).one()
        self.assertEqual(row.verifier_name, "GmailSendVerifier")
        self.assertEqual(row.verdict, "verified")
        self.assertEqual(row.approval_id, 7)
        self.assertIn("a@b.com", row.expected_json or "")

    def test_all_four_verdicts_are_accepted(self) -> None:
        for verdict in ("verified", "failed", "uncertain", "skipped"):
            self.assertIsNotNone(
                audit_service.record_verification(verifier_name="V", verdict=verdict)
            )

    def test_status_mapping(self) -> None:
        self.assertEqual(audit_service.status_for_result("approval_required"), "requires_approval")
        self.assertEqual(audit_service.status_for_result("error"), "failed")
        self.assertEqual(audit_service.status_for_result("clarification_required"), "blocked")
        self.assertEqual(audit_service.status_for_result("gmail_summary"), "success")
        self.assertEqual(audit_service.status_for_result(None), "unknown")


class LiveDatabaseStateTests(unittest.TestCase):
    """Read-only assertions about the real ao.db after migrations 0005/0006.

    These were originally pinned to the exact row counts observed at migration
    time (722/254/8). That was a test-design defect: messages, approvals,
    tool_runs and verification_evidence are LIVE user data that grows every time
    Bunnelby is used, so the assertions failed as soon as the user actually ran
    it. The migration guarantees they were meant to protect are structural and
    monotonic, and are expressed that way here.

    The migration-time snapshot itself remains verified exactly, in the
    pre/post fingerprints recorded under .ao-backups/part10.2-phase*/.
    """

    # Row counts observed immediately after migrations 0004-0006 completed.
    MIGRATION_BASELINE = {"messages": 722, "task_log": 254, "approvals": 8}

    def test_live_database_has_the_new_tables_and_preserved_history(self) -> None:
        import sqlite3

        db = REPO_ROOT / "database/ao.db"
        if not db.exists():
            self.skipTest("live database not present")
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            for required in ("tool_runs", "verification_evidence", "task_log", "messages", "approvals", "user_facts"):
                self.assertIn(required, tables, f"{required} must exist")
            self.assertEqual(
                con.execute("select version_num from alembic_version").fetchone()[0], "0007"
            )

            # task_log is retained as frozen history: nothing writes to it any
            # more, so its count must be exactly the legacy value.
            self.assertEqual(
                con.execute("select count(*) from task_log").fetchone()[0],
                self.MIGRATION_BASELINE["task_log"],
                "task_log is dead-router history and must never gain or lose rows",
            )

            # Live tables may only ever grow. A count BELOW the migration
            # baseline would mean history was destroyed.
            for table in ("messages", "approvals"):
                count = con.execute(f"select count(*) from {table}").fetchone()[0]
                self.assertGreaterEqual(
                    count,
                    self.MIGRATION_BASELINE[table],
                    f"{table} dropped below its post-migration baseline",
                )
        finally:
            con.close()

    def test_every_message_carries_session_identity(self) -> None:
        """Phase D backfill plus live writes must leave no unattributed rows."""
        import sqlite3

        db = REPO_ROOT / "database/ao.db"
        if not db.exists():
            self.skipTest("live database not present")
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            orphans = con.execute(
                "select count(*) from messages where session_id is null or turn_id is null"
            ).fetchone()[0]
            self.assertEqual(orphans, 0, "every message row needs session_id and turn_id")
        finally:
            con.close()

    def test_live_audit_rows_use_only_declared_vocabularies(self) -> None:
        """Production audit rows must satisfy the same constraints as tests."""
        import sqlite3

        db = REPO_ROOT / "database/ao.db"
        if not db.exists():
            self.skipTest("live database not present")
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            statuses = {r[0] for r in con.execute("select distinct status from tool_runs")}
            self.assertLessEqual(
                statuses,
                {"success", "failed", "blocked", "requires_approval", "unknown"},
                f"unexpected tool_runs status in live data: {statuses}",
            )
            verdicts = {r[0] for r in con.execute("select distinct verdict from verification_evidence")}
            self.assertLessEqual(
                verdicts,
                {"verified", "failed", "uncertain", "skipped"},
                f"unexpected verification verdict in live data: {verdicts}",
            )
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
