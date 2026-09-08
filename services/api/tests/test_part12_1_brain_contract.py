"""Part 12.1: the Brain -> typed-request contract, and the verifier transport.

WHY THIS FILE EXISTS
--------------------
Two defects reached real hardware that the controller tests could not have
caught, because both live at an INTEGRATION seam rather than inside the desktop
package:

1. The user typed "Open Notepad" in the Electron UI and nothing happened. The
   live Brain chose the right tool and the right target, and emitted
   action="open" -- because the provider response schema flattened
   DesktopControlRequest's Literal to a bare {"type": "STRING"} and never told
   the model the vocabulary. The typed boundary rejected "open" correctly, so
   the turn failed closed to "I need a bit more detail before I can do that
   safely."

2. Notepad physically opened, the user saw "Notepad is open.", and the ledger
   logged `desktop_control verdict=failed:`. The verifier read DesktopOutcome
   attributes off an OrchestratorResult, so `status` defaulted to "failed".

The existing routing test built a request by hand with the canonical token
already in place, which is precisely the step the live provider got wrong.
These tests exercise the seam instead. No network call is made.
"""

from __future__ import annotations

import unittest

from services.api.app import tool_executor  # noqa: F401  (registers capabilities)
from services.api.app.brain_agent import (
    _extract_json_object,
    _parse_decision,
    decision_response_schema,
    tool_catalog_section,
)
from services.api.app.desktop.models import DesktopAction, DesktopOutcome, WindowInfo
from services.api.app.orchestrator import OrchestratorResult
from services.api.app.tool_requests import (
    DesktopControlRequest,
    ToolRequestValidationError,
    build_request,
    canonical_desktop_action,
)
from services.api.app.verification_service import verify_desktop_control

CANONICAL_ACTIONS = (
    "list_windows",
    "inspect_window",
    "find_control",
    "open_app",
    "focus_app",
    "close_app",
    "safe_shortcut",
)


def desktop_request(**arguments) -> DesktopControlRequest:
    return build_request("desktop_control", "Open Notepad.", arguments)


def transported(outcome: DesktopOutcome, *, action_type: str = "desktop_control"):
    """Exactly what execute_desktop_control hands back to the executor."""
    return OrchestratorResult(
        reply="Notepad is open.",
        action_type=action_type,
        memory_content="",
        spoken_reply="Notepad is open.",
        spoken_metadata=outcome.audit_payload(),
    )


# --------------------------------------------------------------------------- #
# 1: the provider must be told the vocabulary
# --------------------------------------------------------------------------- #


class ProviderSchemaContractTests(unittest.TestCase):
    def test_1_response_schema_exposes_the_canonical_desktop_actions(self) -> None:
        """The defect: this property was published as a bare STRING, so nothing
        in the contract said `open_app` was the only legal spelling."""
        action = decision_response_schema()["properties"]["arguments"]["properties"][
            "action"
        ]
        self.assertEqual(action["type"], "STRING")
        self.assertEqual(list(action.get("enum", ())), list(CANONICAL_ACTIONS))

    def test_the_enum_matches_the_request_model_exactly(self) -> None:
        """The schema must not drift from the type it is meant to describe."""
        published = decision_response_schema()["properties"]["arguments"][
            "properties"
        ]["action"]["enum"]
        model = DesktopControlRequest.model_json_schema()["properties"]["action"]["enum"]
        self.assertEqual(list(published), list(model))

    def test_the_enum_matches_the_controller_action_enum(self) -> None:
        self.assertEqual(
            set(CANONICAL_ACTIONS), {member.value for member in DesktopAction}
        )

    def test_a_free_text_argument_is_not_given_an_enum(self) -> None:
        """Only closed vocabularies are constrained; `target` stays open so the
        registry -- not the schema -- remains the authority on app names."""
        properties = decision_response_schema()["properties"]["arguments"]["properties"]
        self.assertNotIn("enum", properties["target"])

    def test_a_shared_argument_name_keeps_its_enum_only_when_all_agree(self) -> None:
        """`freshness` is declared by two capabilities with the SAME vocabulary,
        so it survives. A future collision with differing vocabularies must
        degrade to a plain STRING rather than over-constrain either caller."""
        from services.api.app.brain_agent import _string_enum_members

        properties = decision_response_schema()["properties"]["arguments"]["properties"]
        self.assertEqual(
            list(properties["freshness"]["enum"]), ["cached_ok", "fresh_required"]
        )
        self.assertIsNone(_string_enum_members({"type": "string"}))
        self.assertIsNone(_string_enum_members({"enum": ["ok", ""]}), "empty member")
        self.assertIsNone(_string_enum_members({"enum": [1, 2]}), "non-string member")

    def test_the_prompt_catalog_states_the_action_vocabulary(self) -> None:
        catalog = tool_catalog_section()
        self.assertIn(
            "action must be exactly one of: " + ", ".join(CANONICAL_ACTIONS), catalog
        )

    def test_every_published_enum_is_provider_safe(self) -> None:
        """Gemini rejects an empty-string enum member with HTTP 400."""
        properties = decision_response_schema()["properties"]["arguments"]["properties"]
        for name, spec in properties.items():
            members = spec.get("enum")
            if members is None:
                continue
            with self.subTest(argument=name):
                self.assertEqual(spec["type"], "STRING")
                self.assertTrue(all(isinstance(m, str) and m.strip() for m in members))
                self.assertEqual(len(members), len(set(members)), "duplicate member")


# --------------------------------------------------------------------------- #
# 2-6: the closed normalization layer
# --------------------------------------------------------------------------- #


class ActionNormalizationTests(unittest.TestCase):
    def test_2_open_normalizes_only_to_open_app(self) -> None:
        """The exact live failure: Gemini emitted action="open"."""
        request = desktop_request(action="open", target="notepad")
        self.assertEqual(request.action, "open_app")
        self.assertEqual(request.target, "notepad")

    def test_3_launch_and_start_normalize_to_open_app(self) -> None:
        for token in ("launch", "start", "OPEN", " Open ", "open-app"):
            with self.subTest(token=token):
                self.assertEqual(
                    desktop_request(action=token, target="notepad").action, "open_app"
                )

    def test_4_switch_and_focus_normalize_to_focus_app(self) -> None:
        for token in ("switch", "switch_to", "switch to", "focus", "activate"):
            with self.subTest(token=token):
                self.assertEqual(
                    desktop_request(action=token, target="calculator").action,
                    "focus_app",
                )

    def test_5_close_normalizes_to_close_app(self) -> None:
        for token in ("close", "quit"):
            with self.subTest(token=token):
                self.assertEqual(
                    desktop_request(action=token, target="calculator").action,
                    "close_app",
                )

    def test_read_only_synonyms_normalize(self) -> None:
        self.assertEqual(desktop_request(action="list").action, "list_windows")
        self.assertEqual(
            desktop_request(action="inspect", target="calculator").action,
            "inspect_window",
        )
        self.assertEqual(
            desktop_request(action="find", target="calculator").action, "find_control"
        )

    def test_6_an_unknown_action_still_fails_closed(self) -> None:
        for token in ("", "delete_everything", "kill", "run", "execute", "type_text"):
            with self.subTest(token=token):
                with self.assertRaises(ToolRequestValidationError):
                    desktop_request(action=token, target="notepad")

    def test_run_and_execute_are_deliberately_not_aliases(self) -> None:
        """A shell-adjacent verb must never acquire a foothold here."""
        self.assertEqual(canonical_desktop_action("run"), "run")
        self.assertEqual(canonical_desktop_action("execute"), "execute")
        self.assertEqual(canonical_desktop_action("powershell"), "powershell")

    def test_normalization_is_exact_never_fuzzy(self) -> None:
        for token in ("opennn", "op", "open the app", "openapp", "close all windows"):
            with self.subTest(token=token):
                self.assertEqual(canonical_desktop_action(token), token.replace(" ", "_"))
                with self.assertRaises(ToolRequestValidationError):
                    desktop_request(action=token, target="notepad")

    def test_9_a_canonical_request_is_unchanged(self) -> None:
        for action in CANONICAL_ACTIONS:
            with self.subTest(action=action):
                arguments = {"action": action}
                if action not in ("list_windows", "safe_shortcut"):
                    arguments["target"] = "notepad"
                if action == "safe_shortcut":
                    arguments["shortcut"] = "show_desktop"
                self.assertEqual(desktop_request(**arguments).action, action)

    def test_normalization_never_touches_the_target(self) -> None:
        """It may refine an action inside a chosen capability. Nothing else."""
        self.assertEqual(desktop_request(action="open", target="notepad").target, "notepad")

    def test_7_a_path_shaped_target_is_still_rejected(self) -> None:
        for target in (
            "C:/Windows/System32/cmd.exe",
            "C:\\Windows\\System32\\cmd.exe",
            "./notepad.exe",
            "notepad.exe",
        ):
            with self.subTest(target=target):
                with self.assertRaises(ToolRequestValidationError):
                    desktop_request(action="open", target=target)

    def test_8_a_command_shaped_target_is_still_rejected(self) -> None:
        for target in (
            "powershell -Command Get-Process",
            "cmd /c dir",
            "notepad & calc",
            "app --quiet",
        ):
            with self.subTest(target=target):
                with self.assertRaises(ToolRequestValidationError):
                    desktop_request(action="open", target=target)

    def test_normalization_cannot_select_the_capability(self) -> None:
        """Structural: the alias map is a validator on DesktopControlRequest, so
        it only ever runs once desktop_control has ALREADY been chosen."""
        from services.api.app import tool_requests

        source = tool_requests.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        anchor = text.index("_DESKTOP_ACTION_ALIASES")
        self.assertIn("DesktopControlRequest", text[anchor:])
        for other in ("gmail_read", "calendar_read", "file_search"):
            self.assertNotIn(
                other,
                text[anchor : text.index("class DesktopControlRequest")],
                "the alias map must not reach another capability",
            )


# --------------------------------------------------------------------------- #
# Integration: the exact live Brain JSON, through the typed boundary
# --------------------------------------------------------------------------- #


class BrainEnvelopeIntegrationTests(unittest.TestCase):
    """The real provider payload observed on hardware, parsed offline."""

    LIVE_PAYLOAD = """{
      "mode": "tool",
      "tool": "desktop_control",
      "confidence": 0.99,
      "arguments": {"action": "open", "target": "notepad"},
      "reply": "Opening Notepad.",
      "spoken_reply": "Opening Notepad.",
      "reason": "The user explicitly asked to open Notepad.",
      "response_policy": "both"
    }"""

    def test_the_live_failure_payload_now_produces_a_canonical_request(self) -> None:
        decision = _parse_decision(self.LIVE_PAYLOAD, "Open Notepad")
        self.assertEqual(decision.mode, "tool")
        self.assertEqual(decision.tool, "desktop_control")

        request = build_request(
            decision.tool, "Open Notepad", dict(decision.arguments)
        )
        self.assertEqual(request.action, "open_app")
        self.assertEqual(request.target, "notepad")

    def test_the_same_envelope_with_a_path_target_still_refuses(self) -> None:
        payload = self.LIVE_PAYLOAD.replace(
            '"target": "notepad"', '"target": "C:/Windows/System32/cmd.exe"'
        )
        decision = _parse_decision(payload, "Open cmd")
        with self.assertRaises(ToolRequestValidationError):
            build_request(decision.tool, "Open cmd", dict(decision.arguments))

    def test_an_envelope_naming_an_unregistered_tool_is_discarded(self) -> None:
        payload = self.LIVE_PAYLOAD.replace(
            '"tool": "desktop_control"', '"tool": "shell_exec"'
        )
        decision = _parse_decision(payload, "run something")
        self.assertIsNone(decision.tool)
        self.assertEqual(decision.mode, "clarify")


# --------------------------------------------------------------------------- #
# 10-16: verifier evidence transport
# --------------------------------------------------------------------------- #


class VerifierTransportTests(unittest.TestCase):
    def _request(self, action="open_app", target="notepad"):
        return desktop_request(action=action, target=target)

    def test_10_the_verifier_reads_the_orchestrator_result_transport(self) -> None:
        """The defect: it read DesktopOutcome attributes off an
        OrchestratorResult, so status defaulted to "failed" every time."""
        outcome = DesktopOutcome(
            action=DesktopAction.OPEN_APP,
            status="succeeded",
            target="notepad",
            detail="Notepad is open.",
            windows=(WindowInfo(handle=1, title="Untitled - Notepad", pid=9, process_name="notepad.exe"),),
            evidence={"launched_pid": 4242, "verified": True},
        )
        verdict = verify_desktop_control(self._request(), transported(outcome))

        self.assertEqual(verdict.verdict, "verified")
        self.assertEqual(verdict.observed["status"], "succeeded")
        self.assertEqual(verdict.observed["window_count"], 1)
        self.assertTrue(verdict.observed["evidence_verified"])

    def test_11_a_succeeded_outcome_yields_verified(self) -> None:
        outcome = DesktopOutcome(
            action=DesktopAction.OPEN_APP, status="succeeded", target="notepad"
        )
        self.assertEqual(
            verify_desktop_control(self._request(), transported(outcome)).verdict,
            "verified",
        )

    def test_12_a_foreground_denial_yields_uncertain(self) -> None:
        outcome = DesktopOutcome(
            action=DesktopAction.FOCUS_APP,
            status="unverified",
            target="calculator",
            error_code="foreground_denied",
            detail="Windows restricts focus changes from background processes.",
        )
        verdict = verify_desktop_control(
            self._request("focus_app", "calculator"), transported(outcome)
        )
        self.assertEqual(verdict.verdict, "uncertain")
        self.assertEqual(verdict.observed["error_code"], "foreground_denied")

    def test_13_needs_clarification_yields_uncertain(self) -> None:
        outcome = DesktopOutcome(
            action=DesktopAction.CLOSE_APP,
            status="needs_clarification",
            target="notepad",
            error_code="confirmation_required",
        )
        self.assertEqual(
            verify_desktop_control(
                self._request("close_app", "notepad"), transported(outcome)
            ).verdict,
            "uncertain",
        )

    def test_14_failed_and_blocked_yield_failed(self) -> None:
        for status in ("failed", "blocked"):
            with self.subTest(status=status):
                outcome = DesktopOutcome(
                    action=DesktopAction.CLOSE_APP, status=status, target="notepad"
                )
                self.assertEqual(
                    verify_desktop_control(
                        self._request("close_app", "notepad"), transported(outcome)
                    ).verdict,
                    "failed",
                )

    def test_15_missing_or_malformed_evidence_never_verifies(self) -> None:
        empty = OrchestratorResult(
            reply="Notepad is open.",
            action_type="desktop_control",
            memory_content="",
            spoken_metadata={},
        )
        self.assertEqual(
            verify_desktop_control(self._request(), empty).verdict, "uncertain"
        )

        for bogus in ("", "ok", "success", "done", None, 1, True):
            with self.subTest(status=bogus):
                result = OrchestratorResult(
                    reply="Notepad is open.",
                    action_type="desktop_control",
                    memory_content="",
                    spoken_metadata={"action": "open_app", "status": bogus},
                )
                self.assertNotEqual(
                    verify_desktop_control(self._request(), result).verdict, "verified"
                )

    def test_15b_a_non_desktop_envelope_fails_closed(self) -> None:
        outcome = DesktopOutcome(
            action=DesktopAction.OPEN_APP, status="succeeded", target="notepad"
        )
        result = transported(outcome, action_type="general_answer")
        self.assertEqual(
            verify_desktop_control(self._request(), result).verdict, "failed"
        )

    def test_16_success_sounding_prose_is_never_treated_as_proof(self) -> None:
        """"Notepad is open." is a sentence, not an observation."""
        result = OrchestratorResult(
            reply="Notepad is open. Everything succeeded and was verified.",
            action_type="desktop_control",
            memory_content="Notepad is open.",
            spoken_reply="Notepad is open.",
            spoken_metadata={},
        )
        verdict = verify_desktop_control(self._request(), result)
        self.assertEqual(verdict.verdict, "uncertain")
        self.assertNotIn("Notepad is open", verdict.evidence_text)

    def test_an_action_mismatch_is_a_failure_not_an_uncertainty(self) -> None:
        """If the system did something other than what was asked, that is a
        correctness failure, not an unproven success."""
        outcome = DesktopOutcome(
            action=DesktopAction.CLOSE_APP, status="succeeded", target="notepad"
        )
        verdict = verify_desktop_control(self._request("open_app"), transported(outcome))
        self.assertEqual(verdict.verdict, "failed")
        self.assertIn("did not match requested", verdict.evidence_text)

    def test_the_audit_projection_carries_no_window_titles(self) -> None:
        """Privacy: the ledger records counts and codes, never the screen."""
        outcome = DesktopOutcome(
            action=DesktopAction.LIST_WINDOWS,
            status="succeeded",
            windows=(
                WindowInfo(handle=1, title="SECRET-PROJECT-Q3.docx", pid=9, process_name="winword.exe"),
            ),
        )
        verdict = verify_desktop_control(
            desktop_request(action="list_windows"), transported(outcome)
        )
        blob = repr(verdict.observed) + verdict.evidence_text
        self.assertNotIn("SECRET-PROJECT", blob)
        self.assertEqual(verdict.observed["window_count"], 1)


# --------------------------------------------------------------------------- #
# _extract_json_object: markdown-fence stripping is now deterministic string
# ops (strip/slice/casefold), not regex, to remove a polynomial-backtracking
# CodeQL finding. These pin the exact behaviour the old regex produced.
# --------------------------------------------------------------------------- #


class ExtractJsonObjectTests(unittest.TestCase):
    def test_plain_json_object(self) -> None:
        self.assertEqual(_extract_json_object('{"a": 1}'), {"a": 1})

    def test_json_fenced_object(self) -> None:
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(_extract_json_object(text), {"a": 1})

    def test_json_fenced_object_case_insensitive_label(self) -> None:
        text = '```JSON\n{"a": 1}\n```'
        self.assertEqual(_extract_json_object(text), {"a": 1})

    def test_generic_fenced_object(self) -> None:
        text = '```\n{"a": 1}\n```'
        self.assertEqual(_extract_json_object(text), {"a": 1})

    def test_fenced_object_with_extra_surrounding_text_still_recovers_via_brace_scan(self) -> None:
        text = 'Here you go:\n```json\n{"a": 1}\n```\nThanks!'
        self.assertEqual(_extract_json_object(text), {"a": 1})

    def test_malformed_fenced_content_falls_through_to_none(self) -> None:
        for text in ("```json\nnot json\n```", "```", "```json", "not json at all", ""):
            with self.subTest(text=text):
                self.assertIsNone(_extract_json_object(text))

    def test_large_whitespace_and_fence_shaped_adversarial_input_does_not_hang(self) -> None:
        """Regression for the ReDoS finding: this used to be O(n^2) via
        `re.sub(r"\\s*```$", ...)` retried at every start position."""
        import time

        adversarial = "```json" + (" " * 200_000) + "not a fence close"
        started = time.monotonic()
        result = _extract_json_object(adversarial)
        elapsed = time.monotonic() - started
        self.assertIsNone(result)
        self.assertLess(elapsed, 1.0, "fence stripping must stay linear-time on adversarial input")


if __name__ == "__main__":
    unittest.main()
