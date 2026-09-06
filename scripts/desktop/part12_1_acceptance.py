"""Part 12.1 live acceptance run. Operator-supervised, non-destructive.

Runs the Part 12.1 acceptance checks against the REAL desktop and prints what
was verified for each. It exists because unit tests prove the policy and the
Windows integration tests prove the reads, but neither proves that "Open
Notepad" actually puts Notepad on screen on this machine.

SAFETY
------
  * Only registered applications are touched.
  * Notepad is opened and focused but NEVER closed: it may hold unsaved text,
    and Part 12.1's rule is that Bunnelby does not make that decision.
  * Calculator IS closed, because it is stateless -- that is the one close in
    the run, and it is graceful.
  * Nothing is forced, no process is terminated, no keystroke is sent.
  * Pass --read-only to skip every state-changing step.

Usage:
    python scripts/desktop/part12_1_acceptance.py
    python scripts/desktop/part12_1_acceptance.py --read-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.app.desktop.controller import DesktopController  # noqa: E402
from services.api.app.desktop.models import DesktopOutcome  # noqa: E402
from services.api.app.desktop.ui_automation import default_provider  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
INFO = "INFO"

#: Failures are classified because the two kinds mean different things. A
#: DISCOVERY failure is a bug: Bunnelby could not see an app that is running.
#: A FOCUS-POLICY failure is Windows exercising its documented right to refuse
#: a foreground change to a background process -- correct, honest behaviour
#: that must NOT be "fixed" by weakening verification.
DISCOVERY_CODES = ("application_not_running", "window_not_found")
FOCUS_POLICY_CODES = ("foreground_denied",)

discovery_failures: list[str] = []
focus_policy_failures: list[str] = []
other_failures: list[str] = []


def report(label: str, outcome: DesktopOutcome, expect: tuple[str, ...] = ("succeeded",)) -> bool:
    ok = outcome.status in expect
    mark = PASS if ok else FAIL
    print(f"  [{mark}] {label}")
    print(f"         status={outcome.status} latency={outcome.latency_ms:.0f}ms")
    if outcome.detail:
        print(f"         {outcome.detail}")
    if outcome.evidence:
        print(f"         evidence={dict(outcome.evidence)}")
    if not ok:
        entry = f"{label} [{outcome.status}/{outcome.error_code}]"
        if outcome.error_code in DISCOVERY_CODES:
            discovery_failures.append(entry)
        elif outcome.error_code in FOCUS_POLICY_CODES:
            focus_policy_failures.append(entry)
        else:
            other_failures.append(entry)
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Part 12.1 desktop acceptance run.")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Observe only: no launch, focus, or close.",
    )
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        print("This acceptance run requires Windows.")
        return 2

    controller = DesktopController()
    results: list[bool] = []

    print("=" * 70)
    print("BUNNELBY PART 12.1 - DESKTOP CONTROL ACCEPTANCE")
    print("=" * 70)
    print(f"UI Automation provider available: {default_provider().is_available()}")
    print()

    print("TEST 0: enumerate windows (read-only)")
    results.append(report("list_windows", controller.list_windows()))
    print()

    if args.read_only:
        print("Read-only mode: skipping state-changing checks.")
    else:
        print("TEST A: open Notepad")
        results.append(report("open_app notepad", controller.open_app("notepad")))
        print()

        print("TEST B: focus Notepad")
        results.append(report("focus_app notepad", controller.focus_app("notepad")))
        print()

        print("TEST C: open Calculator")
        results.append(report("open_app calculator", controller.open_app("calculator")))
        print()

        print("TEST D: switch between them")
        results.append(report("focus_app notepad", controller.focus_app("notepad")))
        results.append(report("focus_app calculator", controller.focus_app("calculator")))
        print()

        print("TEST E: inspect Calculator's UI (bounded)")
        results.append(
            report("inspect_window calculator", controller.inspect_window("calculator"))
        )
        print()

        print("TEST F: find buttons in Calculator")
        results.append(
            report(
                "find_control calculator/Button",
                controller.find_control("calculator", control_type="Button"),
            )
        )
        print()

        print("TEST G: close Calculator gracefully")
        results.append(report("close_app calculator", controller.close_app("calculator")))
        print()

        print("TEST H: Notepad is deliberately LEFT OPEN (may hold unsaved text)")
        print(f"  [{INFO}] no close attempted for notepad, by policy")
        print()

    print("NEGATIVE CHECKS (must all refuse)")
    results.append(
        report(
            "unknown app is refused",
            controller.open_app("definitely_not_a_real_app"),
            expect=("blocked",),
        )
    )
    results.append(
        report(
            "closing the Windows shell is refused",
            controller.close_app("file_explorer"),
            expect=("blocked",),
        )
    )
    results.append(
        report(
            "Ctrl+Alt+Del is refused",
            controller.safe_shortcut("ctrl_alt_delete"),
            expect=("blocked",),
        )
    )
    print()

    passed = sum(1 for item in results if item)
    print("=" * 70)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    print(f"  DISCOVERY failures    : {len(discovery_failures)}  (BUG if non-zero)")
    for item in discovery_failures:
        print(f"      - {item}")
    print(f"  FOCUS-POLICY failures : {len(focus_policy_failures)}  (Windows may refuse)")
    for item in focus_policy_failures:
        print(f"      - {item}")
    if other_failures:
        print(f"  OTHER failures        : {len(other_failures)}")
        for item in other_failures:
            print(f"      - {item}")
    print("No process was terminated. No keystroke was sent. Notepad left open.")
    print("=" * 70)
    # A focus refusal is legitimate Windows behaviour reported honestly, so it
    # does not fail the run. A discovery failure does.
    return 1 if (discovery_failures or other_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
