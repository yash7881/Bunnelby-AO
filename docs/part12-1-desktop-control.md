# Part 12.1 — Desktop Control Foundation

A **safe, bounded, deterministic** foundation for controlling the Windows
desktop. This milestone deliberately implements only control levels 1–3
(official API → native OS API → semantic shortcut). There is no coordinate
clicking and no vision-based control.

---

## 1. Pipeline

```
user message
  └─ brain_agent.decide()              selects desktop_control from the live catalog
       └─ tool_requests.build_request  DesktopControlRequest (extra="forbid", frozen)
            └─ tool_executor.execute   risk decision + tool_runs row
                 └─ tool_execution.execute_desktop_control
                      └─ desktop.controller.DesktopController
                           ├─ app_registry     alias  → reviewed launch argv
                           ├─ windows_api      ctypes user32/kernel32
                           ├─ shortcuts        allowlist → key codes
                           └─ ui_automation    bounded UIA read
                      └─ verification_service.verify_desktop_control
                 └─ evidence recorded, then the reply is produced
```

The reply is produced **after** verification, never before.

## 2. Supported actions

| Action | Risk | Changes state | Verifier |
|---|---|---|---|
| `list_windows` | L0 observe | no | n/a (bounded read) |
| `inspect_window` | L0 observe | no | n/a (bounded read) |
| `find_control` | L0 observe | no | n/a (bounded read) |
| `open_app` | L2 modify local | yes | a matching window must appear |
| `focus_app` | L2 modify local | yes | `GetForegroundWindow` must match |
| `close_app` | L2 modify local | yes | target windows must be gone |
| `safe_shortcut` | L1 safe control | yes | input delivery confirmed |

The capability is declared once at `RiskLevel.L2_MODIFY_LOCAL` /
`ApprovalPolicy.NEVER`, validated at import by `risk_policy.validate_declaration`.

## 3. Application registry

An LLM supplies an **alias**, never a path. `app_registry` is the only place an
alias becomes something executable, and every `launch_argv` is a human-written
literal. Resolution is exact — no fuzzy matching, no PATH search, no Start-menu
lookup.

| App id | Launch | Closable | Note |
|---|---|---|---|
| `notepad` | `notepad.exe` | yes | may hold unsaved work; graceful close only |
| `calculator` | `calc.exe` | yes | stateless |
| `file_explorer` | `explorer.exe` | **no** | Windows shell; system-critical |
| `settings` | `explorer.exe ms-settings:` | yes | packaged app |
| `edge` | `msedge.exe` | yes | browser *control* is Part 16 |
| `chrome` | `chrome.exe` | yes | browser *control* is Part 16 |
| `vscode` | `code.cmd` | **no** | unsaved editor buffers |
| `terminal` | `wt.exe` | **no** | may hold a running job |

Opening Windows Terminal opens a *window*. Bunnelby cannot type into it —
there is no text-entry action in Part 12.1 — so this is not an arbitrary shell.

### 3.1 Window identity: dedicated vs shared host

Attributing a window to an app is **graded**, because Windows itself is graded.
Packaged (Store/UWP) apps are hosted by `ApplicationFrameHost.exe`, which hosts
*every* packaged app — observed live on this hardware with an AFH-hosted
Calculator window and an AFH-hosted Settings window in the **same pid**.

| Evidence | Condition | Strong? | May close? |
|---|---|---|---|
| `DEDICATED_PROCESS` | owner is in `process_names` (exclusive to this app) | yes | yes |
| `SHARED_HOST_TITLE` | owner is in `shared_host_process_names` **and** the title corroborates | yes | yes |
| `TITLE_ONLY` | owner could not be read **and** the entry opts in | **no** | **never** |
| *(no match)* | owner is a readable, different process — whatever the title claims | — | — |

Two consequences worth stating plainly. A shared host is never listed in
`process_names`; the registry raises at import if the two sets overlap. And a
window title alone can describe a window but can never authorise closing it —
any program can name its window "Calculator".

### 3.2 Bounded rediscovery

A packaged app is re-hosted between its own process and the frame host during
its lifetime, and an enumeration taken mid-transition sees no window at all.
Observed on hardware: `open_app` verified Calculator, and a later single
snapshot reported it "not currently running" while both processes were alive.

Absence is therefore **confirmed, not assumed**: `focus_app`, `inspect_window`
and `close_app` poll for up to `DISCOVERY_GRACE_SECONDS` (1.5s) at the usual
0.1s interval through the same `wait_until` deadline helper as every other
wait. No fixed sleep, no retry counter, no unbounded loop. Still absent after
the grace → the existing truthful "not currently running" outcome, unchanged.

`open_app` deliberately does **not** pay this grace on its already-running
pre-check; its own 12s launch verification already covers the case.

## 4. Verification

`succeeded` requires observed evidence. Otherwise:

- **`unverified`** — the action was attempted but the end state could not be
  proven (launch produced no window; Windows refused the foreground change;
  close did not complete). Maps to verdict `uncertain`.
- **`needs_clarification`** — the app raised a confirmation dialog. Maps to
  `uncertain`. **Bunnelby never answers a save prompt.**
- **`blocked`** — policy refused before anything was attempted.
- **`failed`** — the action genuinely failed.

`unverified` is deliberately distinct from both success and failure.

## 5. Security boundaries

| Guarantee | Mechanism |
|---|---|
| No arbitrary executable | Alias-only `target`; validator rejects paths, flags and shell metacharacters; a rejected target fails the **turn** via a model-level validator |
| No shell | `subprocess.Popen(argv, shell=False)`; no `os.system`, no `PowerShell -Command` |
| No process termination | Backend exposes no kill; close is `PostMessage(WM_CLOSE)` only |
| No shell teardown | `explorer.exe` is `system_critical` and permanently not closable |
| No secure-desktop automation | `Ctrl+Alt+Del`, `Win+L` explicitly denied with a reason; also impossible from a normal-integrity process |
| No arbitrary keystrokes | `send()` accepts only a `Shortcut` object obtained from `resolve()` |
| No credential capture | UIA reads identity + state; **never** `CurrentValue`/`ValuePattern` |
| No identity spoofing by title | A readable, non-matching process is a definitive non-match; title-only evidence is weak and never authorises a close |
| No shared-host confusion | `ApplicationFrameHost` is declared a shared host, never a dedicated process; overlap raises at import |
| No keylogging | No `SetWindowsHookEx`, no mouse input |
| Prompt-injection resistant | UI text wrapped as `source_type="screen"` untrusted content; markers neutralised so a forged boundary cannot escape |
| Bounded reads | 40 windows, depth 4, 120 nodes, 120-char names, 160-char titles |
| Bounded waits | All polls have deadlines; no `while True` |
| No races | One re-entrant lock on state-changing actions; reads are unlocked |

## 6. Dependencies

- **Windows/process/focus/close:** stdlib `ctypes` only — **no new dependency**.
- **UI Automation:** `comtypes==1.4.8; sys_platform == "win32"` — Windows-only
  marker so non-Windows installs are unaffected, imported lazily, and the
  provider degrades to "unavailable" if missing.

## 7. Known limitations

1. **Focus can be refused.** Windows restricts foreground changes from
   background processes. Reported as `unverified`, never faked.
2. **Packaged apps** (Calculator, Settings) may be hosted by
   `ApplicationFrameHost.exe`, so their titles must corroborate identity (§3.1).
   A packaged app whose window title does not contain its hint would not be
   attributed to it while hosted — the conservative failure direction.
3. **Confirmation detection is title-based** and English-oriented; a localised
   or unusually-titled dialog may be missed, in which case the close reports
   `unverified` rather than success. It still never clicks anything.
4. **Shortcut effects are not asserted** — only delivery is. "Win+D minimised
   everything" has no single provable end state.
5. **`vscode` launches via `code.cmd`**, which must be on `PATH`.
6. **Multiple windows of one app**: `close_app` closes all matching windows;
   `focus_app` picks foreground-else-lowest-handle (stable, so verification is
   meaningful).

## 8. Manual acceptance

```powershell
cd C:\Users\Dhruv\Desktop\automation\AO-part12-1

# Observation only - safe to run any time
python scripts\desktop\part12_1_acceptance.py --read-only

# Full run: opens Notepad + Calculator, switches, closes Calculator only
python scripts\desktop\part12_1_acceptance.py
```

Notepad is intentionally left open by the full run.

## 9. Deferred

| Part | Deferred work |
|---|---|
| 12.2 | Reality Layer / automated incident repair |
| 13 | Runtime / orb expansion |
| 14 | Context Engine |
| 15 | Screen perception (vision, OCR-driven control) |
| 16 | Browser bridge |
| 18 | Permanent Memory V2 |
| 22 | Full bounded Windows controller (clicking, text entry) |
| 29 | Teach-me-once workflows |
