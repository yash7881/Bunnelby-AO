"""Voice conversation authority: when heard speech may become a command.

PRODUCT MODEL. The microphone stream and the wake detector are ALWAYS live for
as long as Bunnelby runs. There is no enable step and no mic-off state. What is
gated is CONVERSATION:

    STANDBY_WAKE  -- stream live, detector listening, speech is not a command
        | wake phrase (the only way out)
    ACTIVE_TURN   -- capture, STT, /chat, spoken reply
        | response complete
    FOLLOW_UP     -- <=10s to continue without repeating the wake phrase
        | timeout, or the consecutive-follow-up bound
    STANDBY_WAKE

These tests drive the REAL VoiceConversationAuthority and the REAL
VoiceSessionController through a fake turn loop. No microphone is opened, no
audio is played, and no wake recognition is exercised: a wake event is simply
assumed, which is the point -- the detector may emit, the authority decides
what may follow.
"""

from __future__ import annotations

import threading
import unittest

from services.api.app.voice_session import (
    ConversationLease,
    ConversationPhase,
    VoiceActivationDenied,
    VoiceConversationAuthority,
    VoiceSessionController,
    VoiceState,
)

FOLLOW_UP_SECONDS = 10.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakePlaybackGate:
    """Stands in for ExternalPlaybackGate."""

    def __init__(self, blocked: bool = False) -> None:
        self.blocked = blocked

    def is_blocked(self) -> bool:
        return self.blocked


class FakeVoiceRuntime:
    """A faithful model of the runtime turn loop, minus audio and network.

    Counts the four things that must be impossible without a lease, and refuses
    to advance a turn whose permission has lapsed -- exactly where the real
    runtime checks `conversation.may_capture_command()`.
    """

    def __init__(self, *, max_follow_ups: int = 2) -> None:
        self.clock = FakeClock()
        self.gate = FakePlaybackGate()
        self.authority = VoiceConversationAuthority(
            max_consecutive_follow_ups=max_follow_ups, clock=self.clock
        )
        self.controller = VoiceSessionController(
            follow_up_seconds=FOLLOW_UP_SECONDS, clock=self.clock
        )
        self.captures = 0
        self.stt_calls = 0
        self.chat_calls = 0
        self.tts_calls = 0
        self.events: list[str] = []

    # -- what the outside world does ---------------------------------------- #

    def ambient_speech(self, text: str = "some ordinary talking") -> None:
        """Someone speaks with no wake phrase. Nothing may happen."""
        if not self.authority.may_capture_command():
            self.events.append("ignored:no_lease")
            return
        self._run_turn(text)

    def wake_event(self, activation_id: str | None = None, transcript: str = "open notepad") -> None:
        """The existing detector emitted an activation."""
        try:
            self.authority.begin_wake_turn(activation_id)
        except VoiceActivationDenied as denial:
            self.events.append(f"rejected:{denial.reason}")
            return
        if self.controller.state is VoiceState.FOLLOW_UP:
            self.controller.transition(VoiceState.STANDBY, "wake during follow-up")
        self.controller.wake_detected()
        self.controller.begin_listening()
        self.events.append("wake_accepted")
        self._run_turn(transcript)

    def follow_up_speech(self, transcript: str = "give me an example") -> None:
        """The user continues without a wake phrase."""
        if self.controller.state is not VoiceState.FOLLOW_UP:
            self.events.append("ignored:not_in_follow_up")
            return
        if self.clock.now > (self.controller.follow_up_deadline or 0):
            self.events.append("ignored:deadline_passed")
            return
        if not self.authority.may_begin_follow_up(playback_blocked=self.gate.is_blocked()):
            self.events.append("ignored:follow_up_not_permitted")
            return
        try:
            self.authority.begin_follow_up_turn(playback_blocked=self.gate.is_blocked())
        except VoiceActivationDenied as denial:
            self.events.append(f"rejected:{denial.reason}")
            return
        self.controller.accept_follow_up_speech(self.clock.now)
        self.events.append("follow_up_accepted")
        self._run_turn(transcript)

    def wait(self, seconds: float) -> None:
        """Time passes with nobody speaking."""
        self.clock.advance(seconds)
        if self.controller.state is VoiceState.FOLLOW_UP and self.controller.expire_follow_up():
            self.authority.return_to_standby("follow-up window expired")
            self.events.append("follow_up_expired")

    # -- the turn itself ----------------------------------------------------- #

    def _run_turn(self, transcript: str) -> None:
        if not self.authority.may_capture_command():
            self.events.append("blocked:no_lease_at_capture")
            return
        self.captures += 1

        self.stt_calls += 1
        text = transcript.strip()
        if not text:
            self.events.append("empty_transcript")
            self.controller.transition(VoiceState.STANDBY, "empty transcript")
            self.authority.return_to_standby("empty transcript")
            return

        self.controller.utterance_completed(at=self.clock.now)
        self.controller.transcription_completed(at=self.clock.now)

        if not self.authority.may_capture_command():
            self.events.append("blocked:no_lease_at_dispatch")
            return
        self.chat_calls += 1

        # Assistant speaks, then the follow-up window opens on completion.
        self.controller.speaking_started(at=self.clock.now)
        self.gate.blocked = True
        self.tts_calls += 1
        self.clock.advance(1.0)
        self.gate.blocked = False
        self.controller.playback_completed(at=self.clock.now)
        self.authority.open_follow_up_window()
        self.events.append("response_complete")

    def totals(self) -> tuple[int, int, int, int]:
        return (self.captures, self.stt_calls, self.chat_calls, self.tts_calls)


# --------------------------------------------------------------------------- #
# 1-3: standby is wake-only, not deaf and not off
# --------------------------------------------------------------------------- #


class StartupAndStandbyTests(unittest.TestCase):
    def test_1_startup_is_standby_wake_with_no_enable_step(self) -> None:
        """Launch and speak the wake phrase: nothing to click, nothing to arm."""
        authority = VoiceConversationAuthority()
        self.assertIs(authority.phase, ConversationPhase.STANDBY_WAKE)
        self.assertFalse(authority.may_capture_command())

        # The very first wake activation is accepted with no preceding call.
        lease = authority.begin_wake_turn()
        self.assertEqual(lease.source, "wake")
        self.assertTrue(authority.may_capture_command())

    def test_1b_no_enable_api_exists_any_more(self) -> None:
        """The rejected mic-enable design must leave no dead surface behind."""
        authority = VoiceConversationAuthority()
        for removed in ("set_enabled", "is_enabled", "enabled"):
            with self.subTest(attribute=removed):
                self.assertFalse(hasattr(authority, removed))

    def test_2_ambient_speech_in_standby_does_nothing(self) -> None:
        runtime = FakeVoiceRuntime()
        for text in ("la la la la", "so anyway I told him", "the weather is nice"):
            runtime.ambient_speech(text)

        self.assertEqual(runtime.totals(), (0, 0, 0, 0), "capture/STT/chat/TTS")
        self.assertEqual(set(runtime.events), {"ignored:no_lease"})

    def test_3_a_command_without_a_wake_phrase_does_nothing(self) -> None:
        """The exact phrasing that WOULD work after a wake, spoken cold."""
        runtime = FakeVoiceRuntime()
        runtime.ambient_speech("Open Notepad")
        runtime.ambient_speech("Close Calculator")

        self.assertEqual(runtime.totals(), (0, 0, 0, 0))
        self.assertEqual(runtime.chat_calls, 0, "no /chat, so no desktop execution")

    def test_standby_never_stops_the_wake_monitor(self) -> None:
        """Standby is a permission state; it must not imply anything is off."""
        runtime = FakeVoiceRuntime()
        runtime.ambient_speech()
        # A wake event is still honoured immediately afterwards.
        runtime.wake_event()
        self.assertEqual(runtime.chat_calls, 1)


# --------------------------------------------------------------------------- #
# 4-5: one activation, one turn
# --------------------------------------------------------------------------- #


class PrimaryTurnTests(unittest.TestCase):
    def test_4_a_valid_wake_runs_exactly_one_turn(self) -> None:
        runtime = FakeVoiceRuntime()
        runtime.wake_event()
        self.assertEqual(runtime.totals(), (1, 1, 1, 1))

    def test_5_a_duplicate_activation_id_runs_one_turn(self) -> None:
        runtime = FakeVoiceRuntime()
        runtime.wake_event("activation-42")
        runtime.wake_event("activation-42")

        self.assertEqual(runtime.chat_calls, 1, "one activation, one dispatch")
        self.assertIn("rejected:duplicate_activation", runtime.events)

    def test_distinct_activations_each_run(self) -> None:
        runtime = FakeVoiceRuntime()
        runtime.wake_event("a-1")
        runtime.wait(FOLLOW_UP_SECONDS + 1)
        runtime.wake_event("a-2")
        self.assertEqual(runtime.chat_calls, 2)

    def test_a_stale_lease_cannot_act_after_standby(self) -> None:
        authority = VoiceConversationAuthority()
        lease = authority.begin_wake_turn()
        authority.return_to_standby("done")
        self.assertFalse(authority.lease_is_live(lease))
        self.assertFalse(authority.may_capture_command())


# --------------------------------------------------------------------------- #
# 6-9, 11: the follow-up window
# --------------------------------------------------------------------------- #


class FollowUpWindowTests(unittest.TestCase):
    def test_6_a_follow_up_cannot_start_while_the_assistant_speaks(self) -> None:
        """Bunnelby must never hear its own voice as the next command."""
        runtime = FakeVoiceRuntime()
        runtime.wake_event()
        runtime.gate.blocked = True

        runtime.follow_up_speech("this is actually the assistant's own audio")

        self.assertEqual(runtime.chat_calls, 1, "still just the primary turn")
        self.assertIn("ignored:follow_up_not_permitted", runtime.events)

    def test_7_a_follow_up_within_ten_seconds_is_accepted_once(self) -> None:
        runtime = FakeVoiceRuntime()
        runtime.wake_event(transcript="what is rag")
        runtime.wait(3.0)
        runtime.follow_up_speech("give me an example")

        self.assertEqual(runtime.chat_calls, 2)
        self.assertEqual(runtime.events.count("follow_up_accepted"), 1)

    def test_8_after_the_deadline_speech_is_inert(self) -> None:
        """The 10s timer expires PERMISSION, nothing else."""
        runtime = FakeVoiceRuntime()
        runtime.wake_event()
        before = runtime.totals()

        runtime.wait(FOLLOW_UP_SECONDS + 2.0)
        self.assertIn("follow_up_expired", runtime.events)
        self.assertIs(runtime.authority.phase, ConversationPhase.STANDBY_WAKE)

        runtime.ambient_speech("Open Calculator")
        runtime.follow_up_speech("explain more")

        self.assertEqual(runtime.totals(), before, "no capture/STT/chat/TTS")

    def test_9_a_wake_after_timeout_starts_a_normal_new_turn(self) -> None:
        runtime = FakeVoiceRuntime()
        runtime.wake_event()
        runtime.wait(FOLLOW_UP_SECONDS + 2.0)
        runtime.ambient_speech("Open Calculator")
        self.assertEqual(runtime.chat_calls, 1)

        runtime.wake_event(transcript="Open Calculator")

        self.assertEqual(runtime.chat_calls, 2, "the wake phrase restores access")
        self.assertIs(runtime.authority.phase, ConversationPhase.FOLLOW_UP)

    def test_11_the_follow_up_chain_is_bounded(self) -> None:
        runtime = FakeVoiceRuntime(max_follow_ups=2)
        runtime.wake_event()
        runtime.follow_up_speech("one")
        runtime.follow_up_speech("two")
        runtime.follow_up_speech("three")

        self.assertEqual(runtime.chat_calls, 3, "primary + exactly two follow-ups")
        self.assertIn("ignored:follow_up_not_permitted", runtime.events)

    def test_11b_a_wake_phrase_resets_the_budget(self) -> None:
        runtime = FakeVoiceRuntime(max_follow_ups=2)
        runtime.wake_event()
        runtime.follow_up_speech("one")
        runtime.follow_up_speech("two")
        self.assertEqual(runtime.authority.consecutive_follow_ups, 2)

        runtime.wake_event("fresh")
        self.assertEqual(runtime.authority.consecutive_follow_ups, 0)
        runtime.follow_up_speech("allowed again")
        self.assertEqual(runtime.chat_calls, 5)

    def test_ambient_speech_cannot_chain_indefinitely(self) -> None:
        """A song during the window may reach the bound, and then it stops."""
        runtime = FakeVoiceRuntime(max_follow_ups=2)
        runtime.wake_event()
        for _ in range(20):
            runtime.follow_up_speech("la la la")
        self.assertLessEqual(runtime.chat_calls, 3)


# --------------------------------------------------------------------------- #
# 10, 12: the original incident, and what must keep working
# --------------------------------------------------------------------------- #


class GarbledAndGreetingTests(unittest.TestCase):
    def test_10_a_garbled_turn_does_not_manufacture_a_second_assistant_turn(self) -> None:
        """The incident: garbled primary -> clarification -> unprompted greeting.

        The clarification is one legitimate response. What must not happen is a
        further autonomous turn produced from continuing background audio.
        """
        runtime = FakeVoiceRuntime()
        runtime.wake_event(transcript="mumble mumble garbled")
        self.assertEqual(runtime.chat_calls, 1, "one clarification, from one wake")

        # Bunnelby is speaking the clarification: the gate is closed.
        runtime.gate.blocked = True
        runtime.follow_up_speech("hey")
        self.assertEqual(runtime.chat_calls, 1, "self-audio cannot start a turn")

        # It finishes speaking, the singing continues, and the bound stops it.
        runtime.gate.blocked = False
        for _ in range(10):
            runtime.follow_up_speech("hey")
        self.assertLessEqual(runtime.chat_calls, 3)

        # Once the window lapses, nothing further is possible without a wake.
        runtime.wait(FOLLOW_UP_SECONDS + 1)
        settled = runtime.chat_calls
        for _ in range(10):
            runtime.ambient_speech("hey")
        self.assertEqual(runtime.chat_calls, settled)

    def test_an_empty_transcript_never_dispatches(self) -> None:
        runtime = FakeVoiceRuntime()
        runtime.wake_event(transcript="   ")
        self.assertEqual(runtime.chat_calls, 0)
        self.assertIn("empty_transcript", runtime.events)

    def test_12_a_greeting_inside_an_authorized_turn_still_works(self) -> None:
        from services.api.app.brain_agent import decide

        runtime = FakeVoiceRuntime()
        runtime.wake_event(transcript="hello")
        self.assertEqual(runtime.chat_calls, 1, "an authorized greeting is a real turn")

        decision = decide("hello")
        self.assertEqual(decision.mode, "answer")
        self.assertIn("How can I help", decision.reply)

    def test_12b_the_greeting_pattern_is_untouched(self) -> None:
        from services.api.app.persona import SIMPLE_GREETING_PATTERN

        for text in ("hi", "Hello", "hey", "good morning"):
            with self.subTest(text=text):
                self.assertTrue(SIMPLE_GREETING_PATTERN.match(text))


# --------------------------------------------------------------------------- #
# 13: typed chat is a different origin entirely
# --------------------------------------------------------------------------- #


class TypedChatIndependenceTests(unittest.TestCase):
    def test_13_typed_chat_is_unaffected_by_conversation_phase(self) -> None:
        from unittest.mock import patch

        from services.api.app import message_dispatch
        from services.api.app.orchestrator import OrchestratorResult

        authority = VoiceConversationAuthority()
        self.assertIs(authority.phase, ConversationPhase.STANDBY_WAKE)

        executed: list[str] = []

        def stub_execute(decision, user_message, session_id=None, turn_id=None):
            executed.append(user_message)
            return OrchestratorResult(
                reply="(stub)", action_type="desktop_control", memory_content=""
            )

        with patch("services.api.app.tool_executor.execute", stub_execute):
            message_dispatch.handle_message_result("Open Notepad", session_id="typed")

        self.assertEqual(executed, ["Open Notepad"])


# --------------------------------------------------------------------------- #
# 14: the factual-memory fix still holds (detail in test_brain_factual_context)
# --------------------------------------------------------------------------- #


class FactualMemoryCrossCheckTests(unittest.TestCase):
    def test_14_desktop_history_does_not_pollute_a_factual_question(self) -> None:
        from unittest.mock import patch

        from services.api.app import memory_service as ms

        history = [
            ms.MemoryTurn(1, 2, "Open Notepad", "Notepad is open.", "desktop_control (open_app)"),
        ]
        with patch.object(ms, "_load_safe_turns", lambda session_id=None: history):
            context = ms.build_memory_context("What is Notepad?", session_id="s")
        self.assertNotIn("Open Notepad", context)


class ConcurrencyTests(unittest.TestCase):
    def test_the_authority_is_thread_safe(self) -> None:
        """The control listener and the turn loop are different threads."""
        authority = VoiceConversationAuthority()
        errors: list[BaseException] = []

        def mint():
            for index in range(60):
                try:
                    authority.begin_wake_turn(f"{threading.get_ident()}-{index}")
                except VoiceActivationDenied:
                    pass
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

        def cycle():
            for _ in range(60):
                authority.open_follow_up_window()
                authority.return_to_standby("cycle")

        threads = [threading.Thread(target=mint), threading.Thread(target=cycle)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
