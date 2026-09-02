# Part 10 / 10.1 persistent voice runtime

## Implemented architecture

The production path is intentionally small and preserves the existing application:

```text
one 16 kHz mono microphone stream
  -> STANDBY: Silero VAD -> faster-whisper base.en CPU/int8 -> strict "Hey Bunnelby"
  -> LISTENING: Silero conversation VAD -> RAM-only utterance
  -> TRANSCRIBING: faster-whisper small multilingual
  -> THINKING: existing POST /chat and its existing tool/approval policy
  -> SPEAKING: existing POST /tts -> cancellable sounddevice WAV playback
  -> FOLLOW_UP: 10-second speech-onset window -> LISTENING or STANDBY
```

The wake matcher is inactive while Bunnelby is speaking. A conservative correlation
gate compares microphone frames with recently played TTS audio before accepting a
barge-in. This is an echo rejection heuristic, not a claim of acoustic echo
cancellation, so barge-in remains a mandatory real-laptop acceptance test.

## Timing contract

`FOLLOW_UP_SECONDS` defaults to `10`. The deadline uses `time.monotonic()` and is
created from the terminal playback timestamp after the output stream has stopped. If
speech begins by the deadline, its full utterance is captured even if it ends after
the deadline. Every assistant response creates a fresh window.

If TTS is disabled, the spoken reply is empty, synthesis fails, playback cannot start,
or playback fails, the runtime starts the follow-up window from that explicit fallback
completion/failure time. The text response remains visible and the voice runtime does
not deadlock.

## Configuration

Conversation STT defaults remain conservative until the same-corpus laptop benchmark
selects a different profile:

```dotenv
STT_ENABLED=true
STT_MODEL=small
STT_DEVICE=cpu
STT_COMPUTE_TYPE=int8
STT_CPU_THREADS=4
STT_BEAM_SIZE=5
FOLLOW_UP_SECONDS=10
TTS_ENABLED=true
EDGE_TTS_ENABLED=true
TTS_PROVIDER=edge
PIPER_ENABLED=true
```

For GPU operation, use `STT_DEVICE=cuda` and
`STT_COMPUTE_TYPE=int8_float16` only after both preflight and the spoken-corpus
benchmark pass. Windows CUDA directories are registered only in the current process
using `os.add_dll_directory` plus a process-local `PATH` prepend; user and system PATH
are not changed.

## Automated and hardware checks

Run the real CUDA gate. It consumes faster-whisper's lazy segments, so encoder
inference must actually execute:

```powershell
.\.venv\Scripts\python.exe scripts\wakeword\gpu_stt_preflight.py
```

Run the RAM-only same-corpus benchmark. Read each displayed English, Hindi, or
Hinglish sentence once. The six captured utterances remain in memory and are sent to
both profiles with the same `small` model and beam size 5:

```powershell
.\.venv\Scripts\python.exe scripts\wakeword\stt_profile_benchmark.py
```

The benchmark recommends GPU only when its median is at least 15 percent faster and
its mean word error rate is no more than 0.05 worse. An incomplete GPU comparison
fails closed to CPU/int8.

With the FastAPI backend running on port 8000, start the persistent runtime:

```powershell
.\.venv\Scripts\python.exe scripts\wakeword\wake_conversation_runtime.py
```

Useful bounded diagnostic options are `--turns`, `--language`, `--device`,
`--output-device`, `--no-dispatch`, `--no-tts`, and `--no-barge-in`. Production
defaults enable dispatch, TTS, barge-in, persistent operation, and language auto
detection.

## Telemetry and privacy

Startup output states the configured wake model, conversation model/device/compute
type/beam, TTS provider and voices, follow-up duration, and privacy policy. Every turn
prints machine-readable timings for wake ASR, speech capture, endpoint delay, STT,
backend, TTS preparation, first audio, playback, and total turn time.

Ordinary wake and conversation microphone samples stay in RAM. The persistent runtime
contains no temporary-file or audio-write path. The benchmark also retains its corpus
only in RAM and clears its references on exit. Wake certification reports follow the
separate policy in `docs/WAKE_CERTIFICATION.md`.

Voice is activation and input, not authentication. Follow-up turns still use the
existing `/chat` tool registry and deterministic approval policy; they do not bypass
Gmail send or other sensitive-action approvals.

## Failure behavior

- Wake, VAD, empty transcript, STT, and chat failures return to a known standby state.
- A native STT inference failure invalidates and unloads the cached model so the next
  turn cannot reuse a poisoned CTranslate2 instance.
- TTS and output-device failures retain the screen response and enter the documented
  follow-up fallback.
- Playback is cancellable and has queued, started, completed, cancelled, and failed
  terminal reporting.
- Ctrl+C closes the microphone stream and moves the state controller to stopping.

Automated success does not certify microphone acoustics. Wake recall/false positives,
actual audible playback, Hindi/Hinglish quality, follow-up timing, barge-in echo
behavior, and long-run stability must be recorded from the real Windows laptop before
Part 10 can be marked complete.
