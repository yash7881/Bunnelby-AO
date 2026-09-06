# Bunnelby Wake Certification v1

## Status

The active wake path under test is:

`Silero VAD -> faster-whisper base.en CPU/int8 -> strict wake matcher`

Wake phrase: `Hey Bunnelby`.

This document defines the measurement protocol. It does **not** declare the wake path production-certified yet.

## Why this protocol exists

A wake detector has two independent quality dimensions:

1. **Positive recall**: how often an intentional `Hey Bunnelby` attempt wakes the assistant.
2. **False-positive rate**: how often ordinary non-wake speech/audio activates it.

A count such as `18 detections / expected 10` must never be reported as `180% recall`. Recall is attempt-level: each prompted positive trial can contribute at most one success.

## Evidence used for the methodology

- openWakeWord's training/evaluation implementation explicitly evaluates false positives per hour and uses an approximately 11.3-hour false-positive validation set. This supports using multi-hour negative-audio evaluation rather than a few minutes of quiet-room testing.
- faster-whisper officially integrates Silero VAD and exposes VAD parameter tuning. Bunnelby currently uses an external Silero VAD stage and disables faster-whisper's second VAD pass for the already-segmented wake candidate.
- Silero documents its speech threshold as deployment/dataset dependent. Bunnelby's threshold therefore stays frozen during a certification run and may only be changed after a documented failed benchmark.
- sherpa-onnx's VAD API supports explicit `flush()` for finite audio, which the prompted positive harness uses so the final speech segment is not silently lost.

Reference sources:

- https://github.com/dscripka/openWakeWord/blob/main/openwakeword/train.py
- https://github.com/SYSTRAN/faster-whisper/blob/master/README.md
- https://github.com/snakers4/silero-vad/wiki/Quality-Metrics
- https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/generate-subtitles.py

## Frozen v1 detector settings

The certification harness imports the exact detector implementation from `scripts/wakeword/always_on_wake_listener.py` instead of duplicating matcher/VAD/ASR logic.

Important current invariants:

- 16 kHz mono microphone audio.
- Silero ONNX VAD.
- faster-whisper `base.en`, CPU/int8.
- strict accepted forms only: canonical `Hey Bunnelby` plus rare user-observed acoustic spellings already admitted by the listener.
- bare `Bunnelby` remains rejected.
- common confusions such as `hey but there'll be` remain rejected.
- raw wake audio is RAM-only and is not written to disk.
- negative-test normal speech transcripts are not persisted; only transcripts that actually cause a false trigger are retained for diagnosis.

## Positive recall protocol

Run prompted trials. One prompt equals one trial. The user says exactly `Hey Bunnelby` once during each recording window.

Recommended first controlled run: 20 attempts.

Metrics:

`recall = detected_attempts / total_attempts`

Acceptance bands:

- preferred: >= 90%
- minimum: >= 80%
- below 80%: fail/refine before production

A trial with multiple internal candidate matches still counts as **one** successful recall trial.

## Negative false-positive protocol

During a negative run, the wake phrase must not be spoken intentionally.

Use realistic audio:

- normal English speech
- Hindi speech
- Hinglish speech
- ordinary commands and names
- room/background audio
- media/TV/speaker audio where practical

The harness reports both:

1. **raw false matches** — every qualifying matcher activation, with no cooldown hiding later candidates;
2. **debounced wake events** — the user-visible event rate after the configured refractory period.

Certification must not be achieved by cooldown suppression. The raw rate is the primary detector-quality metric.

Target:

`raw false positives/hour <= 0.5`

The harness also reports a one-sided 95% Poisson upper confidence bound. A short zero-event run is therefore reported only as a quick pass, not as statistical certification.

For zero observed false positives, roughly 6 hours are needed before the one-sided 95% upper bound falls below 0.5/hour. A longer 10-12 hour mixed negative corpus is preferred because it gives a stronger real-world estimate and is similar in scale to openWakeWord's published training validation methodology.

## Privacy

- audio saved: NO
- cloud wake/STT audio: NO
- positive reports may contain intentional wake transcripts
- negative reports persist only aggregate metrics and actual false-trigger transcripts
- ordinary negative speech transcripts are console-only when `--debug-transcripts` is explicitly requested

Reports are written under:

`%LOCALAPPDATA%\Bunnelby\wakeword\logs\`

## Commands

Self-test:

```powershell
python scripts\wakeword\wake_certification.py --self-test
```

Positive 20-attempt run:

```powershell
python scripts\wakeword\wake_certification.py --mode positive --attempts 20
```

Quick five-minute negative screen:

```powershell
python scripts\wakeword\wake_certification.py --mode negative-live --duration 300
```

A multi-hour final negative run should only be started after the short positive and negative screens pass, because there is no value spending hours benchmarking a configuration that already fails a five-minute check.

## Current evidence before v1 certification

The previous always-on diagnostic produced 18 wake detections while the user states they intentionally said `Hey Bunnelby` 18 times, with approximately 0.47 s average candidate ASR latency. That is strong functional evidence but is **not** a formal attempt-level recall benchmark because the old command was configured with `--expected-wakes 10` and could not align individual spoken attempts to detections.

The certification harness exists specifically to remove that ambiguity.
