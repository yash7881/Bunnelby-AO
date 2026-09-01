# Bunnelby Wake-Word Engine Lock

Status: LOCKED
Date: 2026-09-01

## Decision

Bunnelby's wake-word path is frozen on the following local-first architecture:

Silero VAD -> faster-whisper local ASR -> strict wake phrase matcher

Target user-facing wake phrase: `Hey Bunnelby`

## Current baseline evidence

Controlled Windows microphone test using `faster-whisper` `base.en` on CPU/int8 with built-in Silero VAD:

- Detections: 7/10
- Recall: 70.0%
- Average ASR latency: 0.47 s
- Audio saved: NO
- Successful transcripts were exact `Hey Bunnelby`
- Two misses were transcribed as `Bunnelby` (ASR dropped the leading `Hey`)
- One miss produced no transcript after an unclear attempt

This is materially stronger than the tested alternatives:

- openWakeWord custom path: failed production recall/false-positive target
- sherpa-onnx exact/alias KWS: 0/60 on controlled alias evaluation
- Porcupine: not adopted because setup requires an external account/company-email gate and an AccessKey dependency

## Safety / matching policy

- Do not use broad fuzzy matching for wake activation.
- Do not accept common phrases observed during diagnostics such as `hey but there'll be`.
- Do not treat wake-word recognition as authentication.
- Sensitive actions remain governed by deterministic approval policy.
- Pre-wake audio is local only and should remain volatile by default.

## Refinement policy

The engine choice is locked. One controlled matcher/ASR refinement is allowed to improve real-user recall without materially increasing false activations. Technology switching is not allowed unless later production evidence objectively fails the acceptance target.

The current acceptance target remains:

- combined real-user recall >= 80%
- false activations <= 0.5 per hour

## Planned runtime direction

Always-on runtime should use lightweight VAD gating and only invoke ASR on detected speech windows. Heavy models must not run continuously when avoidable.

Next milestone after controlled refinement: build the Windows background listener, then measure long-duration false activations before production certification.
