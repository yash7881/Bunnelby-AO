// Authoritative microphone state for the Bunnelby command bar.
//
// WHY THIS EXISTS. The mic highlight used to be computed as
// `isListening || runtimeMicActive` -- App's shared visual `coreState` OR a
// latch kept inside CommandBar from its own event subscription. Two writers
// over one pixel is a race by construction, and the two disagreed in both
// directions: `runtime_error`/`runtime_exit` are not `state` events, so
// CommandBar's latch never cleared and the mic stayed lit after the runtime
// died; meanwhile `coreState` is also written by TTS playback, the typed-chat
// fetch and the mic-click preview, none of which mean "the microphone is
// listening".
//
// So the persistent voice runtime is the single source of truth, and this
// module is the only place its events become a mic decision. It is pure: no
// timers, no DOM, no subscriptions -- which is what makes the event ordering
// testable without React or a real Electron window.

/** State assumed before the runtime has said anything. */
export const INITIAL_VOICE_RUNTIME_STATE = 'standby';

// The microphone is genuinely capturing for the user in exactly these phases.
const MIC_ACTIVE = new Set(['wake_detected', 'listening', 'follow_up']);

// Known phases in which it is not. `wake_candidate` belongs here on purpose:
// a maybe-wake is not a wake, and the rule is that the highlight means "I am
// listening to you now", never "I might be about to".
const MIC_INACTIVE = new Set([
  'standby',
  'wake_candidate',
  'transcribing',
  'thinking',
  'speaking',
  'error_recovery',
  'stopping'
]);

export const MIC_ACTIVE_RUNTIME_STATES = Object.freeze([...MIC_ACTIVE]);
export const MIC_INACTIVE_RUNTIME_STATES = Object.freeze([...MIC_INACTIVE]);

function normalize(value) {
  return String(value ?? '').trim().toLowerCase();
}

/** True for a runtime phase this module recognises either way. */
export function isKnownVoiceRuntimeState(state) {
  const candidate = normalize(state);
  return MIC_ACTIVE.has(candidate) || MIC_INACTIVE.has(candidate);
}

/** Whether the mic icon should be highlighted for this runtime state. */
export function isMicActive(state) {
  return MIC_ACTIVE.has(normalize(state));
}

/**
 * Fold one voice-runtime event into the authoritative runtime state.
 *
 * Unrecognised events and unrecognised phase names PRESERVE the current state
 * rather than guessing. That matters for the real wake sequence, which arrives
 * as three events in a row -- `wake_detected`, then `state: wake_detected`,
 * then `state: listening` -- and must stay continuously active across all of
 * them without a flicker.
 */
export function nextVoiceRuntimeState(current, event) {
  const safeCurrent = isKnownVoiceRuntimeState(current)
    ? normalize(current)
    : INITIAL_VOICE_RUNTIME_STATE;

  if (!event || typeof event !== 'object') return safeCurrent;

  const eventType = normalize(event.event);

  if (eventType === 'state') {
    const nextState = normalize(event.state);
    return isKnownVoiceRuntimeState(nextState) ? nextState : safeCurrent;
  }

  // The runtime emits this alongside `state: wake_detected`; honouring it
  // directly means the highlight does not depend on which of the two lands
  // first.
  if (eventType === 'wake_detected') return 'wake_detected';

  // A runtime that has failed or exited is not listening, whatever it last
  // said. This is the case the old CommandBar latch could not see at all.
  if (
    eventType === 'runtime_error' ||
    eventType === 'runtime_exit' ||
    eventType === 'runtime_ready'
  ) {
    return INITIAL_VOICE_RUNTIME_STATE;
  }

  return safeCurrent;
}
