'use strict';

// Regression tests for the authoritative mic-highlight state mapping.
//
// The bug these guard against: the highlight was computed from two independent
// writers (App's shared visual `coreState` OR a latch inside CommandBar), which
// disagreed in both directions -- notably `runtime_error`/`runtime_exit` are not
// `state` events, so the latch never cleared and the mic stayed lit after the
// runtime had died.
//
// The reducer is pure, so the real event ORDERING is testable here with no
// React, no Electron window, and no timers.

const assert = require('node:assert/strict');
const test = require('node:test');

const MODULE = '../src/voiceMicState.mjs';

/** Fold a sequence of runtime events and report the mic decision after each. */
async function trace(events) {
  const { INITIAL_VOICE_RUNTIME_STATE, isMicActive, nextVoiceRuntimeState } =
    await import(MODULE);

  let state = INITIAL_VOICE_RUNTIME_STATE;
  return events.map((event) => {
    state = nextVoiceRuntimeState(state, event);
    return { state, micActive: isMicActive(state) };
  });
}

const stateEvent = (state) => ({ event: 'state', state, reason: 'test' });

test('the initial state is standby with the mic off', async () => {
  const { INITIAL_VOICE_RUNTIME_STATE, isMicActive } = await import(MODULE);
  assert.equal(INITIAL_VOICE_RUNTIME_STATE, 'standby');
  assert.equal(isMicActive(INITIAL_VOICE_RUNTIME_STATE), false);
});

test('each runtime phase maps to the required mic decision', async () => {
  const expected = {
    standby: false,
    wake_candidate: false,
    wake_detected: true,
    listening: true,
    transcribing: false,
    thinking: false,
    speaking: false,
    follow_up: true,
    error_recovery: false,
    stopping: false
  };

  for (const [phase, micActive] of Object.entries(expected)) {
    const [result] = await trace([stateEvent(phase)]);
    assert.equal(result.state, phase, `state should become ${phase}`);
    assert.equal(
      result.micActive,
      micActive,
      `${phase} must leave the mic ${micActive ? 'ON' : 'OFF'}`
    );
  }
});

test('runtime_error and runtime_exit turn the mic off from listening', async () => {
  for (const failure of ['runtime_error', 'runtime_exit']) {
    const results = await trace([
      stateEvent('listening'),
      { event: failure, message: 'runtime stopped' }
    ]);
    assert.equal(results[0].micActive, true, 'listening must light the mic');
    assert.equal(
      results[1].micActive,
      false,
      `${failure} must clear the mic -- the old CommandBar latch could not see this`
    );
    assert.equal(results[1].state, 'standby');
  }
});

test('the real wake sequence keeps the mic continuously active', async () => {
  // Exactly what the persistent runtime emits, in order.
  const results = await trace([
    { event: 'wake_detected', transcript: 'Hey Bunnelby', latency_seconds: 0.2 },
    stateEvent('wake_detected'),
    stateEvent('listening')
  ]);

  assert.deepEqual(
    results.map((entry) => entry.micActive),
    [true, true, true],
    'the highlight must not flicker across the three wake events'
  );
  assert.equal(results.at(-1).state, 'listening');
});

test('a full turn follows the required on/off sequence', async () => {
  const results = await trace([
    stateEvent('standby'),
    { event: 'wake_detected', transcript: 'Hey Bunnelby' },
    stateEvent('listening'),
    stateEvent('transcribing'),
    stateEvent('thinking'),
    stateEvent('speaking'),
    stateEvent('follow_up'),
    stateEvent('standby')
  ]);

  assert.deepEqual(
    results.map((entry) => entry.micActive),
    [false, true, true, false, false, false, true, false]
  );
});

test('follow_up expiring to standby clears the mic', async () => {
  const results = await trace([stateEvent('follow_up'), stateEvent('standby')]);
  assert.equal(results[0].micActive, true);
  assert.equal(results[1].micActive, false);
});

test('unrelated events preserve the current state', async () => {
  const results = await trace([
    stateEvent('listening'),
    { event: 'user_transcript', text: 'open notepad' },
    { event: 'assistant_response', reply: 'Notepad is open.' },
    { event: 'metrics', turn: 1 }
  ]);

  assert.deepEqual(
    results.map((entry) => entry.state),
    ['listening', 'listening', 'listening', 'listening'],
    'only phase-bearing events may move the authoritative state'
  );
});

test('malformed and unknown input never changes the decision', async () => {
  const { isMicActive, nextVoiceRuntimeState } = await import(MODULE);

  for (const bad of [null, undefined, 'listening', 42, [], { event: 'state' }]) {
    assert.equal(
      nextVoiceRuntimeState('listening', bad),
      'listening',
      `${JSON.stringify(bad)} must be ignored`
    );
  }

  // An unrecognised phase name is preserved rather than guessed at.
  assert.equal(
    nextVoiceRuntimeState('listening', stateEvent('reticulating_splines')),
    'listening'
  );
  assert.equal(isMicActive('reticulating_splines'), false);
});

test('phase names are matched case-insensitively and trimmed', async () => {
  const { isMicActive, nextVoiceRuntimeState } = await import(MODULE);
  assert.equal(nextVoiceRuntimeState('standby', stateEvent('  LISTENING ')), 'listening');
  assert.equal(isMicActive('Follow_Up'), true);
});

test('a corrupt current state falls back to standby, not to ON', async () => {
  const { nextVoiceRuntimeState } = await import(MODULE);
  assert.equal(nextVoiceRuntimeState(undefined, { event: 'metrics' }), 'standby');
  assert.equal(nextVoiceRuntimeState('nonsense', { event: 'metrics' }), 'standby');
});

test('the active set is exactly the three listening phases', async () => {
  const { MIC_ACTIVE_RUNTIME_STATES } = await import(MODULE);
  assert.deepEqual(
    [...MIC_ACTIVE_RUNTIME_STATES].sort(),
    ['follow_up', 'listening', 'wake_detected']
  );
});
