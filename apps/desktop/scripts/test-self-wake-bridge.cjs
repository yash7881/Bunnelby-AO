'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const {
  encodeRendererSpeakingControl
} = require('../voice-control-protocol.cjs');

test('renderer speaking control is strict newline-delimited JSON', () => {
  assert.equal(
    encodeRendererSpeakingControl(true),
    '{"type":"renderer_speaking","speaking":true}\n'
  );
  assert.equal(
    encodeRendererSpeakingControl(false),
    '{"type":"renderer_speaking","speaking":false}\n'
  );
  assert.throws(() => encodeRendererSpeakingControl('true'), TypeError);
});

test('renderer speech guard blocks during playback and speaker-tail cooldown', async () => {
  const {
    createRendererSpeechGuard
  } = await import('../src/rendererSpeechGuard.mjs');

  let now = 1000;
  const guard = createRendererSpeechGuard({
    clock: () => now,
    tailMs: 750
  });

  assert.equal(guard.isActive(), false);
  guard.setSpeaking(true);
  assert.equal(guard.isActive(), true);

  now = 5000;
  assert.equal(guard.isActive(), true);

  guard.setSpeaking(false);
  assert.equal(guard.isActive(), true);

  now += 749;
  assert.equal(guard.isActive(), true);

  now += 2;
  assert.equal(guard.isActive(), false);
});

test('desktop wiring pre-arms suppression before WebAudio playback starts', () => {
  const root = path.resolve(__dirname, '..');
  const playerSource = fs.readFileSync(
    path.join(root, 'src', 'audio', 'aoVoicePlayer.js'),
    'utf8'
  );
  const preloadSource = fs.readFileSync(path.join(root, 'preload.cjs'), 'utf8');
  const electronSource = fs.readFileSync(path.join(root, 'electron.cjs'), 'utf8');

  assert.match(
    playerSource,
    /notifySpeaking\(true\);\s*await wait\(SELF_WAKE_ARM_DELAY_MS\);[\s\S]*sourceNode\.start\(\);/
  );
  assert.match(preloadSource, /setRendererSpeaking\(isSpeaking\)/);
  assert.match(electronSource, /--voice-control-stdin/);
  assert.match(electronSource, /stdio:\s*\['pipe',\s*'pipe',\s*'pipe'\]/);
});
