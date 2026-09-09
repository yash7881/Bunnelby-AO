'use strict';

// Regression tests for voice-runtime interpreter selection and verification.
//
// The bug these guard against: on a checkout with no .venv, resolution fell
// back to a bare `python` on PATH. On this machine that was a global 3.10 with
// none of the voice dependencies, so the child died on its first line
// (`ModuleNotFoundError: No module named 'numpy'`) before the microphone was
// opened -- and nothing in the UI said so, because stderr only reached the
// Electron console and `runtime_exit` carried a generic message.

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');

const {
  LOCAL_CONFIG_FILENAME,
  REQUIRED_RUNTIME_MODULES,
  INTERPRETER_PROBE_SOURCE,
  readLocalInterpreter,
  resolveVoiceRuntime,
  verifyVoiceRuntimeInterpreter,
  describeInterpreterFailure
} = require('../voiceRuntimeInterpreter.cjs');

const APP_DIR = path.join(__dirname, '..');

/** A spawnSync stand-in, so no real interpreter is needed. */
function fakeRunner(result) {
  const calls = [];
  const runner = (file, args, options) => {
    calls.push({ file, args, options });
    return result;
  };
  runner.calls = calls;
  return runner;
}

test('an explicit BUNNELBY_PYTHON wins and is labelled', () => {
  const resolved = resolveVoiceRuntime(APP_DIR, {
    BUNNELBY_PYTHON: 'C:\\custom\\python.exe'
  });
  assert.equal(resolved.pythonExecutable, 'C:\\custom\\python.exe');
  assert.equal(resolved.pythonSource, 'BUNNELBY_PYTHON');
});

test('blank BUNNELBY_PYTHON is ignored rather than used as an executable', () => {
  const resolved = resolveVoiceRuntime(APP_DIR, { BUNNELBY_PYTHON: '   ' });
  assert.notEqual(resolved.pythonSource, 'BUNNELBY_PYTHON');
});

test('there is NO silent PATH fallback', () => {
  // This repository has no .venv, which is exactly the failing configuration.
  // Guessing `python` here is what silently disabled voice in the field.
  const resolved = resolveVoiceRuntime(APP_DIR, {}, () => {
    throw new Error('ENOENT');
  });
  assert.equal(resolved.pythonExecutable, null);
  assert.equal(resolved.pythonSource, 'unresolved');
});

test('a durable local config supplies the interpreter without an env var', () => {
  const reader = () => JSON.stringify({ pythonPath: 'D:\\envs\\bunnelby\\python.exe' });
  const resolved = resolveVoiceRuntime(APP_DIR, {}, reader);
  assert.equal(resolved.pythonExecutable, 'D:\\envs\\bunnelby\\python.exe');
  assert.equal(resolved.pythonSource, LOCAL_CONFIG_FILENAME);
});

test('a malformed or empty local config is ignored, not fatal', () => {
  for (const raw of ['{ not json', '{}', '{"pythonPath": "   "}']) {
    const resolved = resolveVoiceRuntime(APP_DIR, {}, () => raw);
    assert.equal(resolved.pythonExecutable, null, raw);
  }
  assert.equal(readLocalInterpreter('C:\\nope', () => { throw new Error('ENOENT'); }), null);
});

test('an unconfigured interpreter is refused before any spawn', () => {
  const result = verifyVoiceRuntimeInterpreter(null, () => {
    throw new Error('must not be called');
  });
  assert.equal(result.ok, false);
  assert.match(result.reason, /was not configured/);
});

test('the unconfigured failure message still explains the remedy', () => {
  const message = describeInterpreterFailure(null, 'unresolved', 'was not configured', 'C:\\repo');
  assert.match(message, /No voice runtime interpreter is configured/);
  assert.match(message, /\.venv/);
  assert.match(message, /pythonPath/);
  assert.match(message, /Wake detection is disabled/);
});

test('the runtime script path is resolved from the repository root', () => {
  const resolved = resolveVoiceRuntime(APP_DIR, {});
  assert.ok(
    resolved.runtimeScript.endsWith(
      path.join('scripts', 'wakeword', 'wake_conversation_runtime.py')
    ),
    resolved.runtimeScript
  );
});

test('a healthy interpreter verifies', () => {
  const runner = fakeRunner({ status: 0, stdout: '', stderr: '' });
  assert.deepEqual(verifyVoiceRuntimeInterpreter('python', runner), { ok: true });
});

test('missing modules are reported by name', () => {
  const runner = fakeRunner({ status: 0, stdout: 'numpy,sherpa_onnx', stderr: '' });
  const result = verifyVoiceRuntimeInterpreter('python', runner);

  assert.equal(result.ok, false);
  assert.deepEqual(result.missing, ['numpy', 'sherpa_onnx']);
  assert.match(result.reason, /missing required modules: numpy, sherpa_onnx/);
});

test('the probe asks about every dependency the runtime needs', () => {
  const runner = fakeRunner({ status: 0, stdout: '', stderr: '' });
  verifyVoiceRuntimeInterpreter('python', runner);

  const [call] = runner.calls;
  assert.equal(call.args[0], '-c');
  assert.equal(call.args[1], INTERPRETER_PROBE_SOURCE);
  assert.deepEqual(call.args.slice(2), [...REQUIRED_RUNTIME_MODULES]);
  assert.ok(
    REQUIRED_RUNTIME_MODULES.includes('numpy'),
    'numpy is the import that actually failed in the field'
  );
});

test('the probe never imports the modules it checks', () => {
  // Importing faster_whisper/torch at startup would cost seconds.
  assert.match(INTERPRETER_PROBE_SOURCE, /find_spec/);
  assert.doesNotMatch(INTERPRETER_PROBE_SOURCE, /^import numpy/m);
});

test('a non-zero probe exit is reported with the interpreter stderr', () => {
  const runner = fakeRunner({
    status: 9009,
    stdout: '',
    stderr: 'line one\r\nis not recognized as an internal command'
  });
  const result = verifyVoiceRuntimeInterpreter('python', runner);

  assert.equal(result.ok, false);
  assert.match(result.reason, /exit 9009/);
  assert.match(result.reason, /is not recognized/);
});

test('a missing executable is reported, not thrown', () => {
  const runner = fakeRunner({ error: new Error('spawnSync ENOENT') });
  const result = verifyVoiceRuntimeInterpreter('nope.exe', runner);

  assert.equal(result.ok, false);
  assert.match(result.reason, /could not be executed/);
});

test('a throwing runner is contained', () => {
  const runner = () => {
    throw new Error('boom');
  };
  const result = verifyVoiceRuntimeInterpreter('python', runner);

  assert.equal(result.ok, false);
  assert.match(result.reason, /could not be executed \(boom\)/);
});

test('the failure message names the interpreter, its source, and the remedy', () => {
  const message = describeInterpreterFailure(
    'C:\\Python310\\python.exe',
    'PATH fallback',
    'is missing required modules: numpy'
  );

  assert.match(message, /C:\\Python310\\python\.exe/);
  assert.match(message, /PATH fallback/);
  assert.match(message, /numpy/);
  assert.match(message, /BUNNELBY_PYTHON/);
  assert.match(message, /\.venv/);
  assert.match(
    message,
    /Wake detection is disabled/,
    'the user must be told the consequence, not just the cause'
  );
});
