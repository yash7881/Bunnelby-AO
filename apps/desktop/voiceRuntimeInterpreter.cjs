'use strict';

// Which Python runs the persistent voice runtime, and whether it actually can.
//
// WHY THIS EXISTS. `resolveVoiceRuntime` used to end in
// `fs.existsSync(venvPython) ? venvPython : 'python'`, which quietly assumes a
// bare `python` on PATH is interchangeable with the project interpreter. On a
// checkout without a .venv it is not: `python` resolved to a global 3.10 with
// none of the voice dependencies, so the child died on its first line
// (`ModuleNotFoundError: No module named 'numpy'`) before the microphone was
// ever opened. The wake word then did nothing, with no error in the UI.
//
// Kept out of electron.cjs so it is unit-testable without booting Electron,
// exactly as voice-control-protocol.cjs is.

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

// Module-scope imports of wake_conversation_runtime.py, plus the packages it
// needs to open the microphone and transcribe. Any one missing is fatal.
const REQUIRED_RUNTIME_MODULES = Object.freeze([
  'numpy',
  'sherpa_onnx',
  'sounddevice',
  'faster_whisper'
]);

// find_spec locates a module without executing it, so the check stays fast
// enough to run once at startup.
const INTERPRETER_PROBE_SOURCE = [
  'import importlib.util, sys',
  'missing = []',
  'for name in sys.argv[1:]:',
  '    try:',
  '        if importlib.util.find_spec(name) is None:',
  '            missing.append(name)',
  '    except Exception:',
  '        missing.append(name)',
  'sys.stdout.write(",".join(missing))'
].join('\n');

/**
 * Decide which interpreter to use, and record how that decision was made.
 *
 * Order: explicit BUNNELBY_PYTHON, then the repository's own .venv, then
 * whatever `python` is on PATH. The last is a guess and is labelled as one so
 * a failure can name it.
 */
function resolveVoiceRuntime(appDir, env = process.env) {
  const repoRoot = path.resolve(appDir, '..', '..');
  const runtimeScript = path.join(
    repoRoot,
    'scripts',
    'wakeword',
    'wake_conversation_runtime.py'
  );

  const configuredPython = String(env.BUNNELBY_PYTHON || '').trim();
  if (configuredPython) {
    return {
      repoRoot,
      runtimeScript,
      pythonExecutable: configuredPython,
      pythonSource: 'BUNNELBY_PYTHON'
    };
  }

  const venvPython = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
  if (fs.existsSync(venvPython)) {
    return {
      repoRoot,
      runtimeScript,
      pythonExecutable: venvPython,
      pythonSource: 'repo .venv'
    };
  }

  return {
    repoRoot,
    runtimeScript,
    pythonExecutable: 'python',
    pythonSource: 'PATH fallback'
  };
}

/**
 * Confirm an interpreter can actually run the voice runtime.
 *
 * Returns { ok } or { ok: false, reason, missing }. Never throws: a broken
 * check must produce a clear message, not crash the main process.
 */
function verifyVoiceRuntimeInterpreter(pythonExecutable, runner = spawnSync) {
  let probe;
  try {
    probe = runner(
      pythonExecutable,
      ['-c', INTERPRETER_PROBE_SOURCE, ...REQUIRED_RUNTIME_MODULES],
      { encoding: 'utf-8', timeout: 20000, windowsHide: true }
    );
  } catch (error) {
    return { ok: false, reason: `could not be executed (${error.message})` };
  }

  if (!probe || probe.error) {
    const message = probe && probe.error ? probe.error.message : 'no result';
    return { ok: false, reason: `could not be executed (${message})` };
  }

  if (probe.status !== 0) {
    const stderr = String(probe.stderr || '')
      .trim()
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(-2)
      .join(' | ');
    return {
      ok: false,
      reason: `failed its dependency check (exit ${probe.status})${
        stderr ? `: ${stderr}` : ''
      }`
    };
  }

  const missing = String(probe.stdout || '')
    .trim()
    .split(',')
    .map((name) => name.trim())
    .filter(Boolean);

  if (missing.length > 0) {
    return {
      ok: false,
      missing,
      reason: `is missing required modules: ${missing.join(', ')}`
    };
  }

  return { ok: true };
}

/** The message shown when the chosen interpreter cannot run the runtime. */
function describeInterpreterFailure(pythonExecutable, pythonSource, reason) {
  return (
    `Voice runtime interpreter "${pythonExecutable}" (${pythonSource}) ${reason}. ` +
    'Wake detection is disabled. Set BUNNELBY_PYTHON to an interpreter that has ' +
    'the voice dependencies installed, or create a .venv in the repository root.'
  );
}

module.exports = {
  REQUIRED_RUNTIME_MODULES,
  INTERPRETER_PROBE_SOURCE,
  resolveVoiceRuntime,
  verifyVoiceRuntimeInterpreter,
  describeInterpreterFailure
};
