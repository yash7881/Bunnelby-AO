'use strict';

// Which Python runs the persistent voice runtime, and whether it actually can.
//
// WHY THIS EXISTS. Resolution used to end in
// `fs.existsSync(venvPython) ? venvPython : 'python'`, which quietly assumes a
// bare `python` on PATH is interchangeable with the project interpreter. On a
// checkout without a .venv it is not: `python` resolved to a global 3.10 with
// none of the voice dependencies, so the child died on its first line
// (`ModuleNotFoundError: No module named 'numpy'`) before the microphone was
// ever opened. The wake word then did nothing, with no error in the UI.
//
// There is now NO PATH fallback. An interpreter is either explicitly chosen,
// or the repository's own, or the launch fails loudly with the remedy. A guess
// that silently disables voice is worse than a refusal that explains itself.
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

// Durable per-machine override, git-ignored, so a working interpreter is
// configured ONCE instead of exported into the shell on every launch.
const LOCAL_CONFIG_FILENAME = '.bunnelby.local.json';

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

/** Read `pythonPath` from the git-ignored local config, if present and sane. */
function readLocalInterpreter(repoRoot, reader = fs.readFileSync) {
  const configPath = path.join(repoRoot, LOCAL_CONFIG_FILENAME);
  let raw;
  try {
    raw = reader(configPath, 'utf-8');
  } catch {
    return null; // absent is the normal case, not an error
  }
  try {
    const parsed = JSON.parse(raw);
    const candidate = String(parsed?.pythonPath || '').trim();
    return candidate || null;
  } catch {
    return null; // a malformed file must not crash startup
  }
}

/**
 * Decide which interpreter to use, and record how that decision was made.
 *
 * Order: explicit BUNNELBY_PYTHON, the repository's own .venv, then the
 * durable local config. If none resolves there is deliberately no fallback --
 * `pythonExecutable` is null and the caller reports why.
 */
function resolveVoiceRuntime(appDir, env = process.env, reader = fs.readFileSync) {
  const repoRoot = path.resolve(appDir, '..', '..');
  const runtimeScript = path.join(
    repoRoot,
    'scripts',
    'wakeword',
    'wake_conversation_runtime.py'
  );
  const base = { repoRoot, runtimeScript };

  const configuredPython = String(env.BUNNELBY_PYTHON || '').trim();
  if (configuredPython) {
    return { ...base, pythonExecutable: configuredPython, pythonSource: 'BUNNELBY_PYTHON' };
  }

  const venvPython = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
  if (fs.existsSync(venvPython)) {
    return { ...base, pythonExecutable: venvPython, pythonSource: 'repo .venv' };
  }

  const localPython = readLocalInterpreter(repoRoot, reader);
  if (localPython) {
    return { ...base, pythonExecutable: localPython, pythonSource: LOCAL_CONFIG_FILENAME };
  }

  return { ...base, pythonExecutable: null, pythonSource: 'unresolved' };
}

/**
 * Confirm an interpreter can actually run the voice runtime.
 *
 * Returns { ok } or { ok: false, reason, missing }. Never throws: a broken
 * check must produce a clear message, not crash the main process.
 */
function verifyVoiceRuntimeInterpreter(pythonExecutable, runner = spawnSync) {
  if (!pythonExecutable) {
    return { ok: false, reason: 'was not configured' };
  }

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
      reason: `failed its dependency check (exit ${probe.status})${stderr ? `: ${stderr}` : ''}`
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

/** The message shown when no usable interpreter is available. */
function describeInterpreterFailure(pythonExecutable, pythonSource, reason, repoRoot = '') {
  const subject = pythonExecutable
    ? `Voice runtime interpreter "${pythonExecutable}" (${pythonSource}) ${reason}`
    : 'No voice runtime interpreter is configured';

  const localConfig = repoRoot
    ? path.join(repoRoot, LOCAL_CONFIG_FILENAME)
    : LOCAL_CONFIG_FILENAME;

  return (
    `${subject}. Wake detection is disabled. Fix it once by creating a .venv in ` +
    `the repository root and installing services/api/requirements.txt, or by ` +
    `writing {"pythonPath": "<full path to python.exe>"} into ${localConfig}. ` +
    'BUNNELBY_PYTHON also works for a single session.'
  );
}

module.exports = {
  LOCAL_CONFIG_FILENAME,
  REQUIRED_RUNTIME_MODULES,
  INTERPRETER_PROBE_SOURCE,
  readLocalInterpreter,
  resolveVoiceRuntime,
  verifyVoiceRuntimeInterpreter,
  describeInterpreterFailure
};
