const { app, BrowserWindow, session } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const DEV_ORIGIN = 'http://127.0.0.1:5173';
const VOICE_EVENT_PREFIX = 'BUNNELBY_UI_EVENT ';

let mainWindow = null;
let voiceProcess = null;
let rendererReady = false;
let pendingVoiceEvents = [];

function isTrustedRendererUrl(value) {
  try {
    const url = new URL(value);
    const trustedHost = url.hostname === '127.0.0.1' || url.hostname === 'localhost';
    return url.protocol === 'http:' && trustedHost && url.port === '5173';
  } catch {
    return false;
  }
}

function configureSessionSecurity() {
  session.defaultSession.setPermissionCheckHandler((webContents, permission, requestingOrigin, details) => {
    if (permission !== 'media') return false;
    if (!isTrustedRendererUrl(requestingOrigin || webContents?.getURL?.() || '')) return false;
    const mediaTypes = Array.isArray(details?.mediaTypes) ? details.mediaTypes : [];
    return mediaTypes.length > 0 && mediaTypes.every((type) => type === 'audio');
  });

  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback, details) => {
    if (permission !== 'media') {
      callback(false);
      return;
    }

    const pageUrl = webContents?.getURL?.() || '';
    const mediaTypes = Array.isArray(details?.mediaTypes) ? details.mediaTypes : [];
    const audioOnly = mediaTypes.length > 0 && mediaTypes.every((type) => type === 'audio');
    callback(isTrustedRendererUrl(pageUrl) && audioOnly);
  });
}

function configureWebContentsSecurity() {
  app.on('web-contents-created', (_event, contents) => {
    contents.setWindowOpenHandler(() => ({ action: 'deny' }));

    contents.on('will-navigate', (event, targetUrl) => {
      if (!isTrustedRendererUrl(targetUrl)) event.preventDefault();
    });

    contents.on('will-attach-webview', (event) => {
      event.preventDefault();
    });
  });
}

function sendVoiceEvent(payload) {
  if (!payload || typeof payload !== 'object') return;

  if (
    rendererReady &&
    mainWindow &&
    !mainWindow.isDestroyed() &&
    !mainWindow.webContents.isDestroyed()
  ) {
    mainWindow.webContents.send('bunnelby:voice-event', payload);
    return;
  }

  pendingVoiceEvents.push(payload);
  if (pendingVoiceEvents.length > 50) pendingVoiceEvents.shift();
}

function flushVoiceEvents() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const queued = pendingVoiceEvents;
  pendingVoiceEvents = [];
  queued.forEach((payload) => {
    mainWindow.webContents.send('bunnelby:voice-event', payload);
  });
}

function resolveVoiceRuntime() {
  const repoRoot = path.resolve(__dirname, '..', '..');
  const runtimeScript = path.join(
    repoRoot,
    'scripts',
    'wakeword',
    'wake_conversation_runtime.py'
  );

  const configuredPython = (process.env.BUNNELBY_PYTHON || '').trim();
  const venvPython = path.join(
    repoRoot,
    '.venv',
    'Scripts',
    'python.exe'
  );

  return {
    repoRoot,
    runtimeScript,
    pythonExecutable:
      configuredPython ||
      (fs.existsSync(venvPython) ? venvPython : 'python')
  };
}

function startVoiceRuntime() {
  if (voiceProcess || process.env.BUNNELBY_VOICE_BRIDGE === '0') return;

  const {
    repoRoot,
    runtimeScript,
    pythonExecutable
  } = resolveVoiceRuntime();

  if (!fs.existsSync(runtimeScript)) {
    sendVoiceEvent({
      event: 'runtime_error',
      message: `Voice runtime script not found: ${runtimeScript}`
    });
    return;
  }

  voiceProcess = spawn(
    pythonExecutable,
    [
      runtimeScript,
      '--turns', '0',
      '--language', 'auto'
    ],
    {
      cwd: repoRoot,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        PYTHONUTF8: '1',
        PYTHONIOENCODING: 'utf-8'
      },
      stdio: ['ignore', 'pipe', 'pipe']
    }
  );

  const lines = readline.createInterface({
    input: voiceProcess.stdout,
    crlfDelay: Infinity
  });

  lines.on('line', (line) => {
    if (!app.isPackaged) console.log(`[Bunnelby Voice] ${line}`);

    if (!line.startsWith(VOICE_EVENT_PREFIX)) return;

    try {
      const payload = JSON.parse(line.slice(VOICE_EVENT_PREFIX.length));
      sendVoiceEvent(payload);
    } catch (error) {
      console.error('Invalid Bunnelby voice event:', error);
    }
  });

  voiceProcess.stderr.on('data', (chunk) => {
    console.error(`[Bunnelby Voice stderr] ${String(chunk).trim()}`);
  });

  voiceProcess.on('error', (error) => {
    sendVoiceEvent({
      event: 'runtime_error',
      message: `Voice runtime could not start: ${error.message}`
    });
  });

  voiceProcess.on('exit', (code, signal) => {
    const expectedShutdown = app.isQuitting === true;
    voiceProcess = null;

    if (!expectedShutdown) {
      sendVoiceEvent({
        event: 'runtime_exit',
        code,
        signal,
        message: `Voice runtime stopped${code == null ? '' : ` with code ${code}`}.`
      });
    }
  });
}

function stopVoiceRuntime() {
  if (!voiceProcess) return;
  const processToStop = voiceProcess;
  voiceProcess = null;

  try {
    processToStop.kill();
  } catch (error) {
    console.error('Failed to stop Bunnelby voice runtime:', error);
  }
}

const createWindow = () => {
  rendererReady = false;

  mainWindow = new BrowserWindow({
    width: 960,
    height: 700,
    minWidth: 720,
    minHeight: 520,
    title: 'Bunnelby',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      allowRunningInsecureContent: false,
      webviewTag: false,
      navigateOnDragDrop: false,
      devTools: !app.isPackaged
    }
  });

  mainWindow.webContents.on('did-start-loading', () => {
    rendererReady = false;
  });

  mainWindow.webContents.on('did-finish-load', () => {
    rendererReady = true;
    flushVoiceEvents();
  });

  mainWindow.on('closed', () => {
    rendererReady = false;
    mainWindow = null;
  });

  mainWindow.loadURL(DEV_ORIGIN);
};

configureWebContentsSecurity();

app.whenReady().then(() => {
  configureSessionSecurity();
  createWindow();
  startVoiceRuntime();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('before-quit', () => {
  app.isQuitting = true;
  stopVoiceRuntime();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
