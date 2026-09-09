const { contextBridge, ipcRenderer } = require('electron');
const {
  normalizeVoiceEventForRenderer
} = require('./voice-control-protocol.cjs');

contextBridge.exposeInMainWorld('bunnelbyVoice', {
  setRendererSpeaking(isSpeaking) {
    if (typeof isSpeaking !== 'boolean') return;
    ipcRenderer.send('bunnelby:renderer-speaking', isSpeaking);
  },

  onEvent(callback) {
    if (typeof callback !== 'function') return () => {};

    const listener = (_event, payload) => {
      if (!payload || typeof payload !== 'object') return;
      const normalized = normalizeVoiceEventForRenderer(payload);
      if (normalized && typeof normalized === 'object') callback(normalized);
    };

    ipcRenderer.on('bunnelby:voice-event', listener);

    return () => {
      ipcRenderer.removeListener('bunnelby:voice-event', listener);
    };
  }
});
