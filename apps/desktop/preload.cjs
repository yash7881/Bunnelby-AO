const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('bunnelbyVoice', {
  setRendererSpeaking(isSpeaking) {
    if (typeof isSpeaking !== 'boolean') return;
    ipcRenderer.send('bunnelby:renderer-speaking', isSpeaking);
  },

  onEvent(callback) {
    if (typeof callback !== 'function') return () => {};

    const listener = (_event, payload) => {
      if (payload && typeof payload === 'object') callback(payload);
    };

    ipcRenderer.on('bunnelby:voice-event', listener);

    return () => {
      ipcRenderer.removeListener('bunnelby:voice-event', listener);
    };
  }
});
