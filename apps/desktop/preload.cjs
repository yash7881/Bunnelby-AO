const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('bunnelbyVoice', {
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
