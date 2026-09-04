'use strict';

const RENDERER_SPEAKING_CONTROL_TYPE = 'renderer_speaking';

function encodeRendererSpeakingControl(isSpeaking) {
  if (typeof isSpeaking !== 'boolean') {
    throw new TypeError('renderer speaking state must be boolean');
  }
  return `${JSON.stringify({
    type: RENDERER_SPEAKING_CONTROL_TYPE,
    speaking: isSpeaking
  })}\n`;
}

module.exports = {
  RENDERER_SPEAKING_CONTROL_TYPE,
  encodeRendererSpeakingControl
};
