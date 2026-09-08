'use strict';

const RENDERER_SPEAKING_CONTROL_TYPE = 'renderer_speaking';
const UNCLEAR_TRANSCRIPT_RUNTIME_ERROR = 'conversation STT returned no text';

function encodeRendererSpeakingControl(isSpeaking) {
  if (typeof isSpeaking !== 'boolean') {
    throw new TypeError('renderer speaking state must be boolean');
  }
  return `${JSON.stringify({
    type: RENDERER_SPEAKING_CONTROL_TYPE,
    speaking: isSpeaking
  })}\n`;
}

function normalizeVoiceEventForRenderer(payload) {
  if (!payload || typeof payload !== 'object') return payload;

  if (
    payload.event === 'runtime_error' &&
    payload.message === UNCLEAR_TRANSCRIPT_RUNTIME_ERROR
  ) {
    return {
      event: 'assistant_response',
      reply: "I didn't catch that clearly. Say 'Hey Bunnelby' and try again.",
      spoken_reply: "I didn't catch that clearly. Say Hey Bunnelby and try again.",
      spoken_language: 'en',
      action_type: 'clarification_required'
    };
  }

  return payload;
}

module.exports = {
  RENDERER_SPEAKING_CONTROL_TYPE,
  UNCLEAR_TRANSCRIPT_RUNTIME_ERROR,
  encodeRendererSpeakingControl,
  normalizeVoiceEventForRenderer
};
