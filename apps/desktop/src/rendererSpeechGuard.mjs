export const DEFAULT_RENDERER_SPEAKER_TAIL_MS = 750;

export function createRendererSpeechGuard({
  clock = () => performance.now(),
  tailMs = DEFAULT_RENDERER_SPEAKER_TAIL_MS
} = {}) {
  const safeTailMs = Math.max(0, Number(tailMs) || 0);
  let speaking = false;
  let blockedUntil = 0;

  const now = () => {
    const value = Number(clock());
    return Number.isFinite(value) ? value : 0;
  };

  return {
    setSpeaking(isSpeaking) {
      if (typeof isSpeaking !== 'boolean') return false;
      speaking = isSpeaking;
      blockedUntil = isSpeaking ? Number.POSITIVE_INFINITY : now() + safeTailMs;
      return true;
    },

    isActive() {
      return speaking || now() < blockedUntil;
    },

    snapshot() {
      return {
        speaking,
        blockedUntil,
        tailMs: safeTailMs,
        active: speaking || now() < blockedUntil
      };
    }
  };
}
