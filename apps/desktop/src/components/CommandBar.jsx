import { useEffect, useState } from 'react';
import { motion } from 'motion/react';

const ACTIVE_RUNTIME_MIC_STATES = new Set(['wake_detected', 'listening', 'follow_up']);
const INACTIVE_RUNTIME_MIC_STATES = new Set(['transcribing', 'thinking', 'speaking', 'standby']);

export default function CommandBar({
  inputRef,
  message,
  onMessageChange,
  onSubmit,
  onMicrophone,
  isListening,
  isProcessing,
  layoutMode,
  reducedMotion
}) {
  const [runtimeMicActive, setRuntimeMicActive] = useState(false);

  // The microphone indicator reflects the authoritative persistent voice runtime,
  // not only App's shared visual core state. Core state also represents thinking,
  // speaking and renderer transitions, so coupling the mic highlight to it can
  // make a real LISTENING/FOLLOW_UP state disappear visually even though the
  // microphone runtime is still active.
  useEffect(() => {
    const bridge = window.bunnelbyVoice;
    if (!bridge?.onEvent) return undefined;

    return bridge.onEvent((event) => {
      if (!event || typeof event !== 'object' || event.event !== 'state') return;

      const runtimeState = String(event.state || '').toLowerCase();
      if (ACTIVE_RUNTIME_MIC_STATES.has(runtimeState)) {
        setRuntimeMicActive(true);
        return;
      }

      if (INACTIVE_RUNTIME_MIC_STATES.has(runtimeState)) {
        setRuntimeMicActive(false);
      }
    });
  }, []);

  const micActive = isListening || runtimeMicActive;
  const placeholder = isProcessing
    ? 'Processing…'
    : layoutMode === 'response'
      ? 'Ask another command…'
      : 'Ask Bunnelby…';

  return (
    <motion.form
      className="command-bar"
      onSubmit={onSubmit}
      aria-label="Command Bunnelby"
      initial={false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reducedMotion ? 0.01 : 0.24 }}
    >
      <button
        className={`command-bar__icon command-bar__mic ${micActive ? 'is-active' : ''}`}
        type="button"
        onClick={onMicrophone}
        aria-label={micActive ? 'Stop listening preview' : 'Preview listening state'}
        aria-pressed={micActive}
        disabled={isProcessing}
        title="Voice wake is active: say Hey Bunnelby or Hello Bunnelby"
      >
        <span className="mic-glyph" aria-hidden="true" />
      </button>

      <input
        ref={inputRef}
        value={message}
        onChange={(event) => onMessageChange(event.target.value)}
        placeholder={placeholder}
        aria-label="Message Bunnelby"
        disabled={isProcessing}
        autoFocus
        autoComplete="off"
      />

      <button
        className="command-bar__icon command-bar__send"
        type="submit"
        disabled={isProcessing || !message.trim()}
        aria-label="Send command"
      >
        <span aria-hidden="true">↑</span>
      </button>
    </motion.form>
  );
}
