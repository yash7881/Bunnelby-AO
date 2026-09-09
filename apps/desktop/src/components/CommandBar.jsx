import { motion } from 'motion/react';

// Presentational only. The mic highlight is decided by App's authoritative
// voice-runtime state (see src/voiceMicState.mjs) and arrives as `micActive`.
// This component deliberately holds no voice state and opens no event
// subscription of its own: the previous `isListening || runtimeMicActive`
// arrangement had two independent writers for one pixel, and they disagreed.
export default function CommandBar({
  inputRef,
  message,
  onMessageChange,
  onSubmit,
  onMicrophone,
  micActive,
  isProcessing,
  layoutMode,
  reducedMotion
}) {
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
