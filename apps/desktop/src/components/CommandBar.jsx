import { motion } from 'motion/react';

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
        className={`command-bar__icon command-bar__mic ${isListening ? 'is-active' : ''}`}
        type="button"
        onClick={onMicrophone}
        aria-label={isListening ? 'Stop listening preview' : 'Preview listening state'}
        aria-pressed={isListening}
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
