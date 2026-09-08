// Part 10.2 Phase D: one identifier for this desktop chat session. Every /chat
// request carries it so the backend scopes conversational memory to this
// session and never treats an earlier conversation as the active topic.
//
// The identifier MUST be unpredictable: it is trusted to scope memory to a
// session, so a guessable ID would let one session's history bleed into
// another. Only the Web Crypto API is used -- never a non-cryptographic PRNG,
// a timestamp, a process id, or a hand-rolled generator, all of which are
// predictable or low-entropy. If no cryptographically secure source exists,
// generation fails loudly instead of silently falling back to something
// guessable.

const HEX_ID_LENGTH = 16;
const RANDOM_BYTES = HEX_ID_LENGTH / 2;

function toHex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export function createSessionId() {
  const cryptoObj = globalThis.crypto;
  if (cryptoObj && typeof cryptoObj.randomUUID === 'function') {
    return `sess-${cryptoObj.randomUUID().replace(/-/g, '').slice(0, HEX_ID_LENGTH)}`;
  }
  if (cryptoObj && typeof cryptoObj.getRandomValues === 'function') {
    const bytes = new Uint8Array(RANDOM_BYTES);
    cryptoObj.getRandomValues(bytes);
    return `sess-${toHex(bytes)}`;
  }
  throw new Error(
    'createSessionId: no cryptographically secure random source (globalThis.crypto) is available'
  );
}
