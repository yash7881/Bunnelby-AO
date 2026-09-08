'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

function withCrypto(value, fn) {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', {
    value,
    configurable: true,
    writable: true
  });
  return Promise.resolve()
    .then(fn)
    .finally(() => {
      Object.defineProperty(globalThis, 'crypto', original);
    });
}

test('createSessionId never falls back to Math.random', async () => {
  const originalRandom = Math.random;
  let mathRandomCalled = false;
  Math.random = () => {
    mathRandomCalled = true;
    return originalRandom();
  };

  try {
    const { createSessionId } = await import('../src/sessionId.mjs');
    const id = createSessionId();
    assert.match(id, /^sess-[0-9a-f]{16}$/);
    assert.equal(mathRandomCalled, false, 'Math.random must never be called');
  } finally {
    Math.random = originalRandom;
  }
});

test('createSessionId produces distinct ids across calls', async () => {
  const { createSessionId } = await import('../src/sessionId.mjs');
  const ids = new Set([createSessionId(), createSessionId(), createSessionId()]);
  assert.equal(ids.size, 3);
});

test('createSessionId uses getRandomValues when randomUUID is unavailable', async () => {
  let getRandomValuesCalled = false;
  await withCrypto(
    {
      getRandomValues(bytes) {
        getRandomValuesCalled = true;
        for (let i = 0; i < bytes.length; i += 1) {
          bytes[i] = i;
        }
        return bytes;
      }
    },
    async () => {
      const { createSessionId } = await import('../src/sessionId.mjs');
      const id = createSessionId();
      assert.match(id, /^sess-[0-9a-f]{16}$/);
      assert.equal(getRandomValuesCalled, true);
    }
  );
});

test('createSessionId fails explicitly when no secure random source exists', async () => {
  await withCrypto(undefined, async () => {
    const { createSessionId } = await import('../src/sessionId.mjs');
    assert.throws(() => createSessionId(), /no cryptographically secure random source/);
  });
});
