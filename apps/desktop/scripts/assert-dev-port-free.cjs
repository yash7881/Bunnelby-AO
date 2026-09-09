'use strict';

const net = require('net');

const DEV_HOST = '127.0.0.1';
const DEV_PORT = 5173;

const probe = net.createServer();

probe.unref();

probe.once('error', (error) => {
  if (error?.code === 'EADDRINUSE') {
    console.error(
      `[Bunnelby dev] ${DEV_HOST}:${DEV_PORT} is already in use. ` +
      'Close the older Vite/Electron development session before starting AO-main. ' +
      'This prevents Electron from attaching to a stale renderer.'
    );
    process.exitCode = 1;
    return;
  }

  console.error(`[Bunnelby dev] Could not verify dev port ${DEV_PORT}:`, error);
  process.exitCode = 1;
});

probe.once('listening', () => {
  probe.close((error) => {
    if (error) {
      console.error(`[Bunnelby dev] Could not release dev port ${DEV_PORT}:`, error);
      process.exitCode = 1;
      return;
    }

    console.log(`[Bunnelby dev] ${DEV_HOST}:${DEV_PORT} is free.`);
  });
});

probe.listen(DEV_PORT, DEV_HOST);
