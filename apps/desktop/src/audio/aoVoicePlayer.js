import {
  createCleanFallbackGraph,
  createVoiceCharacterGraph,
  disconnectVoiceCharacterGraph
} from './aoVoiceCharacter';

const TTS_URL = 'http://127.0.0.1:8000/tts';
const RMS_NOISE_FLOOR = 0.015;
const RMS_GAIN = 8;
const LEVEL_ATTACK = 0.35;
const LEVEL_DECAY = 0.65;
const SELF_WAKE_ARM_DELAY_MS = 80;

const wait = (milliseconds) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function disconnect(node) {
  try {
    node?.disconnect();
  } catch {
    // A node may already be disconnected during rapid interruption.
  }
}

export function createAOVoicePlayer({
  onSpeakingChange,
  onAudioLevel,
  onCharacterChange,
  characterConfig,
  forceCharacterFailure = false
} = {}) {
  let audioContext = null;
  let abortController = null;
  let sourceNode = null;
  let analyserNode = null;
  let voiceGraph = null;
  let amplitudeFrame = 0;
  let generation = 0;
  let speaking = false;
  let smoothedLevel = 0;
  let disposed = false;
  let characterState = {
    active: false,
    mode: 'idle',
    profile: null,
    language: null,
    amount: 0,
    activeNodes: 0
  };

  const notifySpeaking = (nextSpeaking) => {
    if (disposed) return;
    if (speaking === nextSpeaking) return;
    speaking = nextSpeaking;
    onSpeakingChange?.(nextSpeaking);
  };

  const resetLevel = () => {
    smoothedLevel = 0;
    if (disposed) return;
    onAudioLevel?.(0);
  };

  const notifyCharacter = (nextState) => {
    characterState = { ...characterState, ...nextState };
    if (!disposed) onCharacterChange?.({ ...characterState });
  };

  const clearGraph = () => {
    if (amplitudeFrame) cancelAnimationFrame(amplitudeFrame);
    amplitudeFrame = 0;
    if (sourceNode) sourceNode.onended = null;
    disconnect(sourceNode);
    disconnectVoiceCharacterGraph(voiceGraph);
    sourceNode = null;
    analyserNode = null;
    if (voiceGraph) {
      notifyCharacter({ active: false, activeNodes: 0 });
    }
    voiceGraph = null;
  };

  const ensureAudioContext = async () => {
    if (disposed) throw new Error('AO voice player is disposed.');
    if (!audioContext) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error('Web Audio is unavailable.');
      audioContext = new AudioContextClass();
    }
    if (audioContext.state === 'suspended') await audioContext.resume();
    return audioContext;
  };

  const stop = () => {
    generation += 1;
    abortController?.abort();
    abortController = null;
    if (sourceNode) {
      sourceNode.onended = null;
      try {
        sourceNode.stop();
      } catch {
        // The source may have ended between the interruption and cleanup.
      }
    }
    clearGraph();
    resetLevel();
    notifySpeaking(false);
  };

  const trackAmplitude = (playbackGeneration) => {
    if (playbackGeneration !== generation || !analyserNode) return;

    const waveform = new Uint8Array(analyserNode.fftSize);
    const sample = () => {
      if (playbackGeneration !== generation || !analyserNode) return;

      analyserNode.getByteTimeDomainData(waveform);
      let sum = 0;
      for (let index = 0; index < waveform.length; index += 1) {
        const normalized = (waveform[index] - 128) / 128;
        sum += normalized * normalized;
      }

      const rms = Math.sqrt(sum / waveform.length);
      const level = Math.min(1, Math.max(0, (rms - RMS_NOISE_FLOOR) * RMS_GAIN));
      smoothedLevel = smoothedLevel * LEVEL_DECAY + level * LEVEL_ATTACK;
      onAudioLevel?.(smoothedLevel < 0.004 ? 0 : smoothedLevel);
      amplitudeFrame = requestAnimationFrame(sample);
    };

    sample();
  };

  const play = async ({
    text,
    language,
    characterProfile,
    characterAmount,
    characterEnabled,
    forceDspFailure = false
  }) => {
    if (disposed) return false;
    stop();
    if (!text || !['en', 'hi'].includes(language)) return false;

    const playbackGeneration = generation;
    abortController = new AbortController();

    try {
      const response = await fetch(TTS_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language }),
        signal: abortController.signal
      });
      if (!response.ok) throw new Error(`Local voice returned ${response.status}.`);

      const wavData = await response.arrayBuffer();
      if (playbackGeneration !== generation) return false;

      const context = await ensureAudioContext();
      const audioBuffer = await context.decodeAudioData(wavData.slice(0));
      if (playbackGeneration !== generation) return false;

      sourceNode = context.createBufferSource();
      sourceNode.buffer = audioBuffer;

      try {
        voiceGraph = createVoiceCharacterGraph(context, sourceNode, {
          language,
          enabled: characterEnabled ?? characterConfig?.enabled,
          profile: characterProfile ?? characterConfig?.profile,
          amount: characterAmount ?? characterConfig?.amount,
          forceFailure: forceCharacterFailure || forceDspFailure
        });
      } catch (characterError) {
        console.warn(
          `AO voice character unavailable; using clean Piper: ${characterError?.message || 'unknown error'}`
        );
        voiceGraph = createCleanFallbackGraph(context, sourceNode, language);
      }

      analyserNode = voiceGraph.analyserNode;
      notifyCharacter({
        active: true,
        mode: voiceGraph.mode,
        profile: voiceGraph.profile,
        language,
        amount: voiceGraph.settings.amount,
        activeNodes: voiceGraph.nodes.length
      });

      sourceNode.onended = () => {
        if (playbackGeneration !== generation) return;
        clearGraph();
        resetLevel();
        notifySpeaking(false);
      };

      // Signal Electron/Python before any speaker samples can exist. The tiny
      // arm delay gives the persistent microphone loop at least one control/frame
      // boundary to observe suppression before WebAudio starts.
      notifySpeaking(true);
      await wait(SELF_WAKE_ARM_DELAY_MS);
      if (playbackGeneration !== generation || !sourceNode) return false;

      sourceNode.start();
      trackAmplitude(playbackGeneration);
      return true;
    } catch (error) {
      if (error?.name !== 'AbortError' && playbackGeneration === generation) {
        console.warn(`AO voice unavailable: ${error?.message || 'unknown error'}`);
      }
      if (playbackGeneration === generation) {
        clearGraph();
        resetLevel();
        notifySpeaking(false);
      }
      return false;
    } finally {
      if (playbackGeneration === generation) abortController = null;
    }
  };

  const dispose = async () => {
    stop();
    disposed = true;
    const context = audioContext;
    audioContext = null;
    if (context && context.state !== 'closed') await context.close();
  };

  return {
    play,
    stop,
    unlock: ensureAudioContext,
    dispose,
    isSpeaking: () => speaking,
    getState: () => ({
      speaking,
      audioLevel: smoothedLevel,
      character: { ...characterState }
    })
  };
}
