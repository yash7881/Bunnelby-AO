import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, LayoutGroup, motion, useReducedMotion } from 'motion/react';
import AOCore from './components/AOCore';
import ApprovalCard from './components/ApprovalCard';
import CommandBar from './components/CommandBar';
import HistoryDrawer from './components/HistoryDrawer';
import ResponseSurface from './components/ResponseSurface';
import { createAOVoicePlayer } from './audio/aoVoicePlayer';

const API_BASE_URL = 'http://127.0.0.1:8000';
const API_URL = `${API_BASE_URL}/chat`;

// Part 10.2 Phase D: one identifier for this desktop chat session. Every /chat
// request carries it so the backend scopes conversational memory to this
// session and never treats an earlier conversation as the active topic.
function createSessionId() {
  const random =
    globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function'
      ? globalThis.crypto.randomUUID().replace(/-/g, '').slice(0, 16)
      : Math.random().toString(16).slice(2).padEnd(16, '0').slice(0, 16);
  return `sess-${random}`;
}
const PROCESSING_HINT_DELAY = 700;

const APPROVAL_FIXTURE = {
  role: 'assistant',
  kind: 'approval',
  content: 'I drafted a reply. Nothing will be sent until you explicitly approve it.',
  title: 'Review Gmail reply',
  approval: {
    id: -1,
    task_type: 'gmail_reply',
    preview_content: 'Hi Rahul,\n\nI’ll review the build tonight and send you my notes afterward.\n\nBest,\nParth',
    target: 'rahul@example.com',
    recipient: 'rahul@example.com',
    subject: 'Re: Project update',
    status: 'pending',
    execution_state: 'not_started',
    created_at: new Date().toISOString(),
    resolved_at: null,
    executed_at: null,
    result_message: null
  }
};

const RESPONSE_FIXTURE = {
  kind: 'normal',
  content: 'Your unread email summary is ready.\n\nTwo messages need attention today. The first asks for confirmation on tomorrow’s project review, and the second contains an updated document for your approval.\n\nThe remaining unread messages are informational and do not appear time-sensitive.'
};

const ERROR_FIXTURE = {
  kind: 'error',
  content: 'Bunnelby could not reach its local service. Start the FastAPI service and try your command again.'
};

const HISTORY_FIXTURE_MESSAGES = [
  { role: 'user', content: 'Check my unread emails', time: '2026-08-29T09:20:00.000Z' },
  { role: 'assistant', content: 'Two unread messages need your attention today. The remaining messages are informational.', time: '2026-08-29T09:20:12.000Z' },
  { role: 'user', content: 'Am I free tomorrow afternoon?', time: '2026-08-29T09:24:00.000Z' },
  { role: 'assistant', content: 'Calendar access is not connected in this phase.', time: '2026-08-29T09:24:05.000Z' }
];

function parseAssistantReply(raw = '') {
  const routeMatch = raw.match(/(?:^|\n)Route:\s*([^\n]+)/i);
  const whyMatch = raw.match(/(?:^|\n)Why:\s*([^\n]+)/i);
  const main = raw
    .replace(/(?:^|\n)Route:\s*[^\n]+/i, '')
    .replace(/(?:^|\n)Why:\s*[^\n]+/i, '')
    .trim();

  return {
    main: main || raw,
    route: routeMatch?.[1]?.trim() || '',
    why: whyMatch?.[1]?.trim() || ''
  };
}

function groupCompletedExchanges(messages) {
  const exchanges = [];
  let pendingUser = null;

  messages.forEach((item) => {
    if (item.role === 'user') {
      pendingUser = item;
      return;
    }

    if (item.role === 'assistant' && pendingUser) {
      exchanges.push({ user: pendingUser, assistant: item });
      pendingUser = null;
    }
  });

  return exchanges;
}

function getDevelopmentSetup() {
  if (!import.meta.env.DEV) return { response: null, messages: [], historyOpen: false };
  const params = new URLSearchParams(window.location.search);
  const fixture = params.get('fixture');
  if (fixture === 'approval') return { response: APPROVAL_FIXTURE, messages: [], historyOpen: false };
  if (fixture === 'response') return { response: RESPONSE_FIXTURE, messages: [], historyOpen: false };
  if (fixture === 'error') return { response: ERROR_FIXTURE, messages: [], historyOpen: false };
  if (fixture === 'history') return { response: null, messages: HISTORY_FIXTURE_MESSAGES, historyOpen: true };
  return { response: null, messages: [], historyOpen: false };
}

async function backendError(response) {
  try {
    const payload = await response.json();
    return String(payload?.detail || payload?.message || `Backend returned ${response.status}`);
  } catch {
    return `Backend returned ${response.status}`;
  }
}

export default function App() {
  // Layout and Core state intentionally remain independent for future voice/TTS events.
  // Stable for the lifetime of this window: one desktop chat session.
  const sessionIdRef = useRef(createSessionId());
  const [initialSetup] = useState(getDevelopmentSetup);
  const [layoutMode, setLayoutMode] = useState(initialSetup.response ? 'response' : 'home');
  const [coreState, setCoreState] = useState('idle');
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState(initialSetup.messages);
  const [activeResponse, setActiveResponse] = useState(initialSetup.response);
  const [sending, setSending] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(initialSetup.historyOpen);
  const [showProcessingHint, setShowProcessingHint] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [voiceCharacter, setVoiceCharacter] = useState({
    mode: 'idle',
    profile: null,
    language: null,
    activeNodes: 0
  });
  const inputRef = useRef(null);
  const processingHintTimerRef = useRef(null);
  const voiceFrameRef = useRef(0);
  const voiceTurnRef = useRef(0);
  const voiceMountedRef = useRef(false);
  const layoutModeRef = useRef(layoutMode);
  const voicePlayerRef = useRef(null);
  const reducedMotion = useReducedMotion();

  layoutModeRef.current = layoutMode;
  if (!voicePlayerRef.current) {
    voicePlayerRef.current = createAOVoicePlayer({
      onSpeakingChange: (isSpeaking) => {
        setCoreState((current) => {
          if (isSpeaking && layoutModeRef.current === 'response') return 'speaking';
          if (!isSpeaking && current === 'speaking') return 'idle';
          return current;
        });
      },
      onAudioLevel: setAudioLevel,
      onCharacterChange: setVoiceCharacter
    });
  }

  const completedExchanges = useMemo(() => {
    const exchanges = groupCompletedExchanges(messages);
    return layoutMode === 'response' && exchanges.length > 0
      ? exchanges.slice(0, -1)
      : exchanges;
  }, [layoutMode, messages]);

  useEffect(() => {
    voiceMountedRef.current = true;
    return () => {
      voiceMountedRef.current = false;
      if (processingHintTimerRef.current) window.clearTimeout(processingHintTimerRef.current);
      if (voiceFrameRef.current) window.cancelAnimationFrame(voiceFrameRef.current);
      const player = voicePlayerRef.current;
      // React Strict Mode immediately replays effects in development. Defer disposal
      // one task so replay cannot permanently disable this persistent player.
      window.setTimeout(() => {
        if (!voiceMountedRef.current && voicePlayerRef.current === player) {
          voicePlayerRef.current = null;
          void player?.dispose();
        }
      }, 0);
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape') setHistoryOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  useEffect(() => {
    const bridge = window.bunnelbyVoice;
    if (!bridge?.onEvent) return undefined;

    return bridge.onEvent((event) => {
      if (!event || typeof event !== 'object') return;

      const eventType = String(event.event || '');

      if (eventType === 'runtime_ready') {
        console.info('Bunnelby persistent voice runtime ready', event);
        return;
      }

      if (eventType === 'state') {
        const voiceState = String(event.state || '').toLowerCase();

        if (voiceState === 'listening') {
          setCoreState('listening');
          return;
        }

        if (voiceState === 'follow_up') {
          setCoreState('listening');
          setSending(false);
          return;
        }

        if (voiceState === 'transcribing' || voiceState === 'thinking') {
          setCoreState('thinking');
          setSending(true);
          return;
        }

        if (voiceState === 'speaking') {
          setCoreState('speaking');
          return;
        }

        if (voiceState === 'standby') {
          setCoreState('idle');
          setSending(false);
        }

        return;
      }

      if (eventType === 'wake_detected') {
        if (voiceFrameRef.current) {
          window.cancelAnimationFrame(voiceFrameRef.current);
          voiceFrameRef.current = 0;
        }
        voicePlayerRef.current?.stop();
        setAudioLevel(0);
        setHistoryOpen(false);
        setMessage('');
        setActiveResponse(null);
        setLayoutMode('home');
        setCoreState('listening');
        return;
      }

      if (eventType === 'user_transcript') {
        const transcript = String(event.text || '').trim();
        if (!transcript) return;

        setHistoryOpen(false);
        setMessage('');
        setActiveResponse(null);
        setLayoutMode('processing');
        setCoreState('thinking');
        setSending(true);
        setMessages((current) => [
          ...current,
          {
            role: 'user',
            content: transcript,
            source: 'voice',
            time: new Date().toISOString()
          }
        ]);
        return;
      }

      if (eventType === 'assistant_response') {
        const rawReply = String(event.reply || '').trim();
        const parsed = parseAssistantReply(rawReply);
        const approval =
          event.approval && typeof event.approval === 'object'
            ? event.approval
            : null;
        const hasApproval =
          approval && approval.task_type === 'gmail_reply';

        const assistantMessage = {
          role: 'assistant',
          kind: hasApproval ? 'approval' : 'normal',
          title: hasApproval ? 'Review Gmail reply' : undefined,
          approval: hasApproval ? approval : undefined,
          approvalBusy: false,
          approvalError: '',
          decisionMessage: '',
          content: parsed.main || rawReply || 'Bunnelby completed the voice request.',
          route: parsed.route,
          why: parsed.why,
          source: 'voice',
          time: new Date().toISOString()
        };

        setMessages((current) => [...current, assistantMessage]);
        setActiveResponse(assistantMessage);
        setLayoutMode('response');
        setSending(false);
        return;
      }

      if (eventType === 'runtime_error' || eventType === 'runtime_exit') {
        console.error('Bunnelby voice runtime:', event.message || eventType);
        setSending(false);
        setCoreState('idle');
      }
    });
  }, []);

  useEffect(() => {
    if (!sending && !historyOpen && layoutMode === 'home') {
      inputRef.current?.focus({ preventScroll: true });
    }
  }, [historyOpen, layoutMode, sending]);

  const prepareVoiceTurn = () => {
    const voiceTurn = voiceTurnRef.current + 1;
    voiceTurnRef.current = voiceTurn;
    if (voiceFrameRef.current) window.cancelAnimationFrame(voiceFrameRef.current);
    voiceFrameRef.current = 0;
    voicePlayerRef.current.stop();
    setAudioLevel(0);
    void voicePlayerRef.current.unlock().catch(() => {});
    return voiceTurn;
  };

  const queueSpeech = (text, language, voiceTurn = voiceTurnRef.current) => {
    const safeText = typeof text === 'string' ? text.trim() : '';
    const safeLanguage = ['en', 'hi'].includes(language) ? language : '';
    if (!safeText || !safeLanguage) return;

    voiceFrameRef.current = window.requestAnimationFrame(() => {
      voiceFrameRef.current = 0;
      if (voiceTurnRef.current !== voiceTurn) return;
      void voicePlayerRef.current.play({ text: safeText, language: safeLanguage });
    });
  };

  const sendCommand = async (command) => {
    const trimmed = command.trim();
    if (!trimmed || sending) return;

    const voiceTurn = prepareVoiceTurn();

    if (processingHintTimerRef.current) window.clearTimeout(processingHintTimerRef.current);
    setShowProcessingHint(false);
    processingHintTimerRef.current = window.setTimeout(
      () => setShowProcessingHint(true),
      PROCESSING_HINT_DELAY
    );

    setHistoryOpen(false);
    setMessage('');
    setActiveResponse(null);
    setLayoutMode('processing');
    setCoreState('thinking');
    setSending(true);
    setMessages((current) => [
      ...current,
      { role: 'user', content: trimmed, time: new Date().toISOString() }
    ]);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, session_id: sessionIdRef.current })
      });

      if (!response.ok) throw new Error(await backendError(response));

      const data = await response.json();
      const parsed = parseAssistantReply(data.reply);
      const hasApproval = data.approval && data.approval.task_type === 'gmail_reply';
      const assistantMessage = {
        role: 'assistant',
        kind: hasApproval ? 'approval' : 'normal',
        title: hasApproval ? 'Review Gmail reply' : undefined,
        approval: hasApproval ? data.approval : undefined,
        approvalBusy: false,
        approvalError: '',
        decisionMessage: '',
        content: parsed.main,
        route: parsed.route,
        why: parsed.why,
        time: new Date().toISOString()
      };

      setMessages((current) => [...current, assistantMessage]);
      setActiveResponse(assistantMessage);
      setLayoutMode('response');
      setCoreState('idle');

      queueSpeech(data.spoken_reply ?? data.spoken_ack, data.spoken_language, voiceTurn);
    } catch (requestError) {
      const errorResponse = {
        role: 'assistant',
        kind: 'error',
        content: requestError?.message || 'Bunnelby could not reach its local service. Start the FastAPI service and try your command again.',
        time: new Date().toISOString()
      };

      setMessages((current) => [...current, errorResponse]);
      setActiveResponse(errorResponse);
      setLayoutMode('response');
      setCoreState('idle');
    } finally {
      if (processingHintTimerRef.current) window.clearTimeout(processingHintTimerRef.current);
      processingHintTimerRef.current = null;
      setShowProcessingHint(false);
      setSending(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    sendCommand(message);
  };

  const toggleListeningPreview = () => {
    if (sending) return;
    prepareVoiceTurn();
    if (coreState === 'listening') {
      setCoreState('idle');
      return;
    }

    setActiveResponse(null);
    setLayoutMode('home');
    setCoreState('listening');
  };

  const settleApprovalFixture = (approved) => {
    setActiveResponse((current) => {
      if (!current?.approval) return current;
      return {
        ...current,
        approval: {
          ...current.approval,
          status: approved ? 'approved' : 'rejected',
          execution_state: approved ? 'completed' : 'not_started',
          resolved_at: new Date().toISOString(),
          executed_at: approved ? new Date().toISOString() : null
        },
        decisionMessage: approved
          ? 'Fixture only: approval recorded. No external action was performed.'
          : 'Fixture only: the action was rejected. No external action was performed.',
        approvalBusy: false,
        approvalError: ''
      };
    });
  };

  const handleApprovalDecision = async (decision) => {
    const approval = activeResponse?.approval;
    if (!approval || activeResponse?.approvalBusy || approval.status !== 'pending') return;

    if (approval.id === -1) {
      settleApprovalFixture(decision === 'approve');
      return;
    }

    const voiceTurn = prepareVoiceTurn();
    setActiveResponse((current) => ({
      ...current,
      approvalBusy: true,
      approvalError: '',
      decisionMessage: ''
    }));
    setCoreState('thinking');

    try {
      const response = await fetch(`${API_BASE_URL}/approvals/${approval.id}/${decision}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!response.ok) throw new Error(await backendError(response));

      const data = await response.json();
      setActiveResponse((current) => ({
        ...current,
        approval: data.approval,
        approvalBusy: false,
        approvalError: '',
        decisionMessage: data.message || ''
      }));
      setMessages((current) => current.map((item) => (
        item.role === 'assistant' && item.approval?.id === approval.id
          ? { ...item, approval: data.approval, decisionMessage: data.message || '' }
          : item
      )));
      setCoreState('idle');
      queueSpeech(data.spoken_reply, data.spoken_language, voiceTurn);
    } catch (error) {
      setActiveResponse((current) => ({
        ...current,
        approvalBusy: false,
        approvalError: error?.message || 'The approval action could not be completed safely.'
      }));
      setCoreState('idle');
    }
  };

  const coreTransition = reducedMotion
    ? { duration: 0.01 }
    : { type: 'spring', stiffness: 220, damping: 30, mass: 0.8 };

  return (
    <main className={`production-shell production-shell--${layoutMode}`}>
      <div className="production-atmosphere" aria-hidden="true" />

      <button
        className="history-trigger"
        type="button"
        aria-label="Open conversation history"
        aria-controls="ao-history"
        aria-expanded={historyOpen}
        onClick={() => setHistoryOpen(true)}
      >
        <span aria-hidden="true"><i /><i /><i /></span>
      </button>

      <LayoutGroup id="ao-production-layout">
        <section className={`experience-stage experience-stage--${layoutMode}`} aria-label="Bunnelby assistant">
          <div className={`core-plane core-plane--${layoutMode}`}>
            <motion.div
              className="core-anchor"
              layout
              initial={false}
              animate={{ scale: layoutMode === 'processing' ? 0.93 : 1, opacity: 1 }}
              transition={coreTransition}
              data-layout-mode={layoutMode}
              data-audio-level={audioLevel.toFixed(4)}
              data-voice-character-mode={voiceCharacter.mode}
              data-voice-character-profile={voiceCharacter.profile || ''}
              data-voice-language={voiceCharacter.language || ''}
              data-voice-character-nodes={voiceCharacter.activeNodes || 0}
            >
              <AOCore
                state={coreState}
                audioLevel={audioLevel}
                size={layoutMode === 'response' ? 'docked' : 'large'}
              />
            </motion.div>
          </div>

          <AnimatePresence mode="wait">
            {layoutMode === 'response' && activeResponse && (
              <motion.div
                className="response-layer"
                key="active-response"
                initial={{ opacity: 0, y: reducedMotion ? 0 : 14 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: reducedMotion ? 0 : -10 }}
                transition={{ duration: reducedMotion ? 0.01 : 0.38, ease: [0.22, 1, 0.36, 1] }}
              >
                <ResponseSurface
                  response={activeResponse}
                  onApprove={() => handleApprovalDecision('approve')}
                  onReject={() => handleApprovalDecision('reject')}
                />
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {layoutMode === 'processing' && showProcessingHint && (
              <motion.p
                className="processing-hint"
                initial={{ opacity: 0, y: reducedMotion ? 0 : 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: reducedMotion ? 0.01 : 0.2 }}
                role="status"
              >
                Working…
              </motion.p>
            )}
          </AnimatePresence>
        </section>
      </LayoutGroup>

      <CommandBar
        inputRef={inputRef}
        message={message}
        onMessageChange={setMessage}
        onSubmit={handleSubmit}
        onMicrophone={toggleListeningPreview}
        isListening={coreState === 'listening'}
        isProcessing={sending}
        layoutMode={layoutMode}
        reducedMotion={reducedMotion}
      />

      <HistoryDrawer
        open={historyOpen}
        exchanges={completedExchanges}
        onClose={() => setHistoryOpen(false)}
        reducedMotion={reducedMotion}
      />
    </main>
  );
}

export { ApprovalCard };
