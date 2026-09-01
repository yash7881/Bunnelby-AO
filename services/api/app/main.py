import logging

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .acknowledgments import select_spoken_response
from .approval_service import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    approval_public_dict,
    approval_spoken_language,
    approve_and_execute,
    get_approval,
    reject_approval,
)
from .database import SessionLocal
from .intelligence_dispatch import handle_message_result
from .models import Message
from .schemas import (
    ApprovalDecisionResponse,
    ApprovalResponse,
    ChatRequest,
    ChatResponse,
    STTResponse,
    TTSRequest,
)
from .security import (
    ALLOWED_BROWSER_ORIGINS,
    TRUSTED_HOSTS,
    LocalAPISecurityMiddleware,
    api_docs_enabled,
)
from .stt_service import (
    STTAudioError,
    STTDisabledError,
    STTTranscriptionError,
    STTUnavailableError,
    transcribe_audio,
)
from .tts_service import (
    PiperUnavailableError,
    TTSDisabledError,
    TTSSynthesisError,
    VoiceModelMissingError,
    synthesize_speech,
)

logger = logging.getLogger(__name__)

_docs_enabled = api_docs_enabled()
app = FastAPI(
    title="Bunnelby API",
    version="0.6.1",
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

# The desktop renderer is the only browser origin allowed to call the local API. CORS is
# intentionally not an authentication mechanism; LocalAPISecurityMiddleware also blocks
# hostile browser origins/fetch metadata and bounds request bodies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_BROWSER_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
app.add_middleware(LocalAPISecurityMiddleware)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _approval_response(value) -> ApprovalResponse | None:
    if value is None:
        return None
    public = dict(value) if isinstance(value, dict) else approval_public_dict(value)
    # The current desktop App.jsx still gates approval rendering on gmail_reply.
    # Preserve the durable task_type in SQLite; only the initial /chat transport uses
    # this temporary compatibility value. Calendar-specific fields remain present, so
    # ApprovalCard renders the correct Calendar UI. Approval endpoints return the real type.
    if public.get("task_type") in {"gmail_compose", "calendar_event"}:
        public["task_type"] = "gmail_reply"
    return ApprovalResponse(**public)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    user_message = Message(role="user", content=payload.message)
    db.add(user_message)

    result = handle_message_result(payload.message)
    spoken = select_spoken_response(
        payload.message,
        result.action_type,
        preferred_text=result.spoken_reply,
        metadata=result.spoken_metadata,
    )

    assistant_message = Message(role="assistant", content=result.memory_content)
    db.add(assistant_message)
    db.commit()

    raw_latency = result.spoken_metadata.get("latency_ms")
    latency_ms = dict(raw_latency) if isinstance(raw_latency, dict) else None

    return ChatResponse(
        reply=result.reply,
        spoken_reply=spoken.text,
        spoken_ack=spoken.text,
        spoken_language=spoken.language,
        action_type=spoken.action_type,
        approval=_approval_response(result.approval),
        latency_ms=latency_ms,
    )


@app.post("/stt", response_model=STTResponse)
def speech_to_text(
    audio: bytes = Body(..., media_type="application/octet-stream"),
    language: str = Query(default="auto", min_length=2, max_length=4),
    content_type: str | None = Header(default=None, alias="Content-Type"),
) -> STTResponse:
    """Transcribe one short utterance locally using faster-whisper.

    The desktop client sends raw recorded audio bytes. Audio is not stored after transcription.
    """
    try:
        result = transcribe_audio(
            audio,
            content_type=content_type,
            language=language,
        )
    except STTAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except STTDisabledError as exc:
        raise HTTPException(status_code=503, detail="Local speech recognition is disabled.") from exc
    except STTUnavailableError as exc:
        logger.warning("Bunnelby local STT unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Local speech recognition is unavailable.") from exc
    except STTTranscriptionError as exc:
        logger.warning("Bunnelby local STT failed: %s", exc)
        raise HTTPException(status_code=500, detail="Local speech recognition failed.") from exc

    return STTResponse(
        text=result.text,
        language=result.language,
        language_probability=result.language_probability,
        duration_seconds=result.duration_seconds,
    )


@app.get("/approvals/{approval_id}", response_model=ApprovalResponse)
def read_approval(approval_id: int) -> ApprovalResponse:
    if approval_id < 1:
        raise HTTPException(status_code=422, detail="Invalid approval id.")
    try:
        return ApprovalResponse(**approval_public_dict(get_approval(approval_id)))
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Approval not found.") from exc


@app.post("/approvals/{approval_id}/approve", response_model=ApprovalDecisionResponse)
def approve(approval_id: int) -> ApprovalDecisionResponse:
    if approval_id < 1:
        raise HTTPException(status_code=422, detail="Invalid approval id.")
    try:
        result = approve_and_execute(approval_id)
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Approval not found.") from exc
    except ApprovalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    language = approval_spoken_language(result.approval)
    task_type = result.approval.task_type
    spoken_reply = None

    if result.outcome == "created" and task_type == "calendar_event":
        spoken_reply = "इवेंट बना दिया, सर।" if language == "hi" else "Event created, sir."
    elif result.outcome == "already_created" and task_type == "calendar_event":
        spoken_reply = "यह इवेंट पहले ही बनाया जा चुका है।" if language == "hi" else "That event was already created."
    elif result.outcome == "sent":
        spoken_reply = "भेज दिया, सर।" if language == "hi" else "Sent, sir."
    elif result.outcome == "already_sent":
        spoken_reply = "यह ईमेल पहले ही भेजा जा चुका है।" if language == "hi" else "That email was already sent."
    elif result.outcome == "failed":
        spoken_reply = (
            "कार्रवाई पूरी नहीं हुई। स्क्रीन पर अगला कदम देखें।"
            if language == "hi"
            else "The action was not completed. Check the screen for the next step."
        )
    elif result.outcome == "unknown":
        spoken_reply = (
            "पुष्टि स्पष्ट नहीं है। मैं अपने आप दोबारा कोशिश नहीं करूँगा।"
            if language == "hi"
            else "The confirmation is uncertain. I won't retry automatically."
        )

    return ApprovalDecisionResponse(
        approval=ApprovalResponse(**approval_public_dict(result.approval)),
        outcome=result.outcome,
        message=result.message,
        spoken_reply=spoken_reply,
        spoken_language=language,
    )


@app.post("/approvals/{approval_id}/reject", response_model=ApprovalDecisionResponse)
def reject(approval_id: int) -> ApprovalDecisionResponse:
    if approval_id < 1:
        raise HTTPException(status_code=422, detail="Invalid approval id.")
    try:
        result = reject_approval(approval_id)
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Approval not found.") from exc
    except ApprovalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    language = approval_spoken_language(result.approval)
    return ApprovalDecisionResponse(
        approval=ApprovalResponse(**approval_public_dict(result.approval)),
        outcome="rejected",
        message=result.message,
        spoken_reply="कार्रवाई रद्द कर दी।" if language == "hi" else "Discarded.",
        spoken_language=language,
    )


@app.post("/tts", response_class=Response)
def text_to_speech(payload: TTSRequest) -> Response:
    try:
        wav_bytes = synthesize_speech(payload.text, payload.language)
    except TTSDisabledError as exc:
        raise HTTPException(status_code=503, detail="Local voice output is disabled.") from exc
    except (PiperUnavailableError, VoiceModelMissingError) as exc:
        logger.warning("Bunnelby local voice unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Local voice output is unavailable.") from exc
    except TTSSynthesisError as exc:
        logger.warning("Bunnelby local voice synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Local voice synthesis failed.") from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )
