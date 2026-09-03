from __future__ import annotations

import json
import logging
import os
import re
import stat
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Final, Iterable, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser
from cryptography.fernet import InvalidToken
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tzlocal import get_localzone

from .gmail_service import _get_fernet, _oauth_client_file

logger = logging.getLogger(__name__)

CALENDAR_READONLY_SCOPE: Final[str] = "https://www.googleapis.com/auth/calendar.readonly"
CALENDAR_EVENTS_SCOPE: Final[str] = "https://www.googleapis.com/auth/calendar.events"
CALENDAR_SCOPES: Final[list[str]] = [CALENDAR_READONLY_SCOPE, CALENDAR_EVENTS_SCOPE]
DEFAULT_CALENDAR_ID: Final[str] = "primary"
DEFAULT_WORK_HOURS: Final[tuple[int, int]] = (9, 18)
DEFAULT_EVENT_DURATION_MINUTES: Final[int] = 60
DEFAULT_SLOT_DURATION_MINUTES: Final[int] = 30
MAX_SLOT_RESULTS: Final[int] = 20
EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE
)

DAYPART_HOURS: Final[dict[str, tuple[int, int]]] = {
    "morning": (9, 12),
    "afternoon": (12, 17),
    "evening": (17, 20),
    "tonight": (18, 22),
}

WEEKDAYS: Final[dict[str, int]] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

CalendarAction = Literal["free_busy", "open_slots", "create_event"]


class CalendarServiceError(RuntimeError):
    """Base exception for Google Calendar integration failures."""


class CalendarConfigurationError(CalendarServiceError):
    """Raised when Calendar OAuth/API configuration is incomplete."""


class CalendarAuthorizationError(CalendarServiceError):
    """Raised when Calendar OAuth authorization cannot be completed or refreshed."""


class CalendarRateLimitError(CalendarServiceError):
    """Raised when Google Calendar reports a quota/rate-limit condition."""


class CalendarParseError(CalendarServiceError):
    """Raised when a natural-language calendar request cannot be resolved safely."""


class CalendarConflictError(CalendarServiceError):
    """Raised when an event would overlap an existing busy period."""


class CalendarExecutionUncertainError(CalendarServiceError):
    """Raised when Calendar may have accepted a write but confirmation is uncertain."""


@dataclass(frozen=True)
class ParsedCalendarRequest:
    action: CalendarAction
    start: datetime
    end: datetime
    duration_minutes: int
    title: str | None = None
    attendees: tuple[str, ...] = ()
    calendar_id: str = DEFAULT_CALENDAR_ID
    timezone: str = "UTC"
    work_hours: tuple[int, int] = DEFAULT_WORK_HOURS
    assumed_duration: bool = False
    daypart: str | None = None


def _calendar_token_path() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "AO" / "auth" / "calendar_token.enc"
    return Path.home() / ".ao" / "auth" / "calendar_token.enc"


def _try_private_permissions(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        logger.debug("Could not tighten local permissions for %s", path)


def _payload_has_required_scopes(payload: Mapping[str, Any]) -> bool:
    stored = payload.get("scopes") or []
    if isinstance(stored, str):
        stored_scopes = set(stored.split())
    else:
        stored_scopes = {str(item) for item in stored}
    return set(CALENDAR_SCOPES).issubset(stored_scopes)


def _save_calendar_credentials(creds: Credentials) -> None:
    token_path = _calendar_token_path()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    encrypted = _get_fernet().encrypt(creds.to_json().encode("utf-8"))
    token_path.write_bytes(encrypted)
    _try_private_permissions(token_path)


def _load_calendar_credentials() -> Credentials | None:
    token_path = _calendar_token_path()
    if not token_path.exists():
        return None

    try:
        decrypted = _get_fernet().decrypt(token_path.read_bytes())
        payload = json.loads(decrypted.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("OAuth token payload is not an object")
        if not _payload_has_required_scopes(payload):
            logger.info("Saved Calendar token is missing required scopes; reauthorization required.")
            token_path.unlink(missing_ok=True)
            return None
        return Credentials.from_authorized_user_info(payload, CALENDAR_SCOPES)
    except (InvalidToken, json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Bunnelby could not load its saved Calendar token: %s", exc)
        try:
            token_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _run_calendar_oauth() -> Credentials:
    client_file = _oauth_client_file()
    if not client_file.exists():
        raise CalendarConfigurationError(
            "Google OAuth client JSON was not found. Use the same Desktop OAuth client already configured for Gmail."
        )

    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_file), scopes=CALENDAR_SCOPES)
        creds = flow.run_local_server(
            host="127.0.0.1",
            port=0,
            open_browser=True,
            access_type="offline",
            prompt="consent",
            authorization_prompt_message=(
                "Bunnelby needs Google Calendar read access and permission to create only events you explicitly approve. "
                "Your browser should open now."
            ),
            success_message=(
                "Bunnelby Calendar authorization completed. You can close this browser tab and return to the desktop app."
            ),
        )
    except Exception as exc:
        raise CalendarAuthorizationError(
            "Calendar authorization was not completed. Retry and approve calendar.readonly and calendar.events permissions."
        ) from exc

    _save_calendar_credentials(creds)
    return creds


def _get_calendar_credentials() -> Credentials:
    creds = _load_calendar_credentials()
    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_calendar_credentials(creds)
            return creds
        except RefreshError as exc:
            logger.info("Saved Calendar refresh token is no longer usable; reauthorizing: %s", exc)
            try:
                _calendar_token_path().unlink(missing_ok=True)
            except OSError:
                pass

    return _run_calendar_oauth()


def _calendar_service():
    try:
        return build(
            "calendar",
            "v3",
            credentials=_get_calendar_credentials(),
            cache_discovery=False,
        )
    except (CalendarAuthorizationError, CalendarConfigurationError):
        raise
    except Exception as exc:
        raise CalendarServiceError("Bunnelby could not initialize the Google Calendar API client.") from exc


def _http_error_reason(error: HttpError) -> str:
    try:
        payload = json.loads(error.content.decode("utf-8"))
        details = payload.get("error", {}).get("errors", [])
        reasons = [str(item.get("reason", "")) for item in details if item.get("reason")]
        return ", ".join(reasons)
    except Exception:
        return ""


def _translate_http_error(error: HttpError) -> CalendarServiceError:
    status_code = int(getattr(error.resp, "status", 0) or 0)
    reason = _http_error_reason(error)
    if status_code == 429 or reason in {"rateLimitExceeded", "userRateLimitExceeded"}:
        return CalendarRateLimitError("Google Calendar API rate limit reached. Please retry in a moment.")
    if status_code == 401:
        return CalendarAuthorizationError(
            "Google Calendar authorization expired or was revoked. Reconnect Calendar and retry."
        )
    if status_code == 403 and reason in {"insufficientPermissions", "forbidden"}:
        return CalendarAuthorizationError(
            "Google Calendar needs calendar.readonly and calendar.events permissions. Reconnect Calendar and retry."
        )
    if status_code == 403 and reason in {"accessNotConfigured", "serviceDisabled"}:
        return CalendarConfigurationError(
            "Google Calendar API is not enabled for the configured Google Cloud project. Enable Google Calendar API and retry."
        )
    if status_code == 403:
        return CalendarServiceError(
            "Google Calendar denied the request. Confirm the Calendar API, OAuth scopes, and test-user configuration."
        )
    if status_code == 409:
        return CalendarServiceError("Google Calendar reported a duplicate event identifier.")
    return CalendarServiceError(f"Google Calendar API request failed (HTTP {status_code or 'unknown'}).")


def _execute_read(request: Any) -> dict[str, Any]:
    try:
        result = request.execute()
        return result if isinstance(result, dict) else {}
    except HttpError as exc:
        raise _translate_http_error(exc) from exc
    except (CalendarAuthorizationError, CalendarConfigurationError, CalendarRateLimitError):
        raise
    except Exception as exc:
        raise CalendarServiceError("Google Calendar read request failed.") from exc


def local_timezone() -> ZoneInfo:
    configured = os.getenv("BUNNELBY_TIMEZONE", "").strip()
    if configured:
        try:
            return ZoneInfo(configured)
        except ZoneInfoNotFoundError as exc:
            raise CalendarConfigurationError(
                f"BUNNELBY_TIMEZONE is not a valid IANA timezone: {configured}"
            ) from exc
    zone = get_localzone()
    if isinstance(zone, ZoneInfo):
        return zone
    try:
        return ZoneInfo(str(zone))
    except ZoneInfoNotFoundError as exc:
        raise CalendarConfigurationError(
            "Bunnelby could not determine a stable local timezone. Set BUNNELBY_TIMEZONE, for example Asia/Kolkata."
        ) from exc


def _zone_name(zone: ZoneInfo) -> str:
    return getattr(zone, "key", None) or str(zone)


def _aware(value: datetime, zone: ZoneInfo) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=zone)
    return value.astimezone(zone)


def _coerce_datetime(value: datetime | str, zone: ZoneInfo | None = None) -> datetime:
    target_zone = zone or local_timezone()
    if isinstance(value, datetime):
        return _aware(value, target_zone)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise CalendarServiceError(f"Invalid calendar datetime: {value}") from exc
    return _aware(parsed, target_zone)


def _normalize_range(date_range: Any) -> tuple[datetime, datetime, ZoneInfo]:
    zone = local_timezone()
    if isinstance(date_range, Mapping):
        raw_start = date_range.get("start")
        raw_end = date_range.get("end")
    elif isinstance(date_range, Sequence) and not isinstance(date_range, (str, bytes)) and len(date_range) == 2:
        raw_start, raw_end = date_range[0], date_range[1]
    else:
        raise CalendarServiceError("date_range must provide start and end datetimes.")

    if raw_start is None or raw_end is None:
        raise CalendarServiceError("date_range must include both start and end.")
    start = _coerce_datetime(raw_start, zone)
    end = _coerce_datetime(raw_end, zone)
    if end <= start:
        raise CalendarServiceError("Calendar range end must be after start.")
    return start, end, zone


def check_free_busy(date_range: Any) -> list[dict[str, str]]:
    """Return normalized busy periods for the primary calendar in the requested range."""
    start, end, zone = _normalize_range(date_range)
    service = _calendar_service()
    body = {
        "timeMin": start.isoformat(),
        "timeMax": end.isoformat(),
        "timeZone": _zone_name(zone),
        "items": [{"id": DEFAULT_CALENDAR_ID}],
    }
    result = _execute_read(service.freebusy().query(body=body))
    busy = (((result.get("calendars") or {}).get(DEFAULT_CALENDAR_ID) or {}).get("busy") or [])

    normalized: list[dict[str, str]] = []
    for item in busy:
        try:
            busy_start = _coerce_datetime(str(item.get("start", "")), zone)
            busy_end = _coerce_datetime(str(item.get("end", "")), zone)
        except CalendarServiceError:
            continue
        if busy_end > busy_start:
            normalized.append({"start": busy_start.isoformat(), "end": busy_end.isoformat()})
    normalized.sort(key=lambda item: item["start"])
    return normalized


def _merge_busy(
    busy: Iterable[Mapping[str, str]],
    *,
    window_start: datetime,
    window_end: datetime,
    zone: ZoneInfo,
) -> list[tuple[datetime, datetime]]:
    intervals: list[tuple[datetime, datetime]] = []
    for item in busy:
        try:
            start = max(window_start, _coerce_datetime(item["start"], zone))
            end = min(window_end, _coerce_datetime(item["end"], zone))
        except (KeyError, CalendarServiceError):
            continue
        if end > start:
            intervals.append((start, end))
    intervals.sort(key=lambda pair: pair[0])

    merged: list[tuple[datetime, datetime]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def free_windows(
    start: datetime,
    end: datetime,
    busy: Iterable[Mapping[str, str]],
) -> list[tuple[datetime, datetime]]:
    zone = local_timezone()
    start = _aware(start, zone)
    end = _aware(end, zone)
    merged = _merge_busy(busy, window_start=start, window_end=end, zone=zone)
    cursor = start
    windows: list[tuple[datetime, datetime]] = []
    for busy_start, busy_end in merged:
        if busy_start > cursor:
            windows.append((cursor, busy_start))
        cursor = max(cursor, busy_end)
    if cursor < end:
        windows.append((cursor, end))
    return windows


def find_open_slots(
    target_date: date | datetime | str,
    duration_minutes: int,
    work_hours: tuple[int, int] = DEFAULT_WORK_HOURS,
) -> list[dict[str, str]]:
    """Return chronological candidate slots after subtracting Calendar busy periods."""
    if duration_minutes <= 0 or duration_minutes > 24 * 60:
        raise CalendarServiceError("duration_minutes must be between 1 and 1440.")
    start_hour, end_hour = work_hours
    if not (0 <= start_hour < end_hour <= 24):
        raise CalendarServiceError("work_hours must be a valid increasing hour range.")

    zone = local_timezone()
    now = datetime.now(zone)
    if isinstance(target_date, datetime):
        day = _aware(target_date, zone).date()
    elif isinstance(target_date, date):
        day = target_date
    else:
        parsed = dateparser.parse(
            str(target_date),
            settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now.replace(tzinfo=None)},
        )
        if parsed is None:
            raise CalendarParseError("I couldn't determine the date for the open-slot search.")
        day = parsed.date()

    window_start = datetime.combine(day, time(start_hour, 0), tzinfo=zone)
    window_end = datetime.combine(day, time(0, 0), tzinfo=zone) + timedelta(days=1) if end_hour == 24 else datetime.combine(day, time(end_hour, 0), tzinfo=zone)
    if window_end <= now:
        return []
    if window_start < now:
        rounded_minute = ((now.minute + 14) // 15) * 15
        rounded = now.replace(second=0, microsecond=0)
        if rounded_minute >= 60:
            rounded = rounded.replace(minute=0) + timedelta(hours=1)
        else:
            rounded = rounded.replace(minute=rounded_minute)
        window_start = rounded

    busy = check_free_busy((window_start, window_end))
    duration = timedelta(minutes=duration_minutes)
    slots: list[dict[str, str]] = []
    for free_start, free_end in free_windows(window_start, window_end, busy):
        cursor = free_start
        # Keep slot starts aligned to a 15-minute grid for useful human scheduling.
        if cursor.minute % 15:
            cursor = cursor.replace(second=0, microsecond=0) + timedelta(minutes=15 - cursor.minute % 15)
        while cursor + duration <= free_end and len(slots) < MAX_SLOT_RESULTS:
            slots.append({"start": cursor.isoformat(), "end": (cursor + duration).isoformat()})
            cursor += duration
        if len(slots) >= MAX_SLOT_RESULTS:
            break
    return slots


def _validate_attendees(attendees: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in attendees:
        _, address = parseaddr(str(raw).strip())
        address = address.casefold().strip()
        if not address or not EMAIL_PATTERN.fullmatch(address):
            raise CalendarServiceError(f"Invalid attendee email address: {raw}")
        if address not in seen:
            seen.add(address)
            normalized.append(address)
    return normalized


def create_event(
    title: str,
    start: datetime | str,
    end: datetime | str,
    attendees: Iterable[str] = (),
    *,
    calendar_id: str = DEFAULT_CALENDAR_ID,
    timezone_name: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Create one Calendar event. Production callers must invoke this only after durable approval."""
    title = str(title).strip()
    if not title:
        raise CalendarServiceError("Calendar event title is required.")
    if len(title) > 500:
        raise CalendarServiceError("Calendar event title is unexpectedly long.")

    zone = local_timezone() if not timezone_name else ZoneInfo(timezone_name)
    start_dt = _coerce_datetime(start, zone)
    end_dt = _coerce_datetime(end, zone)
    if end_dt <= start_dt:
        raise CalendarServiceError("Calendar event end must be after start.")
    if start_dt <= datetime.now(zone):
        raise CalendarServiceError("Bunnelby will not create a calendar event in the past.")

    attendee_list = _validate_attendees(attendees)
    body: dict[str, Any] = {
        "summary": title,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": _zone_name(zone)},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": _zone_name(zone)},
    }
    if attendee_list:
        body["attendees"] = [{"email": address} for address in attendee_list]
    if event_id:
        candidate = re.sub(r"[^0-9a-v]", "", event_id.casefold())
        if len(candidate) >= 5:
            body["id"] = candidate[:1024]

    service = _calendar_service()
    try:
        result = service.events().insert(
            calendarId=calendar_id or DEFAULT_CALENDAR_ID,
            body=body,
            sendUpdates="all" if attendee_list else "none",
        ).execute()
        return result if isinstance(result, dict) else {}
    except HttpError as exc:
        raise _translate_http_error(exc) from exc
    except (CalendarAuthorizationError, CalendarConfigurationError, CalendarRateLimitError):
        raise
    except Exception as exc:
        raise CalendarExecutionUncertainError(
            "Calendar creation confirmation is uncertain. Bunnelby will not retry automatically."
        ) from exc


def _extract_duration(text: str) -> int | None:
    match = re.search(
        r"\b(\d{1,3})\s*(?:-|\s)?\s*(minutes?|mins?|min|hours?|hrs?|hr)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).casefold()
    minutes = amount * 60 if unit.startswith(("hour", "hr")) else amount
    if not (1 <= minutes <= 24 * 60):
        raise CalendarParseError("Meeting duration must be between 1 minute and 24 hours.")
    return minutes


def _extract_attendees(text: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for match in EMAIL_PATTERN.finditer(text):
        value = match.group(0).casefold()
        if value not in seen:
            seen.add(value)
            values.append(value)
    return tuple(values)


def _extract_daypart(text: str) -> str | None:
    lower = text.casefold()
    for name in DAYPART_HOURS:
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return name
    return None


def _extract_clock(text: str) -> time | None:
    match = re.search(r"\b(?:at\s+)?(\d{1,2})(?::([0-5]\d))?\s*(am|pm)\b", text, re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        if not 1 <= hour <= 12:
            raise CalendarParseError("The requested clock time is invalid.")
        meridiem = match.group(3).casefold()
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
        return time(hour, minute)

    match = re.search(r"\b(?:at\s+)?([01]?\d|2[0-3]):([0-5]\d)\b", text, re.IGNORECASE)
    if match:
        return time(int(match.group(1)), int(match.group(2)))
    return None


def _parse_absolute_date(text: str, now: datetime) -> date | None:
    patterns = (
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        parsed = dateparser.parse(
            match.group(0),
            settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now.replace(tzinfo=None)},
        )
        if parsed:
            return parsed.date()
    return None


def _extract_date(text: str, now: datetime) -> date:
    lower = text.casefold()
    if "day after tomorrow" in lower:
        return now.date() + timedelta(days=2)
    if re.search(r"\btomorrow\b", lower):
        return now.date() + timedelta(days=1)
    if re.search(r"\btoday\b", lower):
        return now.date()

    weekday_match = re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lower,
    )
    if weekday_match:
        target = WEEKDAYS[weekday_match.group(1)]
        days_ahead = (target - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return now.date() + timedelta(days=days_ahead)

    absolute = _parse_absolute_date(text, now)
    if absolute is not None:
        return absolute

    raise CalendarParseError(
        "I couldn't determine the calendar date. Say something like 'tomorrow', 'Monday', or an explicit date."
    )


def _extract_title(text: str) -> str | None:
    titled = re.search(
        r"\b(?:titled|called|named)\s+(.+?)(?=\s+(?:today|tomorrow|on\b|at\b|monday\b|tuesday\b|wednesday\b|thursday\b|friday\b|saturday\b|sunday\b|for\s+\d)|$)",
        text,
        re.IGNORECASE,
    )
    if titled:
        value = titled.group(1).strip(" ,.-")
        return value or None

    match = re.search(
        r"\b(?:schedule|create|add|book)\s+(?:an?\s+)?(?:calendar\s+)?(?:event\s+|meeting\s+|appointment\s+)?(.+?)(?=\s+(?:today\b|tomorrow\b|on\b|at\b|monday\b|tuesday\b|wednesday\b|thursday\b|friday\b|saturday\b|sunday\b|for\s+\d)|$)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = EMAIL_PATTERN.sub("", match.group(1)).strip(" ,.-")
    value = re.sub(r"^(?:a\s+)?(?:meeting|event|appointment)\s*$", "", value, flags=re.IGNORECASE).strip()
    return value or None


def _is_create_request(text: str) -> bool:
    return bool(re.search(r"\b(?:schedule|book|create|add|set\s*up)\b", text, re.IGNORECASE))


def _is_open_slot_request(text: str) -> bool:
    return bool(
        re.search(r"\b(?:slot|slots|opening|openings|available\s+time|free\s+time)\b", text, re.IGNORECASE)
        or re.search(r"\bfind\b.{0,50}\b(?:free|available)\b", text, re.IGNORECASE)
    )


def parse_calendar_request(
    user_message: str,
    *,
    now: datetime | None = None,
    timezone: ZoneInfo | None = None,
    force_action: CalendarAction | None = None,
    fallback_title: str | None = None,
    fallback_duration_minutes: int | None = None,
) -> ParsedCalendarRequest:
    """Deterministically parse common Calendar requests; fail closed when exact scheduling is ambiguous.

    `force_action` is the Part 10.2 Phase G hinge. Left unset the parser behaves
    exactly as before and derives the action class from the wording -- the legacy
    text-routed path. When the Brain has already chosen the class, the caller
    passes it here and the wording may then only fill in FIELDS (date, clock,
    title, duration, attendees). A create can no longer decay into a free/busy
    read, and read vocabulary such as "book" or "schedule" inside an availability
    question can no longer promote itself into a create.

    `fallback_title` / `fallback_duration_minutes` carry validated request data
    the Brain already supplied, for the case where the deterministic extractors
    cannot recover them from the wording ("put the dentist appointment on my
    calendar" has no create verb for _extract_title to anchor on).
    """
    text = user_message.strip()
    if not text:
        raise CalendarParseError("A calendar request is required.")

    zone = timezone or local_timezone()
    current = _aware(now or datetime.now(zone), zone)
    target_day = _extract_date(text, current)
    explicit_clock = _extract_clock(text)
    daypart = _extract_daypart(text)
    parsed_duration = _extract_duration(text) or fallback_duration_minutes
    attendees = _extract_attendees(text)

    if force_action is None:
        wants_create = _is_create_request(text)
        wants_open_slots = _is_open_slot_request(text)
    else:
        wants_create = force_action == "create_event"
        wants_open_slots = force_action == "open_slots"

    if wants_create:
        title = _extract_title(text) or (fallback_title or "").strip() or None
        if not title:
            raise CalendarParseError("Tell me the event title before I prepare a calendar creation approval.")
        if explicit_clock is None:
            raise CalendarParseError(
                "Tell me the exact event start time, for example 'tomorrow at 3 PM'. A broad daypart is not precise enough to create an event."
            )
        duration = parsed_duration or DEFAULT_EVENT_DURATION_MINUTES
        assumed = parsed_duration is None
        start = datetime.combine(target_day, explicit_clock, tzinfo=zone)
        end = start + timedelta(minutes=duration)
        if start <= current:
            raise CalendarParseError("Bunnelby will not prepare an event whose start time is in the past.")
        return ParsedCalendarRequest(
            action="create_event",
            start=start,
            end=end,
            duration_minutes=duration,
            title=title,
            attendees=attendees,
            timezone=_zone_name(zone),
            work_hours=DEFAULT_WORK_HOURS,
            assumed_duration=assumed,
            daypart=daypart,
        )

    if wants_open_slots:
        duration = parsed_duration or DEFAULT_SLOT_DURATION_MINUTES
        assumed = parsed_duration is None
        work_hours = DAYPART_HOURS.get(daypart or "", DEFAULT_WORK_HOURS)
        start = datetime.combine(target_day, time(work_hours[0], 0), tzinfo=zone)
        end = datetime.combine(target_day, time(work_hours[1], 0), tzinfo=zone)
        return ParsedCalendarRequest(
            action="open_slots",
            start=start,
            end=end,
            duration_minutes=duration,
            timezone=_zone_name(zone),
            work_hours=work_hours,
            assumed_duration=assumed,
            daypart=daypart,
        )

    duration = parsed_duration or DEFAULT_EVENT_DURATION_MINUTES
    if explicit_clock is not None:
        start = datetime.combine(target_day, explicit_clock, tzinfo=zone)
        end = start + timedelta(minutes=duration)
    elif daypart:
        start_hour, end_hour = DAYPART_HOURS[daypart]
        start = datetime.combine(target_day, time(start_hour, 0), tzinfo=zone)
        end = datetime.combine(target_day, time(end_hour, 0), tzinfo=zone)
    else:
        start = datetime.combine(target_day, time(DEFAULT_WORK_HOURS[0], 0), tzinfo=zone)
        end = datetime.combine(target_day, time(DEFAULT_WORK_HOURS[1], 0), tzinfo=zone)

    if end <= current:
        raise CalendarParseError("That calendar time range is already in the past.")
    if start < current < end:
        start = current.replace(second=0, microsecond=0)

    return ParsedCalendarRequest(
        action="free_busy",
        start=start,
        end=end,
        duration_minutes=duration,
        timezone=_zone_name(zone),
        work_hours=(start.hour, end.hour),
        assumed_duration=parsed_duration is None and explicit_clock is not None,
        daypart=daypart,
    )


def _format_time(value: datetime) -> str:
    return value.strftime("%I:%M %p").lstrip("0")


def _format_date(value: datetime) -> str:
    return value.strftime("%A, %B %d").replace(" 0", " ")


def format_free_busy_response(request: ParsedCalendarRequest, busy: list[dict[str, str]]) -> str:
    zone = ZoneInfo(request.timezone)
    start = request.start.astimezone(zone)
    end = request.end.astimezone(zone)
    label = f"{_format_date(start)} from {_format_time(start)} to {_format_time(end)}"
    if not busy:
        return f"You're free on {label}."

    windows = free_windows(start, end, busy)
    if not windows:
        return f"You're busy throughout {label}."

    busy_text = ", ".join(
        f"{_format_time(_coerce_datetime(item['start'], zone))}–{_format_time(_coerce_datetime(item['end'], zone))}"
        for item in busy[:5]
    )
    free_text = ", ".join(f"{_format_time(a)}–{_format_time(b)}" for a, b in windows[:5])
    return f"Busy: {busy_text}. Free windows: {free_text}."


def format_open_slots_response(request: ParsedCalendarRequest, slots: list[dict[str, str]]) -> str:
    if not slots:
        return f"I couldn't find a {request.duration_minutes}-minute opening in that time window."
    zone = ZoneInfo(request.timezone)
    times = [
        _format_time(_coerce_datetime(item["start"], zone))
        for item in slots[:5]
    ]
    assumption = " I used a 30-minute default because no duration was specified." if request.assumed_duration else ""
    return (
        f"I found {len(slots)} available {request.duration_minutes}-minute slot"
        f"{'s' if len(slots) != 1 else ''}. First options: {', '.join(times)}.{assumption}"
    )


def calendar_event_proposal(request: ParsedCalendarRequest) -> dict[str, Any]:
    if request.action != "create_event" or not request.title:
        raise CalendarParseError("This calendar request is not a complete event proposal.")
    return {
        "title": request.title,
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "timezone": request.timezone,
        "attendees": list(request.attendees),
        "calendar_id": request.calendar_id,
        "duration_minutes": request.duration_minutes,
        "assumed_duration": request.assumed_duration,
    }
