import os
import json
import sqlite3
import time
import uuid
import re
import random
import math
import base64
import threading
import traceback
from pathlib import Path
import requests
from datetime import datetime, timezone, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather, Dial
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from openai import OpenAI

# Optional voice-agent dependencies. The app stays deployable without them;
# PREVOLT_VOICE_AGENT_ENABLED must remain false until these are installed.
try:
    from flask_sock import Sock
except Exception:  # pragma: no cover - deploy-time optional dependency
    Sock = None

try:
    import websocket as websocket_client
except Exception:  # pragma: no cover - deploy-time optional dependency
    websocket_client = None



def convo_key_from_request(is_call: bool) -> str:
    frm = (request.values.get("From") or "").strip()
    if frm:
        return frm
    return (request.values.get("CallSid") if is_call else request.values.get("MessageSid")) or "unknown"

# ---------------------------------------------------
# SAFE ZoneInfo Import (Fallback)
# ---------------------------------------------------
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ---------------------------------------------------
# Environment & Clients
# ---------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")

SQUARE_ACCESS_TOKEN = os.environ.get("SQUARE_ACCESS_TOKEN")
SQUARE_LOCATION_ID = os.environ.get("SQUARE_LOCATION_ID")
SQUARE_TEAM_MEMBER_ID = os.environ.get("SQUARE_TEAM_MEMBER_ID")
# ---------------------------------------------------
# Square service variation data (final and verified)
# ---------------------------------------------------
SERVICE_VARIATION_EVAL_ID = "IPCUF6EPOYGWJUEFUZOXL2AZ"
SERVICE_VARIATION_EVAL_VERSION = 1764725435505

SERVICE_VARIATION_INSPECTION_ID = "LYK646AH4NAESCFUZL6PUTZ2"
SERVICE_VARIATION_INSPECTION_VERSION = 1764725393938

SERVICE_VARIATION_TROUBLESHOOT_ID = "64IQNJYO3H6XNTLPIHABDJOQ"
SERVICE_VARIATION_TROUBLESHOOT_VERSION = 1762464315698

BOOKING_START_HOUR = 9
BOOKING_END_HOUR = 16
MAX_TRAVEL_MINUTES = 60

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
DISPATCH_ORIGIN_ADDRESS = os.environ.get("DISPATCH_ORIGIN_ADDRESS")
TECH_CURRENT_ADDRESS = os.environ.get("TECH_CURRENT_ADDRESS")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = (
    Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
    else None
)

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}

# ---------------------------------------------------
# Realtime Voice Agent Flags
# ---------------------------------------------------
# Fail-closed by default. Turning this off returns option 1 to the existing
# voicemail -> transcription -> SMS flow without touching the SRB list.
PREVOLT_VOICE_AGENT_ENABLED = _env_bool("PREVOLT_VOICE_AGENT_ENABLED", False)
PREVOLT_VOICE_AGENT_FAILOVER_TO_VOICEMAIL = _env_bool("PREVOLT_VOICE_AGENT_FAILOVER_TO_VOICEMAIL", True)
PREVOLT_PUBLIC_BASE_URL = (os.environ.get("PREVOLT_PUBLIC_BASE_URL") or "").strip().rstrip("/")
OPENAI_REALTIME_MODEL = (os.environ.get("OPENAI_REALTIME_MODEL") or "gpt-realtime").strip()
OPENAI_REALTIME_VOICE = (os.environ.get("OPENAI_REALTIME_VOICE") or "echo").strip()
OPENAI_REALTIME_WS_URL = (os.environ.get("OPENAI_REALTIME_WS_URL") or "wss://api.openai.com/v1/realtime").strip()
VOICE_AGENT_MAX_SECONDS = int(os.environ.get("VOICE_AGENT_MAX_SECONDS") or "300")
VOICE_AGENT_IDLE_SECONDS = int(os.environ.get("VOICE_AGENT_IDLE_SECONDS") or "30")
VOICE_AGENT_MAX_CUSTOMER_TURNS = int(os.environ.get("VOICE_AGENT_MAX_CUSTOMER_TURNS") or "16")
VOICE_AGENT_HANGUP_SMS_ENABLED = _env_bool("VOICE_AGENT_HANGUP_SMS_ENABLED", True)
VOICE_AGENT_THINKING_SOUND_ENABLED = _env_bool("VOICE_AGENT_THINKING_SOUND_ENABLED", True)
VOICE_AGENT_THINKING_SOUND_MS = int(os.environ.get("VOICE_AGENT_THINKING_SOUND_MS") or "650")
VOICE_AGENT_VAD_SILENCE_MS = int(os.environ.get("VOICE_AGENT_VAD_SILENCE_MS") or "300")
VOICE_AGENT_VAD_THRESHOLD = float(os.environ.get("VOICE_AGENT_VAD_THRESHOLD") or "0.35")
VOICE_RESIDENTIAL_MAX_TRAVEL_MINUTES = int(os.environ.get("VOICE_RESIDENTIAL_MAX_TRAVEL_MINUTES") or str(MAX_TRAVEL_MINUTES or 60))

RULES_FILE = os.environ.get("PREVOLT_RULES_FILE") or os.environ.get("PREVOLT_RULES_PATH")

def load_rule_matrix_text() -> str:
    """Load the existing SRB matrix from repo or env-configured path. No invented rules."""
    candidates = [
        RULES_FILE,
        "prevolt_rules.json",
        "./prevolt_rules.json",
        str(Path(__file__).resolve().parent / "prevolt_rules.json"),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(candidate)
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload, dict):
                rules = payload.get("rules", "")
                if isinstance(rules, str):
                    return rules.strip()
            if isinstance(payload, str):
                return payload.strip()
        except Exception as e:
            print(f"[WARN] rule load failed for {candidate}: {e!r}")

    print("[WARN] No rule matrix file found.")
    return ""

RULE_MATRIX_TEXT = load_rule_matrix_text()


# -------------------------------
# Small shared helpers
# -------------------------------
def humanize_question(core_question: str) -> str:
    """Lightweight wrapper used in route-level prompts.
    The richer variant logic lives inside generate_reply_for_inbound()."""
    core_question = (core_question or "").strip()
    return core_question

# ---------------------------------------------------
# Pre-Routing / Non-Service Thread Guards
# ---------------------------------------------------
EMPLOYMENT_RESUME_EMAIL = "prevoltelectric@gmail.com"

def _intent_text(*parts) -> str:
    """Normalize text for deterministic routing checks."""
    joined = " ".join(str(p or "") for p in parts)
    return re.sub(r"\s+", " ", joined).strip().lower()


# ---------------------------------------------------
# Production Safety Guards — customer intent first
# ---------------------------------------------------
def _loose_bool_text(text: str) -> str:
    return re.sub(r"[^a-z0-9@.+#\-\s]", " ", _intent_text(text))


def looks_like_invalid_voicemail_transcript(text: str) -> bool:
    """Detect prompt leaks / non-voicemails before any SMS is sent."""
    low = _intent_text(text)
    if not low:
        return True
    prompt_leaks = [
        "please provide the voicemail transcription",
        "please provide the transcript",
        "please provide the voicemail",
        "the voicemail transcription you want cleaned up",
        "the voicemail transcription you would like me to clean up",
        "i can help clean up",
        "as an ai",
    ]
    if any(p in low for p in prompt_leaks):
        return True

    # Abandoned voicemail / no actual lead. Do not text these people asking for an address.
    compact = re.sub(r"[^a-z0-9]+", " ", low).strip()
    non_messages = {
        "bye", "bye bye", "goodbye", "hello", "hello hello", "test", "testing",
        "um", "uh", "no message", "nothing", "wrong number",
    }
    if compact in non_messages:
        return True

    service_signal = re.search(
        r"\b(?:electric|electrician|outlet|switch|panel|breaker|light|lighting|fan|charger|ev|wire|wiring|power|smoke|detector|quote|estimate|install|repair|replace|service|call back|appointment|schedule)\b",
        low,
        flags=re.I,
    )
    has_phone_or_address = bool(re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", low)) or bool(extract_service_address_from_text(text))
    if len(compact.split()) <= 3 and not service_signal and not has_phone_or_address:
        return True

    return False


def looks_like_vendor_sales_or_spam(*parts) -> bool:
    """Vendor/sales/prospecting messages must never enter service scheduling."""
    low = _intent_text(*parts)
    if not low:
        return False

    strong = [
        "jobber", "field service pros", "save a few hours of paperwork",
        "stop chasing payments", "software that helps pros", "custom checklists",
        "lead fees", "limited spots", "2 min apply", "vendor call",
        "dlmpropertygroup.com", "selected your company for jobs", "website link",
        "sales rep", "marketing agency", "marketing company", "seo services", "merchant services",
        # Vendor / demo outreach.
        "field service software", "quick demo", "show you a demo",
        # Business loan / funding spam. Keep these here so financing spam is
        # stopped before the address gate tries to treat it like a service lead.
        "prequalified for up to", "pre-qualified for up to", "pre approved for up to", "pre-approved for up to",
        "working capital", "business funding", "business loan", "small business loan",
        "merchant cash advance", "funding specialist", "business line of credit",
        "funds may be available", "funds available", "access to more capital",
    ]
    if any(t in low for t in strong):
        return True

    # Finance spam often avoids a fixed phrase but has the same structure:
    # business + capital/funding + quick money + phone/reply CTA.
    finance_signal = bool(re.search(
        r"\b(?:capital|funding|loan|cash advance|line of credit|credit line)\b",
        low,
        flags=re.I,
    ))
    quick_money_signal = bool(re.search(
        r"\b(?:pre\s*-?\s*qualified|pre\s*-?\s*approved|up to\s*\$?\s*\d{2,4}\s*k|"
        r"as soon as tomorrow|one business day|same day funding|funds? (?:may be|are) available)\b",
        low,
        flags=re.I,
    ))
    sales_cta_signal = bool(re.search(
        r"\b(?:reply\s+(?:yes|y)|call\s+\d{3,4}[-.\s]?\d{3}[-.\s]?\d{4}|"
        r"speak with (?:a|an) .*?(?:specialist|advisor|representative)|more information)\b",
        low,
        flags=re.I,
    ))
    business_context = "business" in low or "company" in low or "owner" in low
    if finance_signal and quick_money_signal and (sales_cta_signal or business_context):
        return True

    if "apply" in low and ("vendor" in low or "property management" in low or "lead" in low or "limited spots" in low):
        return True
    return False

def detect_customer_hard_stop(text: str) -> str | None:
    """Customer's newest message can close the automation regardless of pending_step."""
    low = _intent_text(text)
    if not low:
        return None
    exact = {
        "cancel", "disregard", "never mind", "nevermind", "forget it", "stop", "unsubscribe",
        "no thanks", "no thank you", "pass", "i'll pass", "ill pass", "i will pass", "thanks anyway", "leave me alone"
    }
    if low in exact:
        return "customer_stop"
    phrases = [
        "i found someone", "found someone", "found somebody", "found another electrician",
        "i am all set", "i'm all set", "all set", "we are all set", "we're all set",
        "do not schedule", "don't schedule", "dont schedule",
        "do not book", "don't book", "dont book",
        "i don't want to schedule", "i dont want to schedule",
        "i do not want to schedule", "i don't want an appointment", "i dont want an appointment",
        "no longer need", "don't need service", "dont need service",
        "call someone else", "going with someone else",
        "not interested", "i'm not interested", "im not interested", "not interested anymore",
        "no longer interested", "leave me alone", "please stop", "stop texting", "stop messaging",
        "don't contact me", "dont contact me", "do not contact me", "thanks anyway", "no thanks", "i will pass",
    ]
    if any(p in low for p in phrases):
        return "customer_cancelled_or_found_someone"
    return None


def apply_customer_hard_stop(conv: dict, reason: str, inbound_text: str = "") -> str:
    sched = conv.setdefault("sched", {})
    conv["thread_type"] = "closed_lost"
    sched["customer_hard_stop"] = True
    sched["booking_allowed"] = False
    sched["pending_step"] = None
    sched["manual_only"] = False
    sched["follow_up_due"] = False
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    sched["closed_reason"] = reason
    sched["closed_text"] = _safe_monitor_text(inbound_text, 240) if "_safe_monitor_text" in globals() else inbound_text
    return "No problem. We’ll stop the scheduling messages."



# ---------------------------------------------------
# Flow Lab V4 / V13 backend hardening helpers
# ---------------------------------------------------
def v13_looks_like_smoke_detector_install_request(*parts) -> bool:
    low = _intent_text(*parts)
    if not low:
        return False
    detector = any(x in low for x in [
        "smoke detector", "smoke detectors", "smoke alarm", "smoke alarms",
        "co detector", "co detectors", "carbon monoxide detector", "carbon monoxide detectors"
    ])
    install_context = any(x in low for x in [
        "install", "installed", "replace", "replaced", "hardwire", "hardwired",
        "quote", "estimate", "need smoke", "need co", "need carbon monoxide", "change", "changed"
    ])
    actual_hazard = any(x in low for x in [
        "smoke from", "smoke coming", "burning", "burning smell", "sparking",
        "fire department", "melted", "panel is wet", "water in panel", "hot breaker"
    ])
    return detector and install_context and not actual_hazard


def v13_looks_like_callback_request(*parts) -> bool:
    low = _intent_text(*parts)
    if not low:
        return False
    phrases = [
        "call me", "call back", "call me back", "give me a call", "please call",
        "can someone call", "could someone call", "rather talk", "talk to someone",
        "talk on the phone", "phone call", "call instead", "can you call"
    ]
    return any(p in low for p in phrases)


def v13_apply_callback_request(conv: dict, inbound_text: str = "") -> str:
    sched = conv.setdefault("sched", {})
    conv["thread_type"] = "needs_callback"
    sched["manual_only"] = True
    sched["manual_assist"] = True
    sched["booking_allowed"] = False
    sched["pending_step"] = None
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    sched["manual_reason"] = "customer_requested_callback"
    sched["closed_reason"] = "customer_requested_callback"
    return "Thanks, we got your message. One of our team members will call you back."


def v13_extract_email(text: str) -> str | None:
    m = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", text or "", flags=re.I)
    return m.group(1).strip() if m else None


def v13_save_email(conv: dict, email: str) -> None:
    if not email:
        return
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    profile["active_email"] = email
    profile["email"] = email
    sched["email"] = email


def v13_format_slot_choice_prompt(sched: dict) -> str:
    opts = (sched or {}).get("offered_slot_options") or []
    if opts:
        return f"Which one works best for you — {_format_slot_options(opts[:3])}?"
    return "What day and time works best for you?"


def v13_looks_like_real_person_question(text: str) -> bool:
    low = _loose_text(text)
    if not low:
        return False
    return any(p in low for p in [
        "is this a real person", "are you a real person", "is this ai", "are you ai",
        "is this automated", "is this a bot", "are you a bot", "human"
    ])


def v13_real_person_reply(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    answer = "Yes — this is Prevolt Electric. We use this number to keep scheduling organized while we’re in the field."
    if sched.get("awaiting_slot_offer_choice") and sched.get("offered_slot_options"):
        return f"{answer} {v13_format_slot_choice_prompt(sched)}"
    return answer


def v13_maybe_imply_address_confirmed(conv: dict, inbound_text: str) -> bool:
    """If we asked 'You're at X, right?' and the customer keeps scheduling, treat it as a yes."""
    sched = conv.setdefault("sched", {})
    if sched.get("address_verified"):
        return False
    if starts_with_negative_address_correction(inbound_text) or _intent_text(inbound_text) in {"no", "nope", "nah", "incorrect", "wrong"}:
        return False
    raw = (sched.get("raw_address") or sched.get("address_candidate") or "").strip()
    if not raw or not _address_has_house_number_and_street(raw):
        return False
    missing = (sched.get("address_missing") or "").strip().lower()
    pending = (sched.get("pending_step") or "").strip().lower()
    if missing not in {"confirm", "state", ""} and pending not in {"need_address", ""}:
        return False
    if not (_looks_like_scheduling_reply(inbound_text) or v13_extract_email(inbound_text)):
        return False
    set_raw_address_safe(sched, raw)
    raw = sched.get("raw_address") or raw
    sched["address_candidate"] = raw
    sched["address_verified"] = True
    sched["address_missing"] = None
    if sched.get("pending_step") == "need_address":
        sched["pending_step"] = None
    try:
        update_address_assembly_state(sched)
    except Exception:
        pass
    return True


def v13_handle_email_before_slot(conv: dict, inbound_text: str) -> str | None:
    """Preserve email and ask only for the missing scheduling choice; do not reset the flow."""
    email = v13_extract_email(inbound_text)
    if not email:
        return None
    sched = conv.setdefault("sched", {})
    v13_save_email(conv, email)
    try:
        recompute_pending_step(conv.setdefault("profile", {}), sched)
    except Exception:
        pass
    if not (sched.get("scheduled_date") and sched.get("scheduled_time")):
        if sched.get("awaiting_slot_offer_choice") and sched.get("offered_slot_options"):
            return f"Got it, I have your email. {v13_format_slot_choice_prompt(sched)}"
        return "Got it, I have your email. What day and time works best for you?"
    return None


def v13_time_only_on_offered_slots(conv: dict, inbound_text: str) -> bool:
    """Bind bare time replies like '9', '4', or 'can you do 3 instead' to the first offered date."""
    sched = conv.setdefault("sched", {})
    if not sched.get("awaiting_slot_offer_choice"):
        return False
    options = sched.get("offered_slot_options") or []
    if not options:
        return False
    txt = (inbound_text or "").strip()
    low = _intent_text(txt)
    if not low:
        return False
    # Do not steal ordinal choices. maybe_apply_offered_slot_selection handles those.
    if re.fullmatch(r"[123]", low):
        return False
    time_only = bool(re.fullmatch(r"(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?", low)) or bool(re.search(r"\b(?:can|could)\s+you\s+do\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", low))
    if not time_only:
        return False
    explicit_time = extract_explicit_time_from_text(txt)
    if not explicit_time:
        return False
    chosen = dict(options[0])
    chosen["time"] = explicit_time
    try:
        d = datetime.strptime(str(chosen.get("date") or ""), "%Y-%m-%d")
        chosen["label"] = f"{d.strftime('%A, %B %d').replace(' 0', ' ')} at {humanize_time(explicit_time)}"
    except Exception:
        chosen["label"] = f"{chosen.get('date') or 'that day'} at {humanize_time(explicit_time)}"
    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195").upper()
    sched["appointment_type"] = appt
    conv["appointment_type"] = appt
    sched["scheduled_date"] = chosen.get("date")
    sched["scheduled_time"] = chosen.get("time")
    sched["scheduled_time_source"] = "customer_time_on_first_offered_day"
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    sched["final_confirmation_sent"] = False
    sched["final_confirmation_accepted"] = False
    sched["last_final_confirmation_key"] = None
    sched["slot_choice_locked"] = True
    sched["booking_attempt_nonce"] = str(uuid.uuid4())
    return True


def address_gate_blocks_scheduling(conv: dict) -> bool:
    """Hard stop: scheduling/slot logic must not run while address is still unverified."""
    sched = conv.setdefault("sched", {})
    try:
        update_address_assembly_state(sched)
        recompute_pending_step(conv.setdefault("profile", {}), sched)
    except Exception:
        pass

    if sched.get("address_verified"):
        return False
    if sched.get("customer_hard_stop") or sched.get("manual_only"):
        return False
    if (conv.get("thread_type") or "") in {"closed_lost", "manual_only", "vendor_sales_or_spam", "wrong_number_or_spam"}:
        return False

    raw = (sched.get("raw_address") or sched.get("address_candidate") or "").strip()
    missing = (sched.get("address_missing") or "").strip().lower()
    pending = (sched.get("pending_step") or "").strip().lower()
    return bool(pending == "need_address" or missing in {"street", "number", "confirm", "state", "partial", "missing"} or not raw)


def build_address_gate_reply(conv: dict) -> str:
    """Clear stale scheduling state and ask only for address confirmation/details."""
    sched = conv.setdefault("sched", {})
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["scheduled_date"] = None
    sched["scheduled_time"] = None
    sched["final_confirmation_sent"] = False
    sched["final_confirmation_accepted"] = False
    sched["last_final_confirmation_key"] = None
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    try:
        update_address_assembly_state(sched)
    except Exception:
        pass

    raw = (sched.get("raw_address") or sched.get("address_candidate") or "").strip(" ,.")
    missing = (sched.get("address_missing") or "").strip().lower()

    # If we have a plausible full street/town candidate from voicemail, confirm it.
    if raw and missing in {"confirm", "state"} and _address_has_house_number_and_street(raw):
        return f"You're at {raw}, right?"

    # Otherwise ask for the missing address cleanly.
    try:
        return build_address_prompt(sched)
    except Exception:
        return "What’s the address for the work?"



def _looks_like_scheduling_reply(text: str) -> bool:
    """True if a reply appears to be a scheduling choice/request that should be deferred until address is verified."""
    low = _intent_text(text)
    if not low:
        return False
    if any(w in low for w in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow", "morning", "afternoon", "noon"]):
        return True
    if re.search(r"\b\d{1,2}[:.]?\d{0,2}\s*(?:am|pm)\b", low):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", low):
        return True
    if low in {"first", "second", "third", "option 1", "option 2", "option 3", "1", "2", "3", "anytime", "any time", "whenever"}:
        return True
    return False


def _confirm_address_acceptance_should_run(conv: dict, inbound_text: str) -> bool:
    """Catch yes/correct replies to address confirmation before address gate can loop."""
    if not yes_text(inbound_text):
        return False
    sched = conv.setdefault("sched", {})
    if last_outbound_asked_address_confirmation(conv):
        return True
    if (sched.get("address_missing") or "").strip().lower() == "confirm" and (sched.get("raw_address") or sched.get("address_candidate")):
        return True
    if (sched.get("pending_step") or "").strip().lower() == "need_address" and (sched.get("raw_address") or sched.get("address_candidate")):
        return True
    return False


def handle_address_confirmation_acceptance(conv: dict, inbound_text: str) -> str | None:
    """
    If we asked "You're at X, right?" and the customer says yes, verify the
    address and continue. This prevents yes from being treated as another
    address problem, and it resumes any scheduling text that was deferred while
    the address gate was active.
    """
    if not _confirm_address_acceptance_should_run(conv, inbound_text):
        return None

    sched = conv.setdefault("sched", {})
    raw = (sched.get("raw_address") or sched.get("address_candidate") or "").strip(" ,.")
    if not raw:
        return None

    sched["raw_address"] = raw
    sched["address_verified"] = True
    sched["address_missing"] = None
    sched["pending_step"] = None
    sched["booking_allowed"] = True
    if not sched.get("appointment_type"):
        sched["appointment_type"] = "EVAL_195"
    conv["appointment_type"] = sched.get("appointment_type")

    try:
        try_early_address_normalize(sched)
        update_address_assembly_state(sched)
        # The customer just confirmed the address explicitly, so the confirmation wins.
        sched["address_verified"] = True
        sched["address_missing"] = None
    except Exception:
        pass

    # If a date/time was sent while the address still needed confirmation,
    # resume that request now that the address is verified.
    deferred = (sched.pop("deferred_schedule_text", None) or "").strip()
    if deferred:
        try:
            exact_reply = maybe_handle_exact_slot_before_step4(conv, conv.get("phone") or sched.get("phone") or "", deferred)
            if exact_reply:
                return exact_reply
        except Exception as e:
            print("[WARN] deferred schedule after address confirmation failed:", repr(e))

    # If no deferred schedule exists, move to the normal price + three-day availability handoff.
    try:
        return "Got it. " + build_price_and_availability_prompt(conv, sched.get("appointment_type") or "EVAL_195")
    except Exception:
        return "Got it. What day and time works best for you?"

def last_outbound_asked_address_confirmation(conv: dict) -> bool:
    last = _intent_text((conv or {}).get("last_sms_body") or "")
    if not last:
        return False

    # The humanized first voicemail opener usually says:
    # "You're at 45 Dickerman Ave, Windsor Locks, right?"
    # Treat that as an address confirmation question even though it does not
    # literally contain "is that correct". This prevents a customer reply like
    # "No, 34 Dickerman Ave" from skipping the apology/correction branch.
    explicit_confirm = (
        "is that correct" in last
        or "is this for that address" in last
        or "is this the correct address" in last
        or "is that the correct address" in last
    )
    human_confirm = ("youre at" in last or "you're at" in last) and ("right" in last or "correct" in last)
    has_address_context = (
        "address" in last
        or "i have" in last
        or "on file" in last
        or "for the visit" in last
        or "for the work" in last
        or "youre at" in last
        or "you're at" in last
    )
    return (explicit_confirm or human_confirm) and has_address_context


def is_negative_answer(text: str) -> bool:
    return _intent_text(text) in {"no", "nope", "nah", "incorrect", "not correct", "wrong", "that's wrong", "thats wrong"}


def starts_with_negative_address_correction(text: str) -> bool:
    """True when the customer rejects the heard address and provides the corrected one in the same reply."""
    low = _intent_text(text)
    return bool(re.match(r"^(no|nope|nah|incorrect|wrong|not correct|that's wrong|thats wrong)\b", low))


def extract_address_from_negative_correction(text: str) -> str | None:
    """Extract address from replies like 'No, 34 Dickerman Ave Windsor Locks.'"""
    raw = str(text or "").strip()
    if not raw or not starts_with_negative_address_correction(raw):
        return None
    cleaned = re.sub(r"^\s*(?:no|nope|nah|incorrect|wrong|not correct|that's wrong|thats wrong)\b\s*[,.;:-]?\s*", "", raw, flags=re.I).strip()
    # Prefer the text after the rejection, but fall back to the whole reply if needed.
    candidate = safe_address_candidate_from_text(cleaned) or safe_address_candidate_from_text(raw)
    return candidate

def _address_text_has_location_tail(value: str) -> bool:
    """True when an extracted street address appears to already include town/state/zip context."""
    s = " ".join(str(value or "").split()).strip()
    if not s:
        return False
    if re.search(r"\b(?:CT|C\.?T\.?|Connecticut|MA|M\.?A\.?|Massachusetts|Mass\.?)\b", s, flags=re.I):
        return True
    if re.search(r"\b\d{5}(?:-\d{4})?\b", s):
        return True
    # If there are words after the street suffix, treat them as likely town context.
    suffix = r"st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace|pl|place|park"
    return bool(re.search(rf"\b(?:{suffix})\b\s*,?\s+(?:in\s+)?[A-Za-z][A-Za-z .'\-]+$", s, flags=re.I))


def _location_context_from_text(text: str) -> tuple[str | None, str | None, str | None]:
    """Pull town/state/zip from a prior address-confirmation text such as '2 Main Street, Windsor, right?'"""
    raw = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not raw:
        return (None, None, None)

    # Use the same address extractor first so we look at the address-shaped section, not the whole prompt.
    addr = None
    try:
        addr = extract_service_address_from_text(raw)
    except Exception:
        addr = None
    sample = addr or raw

    suffix = r"st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace|pl|place|park"
    m = re.search(
        rf"\b\d{{1,6}}(?:-\d{{1,6}})?[A-Za-z]?\s+[A-Za-z0-9.'#\- ]+?\b(?:{suffix})\b\s*,?\s*(?:in\s+)?(?P<tail>[A-Za-z][A-Za-z .'\-]{{1,40}})?",
        sample,
        flags=re.I,
    )
    tail = (m.group("tail") or "").strip(" ,.;") if m else ""
    if tail:
        tail = re.split(
            r"\b(?:right|correct|is that|for the work|for the visit|thank|thanks|please|can you|could you|our evaluation|we would|i have)\b",
            tail,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" ,.;")

    state = None
    if re.search(r"\b(?:CT|C\.?T\.?|Connecticut|Conn\.?)\b", sample, flags=re.I):
        state = "CT"
    elif re.search(r"\b(?:MA|M\.?A\.?|Massachusetts|Mass\.?)\b", sample, flags=re.I):
        state = "MA"

    zip_match = re.search(r"\b(\d{5}(?:-\d{4})?)\b", sample)
    zipc = zip_match.group(1) if zip_match else None

    # Guard against junk tails.
    if tail:
        bad = {"right", "correct", "the work", "the visit", "you", "your project"}
        if tail.lower() in bad or len(tail) < 2:
            tail = ""
    return (tail or None, state, zipc)


def inherit_location_context_for_corrected_address(conv: dict, corrected_address: str) -> str:
    """
    If the customer corrects only the street/number after we asked
    'You're at 2 Main Street, Windsor, right?', inherit the town/state from
    the address we just asked about. This prevents the bot from asking
    'What town is that in?' after 'No, 1 Main St.'
    """
    corrected = " ".join(str(corrected_address or "").split()).strip(" ,.;")
    if not corrected:
        return corrected
    if _address_text_has_location_tail(corrected):
        return corrected

    sched = conv.setdefault("sched", {})
    norm = sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None
    if norm:
        city = (norm.get("locality") or "").strip()
        state = (norm.get("administrative_district_level_1") or "").strip()
        zipc = (norm.get("postal_code") or "").strip()
        if city:
            tail = " ".join(x for x in [city, state, zipc] if x)
            return f"{corrected} {tail}".strip()

    candidates = [
        sched.get("rejected_address"),
        sched.get("raw_address"),
        sched.get("address_candidate"),
        conv.get("last_sms_body"),
        conv.get("initial_sms"),
        conv.get("cleaned_transcript"),
    ]
    for candidate in candidates:
        city, state, zipc = _location_context_from_text(candidate or "")
        if city:
            tail = " ".join(x for x in [city, state, zipc] if x)
            return f"{corrected} {tail}".strip()

    return corrected


def should_handle_negative_address_correction(conv: dict, inbound_text: str) -> bool:
    """Broader than last_outbound_asked_address_confirmation; catches 'No, 1 Main St' reliably."""
    if not starts_with_negative_address_correction(inbound_text):
        return False
    if last_outbound_asked_address_confirmation(conv):
        return True
    sched = conv.setdefault("sched", {})
    last = _intent_text(conv.get("last_sms_body"))
    # If we have any address state and the customer starts with no + address,
    # this is almost certainly a correction to our guessed/heard address.
    return bool(
        sched.get("raw_address")
        or sched.get("address_candidate")
        or sched.get("normalized_address")
        or ("right" in last and ("youre at" in last or "you're at" in last or "address" in last))
    )


def handle_address_confirmation_rejection(conv: dict, inbound_text: str) -> str | None:
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})

    corrected_address = extract_address_from_negative_correction(inbound_text)
    if corrected_address and should_handle_negative_address_correction(conv, inbound_text):
        if sched.get("raw_address"):
            sched["rejected_address"] = sched.get("raw_address")
        corrected_address = inherit_location_context_for_corrected_address(conv, corrected_address)
        set_raw_address_safe(sched, corrected_address)
        corrected_address = sched.get("raw_address") or corrected_address
        sched["normalized_address"] = None
        sched["address_verified"] = True
        sched["address_missing"] = None
        sched["pending_step"] = None
        sched["booking_allowed"] = True
        try:
            addresses = profile.setdefault("addresses", [])
            if corrected_address not in addresses:
                addresses.append(corrected_address)
        except Exception:
            pass
        try:
            try_early_address_normalize(sched)
            update_address_assembly_state(sched)
            # A corrected address sent directly by the customer should be trusted even
            # if Google normalization only marks it as a raw street/town candidate.
            sched["address_verified"] = True
            sched["address_missing"] = None
        except Exception:
            pass
        # Address correction into scheduling is always a normal evaluation unless
        # a prior emergency/inspection classification already exists.
        if not sched.get("appointment_type"):
            sched["appointment_type"] = "EVAL_195"
        conv["appointment_type"] = sched.get("appointment_type")
        availability = build_price_and_availability_prompt(conv, sched.get("appointment_type") or "EVAL_195")
        return f"Sorry about that, the voicemail was a little tough to hear. I have it now. {availability}"

    if not last_outbound_asked_address_confirmation(conv):
        return None

    if not is_negative_answer(inbound_text):
        return None

    if sched.get("raw_address"):
        sched["rejected_address"] = sched.get("raw_address")
    sched["raw_address"] = None
    sched["normalized_address"] = None
    sched["address_verified"] = False
    sched["address_missing"] = "street"
    sched["pending_step"] = "need_address"
    sched["booking_allowed"] = True
    return "Sorry about that, the voicemail was a little tough to hear. Can you give me the address again?"


def _address_has_house_number_and_street(value: str) -> bool:
    low = f" {(value or '').lower()} "
    suffixes = (
        " st ", " street ", " ave ", " avenue ", " rd ", " road ", " ln ", " lane ",
        " dr ", " drive ", " ct ", " court ", " cir ", " circle ", " blvd ", " boulevard ",
        " way ", " pkwy ", " parkway ", " ter ", " terrace ", " park "
    )
    return bool(re.match(r"^\s*\d{1,6}(?:-\d{1,6})?[A-Za-z]?\b", value or "")) and any(s in low for s in suffixes)


def safe_address_candidate_from_text(text: str) -> str | None:
    addr = extract_service_address_from_text(text)
    if addr:
        addr = scrub_non_address_tail(addr)
    if addr and is_plausible_address_text(addr):
        return addr
    return None

def is_google_lsa_platform_notice(text: str) -> bool:
    """True for Google LSA wrapper/instruction notices that are not customer-authored content."""
    low = _intent_text(text)
    if not low:
        return False
    return (
        "google local services ads" in low
        or "local services ads" in low
        or "g.co/homeservices" in low
        or "lsa dashboard" in low
        or "replies to this number will be sent to the customer" in low
        or "this customer requested only message replies" in low
        or "respond via your lsa dashboard" in low
        or "reply here or respond via" in low
    )

def extract_lsa_customer_message(text: str) -> str | None:
    """
    Extract the true customer-authored message from Google LSA wrapper text.
    Returns None for pure platform notices with no customer message payload.
    """
    raw = str(text or "").strip()
    if not raw:
        return None

    m = re.search(r"\bMessage:\s*(.+)$", raw, flags=re.I | re.S)
    if not m:
        return None

    msg = m.group(1).strip()
    # Remove common truncation markers and dashboard fragments.
    msg = re.sub(r"\s*\[\.\.\.\]\s*$", "", msg).strip()
    msg = re.split(r"\b(?:Reply here|Replies to this number|respond via LSA dashboard|https://g\.co/homeservices)\b", msg, maxsplit=1, flags=re.I)[0].strip()
    return msg or None

def is_pure_lsa_platform_notice(text: str) -> bool:
    """Wrapper/instruction only. These must update metadata at most, never generate a customer reply."""
    return is_google_lsa_platform_notice(text) and not extract_lsa_customer_message(text)

def looks_like_employment_inquiry(*parts) -> bool:
    low = _intent_text(*parts)
    if not low:
        return False

    positive = [
        "looking for work", "looking for some work", "looking for a job",
        "looking for employment", "employment inquiry", "job inquiry",
        "interested in hiring", "are you hiring", "you hiring",
        "hiring electricians", "hire electricians", "join your team",
        "joining your team", "apprenticeship", "apprentice position",
        "resume", "my resume", "license status", "licensed electrician",
        "unlicensed electrician", "journeyman", "journey man"
    ]
    # Avoid confusing a customer asking to hire Prevolt with a job seeker.
    customer_hire_context = [
        "hire you", "hire prevolt", "hire an electrician to come", "hire someone to come",
        "need to hire an electrician for my house", "looking to hire an electrician for my house"
    ]
    return any(p in low for p in positive) and not any(p in low for p in customer_hire_context)

def looks_like_commercial_bid_context(*parts) -> bool:
    low = _intent_text(*parts)
    if not low:
        return False

    # This guard is ONLY for already-active bid/proposal style conversations.
    # It must NOT steal Google LSA leads or normal commercial service requests
    # that are asking for availability. Python routes; SRB/model still writes.
    strong_terms = [
        "bid", "proposal", "estimating", "estimator", "qualify bids",
        "leveling sheet", "competitive", "putting our best foot forward",
        "change order", "submittal", "addendum", "drawings", "plans",
        "general contractor", "project manager", "project admin",
        "sent you an email", "just sent you an email", "check your email",
        "emailed you", "scope clarification", "scope review",
        "notes on your estimate", "your estimate are great", "your proposal"
    ]
    booking_intent_terms = [
        "availability", "available", "appointment", "schedule", "come out",
        "come take a look", "look at", "replace", "repair", "install",
        "need help", "service", "visit"
    ]

    has_strong_bid_signal = any(t in low for t in strong_terms)
    has_booking_intent = any(t in low for t in booking_intent_terms)

    # A condo association / facility lead asking for availability is still a lead.
    # Do not auto-manual it merely because it sounds commercial.
    if has_booking_intent and not has_strong_bid_signal:
        return False

    estimate_with_bid_context = "estimate" in low and has_strong_bid_signal
    return has_strong_bid_signal or estimate_with_bid_context

def detect_non_service_thread_type(conv: dict, inbound_text: str = "", category: str | None = None, cleaned_text: str | None = None) -> str | None:
    """Return a hard thread type that must override normal residential booking."""
    history = " ".join([
        str(inbound_text or ""),
        str(category or conv.get("category") or ""),
        str(cleaned_text or conv.get("cleaned_transcript") or ""),
        str(conv.get("initial_sms") or ""),
        str(conv.get("last_sms_body") or ""),
    ])

    existing = (conv.get("thread_type") or "").strip()
    if existing in {"employment_inquiry", "commercial_bid_contact", "manual_only", "vendor_sales_or_spam", "closed_lost"}:
        return existing

    if is_pure_lsa_platform_notice(inbound_text):
        return "platform_wrapper"

    if looks_like_employment_inquiry(history):
        return "employment_inquiry"

    if v13_looks_like_callback_request(history):
        return "callback_requested"

    if looks_like_vendor_sales_or_spam(history):
        return "vendor_sales_or_spam"

    # Google Local Services Ads are paid inbound leads. Even if the scope sounds
    # commercial, they should begin the booking/availability flow unless the text
    # is clearly an existing bid/proposal conversation.
    if looks_like_commercial_bid_context(history):
        if (conv.get("source") or "").strip() == "google_lsa" and not any(
            phrase in _intent_text(inbound_text)
            for phrase in ["bid", "proposal", "sent you an email", "just sent you an email", "competitive", "leveling sheet"]
        ):
            return None
        return "commercial_bid_contact"

    return None

def clear_service_booking_state_for_non_service(conv: dict, thread_type: str) -> None:
    """Prevent stale address/date/time pending steps from pulling non-service conversations back into booking."""
    conv["thread_type"] = thread_type
    sched = conv.setdefault("sched", {})
    sched["pending_step"] = None
    sched["non_service_thread"] = True
    sched["manual_only"] = thread_type in {"manual_only", "employment_inquiry", "commercial_bid_contact", "vendor_sales_or_spam", "wrong_number_or_spam", "callback_requested", "needs_callback"}
    sched["appointment_type"] = None
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    sched["address_verified"] = False

def build_employment_inquiry_reply() -> str:
    return (
        f"Hi, thanks for reaching out. This is about employment, not an electrical service visit. "
        f"Please email your resume, license status, and the best phone number to reach you to "
        f"{EMPLOYMENT_RESUME_EMAIL}, and our office can review it."
    )

def build_commercial_bid_reply(inbound_text: str = "") -> str:
    low = _intent_text(inbound_text)
    if "email" in low or "sent" in low or "questions" in low:
        return "Got it, thank you. We will review the email and get back to you."
    if "personal" in low or "save this number" in low:
        return "Sounds good, thank you. We appreciate it."
    return "Got it, thank you. We will review this and get back to you."

def build_commercial_context_reply(conv: dict, inbound_text: str = "") -> str:
    """
    Context-aware commercial/bid reply. Python prevents residential intake;
    SRB/model writes the human response so good commercial conversations do not
    collapse into one canned manual-assist line.
    """
    text = (inbound_text or "").strip()
    if not text:
        return build_commercial_bid_reply(text)

    system = f"""
You are writing one SMS reply for Prevolt Electric to an active commercial, GC, bid, or proposal contact.
Return strict JSON with one key: {{"sms_body": string}}.

Rules:
- Do NOT use the residential intake greeting.
- Do NOT ask for house number, street name, date, or time unless the customer is explicitly trying to schedule a service visit.
- Do NOT mention the $195 evaluation visit unless the customer is clearly asking for a new service appointment/evaluation.
- For bid/proposal/estimate conversations, respond naturally to the actual message.
- If they say they sent an email, acknowledge that our office will review it.
- If they compliment the proposal, acknowledge the compliment and reinforce readiness/fit.
- Keep it concise, professional, and human.
- Do not invent commitments, prices, or schedule availability.

Relevant SRB context:
{RULE_MATRIX_TEXT[:5000]}
"""
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        )
        data = json.loads(completion.choices[0].message.content)
        body = " ".join(str(data.get("sms_body") or "").split()).strip()
        if body:
            return body
    except Exception as e:
        print("[WARN] commercial context reply failed:", repr(e))
    return build_commercial_bid_reply(text)


def looks_like_complex_commercial_coordination_request(*parts) -> bool:
    """
    Only explicit commercial walkthrough / site-walk requests are manual-only.

    Important: Google Local Services and normal inbound service requests for
    commercial or multifamily equipment, including meter-bank replacement, are
    still bookable as a $195 evaluation visit unless the customer specifically
    asks to schedule a walkthrough/site walk/project walk.
    """
    low = _intent_text(*parts)
    if not low:
        return False

    walkthrough_terms = [
        "commercial walkthrough", "walkthrough", "walk through", "site walk",
        "job walk", "walk the job", "walk the site", "project walkthrough",
        "coordinate a walkthrough", "walk-through", "pre bid walk",
        "pre-bid walk", "bid walk", "site meeting"
    ]
    return any(term in low for term in walkthrough_terms)


def looks_like_initial_service_booking_request(conv: dict, inbound_text: str = "") -> bool:
    """
    True for a first customer message that is asking for electrical service /
    availability but has not yet been put into an appointment type.

    This prevents "please let me know availability" from jumping straight to
    slot offers before the $195 evaluation context is established.
    """
    sched = conv.setdefault("sched", {})
    if sched.get("appointment_type") or conv.get("appointment_type"):
        return False
    if conv.get("thread_type") in {"employment_inquiry", "commercial_bid_contact", "manual_only"}:
        return False

    low = _intent_text(inbound_text)
    if not low:
        return False

    service_terms = [
        "need", "replace", "replacement", "repair", "install", "hooked up",
        "not working", "troubleshoot", "take a look", "look at", "come out",
        "service", "electrician", "meter bank", "meter socket", "panel",
        "outlet", "switch", "ev charger", "charger", "fan", "light"
    ]
    availability_terms = [
        "availability", "available", "when can", "please let me know",
        "let me know availability", "appointment", "schedule", "come out"
    ]
    return any(t in low for t in service_terms) and any(t in low for t in availability_terms)


def build_initial_service_booking_reply(conv: dict, inbound_text: str = "") -> str:
    """Start a normal evaluation booking flow from an inbound SMS/GLS lead.

    Price disclosure is intentionally delayed until the lead is moving into
    scheduling. If the address is still missing, ask for the address first so
    the opener does not feel like a paywall.
    """
    sched = conv.setdefault("sched", {})
    sched["appointment_type"] = "EVAL_195"
    conv["appointment_type"] = "EVAL_195"
    sched["intro_sent"] = True
    sched["manual_only"] = False
    sched["non_service_thread"] = False

    try:
        absorb_address_from_mixed_text(conv, inbound_text)
        update_address_assembly_state(sched)
    except Exception:
        pass

    address_ok = bool(sched.get("address_verified")) or _address_has_house_number_and_street(str(sched.get("raw_address") or ""))
    low = _intent_text(inbound_text)

    if not address_ok:
        sched["price_disclosed"] = False
        sched["pending_step"] = "need_address"
        return "Hi, this is Prevolt Electric. We would be more than happy to come out and take a look at your project. What’s the address for the work?"

    sched["price_disclosed"] = True
    sched["pending_step"] = "need_date"
    return "Hi, this is Prevolt Electric. We would be more than happy to come out and take a look at your project. " + build_price_and_availability_prompt(conv, "EVAL_195")


def build_complex_commercial_coordination_reply(inbound_text: str = "") -> str:
    low = _intent_text(inbound_text)
    if "meter bank" in low or "6-gang" in low or "six gang" in low or "gang meter" in low:
        return (
            "Got it, thank you. Our office will review the walkthrough details and follow up directly to coordinate access and timing."
        )
    if "walk" in low or "site visit" in low:
        return (
            "Got it, thank you. Our office will review the details and follow up directly to coordinate the commercial walkthrough."
        )
    return (
        "Thanks for reaching out. This type of commercial project is handled directly by our office so the scope and walkthrough "
        "are coordinated correctly. We will follow up directly."
    )

def is_rejecting_offered_slots(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    rejection_phrases = [
        "none", "none of those", "neither", "no", "nope", "not those",
        "that doesn't work", "that doesnt work", "doesn't work", "doesnt work",
        "not available", "any other", "anything else", "another day", "different day",
        "too early", "too late"
    ]
    return any(p == low or p in low for p in rejection_phrases)

def is_frustrated_with_bot(text: str) -> bool:
    low = _intent_text(text)
    return any(p in low for p in [
        "i don't understand", "i dont understand", "i gave you", "give up",
        "your ai", "not working", "call another company", "stop asking",
        "already told you"
    ])

def outbound_is_duplicate(conv: dict, body: str) -> bool:
    last = _intent_text(conv.get("last_sms_body"))
    current = _intent_text(body)
    return bool(last and current and last == current)

def should_send_no_reply_for_duplicate(inbound_text: str) -> bool:
    low = _intent_text(inbound_text)
    return not any(p in low for p in ["repeat", "again", "what", "which", "?"])


def is_not_a_person_name_reply(text: str) -> bool:
    """
    Guardrail for live scheduling: never treat scheduling preferences,
    slot-choice language, dates, times, or acknowledgements as a last name.
    This prevents replies like "Anytime" from becoming the customer's last name.
    """
    low = _intent_text(text)
    if not low:
        return True

    exact_blocked = {
        "yes", "no", "ok", "okay", "k", "kk", "yep", "yeah", "sure",
        "thanks", "thank you", "thx", "done", "perfect",
        "anytime", "any time", "when ever", "whenever", "whenever works",
        "when ever works", "whenever works for you", "when ever works for you",
        "whatever works", "whatever works for you", "any day", "any works",
        "any of those", "either", "either one", "whichever", "whichever one",
        "you pick", "your choice", "no preference", "all work", "all works",
        "earliest", "soonest", "first available", "next available",
    }
    if low in exact_blocked:
        return True

    blocked_phrases = [
        "works for me", "works for you", "i am home", "i'm home",
        "home all week", "home all day", "available all week",
        "available whenever", "available when ever", "anytime is fine",
        "any time is fine", "whatever is fine", "whichever is fine",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "morning", "afternoon", "evening", "noon", "midday", "tonight", "today", "tomorrow",
    ]
    if any(p in low for p in blocked_phrases):
        return True

    # Do not treat email addresses, dates, times, or numeric replies as names.
    if "@" in low or re.search(r"\d", low):
        return True

    try:
        if is_flexible_schedule_text(text) or is_next_available_request(text) or is_today_request_text(text):
            return True
        if extract_explicit_time_from_text(text) or salvage_relative_date_from_text(text):
            return True
    except Exception:
        pass

    return False


def maybe_apply_flexible_offered_slot_choice(conv: dict, inbound_text: str) -> bool:
    """
    If the customer replies "anytime" / "whatever works" to a concrete list of
    offered slots, choose the first offered slot instead of re-offering options
    or letting the name engine misread it as identity information.
    """
    sched = conv.setdefault("sched", {})
    if not sched.get("awaiting_slot_offer_choice"):
        return False
    options = sched.get("offered_slot_options") or []
    if not options:
        return False
    if not is_flexible_schedule_text(inbound_text):
        return False

    chosen = options[0]
    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195").upper()
    sched["appointment_type"] = appt
    conv["appointment_type"] = appt
    sched["scheduled_date"] = chosen.get("date")
    sched["scheduled_time"] = chosen.get("time")
    sched["scheduled_time_source"] = "offered_slot_flexible"
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    sched["final_confirmation_sent"] = False
    sched["final_confirmation_accepted"] = False
    sched["last_final_confirmation_key"] = None
    sched["slot_choice_locked"] = True
    sched["booking_attempt_nonce"] = str(uuid.uuid4())
    return True


# ---------------------------------------------------
# Availability / Human-Reply Hardening
# ---------------------------------------------------
def extract_service_address_from_text(text: str) -> str | None:
    """Extract a likely street address from a larger sentence without taking phone numbers as addresses."""
    raw = " ".join(str(text or "").replace("\n", " ").split())
    if not raw:
        return None

    # Avoid treating phone numbers as street addresses.
    raw_no_phone = re.sub(r"\b\+?1?\s*\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b", " ", raw)

    suffix = r"st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace|pl|place|park"
    house = r"\d{1,6}(?:-\d{1,6})?[A-Za-z]?"
    state = r"CT|C\.?T\.?|Connecticut|MA|M\.?A\.?|Massachusetts|Mass\.??"

    # First try a rich address with optional city/state/zip after the street.
    pattern = re.compile(
        rf"\b(?P<addr>{house}\s+[A-Za-z0-9.'#\- ]+?\b(?:{suffix})\b"
        rf"(?:\s*,?\s*[A-Za-z.'\- ]{{2,40}})?"
        rf"(?:\s*,?\s*(?:{state}))?"
        rf"(?:\s+\d{{5}}(?:-\d{{4}})?)?)",
        flags=re.I,
    )
    matches = list(pattern.finditer(raw_no_phone))
    if not matches:
        return None

    # Prefer the longest plausible match, then clean trailing non-address prose.
    addr = max((m.group("addr") for m in matches), key=len).strip(" ,.;")
    addr = re.split(
        r"\b(?:can you|could you|please|thank you|thanks|i need|we need|i am|i'm|i live|my name|call me|give me a call|disregard|found someone)\b",
        addr,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" ,.;")

    # Keep a zip if it was separated by a period after MA/CT, e.g. "Springfield ma. 01109".
    after = raw_no_phone[raw_no_phone.lower().find(addr.lower()) + len(addr):]
    z = re.match(r"^[\s,.]*(\d{5}(?:-\d{4})?)\b", after or "")
    if z and not re.search(r"\b\d{5}(?:-\d{4})?\b", addr):
        addr = f"{addr} {z.group(1)}".strip()

    addr = scrub_non_address_tail(addr)
    return addr or None

def absorb_address_from_mixed_text(conv: dict, inbound_text: str) -> bool:
    """Save a service address embedded in a longer customer message."""
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    addr = extract_service_address_from_text(inbound_text)
    if not addr:
        return False

    existing = (sched.get("raw_address") or "").strip()
    update_address_assembly_state(sched)
    existing_verified = bool(sched.get("address_verified"))

    # Do not overwrite a verified full address unless the customer clearly gave a new full address.
    if existing_verified and existing and existing.lower() == addr.lower():
        return False
    if existing_verified and existing and not _address_has_house_number_and_street(addr):
        return False

    # Upgrade partial town/zip-only values like "Springfield, 01109" to full street addresses.
    if (not existing) or (not existing_verified) or _address_has_house_number_and_street(addr):
        set_raw_address_safe(sched, addr)
        addr = sched.get("raw_address") or addr
        sched["normalized_address"] = None
        try:
            if addr not in profile.setdefault("addresses", []):
                profile["addresses"].append(addr)
        except Exception:
            pass
        try:
            parsed = parse_complete_raw_address(addr)
            if parsed:
                sched["normalized_address"] = parsed
            else:
                try_early_address_normalize(sched)
        except Exception:
            pass
        update_address_assembly_state(sched)
        return True
    return False

def _local_now() -> datetime:
    tz = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
    return datetime.now(tz)


def _is_non_emergency_appt(sched: dict) -> bool:
    return "TROUBLESHOOT" not in ((sched.get("appointment_type") or "EVAL_195").upper())


def _is_after_hours_now() -> bool:
    now = _local_now()
    return now.hour >= BOOKING_END_HOUR or now.hour < BOOKING_START_HOUR


def _is_weekend_date(date_str: str) -> bool:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() >= 5
    except Exception:
        return False


def _is_today_date(date_str: str) -> bool:
    try:
        return date_str == _local_now().date().strftime("%Y-%m-%d")
    except Exception:
        return False


def is_flexible_schedule_text(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    phrases = [
        "any day", "anytime", "any time", "whatever works", "whatever works for you",
        "whenever works", "whenever works for you", "when ever works", "when ever works for you",
        "when ever", "whenever", "whenever you want", "when ever you want",
        "whenever you can", "when ever you can", "you pick", "your choice",
        "earliest", "soonest", "first available", "next available", "next availability",
        "open availability", "wide open", "im available whenever", "i am available whenever",
        "i'm available whenever", "im available when ever", "i am available when ever",
        "any availability", "let me know availability", "please let me know availability",
    ]
    return any(p in low for p in phrases)


def is_today_request_text(text: str) -> bool:
    low = _intent_text(text)
    return "today" in low or "same day" in low or "this afternoon" in low or "this morning" in low


def is_date_only_schedule_text(text: str) -> bool:
    """True when the customer gave a day/date but no explicit time."""
    if extract_explicit_time_from_text(text):
        return False
    return bool(salvage_relative_date_from_text(text))


def _format_slot_options(slots: list[dict]) -> str:
    labels = [s.get("label") or _humanize_slot_label(s.get("date"), s.get("time")) for s in (slots or [])[:3]]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _offer_slots_response(conv: dict, slots: list[dict], prefix: str | None = None) -> str:
    sched = conv.setdefault("sched", {})
    if slots:
        sched["awaiting_slot_offer_choice"] = True
        sched["offered_slot_options"] = slots[:3]
        sched["last_slot_unavailable_message"] = None
        opts = _format_slot_options(slots)
        return f"{prefix + ' ' if prefix else ''}I have {opts}. Which one works best?".strip()
    return f"{prefix + ' ' if prefix else ''}What weekday and time work best for you?".strip()


def _slots_for_specific_date(date_str: str, appointment_type: str, limit: int = 3) -> list[dict]:
    try:
        slots = search_square_availability_for_day(date_str, appointment_type)
    except Exception:
        slots = []
    now = _local_now()
    out = []
    seen = set()
    for slot in slots or []:
        key = (slot.get("date"), slot.get("time"))
        if key in seen:
            continue
        try:
            dt = datetime.strptime(f"{slot.get('date')} {slot.get('time')}", "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
            if dt <= now:
                continue
        except Exception:
            pass
        seen.add(key)
        out.append(slot)
        if len(out) >= limit:
            break
    return out


def deterministic_availability_reply(conv: dict, inbound_text: str) -> str | None:
    """
    Handles availability language before the LLM can stack prompts or invent a time.
    Returns an SMS body or None.
    """
    sched = conv.setdefault("sched", {})
    appointment_type = (sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195").upper()
    non_emergency = _is_non_emergency_appt(sched)

    try:
        absorb_address_from_mixed_text(conv, inbound_text)
    except Exception:
        pass

    explicit_time = extract_explicit_time_from_text(inbound_text)
    requested_date = salvage_relative_date_from_text(inbound_text)
    wants_next = is_next_available_request(inbound_text)
    flexible = is_flexible_schedule_text(inbound_text)
    todayish = is_today_request_text(inbound_text)
    date_only = bool(requested_date and not explicit_time)

    if not (wants_next or flexible or todayish or date_only):
        return None

    if explicit_time:
        sched["scheduled_time_source"] = "customer_explicit"
        return None

    # Date-only must never progress to name/email or booking. Offer concrete slots first.
    if date_only:
        sched["scheduled_date"] = requested_date
        sched["scheduled_time"] = None
        sched.pop("scheduled_time_source", None)

        if non_emergency and _is_weekend_date(requested_date):
            slots = get_next_available_slots(appointment_type, limit=3)
            return _offer_slots_response(conv, slots, "We schedule non-emergency visits Monday through Friday.")

        if non_emergency and _is_today_date(requested_date) and _is_after_hours_now():
            slots = get_next_available_slots(appointment_type, limit=3)
            return _offer_slots_response(conv, slots, "We are past the normal non-emergency scheduling window for today.")

        slots = _slots_for_specific_date(requested_date, appointment_type, limit=3)
        if slots:
            day_word = "today" if _is_today_date(requested_date) else _humanize_date_for_sms(requested_date)
            return _offer_slots_response(conv, slots, f"For {day_word},")

        # If the customer requested a future date but did not give a time, do not
        # abandon their date and jump back to near-term openings. Store the date
        # and ask for the missing time; the exact-slot handler will validate Square
        # once they provide it.
        if not _is_today_date(requested_date):
            sched["awaiting_slot_offer_choice"] = False
            sched["offered_slot_options"] = []
            sched["last_slot_unavailable_message"] = None
            sched["pending_step"] = "need_time"
            return f"Got it — what time works for {_date_label_for_sms(requested_date)}?"

        slots = get_next_available_slots(appointment_type, limit=3)
        return _offer_slots_response(conv, slots, "I’m not seeing openings for today.")

    # Flexible / next available requests should get concrete choices, not another generic question.
    if wants_next or flexible:
        slots = get_next_available_slots(appointment_type, limit=3)
        if wants_next:
            if slots:
                sched["awaiting_slot_offer_choice"] = True
                sched["offered_slot_options"] = slots[:3]
                msg = format_next_available_slots_message(slots)
                sched["last_slot_unavailable_message"] = msg
                return msg
            return "I’m not seeing open times right now. What weekday and time work best for you?"
        return _offer_slots_response(conv, slots, "No problem.")

    return None


def maybe_handle_exact_slot_before_step4(conv: dict, phone: str, inbound_text: str) -> str | None:
    """
    Hard guard for customer-selected exact slots like "Monday at 9am".

    This runs before Step 4 / the model so the app cannot say a slot
    "works" or will be "reserved" until Square/state checks have run.
    """
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})

    explicit_time = extract_explicit_time_from_text(inbound_text)
    requested_date = salvage_relative_date_from_text(inbound_text)

    # If the customer already picked a day and now replies with only a time
    # (example: customer says "Tuesday", we offer Tuesday slots, then they say
    # "Can you do 2pm"), bind that time to the stored date before the model can
    # improvise "2pm works" or ask for the date again.
    if explicit_time and not requested_date:
        stored_date = (sched.get("scheduled_date") or "").strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stored_date):
            requested_date = stored_date

    if not (explicit_time and requested_date):
        return None

    appointment_type = (sched.get("appointment_type") or conv.get("appointment_type") or "").upper()
    # If we already offered evaluation slots, an exact date/time reply is a slot
    # choice even if an older state snapshot failed to mirror appointment_type.
    # Default it to EVAL_195 so the handler can check Square and move to name/email.
    if not appointment_type and (sched.get("awaiting_slot_offer_choice") or sched.get("offered_slot_options") or sched.get("address_verified")):
        appointment_type = "EVAL_195"
        sched["appointment_type"] = appointment_type
        conv["appointment_type"] = appointment_type
    if not appointment_type:
        return None

    # Emergency dispatch has a separate confirmation path.
    if "TROUBLESHOOT" in appointment_type or sched.get("awaiting_emergency_confirm") or sched.get("emergency_approved"):
        return None

    update_address_assembly_state(sched)

    # Store the customer's exact slot, but do not confirm it yet.
    sched["scheduled_date"] = requested_date
    sched["scheduled_time"] = explicit_time
    sched["scheduled_time_source"] = "customer_explicit"
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    sched["final_confirmation_sent"] = False
    sched["final_confirmation_accepted"] = False
    sched["last_final_confirmation_key"] = None

    # Non-emergency blocks first.
    if is_weekend(requested_date):
        sched["scheduled_time"] = None
        sched.pop("scheduled_time_source", None)
        slots = get_next_available_slots(appointment_type, limit=3)
        return _offer_slots_response(conv, slots, "We schedule non-emergency visits Monday through Friday.")

    if not is_within_normal_hours(explicit_time):
        sched["scheduled_time"] = None
        sched.pop("scheduled_time_source", None)
        return "We typically schedule between 9:00 AM and 4:00 PM. What time in that window works for you?"

    # The key live bug: check Square BEFORE asking name/email or letting Step 4 speak.
    # First use actual availability when Square returns it. If Square returns no
    # availability data, fall back to the safer conflict-only check below.
    try:
        day_avails = search_square_availability_for_day(requested_date, appointment_type)
        if day_avails:
            available_times = {str(slot.get("time") or "").strip() for slot in day_avails}
            if explicit_time not in available_times:
                result = build_slot_unavailable_result(
                    sched,
                    requested_date,
                    explicit_time,
                    appointment_type,
                    reason="requested_time_not_in_square_availability_pre_step4",
                )
                sched["scheduled_time"] = None
                sched.pop("scheduled_time_source", None)
                sched["final_confirmation_sent"] = False
                sched["final_confirmation_accepted"] = False
                sched["last_final_confirmation_key"] = None
                return result.get("message") or "That time is not showing as available. What other time works for you?"
    except Exception as e:
        print("[WARN] exact slot pre-Step4 availability check failed:", repr(e))

    try:
        if square_slot_has_existing_booking(requested_date, explicit_time, appointment_type, duration_minutes=60):
            result = build_slot_unavailable_result(
                sched,
                requested_date,
                explicit_time,
                appointment_type,
                reason="existing_square_booking_conflict_pre_step4",
            )
            sched["scheduled_time"] = None
            sched.pop("scheduled_time_source", None)
            sched["final_confirmation_sent"] = False
            sched["final_confirmation_accepted"] = False
            sched["last_final_confirmation_key"] = None
            return result.get("message") or "That time is already booked. What other time works for you?"
    except Exception as e:
        print("[WARN] exact slot pre-Step4 conflict check failed:", repr(e))

    # If the slot is not blocked, continue the normal booking data collection.
    # Do NOT say it works, reserved, confirmed, or anything similar yet.
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    recompute_pending_step(profile, sched)
    step = sched.get("pending_step")
    if step in {"need_name", "need_email"}:
        return choose_next_prompt_from_state(conv, inbound_text=inbound_text)

    has_identity_for_booking = bool(
        (get_active_first_name(profile) and get_active_last_name(profile))
        or profile.get("square_customer_id")
    )
    has_contact_for_booking = bool(get_active_email(profile) or profile.get("square_customer_id"))
    ready = (
        sched.get("scheduled_date")
        and sched.get("scheduled_time")
        and sched.get("address_verified")
        and sched.get("appointment_type")
        and has_identity_for_booking
        and has_contact_for_booking
        and not sched.get("pending_step")
        and not sched.get("booking_created")
    )

    if ready:
        attempt = maybe_create_square_booking(phone, conv)
        if isinstance(attempt, dict) and attempt.get("status") == "stale_cancelled":
            attempt = maybe_create_square_booking(phone, conv)
        if isinstance(attempt, dict) and attempt.get("status") == "slot_unavailable":
            return attempt.get("message") or "That time is already booked. What other time works for you?"
        if sched.get("booking_created") and sched.get("square_booking_id"):
            try:
                booked_dt = datetime.strptime(sched["scheduled_date"], "%Y-%m-%d")
                human_day = booked_dt.strftime("%A, %B %d").replace(" 0", " ")
            except Exception:
                human_day = sched.get("scheduled_date") or "that day"
            booked_time = humanize_time(sched.get("scheduled_time") or "") or (sched.get("scheduled_time") or "that time")
            return f"You're all set for {human_day} at {booked_time}. We have you on the schedule."

    return choose_next_prompt_from_state(conv, inbound_text=inbound_text)


def suppress_unbooked_reservation_language(conv: dict, phone: str, inbound_text: str, sms_body: str) -> str:
    """Final fail-safe: never let 'works/reserve/confirming' language leave unless a Square booking exists."""
    sched = conv.setdefault("sched", {})
    body = (sms_body or "").strip()
    if not body:
        return body
    if sched.get("booking_created") and sched.get("square_booking_id"):
        return body

    low = body.lower()
    low_padded = f" {re.sub(r'\\s+', ' ', low)} "
    dangerous = [
        "works great", "works!", "that works", "works for us", " works ",
        "i've noted", "i have noted", "noted that for your appointment",
        "i'll reserve", "i will reserve", "reserve that slot", "reserved",
        "looking forward to confirming", "confirming your appointment",
        "you are all set", "you're all set", "we have you on the schedule",
    ]
    if not any((p in low or p in low_padded) for p in dangerous):
        return body

    # If the inbound was an exact slot, rerun the deterministic guard and use its safe output.
    try:
        safe = maybe_handle_exact_slot_before_step4(conv, phone, inbound_text)
        if safe:
            return safe
    except Exception as e:
        print("[WARN] unbooked reservation language safe fallback failed:", repr(e))

    recompute_pending_step(conv.setdefault("profile", {}), sched)
    return choose_next_prompt_from_state(conv, inbound_text=inbound_text)

app = Flask(__name__)
sock = Sock(app) if Sock is not None else None

@app.route("/", methods=["GET", "HEAD"])
def home():
    return "Prevolt OS running", 200


# ---------------------------------------------------
# In-Memory Conversation Store
# ---------------------------------------------------
conversations = {}

# ---------------------------------------------------
# Realtime Voice Agent Helpers (Option 1 Residential)
# ---------------------------------------------------
def voice_agent_runtime_ready() -> tuple[bool, str]:
    """Return whether the live voice path can safely accept calls."""
    if not PREVOLT_VOICE_AGENT_ENABLED:
        return False, "voice_agent_disabled"
    if Sock is None or sock is None:
        return False, "flask_sock_missing"
    if websocket_client is None:
        return False, "websocket_client_missing"
    if not OPENAI_API_KEY:
        return False, "openai_api_key_missing"
    return True, "ready"


def _request_public_host() -> str:
    if PREVOLT_PUBLIC_BASE_URL:
        return PREVOLT_PUBLIC_BASE_URL.replace("https://", "").replace("http://", "").strip("/")
    host = (request.headers.get("X-Forwarded-Host") or request.host or "").strip()
    return host


def _voice_ws_url(path: str = "/voice/realtime-media") -> str:
    host = _request_public_host()
    return f"wss://{host}{path}"


def hydrate_voice_conversation(phone: str, call_sid: str = "") -> dict:
    key = (phone or "").replace("whatsapp:", "").strip() or (f"call:{call_sid}" if call_sid else "voice:unknown")
    conv = conversations.setdefault(key, {})
    conv["source"] = conv.get("source") or "voice_option_1"
    conv["channel"] = "voice"
    conv["last_call_sid"] = call_sid or conv.get("last_call_sid")
    profile = conv.setdefault("profile", {})
    profile.setdefault("name", None)
    profile.setdefault("addresses", [])
    profile.setdefault("upcoming_appointment", None)
    profile.setdefault("past_jobs", [])
    profile.setdefault("first_name", None)
    profile.setdefault("last_name", None)
    profile.setdefault("email", None)
    profile.setdefault("recognized_first_name", None)
    profile.setdefault("recognized_last_name", None)
    profile.setdefault("recognized_email", None)
    profile.setdefault("active_first_name", None)
    profile.setdefault("active_last_name", None)
    profile.setdefault("active_email", None)
    profile.setdefault("voicemail_first_name", None)
    profile.setdefault("voicemail_last_name", None)
    profile.setdefault("known_people", [])
    profile.setdefault("identity_source", None)
    profile.setdefault("square_customer_id", None)
    profile.setdefault("square_lookup_done", False)
    profile["customer_type"] = "residential"

    sched = conv.setdefault("sched", {})
    sched.setdefault("pending_step", None)
    sched.setdefault("scheduled_date", None)
    sched.setdefault("scheduled_time", None)
    sched.setdefault("raw_address", None)
    sched.setdefault("address_candidate", None)
    sched.setdefault("address_verified", False)
    sched.setdefault("address_missing", None)
    sched.setdefault("address_parts", {})
    sched.setdefault("appointment_type", None)
    sched.setdefault("booking_created", False)
    sched.setdefault("normalized_address", None)
    sched.setdefault("awaiting_emergency_confirm", False)
    sched.setdefault("emergency_approved", False)
    sched.setdefault("manual_only", False)
    sched.setdefault("booking_allowed", True)
    sched.setdefault("voice_started", True)
    sched.setdefault("voice_customer_turn_count", 0)
    sched.setdefault("voice_meaningful_customer_turns", 0)
    sched.setdefault("voice_intro_sms_sent", False)
    sched.setdefault("voice_close_after_reply", False)
    sched.setdefault("voice_drag_handoff_sent", False)

    # IMPORTANT: A new phone call from the same number should keep the customer
    # profile/Square identity, but it must not keep stale live-call scheduling
    # state from the prior test/call. We only clear transient booking fields;
    # saved Square names, emails, and addresses stay in profile and are refreshed
    # below.
    prior_call_sid = sched.get("voice_active_call_sid")
    is_new_voice_call = bool(call_sid and prior_call_sid and prior_call_sid != call_sid)
    if is_new_voice_call:
        for k in [
            "pending_step", "scheduled_date", "scheduled_time", "raw_address",
            "address_candidate", "normalized_address", "address_missing",
            "appointment_type", "last_slot_unavailable_message",
            "voice_last_caller_text_norm", "voice_last_caller_text_ts",
            "last_final_confirmation_key", "scheduled_time_source",
        ]:
            sched[k] = None
        for k in [
            "address_verified", "booking_created", "awaiting_emergency_confirm",
            "emergency_approved", "manual_only", "voice_close_after_reply",
            "voice_drag_handoff_sent", "voice_resume_sms_sent",
            "voice_booking_completed_close", "voice_sms_confirmation_sent",
            "awaiting_slot_offer_choice", "final_confirmation_sent",
            "final_confirmation_accepted",
        ]:
            sched[k] = False
        sched["booking_allowed"] = True
        sched["address_parts"] = {}
        sched["offered_slot_options"] = []
        sched["square_booking_id"] = None
        sched["voice_customer_turn_count"] = 0
        sched["voice_meaningful_customer_turns"] = 0
        conv["voice_transcript"] = []
        conv["cleaned_transcript"] = ""
        conv["last_voice_reply"] = ""
        # Force a Square refresh for this new call. This is what lets an existing
        # customer be recognized again after deploys/tests without relying on
        # stale in-memory flags.
        profile["square_lookup_done"] = False
    if call_sid:
        sched["voice_active_call_sid"] = call_sid

    # Voice calls should get repeat-customer context immediately. Keep profile;
    # refresh it from Square so the assistant can confirm a saved address instead
    # of treating the caller like a new customer.
    try:
        if phone and "hydrate_square_profile_by_phone" in globals():
            hydrate_square_profile_by_phone(profile, phone, force=is_new_voice_call)
            try:
                log_event("VOICE_SQUARE_PROFILE_READY", phone, {
                    "call_sid": call_sid,
                    "square_customer_id": profile.get("square_customer_id"),
                    "addresses_count": len(profile.get("addresses") or []),
                    "active_first_name": profile.get("active_first_name"),
                    "active_last_name": profile.get("active_last_name"),
                    "active_email_present": bool(profile.get("active_email")),
                    "phone_variants_tried": _phone_lookup_variants(phone)[:6] if "_phone_lookup_variants" in globals() else [],
                }, conv)
            except Exception:
                pass
    except Exception as e:
        try:
            log_event("VOICE_SQUARE_HYDRATE_ERROR", phone, {"error": repr(e), "call_sid": call_sid})
        except Exception:
            pass
    return conv


def build_residential_voicemail_twiml(reason: str = "") -> VoiceResponse:
    """Existing residential voicemail path, wrapped so voice failover is safe."""
    response = VoiceResponse()
    if reason:
        try:
            log_event("VOICE_AGENT_FALLBACK_TO_VOICEMAIL", (request.form.get("From") or request.args.get("From") or "").replace("whatsapp:", "").strip(), {"reason": reason, "call_sid": request.form.get("CallSid") or request.args.get("CallSid")})
        except Exception:
            pass
    response.say(
        '<speak>'
            '<prosody rate="95%">'
                'Welcome to Prevolt Electric’s residential service desk.<break time="0.7s"/>'
                'Please leave your name, address, and a brief description of what you need help with.<break time="0.6s"/>'
                'We will text you shortly.'
            '</prosody>'
        '</speak>',
        voice="Polly.Matthew-Neural"
    )
    response.record(
        max_length=60,
        play_beep=True,
        trim="do-not-trim",
        action="/voicemail-complete",
        method="POST"
    )
    response.hangup()
    return response


def build_residential_realtime_twiml(phone: str, call_sid: str) -> VoiceResponse:
    """TwiML for the live voice assistant after the caller presses 1."""
    response = VoiceResponse()
    ready, reason = voice_agent_runtime_ready()
    if not ready:
        if PREVOLT_VOICE_AGENT_FAILOVER_TO_VOICEMAIL:
            return build_residential_voicemail_twiml(reason)
        response.say("We are not able to start the scheduling assistant right now. Please call back shortly.")
        response.hangup()
        return response

    hydrate_voice_conversation(phone, call_sid)
    response.say(
        '<speak><prosody rate="98%">'
        "You have reached Prevolt Electric&apos;s automated scheduling assistant. "
        'This call may be recorded to help with scheduling. '
        "In a couple of words, please tell us why you're calling today."
        '</prosody></speak>',
        voice="Polly.Matthew-Neural"
    )
    connect = response.connect()
    stream = connect.stream(url=_voice_ws_url("/voice/realtime-media"))
    try:
        stream.parameter(name="from", value=phone or "")
        stream.parameter(name="callSid", value=call_sid or "")
    except Exception:
        pass
    return response


def _voice_agent_instructions() -> str:
    # Keep the live prompt compact. The existing Prevolt OS backend/SRB matrix is
    # the source of truth for booking logic; the realtime model is the voice layer.
    return """
You are Prevolt Electric's scheduling assistant for callers who pressed 1 for residential service.
Be calm, natural, brief, and receptionist-like. Speak at a normal call-center pace, not slowly. Do not pretend to be a human. Do not say you are an AI unless asked; say you are Prevolt's scheduling assistant.
Critical rules:
- Your job is to collect scheduling information, not diagnose electrical problems.
- Never give electrical troubleshooting, breaker-flipping, panel-opening, or DIY safety instructions.
- If the caller describes active fire, smoke filling the home, shock/unconscious person, or water touching energized electrical, tell them to call 911 first and stop scheduling until they confirm it is safe.
- Ask one question at a time. Do not ask for the phone number; caller ID already provides it.
- Keep replies short enough for a phone call. No paragraphs.
- Do not say filler phrases like "let me check", "let me lock that in", "one moment", "hold on", "I am noting", "I am finishing", "routing through our scheduling system", "the scheduling system is still processing", "let me respond", "let me move ahead", "let me think", "let me check what we still need", "I will clarify details", "thanks for hanging on", or "I am going to focus on safety" before or after the tool returns. Move directly to the next required question or the final confirmation.
- If a residential caller is in Worcester, Boston, Malden, or another property too far outside the service area, politely say we do not schedule residential repairs that far out, apologize, thank them, and end the call.
- Do not wait for the caller to say okay or thanks before asking the next required scheduling question.
- If the caller confirms an address you just read from file, do not ask for the address again; continue to dispatch confirmation or the next missing booking detail.
- If the caller asks what the emergency troubleshoot and repair visit covers, answer once, then ask if they want dispatch. If they say yes, okay, or let's do it, proceed; do not repeat the same dispatch question.
- Do not say phrases that only make sense in SMS, including: "by text", "reply here", "I'll help you here by text", "I can help you right here by text", or "text thread".
- For every caller utterance that contains scheduling details, job details, address, name, email, price objection, service-area issue, or safety information, call the prevolt_os_turn tool before answering.
- Never answer directly from your own wording after the caller speaks. Always call the prevolt_os_turn tool first, then speak only reply_to_customer. It has already been converted into phone language.
- When the next missing field is an address or address confirmation, the phone reply should sound like: "I can get your appointment scheduled here. What's the missing address piece?" or "I have the address on file. Is this for that address?"
- Do not say any greeting starting with "Thanks for calling" after the Twilio intro has played.
- If the tool returns end_call=true, speak the reply_to_customer, then stop the call naturally.
- If the tool says booking_created is true, tell the caller they are on the schedule, that a confirmation text will be sent, say goodbye, and end the call.
- If the caller asks whether you are automated, answer honestly: "This is Prevolt's scheduling assistant. It helps keep appointments organized while our electricians are in the field."
Opening behavior:
Twilio already plays the opening prompt before the live stream starts: "You have reached Prevolt Electric's automated scheduling assistant. In a couple of words, please tell us why you're calling today."
Do not repeat the greeting or ask the opening question again. Wait for the caller's answer, then call the prevolt_os_turn tool.
If the caller demands a live person, do not argue. Say: "We handle residential scheduling through our automated booking assistant so our electricians can stay in the field. I can help get the request started now."
""".strip()


def _voice_tools_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "prevolt_os_turn",
            "description": "Run the caller's latest spoken turn through Prevolt OS/SRB scheduling logic and return the exact safe reply to speak.",
            "parameters": {
                "type": "object",
                "properties": {
                    "caller_text": {
                        "type": "string",
                        "description": "The caller's latest spoken words, transcribed as accurately as possible."
                    }
                },
                "required": ["caller_text"]
            }
        }
    ]


def _voice_session_update_event() -> dict:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": OPENAI_REALTIME_MODEL,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": VOICE_AGENT_VAD_THRESHOLD,
                        "prefix_padding_ms": 120,
                        "silence_duration_ms": VOICE_AGENT_VAD_SILENCE_MS,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": OPENAI_REALTIME_VOICE,
                },
            },
            "instructions": _voice_agent_instructions(),
            "tools": _voice_tools_schema(),
            # Force every caller utterance through Prevolt OS before the model speaks.
            # This prevents the realtime model from saying a generic partial response
            # such as "I can help get this scheduled" and then waiting for the caller.
            "tool_choice": "required",
        },
    }


def _voice_initial_response_event() -> dict | None:
    # Twilio's Polly voice already plays the live-call opening prompt before
    # the WebSocket stream begins. Returning None prevents the realtime model
    # from repeating a second, different-voice intro.
    return None


def _voice_to_sms_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _voice_dedupe_consecutive_sentences(text: str) -> str:
    """Remove accidental repeated spoken prompts without changing meaning."""
    text = _voice_to_sms_text(text)
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    last_norm = ""
    for part in parts:
        p = part.strip()
        if not p:
            continue
        norm = re.sub(r"[^a-z0-9]+", " ", p.lower()).strip()
        if norm and norm == last_norm:
            continue
        cleaned.append(p)
        last_norm = norm
    # Also catch repeated question fragments that may not have punctuation.
    out = " ".join(cleaned).strip()
    out = re.sub(r"\b(What(?:'| is|’s)? your first and last name\??)\s+\1", r"\1", out, flags=re.I)
    out = re.sub(r"\b(Send the house number and street name for the work\.?)\s+\1", r"\1", out, flags=re.I)
    return _voice_to_sms_text(out)


def _voice_looks_like_live_person_demand(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return any(p in low for p in [
        "speak to a person", "speak to a human", "speak with a person", "speak with a human",
        "talk to a person", "talk to a human", "talk with a person", "talk with a human",
        "real person", "live person", "human being", "representative", "operator",
        "connect me to someone", "transfer me", "get someone on the phone", "i want a person",
        "i need a person", "i want to talk to someone", "i need to talk to someone",
    ])




def _voice_close_with_reply(phone: str, conv: dict, reply: str, reason: str = "") -> dict:
    """Mark a live voice call as complete/closed and return a clean spoken reply."""
    sched = conv.setdefault("sched", {})
    conv["thread_type"] = conv.get("thread_type") or "closed_lost"
    sched["voice_close_after_reply"] = True
    sched["voice_resume_sms_sent"] = True
    sched["voice_intro_sms_sent"] = True
    sched["booking_allowed"] = False
    sched["closed_reason"] = reason or sched.get("closed_reason")
    conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
    conv["last_voice_reply"] = reply
    return {
        "reply_to_customer": reply,
        "booking_created": False,
        "manual_only": bool(sched.get("manual_only")),
        "pending_step": sched.get("pending_step"),
        "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"),
        "end_call": True,
    }


def _voice_is_out_of_area_town(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    # Explicit residential no-go towns from Kyle's service-area rule.
    return bool(re.search(r"\b(?:worcester|worcestor|boston|malden)\b", low, flags=re.I))


def _voice_out_of_area_reply() -> str:
    return (
        "I'm sorry, but we do not schedule residential repairs for properties that far outside our service area. "
        "If there is active smoke, fire, or immediate danger, please call 911 first. "
        "Thank you for calling Prevolt Electric. Goodbye."
    )


def _voice_destination_from_sched(sched: dict) -> str:
    addr_struct = sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None
    if addr_struct:
        destination = (
            f"{(addr_struct.get('address_line_1') or '').strip()}, "
            f"{(addr_struct.get('locality') or '').strip()}, "
            f"{(addr_struct.get('administrative_district_level_1') or '').strip()} "
            f"{(addr_struct.get('postal_code') or '').strip()}"
        ).strip(" ,")
        if destination:
            return destination
    return str(sched.get("raw_address") or sched.get("address_candidate") or "").strip()


def _voice_residential_address_too_far(conv: dict) -> tuple[bool, int | None]:
    """Return true when a residential voice address is beyond the service radius."""
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    dest = _voice_destination_from_sched(sched)
    if not dest:
        return (False, None)
    # Hard block the explicit far-away towns even if travel API is unavailable.
    if _voice_is_out_of_area_town(dest):
        return (True, None)
    origin = TECH_CURRENT_ADDRESS or DISPATCH_ORIGIN_ADDRESS
    if not origin:
        return (False, None)
    try:
        minutes = compute_travel_time_minutes(origin, dest)
    except Exception:
        minutes = None
    if minutes and minutes > VOICE_RESIDENTIAL_MAX_TRAVEL_MINUTES:
        return (True, int(minutes))
    return (False, int(minutes) if minutes else None)


def _voice_looks_like_true_hazard(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return bool(re.search(
        r"\b(?:smoke|smoking|caught fire|fire|burning|burnt|hot to the touch|hot outlet|hot panel|sparking|sparks|arcing|melted|panel is hot|outlet is hot|burning smell|smells like smoke|as soon as possible|immediately|right now|dispatch now)\b",
        low,
        flags=re.I,
    ))


def _voice_text_contains_real_address(text: str) -> bool:
    try:
        addr = extract_service_address_from_text(text or "")
    except Exception:
        addr = None
    return bool(addr and _address_has_house_number_and_street(addr))


def _best_saved_address(profile_obj: dict) -> str | None:
    """Return the best usable saved Square/customer address for voice flows.

    This must be global because emergency and saved-address fast paths call it
    before the normal SMS reply generator's nested helpers exist. The prior v22
    bug defined this only inside the SMS reply function, which made emergency
    voice calls with a known Square customer fall back to asking for address.
    """
    if not isinstance(profile_obj, dict):
        return None
    addresses = profile_obj.get("addresses") or []
    # Prefer complete-looking street + city/state addresses.
    best_partial = None
    for raw in addresses:
        addr = " ".join(str(raw or "").replace("\n", " ").split()).strip(" ,.")
        if not addr:
            continue
        if not re.match(r"^\d{1,6}\b", addr):
            continue
        if not best_partial:
            best_partial = addr
        has_street = _address_has_house_number_and_street(addr)
        has_state = bool(re.search(r"\b(?:CT|Connecticut|MA|Massachusetts)\b", addr, flags=re.I))
        has_city_tail = bool(re.search(r"\b(?:[A-Za-z]+(?:\s+[A-Za-z]+)*)\s*,?\s*(?:CT|Connecticut|MA|Massachusetts)\b", addr, flags=re.I))
        if has_street and (has_state or has_city_tail):
            return addr
    return best_partial


def _saved_address_for_town(profile_obj: dict, town_hint: str) -> str | None:
    """Return saved address matching town when possible; otherwise best saved address."""
    if not isinstance(profile_obj, dict):
        return None
    town_hint = (town_hint or "").strip().lower()
    if town_hint:
        for raw in profile_obj.get("addresses") or []:
            addr = " ".join(str(raw or "").replace("\n", " ").split()).strip(" ,.")
            if addr and re.match(r"^\d{1,6}\b", addr) and town_hint in addr.lower():
                return addr
    return _best_saved_address(profile_obj)


def _voice_hazard_intake_fast_path(phone: str, conv: dict, caller_text: str) -> str | None:
    """Prevent hazard descriptions from being misread as addresses and move to emergency flow."""
    if not _voice_looks_like_true_hazard(caller_text):
        return None
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    sched["appointment_type"] = "TROUBLESHOOT_395"
    conv["appointment_type"] = "TROUBLESHOOT_395"
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    # Never let an emergency description itself become the address.
    raw_low = _intent_text(str(sched.get("raw_address") or ""))
    caller_low = _intent_text(caller_text)
    if sched.get("raw_address") and (raw_low in caller_low or _voice_looks_like_true_hazard(str(sched.get("raw_address")))):
        sched["raw_address"] = None
        sched["address_candidate"] = None
        sched["normalized_address"] = None
        sched["address_verified"] = False
        sched["address_missing"] = "street"
    if not sched.get("address_verified"):
        saved = _best_saved_address(profile)
        if saved and not _voice_text_contains_real_address(caller_text):
            sched["address_candidate"] = saved
            sched["raw_address"] = saved
            sched["address_missing"] = "confirm"
            sched["pending_step"] = "need_address"
            try:
                log_event("VOICE_HAZARD_SAVED_ADDRESS_CONFIRM", phone, {"saved_address": saved, "call_sid": conv.get("last_call_sid")}, conv)
            except Exception:
                pass
            return f"I have {saved} on file. Is this for that address?"
        sched["pending_step"] = "need_address"
        sched["address_missing"] = "street"
        return "If there is active fire or smoke filling the home, please call 911 first. If it is safe for us to come out, what's the address for the work?"
    sched["awaiting_emergency_confirm"] = True
    sched["price_disclosed"] = True
    return "This sounds urgent. We can send someone now, and arrival is usually within one to two hours. The emergency troubleshoot and repair visit is $395. Do you want us to dispatch someone now?"



def _voice_last_reply_text(conv: dict) -> str:
    return str((conv or {}).get("last_voice_reply") or "")


def _voice_last_reply_was_known_address_confirm(conv: dict) -> bool:
    low = _intent_text(_voice_last_reply_text(conv))
    if not low:
        return False
    return (
        ("on file" in low and "is this for that address" in low)
        or ("just to confirm" in low and ("youre at" in low or "you're at" in low))
        or (("youre at" in low or "you're at" in low) and ("right" in low or "correct" in low))
    )


def _voice_extract_address_from_last_reply(conv: dict) -> str | None:
    """Recover a known/saved address that the voice assistant just asked the caller to confirm."""
    prompt = _voice_last_reply_text(conv)
    candidates: list[str] = []
    for pattern in [
        r"I have\s+(.+?)\s+on file\.",
        r"you(?:'|’)?re at\s+(.+?)(?:,?\s+right\??|,?\s+correct\??|\?|$)",
        r"confirm,\s+you(?:'|’)?re at\s+(.+?)(?:,?\s+right\??|,?\s+correct\??|\?|$)",
    ]:
        m = re.search(pattern, prompt, flags=re.I)
        if m:
            candidates.append(m.group(1).strip(" ,.?"))
    try:
        extracted = extract_service_address_from_text(prompt)
        if extracted:
            candidates.append(extracted)
    except Exception:
        pass
    try:
        profile = conv.setdefault("profile", {})
        best = _best_saved_address(profile)
        if best:
            candidates.append(best)
    except Exception:
        pass
    for c in candidates:
        c = _voice_to_sms_text(c).strip(" ,.")
        if c and _address_has_house_number_and_street(c):
            return c
    return candidates[0].strip(" ,.") if candidates else None


def _voice_force_verified_address(conv: dict, address_text: str) -> bool:
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    raw = _voice_to_sms_text(address_text).strip(" ,.")
    if not raw:
        return False
    try:
        set_raw_address_safe(sched, raw)
    except Exception:
        sched["raw_address"] = raw
    sched["address_candidate"] = sched.get("raw_address") or raw
    try:
        result = normalize_address(sched.get("raw_address") or raw)
        if isinstance(result, tuple) and len(result) >= 2:
            status, addr_struct = result[0], result[1]
            if status == "ok" and isinstance(addr_struct, dict):
                sched["normalized_address"] = addr_struct
        elif isinstance(result, dict):
            sched["normalized_address"] = result
    except Exception:
        pass
    try:
        update_address_assembly_state(sched)
    except Exception:
        pass
    # Customer explicitly confirmed the address we read back. Trust that confirmation
    # even if Google parsing is imperfect; otherwise the voice flow loops back to address collection.
    sched["address_verified"] = True
    sched["address_missing"] = None
    if (sched.get("pending_step") or "") == "need_address":
        sched["pending_step"] = None
    try:
        profile.setdefault("addresses", [])
        if raw not in profile["addresses"]:
            profile["addresses"].append(raw)
    except Exception:
        pass
    try:
        recompute_pending_step(profile, sched)
    except Exception:
        pass
    return True


def _voice_known_address_confirm_fast_path(conv: dict, caller_text: str) -> str | None:
    """If caller says yes to an address-on-file confirmation, do not ask for address again."""
    try:
        is_yes = yes_text(caller_text) or _intent_text(caller_text) in {"yes", "yeah", "yep", "yup", "correct", "right", "that's right", "thats right"}
    except Exception:
        is_yes = _intent_text(caller_text) in {"yes", "yeah", "yep", "yup", "correct", "right", "that's right", "thats right"}
    if not is_yes or not _voice_last_reply_was_known_address_confirm(conv):
        return None
    addr = _voice_extract_address_from_last_reply(conv)
    if not addr:
        return None
    if not _voice_force_verified_address(conv, addr):
        return None
    sched = conv.setdefault("sched", {})

    # Recompute can sometimes re-open address collection if Google parsing is imperfect.
    # The caller just confirmed the saved address out loud, so that confirmation wins.
    sched["address_verified"] = True
    sched["address_missing"] = None
    if (sched.get("pending_step") or "").strip().lower() == "need_address":
        sched["pending_step"] = None
    sched["voice_confirmed_saved_address"] = True

    # Emergency/hazard jobs should proceed to dispatch confirmation, not loop back to address.
    last_reply_low = _intent_text(_voice_last_reply_text(conv))
    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "").upper()
    urgent_last_reply = any(x in last_reply_low for x in ["urgent", "emergency", "send someone", "dispatch", "$395", "troubleshoot and repair"])
    if "TROUBLESHOOT" in appt or sched.get("awaiting_emergency_confirm") or urgent_last_reply:
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
        sched["awaiting_emergency_confirm"] = True
        sched["price_disclosed"] = True
        return _voice_naturalize_reply(
            "This sounds urgent. We can send someone now, and arrival is usually within one to two hours. The emergency troubleshoot and repair visit is $395. Do you want us to dispatch someone now?"
        )

    try:
        nxt = choose_next_prompt_from_state(conv, inbound_text="")
        if nxt:
            natural = _voice_naturalize_reply(nxt)
            # Hard guard: after a saved-address yes, never ask for address again.
            if "address for the work" in _intent_text(natural) or "house number and street" in _intent_text(natural):
                return "Got it. What day and time works best?"
            return natural
    except Exception:
        pass
    return "Got it. What day and time works best?"


def _voice_customer_says_address_already_provided(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return any(p in low for p in [
        "you already have it", "you have it", "you have the address", "i already gave it",
        "i gave it", "already gave you", "already told you", "it is on file", "on file",
    ])


def _voice_address_already_have_it_fast_path(conv: dict, caller_text: str) -> str | None:
    """Customer pushed back after an address prompt. Trust the saved/current address and move on."""
    if not _voice_customer_says_address_already_provided(caller_text):
        return None
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    candidates = [
        sched.get("raw_address"),
        sched.get("address_candidate"),
        _best_saved_address(profile),
    ]
    # If the previous voice reply contained a saved address, use that too.
    try:
        candidates.append(_voice_extract_address_from_last_reply(conv))
    except Exception:
        pass
    addr = next((str(c).strip(" ,.") for c in candidates if c and str(c).strip(" ,.")), "")
    if not addr:
        return "Sorry about that. What's the full address for the work?"
    _voice_force_verified_address(conv, addr)
    sched["address_verified"] = True
    sched["address_missing"] = None
    if (sched.get("pending_step") or "").strip().lower() == "need_address":
        sched["pending_step"] = None
    sched["voice_confirmed_saved_address"] = True

    # If we are in an urgent/hazard flow, move to dispatch confirmation immediately.
    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "").upper()
    last_reply_low = _intent_text(_voice_last_reply_text(conv))
    urgent_context = "TROUBLESHOOT" in appt or sched.get("awaiting_emergency_confirm") or any(x in last_reply_low for x in ["urgent", "emergency", "dispatch", "$395", "troubleshoot and repair"])
    if urgent_context:
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
        sched["awaiting_emergency_confirm"] = True
        sched["price_disclosed"] = True
        return "Got it. This looks urgent. We can send someone now, and arrival is usually within one to two hours. The emergency troubleshoot and repair visit is $395. Do you want us to dispatch someone now?"

    try:
        nxt = choose_next_prompt_from_state(conv, inbound_text="")
        if nxt:
            natural = _voice_naturalize_reply(nxt)
            if "address for the work" in _intent_text(natural) or "house number and street" in _intent_text(natural):
                return "Got it. What day and time works best?"
            return natural
    except Exception:
        pass
    return "Got it. What day and time works best?"


def _voice_last_reply_requested_emergency_dispatch(conv: dict) -> bool:
    low = _intent_text(_voice_last_reply_text(conv))
    return bool(
        "dispatch someone now" in low
        or ("troubleshoot and repair" in low and "$395" in low)
        or ("this looks urgent" in low and "send someone" in low)
    )


def _voice_is_emergency_coverage_question(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return any(p in low for p in [
        "what does that cover", "what's that cover", "what does it cover", "what is included",
        "what's included", "what does that include", "what does it include", "what is included", "what am i paying", "what do i get", "what does the 395 cover",
        "what does $395 cover", "what is the 395", "what is that charge",
    ])


def _voice_is_dispatch_confirmation(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return any(p in low for p in [
        "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "go ahead", "let's do it", "lets do it",
        "do it", "send them", "send someone", "dispatch", "send him", "send her", "book it", "schedule it",
        "let's move forward", "lets move forward", "come out", "send somebody", "send an electrician"
    ])


def _voice_transcript_has_urgent_dispatch_context(conv: dict) -> bool:
    """True when the live-call context is clearly an emergency/hazard dispatch thread.

    This keeps follow-up replies like "okay" or "what does that include" anchored
    to the emergency dispatch question even if the realtime model inserted filler or
    the last spoken reply was not stored exactly.
    """
    sched = (conv or {}).setdefault("sched", {})
    appt = (sched.get("appointment_type") or (conv or {}).get("appointment_type") or "").upper()
    # Scheduled troubleshoot/repair also uses TROUBLESHOOT_395, so appointment_type alone
    # must not be treated as emergency dispatch. Only hard-hazard flags or the actual
    # emergency confirmation state should activate the dispatch lane.
    if sched.get("hard_emergency_detected") or sched.get("awaiting_emergency_confirm") or sched.get("emergency_approved"):
        return True
    haystack = _intent_text(
        (conv or {}).get("cleaned_transcript"),
        (conv or {}).get("initial_sms"),
        (conv or {}).get("last_voice_reply"),
        (conv or {}).get("last_sms_body"),
    )
    return bool(re.search(
        r"\b(?:caught\s+fire|on\s+fire|active\s+fire|smoke|smoking|burning|burnt|burned|"
        r"hot\s+to\s+the\s+touch|hot\s+panel|panel\s+is\s+hot|sparking|sparked|"
        r"arcing|arc\s+flash|crackling|popping|sizzling|melted|water\s+in\s+(?:the\s+)?panel)\b",
        haystack,
        flags=re.I,
    ))


def _voice_set_emergency_dispatch_slot(sched: dict) -> None:
    try:
        tz = ZoneInfo("America/New_York")
    except Exception:
        tz = timezone(timedelta(hours=-5))
    now_local = datetime.now(tz)
    dispatch_dt = now_local + timedelta(hours=1)
    minute = dispatch_dt.minute
    if minute == 0:
        rounded_dt = dispatch_dt.replace(second=0, microsecond=0)
    elif minute <= 30:
        rounded_dt = dispatch_dt.replace(minute=30, second=0, microsecond=0)
    else:
        rounded_dt = (dispatch_dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    sched["emergency_approved"] = True
    sched["awaiting_emergency_confirm"] = False
    sched["appointment_type"] = "TROUBLESHOOT_395"
    sched["scheduled_date"] = rounded_dt.strftime("%Y-%m-%d")
    sched["scheduled_time"] = rounded_dt.strftime("%H:%M")
    sched["scheduled_time_source"] = "voice_emergency_dispatch_confirm"
    sched["pending_step"] = None
    sched["price_disclosed"] = True
    sched["booking_attempt_nonce"] = str(uuid.uuid4())


def _voice_emergency_dispatch_fast_path(phone: str, conv: dict, caller_text: str) -> str | None:
    """Handle emergency coverage questions and dispatch acceptance without re-asking the dispatch prompt."""
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "").upper()
    # TROUBLESHOOT_395 is used for both scheduled troubleshoot visits and true emergency
    # dispatch. Do not let a normal outlet/power-loss troubleshoot yes turn become an
    # emergency dispatch acceptance just because appointment_type is TROUBLESHOOT_395.
    emergency_context = bool(
        sched.get("hard_emergency_detected")
        or sched.get("awaiting_emergency_confirm")
        or sched.get("emergency_approved")
        or _voice_last_reply_requested_emergency_dispatch(conv)
        or _voice_transcript_has_urgent_dispatch_context(conv)
    )
    if not emergency_context:
        return None

    if _voice_is_emergency_coverage_question(caller_text):
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
        sched["awaiting_emergency_confirm"] = True
        sched["price_disclosed"] = True
        return (
            "The emergency troubleshoot and repair visit covers sending one of our electricians out, "
            "checking the issue in person, making the area safe, diagnosing the problem, "
            "and completing the repair during the visit when it can be handled right away. "
            "If anything larger is needed, we'll explain the next step before moving forward. "
            "Should we dispatch someone now?"
        )

    if not _voice_is_dispatch_confirmation(caller_text):
        return None

    if not sched.get("address_verified"):
        sched["awaiting_emergency_confirm"] = True
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
        return "Got it. What's the address for the work?"

    _voice_set_emergency_dispatch_slot(sched)
    conv["appointment_type"] = "TROUBLESHOOT_395"
    # Once the caller has approved emergency dispatch, the dispatch question is complete.
    # Do not let downstream SRB logic ask the same dispatch question again.
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = True
    sched["price_disclosed"] = True
    try:
        recompute_pending_step(profile, sched)
    except Exception:
        pass

    # If identity is still missing, keep moving toward booking instead of repeating dispatch confirmation.
    if not ((get_active_first_name(profile) and get_active_last_name(profile)) or profile.get("square_customer_id")):
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
        return "Got it. What's your first and last name?"

    try:
        booking_attempt = maybe_create_square_booking(phone, conv)
    except Exception as e:
        try:
            log_event("VOICE_EMERGENCY_BOOKING_FAST_PATH_ERROR", phone, {"error": repr(e)}, conv)
        except Exception:
            pass
        booking_attempt = {"status": "exception"}

    if sched.get("booking_created") and sched.get("square_booking_id"):
        sched["voice_close_after_reply"] = True
        sched["voice_booking_completed_close"] = True
        return _voice_finalize_booking_reply(conv, "")

    status = booking_attempt.get("status") if isinstance(booking_attempt, dict) else None
    if status in {"created", "success", "booked"} or (isinstance(booking_attempt, dict) and booking_attempt.get("booking_id")):
        sched["booking_created"] = True
        if isinstance(booking_attempt, dict) and booking_attempt.get("booking_id"):
            sched["square_booking_id"] = booking_attempt.get("booking_id")
        sched["voice_close_after_reply"] = True
        sched["voice_booking_completed_close"] = True
        return _voice_finalize_booking_reply(conv, "")

    if status == "missing_identity":
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
        return "Got it. What's your first and last name?"
    if status in {"address_not_verified", "missing_address", "address_normalization_failed", "address_incomplete"}:
        sched["pending_step"] = "need_address"
        return "Got it. What's the full address for the work?"
    return "Got it. I'm going to send a text so we can finish the emergency dispatch details cleanly."



def _voice_yes_no_text(text: str) -> str | None:
    low = _intent_text(text)
    if not low:
        return None
    if low in {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "correct", "that works", "works for me"} or any(p in low for p in ["that works", "works for me", "go ahead", "let's do it", "lets do it", "book it", "schedule it"]):
        return "yes"
    if low in {"no", "nope", "nah", "no thanks", "not interested"} or any(p in low for p in ["too much", "i'll pass", "ill pass", "not interested", "no thanks", "doesn't work", "does not work"]):
        return "no"
    return None


def _voice_looks_like_multiple_addresses(text: str) -> bool:
    low = _intent_text(text)
    return bool(low and (
        re.search(r"\b(?:multiple|several|different|four|three|two|5|4|3|2)\s+(?:different\s+)?(?:addresses|properties|units|apartments|locations|houses)\b", low)
        or "more than one address" in low
        or "multiple addresses" in low
        or "i have four" in low
        or "i have several" in low
    ))


def _voice_multiple_address_reply(conv: dict, caller_text: str) -> str | None:
    if not _voice_looks_like_multiple_addresses(caller_text):
        return None
    sched = conv.setdefault("sched", {})
    sched["multiple_addresses"] = True
    sched["pending_step"] = "need_address"
    sched["state"] = "waiting_for_address"
    return (
        "We can start with the first address and get that visit scheduled. "
        "The $195 evaluation applies to the first location, and we can handle additional addresses after that. "
        "What's the first address for the work?"
    )


def _voice_split_eval_price_and_slots(text: str) -> tuple[str, str | None]:
    raw = _voice_to_sms_text(text)
    if not raw or not re.search(r"\$?195\b", raw):
        return raw, None
    patterns = [
        r"\bWhich one works best for you\??\s*.*$",
        r"\bWe have\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[^.?!]*(?:\?|\.)?.*$",
        r"\b(?:The next openings are|Our next openings are)\s+.*$",
    ]
    for pat in patterns:
        m = re.search(pat, raw, flags=re.I | re.S)
        if m:
            price = raw[:m.start()].strip(" ,.;")
            slots = raw[m.start():].strip()
            return price, slots
    return raw, None


def _voice_eval_price_confirm_fast_path(conv: dict, caller_text: str) -> str | None:
    sched = conv.setdefault("sched", {})
    if not sched.get("awaiting_eval_price_confirm"):
        return None
    yn = _voice_yes_no_text(caller_text)
    if yn == "no":
        sched["voice_close_after_reply"] = True
        sched["closed_reason"] = "declined_eval_price"
        return "No problem. We won't schedule the visit right now. Thank you for calling Prevolt Electric. Goodbye."
    if yn != "yes":
        return "The on-site evaluation visit is $195. Does that work for you?"
    sched["awaiting_eval_price_confirm"] = False
    slots = (sched.get("voice_pending_slot_offer_reply") or "").strip()
    if slots:
        return slots
    if sched.get("awaiting_slot_offer_choice") and sched.get("offered_slot_options"):
        return f"Which one works best for you — {_format_slot_options((sched.get('offered_slot_options') or [])[:3])}?"
    return "What day and time works best for you?"


def _voice_hold_eval_price_for_confirmation(conv: dict, reply: str) -> str:
    sched = conv.setdefault("sched", {})
    if sched.get("awaiting_eval_price_confirm"):
        return reply
    price, slots = _voice_split_eval_price_and_slots(reply)
    if not slots or not re.search(r"\$?195\b", price or ""):
        return reply
    sched["awaiting_eval_price_confirm"] = True
    sched["voice_pending_slot_offer_reply"] = slots
    price = price.strip()
    if not price.endswith((".", "?", "!")):
        price += "."
    return price + " Does that work for you?"


def _voice_remove_filler_sentences(text: str) -> str:
    t = _voice_to_sms_text(text)
    if not t:
        return t
    filler_patterns = [
        r"(?:Okay|Got it|Alright|Sure)?,?\s*that sounds (?:serious|urgent|dangerous)\.\s*Let me (?:think|check|respond|look) [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I(?:'|’)m going to focus on safety [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let(?:'|’)s focus on immediate safety [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me check what we still need [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me check [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I'll just clarify [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I(?:'|’)ll just clarify [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*One moment [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Thanks for hanging on [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I(?:'|’)ll move ahead [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me move ahead [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me set up [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me get [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*The scheduling system [^.?!]*(?:[.?!]|$)",
    ]
    for pat in filler_patterns:
        t = re.sub(pat, "", t, flags=re.I)
    sentences = re.split(r"(?<=[.!?])\s+", t.strip())
    out=[]; urgent_seen=False
    for sent in sentences:
        low=_intent_text(sent)
        if any(x in low for x in ["sounds urgent", "sounds serious", "sounds dangerous", "looks urgent"]):
            if urgent_seen:
                continue
            urgent_seen=True
        if low in {"okay", "got it", "all right", "alright", "sure", "thanks"}:
            continue
        out.append(sent.strip())
    return re.sub(r"\s+", " ", " ".join(out)).strip()

def _voice_naturalize_reply(reply: str) -> str:
    """Make an SMS-safe SRB reply sound normal on the phone without changing meaning."""
    text = _voice_remove_filler_sentences(_voice_to_sms_text(reply))
    if not text:
        return "Sorry, can you say that again?"

    # Remove/rewrite SMS-only openers. The SMS system is still the source of
    # truth, but phone language cannot say "I'll help you here by text" out loud.
    text = re.sub(r"^\s*(hello|hi|hey)[, ]+(?:[A-Za-z]+[, ]+)?(?:this is|you(?:'|’)ve reached)\s+Prevolt Electric\.\s*", "", text, flags=re.I)
    text = re.sub(r"\bI\s+can\s+help\s+you\s+right\s+here\s+by\s+text\.?", "I can help get this started.", text, flags=re.I)
    text = re.sub(r"\bI(?:'|’)ll\s+help\s+you\s+here\s+by\s+text\.?", "I can help get this scheduled.", text, flags=re.I)
    text = re.sub(r"\bwe\s+can\s+help\s+you\s+right\s+here\s+by\s+text\.?", "we can help get this started.", text, flags=re.I)
    text = re.sub(r"\bright\s+here\s+by\s+text\b", "right here", text, flags=re.I)
    text = re.sub(r"\bhere\s+by\s+text\b", "here", text, flags=re.I)

    replacements = [
        (r"\bI(?:'|’)m gonna route your request through our scheduling system to get the right next step\.?", ""),
        (r"\bI'm going to route your request through our scheduling system to get the right next step\.?", ""),
        (r"\bLet me respond with the safest next step\.?", ""),
        (r"\bLet me move ahead with that\.?", ""),
        (r"\bGot it\.\s*Thanks for confirming\.\s*Let me move ahead with that\.?", "Got it."),
        (r"\bThe scheduling system is still processing your request right now\.?", ""),
        (r"\bWhat number is it on [^?]+\?", "What's the house number and street name for the work?"),
        (r"\breply here\b", "tell me"),
        (r"\breply with\b", "tell me"),
        (r"\bsend over\b", "tell me"),
        (r"\bplease send (?:me )?the house number and street name for the work\.?", "what's the house number and street name for the work?"),
        (r"\bsend (?:me )?the house number and street name for the work\.?", "what's the house number and street name for the work?"),
        (r"\bSend (?:me )?the house number and street name for the work\.?", "What's the house number and street name for the work?"),
        (r"\bWhat is your first and last name\??", "What's your first and last name?"),
        (r"\bWe can definitely take care of this\.\s*", "Got it. "),
        (r"\bLet me check the scheduling details for that\.?", ""),
        (r"\bLet me think this through for scheduling\.?", ""),
        (r"\bSure, let me check what(?:\'|’)s included for that visit\.?", ""),
        (r"\bAll right, let me get this dispatch set up for you\.?", "Got it."),
        (r"\bI(?:'|’)m going to get the right safety first next step for this\.?", ""),
        (r"\bOkay, let(?:'|’)s keep this safe and get the next step lined up\.?", "Got it."),
        (r"\bLet me quickly explain what that visit[^.?!]*[.?!]?", ""),
        (r"\bThanks for hanging on\.\s*I(?:'|’)m still waiting for the exact coverage details to come through\.?", ""),
        (r"\bSure, let me quickly explain what that visit[^.?!]*[.?!]?", ""),
        (r"\bGot it\.\s*Let(?:'|’)s move ahead with the next scheduling step now\.?", "Got it."),
        (r"\bOkay, thanks for that detail\.?", "Got it."),
        (r"\bLet me get the next scheduling step ready\.?", ""),
        (r"\bLet me check what details are still needed to finish the booking\.?", ""),
        (r"\bLet me check the details for that [A-Za-z]+ slot\.?", "Got it."),
        (r"\bLet me get that time set for you and finish the booking details\.?", "Got it."),
        (r"\bLet me lock that in and grab the last detail I need\.?", "Got it."),
        (r"\bLet me get the last detail I need to finish scheduling\.?", "Got it."),
        (r"\bI'm just finishing the booking details with that email now\.?", ""),
        (r"\bI(?:'|’)m finalizing the booking details now\.?", ""),
        (r"\bAll right, I(?:'|’)m finalizing the booking details now\.?", ""),
        (r"\bOkay, let me check the details for that [A-Za-z]+ slot\.?", "Got it."),
        (r"\bOkay, let me get that time set for you and finish the booking details\.?", "Got it."),
        (r"\bOkay, let me get the last detail I need to finish scheduling\.?", "Got it."),
        (r"\bOkay, I(?:'|’)m noting that email for your booking\.?", ""),
        (r"\bI(?:'|’)m noting that email for your booking\.?", ""),
        (r"\bOne moment while I finish this\.?", ""),
        (r"\bLet me finish this\.?", ""),
        (r"\bI(?:'|’)ll finish this now\.?", ""),
        (r"\bWhat is the best email address for the appointment\??", "What's the best email address for the appointment?"),
        (r"\bIf that is something you(?:'|’)re interested in,\s*", ""),
        (r"\btext you shortly\b", "send you a confirmation text shortly"),
        (r"\bWe will text you shortly\b", "We'll send you a confirmation text shortly"),
        (r"\bWe'll text you shortly\b", "We'll send you a confirmation text shortly"),
        (r"\bwhat(?:'|’)s the address for the visit\b", "what's the address for the work"),
        (r"\bfor the visit\b", "for the work"),
        (r"\bWhat’s\b", "What's"),
        (r"\bWe’ll\b", "We'll"),
        (r"\byou’re\b", "you're"),
        (r"\bYou’re\b", "You're"),
    ]
    # Emergency voice cleanup: do not stack generic filler before the real dispatch/address prompt.
    replacements.extend([
        (r"\bOkay, that sounds urgent\.\s*Let(?:'|’)s focus on immediate safety and what to do next\.?", "This sounds urgent."),
        (r"\bOkay, that sounds urgent\.?", "This sounds urgent."),
        (r"\bI can help get this scheduled\.\s*This sounds urgent\.", "This sounds urgent."),
        (r"\bI can help get this scheduled\.\s*This looks urgent\.", "This looks urgent."),
        (r"\bGot it\.\s*This looks urgent\.", "This looks urgent."),
    ])

    for pat, repl in replacements:
        text = re.sub(pat, repl, text, flags=re.I)

    # Voice-specific price disclosure: make the $195 sound like an evaluation visit,
    # not simply a fee for a quote.
    text = re.sub(
        r"We do charge \$195 for one of our electricians to come out, take a look, and provide you with a quote\.?",
        "We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step.",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"We do charge \$195 for one of our electricians to come out, take a look, and provide you with a written quote\.?",
        "We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step.",
        text,
        flags=re.I,
    )

    # If the SRB/text reply includes a phone-unfriendly preamble like
    # "I can help get this scheduled. Hi Kyle. We can help with... To get started,"
    # collapse it into one continuous receptionist-style address prompt.
    text = re.sub(
        r"^\s*(?:Okay,?\s*)?(?:Got it\.\s*)?(?:I can help get this (?:scheduled|started)\.\s*)?(?:Hi[, ]+[A-Z][A-Za-z'\-]{1,24}\.\s*)?(?:We can help with [^.?!]+\.\s*)?(?:To get started,\s*)+(?=(?:what|what's|which|please|can you|tell me)\b)",
        "I can get your appointment scheduled here. ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^\s*(?:Okay,?\s*)?(?:Got it\.\s*)?(?:I can help get this (?:scheduled|started)\.\s*)+(?:Hi[, ]+[A-Z][A-Za-z'\-]{1,24}\.\s*)?(?=(?:we can help|what|what's|which|please|can you|tell me)\b)",
        "I can get your appointment scheduled here. ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^\s*I can get your appointment scheduled here,\s*(?:We can help with [^.?!]+\.\s*)?(?:To get started,\s*)?(?=(?:what|what's|which|please|can you|tell me)\b)",
        "I can get your appointment scheduled here. ",
        text,
        flags=re.I,
    )

    # Voice-only UX cleanup: keep the scheduling-assistant transition, but
    # do not let it become a standalone sentence that can sound like a stall.
    # Desired phone shape:
    # "I can get your appointment scheduled here, what's the house number and street name for the work?"
    # or, when the address was already heard:
    # "I can get your appointment scheduled here. Just to confirm, you're at 45 Main Street, right?"
    text = re.sub(
        r"^\s*(?:Okay,?\s*)?(?:let(?:'|’)s get (?:that|this) (?:request )?(?:set up|started)\.?\s*)?(?:Got it\.\s*)?(?:I can help get this (?:scheduled|started)\.\s*)+(?=(?:what|what's|which|we do|we have|please|can you|tell me|you(?:'|’)re at|you're at)\b)",
        "I can get your appointment scheduled here. ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^\s*(?:Okay,?\s*)?(?:let(?:'|’)s get (?:that|this) (?:request )?(?:set up|started)\.?\s*)+(?=(?:what|what's|which|we do|we have|please|can you|tell me|you(?:'|’)re at|you're at)\b)",
        "I can get your appointment scheduled here. ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^\s*I can get your appointment scheduled here,\s*(?:you(?:'|’)re|you're) at\s+",
        "I can get your appointment scheduled here. Just to confirm, you're at ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^\s*I can get your appointment scheduled here,\s*we do charge\b",
        "We do charge",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^\s*I can get your appointment scheduled here,\s*we have\b",
        "We have",
        text,
        flags=re.I,
    )

    # Voice-specific cleanup: do not prefix name questions with the first name;
    # phone TTS creates an awkward long pause after the comma. Also never ask
    # for the phone number on a live call when Twilio already supplied caller ID.
    text = re.sub(r"^\s*[A-Z][A-Za-z'\-]{1,24},\s*(?:what(?:'| is|’s)? your last name\??)", "What's your last name?", text, flags=re.I)
    text = re.sub(r"\bWhat is your last name\??", "What's your last name?", text, flags=re.I)
    text = re.sub(r"\bWhat is the best phone number to reach you about the appointment\??", "What's the best email address for the appointment?", text, flags=re.I)
    text = re.sub(r"\bWhat(?:'|’)?s the best phone number to reach you about the appointment\??", "What's the best email address for the appointment?", text, flags=re.I)

    # Collapse awkward intro leftovers.
    text = re.sub(r"^\s*(I can help get this started\.|I can help get this scheduled\.)\s*(I can help get this started\.|I can help get this scheduled\.)", r"\1", text, flags=re.I)
    text = re.sub(r"^\s*(Perfect|Sounds good|Alright|Okay)\.\s*(Perfect|Sounds good|Alright|Okay)\.\s*", r"\1. ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    # If the SMS intro got prepended to the first real question, keep the
    # conversational question and drop the text-only branding.
    low = _intent_text(text)
    if "what town is the work in" in low and ("help get this" in low or "prevolt electric" in low):
        return "What town is the work in?"
    text = _voice_dedupe_consecutive_sentences(_voice_remove_filler_sentences(text))
    # Final booking language should clearly restate date/time and mention the confirmation text.
    text = re.sub(
        r"You(?:'|’)re all set for ([^.]+?) at ([^.]+?)\.\s*We have you on the schedule\.?",
        r"You're all set. We have you on the schedule for \1 at \2. You'll receive a text shortly with confirmation.",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"You are all set for ([^.]+?) at ([^.]+?)\.\s*We have you on the schedule\.?",
        r"You're all set. We have you on the schedule for \1 at \2. You'll receive a text shortly with confirmation.",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"We have you on the schedule\.?$",
        "You'll receive a text shortly with confirmation.",
        text,
        flags=re.I,
    )

    if len(text) > 320:
        parts = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(parts[:2]).strip()
    return text or "Sorry, can you say that again?"


def _voice_intro_sms_body() -> str:
    return "Hello, this is Prevolt Electric. I can help you right here by text. What town is the work in, and what electrical issue are you looking to schedule?"


def _voice_send_intro_sms_if_abandoned(phone: str, conv: dict, reason: str = "") -> None:
    """If option 1 connected but the caller hung up before giving details, keep the current SMS intake alive."""
    if not VOICE_AGENT_HANGUP_SMS_ENABLED:
        return
    if not phone or not (twilio_client and TWILIO_FROM_NUMBER):
        return
    sched = conv.setdefault("sched", {})
    if sched.get("voice_intro_sms_sent") or sched.get("booking_created") or sched.get("manual_only"):
        return
    if int(sched.get("voice_meaningful_customer_turns") or 0) > 0:
        return
    try:
        body = _voice_intro_sms_body()
        msg = twilio_client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=phone)
        sched["voice_intro_sms_sent"] = True
        sched["voice_intro_sms_reason"] = reason
        conv["last_sms_body"] = body
        log_event("VOICE_ABANDONED_INTRO_SMS_SENT", phone, {"sid": getattr(msg, "sid", None), "reason": reason, "call_sid": conv.get("last_call_sid")}, conv)
    except Exception as e:
        sched["voice_intro_sms_error"] = repr(e)
        try:
            log_event("VOICE_ABANDONED_INTRO_SMS_FAILED", phone, {"error": repr(e), "reason": reason}, conv)
        except Exception:
            pass


def _voice_send_resume_sms_if_needed(phone: str, conv: dict, reason: str = "") -> None:
    """If a live voice call drops mid-flow, continue through the existing SMS path."""
    if not VOICE_AGENT_HANGUP_SMS_ENABLED:
        return
    if not phone or not (twilio_client and TWILIO_FROM_NUMBER):
        return
    sched = conv.setdefault("sched", {})
    if sched.get("booking_created") or sched.get("voice_resume_sms_sent") or sched.get("voice_intro_sms_sent"):
        return
    if sched.get("manual_only"):
        return

    meaningful = int(sched.get("voice_meaningful_customer_turns") or 0)
    if meaningful <= 0:
        _voice_send_intro_sms_if_abandoned(phone, conv, reason or "abandoned_before_details")
        return

    step = (sched.get("pending_step") or "").strip().lower()
    emergency_resume = bool(
        sched.get("hard_emergency_detected")
        or sched.get("awaiting_emergency_confirm")
        or sched.get("emergency_approved")
    )
    address_verified = bool(sched.get("address_verified"))
    if emergency_resume and not address_verified:
        body = "This is Prevolt Electric. We got started over the phone. Please reply with the full address for the work, including the town and state."
    elif emergency_resume and sched.get("awaiting_emergency_confirm") and not sched.get("emergency_approved"):
        body = "This is Prevolt Electric. We got started over the phone. We can send someone now, and arrival is usually within one to two hours. The emergency troubleshoot and repair visit is $395. Reply YES if you want us to dispatch someone now."
    elif not address_verified or step == "need_address":
        body = "This is Prevolt Electric. We got started over the phone. Please reply with the full address for the work, including the town and state."
    elif emergency_resume and step == "need_name":
        body = "This is Prevolt Electric. We got the emergency dispatch started over the phone. Please reply with your first and last name so we can finish booking."
    elif emergency_resume and step == "need_email":
        body = "This is Prevolt Electric. We got the emergency dispatch started over the phone. Please reply with your email address so we can finish booking."
    elif sched.get("awaiting_slot_offer_choice") and sched.get("offered_slot_options"):
        try:
            body = f"This is Prevolt Electric. We got started over the phone. Which appointment option works best — {_format_slot_options((sched.get('offered_slot_options') or [])[:3])}?"
        except Exception:
            body = "This is Prevolt Electric. We got started over the phone. What day and time works best for you?"
    elif step == "need_name":
        body = "This is Prevolt Electric. We got the appointment details started over the phone. Please reply with your first and last name so we can finish booking."
    elif step == "need_email":
        body = "This is Prevolt Electric. We got the appointment details started over the phone. Please reply with your email address so we can finish booking."
    elif step in {"need_date", "need_time"} or not (sched.get("scheduled_date") and sched.get("scheduled_time")):
        body = "This is Prevolt Electric. We got started over the phone. What day and time works best for you?"
    else:
        body = "This is Prevolt Electric. We got started over the phone. Please reply here with any remaining details so we can finish getting you on the schedule."

    try:
        msg = twilio_client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=phone)
        sched["voice_resume_sms_sent"] = True
        sched["voice_resume_sms_reason"] = reason
        sched["voice_resume_sms_sid"] = getattr(msg, "sid", None)
        conv["last_sms_body"] = body
        log_event("VOICE_RESUME_SMS_SENT", phone, {"sid": getattr(msg, "sid", None), "reason": reason, "call_sid": conv.get("last_call_sid")}, conv)
    except Exception as e:
        sched["voice_resume_sms_error"] = repr(e)
        try:
            log_event("VOICE_RESUME_SMS_FAILED", phone, {"error": repr(e), "reason": reason}, conv)
        except Exception:
            pass


def _voice_handoff_sms_body(conv: dict | None = None) -> str:
    """State-aware SMS when the voice call is taking too long. Ask only for what is missing."""
    conv = conv or {}
    sched = conv.setdefault("sched", {}) if isinstance(conv, dict) else {}
    profile = conv.setdefault("profile", {}) if isinstance(conv, dict) else {}
    step = (sched.get("pending_step") or "").strip().lower()
    has_address = bool(sched.get("address_verified"))
    has_slot = bool(sched.get("scheduled_date") and sched.get("scheduled_time"))
    has_name = bool(profile.get("active_first_name") or profile.get("first_name") or profile.get("name"))
    has_email = bool(profile.get("active_email") or profile.get("email") or sched.get("email"))

    emergency_handoff = bool(
        sched.get("hard_emergency_detected")
        or sched.get("awaiting_emergency_confirm")
        or sched.get("emergency_approved")
    )
    if emergency_handoff and not has_address:
        return "This is Prevolt Electric. We got started over the phone. Please reply with the full address for the work, including the town and state."
    if emergency_handoff and sched.get("awaiting_emergency_confirm") and not sched.get("emergency_approved"):
        return "This is Prevolt Electric. We got started over the phone. We can send someone now, and arrival is usually within one to two hours. The emergency troubleshoot and repair visit is $395. Reply YES if you want us to dispatch someone now."
    if not has_address or step == "need_address":
        return "This is Prevolt Electric. We got started over the phone. Please reply with the full address for the work, including the town and state."
    if emergency_handoff and (step == "need_name" or not has_name):
        return "This is Prevolt Electric. We got the emergency dispatch started over the phone. Please reply with your first and last name so we can finish booking."
    if step == "need_name" or not has_name:
        return "This is Prevolt Electric. We got the appointment details started over the phone. Please reply with your first and last name so we can finish booking."
    if emergency_handoff and (step == "need_email" or not has_email):
        return "This is Prevolt Electric. We got the emergency dispatch started over the phone. Please reply with your email address so we can finish booking."
    if step == "need_email" or not has_email:
        return "This is Prevolt Electric. We got the appointment details started over the phone. Please reply with your email address so we can finish booking."
    if not has_slot or step in {"need_date", "need_time"}:
        return "This is Prevolt Electric. We got started over the phone. What day and time works best for you?"
    return "This is Prevolt Electric. We got most of the appointment details started over the phone. Please reply with the electrical issue and any remaining details so we can finish booking."


def _voice_send_drag_handoff_sms(phone: str, conv: dict, reason: str = "") -> None:
    if not phone or not (twilio_client and TWILIO_FROM_NUMBER):
        return
    sched = conv.setdefault("sched", {})
    if sched.get("voice_drag_handoff_sent"):
        return
    try:
        body = _voice_handoff_sms_body(conv)
        msg = twilio_client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=phone)
        sched["voice_drag_handoff_sent"] = True
        sched["voice_drag_handoff_reason"] = reason
        conv["last_sms_body"] = body
        log_event("VOICE_DRAG_HANDOFF_SMS_SENT", phone, {"sid": getattr(msg, "sid", None), "reason": reason, "call_sid": conv.get("last_call_sid")}, conv)
    except Exception as e:
        sched["voice_drag_handoff_error"] = repr(e)
        try:
            log_event("VOICE_DRAG_HANDOFF_SMS_FAILED", phone, {"error": repr(e), "reason": reason}, conv)
        except Exception:
            pass


def _voice_format_scheduled_slot_from_state(conv: dict) -> str:
    sched = (conv or {}).setdefault("sched", {}) if isinstance(conv, dict) else {}
    date_raw = (sched.get("scheduled_date") or "").strip()
    time_raw = (sched.get("scheduled_time") or "").strip()
    try:
        human_day = datetime.strptime(date_raw, "%Y-%m-%d").strftime("%A, %B %d").replace(" 0", " ")
    except Exception:
        human_day = date_raw or "the appointment date"
    try:
        human_t = humanize_time(time_raw) or time_raw
    except Exception:
        human_t = time_raw
    if human_day and human_t:
        return f"{human_day} at {human_t}"
    return human_day or human_t or "the appointment time"


def _voice_finalize_booking_reply(conv: dict, reply: str) -> str:
    """Force a clean live-call closing after Square booking creation."""
    slot = _voice_format_scheduled_slot_from_state(conv)
    sched = (conv or {}).setdefault("sched", {})
    appt = (sched.get("appointment_type") or (conv or {}).get("appointment_type") or "").upper()
    # Prefer a consistent spoken ending over backend/SMS phrasing variants.
    # Include goodbye because the WebSocket will close after this response finishes.
    true_emergency_booking = bool(
        sched.get("hard_emergency_detected")
        or sched.get("awaiting_emergency_confirm")
        or sched.get("emergency_approved")
    )
    if true_emergency_booking:
        return (
            f"You're all set. We have you on the schedule for emergency dispatch at {slot}. "
            "You'll receive a confirmation text shortly. "
            "Thank you for calling Prevolt Electric. Goodbye."
        )
    return (
        f"You're all set. We have you on the schedule for {slot}. "
        "You'll receive a confirmation text shortly. "
        "Thank you for calling Prevolt Electric. Goodbye."
    )

def _voice_maybe_send_booking_sms(phone: str, conv: dict, voice_reply: str) -> None:
    sched = conv.setdefault("sched", {})
    if not (sched.get("booking_created") and sched.get("square_booking_id")):
        return
    if sched.get("voice_sms_confirmation_sent"):
        return
    if not (twilio_client and TWILIO_FROM_NUMBER and phone):
        return
    try:
        msg = twilio_client.messages.create(
            body=_voice_to_sms_text(voice_reply),
            from_=TWILIO_FROM_NUMBER,
            to=phone,
        )
        sched["voice_sms_confirmation_sent"] = True
        sched["voice_sms_confirmation_sid"] = getattr(msg, "sid", None)
        conv["last_sms_body"] = _voice_to_sms_text(voice_reply)
        log_event("VOICE_BOOKING_CONFIRMATION_SMS_SENT", phone, {"sid": getattr(msg, "sid", None), "call_sid": conv.get("last_call_sid")}, conv)
    except Exception as e:
        sched["voice_sms_confirmation_error"] = repr(e)
        try:
            log_event("VOICE_BOOKING_CONFIRMATION_SMS_FAILED", phone, {"error": repr(e)}, conv)
        except Exception:
            pass



def _voice_sanitize_name_fields(profile: dict, sched: dict | None = None) -> None:
    """Prevent voice filler words from becoming customer last names.

    Realtime transcription often turns "my name is Kyle, and I live..." into
    a name candidate of "Kyle And". That should never satisfy the full-name
    requirement or be sent to Square.
    """
    bad_last_names = {
        "and", "or", "in", "at", "from", "with", "for", "about", "because",
        "regarding", "located", "living", "live", "lives", "need", "needs",
        "calling", "looking", "trying", "want", "wants", "the", "a", "an",
    }
    changed = False
    for key in ["active_last_name", "last_name", "recognized_last_name", "voicemail_last_name"]:
        value = (profile.get(key) or "").strip()
        if value and value.lower() in bad_last_names:
            profile[key] = None
            changed = True
    full = " ".join(str(profile.get("name") or "").strip().split())
    if full:
        parts = full.split()
        if len(parts) >= 2 and parts[-1].lower() in bad_last_names:
            profile["name"] = " ".join(parts[:-1]).strip() or None
            changed = True
    if changed and sched is not None:
        try:
            recompute_pending_step(profile, sched)
        except Exception:
            pass


def _voice_apply_offered_slot_fast_path(conv: dict, caller_text: str) -> str | None:
    """Run SMS route-level offered-slot guards for voice turns too.

    The normal SMS route locks offered slot choices before the LLM path. Voice
    bypasses that route, so a date-only reply like "Monday June 1" can otherwise
    be reinterpreted as a fresh date and get the wrong default time.
    """
    sched = conv.setdefault("sched", {})
    if not (sched.get("awaiting_slot_offer_choice") and (sched.get("offered_slot_options") or [])):
        return None
    try:
        date_change_reply = maybe_handle_new_date_during_offered_slots(conv, caller_text)
        if date_change_reply:
            return _voice_naturalize_reply(date_change_reply)
    except Exception as e:
        try:
            log_event("VOICE_OFFERED_SLOT_DATE_CHANGE_ERROR", conv.get("phone") or "", {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
    selected = False
    try:
        selected = maybe_apply_offered_slot_selection(conv, caller_text)
    except Exception as e:
        try:
            log_event("VOICE_OFFERED_SLOT_SELECTION_ERROR", conv.get("phone") or "", {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
    if not selected:
        try:
            selected = v13_time_only_on_offered_slots(conv, caller_text)
        except Exception:
            selected = False
    if selected:
        try:
            profile = conv.setdefault("profile", {})
            _voice_sanitize_name_fields(profile, sched)
            recompute_pending_step(profile, sched)
        except Exception:
            pass
        try:
            return _voice_naturalize_reply(choose_next_prompt_from_state(conv, inbound_text=caller_text))
        except Exception:
            return "Got it. What's your first and last name?"
    return None


def _voice_profile_first_name(profile: dict) -> str:
    return normalize_person_name(
        profile.get("active_first_name")
        or profile.get("first_name")
        or profile.get("recognized_first_name")
        or profile.get("voicemail_first_name")
        or ""
    )


def _voice_profile_last_name(profile: dict) -> str:
    return normalize_person_name(
        profile.get("active_last_name")
        or profile.get("last_name")
        or profile.get("recognized_last_name")
        or profile.get("voicemail_last_name")
        or ""
    )


def _voice_extract_name_words(text: str) -> tuple[str, str]:
    """Extract a first/last name from a short spoken name reply.

    This is intentionally conservative and only runs while the booking state is
    waiting for a name. It prevents replies like "Prevost" from being sent back
    through the full SMS engine and repeatedly producing "what is your last name?".
    """
    raw = str(text or "").strip()
    if not raw:
        return ("", "")
    if v13_extract_email(raw) or re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", raw):
        return ("", "")
    cleaned = re.sub(r"\b(?:my\s+name\s+is|this\s+is|it\s+is|it's|its|last\s+name\s+is|my\s+last\s+name\s+is|first\s+name\s+is|my\s+first\s+name\s+is)\b", " ", raw, flags=re.I)
    cleaned = re.sub(r"[^A-Za-z'\- ]", " ", cleaned).strip()
    words = [normalize_person_name(w) for w in cleaned.split() if normalize_person_name(w)]
    bad = {
        "and", "or", "in", "at", "from", "with", "for", "about", "because", "the", "a", "an",
        "customer", "service", "representative", "operator", "person", "human", "phone", "number",
        "street", "avenue", "ave", "road", "rd", "drive", "dr", "lane", "ln", "windsor", "locks",
        "connecticut", "massachusetts", "framingham", "springfield", "enfield", "suffield",
    }
    words = [w for w in words if w.lower() not in bad]
    if not words:
        return ("", "")
    if len(words) == 1:
        return ("", words[0])
    return (words[0], words[-1])


def _voice_has_email(conv: dict) -> bool:
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    return bool(profile.get("active_email") or profile.get("email") or sched.get("email"))


def _voice_after_name_reply(conv: dict) -> str:
    """After voice captures the missing name, choose the next spoken prompt."""
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    try:
        recompute_pending_step(profile, sched)
    except Exception:
        pass
    # Phone number is already known from Twilio caller ID. Do not ask for it.
    if not _voice_has_email(conv):
        sched["pending_step"] = "need_email"
        sched["state"] = "waiting_for_email"
        return "Thanks. What's the best email address for the appointment?"
    try:
        nxt = choose_next_prompt_from_state(conv, inbound_text="")
        if nxt:
            return _voice_naturalize_reply(nxt)
    except Exception:
        pass
    return "Thanks. We'll send a confirmation text shortly."


def _voice_name_fast_path(conv: dict, caller_text: str) -> str | None:
    """Handle missing-name replies before they hit the SMS engine.

    In live voice, customers often answer a last-name prompt with only one word
    such as "Prevost". The SMS name engine expects a fuller text exchange and can
    loop. This fast path stores that last name and moves to email immediately.
    """
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    step = (sched.get("pending_step") or "").strip().lower()
    state = (sched.get("state") or "").strip().lower()
    name_engine_state = (sched.get("name_engine_state") or "").strip().lower()
    waiting_for_name = step == "need_name" or state in {"waiting_for_name", "need_name"} or "last_name" in name_engine_state or "name" in name_engine_state
    if not waiting_for_name:
        return None

    first_known = _voice_profile_first_name(profile)
    last_known = _voice_profile_last_name(profile)
    # If a phone number is spoken while we still need the name, ignore it and ask the actual missing field.
    if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", caller_text or "") and not last_known:
        return "What's your last name?"

    first, last = _voice_extract_name_words(caller_text)
    if first and not first_known:
        profile["active_first_name"] = first
        profile["first_name"] = first
        first_known = first
    if last and not last_known:
        profile["active_last_name"] = last
        profile["last_name"] = last
        last_known = last
    if first_known and last_known:
        profile["name"] = f"{first_known} {last_known}".strip()
        sched["name_engine_state"] = None
        sched["pending_step"] = None
        try:
            upsert_known_person(profile, first_name=first_known, last_name=last_known, email=profile.get("active_email") or profile.get("email") or "", square_customer_id=None)
        except Exception:
            pass
        return _voice_after_name_reply(conv)
    if first_known and not last_known:
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
        return "What's your last name?"
    return None


def _voice_reply_asks_for_phone(text: str) -> bool:
    low = _intent_text(text)
    return "best phone number" in low or "phone number to reach" in low or "what phone number" in low


def _voice_email_fast_path(phone: str, conv: dict, caller_text: str) -> str | None:
    """When the voice caller gives an email at the end, save it and attempt booking immediately.

    This prevents the live agent from saying a filler phrase like "one moment while I finish this"
    and then closing after the SMS confirmation is sent.
    """
    email = v13_extract_email(caller_text or "")
    if not email:
        return None
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    v13_save_email(conv, email)
    try:
        recompute_pending_step(profile, sched)
    except Exception:
        pass

    first = _voice_profile_first_name(profile)
    last = _voice_profile_last_name(profile)
    if not (first and last):
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
        return "Thanks. What's your first and last name?"

    if not sched.get("address_verified"):
        sched["pending_step"] = "need_address"
        sched["state"] = "waiting_for_address"
        return "Thanks. What's the full address for the work?"

    if not (sched.get("scheduled_date") and sched.get("scheduled_time")):
        try:
            return _voice_naturalize_reply(choose_next_prompt_from_state(conv, inbound_text=caller_text))
        except Exception:
            return "Thanks. What day and time works best?"

    try:
        booking_attempt = maybe_create_square_booking(phone, conv)
    except Exception as e:
        try:
            log_event("VOICE_EMAIL_BOOKING_FAST_PATH_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
        booking_attempt = {"status": "exception"}

    if sched.get("booking_created") and sched.get("square_booking_id"):
        sched["voice_close_after_reply"] = True
        sched["voice_booking_completed_close"] = True
        return _voice_finalize_booking_reply(conv, "")

    status = booking_attempt.get("status") if isinstance(booking_attempt, dict) else None
    if status in {"created", "success", "booked"} or (isinstance(booking_attempt, dict) and booking_attempt.get("booking_id")):
        sched["booking_created"] = True
        if isinstance(booking_attempt, dict) and booking_attempt.get("booking_id"):
            sched["square_booking_id"] = booking_attempt.get("booking_id")
        sched["voice_close_after_reply"] = True
        sched["voice_booking_completed_close"] = True
        return _voice_finalize_booking_reply(conv, "")

    if status == "missing_identity":
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
        return "Thanks. What's your first and last name?"
    if status in {"address_not_verified", "missing_address", "address_normalization_failed", "address_incomplete"}:
        sched["pending_step"] = "need_address"
        sched["state"] = "waiting_for_address"
        return "Thanks. What's the full address for the work?"
    # If Square did not complete immediately, keep the caller in a clean text fallback instead of a dead-air close.
    return "Thanks. I will send you a text so we can finish the last step cleanly."


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    """Run one spoken customer turn through the existing Prevolt OS brain."""
    phone = (phone or "").replace("whatsapp:", "").strip()
    conv = hydrate_voice_conversation(phone, call_sid)
    caller_text = _voice_to_sms_text(caller_text)
    conv.setdefault("voice_transcript", []).append({"role": "customer", "text": caller_text, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})

    if not caller_text:
        return {"reply_to_customer": "Sorry, can you say that again?", "booking_created": False, "manual_only": False, "end_call": False}

    sched = conv.setdefault("sched", {})

    # Realtime can occasionally deliver the same recognized utterance twice with
    # different internal response IDs. Do not count or process duplicate turns;
    # this prevents repeated prompts and false max-turn handoffs.
    now_ts = time.time()
    last_text_norm = sched.get("voice_last_caller_text_norm") or ""
    this_text_norm = re.sub(r"[^a-z0-9]+", " ", caller_text.lower()).strip()
    last_ts = float(sched.get("voice_last_caller_text_ts") or 0)
    confirmation_norms = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "correct", "right"}
    # Do not suppress repeated confirmations. A caller can legitimately say "yes"
    # once to confirm an on-file address and then "yes" again to approve emergency
    # dispatch. Suppressing the second yes caused the dispatch question to repeat.
    if this_text_norm and this_text_norm == last_text_norm and this_text_norm not in confirmation_norms and (now_ts - last_ts) < 8:
        reply = conv.get("last_voice_reply") or "Got it."
        try:
            log_event("VOICE_DUPLICATE_TURN_SUPPRESSED", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": reply, "booking_created": bool(sched.get("booking_created") and sched.get("square_booking_id")), "manual_only": bool(sched.get("manual_only")), "end_call": False}
    sched["voice_last_caller_text_norm"] = this_text_norm
    sched["voice_last_caller_text_ts"] = now_ts

    if _voice_looks_like_live_person_demand(caller_text):
        sched["voice_customer_turn_count"] = int(sched.get("voice_customer_turn_count") or 0) + 1
        reply = "We handle residential scheduling through our automated booking assistant so our electricians can stay in the field. I can help get the request started now. What town is the work in?"
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        try:
            log_event("VOICE_LIVE_PERSON_DEMAND", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "end_call": False}

    sched["voice_customer_turn_count"] = int(sched.get("voice_customer_turn_count") or 0) + 1
    if len(caller_text.strip()) >= 4:
        sched["voice_meaningful_customer_turns"] = int(sched.get("voice_meaningful_customer_turns") or 0) + 1

    # Hard residential service-area closeout before any booking logic.
    if _voice_is_out_of_area_town(caller_text):
        reply = _voice_out_of_area_reply()
        try:
            log_event("VOICE_OUT_OF_AREA_TOWN", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
        except Exception:
            pass
        return _voice_close_with_reply(phone, conv, reply, "residential_out_of_area_town")

    # True hazard/emergency descriptions must not be sent through the generic address parser;
    # it can mistake the issue text for an address and create the "what number is it on..." loop.
    try:
        hazard_reply = _voice_hazard_intake_fast_path(phone, conv, caller_text)
    except Exception as e:
        hazard_reply = None
        try:
            log_event("VOICE_HAZARD_FAST_PATH_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
    if hazard_reply:
        hazard_reply = _voice_naturalize_reply(hazard_reply)
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": hazard_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        conv["last_voice_reply"] = hazard_reply
        try:
            log_event("VOICE_HAZARD_FAST_PATH", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(hazard_reply), "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": hazard_reply, "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    # If we just asked the caller to confirm a saved/on-file address and they say yes,
    # trust that confirmation and move forward. Do not ask for the address again.
    try:
        address_confirm_reply = _voice_known_address_confirm_fast_path(conv, caller_text)
    except Exception as e:
        address_confirm_reply = None
        try:
            log_event("VOICE_ADDRESS_CONFIRM_FAST_PATH_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
    if address_confirm_reply:
        # After a saved/on-file address is confirmed, immediately enforce residential service radius.
        too_far, travel_minutes = _voice_residential_address_too_far(conv)
        if too_far:
            reply = _voice_out_of_area_reply()
            try:
                log_event("VOICE_OUT_OF_AREA_ADDRESS", phone, {"travel_minutes": travel_minutes, "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return _voice_close_with_reply(phone, conv, reply, "residential_out_of_area_address")
        address_confirm_reply = _voice_naturalize_reply(address_confirm_reply)
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": address_confirm_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        conv["last_voice_reply"] = address_confirm_reply
        try:
            log_event("VOICE_ADDRESS_CONFIRM_FAST_PATH", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(address_confirm_reply), "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": address_confirm_reply, "booking_created": bool(sched.get("booking_created") and sched.get("square_booking_id")), "manual_only": bool(sched.get("manual_only")), "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": bool(sched.get("voice_close_after_reply"))}

    # If the caller says "you already have it" after an address prompt, trust the
    # confirmed/saved address and move forward instead of arguing in a loop.
    try:
        address_have_it_reply = _voice_address_already_have_it_fast_path(conv, caller_text)
    except Exception as e:
        address_have_it_reply = None
        try:
            log_event("VOICE_ADDRESS_ALREADY_HAVE_IT_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
    if address_have_it_reply:
        address_have_it_reply = _voice_naturalize_reply(address_have_it_reply)
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": address_have_it_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        conv["last_voice_reply"] = address_have_it_reply
        try:
            log_event("VOICE_ADDRESS_ALREADY_HAVE_IT", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(address_have_it_reply), "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": address_have_it_reply, "booking_created": bool(sched.get("booking_created") and sched.get("square_booking_id")), "manual_only": bool(sched.get("manual_only")), "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    # Emergency dispatch acceptance/coverage is a special voice path. Once the customer
    # says yes/okay/let's do it, do not repeat the $395 dispatch question.
    try:
        emergency_fast_reply = _voice_emergency_dispatch_fast_path(phone, conv, caller_text)
    except Exception as e:
        emergency_fast_reply = None
        try:
            log_event("VOICE_EMERGENCY_FAST_PATH_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
    if emergency_fast_reply:
        emergency_fast_reply = _voice_naturalize_reply(emergency_fast_reply)
        booking_created_fast = bool(sched.get("booking_created") and sched.get("square_booking_id"))
        if booking_created_fast:
            emergency_fast_reply = _voice_finalize_booking_reply(conv, emergency_fast_reply)
            sched["voice_close_after_reply"] = True
            sched["voice_booking_completed_close"] = True
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": emergency_fast_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        conv["last_voice_reply"] = emergency_fast_reply
        _voice_maybe_send_booking_sms(phone, conv, emergency_fast_reply)
        try:
            log_event("VOICE_EMERGENCY_FAST_PATH", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(emergency_fast_reply), "booking_created": booking_created_fast, "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": emergency_fast_reply, "booking_created": booking_created_fast, "manual_only": bool(sched.get("manual_only")), "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": bool(sched.get("voice_close_after_reply"))}

    # If the caller is deciding whether the $195 evaluation works, do not offer slots until they say yes.
    price_confirm_reply = _voice_eval_price_confirm_fast_path(conv, caller_text)
    if price_confirm_reply:
        price_confirm_reply = _voice_naturalize_reply(price_confirm_reply)
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": price_confirm_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        conv["last_voice_reply"] = price_confirm_reply
        return {"reply_to_customer": price_confirm_reply, "booking_created": bool(sched.get("booking_created") and sched.get("square_booking_id")), "manual_only": bool(sched.get("manual_only")), "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": bool(sched.get("voice_close_after_reply"))}

    # Multiple-address callers need a first-location path, not an address loop.
    multi_addr_reply = _voice_multiple_address_reply(conv, caller_text)
    if multi_addr_reply:
        multi_addr_reply = _voice_naturalize_reply(multi_addr_reply)
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": multi_addr_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        conv["last_voice_reply"] = multi_addr_reply
        return {"reply_to_customer": multi_addr_reply, "booking_created": False, "manual_only": bool(sched.get("manual_only")), "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    # If we are waiting for a missing name, handle a one-word last-name reply
    # directly. This prevents loops like: "Prevost" -> "what is your last name?".
    try:
        name_fast_reply = _voice_name_fast_path(conv, caller_text)
    except Exception as e:
        name_fast_reply = None
        try:
            log_event("VOICE_NAME_FAST_PATH_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
    if name_fast_reply:
        name_fast_reply = _voice_naturalize_reply(name_fast_reply)
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": name_fast_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        conv["last_voice_reply"] = name_fast_reply
        try:
            log_event("VOICE_NAME_FAST_PATH", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(name_fast_reply), "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": name_fast_reply, "booking_created": bool(sched.get("booking_created") and sched.get("square_booking_id")), "manual_only": bool(sched.get("manual_only")), "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    # Hard guard against stale address gates after a voice-confirmed saved address.
    if sched.get("address_verified") and (sched.get("pending_step") or "").strip().lower() == "need_address":
        sched["pending_step"] = None
        sched["address_missing"] = None

    # Route-level offered-slot handling for voice. This preserves the exact offered time.
    try:
        offered_slot_reply = _voice_apply_offered_slot_fast_path(conv, caller_text)
    except Exception:
        offered_slot_reply = None
    if offered_slot_reply:
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": offered_slot_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        conv["last_voice_reply"] = offered_slot_reply
        try:
            log_event("VOICE_OFFERED_SLOT_FAST_PATH", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(offered_slot_reply), "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": offered_slot_reply, "booking_created": bool(sched.get("booking_created") and sched.get("square_booking_id")), "manual_only": bool(sched.get("manual_only")), "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    # If the caller gives the email at the end, book immediately and speak the final confirmation,
    # rather than saying a filler phrase and hanging up after the SMS is sent.
    try:
        email_fast_reply = _voice_email_fast_path(phone, conv, caller_text)
    except Exception as e:
        email_fast_reply = None
        try:
            log_event("VOICE_EMAIL_FAST_PATH_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
    if email_fast_reply:
        email_fast_reply = _voice_naturalize_reply(email_fast_reply)
        booking_created_email = bool(sched.get("booking_created") and sched.get("square_booking_id"))
        if booking_created_email:
            sched["voice_close_after_reply"] = True
            sched["voice_booking_completed_close"] = True
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": email_fast_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        conv["last_voice_reply"] = email_fast_reply
        _voice_maybe_send_booking_sms(phone, conv, email_fast_reply)
        try:
            log_event("VOICE_EMAIL_FAST_PATH", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(email_fast_reply), "booking_created": booking_created_email, "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": email_fast_reply, "booking_created": booking_created_email, "manual_only": bool(sched.get("manual_only")), "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": bool(sched.get("voice_close_after_reply"))}

    # If the call is dragging, move it back to SMS instead of letting the agent
    # burn minutes or frustrate the customer.
    if int(sched.get("voice_customer_turn_count") or 0) > VOICE_AGENT_MAX_CUSTOMER_TURNS and not sched.get("booking_created"):
        sched["voice_close_after_reply"] = True
        _voice_send_drag_handoff_sms(phone, conv, "max_customer_turns")
        reply = "I’m going to send you a text so we can finish this cleanly without keeping you on the phone."
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
        return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "end_call": True}

    # Preserve a cumulative transcript for monitor visibility and initial classification context.
    previous = conv.get("cleaned_transcript") or ""
    conv["cleaned_transcript"] = _voice_to_sms_text((previous + "\n" + caller_text).strip())

    sched = conv.setdefault("sched", {})
    result = None
    try:
        with app.test_request_context(
            "/voice/realtime-media",
            method="POST",
            data={
                "From": phone,
                "Body": caller_text,
                "CallSid": call_sid or conv.get("last_call_sid") or "",
                "MessageSid": f"voice-{uuid.uuid4()}",
            },
        ):
            result = generate_reply_for_inbound(
                conv.get("cleaned_transcript") or caller_text,
                conv.get("category"),
                sched.get("appointment_type") or conv.get("appointment_type"),
                conv.get("initial_sms") or "",
                caller_text,
                sched.get("scheduled_date"),
                sched.get("scheduled_time"),
                sched.get("raw_address") or sched.get("address_candidate"),
            )
    except Exception as e:
        try:
            log_event("VOICE_OS_TURN_ERROR", phone, {"error": repr(e), "trace": traceback.format_exc(limit=4), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
        result = {"sms_body": "Sorry, can you say that again?"}

    try:
        _voice_sanitize_name_fields(conv.setdefault("profile", {}), sched)
    except Exception:
        pass
    # If the generic SRB path accepted an address, stop residential jobs that are too far away
    # before offering slots or continuing the booking.
    try:
        if sched.get("address_verified"):
            too_far, travel_minutes = _voice_residential_address_too_far(conv)
            if too_far:
                reply_far = _voice_out_of_area_reply()
                try:
                    log_event("VOICE_OUT_OF_AREA_ADDRESS", phone, {"travel_minutes": travel_minutes, "reply": _safe_monitor_text(reply_far), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return _voice_close_with_reply(phone, conv, reply_far, "residential_out_of_area_address")
    except Exception as e:
        try:
            log_event("VOICE_OUT_OF_AREA_CHECK_ERROR", phone, {"error": repr(e), "call_sid": call_sid}, conv)
        except Exception:
            pass
    reply = _voice_naturalize_reply((result or {}).get("sms_body") or "Sorry, can you say that again?")
    reply = _voice_hold_eval_price_for_confirmation(conv, reply)
    if _voice_reply_asks_for_phone(reply) and phone:
        profile_local = conv.setdefault("profile", {})
        if _voice_profile_first_name(profile_local) and _voice_profile_last_name(profile_local):
            if not _voice_has_email(conv):
                sched["pending_step"] = "need_email"
                sched["state"] = "waiting_for_email"
                reply = "What's the best email address for the appointment?"
        elif _voice_profile_first_name(profile_local):
            sched["pending_step"] = "need_name"
            sched["state"] = "waiting_for_name"
            reply = "What's your last name?"
    booking_created = bool(sched.get("booking_created") and sched.get("square_booking_id"))
    if booking_created:
        reply = _voice_finalize_booking_reply(conv, reply)
        # After the final spoken confirmation, close the call cleanly.
        sched["voice_close_after_reply"] = True
        sched["voice_booking_completed_close"] = True
    conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
    conv["last_voice_reply"] = reply
    _voice_maybe_send_booking_sms(phone, conv, reply)
    try:
        log_event("VOICE_OS_TURN", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "booking_created": booking_created, "call_sid": call_sid}, conv)
    except Exception:
        pass
    return {
        "reply_to_customer": reply,
        "booking_created": booking_created,
        "manual_only": bool(sched.get("manual_only") or conv.get("thread_type") == "manual_only"),
        "pending_step": sched.get("pending_step"),
        "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"),
        "end_call": bool(sched.get("voice_close_after_reply")),
    }


def _extract_twilio_start_payload(start: dict) -> tuple[str, str, str]:
    custom = start.get("customParameters") or {}
    phone = (custom.get("from") or custom.get("From") or "").replace("whatsapp:", "").strip()
    call_sid = custom.get("callSid") or custom.get("CallSid") or start.get("callSid") or ""
    stream_sid = start.get("streamSid") or ""
    return phone, call_sid, stream_sid


def _send_openai_event(openai_ws, event: dict) -> None:
    openai_ws.send(json.dumps(event))

def _wait_for_openai_session_updated(openai_ws, phone: str = "", call_sid: str = "", timeout_seconds: int = 8) -> None:
    """Wait until OpenAI confirms the session update before creating audio output.

    Without this handshake, the first response can be generated using the default
    PCM audio format. Twilio expects raw mulaw/8000, so default PCM played back
    through Twilio sounds like loud static.
    """
    deadline = time.time() + max(1, int(timeout_seconds or 8))
    original_timeout_set = False
    try:
        while time.time() < deadline:
            remaining = max(0.5, deadline - time.time())
            try:
                openai_ws.settimeout(remaining)
                original_timeout_set = True
            except Exception:
                pass
            raw = openai_ws.recv()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except Exception:
                try:
                    log_event("VOICE_OPENAI_HANDSHAKE_NON_JSON", phone, {"raw": str(raw)[:500], "call_sid": call_sid})
                except Exception:
                    pass
                continue

            etype = event.get("type")
            if etype == "session.updated":
                try:
                    session = event.get("session") or {}
                    audio = session.get("audio") or {}
                    log_event("VOICE_OPENAI_SESSION_UPDATED", phone, {
                        "call_sid": call_sid,
                        "input_format": (((audio.get("input") or {}).get("format") or {}).get("type")),
                        "output_format": (((audio.get("output") or {}).get("format") or {}).get("type")),
                        "voice": ((audio.get("output") or {}).get("voice")),
                    })
                except Exception:
                    pass
                return

            if etype in {"error", "invalid_request_error"}:
                try:
                    log_event("VOICE_OPENAI_SESSION_UPDATE_ERROR", phone, {"event": event, "call_sid": call_sid})
                except Exception:
                    pass
                raise RuntimeError(f"OpenAI rejected session update: {event}")

            # Ignore non-critical early server events, but keep a breadcrumb.
            try:
                log_event("VOICE_OPENAI_HANDSHAKE_EVENT", phone, {"type": etype, "call_sid": call_sid})
            except Exception:
                pass

        try:
            log_event("VOICE_OPENAI_SESSION_UPDATE_TIMEOUT", phone, {"call_sid": call_sid})
        except Exception:
            pass
        raise TimeoutError("Timed out waiting for OpenAI session.updated")
    finally:
        if original_timeout_set:
            try:
                openai_ws.settimeout(None)
            except Exception:
                pass



def _send_twilio_media(twilio_ws, stream_sid: str, payload: str) -> None:
    if not (stream_sid and payload):
        return
    twilio_ws.send(json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}}))


def _send_twilio_clear(twilio_ws, stream_sid: str) -> None:
    if not stream_sid:
        return
    try:
        twilio_ws.send(json.dumps({"event": "clear", "streamSid": stream_sid}))
    except Exception:
        pass


def _send_twilio_mark(twilio_ws, stream_sid: str, name: str) -> None:
    if not (stream_sid and name):
        return
    twilio_ws.send(json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": name}}))


def _linear16_to_mulaw(sample: int) -> int:
    """Small local G.711 mu-law encoder so we do not depend on audioop."""
    sample = max(-32768, min(32767, int(sample)))
    BIAS = 0x84
    CLIP = 32635
    sign = 0x80 if sample < 0 else 0
    if sample < 0:
        sample = -sample
    if sample > CLIP:
        sample = CLIP
    sample += BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        mask >>= 1
        exponent -= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def _voice_thinking_click_payload(duration_ms: int | None = None) -> str:
    """Return a smooth low-level PCMU/8000 processing cue.

    This is meant to feel like a modern call-center AI "processing" sound:
    soft flutter/boop, rounded edges, small reverb tail, and fade-out. It is
    raw G.711 mu-law with no WAV headers for Twilio Media Streams.
    """
    ms = max(550, min(1800, int(duration_ms or VOICE_AGENT_THINKING_SOUND_MS or 950)))
    rate = 8000
    total = int(rate * (ms / 1000.0))
    pcm = [0] * total

    # Softer, smoother clustered tones. This avoids the old sharp high-G ping.
    # Frequencies are lower and close together so the result feels like a small
    # flutter/boop rather than separate beeps.
    starts_ms = [70, 150, 245, 365]
    freqs = [255, 335, 292, 385]
    lengths_ms = [210, 230, 250, 330]
    gains = [0.18, 0.16, 0.13, 0.10]

    for n, (start_ms, freq, length_ms, gain) in enumerate(zip(starts_ms, freqs, lengths_ms, gains)):
        start = int(rate * start_ms / 1000.0)
        length = min(int(rate * length_ms / 1000.0), max(0, total - start))
        if length <= 0:
            continue
        phase_offset = n * 0.73
        for i in range(length):
            idx = start + i
            t = i / rate
            x = i / max(1, length - 1)
            # Rounded attack and slow decay so there is no hard click.
            attack = min(1.0, i / max(1, int(rate * 0.035)))
            release = (1.0 - x) ** 1.7
            # Global fade makes the whole cue disappear naturally.
            global_fade = max(0.0, 1.0 - (idx / max(1, total)) ** 1.55)
            # Small pitch movement creates the fluent flutter.
            inst_freq = freq + 22 * math.sin(2 * math.pi * 5.5 * t + phase_offset)
            tone = math.sin(2 * math.pi * inst_freq * t + phase_offset)
            tone += 0.10 * math.sin(2 * math.pi * inst_freq * 1.38 * t + 1.1)
            sample = int(12500 * gain * attack * release * global_fade * tone)
            pcm[idx] = max(-14000, min(14000, pcm[idx] + sample))

            # Soft reverb taps. They are deliberately quiet so they add air
            # without sounding like echo or static over the phone.
            for delay_ms, echo_gain in ((42, 0.20), (88, 0.10), (138, 0.045)):
                eidx = idx + int(rate * delay_ms / 1000.0)
                if eidx < total:
                    pcm[eidx] = max(-14000, min(14000, pcm[eidx] + int(sample * echo_gain)))

    # Low, quiet under-bed to glue the chirps together. Phone compression often
    # eats very quiet audio, so keep it present but far below the voice level.
    bed_len = min(total, int(rate * 0.72))
    for i in range(bed_len):
        t = i / rate
        fade = max(0.0, 1.0 - (i / max(1, bed_len)) ** 1.25)
        bed = int(360 * fade * math.sin(2 * math.pi * 170 * t))
        pcm[i] = max(-14000, min(14000, pcm[i] + bed))

    # Very short leading/trailing fade prevents clicks at the packet boundary.
    fade_samples = int(rate * 0.025)
    for i in range(min(fade_samples, total)):
        mult = i / max(1, fade_samples)
        pcm[i] = int(pcm[i] * mult)
        j = total - 1 - i
        pcm[j] = int(pcm[j] * mult)

    data = bytearray(_linear16_to_mulaw(sample) for sample in pcm)
    return base64.b64encode(bytes(data)).decode("ascii")

def _send_twilio_thinking_sound(twilio_ws, stream_sid: str, phone: str = "", call_sid: str = "", reason: str = "") -> None:
    if not (VOICE_AGENT_THINKING_SOUND_ENABLED and stream_sid):
        return
    try:
        _send_twilio_media(twilio_ws, stream_sid, _voice_thinking_click_payload())
        try:
            log_event("VOICE_THINKING_SOUND_SENT", phone, {"reason": reason, "call_sid": call_sid, "stream_sid": stream_sid})
        except Exception:
            pass
    except Exception as e:
        try:
            log_event("VOICE_THINKING_SOUND_FAILED", phone, {"error": repr(e), "reason": reason, "call_sid": call_sid})
        except Exception:
            pass


def _handle_realtime_function_call(openai_ws, phone: str, call_sid: str, item: dict, handled_call_ids: set | None = None, twilio_ws=None, stream_sid: str = "") -> bool:
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return False
    name = item.get("name") or ""
    if name != "prevolt_os_turn":
        return False
    call_id = item.get("call_id") or item.get("id") or ""
    if handled_call_ids is not None and call_id:
        if call_id in handled_call_ids:
            return True
        handled_call_ids.add(call_id)
    try:
        args = json.loads(item.get("arguments") or "{}")
    except Exception:
        args = {}
    try:
        if twilio_ws is not None and stream_sid:
            _send_twilio_thinking_sound(twilio_ws, stream_sid, phone, call_sid, "before_tool_processing")
    except Exception:
        pass
    output = process_prevolt_voice_turn(phone, call_sid, args.get("caller_text") or "")
    if output.get("end_call"):
        try:
            sched_local = hydrate_voice_conversation(phone, call_sid).setdefault("sched", {})
            sched_local["voice_close_after_reply"] = True
            # Do not close on the function-call response.done event. The final
            # confirmation is spoken by the *next* audio-only response we create below.
            # Closing immediately after the function call caused the confirmation to be
            # sent by SMS but cut off before the caller heard it.
            sched_local["voice_waiting_for_final_audio_done"] = True
        except Exception:
            pass
    _send_openai_event(openai_ws, {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(output),
        },
    })
    exact_reply = _voice_to_sms_text(str(output.get("reply_to_customer") or "Sorry, can you say that again?"))
    _send_openai_event(openai_ws, {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio"],
            "tool_choice": "none",
            "instructions": (
                "Say ONLY this exact phone reply, word for word, including every question in it, then stop speaking. "
                "Do not say anything before it. Do not say you are checking, thinking, waiting, or finishing. "
                "Speak at a natural medium pace. Do not slow down or add dramatic pauses at commas. "
                "Do not add any acknowledgement, greeting, filler phrase, or extra question. "
                "Do not stop after the first clause or first sentence; speak the complete exact reply. "
                "Exact reply: " + json.dumps(exact_reply)
            ),
        },
    })
    return True


if sock is not None:
    @sock.route("/voice/realtime-media")
    def voice_realtime_media(ws):
        """Bidirectional bridge: Twilio Media Streams <-> OpenAI Realtime."""
        openai_ws = None
        stream_sid = ""
        phone = ""
        call_sid = ""
        started_at = time.time()
        try:
            ready, reason = voice_agent_runtime_ready()
            if not ready:
                try:
                    log_event("VOICE_WS_REJECTED", "", {"reason": reason})
                except Exception:
                    pass
                return

            # Twilio Media Streams sends a `connected` message first, then the
            # `start` message with streamSid/customParameters. Do not reject the
            # initial connected event or the call will immediately hang up.
            start_msg = None
            start_deadline = time.time() + 10
            while time.time() < start_deadline:
                remaining = max(1, int(start_deadline - time.time()))
                first_raw = ws.receive(timeout=remaining)
                if not first_raw:
                    continue
                first_msg = json.loads(first_raw)
                first_event = first_msg.get("event")
                if first_event == "connected":
                    try:
                        log_event("VOICE_TWILIO_CONNECTED", "", {"event": first_msg})
                    except Exception:
                        pass
                    continue
                if first_event == "start":
                    start_msg = first_msg
                    break
                if first_event == "stop":
                    try:
                        log_event("VOICE_STREAM_STOPPED_BEFORE_START", "", {"event": first_msg})
                    except Exception:
                        pass
                    return
                try:
                    log_event("VOICE_START_UNEXPECTED", "", {"event": first_msg})
                except Exception:
                    pass

            if not start_msg:
                try:
                    log_event("VOICE_START_TIMEOUT", "", {"reason": "no_twilio_start_after_connected"})
                except Exception:
                    pass
                return

            phone, call_sid, stream_sid = _extract_twilio_start_payload(start_msg.get("start") or {})
            hydrate_voice_conversation(phone, call_sid)
            try:
                log_event("VOICE_STREAM_STARTED", phone, {"call_sid": call_sid, "stream_sid": stream_sid, "model": OPENAI_REALTIME_MODEL, "voice": OPENAI_REALTIME_VOICE})
            except Exception:
                pass

            openai_ws = websocket_client.create_connection(
                f"{OPENAI_REALTIME_WS_URL}?model={OPENAI_REALTIME_MODEL}",
                header=[
                    f"Authorization: Bearer {OPENAI_API_KEY}",
                    "OpenAI-Safety-Identifier: prevolt-os-phone-agent",
                ],
                timeout=10,
            )
            try:
                # Use the timeout only for connect; after the call starts, an
                # otherwise healthy phone conversation can have long silences.
                openai_ws.settimeout(None)
            except Exception:
                pass
            _send_openai_event(openai_ws, _voice_session_update_event())
            _wait_for_openai_session_updated(openai_ws, phone, call_sid, timeout_seconds=8)

            stop_flag = {"stop": False}
            handled_function_calls = set()
            final_mark = {"name": ""}

            def openai_to_twilio():
                nonlocal stream_sid, phone, call_sid
                while not stop_flag["stop"]:
                    try:
                        message = openai_ws.recv()
                        if not message:
                            break
                        event = json.loads(message)
                        etype = event.get("type")
                        if etype in {"response.output_audio.delta", "response.audio.delta"}:
                            _send_twilio_media(ws, stream_sid, event.get("delta") or "")
                        elif etype == "input_audio_buffer.speech_started":
                            _send_twilio_clear(ws, stream_sid)
                        elif etype == "input_audio_buffer.speech_stopped":
                            # Wait until the backend tool actually starts. Sending
                            # a cue here and again before tool processing caused
                            # duplicate buffered sounds and sometimes got swallowed.
                            pass
                        elif etype == "response.done":
                            output = ((event.get("response") or {}).get("output") or [])
                            had_function_call = any(isinstance(item, dict) and item.get("type") == "function_call" for item in output)
                            for item in output:
                                _handle_realtime_function_call(openai_ws, phone, call_sid, item, handled_function_calls, ws, stream_sid)
                            try:
                                conv_local = hydrate_voice_conversation(phone, call_sid)
                                sched_local = conv_local.setdefault("sched", {})
                                if had_function_call:
                                    # A tool call may have queued the final spoken reply with response.create.
                                    # Do NOT close here or the final confirmation will be sent by SMS but never spoken.
                                    if sched_local.get("voice_close_after_reply"):
                                        sched_local["voice_final_reply_queued"] = True
                                    continue
                                if sched_local.get("voice_close_after_reply"):
                                    # This response.done belongs to the spoken final reply.
                                    # Send a Twilio mark and close only after Twilio says the buffered audio played.
                                    mark_name = f"final-{call_sid or stream_sid or uuid.uuid4()}"
                                    final_mark["name"] = mark_name
                                    try:
                                        _send_twilio_mark(ws, stream_sid, mark_name)
                                        log_event("VOICE_FINAL_MARK_SENT", phone, {"mark": mark_name, "call_sid": call_sid, "stream_sid": stream_sid})
                                    except Exception:
                                        time.sleep(7.0)
                                        stop_flag["stop"] = True
                                        try:
                                            ws.close()
                                        except Exception:
                                            pass
                                        break
                            except Exception:
                                pass
                        elif etype == "response.output_item.done":
                            # Do not create the follow-up spoken response here. OpenAI may still
                            # consider the parent response active, which causes
                            # conversation_already_has_active_response errors. Function calls are
                            # handled from response.done instead.
                            pass
                        elif etype in {"error", "invalid_request_error"}:
                            try:
                                log_event("VOICE_OPENAI_ERROR", phone, {"event": event, "call_sid": call_sid})
                            except Exception:
                                pass
                    except Exception as e:
                        if not stop_flag["stop"]:
                            try:
                                log_event("VOICE_OPENAI_RECV_ERROR", phone, {"error": repr(e), "call_sid": call_sid})
                            except Exception:
                                pass
                        break

            t = threading.Thread(target=openai_to_twilio, daemon=True)
            t.start()
            initial_event = _voice_initial_response_event()
            if initial_event:
                _send_openai_event(openai_ws, initial_event)

            while True:
                if time.time() - started_at > VOICE_AGENT_MAX_SECONDS:
                    try:
                        conv = hydrate_voice_conversation(phone, call_sid)
                        conv.setdefault("sched", {})["voice_close_after_reply"] = True
                        _voice_send_drag_handoff_sms(phone, conv, "max_duration")
                        log_event("VOICE_MAX_DURATION_REACHED", phone, {"call_sid": call_sid, "stream_sid": stream_sid})
                    except Exception:
                        pass
                    break
                raw = ws.receive(timeout=VOICE_AGENT_IDLE_SECONDS)
                if raw is None:
                    try:
                        conv = hydrate_voice_conversation(phone, call_sid)
                        _voice_send_resume_sms_if_needed(phone, conv, "idle_timeout")
                        log_event("VOICE_IDLE_TIMEOUT", phone, {"call_sid": call_sid, "stream_sid": stream_sid})
                    except Exception:
                        pass
                    break
                msg = json.loads(raw)
                event_type = msg.get("event")
                if event_type == "start":
                    # Already handled before connecting to OpenAI. Ignore duplicate start messages.
                    continue
                elif event_type == "media":
                    payload = ((msg.get("media") or {}).get("payload") or "")
                    if payload:
                        _send_openai_event(openai_ws, {"type": "input_audio_buffer.append", "audio": payload})
                elif event_type == "mark":
                    mark_name = (((msg.get("mark") or {}).get("name")) or "")
                    if final_mark.get("name") and mark_name == final_mark.get("name"):
                        try:
                            log_event("VOICE_FINAL_MARK_PLAYED", phone, {"mark": mark_name, "call_sid": call_sid, "stream_sid": stream_sid})
                        except Exception:
                            pass
                        stop_flag["stop"] = True
                        try:
                            ws.close()
                        except Exception:
                            pass
                        break
                elif event_type == "stop":
                    try:
                        conv = hydrate_voice_conversation(phone, call_sid)
                        _voice_send_resume_sms_if_needed(phone, conv, "caller_hung_up")
                        log_event("VOICE_STREAM_STOPPED", phone, {"call_sid": call_sid, "stream_sid": stream_sid})
                    except Exception:
                        pass
                    break
        except Exception as e:
            try:
                conv = hydrate_voice_conversation(phone, call_sid) if (phone or call_sid) else {}
                err = repr(e)
                # Twilio/browser side closes often surface as simple_websocket ConnectionClosed(1005).
                # Treat that as a normal dropped/ended call, not an application crash.
                if phone and conv:
                    _voice_send_resume_sms_if_needed(phone, conv, "voice_ws_closed" if "ConnectionClosed" in err else "voice_ws_fatal")
                event_name = "VOICE_WS_CLOSED" if "ConnectionClosed" in err else "VOICE_WS_FATAL"
                log_event(event_name, phone, {"error": err, "trace": traceback.format_exc(limit=6), "call_sid": call_sid, "stream_sid": stream_sid})
            except Exception:
                pass
        finally:
            try:
                stop_flag["stop"] = True
            except Exception:
                pass
            try:
                if openai_ws:
                    openai_ws.close()
            except Exception:
                pass


MONITOR_DB_PATH = os.environ.get("PREVOLT_MONITOR_DB_PATH", "/tmp/prevolt_monitor.db")

def _monitor_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _monitor_connect():
    conn = sqlite3.connect(MONITOR_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_monitor_db() -> None:
    try:
        with _monitor_connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monitor_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    phone TEXT,
                    payload_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_monitor_events_ts ON monitor_events(ts DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_monitor_events_phone ON monitor_events(phone)")
    except Exception as e:
        print("[WARN] monitor db init failed:", repr(e))

def _safe_monitor_text(value, limit: int = 280) -> str:
    s = str(value or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > limit:
        s = s[:limit - 3].rstrip() + "..."
    return s

def _monitor_state_label(conv: dict) -> str:
    sched = (conv or {}).get("sched") or {}
    appt = (sched.get("appointment_type") or "").upper()
    thread_type = (conv or {}).get("thread_type")
    if thread_type in {"closed_lost", "vendor_sales_or_spam", "wrong_number_or_spam"} or sched.get("customer_hard_stop"):
        return "closed"
    if sched.get("booking_created") and sched.get("square_booking_id"):
        return "booked"
    if sched.get("manual_only") or thread_type in {"manual_only", "employment_inquiry", "commercial_bid_contact"}:
        return "manual"
    if sched.get("hard_emergency_detected") or sched.get("emergency_approved") or sched.get("awaiting_emergency_confirm"):
        return "emergency"
    step = (sched.get("pending_step") or "").strip().lower()
    mapping = {
        "need_address": "waiting_for_address",
        "need_date": "waiting_for_date",
        "need_time": "waiting_for_time",
        "need_name": "waiting_for_name",
        "need_email": "waiting_for_email",
    }
    return mapping.get(step, "active")

def _conversation_snapshot(phone: str, conv: dict) -> dict:
    conv = conv or {}
    profile = conv.get("profile") or {}
    sched = conv.get("sched") or {}
    first = (profile.get("active_first_name") or profile.get("first_name") or profile.get("recognized_first_name") or profile.get("voicemail_first_name") or "").strip()
    last = (profile.get("active_last_name") or profile.get("last_name") or profile.get("recognized_last_name") or profile.get("voicemail_last_name") or "").strip()
    name = " ".join([p for p in [first, last] if p]).strip()
    address = scrub_non_address_tail((sched.get("raw_address") or "").strip())
    if address and address != (sched.get("raw_address") or "").strip():
        sched["raw_address"] = address
    return {
        "phone": phone or "",
        "name": name,
        "address": address,
        "scheduled_date": sched.get("scheduled_date"),
        "scheduled_time": sched.get("scheduled_time"),
        "appointment_type": sched.get("appointment_type"),
        "state": _monitor_state_label(conv),
        "booking_created": bool(sched.get("booking_created") and sched.get("square_booking_id")),
        "square_booking_id": sched.get("square_booking_id"),
        "pending_step": sched.get("pending_step"),
        "last_sms_body": conv.get("last_sms_body"),
        "thread_type": conv.get("thread_type"),
        "manual_only": bool(sched.get("manual_only")),
        "customer_hard_stop": bool(sched.get("customer_hard_stop")),
        "booking_allowed": sched.get("booking_allowed", True),
        "closed_reason": sched.get("closed_reason"),
        "address_verified": bool(sched.get("address_verified")),
        "address_missing": sched.get("address_missing"),
        "awaiting_slot_offer_choice": bool(sched.get("awaiting_slot_offer_choice")),
        "offered_slot_options": sched.get("offered_slot_options") or [],
        "last_ai_reason": sched.get("last_ai_reason"),
        "updated_at": _monitor_now_iso(),
    }

def log_event(event_type: str, phone: str = "", payload: dict | None = None, conv: dict | None = None) -> None:
    try:
        init_monitor_db()
        base_payload = dict(payload or {})
        if conv is not None:
            base_payload.setdefault("state", _monitor_state_label(conv))
            base_payload.setdefault("snapshot", _conversation_snapshot(phone, conv))
        # During live voice testing, mirror VOICE_* events to Render stdout so
        # failures are visible immediately in Render logs instead of only in
        # the monitor database.
        if str(event_type or "").startswith("VOICE"):
            try:
                printable = json.dumps(base_payload, ensure_ascii=False)[:1200]
            except Exception:
                printable = str(base_payload)[:1200]
            print(f"[PREVOLT_OS_EVENT] {event_type} phone={phone or ''} payload={printable}", flush=True)
        with _monitor_connect() as conn:
            conn.execute(
                "INSERT INTO monitor_events (ts, event_type, phone, payload_json) VALUES (?, ?, ?, ?)",
                (_monitor_now_iso(), event_type, phone or "", json.dumps(base_payload, ensure_ascii=False)),
            )
    except Exception as e:
        print("[WARN] monitor log_event failed:", repr(e), flush=True)

def _monitor_booking_return(phone: str, status: str, payload: dict, convo: dict | None = None) -> dict:
    try:
        log_event("BOOKING_RESULT", phone, {"status": status, **(payload or {})}, convo)
    except Exception:
        pass
    return payload

init_monitor_db()


# ---------------------------------------------------
# WhatsApp SMS Helper (Testing Path Only)
# ---------------------------------------------------

def humanize_time(t: str) -> str:
    """
    Convert a stored time string into a friendly 12-hour display.
    Accepts formats like "15:00", "15:00:00", "1500", "3pm", "3 pm", "3:00 PM".
    """
    t = (t or "").strip()
    if not t:
        return ""
    # Already contains am/pm -> normalize spacing/case
    m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([aApP])[mM]\s*$", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or "00")
        ap = "AM" if m.group(3).lower() == "a" else "PM"
        if hh == 0:
            hh = 12
        if hh > 12:
            hh = ((hh - 1) % 12) + 1
        return f"{hh}:{mm:02d} {ap}"
    # HHMM (e.g., 1500)
    if re.fullmatch(r"\d{3,4}", t):
        t = t.zfill(4)
        hh = int(t[:2])
        mm = int(t[2:])
    else:
        # HH:MM(:SS)
        m2 = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", t)
        if not m2:
            return t  # fallback: return as-is
        hh = int(m2.group(1))
        mm = int(m2.group(2))
    ap = "AM" if hh < 12 else "PM"
    hh12 = hh % 12
    if hh12 == 0:
        hh12 = 12
    return f"{hh12}:{mm:02d} {ap}"

def extract_explicit_time_from_text(text: str) -> str | None:
    """
    Pull an explicit customer time out of a message without mistaking street
    numbers or phone numbers for appointment times.

    Safe examples:
      - "2pm" -> "14:00"
      - "around 2:30 pm" -> "14:30"
      - "1300" -> "13:00" ONLY when the whole reply is basically "1300"
      - "next Thursday at 1" -> "01:00" by legacy behavior
      - "1251 Washington ST" -> None
    """
    import re

    original = (text or "").strip()
    s = original.lower()
    if not s:
        return None

    if re.search(r"\b(noon|midday)\b", s):
        return "12:00"

    # Explicit AM/PM is always safe.
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\s*m\b", s, flags=re.I)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or "00")
        ap = m.group(3).lower()
        if hh == 12:
            hh = 0
        if ap == "p":
            hh += 12
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    # Explicit HH:MM is safe.
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", s)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"

    # Military-style bare time is safe ONLY when the whole reply is that time.
    # This prevents "1251 Washington ST" from becoming 12:51 PM.
    if re.fullmatch(r"\s*(?:at\s+)?(\d{3,4})\s*", s):
        raw = re.sub(r"\D", "", s).zfill(4)
        hh = int(raw[:2])
        mm = int(raw[2:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    # Human shorthand: "9", "4", "Tuesday at 2", "can you do 3 instead".
    # In appointment context, 1-5 usually means PM; 6-11 means AM; 12 means noon.
    def _business_hour(hh: int) -> int:
        if hh == 12:
            return 12
        if 1 <= hh <= 5:
            return hh + 12
        return hh

    m = re.search(r"\b(?:can|could)\s+you\s+do\s+(\d{1,2})(?::([0-5]\d))?\s*(am|pm)?\b", s)
    if not m:
        m = re.fullmatch(r"\s*(?:at\s+)?(\d{1,2})(?::([0-5]\d))?\s*(am|pm)?\s*", s)
    if not m:
        m = re.search(r"\bat\s+(\d{1,2})(?::([0-5]\d))?\s*(am|pm)?\b", s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or "00")
        marker = (m.group(3) or "").lower()
        if 1 <= hh <= 12:
            if marker == "pm" and hh < 12:
                hh += 12
            elif marker == "am" and hh == 12:
                hh = 0
            elif not marker:
                hh = _business_hour(hh)
            return f"{hh:02d}:{mm:02d}"

    return None

def send_sms(to_number: str, body: str) -> None:
    """
    Send a normal outbound SMS to the actual customer number.
    Requires TWILIO_FROM_NUMBER to be a real SMS-capable Twilio number.
    """
    if not twilio_client:
        print("[WARN] Twilio not configured. SMS not sent.")
        print("Message:", body)
        return

    if not TWILIO_FROM_NUMBER:
        print("[WARN] TWILIO_FROM_NUMBER is not configured. SMS not sent.")
        print("Target:", to_number)
        print("Message:", body)
        return

    try:
        sms_to = (to_number or "").replace("whatsapp:", "").strip()
        sms_from = (TWILIO_FROM_NUMBER or "").replace("whatsapp:", "").strip()

        if not sms_to:
            print("[WARN] Missing destination number. SMS not sent.")
            print("Message:", body)
            return

        msg = twilio_client.messages.create(
            body=body,
            from_=sms_from,
            to=sms_to
        )
        print("[SMS] Sent SID:", msg.sid, "to", sms_to)
        log_event("SMS_OUT", sms_to, {"sid": getattr(msg, "sid", None), "body": _safe_monitor_text(body)})
    except Exception as e:
        print("[ERROR] SMS send failed:", repr(e))
        log_event("SMS_OUT_FAILED", (to_number or "").replace("whatsapp:", "").strip(), {"body": _safe_monitor_text(body), "error": repr(e)})




def sync_confirmed_name_from_context(profile: dict, sched: dict) -> None:
    """
    Promote a full name that is already known inside the current thread into
    the active booking identity so we do not ask the customer for their name again.

    Important: this does NOT override the shared-number name engine. If the
    name engine is waiting for confirmation/selection, we leave it alone.
    """
    if not isinstance(profile, dict) or not isinstance(sched, dict):
        return

    # If the name engine is actively resolving a shared-number/new-person
    # conflict, do not silently choose a name.
    if (sched.get("name_engine_state") or "").strip():
        return

    def _norm_name(value: str) -> str:
        value = " ".join(str(value or "").strip().split())
        if not value:
            return ""
        return " ".join(part[:1].upper() + part[1:].lower() for part in value.split())

    active_first = _norm_name(profile.get("active_first_name") or profile.get("first_name") or "")
    active_last = _norm_name(profile.get("active_last_name") or profile.get("last_name") or "")
    if active_first and active_last:
        return

    # The monitor can display voicemail_first_name/voicemail_last_name, but
    # recompute_pending_step previously ignored those. That caused the bot to
    # ask for first/last name even though the voicemail said "Kyle Prevost".
    candidate_first = _norm_name(profile.get("voicemail_first_name") or "")
    candidate_last = _norm_name(profile.get("voicemail_last_name") or "")

    # Some paths store a full display name under profile["name"]. Use it only
    # when it cleanly looks like a normal two-part person name.
    if not (candidate_first and candidate_last):
        raw_name = " ".join(str(profile.get("name") or "").strip().split())
        parts = [p for p in re.sub(r"[^A-Za-z'\- ]", " ", raw_name).split() if p]
        if len(parts) >= 2 and len(parts) <= 4:
            candidate_first = candidate_first or _norm_name(parts[0])
            candidate_last = candidate_last or _norm_name(" ".join(parts[1:]))

    if not (candidate_first and candidate_last):
        return

    # If there are known people on this number and the voicemail name is not
    # one of them, shared-number logic must confirm it first.
    known_people = profile.get("known_people") or []
    if isinstance(known_people, list) and known_people:
        matched_known = False
        for person in known_people:
            if not isinstance(person, dict):
                continue
            k_first = _norm_name(person.get("first_name") or "")
            k_last = _norm_name(person.get("last_name") or "")
            if k_first and k_first.lower() == candidate_first.lower() and (not k_last or k_last.lower() == candidate_last.lower()):
                matched_known = True
                break
        if not matched_known:
            return

    profile["active_first_name"] = active_first or candidate_first
    profile["first_name"] = active_first or candidate_first
    profile["active_last_name"] = active_last or candidate_last
    profile["last_name"] = active_last or candidate_last
    profile["identity_source"] = profile.get("identity_source") or "thread_full_name"

def recompute_pending_step(profile: dict, sched: dict) -> None:
    if sched.get("non_service_thread") or sched.get("manual_only"):
        sched["pending_step"] = None
        return

    sync_confirmed_name_from_context(profile, sched)

    active_first = (profile.get("active_first_name") or profile.get("first_name") or profile.get("recognized_first_name") or "").strip()
    active_last = (profile.get("active_last_name") or profile.get("last_name") or profile.get("recognized_last_name") or "").strip()
    active_email = (profile.get("active_email") or profile.get("email") or profile.get("recognized_email") or "").strip()

    if not sched.get("appointment_type"):
        sched["pending_step"] = "need_appt_type"
    elif not sched.get("raw_address") or not sched.get("address_verified"):
        sched["pending_step"] = "need_address"
    elif not sched.get("scheduled_date"):
        sched["pending_step"] = "need_date"
    elif not sched.get("scheduled_time"):
        sched["pending_step"] = "need_time"
    elif not (active_first and active_last):
        sched["pending_step"] = "need_name"
    elif not active_email:
        sched["pending_step"] = "need_email"
    else:
        sched["pending_step"] = None

# ---------------------------------------------------
# Step 1 — Transcription (Whisper)
# ---------------------------------------------------
def transcribe_recording(recording_url: str) -> str:
    """
    Downloads voicemail audio (wav/mp3) and transcribes via Whisper.
    """
    wav_url = recording_url + ".wav"
    mp3_url = recording_url + ".mp3"

    def download(url):
        try:
            resp = requests.get(
                url,
                stream=True,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                timeout=12
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except Exception:
            return None

    resp = download(wav_url) or download(mp3_url)
    if resp is None:
        print("[ERROR] Voicemail download failed:", recording_url)
        return ""

    tmp_path = "/tmp/prevolt_voicemail.wav"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return transcript.text.strip()
    except Exception as e:
        print("[ERROR] Whisper transcription failed:", repr(e))
        return ""


# ---------------------------------------------------
# Step 2 — Transcript Cleanup
# ---------------------------------------------------
def clean_transcript_text(raw_text: str) -> str:
    """
    Minor cleanup only; does NOT change meaning.

    Hotfix: if Whisper returned no usable text, do NOT ask the LLM
    to clean an empty string. That was creating prompt-leak transcripts
    like "Please provide the voicemail transcription..." which then
    got treated as real customer voicemails.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You clean up voicemail transcriptions for an electrical contractor. "
                        "Fix transcription errors, preserve meaning exactly. No embellishments."
                    ),
                },
                {"role": "user", "content": raw_text},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("[WARN] Cleanup failed:", repr(e))
        return raw_text


# ---------------------------------------------------
# Voicemail Task Topic Fallback Extractor
# ---------------------------------------------------
def fallback_extract_task_topics(cleaned_text: str) -> list[str]:
    s = (cleaned_text or '').lower()
    if not s:
        return []

    patterns = [
        (r'\bev charger|car charger|tesla charger|level 2 charger\b', 'ev charger install'),
        (r'\bpanel upgrade|service upgrade|upgrade the panel|upgrade my panel\b', 'panel upgrade'),
        (r'\bsmoke.*panel|panel.*smoke|burning smell|sparks|arcing|buzzing panel\b', 'smoke from panel'),
        (r'\boutlet(s)?\b', 'outlet issue'),
        (r'\bgfci\b', 'gfci issue'),
        (r"\bbreaker keeps tripping|breaker wont reset|breaker won't reset|tripping breaker\b", 'breaker issue'),
        (r'\brecessed light|can light|pot light\b', 'recessed lighting'),
        (r'\bceiling fan\b', 'ceiling fan install'),
        (r'\bsubpanel\b', 'subpanel work'),
        (r'\bgenerator\b', 'generator hookup'),
        (r'\bservice mast|meter socket|weatherhead\b', 'service equipment issue'),
        (r'\blight(s)? not working|switch not working|fixture\b', 'lighting issue'),
    ]

    seen = []
    for pat, label in patterns:
        if re.search(pat, s, flags=re.I) and label not in seen:
            seen.append(label)
    return seen[:6]

# ---------------------------------------------------
# Step 3 — Voicemail Classifier (NO SMS GENERATED)
# ---------------------------------------------------
def generate_initial_sms(cleaned_text: str) -> dict:
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured info from a messy voicemail left for an electrical contractor. "
                        "The caller may ramble, pause, self-correct, and mention multiple jobs. "
                        "You DO NOT generate SMS replies.\n\n"
                        "Return strict JSON with these keys only:\n"
                        "{\n"
                        "  'category': str,\n"
                        "  'appointment_type': str,\n"
                        "  'detected_first_name': str|null,\n"
                        "  'detected_last_name': str|null,\n"
                        "  'detected_address': str|null,\n"
                        "  'detected_date': 'YYYY-MM-DD'|null,\n"
                        "  'detected_time': 'HH:MM'|null,\n"
                        "  'intent': 'schedule'|'quote'|'emergency'|'other',\n"
                        "  'task_topics': [str]\n"
                        "}\n\n"
                        "Rules for task_topics:\n"
                        "- Extract the real customer work items, not life story details.\n"
                        "- Return 1 to 6 short plain-English task labels.\n"
                        "- Examples: 'outlet issue', 'ev charger install', 'panel upgrade', 'recessed lighting', 'smoke from panel'.\n"
                        "- Keep them concise and electrician-facing.\n"
                        "- If nothing is clear, return an empty list.\n"
                        "\n"
                        "Non-service classification rules:\n"
                        "- If the caller is asking for employment, hiring, apprenticeship, a job, or to join the team, category must be 'employment inquiry', intent must be 'other', and appointment_type should be 'none'.\n"
                        "- If the caller is a general contractor, estimator, commercial client, facility contact, or is discussing a bid/proposal/email/drawings, category must be 'commercial bid contact' when applicable and intent must be 'other'.\n"
                        "- Do not invent a service appointment for non-service calls."
                    ),
                },
                {"role": "user", "content": cleaned_text},
            ],
        )

        data = json.loads(completion.choices[0].message.content)

        topics = data.get("task_topics") or []
        if not isinstance(topics, list):
            topics = []
        topics = [str(t).strip() for t in topics if str(t).strip()][:6]
        if len(topics) < 2:
            fallback_topics = fallback_extract_task_topics(cleaned_text)
            for topic in fallback_topics:
                if topic not in topics:
                    topics.append(topic)
            topics = topics[:6]

        return {
            "category": data.get("category"),
            "appointment_type": data.get("appointment_type"),
            "detected_first_name": data.get("detected_first_name"),
            "detected_last_name": data.get("detected_last_name"),
            "detected_address": data.get("detected_address"),
            "detected_date": data.get("detected_date"),
            "detected_time": data.get("detected_time"),
            "intent": data.get("intent"),
            "task_topics": topics,
        }

    except Exception as e:
        print("[ERROR] Voicemail classifier failed:", repr(e))
        return {
            "category": "OTHER",
            "appointment_type": "EVAL_195",
            "detected_first_name": None,
            "detected_last_name": None,
            "detected_address": None,
            "detected_date": None,
            "detected_time": None,
            "intent": "other",
            "task_topics": [],
        }




# ---------------------------------------------------
# Initial Voicemail SMS Builder (deterministic first text)
# ---------------------------------------------------
def hydrate_square_profile_by_phone(profile: dict, phone: str, force: bool = False) -> None:
    """Best-effort Square hydrate for repeat callers before the first text/call flow.

    force=True is used by the live voice assistant at the start of each new call
    so an existing Square customer address is available immediately. This does
    NOT reset the customer profile; it refreshes it from Square.
    """
    profile.setdefault("addresses", [])
    profile.setdefault("square_lookup_done", False)
    profile.setdefault("square_customer_id", None)
    profile.setdefault("recognized_first_name", None)
    profile.setdefault("recognized_last_name", None)
    profile.setdefault("recognized_email", None)
    profile.setdefault("active_first_name", None)
    profile.setdefault("active_last_name", None)
    profile.setdefault("active_email", None)
    profile.setdefault("identity_source", None)

    if profile.get("square_lookup_done") and not force:
        return

    try:
        cust = square_lookup_customer_by_phone(phone)
        if cust and cust.get("id"):
            profile["square_customer_id"] = cust.get("id")
            profile["recognized_first_name"] = cust.get("given_name")
            profile["recognized_last_name"] = cust.get("family_name")
            profile["recognized_email"] = cust.get("email_address")

            if not profile.get("active_first_name") and cust.get("given_name"):
                profile["active_first_name"] = cust.get("given_name")
            if not profile.get("active_last_name") and cust.get("family_name"):
                profile["active_last_name"] = cust.get("family_name")
            if not profile.get("active_email") and cust.get("email_address"):
                profile["active_email"] = cust.get("email_address")
            profile["identity_source"] = profile.get("identity_source") or "square_phone_match"

            caddr = cust.get("address") or {}
            line1 = (caddr.get("address_line_1") or "").strip()
            city = (caddr.get("locality") or "").strip()
            state = (caddr.get("administrative_district_level_1") or "").strip()
            zipc = (caddr.get("postal_code") or "").strip()
            if line1 and city and state:
                pretty = f"{line1}, {city}, {state} {zipc}".strip()
                if pretty and pretty not in profile["addresses"]:
                    profile["addresses"].append(pretty)
    except Exception as e:
        print("[WARN] initial Square hydrate failed:", repr(e))
    finally:
        profile["square_lookup_done"] = True


def build_initial_voicemail_sms(conv: dict, classification: dict, phone: str) -> str:
    """Build the very first outbound text after voicemail without relying on downstream reply routing."""
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    hydrate_square_profile_by_phone(profile, phone)

    # Non-service inquiries must never enter the residential booking opener.
    thread_type = detect_non_service_thread_type(
        conv,
        "",
        classification.get("category"),
        conv.get("cleaned_transcript") or ""
    )
    if thread_type == "vendor_sales_or_spam":
        clear_service_booking_state_for_non_service(conv, "vendor_sales_or_spam")
        sched["intro_sent"] = True
        sched["price_disclosed"] = False
        sched["booking_allowed"] = False
        return "Thanks for reaching out. Our office will review this if needed."
    if thread_type == "employment_inquiry":
        clear_service_booking_state_for_non_service(conv, "employment_inquiry")
        sched["intro_sent"] = True
        sched["price_disclosed"] = False
        return build_employment_inquiry_reply()
    if thread_type == "callback_requested":
        clear_service_booking_state_for_non_service(conv, "callback_requested")
        sched["intro_sent"] = True
        sched["price_disclosed"] = False
        return v13_apply_callback_request(conv, conv.get("cleaned_transcript") or "")
    if thread_type == "commercial_bid_contact":
        clear_service_booking_state_for_non_service(conv, "commercial_bid_contact")
        sched["intro_sent"] = True
        sched["price_disclosed"] = False
        return build_commercial_bid_reply(conv.get("cleaned_transcript") or "")

    first_name = (profile.get("active_first_name") or profile.get("recognized_first_name") or profile.get("voicemail_first_name") or "").strip()
    intro = f"Hi {first_name}, this is Prevolt Electric." if first_name else "Hi, this is Prevolt Electric."

    def starts_with_house_number(value: str) -> bool:
        value = (value or "").strip()
        return bool(re.match(r"^\d{1,6}(?:-\d{1,6})?\b", value))

    saved_full = None
    for a in profile.get("addresses") or []:
        a = (a or "").strip()
        if starts_with_house_number(a):
            saved_full = a
            break

    raw_hint = (sched.get("raw_address") or classification.get("detected_address") or "").strip()

    # Keep the first outbound human. Do not inject extracted intent here by default.
    # The task/scope is stored for the monitor and can be used if the customer asks,
    # but the first SMS should not read like a generated summary.
    if raw_hint and starts_with_house_number(raw_hint):
        address_line = f" We would be more than happy to come out and take a look at your project. You're at {raw_hint}, right?"
    elif saved_full:
        address_line = f" We would be more than happy to come out and take a look at your project. Is the work at {saved_full}?"
    else:
        address_line = " We would be more than happy to come out and take a look at your project. What’s the address for the work?"

    # First voicemail text should collect/confirm the address only.
    # Price is disclosed when moving into scheduling.
    sms = (intro + address_line).strip()
    sched["intro_sent"] = True
    sched["price_disclosed"] = False
    return re.sub(r"\s+", " ", sms).strip()

# ---------------------------------------------------
# Voice: Incoming Call (IVR + Spam Filter)
# ---------------------------------------------------
@app.route("/incoming-call", methods=["GET", "POST"])
def incoming_call():
    from twilio.twiml.voice_response import Gather, VoiceResponse

    response = VoiceResponse()
    try:
        log_event("CALL_IN", (request.form.get("From") or request.args.get("From") or "").replace("whatsapp:", "").strip(), {"call_sid": request.form.get("CallSid") or request.args.get("CallSid")})
    except Exception:
        pass

    gather = Gather(
        num_digits=1,
        action="/handle-call-selection",
        method="POST",
        timeout=6
    )

    gather.say(
        '<speak>'
            '<prosody rate="95%">'
                'Thanks for calling Prevolt Electric.<break time="0.7s"/>'
                'To help us direct your call, please choose an option.<break time="0.6s"/>'
                'If you are a residential customer, press 1.<break time="0.6s"/>'
                'If you are a commercial, government, or facility customer, press 2.'
            '</prosody>'
        '</speak>',
        voice="Polly.Matthew-Neural"
    )

    response.append(gather)

    response.say(
        '<speak><prosody rate="95%">Sorry, I did not get that. Let me repeat the options.</prosody></speak>',
        voice="Polly.Matthew-Neural"
    )
    response.redirect("/incoming-call")

    return Response(str(response), mimetype="text/xml")


# ---------------------------------------------------
# Handle Residential vs Commercial (FIXED: schema-hydrated profile/sched)
# ---------------------------------------------------
@app.route("/handle-call-selection", methods=["POST"])
def handle_call_selection():
    from twilio.twiml.voice_response import VoiceResponse

    digit = request.form.get("Digits", "")
    phone = request.form.get("From", "")
    response = VoiceResponse()

    try:
        log_event("CALL_MENU_SELECTION", (phone or "").replace("whatsapp:", "").strip(), {"digit": digit, "call_sid": request.form.get("CallSid")})
    except Exception:
        pass

    
    # ---------------------------------------------------
    # ✅ CRITICAL FIX:
    # conv.setdefault("profile", {}) leaves profile as {} forever,
    # which breaks /incoming-sms when it expects profile["addresses"].
    # So we "hydrate" required keys even if profile already exists.
    # ---------------------------------------------------
    # Establish conversation context (fix NameError: conv undefined)
    call_sid = request.form.get("CallSid", "") or ""
    convo_key = phone.strip() if phone.strip() else (f"call:{call_sid}" if call_sid else "call:unknown")
    conv = conversations.setdefault(convo_key, {})

    # Hydrate schema so later flows never KeyError
    profile = conv.setdefault("profile", {})
    profile.setdefault("name", None)
    profile.setdefault("addresses", [])
    profile.setdefault("upcoming_appointment", None)
    profile.setdefault("past_jobs", [])
    profile.setdefault("first_name", None)
    profile.setdefault("last_name", None)
    profile.setdefault("email", None)
    profile.setdefault("square_customer_id", None)
    profile.setdefault("square_lookup_done", False)

    sched = conv.setdefault("sched", {})
    sched.setdefault("pending_step", None)
    sched.setdefault("scheduled_date", None)
    sched.setdefault("scheduled_time", None)
    sched.setdefault("raw_address", None)
    sched.setdefault("address_verified", False)
    sched.setdefault("appointment_type", None)
    sched.setdefault("booking_created", False)
    sched.setdefault("name_engine_state", None)
    sched.setdefault("name_engine_prompted", False)
    sched.setdefault("name_engine_candidate_first", None)
    sched.setdefault("name_engine_candidate_last", None)
    sched.setdefault("name_engine_expected_known_first", None)
    sched.setdefault("name_engine_selected_first", None)
    sched.setdefault("soft_rejection_state", None)
    sched.setdefault("soft_rejection_open", False)
    sched.setdefault("soft_rejection_ts", None)
    profile.setdefault("name", None)
    profile.setdefault("addresses", [])
    profile.setdefault("upcoming_appointment", None)
    profile.setdefault("past_jobs", [])
    profile.setdefault("first_name", None)
    profile.setdefault("last_name", None)
    profile.setdefault("email", None)
    profile.setdefault("square_customer_id", None)
    profile.setdefault("square_lookup_done", False)
    profile.setdefault("first_name", None)
    profile.setdefault("last_name", None)
    profile.setdefault("email", None)
    profile.setdefault("square_customer_id", None)
    profile.setdefault("square_lookup_done", False)

    # Keep sched present (safe defaults; won't override existing values)
    sched = conv.setdefault("sched", {})
    sched.setdefault("pending_step", None)
    sched.setdefault("scheduled_date", None)
    sched.setdefault("scheduled_time", None)
    sched.setdefault("appointment_type", None)
    sched.setdefault("normalized_address", None)
    sched.setdefault("raw_address", None)
    sched.setdefault("booking_created", False)

    # Address assembly state keys (so later helpers are safe)
    sched.setdefault("address_candidate", None)
    sched.setdefault("address_verified", False)
    sched.setdefault("address_missing", None)
    sched.setdefault("address_parts", {})

    # Emergency flags (safe defaults)
    sched.setdefault("awaiting_emergency_confirm", False)
    sched.setdefault("emergency_approved", False)
    sched.setdefault("final_confirmation_sent", False)
    sched.setdefault("final_confirmation_accepted", False)
    sched.setdefault("last_final_confirmation_key", None)

    # -----------------------------
    # OPTION 1 → RESIDENTIAL
    # -----------------------------
    if digit == "1":
        profile["customer_type"] = "residential"

        # Production voice path: keep the IVR robot filter, then start live
        # voice only after option 1. If the voice runtime is not ready or the
        # env flag is off, fail closed to the existing voicemail/SMS flow.
        realtime_response = build_residential_realtime_twiml(phone, call_sid)
        return Response(str(realtime_response), mimetype="text/xml")

    # -----------------------------
    # OPTION 2 → COMMERCIAL / GOV
    # -----------------------------
    elif digit == "2":
        profile["customer_type"] = "commercial"

        response.say(
            '<speak><prosody rate="90%">Connecting you now.</prosody></speak>',
            voice="Polly.Matthew-Neural"
        )
        response.dial("+18609701727")  # replace with real number
        return Response(str(response), mimetype="text/xml")

    # -----------------------------
    # INVALID INPUT
    # -----------------------------
    else:
        response.say(
            '<speak><prosody rate="90%">Sorry, I didn’t understand that.</prosody></speak>',
            voice="Polly.Matthew-Neural"
        )
        response.redirect("/incoming-call")
        return Response(str(response), mimetype="text/xml")


# ---------------------------------------------------
# Step X — Voicemail Completion
# ---------------------------------------------------
@app.route("/voicemail-complete", methods=["POST"])
def voicemail_complete():
    from twilio.twiml.voice_response import VoiceResponse  # ✅ avoid NameError
    recording_url = request.form.get("RecordingUrl")
    from_number   = request.form.get("From", "").replace("whatsapp:", "")

    resp = VoiceResponse()

    if not recording_url:
        resp.say("We did not receive a recording. Goodbye.")
        return Response(str(resp), mimetype="text/xml")

    # 1) Transcribe
    try:
        transcript = transcribe_recording(recording_url)
        cleaned    = clean_transcript_text(transcript)
    except Exception as e:
        print("[ERROR] voicemail_complete transcription:", repr(e))
        cleaned = ""

    # If the voicemail is empty, abandoned, or a prompt-leak from the cleanup layer,
    # do not classify it and do not text the caller asking for an address.
    conv = conversations.setdefault(from_number, {})
    if looks_like_invalid_voicemail_transcript(cleaned):
        conv["cleaned_transcript"] = cleaned
        conv["thread_type"] = "manual_only"
        sched0 = conv.setdefault("sched", {})
        sched0["manual_only"] = True
        sched0["pending_step"] = None
        sched0["invalid_voicemail_transcript"] = True
        sched0["booking_allowed"] = False
        try:
            log_event("VOICEMAIL_INVALID_SUPPRESSED", from_number, {
                "recording_url": recording_url,
                "raw_transcript": _safe_monitor_text(transcript if 'transcript' in locals() else '', 700),
                "cleaned_transcript": _safe_monitor_text(cleaned, 700),
                "reason": "invalid_or_non_actionable_voicemail",
            }, conv)
        except Exception:
            pass
        resp.say("Thank you. Your message has been recorded.")
        resp.hangup()
        return Response(str(resp), mimetype="text/xml")

    # Command Center manual takeover also suppresses voicemail auto-follow-up.
    # We still transcribe and log the voicemail so the desk can review it, but
    # we do not classify, text, or reopen the automated booking flow.
    sched_manual_vm = conv.setdefault("sched", {})
    if conv.get("thread_type") == "manual_only" or sched_manual_vm.get("manual_only"):
        conv["thread_type"] = "manual_only"
        conv["cleaned_transcript"] = cleaned
        sched_manual_vm["manual_only"] = True
        sched_manual_vm["booking_allowed"] = False
        sched_manual_vm["pending_step"] = None
        sched_manual_vm["awaiting_slot_offer_choice"] = False
        sched_manual_vm["offered_slot_options"] = []
        sched_manual_vm.setdefault("manual_reason", "command_center_manual")
        try:
            log_event("VOICEMAIL_MANUAL_ONLY_SUPPRESSED", from_number, {
                "recording_url": recording_url,
                "raw_transcript": _safe_monitor_text(transcript if 'transcript' in locals() else '', 700),
                "cleaned_transcript": _safe_monitor_text(cleaned, 700),
                "reason": sched_manual_vm.get("manual_reason") or "command_center_manual",
            }, conv)
        except Exception:
            pass
        resp.say("Thank you. Your message has been recorded.")
        resp.hangup()
        return Response(str(resp), mimetype="text/xml")

    # 2) Classification
    classification = generate_initial_sms(cleaned)

    # 3) Save to memory
    # conv was created above so invalid voicemail suppression can log safely.

    # ---------------------------------------------------
    # HYDRATE conversation schema (prevents KeyError later)
    # ---------------------------------------------------
    profile = conv.setdefault("profile", {})
    profile.setdefault("name", None)
    profile.setdefault("addresses", [])
    profile.setdefault("upcoming_appointment", None)
    profile.setdefault("past_jobs", [])
    # Preserve any customer_type set earlier (residential/commercial)

    current_job = conv.setdefault("current_job", {})
    current_job.setdefault("job_type", None)
    current_job.setdefault("raw_description", None)

    sched = conv.setdefault("sched", {})
    sched.setdefault("pending_step", None)
    sched.setdefault("scheduled_date", None)
    sched.setdefault("scheduled_time", None)
    sched.setdefault("appointment_type", None)
    sched.setdefault("normalized_address", None)
    sched.setdefault("raw_address", None)
    sched.setdefault("booking_created", False)

    # Address assembly state
    sched.setdefault("address_candidate", None)
    sched.setdefault("address_verified", False)
    sched.setdefault("address_missing", None)
    sched.setdefault("address_parts", {})

    # Emergency flags
    sched.setdefault("awaiting_emergency_confirm", False)
    sched.setdefault("emergency_approved", False)

    # ---------------------------------------------------
    # Store classification results
    # ---------------------------------------------------
    conv["cleaned_transcript"] = cleaned
    conv["category"] = classification.get("category")
    conv["appointment_type"] = classification.get("appointment_type")
    conv["task_topics"] = classification.get("task_topics") or []
    conv.setdefault("initial_sms", cleaned)

    # Flow Lab V13: smoke detector install/replacement is normal evaluation work,
    # not emergency dispatch unless the caller describes actual smoke/fire/burning.
    if v13_looks_like_smoke_detector_install_request(cleaned):
        classification["intent"] = "quote"
        classification["appointment_type"] = "EVAL_195"
        conv["appointment_type"] = "EVAL_195"
        sched["appointment_type"] = "EVAL_195"
        sched["awaiting_emergency_confirm"] = False
        sched["emergency_approved"] = False

    # Hard pre-routing lock before any address/date scheduling state is saved.
    thread_type = detect_non_service_thread_type(conv, "", classification.get("category"), cleaned)
    if thread_type in {"employment_inquiry", "commercial_bid_contact", "callback_requested"}:
        clear_service_booking_state_for_non_service(conv, thread_type)
    if classification.get("detected_first_name") and not profile.get("voicemail_first_name"):
        profile["voicemail_first_name"] = classification.get("detected_first_name")
    if classification.get("detected_last_name") and not profile.get("voicemail_last_name"):
        profile["voicemail_last_name"] = classification.get("detected_last_name")

    if classification.get("detected_date"):
        sched["scheduled_date"] = classification["detected_date"]
    if classification.get("detected_time"):
        sched["scheduled_time"] = classification["detected_time"]
    if classification.get("detected_address") and conv.get("thread_type") not in {"employment_inquiry", "commercial_bid_contact", "manual_only"}:
        sched["raw_address"] = classification["detected_address"]

    # Refresh address assembly state (safe if normalized/raw changed)
    update_address_assembly_state(sched)

    # --------------------------------------
    # HARD OVERRIDE: Voicemail Emergency Pre-Flag
    # --------------------------------------
    if classification.get("intent") == "emergency" and not v13_looks_like_smoke_detector_install_request(cleaned) and conv.get("thread_type") not in {"employment_inquiry", "commercial_bid_contact", "manual_only", "callback_requested"}:
        sched["appointment_type"] = "TROUBLESHOOT_395"
        sched["awaiting_emergency_confirm"] = True
        sched["emergency_approved"] = False

    try:
        log_event("VOICEMAIL_CAPTURED", from_number, {
            "category": classification.get("category"),
            "appointment_type": classification.get("appointment_type"),
            "intent": classification.get("intent"),
            "task_topics": classification.get("task_topics") or [],
            "transcript": _safe_monitor_text(cleaned, 500),
            "recording_url": recording_url,
        }, conv)
    except Exception:
        pass

    # 4) Trigger First SMS using deterministic voicemail opener so task topics and known addresses land reliably
    try:
        initial_msg = build_initial_voicemail_sms(conv, classification, from_number)
        send_sms(from_number, initial_msg)
    except Exception as e:
        print("[ERROR] voicemail_complete → initial sms:", repr(e))
        try:
            outbound = generate_reply_for_inbound(
                cleaned,
                conv.get("category"),
                conv.get("appointment_type"),
                conv.get("initial_sms"),
                "",
                sched.get("scheduled_date"),
                sched.get("scheduled_time"),
                sched.get("raw_address")
            )
            initial_msg = outbound.get("sms_body") or "Thanks for your voicemail — how can we help?"
            send_sms(from_number, initial_msg)
        except Exception as e2:
            print("[ERROR] voicemail_complete fallback → Step4:", repr(e2))

    resp.say("Thank you. Your message has been recorded.")
    resp.hangup()
    return Response(str(resp), mimetype="text/xml")




# ---------------------------------------------------
# SMS Sanitizer — prevents "one moment" / "hold on" lies
# ---------------------------------------------------
def sanitize_sms_body(s: str, *, booking_created: bool) -> str:
    """
    If we are NOT immediately sending a booking confirmation in this same reply,
    we must NOT say 'one moment', 'securing', 'processing', etc.
    """
    if not s:
        return s

    low = s.lower()

    banned_phrases = [
        "one moment",
        "one sec",
        "one second",
        "hold on",
        "hang tight",
        "please wait",
        "i am finishing",
        "i'm finishing",
        "securing your appointment",
        "processing",
        "working on it",
        "i'll book that now",
        "i will book that now",
        "let me book",
        "let me secure",
        "i'll reserve",
        "i will reserve",
        "reserve that slot",
        "reserved",
        "works great",
        "looking forward to confirming",
        "confirming your appointment",
    ]

    if any(p in low for p in banned_phrases) and not booking_created:
        return "Thanks — got it."

    # Safety net: if the model ever stacks the exact same sentence twice,
    # collapse it before sending so the reply does not look automated.
    try:
        parts = re.split(r"(?<=[.!?])\s+", s.strip())
        cleaned_parts = []
        last_norm = ""
        for part in parts:
            norm = re.sub(r"\s+", " ", part).strip().lower()
            if norm and norm != last_norm:
                cleaned_parts.append(part.strip())
            last_norm = norm
        s = " ".join(cleaned_parts).strip() or s
    except Exception:
        pass

    return s


# ---------------------------------------------------
# Controlled customer-facing price disclosure helpers
# ---------------------------------------------------
def prevolt_eval_price_line(appt_type: str) -> str:
    appt = (appt_type or "").upper()
    if "TROUBLESHOOT" in appt:
        return "Troubleshoot and repair visits are $395."
    if "INSPECTION" in appt:
        return "The whole-home inspection is $395."
    return "We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step."


def format_command_center_slot_list(slots: list[dict]) -> str:
    labels = []
    for slot in (slots or [])[:3]:
        label = (slot.get("label") or _humanize_slot_label(slot.get("date"), slot.get("time")) or "").strip()
        if label:
            labels.append(label)
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def build_price_and_availability_prompt(conv: dict, appt_type: str = "EVAL_195") -> str:
    """Human price disclosure plus concrete slot choices.

    Used only when the customer is moving into scheduling. The price is not
    allowed in the first address-collection message.
    """
    sched = (conv or {}).setdefault("sched", {}) if isinstance(conv, dict) else {}
    appt = (appt_type or sched.get("appointment_type") or "EVAL_195").upper()
    # Once we are disclosing the evaluation price and offering slots, the
    # appointment type is no longer optional. Keep both sched and top-level
    # mirrors in sync so later slot replies do not fall back to need_appt_type.
    if not sched.get("appointment_type"):
        sched["appointment_type"] = appt
    if isinstance(conv, dict):
        conv["appointment_type"] = sched.get("appointment_type") or appt
    sched["booking_allowed"] = True
    try:
        slots = get_next_available_slots(appt, limit=3)
    except Exception as e:
        print("[WARN] build_price_and_availability_prompt slot lookup failed:", repr(e))
        slots = []

    # Keep this wording deliberately human and contained. This is the main
    # customer-facing price + scheduling handoff after the address is accepted.
    if "TROUBLESHOOT" in appt:
        intro = "We have availability to come and take a look. Troubleshoot and repair visits are $395."
    elif "INSPECTION" in appt:
        intro = "We have availability to come and take a look. The whole-home inspection is $395."
    else:
        intro = "We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step."

    if slots:
        sched["awaiting_slot_offer_choice"] = True
        sched["offered_slot_options"] = slots[:3]
        opts = format_command_center_slot_list(slots)
        return f"{intro} If that is something you're interested in, we have {opts}. Or is there a better date/time you're available?"

    return f"{intro} If that is something you're interested in, what date/time are you available?"


def replace_generic_schedule_question_with_availability(body: str, availability_prompt: str) -> str:
    body = " ".join(str(body or "").split()).strip()
    prompt = " ".join(str(availability_prompt or "").split()).strip()
    if not body or not prompt:
        return body or prompt
    low = _intent_text(body)
    generic = [
        "what day and time work best for you",
        "what day and time works best for you",
        "what day works best for you",
        "what day works best",
        "what time works best",
        "what time works for you",
    ]
    if any(g in low for g in generic):
        # Keep short human acknowledgement if present.
        m = re.match(r"^(Got it\.|Perfect\.|Sounds good\.|Alright\.|No problem\.)(?:\s+)?", body, flags=re.I)
        if m:
            return f"{m.group(1).strip()} {prompt}"
        return prompt
    return f"{prompt} {body}".strip()


def message_is_scheduling_prompt(body: str) -> bool:
    low = _intent_text(body)
    if not low:
        return False
    schedule_markers = [
        "what day", "what time", "which one works best", "which time works best",
        "what weekday", "i have monday", "i have tuesday", "i have wednesday",
        "i have thursday", "i have friday", "i have saturday", "i have sunday",
        "at 9 00", "at 10 00", "at 11 00", "at 12 00", "at 1 00", "at 2 00", "at 3 00", "at 4 00",
    ]
    return any(m in low for m in schedule_markers)


def customer_is_asking_price(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return any(p in low for p in [
        "how much", "price", "cost", "$195", "$395", "195", "395",
        "just to come out", "service fee", "trip fee", "diagnostic fee",
        "what do you charge", "what does the evaluation include", "what does that include",
        "what does that cover", "what does this cover", "what does the 195 cover", "what does $195 cover", "what is covered",
        "go toward", "go towards", "goes toward", "goes towards", "applied to the project",
        "credit toward", "credited toward", "deposit", "free estimate", "ballpark", "rough price",
    ])


def strip_unapproved_price_language(body: str) -> str:
    """Remove evaluation pricing if the LLM leaks it before the controlled price gate."""
    s = str(body or "")
    # Remove full sentences containing the standard evaluation fee.
    s = re.sub(r"(?:^|(?<=[.!?])\s+)[^.!?]*\$195[^.!?]*(?:[.!?]|$)", " ", s, flags=re.I)
    s = re.sub(r"(?:^|(?<=[.!?])\s+)[^.!?]*evaluation visit[^.!?]*(?:[.!?]|$)", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s or body


# ---------------------------------------------------
# build_system_prompt (RESTORED + PATCHED)
# ---------------------------------------------------
def build_system_prompt(
    cleaned_transcript,
    category,
    appointment_type,
    initial_sms,
    scheduled_date,
    scheduled_time,
    address,
    today_date_str,
    today_weekday,
    conv
):
    """
    Reconstructed stable prompt used by Step 4.
    Tailored to the B-3 state-machine logic.
    """
    rules_blob = RULE_MATRIX_TEXT or ""
    return f"""
You are Prevolt OS — a deterministic scheduling engine.
You ALWAYS return strict JSON with fields:
  sms_body, scheduled_date, scheduled_time, address.

NEVER forget known values. NEVER reset fields unless the user changes them.
NEVER invent policy. Follow the existing SRB matrix below when it applies.
Python is the execution engine. The SRB matrix is the policy source.

Known transcript: {cleaned_transcript}
Category: {category}
Initial appointment type: {appointment_type}
Initial SMS: {initial_sms}

Current stored values:
  scheduled_date = {scheduled_date}
  scheduled_time = {scheduled_time}
  address = {address}

Today's date: {today_date_str} ({today_weekday})

Core behavioral constraints:
- Non-service threads override scheduling. Employment inquiries, job applicants, commercial bid/proposal conversations, and platform wrapper notices must NOT receive address/date/time collection or evaluation pricing.
- NEVER ask questions already answered.
- If only one field is missing, ask ONLY for that field.
- Do not treat vague time phrases as explicit times.
- Do not confirm a booking unless a real Square booking exists.
- Use simple, direct language.
- Do NOT mention $195 or $395 unless the customer directly asks about price. Python will add required price disclosure at the right scheduling step.
- Do NOT say "for the visit" when asking for an address. Say "for the work".
- Do NOT mention Kyle by name in customer-facing replies. Use "our electrician", "one of our licensed electricians", "our technician", "our office", or "we".
- Do NOT insert a robotic project-intent summary like "It sounds like you are looking for help with..." unless the customer directly asks what we are coming out for.
- NEVER say "one moment", "please wait", "hold on", "securing your appointment", or anything implying background processing.
- If the address is incomplete, ask only for the missing address atom.
- If date is known but time is missing, ask only for time.
- If time is known but date is missing, ask only for date.
- Never overwrite stored name, email, date, time, or address with blank values.

Existing SRB matrix:
{rules_blob}
"""


# ---------------------------------------------------
# Address Assembly State Helper (Step 2) — FIXED (requires house number)
# ---------------------------------------------------
def parse_complete_raw_address(raw: str) -> dict | None:
    """
    Parse a fully spoken/typed residential address directly from raw text.

    Accepts examples like:
      - 97 Maybeth Street, Springfield, MA 01119
      - 97 Maybeth Street Springfield MA 01119
      - 2 Main St, Springfield, Massachusetts 01103
      - 97 Maybeth Street, Springfield, Mass. 01119

    Returns a normalized-address-shaped dict or None.
    """
    import re

    s = " ".join((raw or "").strip().replace("\n", " ").split()).strip(" ,")
    if not s:
        return None

    # Accept leading unit suffixes like "72B Dakota Lane" by normalizing to
    # "72 Dakota Lane, Unit B" before parsing/normalizing.
    m_unit = re.match(r"^(?P<num>\d{1,6})(?P<unit>[A-Za-z])\s+(?P<rest>.+)$", s)
    if m_unit and not re.search(r"\b(?:apt|apartment|unit|suite|ste|#)\b", s, flags=re.I):
        s = f"{m_unit.group('num')} {m_unit.group('rest')}, Unit {m_unit.group('unit').upper()}"

    state_token = r"CT|C\.?T\.?|Connecticut|Conn\.?|MA|M\.?A\.?|Massachusetts|Mass\.?"
    patterns = [
        rf"^(?P<line1>\d{{1,6}}(?:-\d{{1,6}})?[A-Za-z]?\s+.+?),\s*(?P<city>[A-Za-z .'\-]+?),\s*(?P<state>{state_token})\s+(?P<zip>\d{{5}}(?:-\d{{4}})?)$",
        rf"^(?P<line1>\d{{1,6}}(?:-\d{{1,6}})?[A-Za-z]?\s+.+?)\s+(?P<city>[A-Za-z .'\-]+?)\s+(?P<state>{state_token})\s+(?P<zip>\d{{5}}(?:-\d{{4}})?)$",
    ]

    def normalize_state_token(state_raw: str) -> str | None:
        token = re.sub(r"[^A-Za-z]", "", state_raw or "").upper()
        if token in {"CT", "CONNECTICUT", "CONN"}:
            return "CT"
        if token in {"MA", "MASS", "MASSACHUSETTS"}:
            return "MA"
        return None

    for pat in patterns:
        m = re.match(pat, s, flags=re.I)
        if not m:
            continue

        line1 = " ".join((m.group("line1") or "").split()).strip(" ,")
        city = " ".join((m.group("city") or "").split()).strip(" ,")
        state_raw = (m.group("state") or "").strip()
        zipc = (m.group("zip") or "").strip()

        if not re.match(r"^\d{1,6}(?:-\d{1,6})?[A-Za-z]?\b", line1):
            continue
        if not line1 or not city or not state_raw or not zipc:
            continue

        state_up = normalize_state_token(state_raw)
        if state_up not in {"CT", "MA"}:
            continue

        city = " ".join(w.capitalize() for w in city.split())

        return {
            "address_line_1": line1,
            "locality": city,
            "administrative_district_level_1": state_up,
            "postal_code": zipc,
            "country": "US",
        }

    return None

def update_address_assembly_state(sched: dict) -> None:
    """
    Derives address state atoms from raw_address / normalized_address.

    FIX:
      - Normalized addresses are ONLY considered "verified" if address_line_1
        includes a leading street number (ex: "24 Dickerman Ave").
      - Route-only normalization (ex: "Dickerman Ave, Windsor Locks CT 06096")
        is NOT sufficient to book.
    """
    import re

    # Defaults (never assume keys exist)
    sched.setdefault("address_candidate", None)
    sched.setdefault("address_verified", False)
    sched.setdefault("address_missing", None)
    sched.setdefault("address_parts", {})

    raw = (sched.get("raw_address") or "").strip()
    try:
        cleaned_raw = scrub_non_address_tail(raw)
        if cleaned_raw and cleaned_raw != raw:
            raw = cleaned_raw
            sched["raw_address"] = raw
    except Exception:
        pass
    # Normalize leading unit suffixes like 72B Dakota Lane -> 72 Dakota Lane, Unit B.
    try:
        m_unit = re.match(r"^(?P<num>\d{1,6})(?P<unit>[A-Za-z])\s+(?P<rest>.+)$", raw)
        if m_unit and not re.search(r"\b(?:apt|apartment|unit|suite|ste|#)\b", raw, flags=re.I):
            raw = f"{m_unit.group('num')} {m_unit.group('rest')}, Unit {m_unit.group('unit').upper()}"
            sched["raw_address"] = raw
    except Exception:
        pass
    norm = sched.get("normalized_address")

    # Helper: does a line start with a street number?
    def line_has_number(line: str) -> bool:
        line = (line or "").strip()
        return bool(re.match(r"^\d{1,6}(?:-\d{1,6})?[A-Za-z]?\b", line))

    # -----------------------------
    # 1) If we have normalized_address, validate it properly
    # -----------------------------
    if isinstance(norm, dict):
        line1 = (norm.get("address_line_1") or "").strip()
        city  = (norm.get("locality") or "").strip()
        state = (norm.get("administrative_district_level_1") or "").strip()
        zipc  = (norm.get("postal_code") or "").strip()

        has_core_fields = bool(line1 and city and state and zipc)

        # ✅ FIX: require street number in line1
        has_number = line_has_number(line1)

        if has_core_fields and has_number:
            sched["address_verified"] = True
            sched["address_missing"] = None
            sched["address_candidate"] = raw or sched.get("address_candidate")
            sched["address_parts"] = {
                "street": True,
                "number": True,
                "city": True,
                "state": True,
                "zip": True,
                "source": "normalized_address"
            }
            return

        # If we have route-level normalization but no house number:
        if has_core_fields and not has_number:
            sched["address_verified"] = False
            sched["address_missing"] = "number"
            sched["address_candidate"] = raw or line1
            sched["address_parts"] = {
                "street": True,
                "number": False,
                "city": True,
                "state": True,
                "zip": True,
                "source": "normalized_address_missing_number"
            }
            return

    # -----------------------------
    # 2) Full raw address shortcut
    # If voicemail/SMS already gave us a complete address string,
    # trust that structure immediately instead of downgrading to
    # "confirm" and asking for town again.
    # -----------------------------
    parsed_raw = parse_complete_raw_address(raw)
    if parsed_raw:
        sched["normalized_address"] = parsed_raw
        sched["address_verified"] = True
        sched["address_missing"] = None
        sched["address_candidate"] = raw or parsed_raw.get("address_line_1")
        sched["address_parts"] = {
            "street": True,
            "number": True,
            "city": True,
            "state": True,
            "zip": True,
            "source": "raw_full_address"
        }
        return

    # -----------------------------
    # 2) No normalized verified address → evaluate raw candidate quality
    # -----------------------------
    sched["address_verified"] = False
    if raw:
        sched["address_candidate"] = raw

    # If nothing at all
    if not raw:
        sched["address_missing"] = "street"
        sched["address_parts"] = {"street": False, "number": False, "city": False, "state": False, "zip": False, "source": "none"}
        return

    low = raw.lower()

    # Heuristic: detect street words
    street_suffixes = (
        " st", " street", " ave", " avenue", " rd", " road", " ln", " lane",
        " dr", " drive", " ct", " court", " cir", " circle", " blvd", " boulevard",
        " way", " pkwy", " parkway", " ter", " terrace", " park"
    )
    has_street_word = any(suf in low for suf in street_suffixes)

    # Has a house number at the start?
    starts_with_number = bool(re.match(r"^\s*\d{1,6}(?:-\d{1,6})?[A-Za-z]?\b", raw))

    # Has explicit CT/MA?
    has_state = (" ct" in f" {low} ") or (" connecticut" in low) or (" ma" in f" {low} ") or (" massachusetts" in low)

    # Street, no number: "Dickerman Ave"
    if has_street_word and not starts_with_number:
        sched["address_missing"] = "number"
        sched["address_parts"] = {"street": True, "number": False, "city": False, "state": has_state, "zip": False, "source": "raw_address"}
        return

    # Number + street: "24 Main St" (needs town confirm possibly)
    if has_street_word and starts_with_number:
        sched["address_missing"] = "confirm"
        sched["address_parts"] = {"street": True, "number": True, "city": False, "state": has_state, "zip": False, "source": "raw_address"}
        return

    # Town-only like "Windsor Locks"
    sched["address_missing"] = "street"
    sched["address_parts"] = {"street": False, "number": False, "city": True, "state": has_state, "zip": False, "source": "raw_address"}

import hashlib

def _stable_choice_key(sched: dict, label: str) -> str:
    """
    Generates a stable key for deterministic phrasing selection.
    Prefer a per-conversation identifier if you have one.
    """
    # Use whatever you already store; these are safe fallbacks.
    cid = (
        str(sched.get("conversation_id") or "")
        or str(sched.get("from_number") or "")
        or str(sched.get("contact") or "")
        or str(sched.get("raw_address") or "")
    ).strip()

    base = f"{cid}|{label}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def pick_variant_once(sched: dict, label: str, options: list[str]) -> str:
    """
    Deterministically picks one option and stores it in sched so it never changes
    mid-thread, even if the function runs multiple times.
    """
    if not options:
        return ""

    store = sched.setdefault("prompt_variants", {})
    if label in store and store[label] in options:
        return store[label]

    h = _stable_choice_key(sched, label)
    idx = int(h[:8], 16) % len(options)
    chosen = options[idx]
    store[label] = chosen
    return chosen


# ---------------------------------------------------
# Address Prompt Builder (Step 4) — HUMAN + DETERMINISTIC VARIATION
# ---------------------------------------------------
def build_address_prompt(sched: dict) -> str:
    """
    Human-sounding prompts that ask for the missing address atom.
    Deterministic phrasing variation (no AI tells, no em dashes).
    """
    update_address_assembly_state(sched)

    missing = (sched.get("address_missing") or "").strip().lower()
    candidate = (sched.get("address_candidate") or sched.get("raw_address") or "").strip()
    parts = sched.get("address_parts") or {}

    norm = sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None
    norm_line1 = (norm.get("address_line_1") or "").strip() if norm else ""
    norm_city  = (norm.get("locality") or "").strip() if norm else ""
    norm_state = (norm.get("administrative_district_level_1") or "").strip() if norm else ""
    norm_zip   = (norm.get("postal_code") or "").strip() if norm else ""

    # STREET
    if missing == "street":
        if parts.get("city") and candidate and not parts.get("street"):
            options = [
                f"What is the house number and street name in {candidate}?",
                f"What’s the house number and street for the place in {candidate}?",
                f"Send the house number and street name in {candidate}.",
            ]
            return pick_variant_once(sched, "addr_missing_street_with_city", options)

        options = [
            "What is the house number and street name?",
            "Send the house number and street name for the work.",
            "What’s the house number and street for the work?",
        ]
        return pick_variant_once(sched, "addr_missing_street", options)

    # NUMBER
    if missing == "number":
        if norm_line1 and norm_city and norm_state:
            tail = f"{norm_city}, {norm_state}"
            if norm_zip:
                tail += f" {norm_zip}"

            options = [
                f"What’s the house number on {norm_line1} in {tail}?",
                f"What number is the place on {norm_line1} in {tail}?",
                f"What’s the street number on {norm_line1} in {tail}?",
            ]
            return pick_variant_once(sched, "addr_missing_number_with_norm", options)

        if candidate:
            options = [
                f"What’s the house number on {candidate}?",
                f"What number is it on {candidate}?",
                f"What’s the street number on {candidate}?",
            ]
            return pick_variant_once(sched, "addr_missing_number_with_candidate", options)

        options = [
            "What’s the house number?",
            "What number is it?",
            "What’s the street number?",
        ]
        return pick_variant_once(sched, "addr_missing_number", options)

    # CONFIRM TOWN/STATE
    if missing == "confirm":
        need_state = not bool(parts.get("state"))
        if need_state:
            options = [
                "What town is it in, Connecticut or Massachusetts?",
                "Which town is it in, and is that Connecticut or Massachusetts?",
                "What town is this in, CT or MA?",
            ]
            return pick_variant_once(sched, "addr_missing_confirm_need_state", options)

        options = [
            "What town is it in?",
            "Which town is this in?",
            "What town is that in?",
        ]
        return pick_variant_once(sched, "addr_missing_confirm", options)

    # FALLBACK
    options = [
        "What’s the address for the work?",
        "What address should we use?",
        "Send the address for the work.",
    ]
    return pick_variant_once(sched, "addr_fallback", options)



# ---------------------------------------------------
# Step 5B — Early Google Maps normalization for partial addresses
# NOTE: Works with normalize_address(raw) returning either:
#   - (status, dict)  OR
#   - dict (assumed ok)
# ---------------------------------------------------
def try_early_address_normalize(sched: dict) -> None:
    """
    Tries Google Maps normalization as soon as we have a plausible street.
    Safe rules:
      - Never clears data
      - Only sets normalized_address if normalize_address() returns a dict-shaped address
      - Recomputes address assembly state afterward
    """
    update_address_assembly_state(sched)

    if sched.get("address_verified"):
        return

    raw = (sched.get("raw_address") or "").strip()
    if not raw:
        return

    missing = (sched.get("address_missing") or "").strip().lower()

    # If we only have town/city (missing street), do not normalize yet.
    if missing == "street":
        return

    # Only normalize if it looks like a street input (avoid spamming maps on random text)
    low = raw.lower()
    street_suffixes = (" st", " street", " ave", " avenue", " rd", " road", " ln", " lane", " dr", " drive", " blvd", " boulevard", " way", " ct", " court", " cir", " circle", " ter", " terrace", " park", " pkwy", " parkway")
    if not any(suf in low for suf in street_suffixes):
        return

    try:
        result = normalize_address(raw)
    except Exception as e:
        print("[WARN] try_early_address_normalize normalize_address failed:", repr(e))
        return

    status = None
    addr_struct = None

    # Support both signatures
    if isinstance(result, tuple) and len(result) >= 2:
        status, addr_struct = result[0], result[1]
    elif isinstance(result, dict):
        status, addr_struct = "ok", result
    else:
        return

    if status == "ok" and isinstance(addr_struct, dict):
        sched["normalized_address"] = addr_struct
        update_address_assembly_state(sched)






# ---------------------------------------------------
# Address Tail Contamination Guard
# Prevents customer questions like "Are you licensed?" from becoming city text.
# ---------------------------------------------------
ADDRESS_SUFFIX_RE_SAFE = r"(?:st|street|ave|avenue|rd|road|ln|lane|dr|drive|court|ct|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace|pl|place|park)"
QUESTION_START_RE_SAFE = re.compile(
    r"^\s*(?:are|is|do|does|did|can|could|would|will|why|what|when|where|who|how)\b",
    re.I,
)
NON_ADDRESS_WORD_RE_SAFE = re.compile(
    r"\b(?:licensed|license|insured|insurance|permit|employment|job|hiring|resume|"
    r"coming|still\s+coming|appointment|schedule|email|phone|quote|estimate|service|"
    r"work|visit|price|cost|charge|thank|thanks|hello|hi|ai|bot|real\s+person)\b",
    re.I,
)
DAY_TIME_WORD_RE_SAFE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|"
    r"morning|afternoon|evening|noon|tonight|asap|soonest|earliest|whenever|anytime|any\s+time)\b",
    re.I,
)


def looks_like_non_address_fragment(text: str) -> bool:
    value = " ".join(str(text or "").strip().split())
    if not value:
        return True
    low = value.lower()
    compact = re.sub(r"[^a-z0-9']+", " ", low).strip()
    if compact in {"yes", "y", "yeah", "yep", "correct", "right", "ok", "okay", "sure", "no", "nope", "nah"}:
        return True
    if "?" in value:
        return True
    if QUESTION_START_RE_SAFE.search(value):
        return True
    if NON_ADDRESS_WORD_RE_SAFE.search(low):
        return True
    if DAY_TIME_WORD_RE_SAFE.search(low):
        return True
    if "@" in value:
        return True
    if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", value):
        return True
    return False


def _is_state_token_safe(text: str) -> bool:
    token = re.sub(r"[^A-Za-z]", "", str(text or "")).upper()
    return token in {"CT", "CONNECTICUT", "CONN", "MA", "MASS", "MASSACHUSETTS"}


def _looks_like_city_name_fragment_safe(text: str) -> bool:
    value = " ".join(str(text or "").strip(" ,.;").split())
    if not value:
        return False
    if looks_like_non_address_fragment(value):
        return False
    if re.search(r"\d", value):
        return False
    if re.search(rf"\b{ADDRESS_SUFFIX_RE_SAFE}\b", value, flags=re.I):
        return False
    if len(value.split()) > 4:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", value):
        return False
    return True


def _city_state_zip_fragment_ok_safe(fragment: str) -> bool:
    value = " ".join(str(fragment or "").strip(" ,.;").split())
    if not value:
        return False
    value = re.sub(r"\b\d{5}(?:-\d{4})?\b", "", value).strip(" ,.;")
    value = re.sub(
        r"\b(?:CT|C\.?T\.?|Connecticut|Conn\.?|MA|M\.?A\.?|Mass\.?|Massachusetts)\b",
        "",
        value,
        flags=re.I,
    ).strip(" ,.;")
    if not value:
        return True
    return _looks_like_city_name_fragment_safe(value)


def scrub_non_address_tail(address_text: str) -> str:
    """
    Keeps normal street/city/state/zip tails and drops customer prose/questions.
    Example: '34 Dickerman Ave, Are You Licensed' -> '34 Dickerman Ave'.
    """
    value = " ".join(str(address_text or "").replace("\n", " ").split()).strip(" ,.;")
    if not value:
        return ""

    m = re.match(
        rf"^(?P<line>\d{{1,6}}(?:-\d{{1,6}})?[A-Za-z]?\s+[A-Za-z0-9.'#\- ]+?\b{ADDRESS_SUFFIX_RE_SAFE}\b)(?P<tail>.*)$",
        value,
        flags=re.I,
    )
    if not m:
        return value

    line = m.group("line").strip(" ,.;")
    tail = (m.group("tail") or "").strip(" ,.;")
    if not tail:
        return line

    tail = re.split(
        r"\b(?:are\s+you|do\s+you|can\s+you|could\s+you|would\s+you|will\s+you|what|when|where|why|how|"
        r"right|correct|please|thank\s+you|thanks|is\s+this|is\s+that)\b",
        tail,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" ,.;")

    if not tail or looks_like_non_address_fragment(tail):
        return line

    parts = [p.strip(" ,.;") for p in tail.split(",") if p.strip(" ,.;")]
    if not parts and tail:
        parts = [tail]

    safe_parts = []
    for part in parts:
        if _is_state_token_safe(part):
            token = re.sub(r"[^A-Za-z]", "", part).upper()
            safe_parts.append("MA" if token.startswith("MA") or token == "MASSACHUSETTS" else "CT")
            continue
        if re.fullmatch(r"\d{5}(?:-\d{4})?", part):
            safe_parts.append(part)
            continue
        if _city_state_zip_fragment_ok_safe(part):
            safe_parts.append(part)
            continue
        break

    return ", ".join([line] + safe_parts).strip(" ,.;") if safe_parts else line


def set_raw_address_safe(sched: dict, value: str) -> None:
    cleaned = scrub_non_address_tail(value)
    if cleaned:
        sched["raw_address"] = cleaned

def extract_city_state_from_reply(text: str) -> tuple[str | None, str | None]:
    """
    Parse compact city/state replies like 'Windsor CT' or 'Windsor Locks'.
    Customer questions/prose such as 'Are you licensed?' must never become city text.
    """
    txt = " ".join((text or "").strip().replace(",", " ").split())
    if not txt:
        return None, None

    if looks_like_non_address_fragment(txt):
        return None, None

    low = txt.lower()
    state = None
    if re.search(r"\bct\b|\bconnecticut\b", low):
        state = "CT"
        txt = re.sub(r"\bct\b|\bconnecticut\b", "", txt, flags=re.I).strip(" ,")
    elif re.search(r"\bma\b|\bmass\b|\bmassachusetts\b", low):
        state = "MA"
        txt = re.sub(r"\bma\b|\bmass\b|\bmassachusetts\b", "", txt, flags=re.I).strip(" ,")

    if not txt:
        return None, state
    if not _looks_like_city_name_fragment_safe(txt):
        return None, state

    city = " ".join(w.capitalize() for w in txt.split())
    return city or None, state

def apply_partial_address_reply(sched: dict, inbound_text: str) -> bool:
    """
    Merge compact follow-up address replies into the existing street address.
    Examples:
      raw_address='54 Bloomfield Ave' + inbound='Windsor CT'
      raw_address='54 Bloomfield Ave' + inbound='Windsor'
      raw_address='54 Bloomfield Ave, Windsor' + inbound='CT'
    Returns True when sched was updated.
    """
    inbound = (inbound_text or "").strip()
    if not inbound:
        return False
    if looks_like_non_address_fragment(inbound):
        return False

    # Live guard: acknowledgements like "Yes" are address confirmations,
    # not city/state fragments. Never append them to raw_address.
    low_ack = re.sub(r"[^a-z0-9' ]+", "", inbound.lower()).strip()
    if low_ack in {"yes", "y", "yeah", "yep", "correct", "right", "that is correct", "thats right", "that's right", "ok", "okay", "sure", "no", "nope"}:
        return False

    update_address_assembly_state(sched)
    missing = (sched.get("address_missing") or "").strip().lower()
    if missing not in {"confirm", "state"}:
        return False

    raw = (sched.get("raw_address") or "").strip()
    if not raw:
        return False

    city, state = extract_city_state_from_reply(inbound)

    pieces = [p.strip() for p in raw.split(",") if p.strip()]
    last_piece = pieces[-1] if pieces else ""
    second_last_piece = pieces[-2] if len(pieces) >= 2 else ""

    # State-only reply like 'CT'
    if not city and state:
        if re.fullmatch(r"CT|MA", last_piece, flags=re.I):
            merged = ", ".join(pieces[:-1] + [state])
        else:
            merged = raw + f", {state}"
    # City + optional state reply
    elif city:
        # If the raw address already ends with the same city, keep it as-is.
        if last_piece.lower() == city.lower() or second_last_piece.lower() == city.lower():
            merged = raw
            if state:
                if re.fullmatch(r"CT|MA", last_piece, flags=re.I):
                    merged = ", ".join(pieces[:-1] + [state])
                elif not re.fullmatch(r"CT|MA", last_piece, flags=re.I):
                    merged = raw if last_piece.lower() == city.lower() else raw + f", {state}"
        else:
            merged = f"{raw}, {city}" + (f", {state}" if state else "")
    else:
        return False

    set_raw_address_safe(sched, re.sub(r"\s+,", ",", merged).strip(" ,"))

    # v58: Do not let Google invent the missing town/state from a partial address.
    # Only normalize after the caller supplied a real town/state or a known local town.
    try:
        allowed, safe_raw, forced_state, town, why = _v58_prepare_address_for_maps(sched["raw_address"])
    except Exception:
        allowed, safe_raw, forced_state, town, why = False, None, None, None, "v58_guard_error"

    if allowed and safe_raw:
        set_raw_address_safe(sched, safe_raw)
        try:
            result = normalize_address(safe_raw, forced_state=forced_state)
            if isinstance(result, tuple) and len(result) >= 2:
                status, addr_struct = result[0], result[1]
                if status == "ok" and isinstance(addr_struct, dict) and _v58_locality_ok(addr_struct, town, forced_state):
                    sched["normalized_address"] = addr_struct
            elif isinstance(result, dict) and _v58_locality_ok(result, town, forced_state):
                sched["normalized_address"] = result
        except Exception as e:
            print("[WARN] apply_partial_address_reply normalize failed:", repr(e))
    else:
        sched["address_verified"] = False
        sched["address_missing"] = "state" if city and not state else "confirm"

    update_address_assembly_state(sched)
    return True

def choose_next_prompt_from_state(conv: dict, inbound_text: str = "") -> str:
    """Single deterministic next-step selector. Python enforces state; SRBs drive prompt choice."""
    import re

    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    update_address_assembly_state(sched)
    recompute_pending_step(profile, sched)

    inbound_low = (inbound_text or "").strip().lower()
    appt = (sched.get("appointment_type") or "").upper()
    is_emergency = ("TROUBLESHOOT" in appt) or bool(sched.get("emergency_approved")) or bool(sched.get("awaiting_emergency_confirm"))

    immediate_dispatch_phrases = {
        "now", "right now", "immediately", "asap", "as soon as possible",
        "come now", "dispatch now", "today", "today please", "i need somebody today",
        "need somebody today", "send someone now", "whenever you can get here"
    }
    wants_immediate_dispatch = any(p in inbound_low for p in immediate_dispatch_phrases)

    if is_emergency and sched.get("address_verified") and not sched.get("emergency_approved"):
        return "This looks urgent. We can send someone now and arrival is usually within 1 to 2 hours. Troubleshoot and repair visits are $395. Do you want us to dispatch someone now?"

    ambiguous_times = {
        "any time", "anytime", "whenever", "later", "sometime", "around",
        "as soon as possible", "asap", "i'm around", "im around", "i'm home today",
        "im home today", "i'm here all day", "im here all day", "it doesn't matter",
        "it doesnt matter", "whenever works", "whenever you can", "sometime today", "later today"
    }
    time_of_day_phrases = {
        "today", "this morning", "this afternoon", "this evening", "later today",
        "sometime today", "i'm around today", "im around today", "i'll be home this afternoon",
        "ill be home this afternoon", "today works", "i'm available today", "im available today"
    }
    provided_ambiguous_time = any(p in inbound_low for p in ambiguous_times | time_of_day_phrases)

    step = sched.get("pending_step")
    if step == "need_address":
        return build_address_prompt(sched)
    if step == "need_date":
        if sched.get("offered_slot_options"):
            return v13_format_slot_choice_prompt(sched)
        if is_emergency and wants_immediate_dispatch and sched.get("address_verified"):
            return "This looks urgent. We can send someone now and arrival is usually within 1 to 2 hours. Troubleshoot and repair visits are $395. Do you want us to dispatch someone now?"
        if not sched.get("scheduled_time"):
            return humanize_question("What day and time work best for you?")
        return humanize_question("What day works best for you?")
    if step == "need_time":
        if sched.get("offered_slot_options"):
            return v13_format_slot_choice_prompt(sched)
        if is_emergency and wants_immediate_dispatch and sched.get("address_verified"):
            return "This looks urgent. We can send someone now and arrival is usually within 1 to 2 hours. Troubleshoot and repair visits are $395. Do you want us to dispatch someone now?"
        if not sched.get("scheduled_date"):
            return humanize_question("What day and time work best for you?")
        if provided_ambiguous_time:
            if is_emergency and sched.get("address_verified"):
                return "This looks urgent. We can send someone now and arrival is usually within 1 to 2 hours. Troubleshoot and repair visits are $395. Do you want us to dispatch someone now?"
            if is_emergency:
                return humanize_question("What’s the address for the work?")
            return humanize_question("What time works for you?")
        return humanize_question("What time works best for you?")
    if step == "need_name":
        first_name = get_active_first_name(profile)
        last_name = get_active_last_name(profile)
        if first_name and not last_name:
            return humanize_question(f"{first_name}, what is your last name?")
        return humanize_question("What is your first and last name?")
    if step == "need_email":
        return humanize_question("What is the best email address for the appointment?")

    if step is None and not (sched.get("booking_created") and sched.get("square_booking_id")):
        if sched.get("scheduled_date") and sched.get("scheduled_time"):
            final_key = f"{sched.get('scheduled_date')}|{sched.get('scheduled_time')}"
            if sched.get("last_final_confirmation_key") == final_key and (
                sched.get("final_confirmation_sent") or sched.get("final_confirmation_accepted")
            ):
                return "Okay."
            try:
                if isinstance(sched.get("scheduled_date"), str) and re.match(r"^\d{4}-\d{2}-\d{2}$", sched["scheduled_date"]):
                    d = datetime.strptime(sched["scheduled_date"], "%Y-%m-%d")
                    human_d = d.strftime("%A, %B %d").replace(" 0", " ")
                else:
                    human_d = (sched.get("scheduled_date") or "that day").strip()
            except Exception:
                human_d = (sched.get("scheduled_date") or "that day").strip()
            human_t = humanize_time(sched.get("scheduled_time")) if sched.get("scheduled_time") else "that time"
            sched["final_confirmation_sent"] = True
            sched["last_final_confirmation_key"] = final_key
            return f"Just to confirm, {human_d} at {human_t}. Is that still good?"
        return "Okay."

    return (conv.get("last_sms_body") or "Okay.").strip() or "Okay."

# ---------------------------------------------------
# Incoming SMS (B-3 State Machine, Option A)
# ---------------------------------------------------
@app.route("/incoming-sms", methods=["POST"])
def incoming_sms():
    inbound_text = request.form.get("Body", "") or ""
    phone_raw   = request.form.get("From", "")
    phone       = (phone_raw or "").replace("whatsapp:", "")
    inbound_sid = request.form.get("MessageSid") or request.form.get("SmsSid") or ""
    convo_key   = phone or inbound_sid or request.form.get("CallSid") or "unknown"
    inbound_low  = inbound_text.lower().strip()

    # Google LSA can send platform wrappers through the same SMS webhook.
    # Extract customer-authored payloads and suppress pure platform notices.
    lsa_customer_message = extract_lsa_customer_message(inbound_text)
    if is_pure_lsa_platform_notice(inbound_text):
        try:
            log_event("SMS_PLATFORM_SUPPRESSED", phone, {"sid": inbound_sid, "body": _safe_monitor_text(inbound_text)})
        except Exception:
            pass
        return Response(str(MessagingResponse()), mimetype="text/xml")
    if lsa_customer_message:
        inbound_text = lsa_customer_message
        inbound_low = inbound_text.lower().strip()

    try:
        log_event("SMS_IN", phone, {"sid": inbound_sid, "body": _safe_monitor_text(inbound_text)})
    except Exception:
        pass

    # SECRET RESET COMMAND
    if inbound_low == "mobius1":
        conversations[convo_key] = {
            "profile": {"name": None, "first_name": None, "last_name": None, "email": None, "recognized_first_name": None, "recognized_last_name": None, "recognized_email": None, "active_first_name": None, "active_last_name": None, "active_email": None, "voicemail_first_name": None, "voicemail_last_name": None, "known_people": [], "identity_source": None, "square_customer_id": None, "square_lookup_done": False, "addresses": [], "upcoming_appointment": None, "past_jobs": []},
            "current_job": {"job_type": None, "raw_description": None},
            "sched": {
                "pending_step": None,
                "scheduled_date": None,
                "scheduled_time": None,
                "appointment_type": None,
                "normalized_address": None,
                "raw_address": None,
                "booking_created": False,

                # Address Assembly State
                "address_candidate": None,
                "address_verified": False,
                "address_missing": None,
                "address_parts": {},

                # Emergency state flags (safe defaults)
                "awaiting_emergency_confirm": False,
                "emergency_approved": False,
                "name_engine_state": None,
                "name_engine_prompted": False,
                "name_engine_candidate_first": None,
                "name_engine_candidate_last": None,
                "name_engine_expected_known_first": None,
                "name_engine_selected_first": None,
                "soft_rejection_state": None,
                "soft_rejection_open": False,
                "soft_rejection_ts": None,
            }
        }
        resp = MessagingResponse()
        resp.message("✔ Memory reset complete for this number.")
        return Response(str(resp), mimetype="text/xml")

    # ---------------------------------------------------
    # Initialize layers (HARDENED: never assume dict shape)
    # ---------------------------------------------------
    conv = conversations.setdefault(phone, {})
    if lsa_customer_message:
        conv["source"] = "google_lsa"

    # Command Center manual takeover hard lock.
    # If Kyle marks a thread Manual in Prevolt Command Center, the backend must
    # keep logging inbound customer messages but must not send automated SMS,
    # continue address/date/email prompts, or create bookings.
    manual_sched = conv.setdefault("sched", {})
    if conv.get("thread_type") == "manual_only" or manual_sched.get("manual_only"):
        conv["thread_type"] = "manual_only"
        manual_sched["manual_only"] = True
        manual_sched["booking_allowed"] = False
        manual_sched["pending_step"] = None
        manual_sched["awaiting_slot_offer_choice"] = False
        manual_sched["offered_slot_options"] = []
        manual_sched.setdefault("manual_reason", "command_center_manual")
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        try:
            log_event("MANUAL_ONLY_SMS_SUPPRESSED", phone, {
                "sid": inbound_sid,
                "body": _safe_monitor_text(inbound_text),
                "reason": manual_sched.get("manual_reason") or "command_center_manual",
            }, conv)
        except Exception:
            pass
        return Response(str(MessagingResponse()), mimetype="text/xml")

    # Twilio and WhatsApp can occasionally retry the same inbound webhook.
    # Guard both by inbound SID and by a short body fingerprint window.
    inbound_fingerprint = re.sub(r"\s+", " ", inbound_low).strip()
    now_ts = time.time()
    if inbound_sid and conv.get("last_inbound_sid") == inbound_sid and conv.get("last_sms_body"):
        tw = MessagingResponse()
        tw.message(conv.get("last_sms_body"))
        return Response(str(tw), mimetype="text/xml")
    if (
        inbound_fingerprint
        and conv.get("last_inbound_fingerprint") == inbound_fingerprint
        and conv.get("last_inbound_fingerprint_ts")
        and (now_ts - float(conv.get("last_inbound_fingerprint_ts") or 0)) <= 90
        and conv.get("last_sms_body")
    ):
        tw = MessagingResponse()
        tw.message(conv.get("last_sms_body"))
        return Response(str(tw), mimetype="text/xml")

    profile = conv.setdefault("profile", {})
    profile.setdefault("name", None)
    profile.setdefault("addresses", [])
    profile.setdefault("upcoming_appointment", None)
    profile.setdefault("past_jobs", [])

    profile.setdefault("first_name", None)
    profile.setdefault("last_name", None)
    profile.setdefault("email", None)

    profile.setdefault("recognized_first_name", None)
    profile.setdefault("recognized_last_name", None)
    profile.setdefault("recognized_email", None)

    profile.setdefault("active_first_name", None)
    profile.setdefault("active_last_name", None)
    profile.setdefault("active_email", None)
    profile.setdefault("voicemail_first_name", None)
    profile.setdefault("voicemail_last_name", None)
    profile.setdefault("known_people", [])
    profile.setdefault("voicemail_first_name", None)
    profile.setdefault("voicemail_last_name", None)
    profile.setdefault("known_people", [])

    profile.setdefault("identity_source", None)
    profile.setdefault("square_customer_id", None)
    profile.setdefault("square_lookup_done", False)
    # keep customer_type if it exists (from call flow)
    profile.setdefault("customer_type", profile.get("customer_type"))

    current_job = conv.setdefault("current_job", {})
    current_job.setdefault("job_type", None)
    current_job.setdefault("raw_description", None)

    sched = conv.setdefault("sched", {})
    sched.setdefault("pending_step", None)
    sched.setdefault("scheduled_date", None)
    sched.setdefault("scheduled_time", None)
    sched.setdefault("appointment_type", None)
    sched.setdefault("normalized_address", None)
    sched.setdefault("raw_address", None)
    sched.setdefault("booking_created", False)
    sched.setdefault("soft_rejection_state", None)
    sched.setdefault("soft_rejection_open", False)
    sched.setdefault("soft_rejection_ts", None)

    # Address Assembly State
    sched.setdefault("address_candidate", None)
    sched.setdefault("address_verified", False)
    sched.setdefault("address_missing", None)
    sched.setdefault("address_parts", {})

    # Emergency flags (used by incoming_sms patch logic too)
    sched.setdefault("awaiting_emergency_confirm", False)
    sched.setdefault("emergency_approved", False)

    # Customer hard-stop and correction gates run before every pending_step handler.
    hard_stop_reason = detect_customer_hard_stop(inbound_text)
    if hard_stop_reason:
        reply_body = apply_customer_hard_stop(conv, hard_stop_reason, inbound_text)
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = reply_body
        try:
            log_event("CUSTOMER_HARD_STOP", phone, {"body": _safe_monitor_text(inbound_text), "reason": hard_stop_reason, "reply": _safe_monitor_text(reply_body)}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(reply_body)
        return Response(str(tw), mimetype="text/xml")

    # Customer asked for a call instead of continuing automation.
    if v13_looks_like_callback_request(inbound_text):
        reply_body = v13_apply_callback_request(conv, inbound_text)
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = reply_body
        try:
            log_event("CUSTOMER_CALLBACK_REQUESTED", phone, {"body": _safe_monitor_text(inbound_text), "reply": _safe_monitor_text(reply_body)}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(reply_body)
        return Response(str(tw), mimetype="text/xml")

    address_reject_reply = handle_address_confirmation_rejection(conv, inbound_text)
    if address_reject_reply:
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = address_reject_reply
        try:
            log_event("ADDRESS_CONFIRMATION_REJECTED", phone, {"body": _safe_monitor_text(inbound_text), "reply": _safe_monitor_text(address_reject_reply)}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(address_reject_reply)
        return Response(str(tw), mimetype="text/xml")

    address_accept_reply = handle_address_confirmation_acceptance(conv, inbound_text)
    if address_accept_reply:
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = address_accept_reply
        try:
            log_event("ADDRESS_CONFIRMATION_ACCEPTED", phone, {"body": _safe_monitor_text(inbound_text), "reply": _safe_monitor_text(address_accept_reply)}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(address_accept_reply)
        return Response(str(tw), mimetype="text/xml")

    if looks_like_vendor_sales_or_spam(inbound_text):
        clear_service_booking_state_for_non_service(conv, "vendor_sales_or_spam")
        sched["booking_allowed"] = False
        sched["manual_reason"] = "vendor_sales_or_spam"
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        try:
            log_event("VENDOR_OR_SPAM_SUPPRESSED", phone, {"body": _safe_monitor_text(inbound_text)}, conv)
        except Exception:
            pass
        return Response(str(MessagingResponse()), mimetype="text/xml")

    # Non-service / thread-type hard guard before normal booking state can run.
    thread_type = detect_non_service_thread_type(conv, inbound_text, conv.get("category"), conv.get("cleaned_transcript"))
    if thread_type == "vendor_sales_or_spam":
        clear_service_booking_state_for_non_service(conv, "vendor_sales_or_spam")
        sched["booking_allowed"] = False
        sched["manual_reason"] = "vendor_sales_or_spam"
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        try:
            log_event("VENDOR_OR_SPAM_SUPPRESSED", phone, {"body": _safe_monitor_text(inbound_text), "thread_type": "vendor_sales_or_spam"}, conv)
        except Exception:
            pass
        return Response(str(MessagingResponse()), mimetype="text/xml")
    if thread_type == "employment_inquiry":
        clear_service_booking_state_for_non_service(conv, "employment_inquiry")
        reply_body = build_employment_inquiry_reply()
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = reply_body
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(reply_body), "thread_type": "employment_inquiry"}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(reply_body)
        return Response(str(tw), mimetype="text/xml")
    if thread_type == "callback_requested":
        clear_service_booking_state_for_non_service(conv, "callback_requested")
        reply_body = v13_apply_callback_request(conv, inbound_text)
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = reply_body
        try:
            log_event("CUSTOMER_CALLBACK_REQUESTED", phone, {"body": _safe_monitor_text(inbound_text), "thread_type": "callback_requested"}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(reply_body)
        return Response(str(tw), mimetype="text/xml")
    if thread_type == "commercial_bid_contact":
        clear_service_booking_state_for_non_service(conv, "commercial_bid_contact")
        reply_body = build_commercial_context_reply(conv, inbound_text)
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = reply_body
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(reply_body), "thread_type": "commercial_bid_contact"}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(reply_body)
        return Response(str(tw), mimetype="text/xml")

    # Explicit commercial walkthrough/site-walk requests are manual-only.
    # Commercial/multifamily equipment requests that simply ask for availability
    # remain normal $195 evaluation leads.
    try:
        if looks_like_complex_commercial_coordination_request(inbound_text):
            clear_service_booking_state_for_non_service(conv, "manual_only")
            sched["manual_reason"] = "commercial_walkthrough_coordination"
            # Preserve the address for office review without treating it as a bookable Square visit.
            try:
                extracted_addr = extract_service_address_from_mixed_text(inbound_text)
                if extracted_addr:
                    sched["raw_address"] = extracted_addr
            except Exception:
                pass
            reply_body = build_complex_commercial_coordination_reply(inbound_text)
            conv["last_inbound_sid"] = inbound_sid
            conv["last_inbound_fingerprint"] = inbound_fingerprint
            conv["last_inbound_fingerprint_ts"] = now_ts
            conv["last_sms_body"] = reply_body
            try:
                log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(reply_body), "thread_type": "manual_only", "manual_reason": "commercial_walkthrough_coordination"}, conv)
            except Exception:
                pass
            tw = MessagingResponse()
            tw.message(reply_body)
            return Response(str(tw), mimetype="text/xml")
    except Exception as e:
        print("[WARN] incoming_sms commercial walkthrough coordination guard failed:", repr(e))

    # Before any booking branch, upgrade partial address state from the newest customer text.
    try:
        if absorb_address_from_mixed_text(conv, inbound_text):
            log_event("ADDRESS_ABSORBED_FROM_INBOUND", phone, {"body": _safe_monitor_text(inbound_text), "address": _safe_monitor_text(sched.get("raw_address"), 240)}, conv)
    except Exception as e:
        print("[WARN] incoming_sms early address absorb failed:", repr(e))

    # First-message GLS/SMS service leads should establish the $195 evaluation
    # context before availability handling offers slots. This includes commercial
    # or multifamily equipment work like a meter-bank replacement.
    try:
        if looks_like_initial_service_booking_request(conv, inbound_text):
            reply_body = build_initial_service_booking_reply(conv, inbound_text)
            conv["last_inbound_sid"] = inbound_sid
            conv["last_inbound_fingerprint"] = inbound_fingerprint
            conv["last_inbound_fingerprint_ts"] = now_ts
            conv["last_sms_body"] = reply_body
            try:
                log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(reply_body), "initial_service_booking": True}, conv)
            except Exception:
                pass
            tw = MessagingResponse()
            tw.message(reply_body)
            return Response(str(tw), mimetype="text/xml")
    except Exception as e:
        print("[WARN] incoming_sms initial service booking guard failed:", repr(e))

    cleaned_transcript = conv.get("cleaned_transcript")
    category = conv.get("category")
    appointment_type = sched.get("appointment_type")
    initial_sms = conv.get("initial_sms")

    scheduled_date = sched.get("scheduled_date")
    scheduled_time = sched.get("scheduled_time")
    address = sched.get("raw_address") or sched.get("normalized_address")

    # ---------------------------------------------------
    # PATCH: Extract address from initial SMS if missing
    # ---------------------------------------------------
    try:
        if initial_sms:
            m = re.search(r"\((.*?)\)", initial_sms)
            if m and not sched.get("raw_address"):
                extracted = m.group(1).strip()
                sched["raw_address"] = extracted
                address = extracted
                if extracted and extracted not in profile["addresses"]:
                    profile["addresses"].append(extracted)

            if not sched.get("raw_address"):
                street_pattern = (
                    r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+"
                    r"(st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|circle|blvd|way)\b"
                )
                m2 = re.search(street_pattern, initial_sms, flags=re.IGNORECASE)
                if m2:
                    extracted2 = m2.group(0).strip()
                    sched["raw_address"] = extracted2
                    address = extracted2
                    if extracted2 and extracted2 not in profile["addresses"]:
                        profile["addresses"].append(extracted2)

    except Exception as e:
        print("[WARN] initial_sms address extraction failed:", repr(e))

        # Only set pending_step here as a fallback; Step 4 will recompute properly.
        if not sched.get("emergency_approved"):
            if not sched.get("scheduled_date"):
                sched["pending_step"] = "need_date"
            elif not sched.get("scheduled_time"):
                sched["pending_step"] = "need_time"
            elif not sched.get("raw_address") and not sched.get("address_verified"):
                sched["pending_step"] = "need_address"
            elif not ((profile.get("active_first_name") or profile.get("first_name") or "").strip() and (profile.get("active_last_name") or profile.get("last_name") or "").strip()):
                sched["pending_step"] = "need_name"
            elif not ((profile.get("active_email") or profile.get("email") or "").strip()):
                sched["pending_step"] = "need_email"
            else:
                sched["pending_step"] = None

    # Keep address state fresh (pre-Step4)
    update_address_assembly_state(sched)

    # Pre-Step4 smart merge for compact city/state replies.
    try:
        apply_partial_address_reply(sched, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms partial address merge failed:", repr(e))

    # Absolute address gate: never schedule, offer slots, or ask for time while address is truly missing.
    # If we only needed confirmation and the customer keeps scheduling, treat that as implied confirmation.
    try:
        if address_gate_blocks_scheduling(conv):
            if v13_maybe_imply_address_confirmed(conv, inbound_text):
                recompute_pending_step(profile, sched)
            else:
                if _looks_like_scheduling_reply(inbound_text):
                    sched["deferred_schedule_text"] = inbound_text
                reply_body = build_address_gate_reply(conv)
                conv["last_inbound_sid"] = inbound_sid
                conv["last_inbound_fingerprint"] = inbound_fingerprint
                conv["last_inbound_fingerprint_ts"] = now_ts
                conv["last_sms_body"] = reply_body
                try:
                    log_event("ADDRESS_GATE_BLOCKED_SCHEDULING", phone, {"body": _safe_monitor_text(inbound_text), "reply": _safe_monitor_text(reply_body)}, conv)
                except Exception:
                    pass
                tw = MessagingResponse()
                tw.message(reply_body)
                return Response(str(tw), mimetype="text/xml")
    except Exception as e:
        print("[WARN] incoming_sms address gate failed:", repr(e))

    # If customer answers an offered slot list with "anytime" / "whatever works",
    # take the earliest offered slot before any name capture can misread it.
    try:
        if maybe_apply_flexible_offered_slot_choice(conv, inbound_text):
            recompute_pending_step(profile, sched)
            reply_body = choose_next_prompt_from_state(conv, inbound_text=inbound_text)
            reply_body = sanitize_sms_body(collapse_duplicate_sms(reply_body.strip()), booking_created=bool(sched.get("booking_created") and sched.get("square_booking_id")))
            conv["last_inbound_sid"] = inbound_sid
            conv["last_inbound_fingerprint"] = inbound_fingerprint
            conv["last_inbound_fingerprint_ts"] = now_ts
            conv["last_sms_body"] = reply_body
            try:
                log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(reply_body), "flexible_offered_slot_choice": True}, conv)
            except Exception:
                pass
            tw = MessagingResponse()
            tw.message(reply_body)
            return Response(str(tw), mimetype="text/xml")
    except Exception as e:
        print("[WARN] incoming_sms flexible offered slot choice failed:", repr(e))

    # If the customer asks for a different date while viewing fallback options,
    # respect the new date before words like "first" can be misread as "first option".
    try:
        date_change_reply = maybe_handle_new_date_during_offered_slots(conv, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms offered-slot date change handling failed:", repr(e))
        date_change_reply = None
    if date_change_reply:
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = date_change_reply.strip()
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(date_change_reply), "offered_slot_date_change": True}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(date_change_reply.strip())
        return Response(str(tw), mimetype="text/xml")

    # If customer chooses one of the offered slots, lock it in at the route level
    # before exact-slot fallback or the model can reinterpret the reply.
    try:
        if maybe_apply_offered_slot_selection(conv, inbound_text):
            recompute_pending_step(profile, sched)
            reply_body = choose_next_prompt_from_state(conv, inbound_text=inbound_text)
            reply_body = sanitize_sms_body(collapse_duplicate_sms(reply_body.strip()), booking_created=bool(sched.get("booking_created") and sched.get("square_booking_id")))
            conv["last_inbound_sid"] = inbound_sid
            conv["last_inbound_fingerprint"] = inbound_fingerprint
            conv["last_inbound_fingerprint_ts"] = now_ts
            conv["last_sms_body"] = reply_body
            try:
                log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(reply_body), "offered_slot_choice": True}, conv)
            except Exception:
                pass
            tw = MessagingResponse()
            tw.message(reply_body)
            return Response(str(tw), mimetype="text/xml")
    except Exception as e:
        print("[WARN] incoming_sms offered slot choice failed:", repr(e))

    # Time-only replies against an offered slot list: bind to the first offered date.
    try:
        if v13_time_only_on_offered_slots(conv, inbound_text):
            recompute_pending_step(profile, sched)
            reply_body = choose_next_prompt_from_state(conv, inbound_text=inbound_text)
            reply_body = sanitize_sms_body(collapse_duplicate_sms(reply_body.strip()), booking_created=bool(sched.get("booking_created") and sched.get("square_booking_id")))
            conv["last_inbound_sid"] = inbound_sid
            conv["last_inbound_fingerprint"] = inbound_fingerprint
            conv["last_inbound_fingerprint_ts"] = now_ts
            conv["last_sms_body"] = reply_body
            try:
                log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(reply_body), "time_only_on_offered_slot": True}, conv)
            except Exception:
                pass
            tw = MessagingResponse()
            tw.message(reply_body)
            return Response(str(tw), mimetype="text/xml")
    except Exception as e:
        print("[WARN] incoming_sms time-only offered slot handling failed:", repr(e))

    # Email can arrive before the slot is locked. Save it and ask only for the missing slot.
    try:
        email_guard_reply = v13_handle_email_before_slot(conv, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms email-before-slot guard failed:", repr(e))
        email_guard_reply = None
    if email_guard_reply:
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = email_guard_reply
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(email_guard_reply), "email_before_slot_guard": True}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(email_guard_reply)
        return Response(str(tw), mimetype="text/xml")

    # LIVE HOTFIX: exact date/time replies must be handled before any model,
    # interruption, name-capture, or availability fallback logic. This catches
    # messages like "Could we do Tuesday at 2pm" while pending_step is need_date.
    try:
        exact_slot_reply = maybe_handle_exact_slot_before_step4(conv, phone, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms early exact slot pre-Step4 failed:", repr(e))
        exact_slot_reply = None

    if exact_slot_reply:
        exact_slot_reply = sanitize_sms_body(
            collapse_duplicate_sms(exact_slot_reply.strip()),
            booking_created=bool(sched.get("booking_created") and sched.get("square_booking_id"))
        )
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = exact_slot_reply
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(exact_slot_reply), "exact_slot_early_pre_step4": True}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(exact_slot_reply)
        return Response(str(tw), mimetype="text/xml")

    # Route-level last-name salvage before Step 4.
    try:
        if maybe_capture_last_name_only(profile, sched, inbound_text):
            recompute_pending_step(profile, sched)
    except Exception as e:
        print("[WARN] incoming_sms last-name salvage failed:", repr(e))

    # If customer rejects all offered slot options, do not repeat the same options.
    try:
        if sched.get("awaiting_slot_offer_choice") and is_rejecting_offered_slots(inbound_text) and not detect_customer_hard_stop(inbound_text):
            sched["awaiting_slot_offer_choice"] = False
            sched["offered_slot_options"] = []
            reply_body = "No problem. What day or time range works better for you?"
            conv["last_inbound_sid"] = inbound_sid
            conv["last_inbound_fingerprint"] = inbound_fingerprint
            conv["last_inbound_fingerprint_ts"] = now_ts
            conv["last_sms_body"] = reply_body
            try:
                log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(reply_body), "slot_rejection": True}, conv)
            except Exception:
                pass
            tw = MessagingResponse()
            tw.message(reply_body)
            return Response(str(tw), mimetype="text/xml")
    except Exception as e:
        print("[WARN] incoming_sms slot rejection handling failed:", repr(e))

    # Deterministic availability and date-only handling before any generic interruption handling.
    try:
        availability_reply = deterministic_availability_reply(conv, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms deterministic availability failed:", repr(e))
        availability_reply = None

    if availability_reply:
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = availability_reply.strip()
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(availability_reply), "availability_hardened": True}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(availability_reply.strip())
        return Response(str(tw), mimetype="text/xml")

    # Direct availability questions should return real next slots before any generic interruption handling.
    try:
        if is_next_available_request(inbound_text):
            slot_options = get_next_available_slots((sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195"), limit=3)
            if slot_options:
                sched["awaiting_slot_offer_choice"] = True
                sched["offered_slot_options"] = slot_options
                sched["last_slot_unavailable_message"] = format_next_available_slots_message(slot_options)
                sched["booking_created"] = False
                sched["square_booking_id"] = None
                conv["last_inbound_sid"] = inbound_sid
                conv["last_inbound_fingerprint"] = inbound_fingerprint
                conv["last_inbound_fingerprint_ts"] = now_ts
                conv["last_sms_body"] = sched["last_slot_unavailable_message"]
                try:
                    log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(sched["last_slot_unavailable_message"])}, conv)
                except Exception:
                    pass
                tw = MessagingResponse()
                tw.message(sched["last_slot_unavailable_message"])
                return Response(str(tw), mimetype="text/xml")
    except Exception as e:
        print("[WARN] incoming_sms next available handling failed:", repr(e))

    # Deterministic fast path for pure interruption questions.
    # This prevents price / payment / permit questions from getting lost
    # behind the address collector when the inbound does not contain a new slot value.
    fast_interrupt = None
    try:
        if should_short_circuit_interrupt(conv, inbound_text):
            fast_interrupt = interruption_answer_and_return_prompt(conv, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms fast interrupt failed:", repr(e))

    if fast_interrupt:
        try:
            update_address_assembly_state(sched)
            recompute_pending_step(profile, sched)

            has_identity_for_booking = bool(
                (get_active_first_name(profile) and get_active_last_name(profile))
                or profile.get("square_customer_id")
            )
            has_contact_for_booking = bool(get_active_email(profile) or profile.get("square_customer_id"))
            appt_upper = (sched.get("appointment_type") or "").upper()
            emergency_mode = (
                "TROUBLESHOOT" in appt_upper
                or bool(sched.get("emergency_approved"))
                or bool(sched.get("awaiting_emergency_confirm"))
            )

            ready_after_interrupt = (
                bool(sched.get("scheduled_date")) and
                bool(sched.get("scheduled_time")) and
                bool(sched.get("address_verified")) and
                bool(sched.get("appointment_type")) and
                has_identity_for_booking and
                has_contact_for_booking and
                not sched.get("pending_step") and
                not sched.get("booking_created")
            )

            if ready_after_interrupt and not emergency_mode:
                booking_attempt = maybe_create_square_booking(phone, conv)
                if isinstance(booking_attempt, dict) and booking_attempt.get("status") == "stale_cancelled":
                    booking_attempt = maybe_create_square_booking(phone, conv)

                if sched.get("booking_created") and sched.get("square_booking_id"):
                    try:
                        booked_dt = datetime.strptime(sched["scheduled_date"], "%Y-%m-%d")
                        human_day = booked_dt.strftime("%A, %B %d").replace(" 0", " ")
                    except Exception:
                        human_day = sched.get("scheduled_date") or "that day"
                    booked_time = humanize_time(sched.get("scheduled_time") or "") or (sched.get("scheduled_time") or "that time")
                    fast_interrupt = f"{fast_interrupt.strip()} You're all set for {human_day} at {booked_time}. We have you on the schedule."
                elif isinstance(booking_attempt, dict) and booking_attempt.get("status") == "slot_unavailable":
                    fast_interrupt = f"{fast_interrupt.strip()} {booking_attempt.get('message') or ''}".strip()
                elif isinstance(booking_attempt, dict) and booking_attempt.get("status") == "missing_explicit_time":
                    fast_interrupt = f"{fast_interrupt.strip()} What time works best that day?".strip()
        except Exception as e:
            print("[WARN] fast interrupt immediate booking failed:", repr(e))

        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = fast_interrupt.strip()
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(fast_interrupt)}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(fast_interrupt.strip())
        return Response(str(tw), mimetype="text/xml")

    # Warm soft rejection handling during booking.
    try:
        soft_reject_reply = build_soft_rejection_reply(conv, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms soft rejection failed:", repr(e))
        soft_reject_reply = None

    if soft_reject_reply:
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = soft_reject_reply.strip()
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(soft_reject_reply)}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(soft_reject_reply.strip())
        return Response(str(tw), mimetype="text/xml")

    # Resume paused conversation if the customer comes back later.
    try:
        if conv.setdefault("sched", {}).get("soft_rejection_open"):
            absorb_obvious_booking_details(conv, inbound_text)
        resumed_reply = maybe_resume_paused_conversation(conv, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms resume failed:", repr(e))
        resumed_reply = None

    if resumed_reply:
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = resumed_reply.strip()
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(resumed_reply)}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(resumed_reply.strip())
        return Response(str(tw), mimetype="text/xml")

    # Post-booking questions should answer cleanly without reopening the booking flow.
    try:
        post_booking_reply = handle_post_booking_question(conv, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms post booking question failed:", repr(e))
        post_booking_reply = None

    if post_booking_reply:
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = post_booking_reply.strip()
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(post_booking_reply)}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(post_booking_reply.strip())
        return Response(str(tw), mimetype="text/xml")

    # ---------------------------------------------------
    # Pre-Step4 salvage: always absorb obvious slot payloads so
    # explicit replies like "next Friday at 2pm" are stored before
    # any LLM call. This prevents date/time loops.
    # ---------------------------------------------------
    try:
        absorb_obvious_booking_details(conv, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms absorb obvious booking details failed:", repr(e))

    # Hard exact-slot gate BEFORE Step 4.
    # If customer says "Monday at 9am", check Square/state now.
    # Never let the model say "works" or "I'll reserve it" first.
    try:
        exact_slot_reply = maybe_handle_exact_slot_before_step4(conv, phone, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms exact slot pre-Step4 failed:", repr(e))
        exact_slot_reply = None

    if exact_slot_reply:
        exact_slot_reply = sanitize_sms_body(collapse_duplicate_sms(exact_slot_reply.strip()), booking_created=bool(sched.get("booking_created") and sched.get("square_booking_id")))
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = exact_slot_reply
        try:
            log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(exact_slot_reply), "exact_slot_pre_step4": True}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(exact_slot_reply)
        return Response(str(tw), mimetype="text/xml")

    # ---------------------------------------------------
    # Run Step 4
    # ---------------------------------------------------
    reply = generate_reply_for_inbound(
        cleaned_transcript,
        category,
        appointment_type,
        initial_sms,
        inbound_text,
        scheduled_date,
        scheduled_time,
        address
    )

    # SAVE STEP 4 RESULTS
    if reply.get("scheduled_date"):
        sched["scheduled_date"] = reply.get("scheduled_date")
    if reply.get("scheduled_time"):
        if extract_explicit_time_from_text(inbound_text) or sched.get("scheduled_time_source") in {"customer_explicit", "offered_slot", "voicemail_explicit"}:
            sched["scheduled_time"] = reply.get("scheduled_time")
            sched["scheduled_time_source"] = sched.get("scheduled_time_source") or "customer_explicit"
        elif not salvage_relative_date_from_text(inbound_text):
            # Last-resort compatibility: do not accept model/current-clock times from date-only replies.
            sched["scheduled_time"] = reply.get("scheduled_time")
            sched["scheduled_time_source"] = "model_non_date_only"

    if reply.get("address"):
        candidate_address = scrub_non_address_tail(str(reply["address"] or "").strip())
        existing_address = scrub_non_address_tail(str(sched.get("raw_address") or "").strip())
        inbound_has_address = bool(extract_service_address_from_text(inbound_text))
        safe_candidate = safe_address_candidate_from_text(candidate_address) or (candidate_address if is_plausible_address_text(candidate_address) else None)
        if safe_candidate and not (
            yes_text(inbound_text)
            and existing_address
            and (safe_candidate == f"{existing_address}, Yes" or safe_candidate == f"{existing_address} Yes")
        ):
            # Do not let customer questions get appended into the address field.
            if not (existing_address and looks_like_non_address_fragment(inbound_text) and not inbound_has_address):
                if (not existing_address) or (not sched.get("address_verified")) or _address_has_house_number_and_street(safe_candidate):
                    set_raw_address_safe(sched, safe_candidate)
                    sched["normalized_address"] = None
                    # addresses list is guaranteed above
                    saved_candidate = sched.get("raw_address") or safe_candidate
                    if saved_candidate and saved_candidate not in profile["addresses"]:
                        profile["addresses"].append(saved_candidate)
        elif candidate_address and existing_address:
            try:
                log_event("ADDRESS_MODEL_OUTPUT_REJECTED", phone, {"model_address": _safe_monitor_text(candidate_address), "kept_address": _safe_monitor_text(existing_address)}, conv)
            except Exception:
                pass

    # Re-derive address assembly state after Step 4 updates
    update_address_assembly_state(sched)

    recompute_pending_step(profile, sched)

    sms_body = collapse_duplicate_sms((reply.get("sms_body") or "").strip())
    sms_body = suppress_unbooked_reservation_language(conv, phone, inbound_text, sms_body)

    # Regular-booking hard guard:
    # If Step 4 has already captured a full slot + identity + contact + verified address,
    # attempt the normal Square booking path immediately before asking anything again.
    has_identity_for_booking = bool(
        (get_active_first_name(profile) and get_active_last_name(profile))
        or profile.get("square_customer_id")
    )
    has_contact_for_booking = bool(get_active_email(profile) or profile.get("square_customer_id"))
    appt_upper = (sched.get("appointment_type") or "").upper()
    emergency_mode = (
        "TROUBLESHOOT" in appt_upper
        or bool(sched.get("emergency_approved"))
        or bool(sched.get("awaiting_emergency_confirm"))
    )

    ready_for_route_booking = (
        bool(sched.get("scheduled_date")) and
        bool(sched.get("scheduled_time")) and
        bool(sched.get("address_verified")) and
        bool(sched.get("appointment_type")) and
        has_identity_for_booking and
        has_contact_for_booking and
        not sched.get("pending_step") and
        not sched.get("booking_created")
    )

    if ready_for_route_booking and not emergency_mode:
        try:
            booking_attempt = maybe_create_square_booking(phone, conv)
            if isinstance(booking_attempt, dict) and booking_attempt.get("status") == "stale_cancelled":
                booking_attempt = maybe_create_square_booking(phone, conv)

            if isinstance(booking_attempt, dict) and booking_attempt.get("status") == "slot_unavailable":
                sms_body = booking_attempt.get("message") or "That time is already booked. Here are three other times that work."
            elif isinstance(booking_attempt, dict) and booking_attempt.get("status") == "missing_explicit_time":
                sms_body = "What time works best that day?"
            elif isinstance(booking_attempt, dict) and booking_attempt.get("status") == "outside_hours":
                if sched.get("scheduled_date"):
                    sms_body = "We typically schedule between 9am and 4pm. What time in that window works for you?"
                else:
                    sms_body = "We typically schedule between 9am and 4pm. What day and time in that window work best for you?"
            elif isinstance(booking_attempt, dict) and booking_attempt.get("status") == "weekend_blocked":
                sms_body = "We schedule non-emergency visits Monday through Friday. What day and time work best for you?"
            elif sched.get("booking_created") and sched.get("square_booking_id"):
                try:
                    booked_dt = datetime.strptime(sched["scheduled_date"], "%Y-%m-%d")
                    human_day = booked_dt.strftime("%A, %B %d").replace(" 0", " ")
                except Exception:
                    human_day = sched.get("scheduled_date") or "that day"
                booked_time = humanize_time(sched.get("scheduled_time") or "") or (sched.get("scheduled_time") or "that time")
                sms_body = f"You're all set for {human_day} at {booked_time}. We have you on the schedule."
        except Exception as e:
            print("[WARN] route-level booking attempt failed:", repr(e))

    next_prompt = choose_next_prompt_from_state(conv, inbound_text=inbound_text)

    # Route-level guardrail: only override when Step 4 returned a stall / generic filler.
    generic_fillers = {"", "Okay.", "Okay", "ok", "ok.", "sure.", "Sure."}
    if sms_body in generic_fillers:
        sms_body = next_prompt

    # Frustration and duplicate-prompt protection.
    if is_frustrated_with_bot(inbound_text):
        sms_body = "Sorry about that. I have the information you sent. Our office will review this manually."
        conv["thread_type"] = "manual_only"
        sched["pending_step"] = None
    elif outbound_is_duplicate(conv, sms_body) and should_send_no_reply_for_duplicate(inbound_text):
        try:
            log_event("SMS_DUPLICATE_SUPPRESSED", phone, {"body": _safe_monitor_text(sms_body)}, conv)
        except Exception:
            pass
        return Response(str(MessagingResponse()), mimetype="text/xml")

    conv["last_inbound_sid"] = inbound_sid
    conv["last_inbound_fingerprint"] = inbound_fingerprint
    conv["last_inbound_fingerprint_ts"] = now_ts
    conv["last_sms_body"] = sms_body
    try:
        log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(sms_body)}, conv)
    except Exception:
        pass

    tw = MessagingResponse()
    tw.message(sms_body)
    return Response(str(tw), mimetype="text/xml")

import re
from datetime import datetime, timedelta

# ---------------------------------------------------
# Human SMS polish + anti-AI telltales + ack memory
# ---------------------------------------------------

ACK_PHRASES = {
    "ok", "okay", "k", "kk", "sure", "sounds good", "that works", "works", "yep", "yeah", "yes",
    "thanks", "thank you", "thx", "got it", "done", "perfect"
}

def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def collapse_duplicate_sms(text: str) -> str:
    text = " ".join((text or "").split()).strip()
    if not text:
        return text

    # Exact repeated whole-string halves
    half = len(text) // 2
    if len(text) % 2 == 0 and text[:half].strip() == text[half:].strip():
        return text[:half].strip()

    # Repeated sentence blocks
    parts = [p.strip() for p in re.split(r'(?<=[?.!])\s+', text) if p.strip()]
    out = []
    for p in parts:
        if not out or out[-1].lower() != p.lower():
            out.append(p)
    return " ".join(out).strip()

def is_ack_message(inbound_text: str) -> bool:
    t = _norm_text(inbound_text).lower()
    if not t:
        return False
    # short ack-only replies
    if t in ACK_PHRASES:
        return True
    # "ok thanks" style
    if len(t) <= 20 and any(p in t for p in ["ok", "okay", "thanks", "thank you", "thx", "got it"]):
        return True
    return False

def remove_ai_punctuation(text: str) -> str:
    if not text:
        return text
    # em dash / en dash / spaced hyphen patterns
    text = text.replace("—", ".").replace("–", ".")
    text = text.replace(" - ", " ")
    return text

def strip_ai_telltales(text: str) -> str:
    """
    Removes common bot openers and filler that scream "assistant".
    Keeps the message meaning intact.
    """
    if not text:
        return text

    t = _norm_text(text)

    # Kill leading filler openers
    # Examples: "Got it." "Thanks." "Sure." "Absolutely."
    t = re.sub(r"^(got it|thanks|thank you|sure|absolutely|no problem|ok|okay)\b[\s,.:;!-]*", "", t, flags=re.I)

    # Kill "one moment" / wait-text (you said never do this)
    t = re.sub(r"\b(one moment|one sec|one second|give me a moment|please wait|hang tight)\b[\s,.:;!-]*", "", t, flags=re.I)

    # Kill "you're all set" if not actually booked (you also gate this elsewhere, but belt + suspenders)
    # We'll let booking-confirm strings happen only when booking_created True upstream.
    # Here we just remove the phrase if it appears alone at the start.
    t = re.sub(r"^(you'?re all set)\b[\s,.:;!-]*", "", t, flags=re.I)

    return _norm_text(t)

def shorten_for_texting(text: str, max_chars: int = 220) -> str:
    """
    Makes messages feel like a human text:
    - fewer clauses
    - fewer stacked sentences
    - keeps it under a reasonable length
    """
    if not text:
        return text

    t = _norm_text(text)

    # Replace overly formal connectors
    t = re.sub(r"\b(perfect|certainly|additionally|therefore|however)\b", "", t, flags=re.I)
    t = _norm_text(t)

    # If it's long, split into 2 sentences max by punctuation
    if len(t) > max_chars:
        parts = re.split(r"(?<=[.!?])\s+", t)
        t = " ".join(parts[:2]).strip()

    # If still long, hard-trim but keep clean end
    if len(t) > max_chars:
        t = t[:max_chars].rstrip()
        t = re.sub(r"[\s,;:]+$", "", t)
        t += "."

    return _norm_text(t)

def postprocess_sms(sms_body: str, inbound_text: str, sched: dict, booking_created: bool = False) -> str:
    """
    Final pass: removes AI tells, avoids repeated acknowledgements, and keeps it human.
    """
    sms = sms_body or ""
    sms = remove_ai_punctuation(sms)
    sms = strip_ai_telltales(sms)

    # -----------------------------
    # Acknowledgement memory
    # -----------------------------
    # If user just sent an acknowledgement ("ok", "thanks"), do NOT reply with another acknowledgement.
    if is_ack_message(inbound_text):
        # If our message is now empty (because it was only "Thanks."), replace with a useful next step.
        # Prefer next missing atom if present.
        update_address_assembly_state(sched)

        if not sched.get("address_verified"):
            sms = (_known_address_question() or build_address_prompt(sched))
        elif not sched.get("scheduled_date"):
            sms = "What day works for you?"
        elif not sched.get("scheduled_time"):
            sms = "What time works best?"
        else:
            # If everything is present and booking is not created, just keep it simple.
            sms = "Got it."

        # Even here, avoid "Got it." repeats:
        # If we said "Got it." recently, say nothing new or ask a clarifier.
        last_ack = sched.get("last_ack_text")
        last_ack_ts = sched.get("last_ack_ts")
        if last_ack and last_ack.lower() == sms.lower():
            sms = "Okay."
        sched["last_ack_text"] = sms
        sched["last_ack_ts"] = datetime.utcnow().isoformat()

    # If booking is not actually created, remove any "confirmation" talk (extra safety)
    if not booking_created:
        confirmation_markers = [
            "confirmation number", "confirming", "confirmed", "your appointment is", "booked for", "scheduled for"
        ]
        low = sms.lower()
        if any(m in low for m in confirmation_markers):
            update_address_assembly_state(sched)
            if not sched.get("address_verified"):
                sms = (_known_address_question() or build_address_prompt(sched))
            elif not sched.get("scheduled_date"):
                sms = "What day works for you?"
            elif not sched.get("scheduled_time"):
                sms = "What time works best?"
            else:
                sms = "Okay."

    sms = shorten_for_texting(sms)

    # If empty after stripping, never send empty: pick a safe next line
    if not sms:
        update_address_assembly_state(sched)
        if not sched.get("address_verified"):
            sms = build_address_prompt(sched)
        elif not sched.get("scheduled_date"):
            sms = "What day works for you?"
        elif not sched.get("scheduled_time"):
            sms = "What time works best?"
        else:
            sms = "Okay."

    return sms



def get_active_first_name(profile: dict) -> str:
    return (profile.get("active_first_name") or profile.get("first_name") or profile.get("recognized_first_name") or "").strip()

def get_active_last_name(profile: dict) -> str:
    return (profile.get("active_last_name") or profile.get("last_name") or profile.get("recognized_last_name") or "").strip()

def get_active_email(profile: dict) -> str:
    return (profile.get("active_email") or profile.get("email") or profile.get("recognized_email") or "").strip()

def get_display_first_name(profile: dict) -> str:
    return (
        profile.get("active_first_name")
        or profile.get("voicemail_first_name")
        or profile.get("recognized_first_name")
        or profile.get("first_name")
        or ""
    ).strip()

def normalize_person_name(s: str) -> str:
    s = " ".join((s or "").strip().split())
    if not s:
        return ""
    return " ".join(p[:1].upper() + p[1:].lower() for p in s.split())

def maybe_capture_last_name_only(profile: dict, sched: dict, inbound_text: str) -> bool:
    """
    Route-level salvage:
    If first name is already known, last name is still missing, and the customer
    replies with a simple last-name-looking token, persist it before Step 4.
    """
    if (sched.get("pending_step") or "").strip().lower() != "need_name":
        return False

    first_name = get_active_first_name(profile)
    last_name = get_active_last_name(profile)
    if not first_name or last_name:
        return False

    cleaned = re.sub(r"[^A-Za-z'\- ]", " ", inbound_text or "").strip()
    parts = [p for p in cleaned.split() if p]
    if not parts:
        return False

    low = " ".join(parts).lower().strip()
    if is_not_a_person_name_reply(inbound_text):
        return False

    blocked = {
        "yes", "no", "okay", "ok", "thanks", "thank you", "yep", "yeah", "sure",
        "tomorrow", "today", "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "morning", "afternoon", "evening", "noon", "midday",
        "anytime", "any time", "whenever", "when ever", "whatever works",
        "whatever works for you", "any of those", "either", "whichever",
        "you pick", "your choice", "no preference"
    }
    if low in blocked:
        return False

    # Common real-world rescue:
    # "Smith"
    # "my last name is Smith"
    # "it is Smith"
    candidate = None
    if len(parts) == 1:
        candidate = parts[0]
    else:
        m = re.search(r"\b(?:last name is|lastname is|my last name is|it is|its|it's)\s+([A-Za-z][A-Za-z'\-]{1,})\b", cleaned, flags=re.I)
        if m:
            candidate = m.group(1)
        elif len(parts) <= 3 and parts[-1].lower() not in {"name", "last", "is", "my", "it"}:
            candidate = parts[-1]

    candidate = normalize_person_name(candidate or "")
    if not candidate:
        return False

    profile["active_last_name"] = candidate
    profile["last_name"] = candidate
    if not profile.get("identity_source"):
        profile["identity_source"] = "customer_provided_last_name"
    return True

def extract_possible_person_name(text: str) -> tuple[str | None, str | None]:
    """
    Conservative name extraction for voicemail identity only.

    IMPORTANT:
    - Do NOT infer names from phrases like "I'm looking to..." or "I am calling about..."
    - Only accept explicit self-identification patterns.
    - If the caller did not clearly say their name, return (None, None).
    """
    txt = " ".join((text or "").strip().split())
    if not txt:
        return None, None

    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z'\-]{1,})(?:\s+([A-Za-z][A-Za-z'\-]{1,}))?",
        r"\bname\s+is\s+([A-Za-z][A-Za-z'\-]{1,})(?:\s+([A-Za-z][A-Za-z'\-]{1,}))?",
        r"\bthis is\s+([A-Za-z][A-Za-z'\-]{1,})(?:\s+([A-Za-z][A-Za-z'\-]{1,}))?",
    ]

    stop_words = {
        "looking", "calling", "trying", "needing", "need", "wanting", "want",
        "with", "for", "about", "because", "regarding", "located", "living",
        "from", "at", "in", "the", "a", "an", "and", "or"
    }

    for pat in patterns:
        m = re.search(pat, txt, flags=re.I)
        if not m:
            continue

        first = normalize_person_name(m.group(1) or "")
        last = normalize_person_name(m.group(2) or "")

        if not first:
            continue
        if first.lower() in stop_words:
            continue
        if last and last.lower() in stop_words:
            last = ""

        return first or None, last or None

    return None, None

def ensure_name_engine_defaults(profile: dict, sched: dict) -> None:
    profile.setdefault("recognized_first_name", None)
    profile.setdefault("recognized_last_name", None)
    profile.setdefault("recognized_email", None)
    profile.setdefault("active_first_name", None)
    profile.setdefault("active_last_name", None)
    profile.setdefault("active_email", None)
    profile.setdefault("voicemail_first_name", None)
    profile.setdefault("voicemail_last_name", None)
    profile.setdefault("known_people", [])
    profile.setdefault("identity_source", None)

    sched.setdefault("name_engine_state", None)
    sched.setdefault("name_engine_prompted", False)
    sched.setdefault("name_engine_candidate_first", None)
    sched.setdefault("name_engine_candidate_last", None)
    sched.setdefault("name_engine_expected_known_first", None)
    sched.setdefault("name_engine_selected_first", None)
    sched.setdefault("soft_rejection_state", None)
    sched.setdefault("soft_rejection_open", False)
    sched.setdefault("soft_rejection_ts", None)
    sched.setdefault("name_engine_branded", False)


def wrap_name_engine_message(sched: dict, message: str, first_name: str | None = None) -> str:
    msg = (message or "").strip()
    if not msg:
        return msg

    intro = f"Hello {first_name}, this is Prevolt Electric." if (first_name or "").strip() else "Hello, this is Prevolt Electric."
    context = "I'm following up on the service request from this number."

    # Name-engine messages should NEVER read like a cold identity verification text.
    # Always include business identity and service context in the message itself.
    sched["name_engine_branded"] = True
    sched["intro_sent"] = True

    full = f"{intro} {context} {msg}".strip()
    full = " ".join(full.split())
    return full

def get_known_people(profile: dict) -> list:
    ppl = profile.setdefault("known_people", [])
    if isinstance(ppl, list):
        return ppl
    profile["known_people"] = []
    return profile["known_people"]

def find_known_person_by_first(profile: dict, first_name: str) -> dict | None:
    target = (first_name or "").strip().lower()
    if not target:
        return None
    for p in get_known_people(profile):
        if (p.get("first_name") or "").strip().lower() == target:
            return p
    return None

def upsert_known_person(profile: dict, *, first_name: str, last_name: str, email: str, square_customer_id: str | None = None) -> None:
    first_name = normalize_person_name(first_name)
    last_name = normalize_person_name(last_name)
    email = (email or "").strip()
    if not first_name:
        return
    ppl = get_known_people(profile)
    for p in ppl:
        if (p.get("first_name") or "").strip().lower() == first_name.lower() and (p.get("last_name") or "").strip().lower() == (last_name or "").lower():
            if last_name:
                p["last_name"] = last_name
            if email:
                p["email"] = email
            if square_customer_id:
                p["square_customer_id"] = square_customer_id
            return
    ppl.append({
        "first_name": first_name,
        "last_name": last_name or "",
        "email": email or "",
        "square_customer_id": square_customer_id or "",
    })

def list_known_first_names(profile: dict) -> list[str]:
    names = []
    for p in get_known_people(profile):
        fn = normalize_person_name(p.get("first_name") or "")
        if fn and fn not in names:
            names.append(fn)
    return names

def normalize_short_reply(text: str) -> str:
    low = " ".join((text or "").strip().lower().split())
    low = re.sub(r"[^a-z0-9' ]+", "", low)
    low = " ".join(low.split())
    return low

def yes_text(text: str) -> bool:
    low = normalize_short_reply(text)
    if low in {"yes", "y", "yeah", "yep", "correct", "that is correct", "right", "it is", "it is correct", "thats right", "that's right"}:
        return True
    if low.startswith("yes ") and any(x in low for x in ["correct", "right"]):
        return True
    return False

def no_text(text: str) -> bool:
    low = normalize_short_reply(text)
    return low in {"no", "n", "nope", "not me", "wrong", "incorrect", "that is wrong", "that's wrong"}

def confirmation_accept_text(text: str) -> bool:
    low = normalize_short_reply(text)
    if not low:
        return False

    direct = {
        "yes", "y", "yeah", "yep", "sure", "ok", "okay", "correct",
        "that works", "sounds good", "book it", "schedule it", "lets do it",
        "let's do it", "do it", "go ahead", "yes please", "works for me"
    }
    if low in direct:
        return True

    yes_words = ["yes", "yeah", "yep", "correct", "right", "works", "good", "book", "schedule"]
    return any(w in low for w in yes_words)

def apply_known_person_to_active(profile: dict, person: dict, *, source: str) -> None:
    if not isinstance(person, dict):
        return
    fn = normalize_person_name(person.get("first_name") or "")
    ln = normalize_person_name(person.get("last_name") or "")
    em = (person.get("email") or "").strip()
    if fn:
        profile["active_first_name"] = fn
        profile["first_name"] = fn
    if ln:
        profile["active_last_name"] = ln
        profile["last_name"] = ln
    if em:
        profile["active_email"] = em
        profile["email"] = em
    profile["identity_source"] = source

def maybe_apply_name_engine_from_context(profile: dict, sched: dict, cleaned_transcript: str, initial_sms: str) -> str | None:
    ensure_name_engine_defaults(profile, sched)

    voicemail_first = profile.get("voicemail_first_name")
    voicemail_last = profile.get("voicemail_last_name")
    if not voicemail_first:
        first, last = extract_possible_person_name(cleaned_transcript or initial_sms or "")
        if first:
            profile["voicemail_first_name"] = first
            profile["voicemail_last_name"] = last
            voicemail_first = first
            voicemail_last = last

    known_names = list_known_first_names(profile)
    recognized_first = normalize_person_name(profile.get("recognized_first_name") or "")
    recognized_last = normalize_person_name(profile.get("recognized_last_name") or "")

    if not known_names and recognized_first:
        upsert_known_person(
            profile,
            first_name=recognized_first,
            last_name=recognized_last or "",
            email=profile.get("recognized_email") or "",
            square_customer_id=profile.get("square_customer_id") or None,
        )
        known_names = list_known_first_names(profile)

    if sched.get("name_engine_state"):
        return None

    # If voicemail clearly identified a person and that person is already known on this number,
    # use that caller as the active identity for this booking.
    if voicemail_first:
        known = find_known_person_by_first(profile, voicemail_first)
        if known:
            apply_known_person_to_active(profile, known, source="known_person_match")
            return None

        # Exactly one known person on file but voicemail named someone different -> clarify.
        if len(known_names) == 1:
            known_first = known_names[0]
            sched["name_engine_state"] = "awaiting_new_person_confirmation"
            sched["name_engine_candidate_first"] = voicemail_first
            sched["name_engine_candidate_last"] = voicemail_last
            sched["name_engine_expected_known_first"] = known_first
            return wrap_name_engine_message(
                sched,
                f"We worked with {known_first} on this number before, but the voicemail sounded like {voicemail_first}. Is {voicemail_first} the correct name for this visit?",
                voicemail_first,
            )

        # Multiple known people and voicemail gives a new first name -> clarify against known names.
        if len(known_names) >= 2:
            joined = ", ".join(known_names[:-1]) + f" or {known_names[-1]}" if len(known_names) > 1 else known_names[0]
            sched["name_engine_state"] = "awaiting_new_person_confirmation_multi"
            sched["name_engine_candidate_first"] = voicemail_first
            sched["name_engine_candidate_last"] = voicemail_last
            return wrap_name_engine_message(
                sched,
                f"I have {joined} on this number, but the voicemail sounded like {voicemail_first}. Is {voicemail_first} the correct name for this visit?",
                voicemail_first,
            )

        # No known people yet -> trust the explicit voicemail name.
        profile["active_first_name"] = voicemail_first
        profile["first_name"] = voicemail_first
        if voicemail_last:
            profile["active_last_name"] = voicemail_last
            profile["last_name"] = voicemail_last
            profile["identity_source"] = "voicemail_full_name"
        else:
            profile["identity_source"] = "voicemail_first_name"
        return None

    # No voicemail name was given.
    # If there is exactly one known person on file, use that person quietly.
    if len(known_names) == 1 and not get_active_first_name(profile):
        known = find_known_person_by_first(profile, known_names[0])
        if known:
            apply_known_person_to_active(profile, known, source="single_known_person_phone_match")
            return None

    # No voicemail name and multiple known people on the number -> ask who is calling.
    if len(known_names) >= 2 and not get_active_first_name(profile):
        joined = ", ".join(known_names[:-1]) + f" or {known_names[-1]}" if len(known_names) > 1 else known_names[0]
        sched["name_engine_state"] = "awaiting_known_person_selection"
        return wrap_name_engine_message(sched, f"I have {joined} on this number. Which person is calling today?")

    return None

def handle_name_engine_response(conv: dict, inbound_text: str) -> str | None:
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    ensure_name_engine_defaults(profile, sched)
    state = sched.get("name_engine_state")
    low = (inbound_text or "").strip().lower()
    if not state:
        return None

    if state in {"awaiting_new_person_confirmation", "awaiting_new_person_confirmation_multi"}:
        candidate_first = normalize_person_name(sched.get("name_engine_candidate_first") or "")
        if yes_text(low):
            profile["active_first_name"] = candidate_first
            profile["first_name"] = candidate_first
            profile["identity_source"] = "new_person_confirmed_from_voicemail"
            sched["name_engine_state"] = "awaiting_new_person_last_name"
            return "Perfect. What is your last name?"
        if no_text(low):
            known_names = list_known_first_names(profile)
            if known_names:
                joined = ", ".join(known_names[:-1]) + f" or {known_names[-1]}" if len(known_names) > 1 else known_names[0]
                sched["name_engine_state"] = "awaiting_known_person_selection"
                return wrap_name_engine_message(sched, f"No problem. Which person is calling today, {joined}?")
            sched["name_engine_state"] = "awaiting_manual_first_name"
            return wrap_name_engine_message(sched, "No problem. What first name should I use today?")
        return wrap_name_engine_message(sched, f"I heard the name {candidate_first} in the voicemail. Did I get that right?", candidate_first)

    if state == "awaiting_known_person_selection":
        for p in get_known_people(profile):
            fn = normalize_person_name(p.get("first_name") or "")
            if fn and fn.lower() in low:
                apply_known_person_to_active(profile, p, source="known_person_selection")
                sched["name_engine_state"] = None
                return None
        return wrap_name_engine_message(sched, "Which first name should I use today?")

    if state == "awaiting_manual_first_name":
        first, last = extract_possible_person_name(inbound_text)
        if not first:
            cleaned = re.sub(r"[^A-Za-z'\- ]", " ", inbound_text).strip()
            first = normalize_person_name(cleaned.split()[0]) if cleaned.split() else ""
        if first:
            profile["active_first_name"] = first
            profile["first_name"] = first
            profile["identity_source"] = "manual_first_name"
            if last:
                profile["active_last_name"] = last
                profile["last_name"] = last
                sched["name_engine_state"] = "awaiting_new_person_email"
                return "What is the best email address for the appointment?"
            sched["name_engine_state"] = "awaiting_new_person_last_name"
            return wrap_name_engine_message(sched, "What is your last name?")
        return wrap_name_engine_message(sched, "What first name should I use today?")

    if state == "awaiting_new_person_last_name":
        if is_not_a_person_name_reply(inbound_text):
            return wrap_name_engine_message(sched, "What is your last name?")
        cleaned = re.sub(r"[^A-Za-z'\- ]", " ", inbound_text).strip()
        parts = cleaned.split()
        if parts:
            last = normalize_person_name(" ".join(parts[-2:]) if len(parts) > 1 and parts[0].lower() in {"my", "is"} else parts[-1])
            if last:
                profile["active_last_name"] = last
                profile["last_name"] = last
                sched["name_engine_state"] = "awaiting_new_person_email"
                return "What is the best email address for the appointment?"
        return wrap_name_engine_message(sched, "What is your last name?")

    if state == "awaiting_new_person_email":
        m = re.search(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", inbound_text or "", flags=re.I)
        if m:
            email = m.group(1).strip()
            profile["active_email"] = email
            profile["email"] = email
            sched["name_engine_state"] = None
            sched["pending_step"] = None
            upsert_known_person(
                profile,
                first_name=get_active_first_name(profile),
                last_name=get_active_last_name(profile),
                email=get_active_email(profile),
                square_customer_id=None,
            )
            return None
        return "What is the best email address for the appointment?"

    return None


ADDRESS_WORD_RE = re.compile(r"\b(?:st|street|ave|avenue|rd|road|ln|lane|dr|drive|court|ct|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace)\b", flags=re.I)
QUESTIONISH_RE = re.compile(r"\b(?:are|is|do|does|did|can|could|would|will|why|what|when|where|who|how)\b", flags=re.I)


def is_plausible_address_text(text: str) -> bool:
    txt = (text or "").strip()
    if not txt:
        return False
    low = txt.lower()
    banned = [
        "licensed", "insured", "insurance", "permit", "ai", "artificial", "favorite color",
        "how much", "price", "cost", "what does", "include", "where are you", "who are you",
        "do you", "are you", "can you", "would you", "will you"
    ]
    if any(b in low for b in banned):
        return False
    if "?" in txt and not re.search(r"\b\d{1,6}\b", txt):
        return False
    if re.search(r"\b\d{1,6}\b", txt) and ADDRESS_WORD_RE.search(low):
        return True
    if "," in txt and ADDRESS_WORD_RE.search(low):
        return True
    if ADDRESS_WORD_RE.search(low) and len(txt.split()) <= 6 and not QUESTIONISH_RE.search(low):
        return True
    return False


def is_conversation_hesitation(text: str) -> bool:
    low = _loose_text(text)
    markers = [
        "not sure", "nervous", "hesitant", "hesitation", "is this ai", "are you ai", "real person",
        "licensed", "insured", "insurance", "permit", "where are you located", "where are you guys located",
        "where are you based", "where are you out of", "do you service", "do you work in", "trust",
        "not ready", "want to think", "shopping around", "just looking", "want to feel good", "not going to book"
    ]
    return any(m in low for m in markers)


def _question_count(text: str) -> int:
    t = (text or "")
    c = t.count("?")
    if c:
        return c
    low = _loose_text(t)
    return 1 if QUESTIONISH_RE.search(low) else 0


def hybrid_rule_excerpt(low: str) -> str:
    rb = RULE_MATRIX_TEXT or ""
    chunks = []
    key_groups = [
        (["licensed", "insured", "insurance", "where are you", "service", "permit", "ai"], "SRB-16"),
        (["price", "cost", "195", "395", "estimate", "quote"], "SRB-5"),
        (["safe", "danger", "shut", "breaker", "sparks", "burning", "smoke"], "SRB-2"),
        (["address", "street", "town", "ct", "ma"], "SRB-3"),
        (["not sure", "book", "hesitant", "think about it", "shopping around"], "SRB-11"),
        (["question", "?"], "Rule 20."),
    ]
    for needles, anchor in key_groups:
        if any(n in low for n in needles) and anchor in rb:
            i = rb.find(anchor)
            if i >= 0:
                chunks.append(rb[i:i+2200])
    if not chunks:
        return rb[:2200]
    return "\n\n".join(chunks[:3])


def llm_trust_reply(conv: dict, inbound_text: str) -> dict | None:
    low = _loose_text(inbound_text)
    if not low:
        return None
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    if not (_question_count(inbound_text) or is_conversation_hesitation(inbound_text)):
        return None

    next_prompt = choose_next_prompt_from_state(conv, inbound_text="")
    if next_prompt in {"Okay.", "Okay", "Everything looks good here."}:
        next_prompt = ""

    system = f"""
You are writing one SMS reply for Prevolt Electric.
Return strict JSON with keys:
{{
  "answer": string,
  "resume_booking": boolean,
  "next_prompt": string,
  "address": string|null
}}

Priorities:
- answer the customer's actual question naturally
- protect Prevolt's interests
- keep trust high
- never sound robotic
- never invent license numbers, policy numbers, addresses, permits, or legal promises
- never restate price unless the user is actively asking about price in this message
- if the customer is still deciding, do NOT force the booking flow
- if the customer is cooperative and the trust question is answered, you may gently return to the next missing booking step
- never turn the customer's question into an address or rewrite the question as an address
- do not output paragraphs
- one clean text

Known state:
- customer name: {(profile.get('active_first_name') or profile.get('first_name') or '').strip()}
- stored address: {(sched.get('raw_address') or '').strip()}
- pending step: {sched.get('pending_step')}
- price already disclosed: {bool(sched.get('price_disclosed'))}
- next_prompt_if_resuming: {next_prompt}

Use these rules as the source of truth:
{hybrid_rule_excerpt(low)}
"""
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": inbound_text},
            ],
        )
        data = json.loads(completion.choices[0].message.content)
        answer = " ".join(str(data.get("answer") or "").split()).strip()
        if not answer:
            return None
        resume = bool(data.get("resume_booking"))
        np = " ".join(str(data.get("next_prompt") or "").split()).strip()
        out_addr = data.get("address")
        if isinstance(out_addr, str) and not is_plausible_address_text(out_addr):
            out_addr = None
        return {"answer": answer, "resume_booking": resume, "next_prompt": np, "address": out_addr}
    except Exception as e:
        print("[WARN] llm_trust_reply failed:", repr(e))
        return None

def interruption_answer_and_return_prompt(conv: dict, inbound_text: str, *, allow_post_booking: bool = False) -> str | None:
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    low = _loose_text(inbound_text)
    if not low:
        return None

    booked = bool(sched.get("booking_created") and sched.get("square_booking_id"))
    if booked and not allow_post_booking:
        return None

    if v13_looks_like_real_person_question(inbound_text):
        return v13_real_person_reply(conv)

    # Let the LLM handle trust / hesitation / multi-question turns first.
    trust = llm_trust_reply(conv, inbound_text)
    if trust and trust.get("answer"):
        answer = trust["answer"].strip()
        resume = bool(trust.get("resume_booking"))
        next_prompt = (trust.get("next_prompt") or "").strip()
        if booked or sched.get("soft_rejection_open") or is_conversation_hesitation(inbound_text):
            resume = False
        if resume and next_prompt and next_prompt not in {"Okay.", "Okay", answer, "Everything looks good here."}:
            return f"{answer} {next_prompt}"
        return answer

    answer = None
    if any(x in low for x in ["dog", "dogs", "pet", "pets"]):
        answer = "Yes, that is fine. Just make sure we can safely get to the panel when we arrive."
    elif any(x in low for x in ["licensed", "license", "insured", "insurance"]):
        if "copy" in low or "certificate" in low or "proof" in low:
            answer = "We can provide insurance documentation when the work is moving forward."
        else:
            answer = "Yes, we're licensed and insured."
    elif any(x in low for x in ["where are you located", "where are you guys located", "where are you based", "where are you out of"]):
        answer = "We service Connecticut and Massachusetts."
    elif any(x in low for x in ["call when", "text when", "on the way", "arrival window", "when close", "when you re close", "when you're close", "when youre close"]):
        answer = "Yes, you'll get a text when we're on the way."
    elif any(x in low for x in ["do i need to buy", "bring anything", "materials", "should i buy", "do i need anything"]):
        answer = "No, you do not need to buy anything ahead of time."
    elif any(x in low for x in ["permit", "permit required"]):
        answer = "If anything needs a permit, we'll go over that during the visit."
    elif any(x in low for x in ["card", "cash", "check", "payment", "pay by", "how do i pay"]):
        answer = "Card or cash after the work is fine."
    elif any(x in low for x in [
        "how much", "price", "cost", "$195", "$395", "195", "395",
        "just to come out", "just to come", "service fee", "trip fee", "diagnostic fee",
        "quote", "estimate", "free estimate", "ballpark", "rough price", "firm number",
        "what do you charge", "what does the visit include", "what does that include",
        "what does that cover", "what does this cover", "what does the 195 cover", "what does $195 cover", "what is covered",
        "go towards the project", "go toward the project", "goes towards the project", "goes toward the project",
        "apply to the project", "applied to the project", "credit toward the project", "credited toward the project",
        "does it go toward", "does it go towards", "does that go toward", "does that go towards",
        "does the fee go toward", "does the fee go towards", "does the service fee go toward", "does the service fee go towards",
        "deposit", "credited back", "applied back"
    ]):
        appt = (sched.get("appointment_type") or "").upper()
        if any(x in low for x in [
            "go towards the project", "go toward the project", "goes towards the project", "goes toward the project",
            "apply to the project", "applied to the project", "credit toward the project", "credited toward the project",
            "does it go toward", "does it go towards", "does that go toward", "does that go towards",
            "does the fee go toward", "does the fee go towards", "does the service fee go toward", "does the service fee go towards",
            "deposit", "credited back", "applied back"
        ]):
            if "TROUBLESHOOT" in appt:
                answer = "The $395 covers the troubleshoot and repair visit itself. If larger repair work is needed, we go over that separately on site first."
            elif "INSPECTION" in appt:
                answer = "The inspection fee covers the inspection visit itself. If you need additional work after that, we would go over it separately."
            else:
                answer = "Not as a separate credit. The $195 covers our time and fuel to send one of our electricians out, look everything over, and get you a written quote."
        elif any(x in low for x in [
            "what does that cover", "what does this cover", "what does the 195 cover", "what does $195 cover",
            "what is covered", "what does the visit include", "what does that include", "what does the evaluation include"
        ]):
            if "TROUBLESHOOT" in appt:
                answer = "The $395 covers the troubleshoot and repair visit itself. If larger repair work is needed, we go over that separately on site first."
            elif "INSPECTION" in appt:
                answer = "The inspection fee covers sending one of our electricians out to review the home and go over the next step with you."
            else:
                answer = "It covers sending one of our electricians out, reviewing the work in person, going over the next step with you, and putting together a quote."
        elif any(x in low for x in ["quote", "estimate", "free estimate", "ballpark", "firm number", "rough price"]):
            answer = "We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step."
        elif "TROUBLESHOOT" in appt:
            answer = "The $395 is the troubleshoot and repair visit to come out, diagnose the issue, and handle minor repairs if it makes sense on site."
        elif "INSPECTION" in appt:
            answer = "Whole-home inspections are $395, and larger homes can run higher depending on square footage."
        else:
            answer = "We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step."
        sched["price_disclosed"] = True
    elif any(x in low for x in ["availability", "available", "openings", "how soon", "come sooner", "earliest", "soonest", "when can you come", "when can you come out"]):
        # Availability questions should never fall into a vague canned response.
        # Offer concrete openings if possible; otherwise ask for a weekday/time window.
        try:
            slot_options = get_next_available_slots((sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195"), limit=3)
        except Exception:
            slot_options = []
        if slot_options:
            sched["awaiting_slot_offer_choice"] = True
            sched["offered_slot_options"] = slot_options[:3]
            answer = format_next_available_slots_message(slot_options)
            sched["last_slot_unavailable_message"] = answer
        else:
            answer = "What weekday and time work best for you?"
    elif any(x in low for x in ["how long", "visit take", "how long does it take", "how long is the visit"]):
        answer = "Most visits are about an hour, depending on what you have going on."
    elif any(x in low for x in ["panel upgrade", "do you do panel", "service change", "panel replacement"]):
        answer = "Yes, we handle panel upgrades and replacements."
    elif "favorite color" in low:
        answer = "Let's keep it on the visit details."
    elif any(x in low for x in ["roll me over", "pick me up", "carry me"]):
        answer = "We handle the electrical work itself. If someone needs to help with access, just make sure that is covered when we come out."

    if not answer:
        return None

    recompute_pending_step(profile, sched)

    if booked:
        return answer

    if sched.get("soft_rejection_open") or is_conversation_hesitation(inbound_text):
        return answer

    if sched.get("final_confirmation_sent") or sched.get("final_confirmation_accepted"):
        if not sched.get("pending_step") and sched.get("scheduled_date") and sched.get("scheduled_time"):
            return answer

    next_prompt = choose_next_prompt_from_state(conv, inbound_text="")
    if next_prompt and next_prompt not in {"Okay.", "Okay", answer, "Everything looks good here."}:
        return f"{answer} {next_prompt}"
    return answer

def looks_like_slot_payload(inbound_text: str) -> bool:
    txt = (inbound_text or "").strip()
    low = _loose_text(txt)
    if not txt:
        return False
    if "?" in txt or QUESTIONISH_RE.search(low):
        # Questions are not slot payloads unless they also contain a clear address/date/time/email.
        pass
    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", txt, flags=re.I):
        return True
    if is_plausible_address_text(txt):
        return True
    if extract_explicit_time_from_text(txt):
        return True
    if salvage_relative_date_from_text(txt):
        return True
    # Only treat a very short alpha-only reply as slot data when it looks like a name, not a question.
    if len(txt.split()) <= 3 and re.fullmatch(r"[A-Za-z'\- ]+", txt) and not QUESTIONISH_RE.search(low):
        return True
    return False


def should_short_circuit_interrupt(conv: dict, inbound_text: str) -> bool:
    sched = conv.setdefault("sched", {})
    if sched.get("booking_created") and sched.get("square_booking_id"):
        return False
    low = _loose_text(inbound_text)
    if not low:
        return False
    if is_next_available_request(inbound_text):
        return False
    if _question_count(inbound_text) or is_conversation_hesitation(inbound_text):
        return True
    if looks_like_slot_payload(inbound_text):
        return False
    interrupt_markers = [
        "how much", "price", "cost", "$195", "$395", "195", "395", "just to come out", "just to come",
        "quote", "estimate", "free estimate", "ballpark", "firm number", "rough price",
        "go towards the project", "go toward the project", "goes towards the project", "goes toward the project",
        "apply to the project", "applied to the project", "credit toward the project", "credited toward the project",
        "does it go toward", "does it go towards", "does that go toward", "does that go towards",
        "does the fee go toward", "does the fee go towards", "does the service fee go toward", "does the service fee go towards",
        "deposit", "credited back", "applied back",
        "licensed", "license", "insured", "insurance", "where are you", "where are you located", "where are you guys located",
        "permit", "permit required", "dog", "dogs", "pet", "pets", "call when",
        "text when", "on the way", "arrival window", "when close", "when you're close", "when youre close",
        "bring anything", "materials", "do i need to buy", "do you do panel", "panel upgrade",
        "availability", "available", "how soon", "come sooner", "earliest", "soonest", "when can you come", "when can you come out",
        "how long does it take", "how long is the visit", "what does the visit include", "what does that include",
        "favorite color", "is this ai", "are you ai", "real person", "talk to my husband", "talk to my wife", "copy of your insurance"
    ]
    return any(x in low for x in interrupt_markers)

def _loose_text(s: str) -> str:
    s = (s or "").lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def parse_ordinal_weekday_of_month_date(text: str, today=None) -> str | None:
    """
    Parse customer date requests such as:
      - first Monday of July
      - 1st Monday in July
      - second Tuesday of June 2026
      - last Friday of August

    This must run before generic weekday parsing so "first Monday of July"
    does not collapse to the next upcoming Monday or the first offered slot.
    """
    low = _loose_text(text)
    if not low:
        return None

    if today is None:
        now_local = datetime.now(ZoneInfo("America/New_York")) if ZoneInfo else datetime.now()
        today = now_local.date()

    month_names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    ordinals = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
        "last": -1,
    }

    pattern = re.compile(
        r"\b(?:the\s+)?(?P<ord>first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last)\s+"
        r"(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
        r"(?:(?:of|in)\s+)?"
        r"(?P<month>jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
        r"(?:\s+(?P<year>\d{2,4}))?\b",
        flags=re.I,
    )
    m = pattern.search(low)
    if not m:
        return None

    ordinal = ordinals.get(m.group("ord").lower())
    weekday_idx = weekdays.get(m.group("weekday").lower())
    month = month_names.get(m.group("month").lower())
    yr_raw = m.group("year")
    if ordinal is None or weekday_idx is None or month is None:
        return None

    year = today.year
    if yr_raw:
        year = int(yr_raw)
        if year < 100:
            year += 2000

    try:
        if ordinal == -1:
            if month == 12:
                last_day = (datetime(year + 1, 1, 1).date() - timedelta(days=1))
            else:
                last_day = (datetime(year, month + 1, 1).date() - timedelta(days=1))
            delta = (last_day.weekday() - weekday_idx) % 7
            target = last_day - timedelta(days=delta)
        else:
            first_day = datetime(year, month, 1).date()
            delta = (weekday_idx - first_day.weekday()) % 7
            target = first_day + timedelta(days=delta + (ordinal - 1) * 7)
            if target.month != month:
                return None

        if not yr_raw and target < today:
            # Customer did not specify a year and that ordinal date already passed this year.
            return parse_ordinal_weekday_of_month_date(f"{m.group('ord')} {m.group('weekday')} {m.group('month')} {today.year + 1}", today=today)
        return target.strftime("%Y-%m-%d")
    except Exception:
        return None


def _date_label_for_sms(date_str: str) -> str:
    try:
        return _humanize_date_for_sms(date_str)
    except Exception:
        try:
            d = datetime.strptime(str(date_str or ""), "%Y-%m-%d")
            return d.strftime("%A, %B %d").replace(" 0", " ")
        except Exception:
            return str(date_str or "that day")


def maybe_handle_new_date_during_offered_slots(conv: dict, inbound_text: str) -> str | None:
    """
    If the customer is looking at offered fallback slots but asks for a different
    date, respect the new date instead of treating words like "first" as
    "first option".

    Example failure this prevents:
      Offered: Tue May 12 2 PM, Wed May 13 3 PM, Thu May 14 4 PM
      Customer: "Can we do the first Monday of July?"
      Bad old behavior: selects the first offered May slot.
      Correct behavior: stores 2026-07-06 and asks for the time.
    """
    sched = conv.setdefault("sched", {})
    if not sched.get("awaiting_slot_offer_choice"):
        return None

    options = sched.get("offered_slot_options") or []
    if not options:
        return None

    requested_date = salvage_relative_date_from_text(inbound_text)
    if not requested_date:
        return None

    option_dates = {str(opt.get("date") or "").strip() for opt in options if opt.get("date")}
    # If the requested date is one of the offered choices, let normal offered-slot
    # selection handle it so "Tuesday works" still selects the Tuesday option.
    if requested_date in option_dates:
        return None

    explicit_time = extract_explicit_time_from_text(inbound_text)
    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195").upper()
    sched["appointment_type"] = appt
    conv["appointment_type"] = appt
    sched["scheduled_date"] = requested_date
    sched["scheduled_time"] = explicit_time if explicit_time else None
    if explicit_time:
        sched["scheduled_time_source"] = "customer_explicit_new_date_after_offer"
    else:
        sched.pop("scheduled_time_source", None)
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    sched["final_confirmation_sent"] = False
    sched["final_confirmation_accepted"] = False
    sched["last_final_confirmation_key"] = None
    sched["slot_choice_locked"] = False
    sched["booking_attempt_nonce"] = str(uuid.uuid4())

    if explicit_time:
        # Let maybe_handle_exact_slot_before_step4 validate Square availability later in the route.
        return None

    if "TROUBLESHOOT" not in appt and is_weekend(requested_date):
        sched["pending_step"] = "need_date"
        return "We schedule non-emergency visits Monday through Friday. What weekday and time work better?"

    sched["pending_step"] = "need_time"
    return f"Got it — what time works for {_date_label_for_sms(requested_date)}?"


def salvage_relative_date_from_text(inbound_text: str) -> str | None:
    low = _loose_text(inbound_text)
    if not low:
        return None
    now_local = datetime.now(ZoneInfo("America/New_York")) if ZoneInfo else datetime.now()
    today = now_local.date()

    ordinal_weekday_date = parse_ordinal_weekday_of_month_date(low, today=today)
    if ordinal_weekday_date:
        return ordinal_weekday_date

    if "today" in low:
        return today.strftime("%Y-%m-%d")
    if "tomorrow" in low:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for name, idx in weekdays.items():
        if f"next {name}" in low:
            delta = (idx - today.weekday()) % 7
            if delta == 0:
                delta = 7
            delta += 7
            return (today + timedelta(days=delta)).strftime("%Y-%m-%d")
        if re.search(rf"\b(?:this )?{name}\b", low):
            delta = (idx - today.weekday()) % 7
            return (today + timedelta(days=delta)).strftime("%Y-%m-%d")

    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", low)
    if m:
        mo = int(m.group(1))
        day = int(m.group(2))
        yr_raw = m.group(3)
        year = today.year
        if yr_raw:
            year = int(yr_raw)
            if year < 100:
                year += 2000
        try:
            dt = datetime(year, mo, day)
            if not yr_raw and dt.date() < today:
                dt = datetime(year + 1, mo, day)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    month_names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    m = re.search(r"\b(" + "|".join(month_names.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{2,4}))?\b", low)
    if m:
        mo = month_names[m.group(1)]
        day = int(m.group(2))
        yr_raw = m.group(3)
        year = today.year
        if yr_raw:
            year = int(yr_raw)
            if year < 100:
                year += 2000
        try:
            dt = datetime(year, mo, day)
            if not yr_raw and dt.date() < today:
                dt = datetime(year + 1, mo, day)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    return None


def absorb_obvious_booking_details(conv: dict, inbound_text: str) -> None:
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    txt = (inbound_text or "").strip()
    if not txt:
        return

    if not sched.get("scheduled_date"):
        d = salvage_relative_date_from_text(txt)
        if d:
            sched["scheduled_date"] = d

    if not sched.get("scheduled_time"):
        t = extract_explicit_time_from_text(txt)
        if t:
            sched["scheduled_time"] = t
            sched["scheduled_time_source"] = "customer_explicit"

    if not sched.get("raw_address"):
        extracted_addr = extract_service_address_from_text(txt)
        if extracted_addr:
            sched["raw_address"] = extracted_addr
            try:
                if extracted_addr not in profile.setdefault("addresses", []):
                    profile["addresses"].append(extracted_addr)
            except Exception:
                pass
            try:
                try_early_address_normalize(sched)
            except Exception:
                pass

    update_address_assembly_state(sched)
    recompute_pending_step(profile, sched)


def detect_soft_rejection(inbound_text: str) -> str | None:
    low = _loose_text(inbound_text)
    if not low:
        return None

    if (("wife" in low or "husband" in low or "spouse" in low) and any(x in low for x in ["talk", "tell", "check", "ask", "run it by", "first"])):
        return "spouse"
    if any(p in low for p in ["call around", "shop around", "check around", "come sooner", "find someone sooner", "compare prices", "get a few quotes", "get other quotes"]):
        return "shopping"
    if any(p in low for p in ["not ready yet", "maybe later", "let me think about it", "ill let you know", "i ll let you know", "i will get back to you", "i ll get back to you", "not sure what day", "not sure ill be free", "not sure i will be free", "have to check", "need to check"]):
        return "timing"
    return None


def build_soft_rejection_reply(conv: dict, inbound_text: str) -> str | None:
    sched = conv.setdefault("sched", {})
    kind = detect_soft_rejection(inbound_text)
    if not kind:
        return None

    sched["soft_rejection_state"] = kind
    sched["soft_rejection_open"] = True
    sched["soft_rejection_ts"] = time.time()

    if kind == "spouse":
        return "No problem at all. Talk it over and if you want to move forward just message me back here and I’ll pick it up where we left off."
    if kind == "shopping":
        return "No problem at all. If you want to move forward later just message me back here and I’ll pick it up where we left off."
    return "No problem at all. When you know what day works, just message me back here and I'll pick it up where we left off."


def maybe_resume_paused_conversation(conv: dict, inbound_text: str) -> str | None:
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    if not sched.get("soft_rejection_open"):
        return None

    low = _loose_text(inbound_text)
    if not low:
        return None

    # While paused, answer questions without forcing the booking flow back open.
    if _question_count(inbound_text) or is_conversation_hesitation(inbound_text):
        answered = interruption_answer_and_return_prompt(conv, inbound_text)
        if answered:
            return answered
        trust = llm_trust_reply(conv, inbound_text)
        if trust and trust.get("answer"):
            return trust.get("answer").strip()

    resume_markers = [
        "okay", "ok", "yes", "yeah", "yep", "that works", "lets do it", "let's do it",
        "book it", "go ahead", "move forward", "i'm ready", "im ready", "can we schedule",
        "schedule it", "tomorrow", "today", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "its a go", "it's a go", "wife says", "husband says",
        "spouse says", "when can you come", "when can you come out", "ready to book", "lets book", "let's book"
    ]
    if looks_like_slot_payload(inbound_text) or any(x in low for x in resume_markers):
        sched["soft_rejection_open"] = False
        absorb_obvious_booking_details(conv, inbound_text)
        recompute_pending_step(profile, sched)
        next_prompt = choose_next_prompt_from_state(conv, inbound_text=inbound_text)
        if next_prompt and next_prompt not in {"Okay.", "Okay", "Everything looks good here."}:
            return f"Sounds good. {next_prompt}"
        return "Sounds good."
    return None


def handle_post_booking_question(conv: dict, inbound_text: str) -> str | None:
    sched = conv.setdefault("sched", {})
    if not (sched.get("booking_created") and sched.get("square_booking_id")):
        return None

    answered = interruption_answer_and_return_prompt(conv, inbound_text, allow_post_booking=True)
    if answered:
        return answered

    low = (inbound_text or "").lower().strip()
    if any(x in low for x in ["what day", "what time", "when am i on", "when are you coming", "confirm", "appointment time"]):
        try:
            booked_dt = datetime.strptime(sched.get("scheduled_date") or "", "%Y-%m-%d")
            human_day = booked_dt.strftime("%A, %B %d").replace(" 0", " ")
        except Exception:
            human_day = (sched.get("scheduled_date") or "").strip() or "the scheduled day"
        human_t = humanize_time(sched.get("scheduled_time") or "") or "the scheduled time"
        return f"You’re set for {human_day} at {human_t}."

    return None

def looks_like_new_booking_request(inbound_text: str) -> bool:
    low = (inbound_text or "").lower().strip()
    if not low:
        return False

    restart_keywords = [
        "reschedule", "change", "different", "another", "new appointment",
        "move it", "push it", "cancel", "need a new time", "need a new day",
        "book", "schedule", "come out", "come by"
    ]
    if any(k in low for k in restart_keywords):
        return True

    if any(w in low for w in [
        "tomorrow", "today", "next monday", "next tuesday", "next wednesday",
        "next thursday", "next friday", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday"
    ]):
        return True

    if re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", low, flags=re.I):
        return True

    if re.search(
        r"\b\d{1,6}\b.*\b(st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace)\b",
        low,
        flags=re.I
    ):
        return True

    return False

def handle_post_booking(conv: dict, inbound_text: str) -> str | None:
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    appt = profile.get("upcoming_appointment") or {}

    if not (sched.get("booking_created") and sched.get("square_booking_id")):
        return None

    low = (inbound_text or "").lower().strip()
    if not low:
        return None

    # Let hazards break out of post-booking mode immediately.
    emergency_words = [
        "sparking", "burning", "smoke", "water pouring", "water in panel",
        "panel has water", "no power", "arcing", "buzzing", "hot panel",
        "fire", "melted", "shocked", "shock", "tree fell", "service ripped"
    ]
    if any(x in low for x in emergency_words):
        return None

    if looks_like_new_booking_request(inbound_text):
        sched["booking_created"] = False
        sched["square_booking_id"] = None
        sched["scheduled_date"] = None
        sched["scheduled_time"] = None
        sched["raw_address"] = None
        sched["normalized_address"] = None
        sched["address_candidate"] = None
        sched["address_verified"] = False
        sched["address_missing"] = None
        sched["address_parts"] = {}
        sched["pending_step"] = None
        return None

    if any(x in low for x in ["dog", "dogs", "pet", "pets"]):
        return "Yes, please make sure we can safely get to the panel when we arrive."

    if any(x in low for x in ["gate", "code", "lock", "locked", "access", "doorbell", "call when outside"]):
        return "That's fine. Just make sure we can get to the panel when we arrive."

    if any(x in low for x in ["what time", "when are you coming", "what day", "when is my appointment", "are we good"]):
        date_txt = (appt.get("date") or sched.get("scheduled_date") or "").strip()
        time_txt = humanize_time(appt.get("time") or sched.get("scheduled_time") or "")
        if date_txt and time_txt:
            return f"You're all set for {date_txt} at {time_txt}."
        if date_txt:
            return f"You're all set for {date_txt}."
        return "You're all set."

    if any(x in low for x in ["who is coming", "who's coming", "whos coming", "on the way", "arrival window", "will they call"]):
        return "You'll get a text when we're on the way."

    if any(x in low for x in ["price", "how much", "cost"]):
        return "You're all set."

    if any(x in low for x in ["address", "coming to", "where are you going"]):
        return "We've got the address already attached to the visit."

    if any(x in low for x in ["thanks", "thank you", "ok", "okay", "got it", "perfect"]):
        return ""

    return "You're all set."

# ---------------------------------------------------
# Step 4 — Generate Replies (Hybrid Logic + Deterministic State Machine)
# PATCHED (HUMAN + HOUSE# + NO WAIT-TEXT + ACK MEMORY)
#   - Longer, more human intros + prompts (deterministic per conversation)
#   - No em dash "—" and no " - " telltales
#   - No "Got it", "Thanks" filler
#   - No "one moment / securing" messages (ever)
#   - House-number patch: handles
#   - Re-check address state AFTER model output (prevents booking without house number)
#   - If Square did NOT book, never "confirm"; always ask the next missing atom (or send a neutral line)
#   - Adds acknowledgement memory so it doesn't repeat acknowledgements
# ---------------------------------------------------
def generate_reply_for_inbound(
    cleaned_transcript,
    category,
    appointment_type,
    initial_sms,
    inbound_text,
    scheduled_date,
    scheduled_time,
    address,
) -> dict:

    try:
        import re
        import hashlib
        from datetime import datetime, timedelta

        # --------------------------------------
        # Timezone setup
        # --------------------------------------
        try:
            tz = ZoneInfo("America/New_York")
        except Exception:
            tz = timezone(timedelta(hours=-5))

        now_local      = datetime.now(tz)
        today_date_str = now_local.strftime("%Y-%m-%d")
        today_weekday  = now_local.strftime("%A")

        # --------------------------------------
        # Conversation + scheduler layers (HARDENED)
        # --------------------------------------
        phone = request.form.get("From", "").replace("whatsapp:", "")
        convo_key = phone or request.form.get("MessageSid") or request.form.get("SmsSid") or request.form.get("CallSid") or "unknown"
        conv  = conversations.setdefault(convo_key, {})

        profile = conv.setdefault("profile", {})
        profile.setdefault("addresses", [])
        profile.setdefault("past_jobs", [])
        profile.setdefault("upcoming_appointment", None)

        profile.setdefault("first_name", None)
        profile.setdefault("last_name", None)
        profile.setdefault("email", None)

        profile.setdefault("recognized_first_name", None)
        profile.setdefault("recognized_last_name", None)
        profile.setdefault("recognized_email", None)

        profile.setdefault("active_first_name", None)
        profile.setdefault("active_last_name", None)
        profile.setdefault("active_email", None)

        profile.setdefault("identity_source", None)
        profile.setdefault("square_customer_id", None)
        profile.setdefault("square_lookup_done", False)

        # One-time repeat-customer hydrate from Square by phone.
        # IMPORTANT: recognized identity is separate from active booking identity.
        if not profile.get("square_lookup_done"):
            try:
                cust = square_lookup_customer_by_phone(phone)
                if cust and cust.get("id"):
                    profile["square_customer_id"] = cust.get("id")

                    profile["recognized_first_name"] = cust.get("given_name")
                    profile["recognized_last_name"] = cust.get("family_name")
                    profile["recognized_email"] = cust.get("email_address")

                    if not (profile.get("active_first_name") and profile.get("active_last_name")):
                        if cust.get("given_name"):
                            profile["active_first_name"] = cust.get("given_name")
                        if cust.get("family_name"):
                            profile["active_last_name"] = cust.get("family_name")
                        profile["identity_source"] = profile.get("identity_source") or "square_phone_match"

                    if not profile.get("active_email") and cust.get("email_address"):
                        profile["active_email"] = cust.get("email_address")

                    if cust.get("given_name"):
                        upsert_known_person(
                            profile,
                            first_name=cust.get("given_name") or "",
                            last_name=cust.get("family_name") or "",
                            email=cust.get("email_address") or "",
                            square_customer_id=cust.get("id") or None,
                        )

                    caddr = cust.get("address") or {}
                    line1 = (caddr.get("address_line_1") or "").strip()
                    city = (caddr.get("locality") or "").strip()
                    state = (caddr.get("administrative_district_level_1") or "").strip()
                    zipc = (caddr.get("postal_code") or "").strip()
                    if line1 and city and state:
                        pretty = f"{line1}, {city}, {state} {zipc}".strip()
                        if pretty and pretty not in profile["addresses"]:
                            profile["addresses"].append(pretty)
            except Exception as e:
                print("[WARN] Square lookup hydrate failed:", repr(e))
            finally:
                profile["square_lookup_done"] = True

        def _best_saved_address(profile_obj: dict) -> str | None:
            for a in profile_obj.get("addresses") or []:
                a = (a or "").strip()
                if re.match(r"^\d{1,6}\b", a):
                    return a
            return None

        def _saved_address_for_town(profile_obj: dict, town_hint: str) -> str | None:
            town_hint = (town_hint or "").strip().lower()
            if not town_hint:
                return _best_saved_address(profile_obj)
            for a in profile_obj.get("addresses") or []:
                s = (a or "").strip()
                if s and re.match(r"^\d{1,6}\b", s) and town_hint in s.lower():
                    return s
            return _best_saved_address(profile_obj)

        def _task_topics_text() -> str:
            topics = conv.get("task_topics") or []
            topics = [str(t).strip() for t in topics if str(t).strip()]
            if not topics:
                return ""
            topics = topics[:4]
            if len(topics) == 1:
                return topics[0]
            if len(topics) == 2:
                return f"{topics[0]} and {topics[1]}"
            return ", ".join(topics[:-1]) + f", and {topics[-1]}"

        def _known_address_question() -> str | None:
            saved = _saved_address_for_town(profile, sched.get("raw_address") or "")
            if not saved:
                return None
            return f"I have {saved} on file. Is this for that address?"

        # Helpers to capture name/email from inbound text
        def _extract_email(txt: str) -> str | None:
            txt = (txt or "").strip()
            m = re.search(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", txt, flags=re.I)
            return m.group(1).strip() if m else None

        def _extract_first_last(txt: str) -> tuple[str | None, str | None]:
            txt = (txt or "").strip()
            # avoid treating address/phone/email as a name
            if any(ch.isdigit() for ch in txt):
                return None, None
            if "@" in txt:
                return None, None
            cleaned = re.sub(r"[^A-Za-z\-\'\s]", " ", txt)
            cleaned = " ".join(cleaned.split()).strip()
            if not cleaned:
                return None, None
            parts = cleaned.split(" ")
            if len(parts) < 2:
                return None, None
            first = parts[0]
            last = " ".join(parts[1:])
            return first, last



        sched = conv.setdefault("sched", {})

        # Core state
        sched.setdefault("raw_address", None)
        sched.setdefault("normalized_address", None)
        sched.setdefault("pending_step", None)
        sched.setdefault("intro_sent", False)
        sched.setdefault("price_disclosed", False)
        sched.setdefault("awaiting_emergency_confirm", False)
        sched.setdefault("emergency_approved", False)
        sched.setdefault("final_confirmation_sent", False)
        sched.setdefault("final_confirmation_accepted", False)
        sched.setdefault("last_final_confirmation_key", None)

        # Booking flags
        sched.setdefault("booking_created", False)
        sched.setdefault("square_booking_id", None)
        sched.setdefault("awaiting_slot_offer_choice", False)
        sched.setdefault("offered_slot_options", [])
        sched.setdefault("last_slot_unavailable_message", None)

        # Address atoms
        sched.setdefault("address_candidate", None)
        sched.setdefault("address_verified", False)
        sched.setdefault("address_missing", None)
        sched.setdefault("address_parts", {})

        # Ack memory
        sched.setdefault("last_ack_text", None)
        sched.setdefault("last_ack_ts", None)

        # Human phrasing memory
        sched.setdefault("prompt_variants", {})

        inbound_text  = (inbound_text or "").strip()
        inbound_lower = inbound_text.lower().strip()

        ensure_name_engine_defaults(profile, sched)
        name_engine_prompt = maybe_apply_name_engine_from_context(profile, sched, cleaned_transcript, initial_sms)

        # If the customer asks for a different date while viewing fallback options,
        # handle that before offered-slot selection can treat "first" as "first option".
        date_change_reply = maybe_handle_new_date_during_offered_slots(conv, inbound_text)
        if date_change_reply:
            return {
                "sms_body": date_change_reply,
                "scheduled_date": sched.get("scheduled_date"),
                "scheduled_time": sched.get("scheduled_time"),
                "address": sched.get("raw_address"),
                "booking_complete": False
            }

        # If the customer picks one of the offered fallback slots, lock it in before any LLM work.
        maybe_apply_offered_slot_selection(conv, inbound_text)
        v13_time_only_on_offered_slots(conv, inbound_text)
        email_guard_reply = v13_handle_email_before_slot(conv, inbound_text)
        if email_guard_reply:
            return {
                "sms_body": email_guard_reply,
                "scheduled_date": sched.get("scheduled_date"),
                "scheduled_time": sched.get("scheduled_time"),
                "address": sched.get("raw_address"),
                "booking_complete": False
            }

        # Hazard detection must override post-booking mode.
        EMERGENCY_KEYWORDS = [
            "tree fell", "tree down", "power line", "lines down",
            "service ripped", "service wire", "service wires", "service wire damage",
            "service wire down", "service wires down", "sparking", "burning", "fire",
            "smoke", "no power", "power outage", "water pouring",
            "water in panel", "panel has water", "arcing", "buzzing",
            "hot panel", "burning smell", "main breaker", "melted",
            "shocked", "shock"
        ]
        IS_EMERGENCY = (not v13_looks_like_smoke_detector_install_request(inbound_text, cleaned_transcript, initial_sms) and any(k in inbound_lower for k in EMERGENCY_KEYWORDS))

        # Name engine gets first shot before the normal scheduler.
        name_engine_reply = handle_name_engine_response(conv, inbound_text)
        if name_engine_reply is not None:
            return {
                "sms_body": name_engine_reply,
                "scheduled_date": sched.get("scheduled_date"),
                "scheduled_time": sched.get("scheduled_time"),
                "address": sched.get("raw_address"),
                "booking_complete": False
            }
        if name_engine_prompt:
            return {
                "sms_body": name_engine_prompt,
                "scheduled_date": sched.get("scheduled_date"),
                "scheduled_time": sched.get("scheduled_time"),
                "address": sched.get("raw_address"),
                "booking_complete": False
            }

        # --------------------------------------
        # Post-booking handling
        # --------------------------------------
        if not IS_EMERGENCY:
            post_booking_reply = handle_post_booking(conv, inbound_text)
            if post_booking_reply is not None:
                booking_created = bool(sched.get("booking_created") and sched.get("square_booking_id"))
                post_booking_reply = sanitize_sms_body(post_booking_reply, booking_created=booking_created)
                return {
                    "sms_body": post_booking_reply,
                    "scheduled_date": sched.get("scheduled_date"),
                    "scheduled_time": sched.get("scheduled_time"),
                    "address": sched.get("raw_address"),
                    "booking_complete": True
                }
        # Opportunistic capture.
        # IMPORTANT: customer-provided identity overrides Square-recognized identity for this booking.
        if (sched.get("pending_step") == "need_name" or "my name" in inbound_lower or inbound_lower.startswith("name is")) and not is_not_a_person_name_reply(inbound_text):
            fn, ln = _extract_first_last(inbound_text)
            if fn and ln:
                profile["active_first_name"] = fn
                profile["active_last_name"] = ln
                profile["first_name"] = fn
                profile["last_name"] = ln
                profile["identity_source"] = "customer_provided"

        if sched.get("pending_step") == "need_email" or "@" in inbound_text:
            em = _extract_email(inbound_text)
            if em:
                profile["active_email"] = em
                profile["email"] = em


        # --------------------------------------
        # Local helpers (so this block is self-contained)
        # --------------------------------------
        ACK_SET = {
            "ok", "okay", "k", "kk", "sure", "yep", "yeah", "yes",
            "thanks", "thank you", "thx", "got it", "done", "perfect"
        }

        def _norm(s: str) -> str:
            return " ".join((s or "").strip().split())

        def _is_ack_only(msg: str) -> bool:
            t = _norm(msg).lower()
            if not t:
                return False
            if t in ACK_SET:
                return True
            if len(t) <= 20 and any(w in t for w in ["ok", "okay", "thanks", "thank", "thx", "got it"]):
                return True
            return False

        def _stable_choice_key(label: str) -> str:
            cid = (
                str(sched.get("conversation_id") or "")
                or str(phone or "")
                or str(sched.get("raw_address") or "")
            ).strip()
            base = f"{cid}|{label}"
            return hashlib.sha256(base.encode("utf-8")).hexdigest()

        def pick_variant_once(*args) -> str:
            """
            Deterministically pick and persist a prompt variant.

            Supports both call styles:
              - pick_variant_once(label, options)              (uses outer 'sched')
              - pick_variant_once(sched_dict, label, options)  (uses provided dict)
            """
            if len(args) == 2:
                _sched = sched
                label, options = args
            elif len(args) == 3:
                _sched, label, options = args
            else:
                return ""

            if not options:
                return ""

            if not isinstance(_sched, dict):
                _sched = sched

            store = _sched.setdefault("prompt_variants", {})
            # If we've already chosen a variant for this label, reuse it (but only if still valid).
            if label in store and store[label] in options:
                return store[label]

            h = _stable_choice_key(str(label))
            idx = int(h[:8], 16) % len(options)
            chosen = options[idx]
            store[label] = chosen
            return chosen

        def _strip_ai_tells(s: str) -> str:
            s = _norm(s)

            # normalize dashes and remove spaced hyphen tell
            s = s.replace("—", ".").replace("–", ".")
            s = s.replace(" - ", " ")

            # remove leading filler openers
            s = re.sub(
                r"^(got it|thanks|thank you|sure|absolutely|no problem|ok|okay)\b[\s,.:;!-]*",
                "",
                s,
                flags=re.I
            )

            # remove filler that appears right after the intro sentence
            s = re.sub(
                r"^(hi[, ]+)?(hey[, ]+)?(you(?:'|’)ve reached|this is)\s+prevolt electric\.\s*"
                r"(got it|thanks|thank you|sure|absolutely|no problem|ok|okay)\b[\s,.:;!-]*",
                r"this is prevolt electric. ",
                s,
                flags=re.I
            )

            # remove wait-text (never allowed)
            s = re.sub(
                r"\b(one moment|one sec|one second|give me a moment|please wait|hang tight|securing your appointment)\b[\s,.:;!-]*",
                "",
                s,
                flags=re.I
            )

            # tighten punctuation spacing
            s = re.sub(r"\s+\.", ".", s)
            s = re.sub(r"\.\.+", ".", s)
            s = _norm(s)

            return s

        def _shorten_texty(s: str, max_chars: int = 260) -> str:
            s = _norm(s)

            # Do not truncate the controlled price + slot handoff. The slots are
            # the whole point of that message; earlier versions shortened the
            # message after the price sentence and accidentally removed the three
            # available times.
            low_s = _intent_text(s)
            if ("$195" in s or "$395" in s) and ("we have" in low_s or "i have" in low_s) and ("or is there a better" in low_s):
                return s

            if len(s) <= max_chars:
                return s
            parts = re.split(r"(?<=[.!?])\s+", s)
            s = " ".join(parts[:2]).strip()
            if len(s) > max_chars:
                s = s[:max_chars].rstrip()
                s = re.sub(r"[\s,;:]+$", "", s)
                s += "."
            return _norm(s)

        def build_human_intro_line() -> str:
            greet_name = get_display_first_name(profile)
            if sched.get("name_engine_state") == "awaiting_known_person_selection":
                greet_name = ""

            if greet_name:
                options = [
                    f"Hello {greet_name}, this is Prevolt Electric. I can help you right here by text.",
                    f"Hello {greet_name}, this is Prevolt Electric. Let's get this lined up.",
                    f"Hello {greet_name}, you've reached Prevolt Electric. I'll help you here by text.",
                ]
            else:
                options = [
                    "Hello, this is Prevolt Electric. I can help you right here by text.",
                    "Hello, this is Prevolt Electric. Let's get this lined up.",
                    "Hello, you've reached Prevolt Electric. I'll help you here by text.",
                ]
            return pick_variant_once(sched, "intro_line", options)

        def humanize_question(core_question: str) -> str:
            core_question = _norm(core_question)

            # Deterministic wrapper per question type so we do not sound random mid-thread.
            # Keep it subtle, not salesy.
            options = [
                core_question,
                f"Perfect. {core_question}",
                f"Sounds good. {core_question}",
                f"Alright. {core_question}",
            ]
            return pick_variant_once(f"qwrap::{core_question[:18].lower()}", options)

        def _apply_intro_once(s: str) -> str:
            s = _norm(s)
            if not sched.get("intro_sent"):
                intro = build_human_intro_line()
                # Do not inject extracted task topics into the opener by default.
                # It tends to sound robotic; keep scope in the monitor unless the customer asks.
                s = f"{intro} {s}".strip()
                sched["intro_sent"] = True
            return _norm(s)

        def _maybe_price_once(s: str, appt_type_local: str) -> str:
            # Price is controlled by Python so it cannot leak into address collection.
            # Best point: after the lead feels understood and the address is verified,
            # immediately before asking for a day/time or offering slots.
            s = _norm(s)
            appt_local = (appt_type_local or "").upper()

            # Emergency pricing is disclosed only at the emergency dispatch confirmation.
            if appt_local == "TROUBLESHOOT_395":
                return s

            allowed = bool(sched.get("address_verified")) and (message_is_scheduling_prompt(s) or customer_is_asking_price(inbound_text))

            if "$195" in s or "$395" in s:
                if allowed or customer_is_asking_price(inbound_text):
                    sched["price_disclosed"] = True
                    return s
                return _norm(strip_unapproved_price_language(s))

            if allowed and not sched.get("price_disclosed"):
                try:
                    s = apply_price_injection(appt_local, s, conv)
                    sched["price_disclosed"] = True
                except Exception:
                    pass
            return _norm(s)

        def _finalize_sms(s: str, appt_type_local: str, booking_created: bool) -> str:
            s = _apply_intro_once(s)
            s = _maybe_price_once(s, appt_type_local)

            s = _strip_ai_tells(s)
            s = _shorten_texty(s)

            # If user just sent ack-only, do NOT reply with ack-only again.
            if _is_ack_only(inbound_text):
                low = s.lower()
                looks_like_ack = (low in {"ok", "okay", "yep", "yeah", "yes"} or len(low) <= 6)

                if looks_like_ack:
                    update_address_assembly_state(sched)
                    if not sched.get("address_verified"):
                        s = build_address_prompt(sched)
                        s = _apply_intro_once(s)
                        s = _strip_ai_tells(s)
                        s = _shorten_texty(s)
                    elif not sched.get("scheduled_date") and not sched.get("scheduled_time"):
                        s = humanize_question("What day and time work best for you?")
                        s = _apply_intro_once(s)
                    elif not sched.get("scheduled_date"):
                        s = humanize_question("What day works best for you?")
                        s = _apply_intro_once(s)
                    elif not sched.get("scheduled_time"):
                        s = humanize_question("What time works best?")
                        s = _apply_intro_once(s)
                    else:
                        s = "Okay."

            # Extra safety: if not booked, strip any confirmation language
            if not booking_created:
                conf_markers = [
                    "confirmation number", "confirmed", "your appointment is", "booked for", "scheduled for"
                ]
                if any(m in s.lower() for m in conf_markers):
                    update_address_assembly_state(sched)
                    if not sched.get("address_verified"):
                        s = build_address_prompt(sched)
                        s = _apply_intro_once(s)
                    elif not sched.get("scheduled_date") and not sched.get("scheduled_time"):
                        s = humanize_question("What day and time work best for you?")
                        s = _apply_intro_once(s)
                    elif not sched.get("scheduled_date"):
                        s = humanize_question("What day works best for you?")
                        s = _apply_intro_once(s)
                    elif not sched.get("scheduled_time"):
                        s = humanize_question("What time works best?")
                        s = _apply_intro_once(s)
                    else:
                        s = "Okay."

            try:
                s = sanitize_sms_body(s, booking_created=bool(booking_created))
            except Exception:
                pass

            # Final cleanup: ensure no em dash / spaced hyphen survives
            s = s.replace("—", ".").replace("–", ".").replace(" - ", " ")
            s = _norm(s)

            # Prevent repeating the exact same final ack text twice in a row
            if _is_ack_only(inbound_text):
                if sched.get("last_ack_text") and sched["last_ack_text"].lower() == s.lower():
                    s = "Okay."
                sched["last_ack_text"] = s
                sched["last_ack_ts"] = datetime.utcnow().isoformat()

            return s

        if inbound_text:
            on_file_patterns = [
                "on file", "have me on file", "my address", "already have my address", "already have me",
                "same address", "saved address"
            ]
            if any(p in inbound_lower for p in on_file_patterns):
                known_q = _known_address_question()
                if known_q:
                    return {
                        "sms_body": _finalize_sms(known_q, sched.get("appointment_type") or appointment_type or "EVAL_195", booking_created=False),
                        "scheduled_date": sched.get("scheduled_date"),
                        "scheduled_time": sched.get("scheduled_time"),
                        "address": sched.get("raw_address"),
                        "booking_complete": False
                    }

        # If the customer confirms a just-stated address, promote that address into a verified state
        # before any other prompt logic can loop them back into town collection.
        if inbound_text and yes_text(inbound_text) and (sched.get("pending_step") or "").strip().lower() == "need_address":
            candidate_addr = (sched.get("raw_address") or "").strip()
            if not candidate_addr:
                candidate_addr = (_best_saved_address(profile) or "").strip()
            if candidate_addr:
                sched["raw_address"] = candidate_addr
                try:
                    normalized = normalize_address(candidate_addr)
                    if isinstance(normalized, tuple) and len(normalized) >= 2:
                        status, addr_struct = normalized[0], normalized[1]
                        if status == "ok" and isinstance(addr_struct, dict):
                            sched["normalized_address"] = addr_struct
                    elif isinstance(normalized, dict):
                        sched["normalized_address"] = normalized
                except Exception as e:
                    print("[WARN] address confirmation normalize failed:", repr(e))

                update_address_assembly_state(sched)
                if not sched.get("address_verified"):
                    parsed = parse_complete_raw_address(candidate_addr)
                    if parsed:
                        sched["normalized_address"] = parsed
                        update_address_assembly_state(sched)

                if sched.get("address_verified"):
                    recompute_pending_step(profile, sched)
                    next_prompt = choose_next_prompt_from_state(conv, inbound_text="")
                    return {
                        "sms_body": _finalize_sms(next_prompt, sched.get("appointment_type") or appointment_type or "EVAL_195", booking_created=False),
                        "scheduled_date": sched.get("scheduled_date"),
                        "scheduled_time": sched.get("scheduled_time"),
                        "address": sched.get("raw_address"),
                        "booking_complete": False
                    }

        # --------------------------------------
        # Emergency flow (2-step confirmation)
        # --------------------------------------
        EMERGENCY = IS_EMERGENCY

        if IS_EMERGENCY and not sched["awaiting_emergency_confirm"] and not sched["emergency_approved"]:
            sched["appointment_type"] = "TROUBLESHOOT_395"
            sched["pending_step"] = None
            try:
                try_early_address_normalize(sched)
            except Exception:
                pass

            if not sched.get("address_verified"):
                sched["awaiting_emergency_confirm"] = False
                known_q = _known_address_question()
                if known_q:
                    msg = f"This sounds urgent. {known_q}"
                else:
                    msg = "This sounds urgent. " + build_address_prompt(sched)
                msg = _finalize_sms(msg, "TROUBLESHOOT_395", booking_created=False)
                return {
                    "sms_body": msg,
                    "scheduled_date": None,
                    "scheduled_time": None,
                    "address": sched.get("raw_address"),
                    "booking_complete": False
                }

            sched["awaiting_emergency_confirm"] = True
            msg = (
                "This looks urgent. We can send someone now and arrival is usually within 1 to 2 hours. "
                "Troubleshoot and repair visits are $395. Do you want us to dispatch someone now?"
            )
            msg = _finalize_sms(msg, "TROUBLESHOOT_395", booking_created=False)
            return {
                "sms_body": msg,
                "scheduled_date": sched.get("scheduled_date"),
                "scheduled_time": sched.get("scheduled_time"),
                "address": sched.get("raw_address"),
                "booking_complete": False
            }

        CONFIRM_PHRASES = ["yes", "yeah", "yup", "ok", "okay", "sure", "book", "send", "do it", "dispatch", "come now", "now", "immediately", "asap"]

        if sched["awaiting_emergency_confirm"] and any(p in inbound_lower for p in CONFIRM_PHRASES):
            dispatch_dt = now_local + timedelta(hours=1)
            minute = dispatch_dt.minute
            if minute == 0:
                rounded_dt = dispatch_dt.replace(second=0, microsecond=0)
            elif minute <= 30:
                rounded_dt = dispatch_dt.replace(minute=30, second=0, microsecond=0)
            else:
                rounded_dt = (dispatch_dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

            sched["emergency_approved"] = True
            sched["awaiting_emergency_confirm"] = False
            sched["appointment_type"] = "TROUBLESHOOT_395"
            sched["scheduled_date"] = rounded_dt.strftime("%Y-%m-%d")
            sched["scheduled_time"] = rounded_dt.strftime("%H:%M")
            sched["pending_step"] = None
            if not sched.get("price_disclosed"):
                sched["price_disclosed"] = True

            scheduled_date = sched["scheduled_date"]
            scheduled_time = sched["scheduled_time"]

        # --------------------------------------
        # Appointment type fallback
        # --------------------------------------
        appt_type = sched.get("appointment_type") or appointment_type
        if not appt_type:
            if any(w in inbound_lower for w in [
                "not working", "no power", "dead", "sparking", "burning",
                "breaker keeps", "gfci", "outlet not", "troubleshoot"
            ]):
                appt_type = "TROUBLESHOOT_395"
            elif any(w in inbound_lower for w in ["inspection", "whole home inspection", "electrical inspection"]):
                appt_type = "WHOLE_HOME_INSPECTION"
            else:
                appt_type = "EVAL_195"
            sched["appointment_type"] = appt_type

        # --------------------------------------
        # Missing-info resolver (Pre-LLM)
        # --------------------------------------
        update_address_assembly_state(sched)

        # ✅ HOUSE NUMBER PATCH (pre-LLM) — IMPROVED
        try:
            update_address_assembly_state(sched)
            missing_atom = (sched.get("address_missing") or "").strip().lower()
            inbound_clean = inbound_text.strip()

            low = inbound_clean.lower()
            street_suffixes = (
                " st", " street", " ave", " avenue", " rd", " road", " ln", " lane",
                " dr", " drive", " ct", " court", " cir", " circle", " blvd", " boulevard",
                " way", " pkwy", " parkway", " ter", " terrace", " park"
            )
            inbound_has_street_word = any(suf in f" {low} " for suf in street_suffixes)
            inbound_starts_with_number = bool(re.match(r"^\s*\d{1,6}\b", inbound_clean))
            m_num = re.search(r"\b(\d{1,6})\b", inbound_clean)
            num = m_num.group(1) if m_num else None

            raw = (sched.get("raw_address") or "").strip()
            norm = sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None
            norm_line1 = (norm.get("address_line_1") or "").strip() if norm else ""

            if missing_atom == "street":
                if inbound_has_street_word and not inbound_starts_with_number:
                    city_hint = raw.strip()
                    if city_hint and city_hint.lower() not in inbound_clean.lower():
                        set_raw_address_safe(sched, f"{inbound_clean}, {city_hint}".strip(" ,"))
                    else:
                        set_raw_address_safe(sched, inbound_clean)
                    sched["normalized_address"] = None
                    update_address_assembly_state(sched)

                elif inbound_starts_with_number and inbound_has_street_word:
                    city_hint = raw.strip()
                    if city_hint and city_hint.lower() not in inbound_clean.lower():
                        set_raw_address_safe(sched, f"{inbound_clean}, {city_hint}".strip(" ,"))
                    else:
                        set_raw_address_safe(sched, inbound_clean)
                    sched["normalized_address"] = None
                    update_address_assembly_state(sched)

            elif missing_atom == "number":
                if inbound_starts_with_number and inbound_has_street_word:
                    set_raw_address_safe(sched, inbound_clean)
                    sched["normalized_address"] = None
                    update_address_assembly_state(sched)

                elif num:
                    base = raw or norm_line1
                    if base and not re.match(r"^\d{1,6}\b", base):
                        set_raw_address_safe(sched, f"{num} {base}".strip())
                        sched["normalized_address"] = None
                    update_address_assembly_state(sched)

        except Exception as e:
            print("[WARN] house-number merge patch failed:", repr(e))

        # ✅ CITY/STATE FOLLOW-UP PATCH
        try:
            apply_partial_address_reply(sched, inbound_text)
            # If we now have a complete verified address, clear any stale address-step loop.
            if sched.get("address_verified"):
                if sched.get("pending_step") in {"need_address", None}:
                    recompute_pending_step(profile, sched)
        except Exception as e:
            print("[WARN] partial address merge patch failed:", repr(e))

        # If the customer is accepting the last final confirmation, lock that immediately.
        current_final_key = None
        if sched.get("scheduled_date") and sched.get("scheduled_time"):
            current_final_key = f"{sched.get('scheduled_date')}|{sched.get('scheduled_time')}"

        if (
            sched.get("final_confirmation_sent")
            and not sched.get("booking_created")
            and confirmation_accept_text(inbound_text)
            and current_final_key
            and sched.get("last_final_confirmation_key") == current_final_key
        ):
            sched["final_confirmation_accepted"] = True

        # --------------------------------------
        # Initial outbound shortcut: if this is the first text after voicemail and
        # we already have a full saved address on file, confirm that address instead
        # of asking from scratch for house number/street.
        # --------------------------------------
        if not inbound_text:
            try:
                update_address_assembly_state(sched)
                if not sched.get("address_verified"):
                    known_q = _known_address_question()
                    if known_q:
                        msg = _finalize_sms(known_q, appt_type, booking_created=False)
                        return {
                            "sms_body": msg,
                            "scheduled_date": sched.get("scheduled_date"),
                            "scheduled_time": sched.get("scheduled_time"),
                            "address": sched.get("raw_address"),
                            "booking_complete": False
                        }
            except Exception as e:
                print("[WARN] initial known-address shortcut failed:", repr(e))

        # --------------------------------------
        # Build System Prompt
        # --------------------------------------
        system_prompt = build_system_prompt(
            cleaned_transcript,
            category,
            appt_type,
            initial_sms,
            sched.get("scheduled_date") or scheduled_date,
            sched.get("scheduled_time") or scheduled_time,
            sched.get("raw_address") or address,
            today_date_str,
            today_weekday,
            conv
        )

        # --------------------------------------
        # LLM CALL (STRICT JSON)
        # --------------------------------------
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": inbound_text},
            ],
        )

        ai_raw = json.loads(
            completion.choices[0].message.content
            .strip()
            .replace("None", "null")
            .replace("none", "null")
        )
        try:
            log_event("AI_STEP4_RESULT", phone, {
                "inbound": _safe_monitor_text(inbound_text, 700),
                "pending_before": sched.get("pending_step"),
                "raw_address_before": _safe_monitor_text(sched.get("raw_address"), 240),
                "ai_raw": _safe_monitor_text(json.dumps(ai_raw, ensure_ascii=False), 1000),
            }, conv)
        except Exception:
            pass

        sms_body   = (ai_raw.get("sms_body") or "").strip()
        model_date = ai_raw.get("scheduled_date")
        model_time = ai_raw.get("scheduled_time")
        model_addr = ai_raw.get("address")

        # Heuristic slot salvage: preserve explicit times embedded in mixed messages
        # like "2pm and I have dogs is that ok?" when the LLM only answers the question.
        explicit_customer_time = extract_explicit_time_from_text(inbound_text)
        if not model_time:
            salvaged_time = explicit_customer_time
            if salvaged_time:
                model_time = salvaged_time
                sched["scheduled_time_source"] = "customer_explicit"
        elif explicit_customer_time:
            sched["scheduled_time_source"] = "customer_explicit"
        elif salvage_relative_date_from_text(inbound_text) and not sched.get("scheduled_time_source"):
            # Date-only replies like "today" or "Monday" must not create a time.
            model_time = None

        # --------------------------------------
        # RESET-LOCK (never lose good stored values)
        # --------------------------------------
        if sched.get("scheduled_date") and not model_date:
            model_date = sched["scheduled_date"]
        if sched.get("scheduled_time") and not model_time:
            model_time = sched["scheduled_time"]
        if sched.get("raw_address") and not model_addr:
            model_addr = sched["raw_address"]

        if isinstance(model_addr, str) and len(model_addr.strip()) > 3:
            model_addr = scrub_non_address_tail(model_addr)
            safe_model_addr = safe_address_candidate_from_text(model_addr) or (model_addr.strip() if is_plausible_address_text(model_addr) else None)
            inbound_has_address = bool(extract_service_address_from_text(inbound_text))
            if safe_model_addr:
                existing_addr = scrub_non_address_tail((sched.get("raw_address") or "").strip())
                if not (existing_addr and looks_like_non_address_fragment(inbound_text) and not inbound_has_address):
                    if (not existing_addr) or (not sched.get("address_verified")) or _address_has_house_number_and_street(safe_model_addr):
                        set_raw_address_safe(sched, safe_model_addr)
                        sched["normalized_address"] = None
            else:
                try:
                    log_event("ADDRESS_MODEL_OUTPUT_REJECTED", phone, {"model_address": _safe_monitor_text(model_addr), "kept_address": _safe_monitor_text(sched.get("raw_address"))}, conv)
                except Exception:
                    pass

        # ---------------------------------------------------
        # CRITICAL PATCH: re-run address state AFTER model changes
        # ---------------------------------------------------
        update_address_assembly_state(sched)
        try:
            try_early_address_normalize(sched)
        except Exception as e:
            print("[WARN] try_early_address_normalize post-LLM failed:", repr(e))
        update_address_assembly_state(sched)

        model_addr = sched.get("raw_address")

        # Human-readable time
        try:
            human_time = datetime.strptime(model_time, "%H:%M").strftime("%-I:%M %p") if model_time else None
        except Exception:
            human_time = model_time

        if model_time and human_time:
            sms_body = sms_body.replace(model_time, human_time)

        # Save new values
        if model_date:
            sched["scheduled_date"] = model_date
        if model_time:
            if extract_explicit_time_from_text(inbound_text) or sched.get("scheduled_time_source") in {"customer_explicit", "offered_slot", "voicemail_explicit", "model_non_date_only"}:
                sched["scheduled_time"] = model_time
                sched["scheduled_time_source"] = sched.get("scheduled_time_source") or "customer_explicit"

        # CRITICAL: once new slot values are saved, recompute state immediately so
        # the autobooking gate sees the updated step instead of a stale need_date / need_time.
        update_address_assembly_state(sched)
        recompute_pending_step(profile, sched)

        # Mid-flow customer interruptions: answer briefly, then return to the next step.
        # Run this AFTER model extraction so combined messages like
        # "Tuesdays usually work best. Is the $195 just to come out?" can
        # both save the day preference and answer pricing without losing state.
        # IMPORTANT: this must happen BEFORE the address-prompt override so
        # pricing and other short questions still get answered while we are
        # collecting the address.
        interruption_reply = interruption_answer_and_return_prompt(conv, inbound_text)
        if interruption_reply and not IS_EMERGENCY and not sched.get("booking_created"):
            interruption_reply = _finalize_sms(interruption_reply, appt_type, booking_created=False)
            return {
                "sms_body": interruption_reply,
                "scheduled_date": sched.get("scheduled_date"),
                "scheduled_time": sched.get("scheduled_time"),
                "address": sched.get("raw_address"),
                "booking_complete": False
            }

        # If address STILL not verified, override immediately (humanized prompt stays in builder)
        if not sched.get("address_verified"):
            msg = build_address_prompt(sched)
            msg = _finalize_sms(msg, appt_type, booking_created=False)
            return {
                "sms_body": msg,
                "scheduled_date": sched.get("scheduled_date") or model_date,
                "scheduled_time": sched.get("scheduled_time") or model_time,
                "address": sched.get("raw_address"),
                "booking_complete": False
            }

        # --------------------------------------
        # AUTOBOOKING (Step 5)
        # --------------------------------------
        has_identity_for_booking = bool(
            (get_active_first_name(profile) and get_active_last_name(profile))
            or profile.get("square_customer_id")
        )
        has_contact_for_booking = bool(get_active_email(profile) or profile.get("square_customer_id"))

        # If the same slot is already saved as the upcoming appointment, do not try to recreate it.
        upcoming = profile.get("upcoming_appointment") or {}
        if (
            upcoming.get("date")
            and upcoming.get("time")
            and sched.get("scheduled_date") == upcoming.get("date")
            and sched.get("scheduled_time") == upcoming.get("time")
            and not sched.get("booking_created")
        ):
            sched["booking_created"] = True
            sched["square_booking_id"] = upcoming.get("square_id") or "existing_on_file"
            try:
                booked_dt = datetime.strptime(sched["scheduled_date"], "%Y-%m-%d")
                human_day = booked_dt.strftime("%A, %B %d").replace(" 0", " ")
            except Exception:
                human_day = sched.get("scheduled_date") or "that day"
            human_t = humanize_time(sched.get("scheduled_time") or "") or (sched.get("scheduled_time") or "that time")
            booked_sms = f"You're already all set for {human_day} at {human_t}."
            booked_sms = _finalize_sms(booked_sms, appt_type, booking_created=True)
            return {
                "sms_body": booked_sms,
                "scheduled_date": sched.get("scheduled_date"),
                "scheduled_time": sched.get("scheduled_time"),
                "address": sched.get("raw_address"),
                "booking_complete": True
            }

        ready_for_booking = (
            bool(sched.get("scheduled_date")) and
            bool(sched.get("scheduled_time")) and
            bool(sched.get("address_verified")) and
            bool(sched.get("appointment_type")) and
            has_identity_for_booking and
            has_contact_for_booking and
            not sched.get("pending_step") and
            not sched.get("booking_created")
        )

        if ready_for_booking or (EMERGENCY and has_identity_for_booking):
            try:
                booking_attempt = maybe_create_square_booking(phone, conv)

                if isinstance(booking_attempt, dict) and booking_attempt.get("status") == "slot_unavailable":
                    unavailable_sms = booking_attempt.get("message") or choose_next_prompt_from_state(conv, inbound_text=inbound_text)
                    unavailable_sms = _finalize_sms(unavailable_sms, appt_type, booking_created=False)
                    return {
                        "sms_body": unavailable_sms,
                        "scheduled_date": sched.get("scheduled_date"),
                        "scheduled_time": sched.get("scheduled_time"),
                        "address": sched.get("raw_address"),
                        "booking_complete": False
                    }

                if isinstance(booking_attempt, dict) and booking_attempt.get("status") == "outside_hours":
                    sched["final_confirmation_sent"] = False
                    sched["final_confirmation_accepted"] = False
                    sched["last_final_confirmation_key"] = None
                    if sched.get("scheduled_date"):
                        outside_sms = "We typically schedule between 9am and 4pm. What time in that window works for you?"
                    else:
                        outside_sms = "We typically schedule between 9am and 4pm. What day and time in that window work best for you?"
                    outside_sms = _finalize_sms(outside_sms, appt_type, booking_created=False)
                    return {
                        "sms_body": outside_sms,
                        "scheduled_date": sched.get("scheduled_date"),
                        "scheduled_time": sched.get("scheduled_time"),
                        "address": sched.get("raw_address"),
                        "booking_complete": False
                    }

                if isinstance(booking_attempt, dict) and booking_attempt.get("status") == "weekend_blocked":
                    sched["final_confirmation_sent"] = False
                    sched["final_confirmation_accepted"] = False
                    sched["last_final_confirmation_key"] = None
                    weekend_sms = "We schedule non-emergency visits Monday through Friday. What day and time work best for you?"
                    weekend_sms = _finalize_sms(weekend_sms, appt_type, booking_created=False)
                    return {
                        "sms_body": weekend_sms,
                        "scheduled_date": sched.get("scheduled_date"),
                        "scheduled_time": sched.get("scheduled_time"),
                        "address": sched.get("raw_address"),
                        "booking_complete": False
                    }

                if sched.get("booking_created") and sched.get("square_booking_id"):
                    try:
                        booked_dt = datetime.strptime(sched["scheduled_date"], "%Y-%m-%d")
                        human_day = booked_dt.strftime("%A, %B %d").replace(" 0", " ")
                    except Exception:
                        human_day = sched["scheduled_date"]

                    sched["final_confirmation_sent"] = False
                    sched["final_confirmation_accepted"] = False
                    sched["last_final_confirmation_key"] = None
                    booked_sms = (
                        f"You're all set for {human_day} at {human_time}. "
                        "We have you on the schedule."
                    )
                    booked_sms = _finalize_sms(booked_sms, appt_type, booking_created=True)
                    return {
                        "sms_body": booked_sms,
                        "scheduled_date": sched["scheduled_date"],
                        "scheduled_time": sched["scheduled_time"],
                        "address": model_addr,
                        "booking_complete": True
                    }
            except Exception as e:
                print("[ERROR] Autobooking:", repr(e))

        # --------------------------------------
        # HARD SAFETY: If Square didn't book, never confirm. Ask next missing atom.
        # --------------------------------------
        if not (sched.get("booking_created") and sched.get("square_booking_id")):
            update_address_assembly_state(sched)
            recompute_pending_step(profile, sched)
            sms_body = choose_next_prompt_from_state(conv, inbound_text=inbound_text)

            # If we already have a full slot and booking details but somehow fell through,
            # do not send a dead-end "Okay.". Re-attempt the booking path once.
            if sms_body in {"Okay.", "Okay", "ok", "ok."}:
                has_identity_for_booking = bool(
                    (get_active_first_name(profile) and get_active_last_name(profile))
                    or profile.get("square_customer_id")
                )
                has_contact_for_booking = bool(get_active_email(profile) or profile.get("square_customer_id"))
                ready_for_booking_retry = (
                    bool(sched.get("scheduled_date")) and
                    bool(sched.get("scheduled_time")) and
                    bool(sched.get("address_verified")) and
                    bool(sched.get("appointment_type")) and
                    has_identity_for_booking and
                    has_contact_for_booking and
                    not sched.get("pending_step") and
                    not sched.get("booking_created")
                )
                if ready_for_booking_retry:
                    try:
                        booking_attempt = maybe_create_square_booking(phone, conv)
                        if isinstance(booking_attempt, dict) and booking_attempt.get("status") == "slot_unavailable":
                            sms_body = booking_attempt.get("message") or "That time is already booked. Here are three other times that work."
                        elif isinstance(booking_attempt, dict) and booking_attempt.get("status") == "outside_hours":
                            sms_body = "We typically schedule between 9am and 4pm. What time in that window works for you?"
                        elif isinstance(booking_attempt, dict) and booking_attempt.get("status") == "weekend_blocked":
                            sms_body = "We schedule non-emergency visits Monday through Friday. What day and time work best for you?"
                        elif sched.get("booking_created") and sched.get("square_booking_id"):
                            try:
                                booked_dt = datetime.strptime(sched["scheduled_date"], "%Y-%m-%d")
                                human_day = booked_dt.strftime("%A, %B %d").replace(" 0", " ")
                            except Exception:
                                human_day = sched.get("scheduled_date") or "that day"
                            booked_time = humanize_time(sched.get("scheduled_time") or "") or (sched.get("scheduled_time") or "that time")
                            sms_body = f"You're all set for {human_day} at {booked_time}. We have you on the schedule."
                    except Exception as e:
                        print("[WARN] fallback booking retry failed:", repr(e))

        booking_created = bool(sched.get("booking_created") and sched.get("square_booking_id"))
        sms_body = _finalize_sms(sms_body, appt_type, booking_created=booking_created)
        sms_body = collapse_duplicate_sms(sms_body)

        return {
            "sms_body": sms_body,
            "scheduled_date": model_date,
            "scheduled_time": model_time,
            "address": model_addr,
            "booking_complete": False
        }

    except Exception as e:
        print("[ERROR] generate_reply_for_inbound:", repr(e))
        return {
            "sms_body": "Sorry, can you say that again?",
            "scheduled_date": scheduled_date,
            "scheduled_time": scheduled_time,
            "address": address,
            "booking_complete": False
        }





# ---------------------------------------------------
# PRICE INJECTION HELPER (PATCH 2)
# ---------------------------------------------------
def apply_price_injection(appt_type: str, body: str, conv: dict | None = None) -> str:
    body = " ".join(str(body or "").split()).strip()
    if "$" in body:
        return body

    prompt = build_price_and_availability_prompt(conv or {}, appt_type)
    return replace_generic_schedule_question_with_availability(body, prompt)


# ---------------------------------------------------
# Google Maps — Address Normalization
# ---------------------------------------------------
def normalize_address(raw_address: str, forced_state: str | None = None):
    if not GOOGLE_MAPS_API_KEY or not raw_address:
        print("normalize_address: missing API key or address")
        return "error", None

    try:
        params = {
            "address": raw_address,
            "key": GOOGLE_MAPS_API_KEY,
        }

        if forced_state:
            params["components"] = f"country:US|administrative_area:{forced_state}"
        else:
            params["components"] = "country:US"

        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params=params,
            timeout=8,
        )
        data = resp.json()

        status = data.get("status")
        if status != "OK" or not data.get("results"):
            print("normalize_address: status:", status)
            return "error", None

        result = data["results"][0]
        comps = result.get("address_components", [])

        line1 = None
        city = None
        state = None
        zipcode = None

        for c in comps:
            types = c.get("types", [])
            if "street_number" in types:
                line1 = c["long_name"]
            if "route" in types:
                if line1:
                    line1 = f"{line1} {c['long_name']}"
                else:
                    line1 = c["long_name"]
            if "locality" in types:
                city = c["long_name"]
            if "administrative_area_level_1" in types:
                state = c["short_name"]
            if "postal_code" in types:
                zipcode = c["long_name"]

        if not state and not forced_state:
            return "needs_state", None

        if state and state not in ("CT", "MA") and not forced_state:
            return "needs_state", None

        final_state = forced_state or state

        if not (line1 and city and final_state and zipcode):
            return "error", None

        addr_struct = {
            "address_line_1": line1,
            "locality": city,
            "administrative_district_level_1": final_state,
            "postal_code": zipcode,
            "country": "US",
        }

        return "ok", addr_struct

    except Exception as e:
        print("[ERROR] normalize_address:", repr(e))
        return "error", None


# ---------------------------------------------------
# Square Customer Creation
# ---------------------------------------------------
def square_headers() -> dict:
    if not SQUARE_ACCESS_TOKEN:
        raise RuntimeError("SQUARE_ACCESS_TOKEN missing")
    return {
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _phone_lookup_variants(phone: str) -> list[str]:
    """Return practical phone formats Square may have stored for the same caller."""
    raw = str(phone or "").strip().replace("whatsapp:", "")
    digits = re.sub(r"\D+", "", raw)
    variants: list[str] = []

    def add(v: str):
        v = str(v or "").strip()
        if v and v not in variants:
            variants.append(v)

    add(raw)
    if digits:
        add(digits)
        if len(digits) == 11 and digits.startswith("1"):
            ten = digits[1:]
            add("+" + digits)
            add(ten)
        elif len(digits) == 10:
            ten = digits
            add("+1" + ten)
            add("1" + ten)
        else:
            ten = digits[-10:] if len(digits) >= 10 else ""
        if len(digits) >= 10:
            ten = digits[-10:]
            add(ten)
            add(f"+1{ten}")
            add(f"1{ten}")
            add(f"{ten[0:3]}-{ten[3:6]}-{ten[6:10]}")
            add(f"({ten[0:3]}) {ten[3:6]}-{ten[6:10]}")
            add(f"{ten[0:3]}.{ten[3:6]}.{ten[6:10]}")
    return variants


def square_lookup_customer_by_phone(phone: str) -> dict | None:
    """Return the first Square customer match for this phone, or None.

    Square's exact phone filter can be format-sensitive depending on how the
    customer was created/imported. Try common E.164 and local formats so an
    existing customer is recognized from the live voice call.
    """
    try:
        for candidate in _phone_lookup_variants(phone):
            payload = {"query": {"filter": {"phone_number": {"exact": candidate}}}}
            resp = requests.post(
                "https://connect.squareup.com/v2/customers/search",
                json=payload,
                headers=square_headers(),
                timeout=10,
            )
            if resp.status_code not in (200, 201):
                continue
            data = resp.json() or {}
            custs = data.get("customers") or []
            if custs:
                return custs[0]
    except Exception as e:
        print("[WARN] customer search failed:", repr(e))
    return None


def square_create_or_get_customer(
    phone: str,
    profile: dict | None = None,
    addr_struct: dict | None = None,
):
    """
    Repeat-customer optimization with identity protection:
      - Search by phone first.
      - Reuse the Square customer only if the active booking identity matches.
      - If the caller provided a different name for this booking, create a fresh customer.
    """
    profile = profile or {}

    active_first = (profile.get("active_first_name") or profile.get("first_name") or "").strip()
    active_last = (profile.get("active_last_name") or profile.get("last_name") or "").strip()
    active_email = (profile.get("active_email") or profile.get("email") or "").strip()

    cust = square_lookup_customer_by_phone(phone)
    if cust and cust.get("id"):
        cid = cust["id"]

        profile.setdefault("square_customer_id", cid)
        profile["recognized_first_name"] = cust.get("given_name")
        profile["recognized_last_name"] = cust.get("family_name")
        profile["recognized_email"] = cust.get("email_address")

        existing_first = (cust.get("given_name") or "").strip().lower()
        existing_last = (cust.get("family_name") or "").strip().lower()
        desired_first = active_first.strip().lower()
        desired_last = active_last.strip().lower()

        same_identity = bool(
            desired_first and desired_last and
            existing_first == desired_first and
            existing_last == desired_last
        )

        if not desired_first and not desired_last:
            same_identity = True

        try:
            caddr = cust.get("address") or {}
            line1 = (caddr.get("address_line_1") or "").strip()
            city = (caddr.get("locality") or "").strip()
            state = (caddr.get("administrative_district_level_1") or "").strip()
            zipc = (caddr.get("postal_code") or "").strip()
            if line1 and city and state:
                pretty = f"{line1}, {city}, {state} {zipc}".strip()
                addrs = profile.setdefault("addresses", [])
                if pretty and pretty not in addrs:
                    addrs.append(pretty)
        except Exception:
            pass

        if same_identity:
            return cid

    try:
        payload = {
            "idempotency_key": str(uuid.uuid4()),
            "phone_number": phone,
        }

        if active_first:
            payload["given_name"] = active_first
        if active_last:
            payload["family_name"] = active_last
        if active_email:
            payload["email_address"] = active_email

        if not payload.get("given_name") and not payload.get("family_name"):
            payload["given_name"] = "Prevolt Lead"

        if addr_struct:
            payload["address"] = addr_struct

        resp = requests.post(
            "https://connect.squareup.com/v2/customers",
            json=payload,
            headers=square_headers(),
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            print("[ERROR] customer create failed:", resp.status_code, resp.text)
            return None

        data = resp.json() or {}
        cid = (data.get("customer") or {}).get("id")

        if cid:
            profile["square_customer_id"] = cid
            upsert_known_person(
                profile,
                first_name=active_first or profile.get("recognized_first_name") or "",
                last_name=active_last or profile.get("recognized_last_name") or "",
                email=active_email or profile.get("recognized_email") or "",
                square_customer_id=cid,
            )

        return cid

    except Exception as e:
        print("[ERROR] customer create failed:", repr(e))
        return None


# ---------------------------------------------------
# parse_local_datetime (corrected)
# ---------------------------------------------------
def parse_local_datetime(date_str: str, time_str: str):
    try:
        tz = ZoneInfo("America/New_York")
    except:
        tz = timezone(timedelta(hours=-5))

    now = datetime.now(tz)
    if not date_str:
        return None

    t_lower = (time_str or "").lower().strip()

    # Emergency logic
    if t_lower in ("now", "asap", "right now", "immediately"):
        minute = (now.minute + 4) // 5 * 5
        if minute >= 60:
            now = now.replace(minute=0) + timedelta(hours=1)
        else:
            now = now.replace(minute=minute)
        final_t = now.time()

    elif t_lower in ("any time", "anytime", "whenever"):
        future = now + timedelta(hours=1)
        final_t = future.time()

    elif t_lower == "tonight":
        tonight = now.replace(hour=19, minute=0, second=0, microsecond=0)
        if tonight < now:
            tonight = now + timedelta(minutes=30)
        final_t = tonight.time()

    else:
        try:
            final_t = datetime.strptime(t_lower, "%H:%M").time()
        except Exception:
            return None

    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        combined = datetime.combine(d, final_t).replace(tzinfo=tz)
    except Exception:
        return None

    if combined < now:
        combined = now + timedelta(minutes=15)
        combined = combined.replace(second=0, microsecond=0)

    return combined.astimezone(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------
# Appointment Type → Square Variation Mapping
# ---------------------------------------------------
def map_appointment_type_to_variation(appt: str):
    if not appt:
        return None, None

    appt = appt.upper()

    if "EVAL" in appt:
        return SERVICE_VARIATION_EVAL_ID, SERVICE_VARIATION_EVAL_VERSION
    if "INSPECTION" in appt:
        return SERVICE_VARIATION_INSPECTION_ID, SERVICE_VARIATION_INSPECTION_VERSION
    if "TROUBLESHOOT" in appt or "REPAIR" in appt:
        return SERVICE_VARIATION_TROUBLESHOOT_ID, SERVICE_VARIATION_TROUBLESHOOT_VERSION

    return None, None


def _humanize_date_for_sms(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.strftime("%A, %B %d").replace(" 0", " ")
    except Exception:
        return (date_str or "").strip()


def _humanize_slot_label(date_str: str, time_str: str) -> str:
    human_t = humanize_time(time_str) if time_str else ""
    if date_str:
        return f"{_humanize_date_for_sms(date_str)} at {human_t}".strip()
    return human_t.strip()


def is_next_available_request(inbound_text: str) -> bool:
    low = _loose_text(inbound_text)
    if not low:
        return False

    patterns = [
        "next available",
        "next availability",
        "next opening",
        "next appointment",
        "next available appointment",
        "earliest appointment",
        "earliest available",
        "soonest appointment",
        "soonest available",
        "first available",
        "what is your next available",
        "when is your next available",
        "when is your next availability",
        "when are you available next",
        "what is your next available time",
        "when is your next available time",
        "what do you have open",
        "what do you have available",
        "when can you come out next",
        "when can you come next",
        "when can you come out",
        "when can you come",
        "next available slot",
        "let me know availability",
        "please let me know availability",
        "what availability",
        "your availability",
        "availability?",
    ]
    if any(p in low for p in patterns):
        return True
    # Generic availability asks should offer concrete openings, not fall through
    # to the canned booking-details fallback. Keep this narrow so ordinary words
    # like "available" in a full scheduling answer do not override explicit dates/times.
    return bool(re.search(r"\b(?:availability|available openings|openings|available times)\b", low))


def get_next_available_slots(appointment_type: str, limit: int = 3, days_ahead: int = 14) -> list[dict]:
    """Return up to `limit` available options on different calendar days.

    Prevolt's customer-facing flow should not offer three times on the same day
    by default. It feels narrow and robotic. For the scheduling handoff, offer
    the next available day/time, then a later day/time, then a third later
    day/time. Each day gets one available time so the customer sees real choice
    across the week.
    """
    appointment_type = (appointment_type or "EVAL_195").upper()
    tz = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
    now_local = datetime.now(tz)
    collected = []
    seen_days = set()
    rng = random.SystemRandom()

    non_emergency = "TROUBLESHOOT" not in appointment_type

    for offset in range(days_ahead + 1):
        target_day = now_local.date() + timedelta(days=offset)
        if non_emergency and target_day.weekday() >= 5:
            continue
        target_date = target_day.strftime("%Y-%m-%d")
        if target_date in seen_days:
            continue

        try:
            day_slots = search_square_availability_for_day(target_date, appointment_type)
        except Exception as e:
            print("[WARN] get_next_available_slots day lookup failed:", target_date, repr(e))
            day_slots = []

        valid_slots = []
        seen_times = set()
        for slot in day_slots or []:
            slot_date = str(slot.get("date") or target_date).strip()
            slot_time = str(slot.get("time") or "").strip()
            if not slot_time or slot_time in seen_times:
                continue
            try:
                slot_dt = datetime.strptime(f"{slot_date} {slot_time}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            except Exception:
                continue
            if slot_dt <= now_local:
                continue
            seen_times.add(slot_time)
            valid_slots.append(slot)

        if not valid_slots:
            continue

        # Pick one valid time for this date so the three options are three
        # different days, not the same day repeated at 9/10/12.
        chosen = rng.choice(valid_slots)
        seen_days.add(target_date)
        collected.append(chosen)
        if len(collected) >= limit:
            return collected

    return collected


def format_next_available_slots_message(slots: list[dict]) -> str:
    if not slots:
        return "I’m not seeing open times right now. What day and time work best for you?"

    labels = [s.get("label") or _humanize_slot_label(s.get("date"), s.get("time")) for s in slots[:3]]
    if len(labels) == 1:
        opts = labels[0]
    elif len(labels) == 2:
        opts = f"{labels[0]} or {labels[1]}"
    else:
        opts = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return f"My next three openings are {opts}. Which one works best?"


def _local_day_range_to_utc(date_str: str) -> tuple[str | None, str | None]:
    try:
        tz = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
        local_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
        local_end = local_start + timedelta(days=1)
        return (
            local_start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            local_end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    except Exception:
        return None, None


def search_square_availability_for_day(date_str: str, appointment_type: str) -> list[dict]:
    variation_id, _variation_version = map_appointment_type_to_variation(appointment_type)
    if not (variation_id and SQUARE_LOCATION_ID and SQUARE_TEAM_MEMBER_ID and SQUARE_ACCESS_TOKEN):
        return []

    start_at, end_at = _local_day_range_to_utc(date_str)
    if not (start_at and end_at):
        return []

    payload = {
        "query": {
            "filter": {
                "start_at_range": {
                    "start_at": start_at,
                    "end_at": end_at
                },
                "location_id": SQUARE_LOCATION_ID,
                "segment_filters": [
                    {
                        "service_variation_id": variation_id,
                        "team_member_id_filter": {
                            "any": [SQUARE_TEAM_MEMBER_ID]
                        }
                    }
                ]
            }
        }
    }

    try:
        resp = requests.post(
            "https://connect.squareup.com/v2/bookings/availability/search",
            headers=square_headers(),
            json=payload,
            timeout=12,
        )
        if resp.status_code not in (200, 201):
            print("[WARN] Square availability search failed:", resp.status_code, resp.text)
            return []

        data = resp.json() or {}
        avails = data.get("availabilities") or []
        out = []
        tz = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
        for av in avails:
            start_at_raw = av.get("start_at")
            segs = av.get("appointment_segments") or []
            seg0 = segs[0] if segs and isinstance(segs, list) else {}
            if not start_at_raw:
                continue
            try:
                utc_dt = datetime.strptime(start_at_raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                local_dt = utc_dt.astimezone(tz)
            except Exception:
                continue
            out.append({
                "date": local_dt.strftime("%Y-%m-%d"),
                "time": local_dt.strftime("%H:%M"),
                "start_at": start_at_raw,
                "team_member_id": seg0.get("team_member_id") or SQUARE_TEAM_MEMBER_ID,
                "service_variation_id": seg0.get("service_variation_id") or variation_id,
                "service_variation_version": seg0.get("service_variation_version"),
                "duration_minutes": seg0.get("duration_minutes") or 60,
                "label": _humanize_slot_label(local_dt.strftime("%Y-%m-%d"), local_dt.strftime("%H:%M")),
            })
        return out
    except Exception as e:
        print("[WARN] Square availability exception:", repr(e))
        return []


def _nearest_same_day_slots(avails: list[dict], requested_time: str, limit: int = 3) -> list[dict]:
    if not avails:
        return []
    try:
        req = datetime.strptime(requested_time, "%H:%M")
    except Exception:
        req = None

    def _distance(slot: dict):
        try:
            cur = datetime.strptime(slot.get("time") or "", "%H:%M")
            if req:
                return abs(int((cur - req).total_seconds()))
            return int(cur.strftime("%H")) * 60 + int(cur.strftime("%M"))
        except Exception:
            return 10**9

    unique = {}
    for s in avails:
        key = (s.get("date"), s.get("time"))
        if key not in unique:
            unique[key] = s
    ordered = sorted(unique.values(), key=_distance)
    return ordered[:limit]


def _rolling_same_weekday_slots(date_str: str, appointment_type: str, limit: int = 3) -> list[dict]:
    try:
        base = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return []

    gathered = []
    for weeks_ahead in range(1, 5):
        target = (base + timedelta(days=7 * weeks_ahead)).strftime("%Y-%m-%d")
        avails = search_square_availability_for_day(target, appointment_type)
        if avails:
            gathered.extend(avails[:limit])
            break
    return gathered[:limit]


def _format_slot_offer_message(requested_date: str, requested_time: str, same_day_slots: list[dict], rolled_slots: list[dict]) -> str:
    requested_human = humanize_time(requested_time) if requested_time else "that time"
    if same_day_slots:
        labels = [humanize_time(s.get("time") or "") for s in same_day_slots]
        if len(labels) == 1:
            opts = labels[0]
        elif len(labels) == 2:
            opts = f"{labels[0]} or {labels[1]}"
        else:
            opts = ", ".join(labels[:-1]) + f", or {labels[-1]}"
        return f"We’re booked at {requested_human}. I have {opts} open that day. Which one works best?"

    if rolled_slots:
        labels = [s.get("label") or _humanize_slot_label(s.get("date"), s.get("time")) for s in rolled_slots]
        if len(labels) == 1:
            opts = labels[0]
        elif len(labels) == 2:
            opts = f"{labels[0]} or {labels[1]}"
        else:
            opts = ", ".join(labels[:-1]) + f", or {labels[-1]}"
        return f"That day is full. I have {opts}. Which one works best?"

    human_day = _humanize_date_for_sms(requested_date)
    return f"We’re fully booked for {human_day}. Would another day work?"


def _requested_local_interval(date_str: str, time_str: str, duration_minutes: int = 60) -> tuple[datetime | None, datetime | None]:
    """Return the requested appointment interval in America/New_York local time."""
    try:
        tz = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
        start = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        end = start + timedelta(minutes=duration_minutes or 60)
        return start, end
    except Exception:
        return None, None


def _square_booking_is_active(status: str) -> bool:
    """Treat non-cancelled Square bookings as time blockers."""
    status = (status or "").upper()
    cancelled_tokens = ("CANCEL", "CANCELED", "NO_SHOW")
    return bool(status) and not any(tok in status for tok in cancelled_tokens)


def square_slot_has_existing_booking(date_str: str, time_str: str, appointment_type: str, duration_minutes: int = 60) -> bool:
    """
    Narrow conflict check used before creating a booking.

    This intentionally does NOT use Square availability search as final truth,
    because that search can omit a valid empty slot. Instead it lists actual
    bookings for that day and blocks only if an active booking overlaps the
    exact requested interval for the assigned team member.
    """
    if not (date_str and time_str and SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID and SQUARE_TEAM_MEMBER_ID):
        return False

    req_start, req_end = _requested_local_interval(date_str, time_str, duration_minutes)
    if not (req_start and req_end):
        return False

    start_at_min, start_at_max = _local_day_range_to_utc(date_str)
    if not (start_at_min and start_at_max):
        return False

    params = {
        "location_id": SQUARE_LOCATION_ID,
        "team_member_id": SQUARE_TEAM_MEMBER_ID,
        "start_at_min": start_at_min,
        "start_at_max": start_at_max,
        "limit": 100,
    }

    try:
        resp = requests.get(
            "https://connect.squareup.com/v2/bookings",
            headers=square_headers(),
            params=params,
            timeout=12,
        )
        if resp.status_code not in (200, 201):
            print("[WARN] Square booking conflict check failed:", resp.status_code, resp.text)
            return False

        data = resp.json() or {}
        bookings = data.get("bookings") or []
        tz = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc

        for booking in bookings:
            if not isinstance(booking, dict):
                continue
            if not _square_booking_is_active(booking.get("status") or ""):
                continue

            start_raw = booking.get("start_at") or ""
            if not start_raw:
                continue
            try:
                existing_start = datetime.strptime(start_raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(tz)
            except Exception:
                continue

            segs = booking.get("appointment_segments") or []
            if not segs or not isinstance(segs, list):
                existing_end = existing_start + timedelta(minutes=60)
                if req_start < existing_end and req_end > existing_start:
                    return True
                continue

            for seg in segs:
                if not isinstance(seg, dict):
                    continue
                team_member_id = (seg.get("team_member_id") or "").strip()
                if team_member_id and team_member_id != SQUARE_TEAM_MEMBER_ID:
                    continue
                duration = int(seg.get("duration_minutes") or 60)
                existing_end = existing_start + timedelta(minutes=duration)
                if req_start < existing_end and req_end > existing_start:
                    print("[BLOCKED] Existing Square booking overlaps requested slot:", booking.get("id"), start_raw)
                    return True
    except Exception as e:
        print("[WARN] Square booking conflict check exception:", repr(e))
        return False

    return False


def build_slot_unavailable_result(sched: dict, scheduled_date: str, scheduled_time: str, appointment_type: str, *, reason: str = "slot_unavailable") -> dict:
    """Offer same-day nearby slots, then same weekday future slots, for a blocked requested slot."""
    try:
        day_avails = search_square_availability_for_day(scheduled_date, appointment_type)
        same_day_options = _nearest_same_day_slots(day_avails, scheduled_time, limit=3)
        rolled_slots = []
        if not same_day_options:
            rolled_slots = _rolling_same_weekday_slots(scheduled_date, appointment_type, limit=3)
        offered = same_day_options or rolled_slots
        if offered:
            sched["awaiting_slot_offer_choice"] = True
            sched["offered_slot_options"] = offered
            sched["last_slot_unavailable_message"] = _format_slot_offer_message(
                scheduled_date,
                scheduled_time,
                same_day_options,
                rolled_slots,
            )
            return {
                "status": "slot_unavailable",
                "message": sched["last_slot_unavailable_message"],
                "options": offered,
                "reason": reason,
            }
    except Exception as e:
        print("[WARN] slot unavailable helper failed:", repr(e))

    return {
        "status": "slot_unavailable",
        "message": "That time is already booked. What other day or time works for you?",
        "reason": reason,
    }



def maybe_apply_offered_slot_selection(conv: dict, inbound_text: str) -> bool:
    """Lock one of the currently offered slots when the customer chooses it.

    Handles natural replies like:
    - "first one"
    - "Tuesday works"
    - "Tuesday at 2"
    - "2pm"
    - "May 12"

    This must run before Step 4 and before exact-slot fallback so the system does
    not treat an offered-slot choice as a new scheduling request or reopen price.
    """
    sched = conv.setdefault("sched", {})
    if not sched.get("awaiting_slot_offer_choice"):
        return False

    options = sched.get("offered_slot_options") or []
    if not options:
        sched["awaiting_slot_offer_choice"] = False
        return False

    txt = (inbound_text or "").strip()
    low = _intent_text(txt)
    explicit_time = extract_explicit_time_from_text(txt)
    requested_date = salvage_relative_date_from_text(txt)
    option_dates = {str(opt.get("date") or "").strip() for opt in options if opt.get("date")}

    # A real date request that is not one of the offered dates is not a slot choice.
    # This prevents "first Monday of July" from matching the word "first" and
    # selecting option 1 from a May fallback list.
    if requested_date and requested_date not in option_dates:
        return False

    chosen = None

    ordinal_map = [
        ("first", 0), ("1st", 0), ("option 1", 0), ("number 1", 0), ("one", 0),
        ("second", 1), ("2nd", 1), ("option 2", 1), ("number 2", 1), ("two", 1),
        ("third", 2), ("3rd", 2), ("option 3", 2), ("number 3", 2), ("three", 2),
    ]
    for key, idx in ordinal_map:
        if re.search(rf"\b{re.escape(key)}\b", low):
            if idx < len(options):
                chosen = options[idx]
                break

    if chosen is None and re.fullmatch(r"\d+", low or ""):
        idx = int(low) - 1
        if 0 <= idx < len(options):
            chosen = options[idx]

    # Bare/custom time with an active option list means: use first offered date at that time.
    # Example: offered Mon/Tue/Wed, customer says "9", "4", or "can you do 3 instead".
    if chosen is None and explicit_time and not requested_date:
        time_only = bool(re.fullmatch(r"(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?", low)) or bool(re.search(r"\b(?:can|could)\s+you\s+do\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", low))
        if time_only and not re.fullmatch(r"[123]", low or ""):
            chosen = dict(options[0])
            chosen["time"] = explicit_time
            try:
                d = datetime.strptime(str(chosen.get("date") or ""), "%Y-%m-%d")
                chosen["label"] = f"{d.strftime('%A, %B %d').replace(' 0', ' ')} at {humanize_time(explicit_time)}"
            except Exception:
                chosen["label"] = f"{chosen.get('date') or 'that day'} at {humanize_time(explicit_time)}"

    def _slot_day_names(opt):
        try:
            d = datetime.strptime(str(opt.get("date") or ""), "%Y-%m-%d")
            return {d.strftime("%A").lower(), d.strftime("%a").lower()}
        except Exception:
            return set()

    # Match a specific offered date/day, optionally with a time.
    if chosen is None:
        matches = []
        for opt in options:
            opt_date = str(opt.get("date") or "").strip()
            opt_time = str(opt.get("time") or "").strip()
            day_match = any(re.search(rf"\b{re.escape(day)}\b", low) for day in _slot_day_names(opt))
            date_match = bool(requested_date and requested_date == opt_date)
            label_match = bool((opt.get("label") or "").strip().lower() and (opt.get("label") or "").strip().lower() in low)
            if day_match or date_match or label_match:
                if explicit_time and explicit_time != opt_time:
                    continue
                matches.append(opt)
        if len(matches) == 1:
            chosen = matches[0]

    # Match by time when the offered times are unique. This catches "2pm".
    if chosen is None and explicit_time:
        matches = [opt for opt in options if str(opt.get("time") or "").strip() == explicit_time]
        if len(matches) == 1:
            chosen = matches[0]

    if not chosen:
        return False

    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195").upper()
    sched["appointment_type"] = appt
    conv["appointment_type"] = appt
    sched["scheduled_date"] = chosen.get("date")
    sched["scheduled_time"] = chosen.get("time")
    sched["scheduled_time_source"] = "offered_slot"
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    sched["final_confirmation_sent"] = False
    sched["final_confirmation_accepted"] = False
    sched["last_final_confirmation_key"] = None
    sched["slot_choice_locked"] = True
    sched["booking_attempt_nonce"] = str(uuid.uuid4())
    return True



# ---------------------------------------------------
# Business Hours Validator
# ---------------------------------------------------
def is_within_normal_hours(time_str: str) -> bool:
    try:
        dt = datetime.strptime(time_str, "%H:%M")
        minutes = dt.hour * 60 + dt.minute
        # Allow exact 4:00 PM starts; block 4:30 PM and later.
        return (9 * 60) <= minutes <= (16 * 60)
    except:
        return False


def is_weekend(date_str: str) -> bool:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() >= 5
    except:
        return False


# ---------------------------------------------------
# Create Square Booking (Corrected + Step 5B Compatible) — PATCHED (requires house number)
#   ✅ Creates booking
#   ✅ Retrieves booking to confirm status/version
#   ✅ Accepts booking if needed (often required to trigger notifications + calendar sync)
#   ✅ Only sets booking_created / square_booking_id AFTER verify+accept succeeds
#   ✅ Prevents stale booking flags from causing ghost confirmations
# ---------------------------------------------------
def maybe_create_square_booking(phone: str, convo: dict):
    try:
        log_event("BOOKING_ATTEMPT", phone, {}, convo)
    except Exception:
        pass
    import re

    sched = convo.setdefault("sched", {})
    if sched.get("booking_allowed") is False or sched.get("customer_hard_stop") or convo.get("thread_type") in {"closed_lost", "vendor_sales_or_spam", "wrong_number_or_spam", "employment_inquiry", "commercial_bid_contact", "manual_only"}:
        return _monitor_booking_return(phone, "booking_blocked_by_thread_state", {"status": "booking_blocked_by_thread_state", "thread_type": convo.get("thread_type"), "closed_reason": sched.get("closed_reason")}, convo)
    profile = convo.setdefault("profile", {})
    current_job = convo.setdefault("current_job", {})

    profile.setdefault("addresses", [])
    profile.setdefault("past_jobs", [])
    profile.setdefault("upcoming_appointment", None)

    profile.setdefault("first_name", None)
    profile.setdefault("last_name", None)
    profile.setdefault("email", None)
    profile.setdefault("active_first_name", None)
    profile.setdefault("active_last_name", None)
    profile.setdefault("active_email", None)
    profile.setdefault("known_people", [])

    sched.setdefault("awaiting_slot_offer_choice", False)
    sched.setdefault("offered_slot_options", [])
    sched.setdefault("last_slot_unavailable_message", None)

    active_first = (profile.get("active_first_name") or profile.get("first_name") or "").strip()
    active_last = (profile.get("active_last_name") or profile.get("last_name") or "").strip()
    active_email = (profile.get("active_email") or profile.get("email") or "").strip()

    profile["first_name"] = active_first
    profile["last_name"] = active_last
    profile["email"] = active_email

    if not ((active_first and active_last) or profile.get("square_customer_id")):
        return {"status": "missing_identity"}

    if sched.get("booking_created") and sched.get("square_booking_id"):
        return {"status": "already_booked"}

    sched["booking_created"] = False
    sched["square_booking_id"] = None

    scheduled_date = sched.get("scheduled_date")
    scheduled_time = sched.get("scheduled_time")
    raw_address = (sched.get("raw_address") or "").strip()
    appointment_type = sched.get("appointment_type")

    if not sched.get("address_verified"):
        return {"status": "address_not_verified"}

    if not (scheduled_date and scheduled_time and appointment_type):
        return {"status": "missing_fields"}

    if appointment_type != "TROUBLESHOOT_395" and not sched.get("scheduled_time_source"):
        return {"status": "missing_explicit_time"}

    variation_id, variation_version = map_appointment_type_to_variation(appointment_type)
    if not variation_id:
        print("[ERROR] Unknown appt_type:", appointment_type)
        return {"status": "unknown_appointment_type"}

    if is_weekend(scheduled_date) and appointment_type != "TROUBLESHOOT_395":
        print("[BLOCKED] Weekend not allowed for non-emergency.")
        return _monitor_booking_return(phone, "weekend_blocked", {"status": "weekend_blocked"}, convo)

    if appointment_type != "TROUBLESHOOT_395":
        if not is_within_normal_hours(scheduled_time):
            print("[BLOCKED] Time outside 9–4")
            return _monitor_booking_return(phone, "outside_hours", {"status": "outside_hours"}, convo)

    if not (SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID and SQUARE_TEAM_MEMBER_ID):
        print("[ERROR] Square not configured.")
        return {"status": "square_not_configured"}

    # IMPORTANT: Do NOT preflight-reject an exact customer-selected slot merely because
    # Square's availability search did not list it. In production, that search can omit
    # valid bookable times depending on service variation, team member, duration, or API
    # availability-window behavior. The source of truth is the actual Square create-booking
    # response. Only offer alternate slots after Square rejects the create request.
    exact_avail = {}

    addr_struct = sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None

    def _is_complete_norm(a: dict) -> bool:
        return bool(
            isinstance(a, dict)
            and (a.get("address_line_1") or "").strip()
            and (a.get("locality") or "").strip()
            and (a.get("administrative_district_level_1") or "").strip()
            and (a.get("postal_code") or "").strip()
        )

    def _has_house_number(line1: str) -> bool:
        line1 = (line1 or "").strip()
        return bool(re.match(r"^\d{1,6}\b", line1))

    if not _is_complete_norm(addr_struct):
        if not raw_address:
            sched["address_verified"] = False
            sched["address_missing"] = "street"
            return {"status": "missing_address"}

        try:
            status, fresh = normalize_address(raw_address)
        except Exception as e:
            print("[ERROR] normalize_address exception:", repr(e))
            return {"status": "normalize_exception"}

        if status == "needs_state":
            send_sms(phone, "Just to confirm, is this address in Connecticut or Massachusetts?")
            sched["address_verified"] = False
            sched["address_missing"] = "state"
            return {"status": "needs_state"}

        if status != "ok" or not isinstance(fresh, dict):
            print("[ERROR] Address normalization failed. status=", status)
            sched["address_verified"] = False
            sched["address_missing"] = "confirm"
            return {"status": "address_normalization_failed"}

        addr_struct = fresh
        sched["normalized_address"] = addr_struct

    if not _is_complete_norm(addr_struct):
        print("[ERROR] Normalized address missing required fields.")
        sched["address_verified"] = False
        sched["address_missing"] = "confirm"
        return {"status": "address_incomplete"}

    line1 = (addr_struct.get("address_line_1") or "").strip()
    if not _has_house_number(line1):
        sched["address_verified"] = False
        sched["address_missing"] = "number"
        return {"status": "missing_house_number"}

    booking_address = dict(addr_struct)
    booking_address.pop("country", None)

    origin = TECH_CURRENT_ADDRESS or DISPATCH_ORIGIN_ADDRESS
    if origin:
        destination = (
            f"{(addr_struct.get('address_line_1') or '').strip()}, "
            f"{(addr_struct.get('locality') or '').strip()}, "
            f"{(addr_struct.get('administrative_district_level_1') or '').strip()} "
            f"{(addr_struct.get('postal_code') or '').strip()}"
        ).strip()

        travel_minutes = compute_travel_time_minutes(origin, destination)
        if travel_minutes and travel_minutes > MAX_TRAVEL_MINUTES:
            print("[BLOCKED] Travel too long.")
            return {"status": "travel_too_long"}

    customer_id = square_create_or_get_customer(phone, profile, addr_struct)
    if not customer_id:
        print("[ERROR] Can't create/fetch customer.")
        return {"status": "customer_unavailable"}

    start_at_utc = parse_local_datetime(scheduled_date, scheduled_time)
    if not start_at_utc:
        print("[ERROR] Invalid start time.")
        return {"status": "invalid_start_time"}

    # Preflight only against actual existing Square bookings, not availability search.
    # This prevents duplicate bookings at an already-booked exact slot while avoiding
    # the earlier false-negative issue where Square availability omitted an empty time.
    if appointment_type != "TROUBLESHOOT_395":
        if square_slot_has_existing_booking(scheduled_date, scheduled_time, appointment_type, duration_minutes=60):
            return build_slot_unavailable_result(
                sched,
                scheduled_date,
                scheduled_time,
                appointment_type,
                reason="existing_square_booking_conflict",
            )

    booking_nonce = (sched.get("booking_attempt_nonce") or "").strip()
    if not booking_nonce:
        booking_nonce = "base"

    idempotency_key = (
        f"prevolt-{phone}-{scheduled_date}-{scheduled_time}-{appointment_type}-"
        f"{(addr_struct.get('address_line_1') or '').strip()}-"
        f"{(addr_struct.get('postal_code') or '').strip()}-"
        f"{booking_nonce}"
    )

    booking_payload = {
        "idempotency_key": idempotency_key,
        "booking": {
            "location_id": SQUARE_LOCATION_ID,
            "customer_id": customer_id,
            "start_at": exact_avail.get("start_at") or start_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location_type": "CUSTOMER_LOCATION",
            "address": booking_address,
            "appointment_segments": [
                {
                    "duration_minutes": exact_avail.get("duration_minutes") or 60,
                    "service_variation_id": exact_avail.get("service_variation_id") or variation_id,
                    "service_variation_version": exact_avail.get("service_variation_version") or variation_version,
                    "team_member_id": exact_avail.get("team_member_id") or SQUARE_TEAM_MEMBER_ID
                }
            ],
            "customer_note": (
                "Auto-booked by Prevolt OS. "
                f"Raw address: {raw_address or '[none]'} | "
                f"Normalized: {addr_struct.get('address_line_1')}, {addr_struct.get('locality')}, "
                f"{(addr_struct.get('administrative_district_level_1') or '').strip()} {(addr_struct.get('postal_code') or '').strip()}"
            )
        }
    }

    try:
        resp = requests.post(
            "https://connect.squareup.com/v2/bookings",
            headers=square_headers(),
            json=booking_payload,
            timeout=12,
        )
        if resp.status_code not in (200, 201):
            print("[ERROR] Square booking failed:", resp.status_code, resp.text)

            # Only now, after Square rejects the actual create request, offer alternate
            # availability. This avoids false "booked" messages when the availability
            # search omits a valid empty time.
            if appointment_type != "TROUBLESHOOT_395":
                try:
                    day_avails = search_square_availability_for_day(scheduled_date, appointment_type)
                    same_day_options = _nearest_same_day_slots(day_avails, scheduled_time, limit=3)
                    rolled_slots = []
                    if not same_day_options:
                        rolled_slots = _rolling_same_weekday_slots(scheduled_date, appointment_type, limit=3)
                    offered = same_day_options or rolled_slots
                    if offered:
                        sched["awaiting_slot_offer_choice"] = True
                        sched["offered_slot_options"] = offered
                        sched["last_slot_unavailable_message"] = _format_slot_offer_message(
                            scheduled_date,
                            scheduled_time,
                            same_day_options,
                            rolled_slots
                        )
                        return {
                            "status": "slot_unavailable",
                            "message": sched["last_slot_unavailable_message"],
                            "options": offered,
                            "http_status": resp.status_code,
                            "body": resp.text,
                        }
                except Exception as e:
                    print("[WARN] slot fallback after create failure failed:", repr(e))

            return {"status": "create_booking_failed", "http_status": resp.status_code, "body": resp.text}

        data = resp.json()
        booking = data.get("booking") or {}
        booking_id = booking.get("id")

        if not booking_id:
            print("[ERROR] booking_id missing. payload=", data)
            return {"status": "missing_booking_id"}

        r2 = requests.get(
            f"https://connect.squareup.com/v2/bookings/{booking_id}",
            headers=square_headers(),
            timeout=12,
        )
        if r2.status_code not in (200, 201):
            print("[ERROR] retrieve booking failed:", r2.status_code, r2.text)
            return {"status": "retrieve_failed", "http_status": r2.status_code, "body": r2.text}

        b2 = (r2.json() or {}).get("booking") or {}
        status = (b2.get("status") or "").upper()
        version = b2.get("version")

        try:
            segs = b2.get("appointment_segments") or []
            tm = segs[0].get("team_member_id") if segs and isinstance(segs, list) else None
        except Exception:
            tm = None

        print("[DEBUG] booking retrieved:", {
            "id": booking_id,
            "status": status,
            "version": version,
            "location_id": b2.get("location_id"),
            "team_member_id": tm
        })

        # If Square gives back a cancelled booking record for this idempotency key,
        # treat it as stale and do not mark the booking as created.
        if status == "CANCELLED_BY_CUSTOMER":
            print("[WARN] stale cancelled booking returned by Square; forcing fresh booking attempt")
            sched["booking_created"] = False
            sched["square_booking_id"] = None
            sched["booking_attempt_nonce"] = str(uuid.uuid4())
            return _monitor_booking_return(phone, "stale_cancelled", {"status": "stale_cancelled"}, convo)

        # Do not attempt to update booking.status here.
        # The current Square response is showing booking.status as read-only on update,
        # which turns a successful create into a false failure at the very end.
        if status not in ("ACCEPTED", "CONFIRMED"):
            print("[WARN] booking retrieved with non-final status; proceeding without status update:", status)

        sched["booking_created"] = True
        sched["square_booking_id"] = booking_id
        sched["awaiting_slot_offer_choice"] = False
        sched["offered_slot_options"] = []
        sched["last_slot_unavailable_message"] = None

        profile["upcoming_appointment"] = {
            "date": scheduled_date,
            "time": scheduled_time,
            "type": appointment_type,
            "square_id": booking_id,
            "first_name": active_first,
            "last_name": active_last,
            "email": active_email
        }

        if current_job.get("job_type"):
            profile.setdefault("past_jobs", []).append({
                "type": current_job["job_type"],
                "date": scheduled_date
            })

        print("[SUCCESS] Booking created and accepted:", booking_id)
        return _monitor_booking_return(phone, "booked", {"status": "booked", "booking_id": booking_id}, convo)

    except Exception as e:
        print("[ERROR] Square exception:", repr(e))
        return _monitor_booking_return(phone, "square_exception", {"status": "square_exception", "detail": repr(e)}, convo)



@app.route("/monitor/health", methods=["GET"])
def monitor_health():
    return {"ok": True}, 200


@app.route("/monitor/voice-health", methods=["GET"])
def monitor_voice_health():
    ready, reason = voice_agent_runtime_ready()
    return {
        "ok": bool(ready),
        "reason": reason,
        "enabled": PREVOLT_VOICE_AGENT_ENABLED,
        "flask_sock": Sock is not None,
        "websocket_client": websocket_client is not None,
        "model": OPENAI_REALTIME_MODEL,
        "voice": OPENAI_REALTIME_VOICE,
        "thinking_sound_enabled": VOICE_AGENT_THINKING_SOUND_ENABLED,
        "max_customer_turns": VOICE_AGENT_MAX_CUSTOMER_TURNS,
        "vad_silence_ms": VOICE_AGENT_VAD_SILENCE_MS,
        "vad_threshold": VOICE_AGENT_VAD_THRESHOLD,
        "residential_max_travel_minutes": VOICE_RESIDENTIAL_MAX_TRAVEL_MINUTES,
    }, (200 if ready else 503)


# ---------------------------------------------------
# Prevolt Command Center Security + Control Endpoints
# ---------------------------------------------------
def _command_center_authorized() -> bool:
    """Require the Command Center API key for monitor/control routes.
    Set COMMAND_CENTER_API_KEY in Render and the same value in the desktop app settings.
    """
    required = (os.environ.get("COMMAND_CENTER_API_KEY") or "").strip()
    supplied = (
        request.headers.get("X-Prevolt-Command-Key")
        or request.args.get("command_key")
        or request.form.get("command_key")
        or ""
    ).strip()
    return bool(required) and bool(supplied) and supplied == required


def _command_center_auth_error():
    return {"ok": False, "error": "unauthorized_command_center"}, 401


def _cc_phone(raw: str) -> str:
    raw = (raw or "").strip().replace("whatsapp:", "")
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    if raw.startswith("+"):
        return raw
    return "+" + digits if digits else raw


def _safe_twilio_recording_url(raw_url: str) -> str | None:
    """Return a safe Twilio recording media URL or None.

    Twilio Recording URLs require HTTP Basic Auth. The desktop app should
    never open api.twilio.com directly because the browser will prompt for credentials
    for Twilio credentials. Instead Command Center calls this Render proxy;
    Render holds TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN securely.
    """
    raw = str(raw_url or "").strip()
    if not raw:
        return None
    # Permit only the authenticated Twilio API recording resource for this account.
    acct = (TWILIO_ACCOUNT_SID or "").strip()
    if not acct or "api.twilio.com" not in raw or "/Recordings/" not in raw:
        return None
    if f"/Accounts/{acct}/" not in raw:
        return None
    # Strip query fragments and force playable MP3 media.
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    if not (raw.endswith(".mp3") or raw.endswith(".wav")):
        raw = raw + ".mp3"
    return raw


@app.route("/monitor/recording-audio", methods=["GET"])
def monitor_recording_audio():
    """Secure proxy for voicemail/call recording playback.

    The desktop app sends a Twilio recording URL. Render validates that it is a
    recording URL for this Twilio account, fetches it with Twilio credentials,
    and streams the MP3 back to the desktop app. No Twilio secrets are stored on
    the PC and the browser no longer shows a Twilio login prompt.
    """
    if not _command_center_authorized():
        return _command_center_auth_error()
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return {"ok": False, "error": "twilio_auth_not_configured"}, 500
    recording_url = request.args.get("url") or request.args.get("recording_url") or ""
    media_url = _safe_twilio_recording_url(recording_url)
    if not media_url:
        return {"ok": False, "error": "invalid_or_unsafe_recording_url"}, 400
    try:
        rr = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
        if rr.status_code >= 400:
            return {"ok": False, "error": "twilio_recording_fetch_failed", "status_code": rr.status_code, "detail": rr.text[:300]}, 502
        content_type = rr.headers.get("Content-Type") or "audio/mpeg"
        return Response(rr.content, status=200, mimetype=content_type, headers={
            "Cache-Control": "private, max-age=60",
            "Content-Disposition": "inline; filename=prevolt-recording.mp3",
        })
    except Exception as e:
        return {"ok": False, "error": "recording_proxy_failed", "detail": repr(e)}, 500


@app.route("/monitor/events", methods=["GET"])
def monitor_events():
    if not _command_center_authorized():
        return _command_center_auth_error()
    init_monitor_db()
    limit_raw = request.args.get("limit", "250")
    phone = (request.args.get("phone") or "").replace("whatsapp:", "").strip()
    try:
        limit = max(1, min(1000, int(limit_raw)))
    except Exception:
        limit = 250

    query = "SELECT id, ts, event_type, phone, payload_json FROM monitor_events"
    params = []
    if phone:
        query += " WHERE phone = ?"
        params.append(phone)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    items = []
    try:
        with _monitor_connect() as conn:
            rows = conn.execute(query, params).fetchall()
        for row in rows:
            payload = {}
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                payload = {}
            items.append({
                "id": row["id"],
                "ts": row["ts"],
                "event_type": row["event_type"],
                "phone": row["phone"],
                "payload": payload,
            })
    except Exception as e:
        return {"ok": False, "error": repr(e), "items": []}, 500

    return {"ok": True, "items": items}, 200


@app.route("/monitor/conversations", methods=["GET"])
def monitor_conversations():
    if not _command_center_authorized():
        return _command_center_auth_error()
    items = []
    for phone, conv in conversations.items():
        if not isinstance(conv, dict):
            continue
        try:
            items.append(_conversation_snapshot(phone, conv))
        except Exception:
            continue

    items.sort(key=lambda x: (
        0 if x.get("state") == "emergency" else 1,
        0 if x.get("booking_created") else 1,
        x.get("phone") or "",
    ))
    return {"ok": True, "items": items}, 200



@app.route("/monitor/conversation/close", methods=["POST"])
def monitor_close_conversation():
    if not _command_center_authorized():
        return _command_center_auth_error()
    data = request.get_json(silent=True) or request.form or {}
    phone = (data.get("phone") or "").replace("whatsapp:", "").strip()
    reason = (data.get("reason") or "manual_closed").strip() or "manual_closed"
    if not phone or phone not in conversations:
        return {"ok": False, "error": "conversation_not_found"}, 404
    conv = conversations.setdefault(phone, {})
    sched = conv.setdefault("sched", {})
    conv["thread_type"] = "closed_lost"
    sched["customer_hard_stop"] = True
    sched["booking_allowed"] = False
    sched["pending_step"] = None
    sched["closed_reason"] = reason
    try:
        log_event("MONITOR_CLOSED_CONVERSATION", phone, {"reason": reason}, conv)
    except Exception:
        pass
    return {"ok": True, "conversation": _conversation_snapshot(phone, conv)}, 200



@app.route("/monitor/send-sms", methods=["POST"])
def monitor_send_sms():
    if not _command_center_authorized():
        return _command_center_auth_error()
    data = request.get_json(silent=True) or request.form or {}
    to_phone = _cc_phone(data.get("to") or data.get("phone") or "")
    body = str(data.get("body") or "").strip()
    if not to_phone or not body:
        return {"ok": False, "error": "missing_to_or_body"}, 400
    if not twilio_client or not TWILIO_FROM_NUMBER:
        return {"ok": False, "error": "twilio_not_configured"}, 500
    try:
        msg = twilio_client.messages.create(to=to_phone, from_=TWILIO_FROM_NUMBER, body=body)
        conv = conversations.setdefault(to_phone, {"profile": {}, "sched": {}})
        conv["last_sms_body"] = body
        log_event("MANUAL_SMS_OUT", to_phone, {"body": _safe_monitor_text(body), "sid": getattr(msg, "sid", None), "source": "command_center"}, conv)
        return {"ok": True, "sid": getattr(msg, "sid", None), "to": to_phone}, 200
    except Exception as e:
        try:
            log_event("MANUAL_SMS_FAILED", to_phone, {"body": _safe_monitor_text(body), "error": repr(e), "source": "command_center"})
        except Exception:
            pass
        return {"ok": False, "error": "twilio_sms_failed", "detail": repr(e)}, 500


@app.route("/monitor/start-call", methods=["POST"])
def monitor_start_call():
    """
    Secure click-to-call bridge:
    1. Command Center asks Render to start the call.
    2. Twilio calls the operator phone first.
    3. When answered, Twilio fetches /monitor/twiml/bridge and dials the customer.
    """
    if not _command_center_authorized():
        return _command_center_auth_error()
    data = request.get_json(silent=True) or request.form or {}
    customer = _cc_phone(data.get("to") or data.get("customer_phone") or data.get("phone") or "")
    operator = _cc_phone(data.get("operator_phone") or data.get("operator") or os.environ.get("COMMAND_CENTER_OPERATOR_PHONE") or "")
    if not customer or not operator:
        return {"ok": False, "error": "missing_customer_or_operator_phone"}, 400
    if not twilio_client or not TWILIO_FROM_NUMBER:
        return {"ok": False, "error": "twilio_not_configured"}, 500

    from urllib.parse import quote_plus
    base = (os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or request.url_root).rstrip("/")
    bridge_token = (os.environ.get("PREVOLT_CALL_BRIDGE_TOKEN") or "").strip()
    if not bridge_token:
        return {"ok": False, "error": "missing_prevolt_call_bridge_token"}, 500
    twiml_url = f"{base}/monitor/twiml/bridge?to={quote_plus(customer)}&token={quote_plus(bridge_token)}"
    try:
        call = twilio_client.calls.create(
            to=operator,
            from_=TWILIO_FROM_NUMBER,
            url=twiml_url,
            method="POST",
        )
        conv = conversations.setdefault(customer, {"profile": {}, "sched": {}})
        log_event("MONITOR_CALL_STARTED", customer, {
            "to": customer,
            "operator_phone": operator,
            "call_sid": getattr(call, "sid", None),
            "source": "command_center",
        }, conv)
        return {"ok": True, "call_sid": getattr(call, "sid", None), "to": customer, "operator_phone": operator}, 200
    except Exception as e:
        return {"ok": False, "error": "twilio_call_failed", "detail": repr(e)}, 500


@app.route("/monitor/twiml/bridge", methods=["GET", "POST"])
def monitor_twiml_bridge():
    token = (request.values.get("token") or "").strip()
    expected = (os.environ.get("PREVOLT_CALL_BRIDGE_TOKEN") or "").strip()
    vr = VoiceResponse()
    if not expected or token != expected:
        vr.say("Unauthorized Prevolt call bridge request.")
        return Response(str(vr), mimetype="text/xml")
    customer = _cc_phone(request.values.get("to") or "")
    if not customer:
        vr.say("No customer phone number was provided.")
        return Response(str(vr), mimetype="text/xml")
    vr.say("Connecting Prevolt Command Center.")
    dial = Dial(caller_id=TWILIO_FROM_NUMBER, answer_on_bridge=True, timeout=25)
    dial.number(customer)
    vr.append(dial)
    return Response(str(vr), mimetype="text/xml")


@app.route("/monitor/conversation/status", methods=["POST"])
def monitor_conversation_status():
    if not _command_center_authorized():
        return _command_center_auth_error()
    data = request.get_json(silent=True) or request.form or {}
    phone = _cc_phone(data.get("phone") or "")
    status = str(data.get("status") or "manual_only").strip() or "manual_only"
    reason = str(data.get("reason") or status).strip() or status
    if not phone:
        return {"ok": False, "error": "missing_phone"}, 400
    conv = conversations.setdefault(phone, {"profile": {}, "sched": {}})
    sched = conv.setdefault("sched", {})

    if status in {"closed_lost", "vendor_sales_or_spam", "wrong_number_or_spam", "spam", "found_someone"}:
        conv["thread_type"] = "vendor_sales_or_spam" if status in {"vendor_sales_or_spam", "spam"} else "closed_lost"
        sched["customer_hard_stop"] = True
        sched["booking_allowed"] = False
        sched["pending_step"] = None
        sched["closed_reason"] = reason
    elif status in {"manual_only", "manual", "needs_callback", "callback"}:
        conv["thread_type"] = "manual_only"
        conv["manual_takeover"] = True
        sched["manual_only"] = True
        sched["manual_reason"] = reason
        sched["booking_allowed"] = False
        sched["pending_step"] = None
        sched["awaiting_slot_offer_choice"] = False
        sched["offered_slot_options"] = []
    elif status == "booked_manual":
        sched["booking_created"] = True
        sched["manual_booked"] = True
        sched["manual_reason"] = reason
    else:
        sched["manual_status"] = status
        sched["manual_reason"] = reason

    log_event("MONITOR_STATUS_UPDATED", phone, {"status": status, "reason": reason, "source": "command_center"}, conv)
    return {"ok": True, "conversation": _conversation_snapshot(phone, conv)}, 200


@app.route("/monitor/conversation/note", methods=["POST"])
def monitor_conversation_note():
    if not _command_center_authorized():
        return _command_center_auth_error()
    data = request.get_json(silent=True) or request.form or {}
    phone = _cc_phone(data.get("phone") or "")
    note = str(data.get("note") or "").strip()
    if not phone or not note:
        return {"ok": False, "error": "missing_phone_or_note"}, 400
    conv = conversations.setdefault(phone, {"profile": {}, "sched": {}})
    conv.setdefault("notes", []).append({"ts": _monitor_now_iso(), "note": note, "source": "command_center"})
    log_event("MONITOR_NOTE", phone, {"note": _safe_monitor_text(note, 800), "source": "command_center"}, conv)
    return {"ok": True}, 200


# ---------------------------------------------------
# Local Development Entrypoint
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
# ---------------------------------------------------
# Voice Agent v26 hardening overrides
# Appended after v25 so process_prevolt_voice_turn resolves these globals at runtime.
# ---------------------------------------------------

def _voice_looks_like_true_hazard(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    # Smoke detector / smoke alarm installation or replacement is ordinary evaluation work,
    # not an active smoke/fire emergency.
    if re.search(r"\b(?:smoke|co|carbon monoxide)\s+(?:detector|alarm)s?\b", low) and re.search(r"\b(?:install|installed|replace|replaced|change|changed|hardwire|quote|estimate)\b", low):
        return False
    hazard_terms = [
        r"smoke\s+(?:coming|pouring|from|out|filling)", r"started\s+smoking", r"caught\s+fire", r"active\s+fire",
        r"burning\s+(?:smell|odor)", r"smells?\s+like\s+smoke", r"burnt\s+(?:wire|outlet|breaker|panel)",
        r"hot\s+to\s+the\s+touch", r"hot\s+(?:outlet|panel|breaker|device|plug)",
        r"sparking|sparks|arcing|melted", r"panel\s+is\s+hot", r"outlet\s+is\s+hot",
        r"no\s+power", r"partial\s+outage", r"half\s+the\s+house\s+has\s+no\s+power",
        r"breaker\s+(?:won'?t|will\s+not|keeps?)\s+(?:reset|tripping)",
        r"water\s+(?:in|inside|got\s+into|leaking\s+into)\s+(?:the\s+)?(?:panel|breaker|electrical)",
        r"(?:panel|breaker|electrical)\s+(?:is\s+)?wet", r"flood(?:ing)?\s+(?:near|around)\s+(?:outlets?|panel|electrical)",
        r"service\s+(?:drop|entrance)\s+(?:ripped|torn|down)", r"meter\s+(?:socket\s+)?(?:ripped|damaged)",
        r"power\s+line\s+down", r"tree\s+ripped\s+wires", r"ev\s+charger\s+(?:plug\s+)?melted",
    ]
    return any(re.search(p, low, flags=re.I) for p in hazard_terms)


def _voice_looks_like_multiple_addresses(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return bool(
        re.search(r"\b(?:multiple|several|different|many|various|four|three|two|five|couple|few|5|4|3|2)\s+(?:different\s+)?(?:rental\s+)?(?:addresses|properties|units|apartments|locations|houses|buildings|condos)\b", low)
        or re.search(r"\b(?:a\s+couple|a\s+few)\s+(?:of\s+)?(?:apartments|properties|rentals|units|addresses|locations|houses|buildings)\b", low)
        or re.search(r"\b(?:several|multiple)\s+(?:rental\s+)?(?:properties|apartments|units|addresses|locations|buildings)\b", low)
        or "more than one address" in low
        or "multiple addresses" in low
        or "different addresses" in low
        or "i have four" in low
        or "i have several" in low
        or "rental properties" in low and any(x in low for x in ["several", "multiple", "couple", "few", "two", "three", "four", "5", "4", "3", "2"])
    )


def _voice_looks_like_live_person_demand(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return any(p in low for p in [
        "speak to a person", "speak to a human", "speak with a person", "speak with a human",
        "talk to a person", "talk to a human", "talk with a person", "talk with a human",
        "real person", "live person", "human being", "representative", "operator", "customer service", "live rep", "service rep",
        "connect me to someone", "transfer me", "get someone on the phone", "i want a person", "i need a person",
        "i want to talk to someone", "i need to talk to someone", "call me back", "call me", "can kyle call", "have kyle call",
    ])


def _voice_yes_no_text(text: str) -> str | None:
    low = _intent_text(text)
    if not low:
        return None
    yes_exact = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "correct", "right", "fine", "yes please", "absolutely", "sounds good", "that works", "works for me"}
    if low in yes_exact or any(p in low for p in [
        "that works", "works for me", "go ahead", "let's do it", "lets do it", "let s do it", "book it", "schedule it",
        "move forward", "move ahead", "send someone", "dispatch", "yes please", "sounds good", "that is fine", "that's fine", "sure thing"
    ]):
        return "yes"
    no_exact = {"no", "nope", "nah", "no thanks", "not interested", "pass"}
    if low in no_exact or any(p in low for p in [
        "too much", "i'll pass", "ill pass", "not interested", "no thanks", "doesn't work", "does not work", "free quote",
        "i don't pay", "i dont pay", "do not pay", "won't pay", "wont pay", "no charge", "free estimate"
    ]):
        return "no"
    return None


def v13_extract_email(text: str) -> str | None:
    raw = str(text or "").strip()
    m = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", raw, flags=re.I)
    if m:
        return m.group(1).strip()
    # Speech transcripts often produce "name at gmail dot com".
    spoken = raw.lower()
    spoken = re.sub(r"\s+at\s+", "@", spoken)
    spoken = re.sub(r"\s+dot\s+", ".", spoken)
    spoken = re.sub(r"\s+", "", spoken)
    m = re.search(r"([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})", spoken, flags=re.I)
    return m.group(1).strip() if m else None


def _voice_remove_filler_sentences(text: str) -> str:
    t = _voice_to_sms_text(text)
    if not t:
        return t
    # Remove full internal-process narration sentences before applying original-style cleanup.
    filler_patterns = [
        r"(?:Okay|Got it|Alright|Sure)?,?\s*that sounds (?:serious|urgent|dangerous)\.\s*Let me [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I(?:'|’)m going to focus on safety [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let(?:'|’)s focus on immediate safety [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me check what we still need [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me check [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I'll just clarify [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I(?:'|’)ll just clarify [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*One moment [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Thanks for hanging on [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I(?:'|’)ll move ahead [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I(?:'|’)ll get dispatch started [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me move ahead [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me set up [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me get [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*The scheduling system [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I(?:'|’)m gonna route [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I'm going to route [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*Let me respond [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*with scheduling in mind [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I(?:'|’)m noting [^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Got it|Alright|Sure)?,?\s*I'll confirm the plan [^.?!]*(?:[.?!]|$)",
    ]
    for pat in filler_patterns:
        t = re.sub(pat, "", t, flags=re.I)
    sentences = re.split(r"(?<=[.!?])\s+", t.strip())
    out=[]; urgent_seen=False
    for sent in sentences:
        low=_intent_text(sent)
        if any(x in low for x in ["sounds urgent", "sounds serious", "sounds dangerous", "looks urgent"]):
            if urgent_seen:
                continue
            urgent_seen=True
        if low in {"okay", "ok", "got it", "all right", "alright", "sure", "thanks", "thank you"}:
            continue
        if any(x in low for x in ["let me think", "let me check", "one moment", "scheduling system", "move ahead", "thanks for hanging on", "finish booking"]):
            continue
        out.append(sent.strip())
    return re.sub(r"\s+", " ", " ".join(out)).strip()

# ---------------------------------------------------
# Voice Agent v27 chaos-test hardening overrides
# ---------------------------------------------------

def _voice_yes_no_text(text: str) -> str | None:
    low = _intent_text(text)
    raw = str(text or "").lower()
    if not low and not raw:
        return None
    yes_exact = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "correct", "right", "fine", "yes please", "absolutely", "sounds good", "that works", "works for me"}
    yes_phrases = [
        "that works", "works for me", "go ahead", "let's do it", "lets do it", "let’s do it", "book it", "schedule it",
        "move forward", "send someone", "dispatch", "yes please", "sounds good", "that is fine", "that's fine", "sure thing", "works", "do it"
    ]
    if low in yes_exact or any(p in low for p in yes_phrases) or any(p in raw for p in ["let’s do it", "that’s fine"]):
        return "yes"
    no_exact = {"no", "nope", "nah", "no thanks", "not interested", "pass"}
    no_phrases = [
        "too much", "i'll pass", "ill pass", "not interested", "no thanks", "doesn't work", "does not work", "free quote",
        "i don't pay", "i dont pay", "do not pay", "won't pay", "wont pay", "no charge", "free estimate"
    ]
    if low in no_exact or any(p in low for p in no_phrases) or any(p in raw for p in ["don’t pay", "doesn’t work", "won’t pay"]):
        return "no"
    return None


def _voice_looks_like_live_person_demand(text: str) -> bool:
    low = _intent_text(text)
    raw = str(text or "").lower()
    if not low and not raw:
        return False
    phrases = [
        "speak to a person", "speak to a human", "speak with a person", "speak with a human",
        "talk to a person", "talk to a human", "talk with a person", "talk with a human",
        "real person", "live person", "human being", "representative", "operator", "customer service", "live rep", "service rep",
        "connect me to someone", "transfer me", "get someone on the phone", "i want a person", "i need a person",
        "i want to talk to someone", "i need to talk to someone", "call me back", "call me", "can kyle call", "have kyle call",
        "do not want a bot", "don't want a bot", "dont want a bot", "do not want ai", "don't want ai", "dont want ai",
        "not talking to a bot", "not talking to ai", "live agent", "human agent"
    ]
    return any(p in low for p in phrases) or any(p in raw for p in ["don’t want a bot", "don’t want ai"])


def _voice_looks_like_multiple_addresses(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return bool(
        re.search(r"\b(?:multiple|several|different|many|various|four|three|two|five|couple|few|5|4|3|2)\s+(?:different\s+)?(?:rental\s+)?(?:addresses|properties|units|apartments|locations|houses|buildings|condos)\b", low)
        or re.search(r"\b(?:a\s+couple|a\s+few)\s+(?:of\s+)?(?:apartments|properties|rentals|units|addresses|locations|houses|buildings)\b", low)
        or re.search(r"\b(?:several|multiple)\s+(?:rental\s+)?(?:properties|apartments|units|addresses|locations|buildings)\b", low)
        or "more than one address" in low
        or "multiple addresses" in low
        or "different addresses" in low
        or "property portfolio" in low
        or "portfolio" in low and any(w in low for w in ["property", "properties", "rental", "rentals", "addresses"])
        or "i have four" in low
        or "i have several" in low
        or "rental properties" in low and any(x in low for x in ["several", "multiple", "couple", "few", "two", "three", "four", "5", "4", "3", "2"])
    )


def _voice_looks_like_slot_choice_text(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return bool(
        re.search(r"\b(?:first|second|third|1st|2nd|3rd|option\s*[123]|one|two|three)\b", low)
        or re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|may\s*28|may\s*29|june\s*1)\b", low)
        or re.search(r"\b\d{1,2}\s*(?:am|pm|a m|p m)\b", low)
    )


def _voice_eval_price_confirm_fast_path(conv: dict, caller_text: str) -> str | None:
    sched = conv.setdefault("sched", {})
    if not sched.get("awaiting_eval_price_confirm"):
        return None
    yn = _voice_yes_no_text(caller_text)
    # Some callers answer the pending slot choice immediately ("the second one", "Thursday")
    # after hearing/remembering openings. Treat that as price acceptance plus slot selection.
    if yn != "no" and _voice_looks_like_slot_choice_text(caller_text):
        sched["awaiting_eval_price_confirm"] = False
        try:
            slot_reply = _voice_apply_offered_slot_fast_path(conv, caller_text)
            if slot_reply:
                return slot_reply
        except Exception:
            pass
        slots = (sched.get("voice_pending_slot_offer_reply") or "").strip()
        return slots or "Which one works best for you?"
    if yn == "no":
        sched["voice_close_after_reply"] = True
        sched["closed_reason"] = "declined_eval_price"
        return "No problem. We won't schedule the visit right now. Thank you for calling Prevolt Electric. Goodbye."
    if yn != "yes":
        return "The on-site evaluation visit is $195. Does that work for you?"
    sched["awaiting_eval_price_confirm"] = False
    slots = (sched.get("voice_pending_slot_offer_reply") or "").strip()
    if slots:
        return slots
    if sched.get("awaiting_slot_offer_choice") and sched.get("offered_slot_options"):
        return f"Which one works best for you — {_format_slot_options((sched.get('offered_slot_options') or [])[:3])}?"
    return "What day and time works best for you?"

# ---------------------------------------------------
# Voice Agent v28 name pre-capture override
# ---------------------------------------------------
_ORIGINAL_PROCESS_PREVOLT_VOICE_TURN_V28 = process_prevolt_voice_turn

def _voice_precapture_name_from_intro(conv: dict, caller_text: str) -> None:
    profile = conv.setdefault("profile", {})
    if profile.get("active_first_name"):
        return
    raw = str(caller_text or "")
    patterns = [
        r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?\b",
        r"\bthis\s+is\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?\b",
        r"\bit(?:'|’)s\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?\b",
        r"\bi\s+am\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?\b",
        r"\bi(?:'|’)m\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?\b",
    ]
    stop = {"and", "calling", "looking", "from", "in", "with", "for", "need", "have", "we", "i"}
    for pat in patterns:
        m = re.search(pat, raw, flags=re.I)
        if not m:
            continue
        first = (m.group(1) or "").strip().title()
        last = (m.group(2) or "").strip().title()
        if first and first.lower() not in stop:
            profile["active_first_name"] = first
            profile["first_name"] = first
            profile["name_source"] = "voice_intro_precapture"
        if last and last.lower() not in stop:
            profile["active_last_name"] = last
            profile["last_name"] = last
        return

def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    conv = hydrate_voice_conversation((phone or "").replace("whatsapp:", "").strip(), call_sid)
    try:
        _voice_precapture_name_from_intro(conv, caller_text)
    except Exception:
        pass
    return _ORIGINAL_PROCESS_PREVOLT_VOICE_TURN_V28(phone, call_sid, caller_text)


# ---------------------------------------------------
# Voice Agent v29 production hardening overrides
# Focus: eliminate phone filler, enforce price consent, multiple-address first-location flow,
# and use Twilio TwiML for the final booked confirmation so the caller hears the goodbye.
# ---------------------------------------------------

try:
    # Live callers routinely need more room than 10-16 turns when they ask coverage/price questions
    # or have multiple addresses. Do not let a low Render env prematurely kill a legitimate call.
    VOICE_AGENT_MAX_CUSTOMER_TURNS = max(int(VOICE_AGENT_MAX_CUSTOMER_TURNS or 0), 24)
except Exception:
    VOICE_AGENT_MAX_CUSTOMER_TURNS = 24


def _voice_remove_filler_sentences(text: str) -> str:
    """Aggressively remove internal process narration from spoken replies.

    The phone assistant should sound like a receptionist asking the next needed
    question, not like an AI narrating its reasoning. This runs after SRB logic
    produces the reply, so it does not change the scheduling state.
    """
    t = _voice_to_sms_text(text)
    if not t:
        return t

    # Remove common internal-process clauses even when they appear mid-reply.
    filler_patterns = [
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*that sounds (?:serious|urgent|dangerous)\.\s*Let me (?:think|check|respond|look)[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*this sounds (?:urgent|serious|dangerous)\.\s*Let me (?:think|check|respond|look)[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I(?:'|’)m going to focus on safety[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I'm going to focus on safety[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Let(?:'|’)s focus on immediate safety[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Let me check what (?:we|I) still need[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Let me check the next detail[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Let me check the details[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Let me check[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I'll just clarify[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I(?:'|’)ll just clarify[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*One moment[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Thanks for hanging on[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I(?:'|’)ll move ahead[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I'll move ahead[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I(?:'|’)ll get dispatch started[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I'll get dispatch started[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Let me move ahead[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Let me set up[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Let me get[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*The scheduling system[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I(?:'|’)m gonna route[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I'm going to route[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Let me respond[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I'll confirm the plan[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I'll use that email[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I'm finalizing[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I'm noting[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*I've got your name[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Next,? I just need[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Thanks for that address[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Thanks for confirming[^.?!]*(?:[.?!]|$)",
        r"(?:Okay|Ok|Got it|Alright|All right|Sure|Thanks)?,?\s*Perfect\.\s*Thanks[^.?!]*(?:[.?!]|$)",
    ]
    for pat in filler_patterns:
        t = re.sub(pat, "", t, flags=re.I)

    # Sentence-level cleanup: keep one urgent acknowledgement max, remove pure acknowledgements.
    sentences = re.split(r"(?<=[.!?])\s+", t.strip())
    out = []
    urgent_seen = False
    for sent in sentences:
        s = sent.strip(" ,")
        if not s:
            continue
        low = _intent_text(s)
        if not low:
            continue
        if any(x in low for x in ["sounds urgent", "sounds serious", "sounds dangerous", "looks urgent", "sounds dangerous"]):
            if urgent_seen:
                continue
            urgent_seen = True
            # Keep only the short sentence, not any appended reasoning.
            if low not in {"this sounds urgent", "that sounds urgent", "this looks urgent", "okay that sounds urgent"}:
                s = "This sounds urgent."
        if low in {"okay", "ok", "got it", "all right", "alright", "sure", "thanks", "thank you", "perfect"}:
            continue
        if any(x in low for x in [
            "let me think", "let me check", "one moment", "scheduling system", "move ahead",
            "thanks for hanging on", "finish booking", "finish getting", "last detail", "next detail",
            "i'll use that email", "i will use that email", "i've got your name", "i have got your name"
        ]):
            continue
        out.append(s)
    cleaned = re.sub(r"\s+", " ", " ".join(out)).strip()
    return cleaned or "Sorry, can you say that again?"


def _voice_naturalize_reply(reply: str) -> str:
    text = _voice_remove_filler_sentences(_voice_to_sms_text(reply))
    if not text:
        return "Sorry, can you say that again?"
    # SMS-to-phone wording cleanup.
    text = re.sub(r"^\s*(hello|hi|hey)[, ]+(?:[A-Za-z]+[, ]+)?(?:this is|you(?:'|’)ve reached)\s+Prevolt Electric\.\s*", "", text, flags=re.I)
    replacements = [
        (r"\bI\s+can\s+help\s+you\s+right\s+here\s+by\s+text\.?", "I can help get this started."),
        (r"\bI(?:'|’)ll\s+help\s+you\s+here\s+by\s+text\.?", "I can help get this scheduled."),
        (r"\bright\s+here\s+by\s+text\b", "right here"),
        (r"\breply here\b", "tell me"),
        (r"\breply with\b", "tell me"),
        (r"\bsend over\b", "tell me"),
        (r"\bplease send (?:me )?the house number and street name for the work\.?", "what's the house number and street name for the work?"),
        (r"\bsend (?:me )?the house number and street name for the work\.?", "what's the house number and street name for the work?"),
        (r"\bWhat number is it on [^?]+\?", "What's the house number and street name for the work?"),
        (r"\bWe do charge \$195 for one of our electricians to come out, take a look, and provide you with a quote\.",
         "We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step."),
        (r"\bWhat is your first and last name\?", "What's your first and last name?"),
        (r"\bWhat is the best email address for the appointment\?", "What's the best email address for the appointment?"),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text, flags=re.I)
    # Prevent double scheduling lead-ins.
    text = re.sub(r"\bI can help get this scheduled\.\s*I can get your appointment scheduled here\.", "I can get your appointment scheduled here.", text, flags=re.I)
    text = re.sub(r"\bI can get your appointment scheduled here\.\s*I can get your appointment scheduled here\.", "I can get your appointment scheduled here.", text, flags=re.I)
    text = _voice_remove_filler_sentences(text)
    text = _voice_dedupe_consecutive_sentences(text)
    return re.sub(r"\s+", " ", text).strip() or "Sorry, can you say that again?"


def _voice_split_eval_price_and_slots(text: str) -> tuple[str, str | None]:
    raw = _voice_to_sms_text(text)
    if not raw or not re.search(r"\$?195\b", raw):
        return raw, None
    patterns = [
        r"\bWhich one works best for you\??\s*.*$",
        r"\bWe have\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[^.?!]*(?:\?|\.)?.*$",
        r"\b(?:The next openings are|Our next openings are)\s+.*$",
    ]
    for pat in patterns:
        m = re.search(pat, raw, flags=re.I | re.S)
        if m:
            price = raw[:m.start()].strip(" ,.;")
            slots = raw[m.start():].strip()
            return price, slots
    return raw.strip(), None


def _voice_hold_eval_price_for_confirmation(conv: dict, reply: str) -> str:
    """Always ask for explicit consent after $195 before offering slots."""
    sched = conv.setdefault("sched", {})
    if sched.get("awaiting_eval_price_confirm"):
        return _voice_naturalize_reply(reply)
    natural = _voice_naturalize_reply(reply)
    if not re.search(r"\$?195\b", natural or ""):
        return natural
    if re.search(r"does that work for you\?", natural, flags=re.I):
        sched["awaiting_eval_price_confirm"] = True
        return natural
    price, slots = _voice_split_eval_price_and_slots(natural)
    sched["awaiting_eval_price_confirm"] = True
    if slots:
        sched["voice_pending_slot_offer_reply"] = _voice_naturalize_reply(slots)
    price = price.strip(" ,.;")
    if not price:
        price = "We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step"
    if not price.endswith(('.', '?', '!')):
        price += "."
    return price + " Does that work for you?"


def _voice_looks_like_multiple_addresses(text: str) -> bool:
    low = _intent_text(text)
    if not low:
        return False
    return bool(
        re.search(r"\b(?:multiple|several|different|many|various|four|three|two|five|six|seven|eight|nine|ten|couple|few|\d+)\s+(?:different\s+)?(?:rental\s+)?(?:addresses|properties|units|apartments|locations|houses|buildings|condos|stores|sites)\b", low)
        or re.search(r"\b(?:a\s+couple|a\s+few)\s+(?:of\s+)?(?:apartments|properties|rentals|units|addresses|locations|houses|buildings|condos)\b", low)
        or re.search(r"\b(?:several|multiple|many|various)\s+(?:rental\s+)?(?:properties|apartments|units|addresses|locations|buildings|condos|sites)\b", low)
        or "more than one address" in low
        or "multiple addresses" in low
        or "different addresses" in low
        or "property portfolio" in low
        or ("portfolio" in low and any(w in low for w in ["property", "properties", "rental", "rentals", "addresses"]))
        or ("rental properties" in low and any(x in low for x in ["several", "multiple", "couple", "few", "two", "three", "four", "5", "4", "3", "2"]))
    )


def _voice_multiple_address_reply(conv: dict, caller_text: str) -> str | None:
    if not _voice_looks_like_multiple_addresses(caller_text):
        return None
    sched = conv.setdefault("sched", {})
    sched["multiple_addresses"] = True
    sched["pending_step"] = "need_address"
    sched["state"] = "waiting_for_address"
    return (
        "We can start with the first address and get that visit scheduled. "
        "The $195 evaluation applies to the first location, and we can handle additional addresses after that. "
        "What's the first address for the work?"
    )


def _voice_call_update_with_final_twiml(call_sid: str, reply: str, phone: str = "") -> bool:
    """Use Twilio itself to play the final confirmation and hang up.

    This avoids the recurring bug where the booking is created and the SMS is sent,
    but the realtime audio stream closes before the caller hears the final booked
    confirmation. Twilio owns this final Say/Hangup step.
    """
    if not (twilio_client and call_sid and reply):
        return False
    try:
        vr = VoiceResponse()
        vr.say(_voice_to_sms_text(reply), voice="Polly.Matthew-Neural")
        vr.hangup()
        twilio_client.calls(call_sid).update(twiml=str(vr))
        try:
            log_event("VOICE_FINAL_TWIML_UPDATE_SENT", phone, {"call_sid": call_sid, "reply": _safe_monitor_text(reply, 500)})
        except Exception:
            pass
        return True
    except Exception as e:
        try:
            log_event("VOICE_FINAL_TWIML_UPDATE_FAILED", phone, {"call_sid": call_sid, "error": repr(e), "reply": _safe_monitor_text(reply, 500)})
        except Exception:
            pass
        return False


_ORIGINAL_PROCESS_PREVOLT_VOICE_TURN_V29 = process_prevolt_voice_turn

def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    conv = hydrate_voice_conversation((phone or "").replace("whatsapp:", "").strip(), call_sid)
    sched = conv.setdefault("sched", {})
    # Highest-priority deterministic multi-address handler. Do this before the
    # generic SRB address gate can loop on "what's the house number".
    try:
        multi = _voice_multiple_address_reply(conv, caller_text)
        if multi:
            multi = _voice_naturalize_reply(multi)
            conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": multi, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
            conv["last_voice_reply"] = multi
            try:
                log_event("VOICE_MULTIPLE_ADDRESS_FAST_PATH_V29", phone, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(multi), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return {"reply_to_customer": multi, "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_MULTIPLE_ADDRESS_FAST_PATH_ERROR_V29", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid}, conv)
        except Exception:
            pass

    output = _ORIGINAL_PROCESS_PREVOLT_VOICE_TURN_V29(phone, call_sid, caller_text)
    try:
        reply = str(output.get("reply_to_customer") or "")
        # Remove backend/openai-style filler one last time and force the $195 consent question.
        if reply:
            reply = _voice_naturalize_reply(reply)
            if not output.get("booking_created") and not output.get("end_call"):
                reply = _voice_hold_eval_price_for_confirmation(conv, reply)
            output["reply_to_customer"] = reply
            conv["last_voice_reply"] = reply
        # If this turn created a booking, force our clean final phrase every time.
        if bool(conv.setdefault("sched", {}).get("booking_created") and conv.setdefault("sched", {}).get("square_booking_id")):
            final_reply = _voice_finalize_booking_reply(conv, "")
            output["reply_to_customer"] = final_reply
            output["booking_created"] = True
            output["end_call"] = True
            conv["last_voice_reply"] = final_reply
            conv.setdefault("sched", {})["voice_close_after_reply"] = True
            conv.setdefault("sched", {})["voice_booking_completed_close"] = True
    except Exception as e:
        try:
            log_event("VOICE_POSTPROCESS_ERROR_V29", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid}, conv)
        except Exception:
            pass
    return output


_ORIGINAL_HANDLE_REALTIME_FUNCTION_CALL_V29 = _handle_realtime_function_call

def _handle_realtime_function_call(openai_ws, phone: str, call_sid: str, item: dict, handled_call_ids: set | None = None, twilio_ws=None, stream_sid: str = "") -> bool:
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return False
    name = item.get("name") or ""
    if name != "prevolt_os_turn":
        return False
    call_id = item.get("call_id") or item.get("id") or ""
    if handled_call_ids is not None and call_id:
        if call_id in handled_call_ids:
            return True
        handled_call_ids.add(call_id)
    try:
        args = json.loads(item.get("arguments") or "{}")
    except Exception:
        args = {}
    try:
        if twilio_ws is not None and stream_sid:
            _send_twilio_thinking_sound(twilio_ws, stream_sid, phone, call_sid, "before_tool_processing")
    except Exception:
        pass

    output = process_prevolt_voice_turn(phone, call_sid, args.get("caller_text") or "")
    exact_reply = _voice_to_sms_text(str(output.get("reply_to_customer") or "Sorry, can you say that again?"))
    booking_final = bool(output.get("end_call") and output.get("booking_created"))

    # Always return the tool result so OpenAI's function call can complete cleanly.
    try:
        _send_openai_event(openai_ws, {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id, "output": json.dumps(output)},
        })
    except Exception:
        pass

    if booking_final and _voice_call_update_with_final_twiml(call_sid, exact_reply, phone):
        # Twilio is now responsible for saying the final confirmation and hanging up.
        # Do not also ask OpenAI to speak it, which has repeatedly caused cutoffs/races.
        try:
            conv = hydrate_voice_conversation(phone, call_sid)
            conv.setdefault("sched", {})["voice_close_after_reply"] = True
            conv.setdefault("sched", {})["voice_final_twiML_sent"] = True
        except Exception:
            pass
        return True

    if output.get("end_call"):
        try:
            sched_local = hydrate_voice_conversation(phone, call_sid).setdefault("sched", {})
            sched_local["voice_close_after_reply"] = True
            sched_local["voice_waiting_for_final_audio_done"] = True
        except Exception:
            pass

    _send_openai_event(openai_ws, {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio"],
            "tool_choice": "none",
            "instructions": (
                "Say ONLY this exact phone reply, word for word, including every question in it, then stop speaking. "
                "Do not say anything before it. Do not say you are checking, thinking, waiting, routing, processing, or finishing. "
                "Speak at a natural medium pace. Do not slow down or add dramatic pauses at commas. "
                "Do not add any acknowledgement, greeting, filler phrase, or extra question. "
                "Do not stop after the first clause or first sentence; speak the complete exact reply. "
                "Exact reply: " + json.dumps(exact_reply)
            ),
        },
    })
    return True


# ---------------------------------------------------
# Voice Agent v30 small cleanup override
# Avoid repeating the exact multi-address policy when the caller pushes back.
# ---------------------------------------------------

def _voice_multiple_address_reply(conv: dict, caller_text: str) -> str | None:
    sched = conv.setdefault("sched", {})
    low = _intent_text(caller_text)
    if not _voice_looks_like_multiple_addresses(caller_text):
        return None
    sched["multiple_addresses"] = True
    sched["pending_step"] = "need_address"
    sched["state"] = "waiting_for_address"
    if sched.get("voice_multi_address_explained"):
        return "I understand. For scheduling, we will start with the first property first. What's the first address for the work?"
    sched["voice_multi_address_explained"] = True
    return (
        "We can start with the first address and get that visit scheduled. "
        "The $195 evaluation applies to the first location, and we can handle additional addresses after that. "
        "What's the first address for the work?"
    )

# ---------------------------------------------------
# Voice Agent v31 name/email hardening overrides
# Fixes: intro name over-capture ("My name's Sean and I live...") and
# voice email misrecognition by requiring confirmation before booking.
# ---------------------------------------------------

_VOICE_BAD_NAME_WORDS_V31 = {
    "and", "or", "in", "at", "from", "with", "for", "about", "because", "the", "a", "an",
    "live", "lives", "living", "looking", "need", "needs", "want", "wants", "was", "were",
    "ev", "charger", "installed", "install", "outlet", "outlets", "panel", "smoke", "smoking",
    "windsor", "locks", "connecticut", "massachusetts", "ct", "ma", "garage", "basement",
}

def _voice_extract_intro_name_v31(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return ("", "")
    # Normalize common speech/transcript variants: "my name's", "my name s".
    pats = [
        r"\bmy\s+name\s*(?:is|'s|’s|s)?\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?",
        r"\bthis\s+is\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?",
        r"\bit(?:'|’)?s\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?",
        r"\bi\s+am\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?",
        r"\bi'm\s+([A-Za-z][A-Za-z'\-]+)(?:\s+([A-Za-z][A-Za-z'\-]+))?",
    ]
    for pat in pats:
        m = re.search(pat, raw, flags=re.I)
        if not m:
            continue
        first = normalize_person_name(m.group(1) or "")
        second = normalize_person_name(m.group(2) or "")
        if first.lower() in _VOICE_BAD_NAME_WORDS_V31:
            first = ""
        if second.lower() in _VOICE_BAD_NAME_WORDS_V31:
            second = ""
        # If the second captured word is immediately followed by a stop phrase,
        # keep it only if it is not a filler word. This handles "Sean and I live...".
        return (first, second)
    return ("", "")


def _voice_name_field_is_malformed_v31(value: str) -> bool:
    v = " ".join(str(value or "").strip().split())
    if not v:
        return False
    low = v.lower()
    if len(v.split()) >= 4:
        return True
    bad_phrases = [
        "my name", "name s", "i live", "live in", "looking to", "looking for", "ev charger",
        "charger installed", "outlet replaced", "outlets replaced", "smoke coming", "electrical panel",
        "need somebody", "hoping to", "was looking", "i was looking", "this is kyle and",
    ]
    return any(p in low for p in bad_phrases)


def _voice_repair_malformed_name_v31(conv: dict, caller_text: str = "") -> None:
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    fields = [
        profile.get("name"), profile.get("active_first_name"), profile.get("active_last_name"),
        profile.get("first_name"), profile.get("last_name"), profile.get("recognized_first_name"), profile.get("recognized_last_name"),
    ]
    combined = " ".join(str(x or "") for x in fields if x)
    malformed = any(_voice_name_field_is_malformed_v31(str(x or "")) for x in fields)
    intro_first, intro_last = _voice_extract_intro_name_v31((caller_text or "") + " " + combined)
    if not malformed and not intro_first:
        return
    # If malformed, throw away the bad long name and keep only a clean spoken name if available.
    clean_first = intro_first or ""
    clean_last = intro_last or ""
    # Existing good Square/customer names should not be replaced by empty strings.
    if not clean_first and not malformed:
        return
    for key in ["name", "active_first_name", "active_last_name", "first_name", "last_name", "recognized_first_name", "recognized_last_name"]:
        profile[key] = None
    if clean_first:
        profile["active_first_name"] = clean_first
        profile["first_name"] = clean_first
        profile["name"] = clean_first
    if clean_first and clean_last:
        profile["active_last_name"] = clean_last
        profile["last_name"] = clean_last
        profile["name"] = f"{clean_first} {clean_last}".strip()
    try:
        recompute_pending_step(profile, sched)
    except Exception:
        pass


def _voice_speak_email_v31(email: str) -> str:
    e = str(email or "").strip().lower()
    if not e:
        return ""
    e = e.replace("@", " at ")
    e = e.replace(".", " dot ")
    e = re.sub(r"\s+", " ", e).strip()
    return e


def _voice_save_confirmed_email_and_maybe_book_v31(phone: str, conv: dict, email: str) -> dict:
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    v13_save_email(conv, email)
    try:
        _voice_repair_malformed_name_v31(conv, "")
        recompute_pending_step(profile, sched)
    except Exception:
        pass
    first = _voice_profile_first_name(profile)
    last = _voice_profile_last_name(profile)
    if not first:
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
        reply = "Thanks. What's your first and last name?"
        return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
    if first and not last:
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
        reply = "Thanks. What's your last name?"
        return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
    if not sched.get("address_verified"):
        sched["pending_step"] = "need_address"
        sched["state"] = "waiting_for_address"
        reply = "Thanks. What's the full address for the work?"
        return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
    if not (sched.get("scheduled_date") and sched.get("scheduled_time")):
        try:
            reply = _voice_naturalize_reply(choose_next_prompt_from_state(conv, inbound_text=""))
        except Exception:
            reply = "Thanks. What day and time works best?"
        return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
    try:
        booking_attempt = maybe_create_square_booking(phone, conv)
    except Exception as e:
        try:
            log_event("VOICE_EMAIL_CONFIRM_BOOKING_ERROR_V31", phone, {"error": repr(e)}, conv)
        except Exception:
            pass
        booking_attempt = {"status": "exception"}
    status = booking_attempt.get("status") if isinstance(booking_attempt, dict) else None
    if sched.get("booking_created") and sched.get("square_booking_id") or status in {"created", "success", "booked"}:
        if isinstance(booking_attempt, dict) and booking_attempt.get("booking_id"):
            sched["square_booking_id"] = booking_attempt.get("booking_id")
            sched["booking_created"] = True
        reply = _voice_finalize_booking_reply(conv, "")
        sched["voice_close_after_reply"] = True
        sched["voice_booking_completed_close"] = True
        _voice_maybe_send_booking_sms(phone, conv, reply)
        return {"reply_to_customer": reply, "booking_created": True, "manual_only": False, "pending_step": None, "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": True}
    reply = _voice_naturalize_reply(choose_next_prompt_from_state(conv, inbound_text="")) if 'choose_next_prompt_from_state' in globals() else "Thanks. What else is needed for the appointment?"
    return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}


_ORIGINAL_PROCESS_PREVOLT_VOICE_TURN_V31 = process_prevolt_voice_turn

def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    conv = hydrate_voice_conversation((phone or "").replace("whatsapp:", "").strip(), call_sid)
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    try:
        _voice_repair_malformed_name_v31(conv, caller_text)
    except Exception as e:
        try:
            log_event("VOICE_NAME_REPAIR_PRE_ERROR_V31", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass

    # Email confirmation gate: voice ASR can mishear names inside email addresses.
    # Do not create the Square booking until the caller confirms the email we heard.
    yn = _voice_yes_no_text(caller_text)
    if sched.get("voice_awaiting_email_confirm"):
        pending_email = (sched.get("voice_pending_email") or "").strip().lower()
        if yn == "yes" and pending_email:
            sched["voice_awaiting_email_confirm"] = False
            sched["voice_pending_email"] = None
            out = _voice_save_confirmed_email_and_maybe_book_v31(phone, conv, pending_email)
            out["reply_to_customer"] = _voice_naturalize_reply(out.get("reply_to_customer") or "")
            conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": out["reply_to_customer"], "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
            conv["last_voice_reply"] = out["reply_to_customer"]
            return out
        if yn == "no" or any(p in _intent_text(caller_text) for p in ["wrong", "not correct", "incorrect", "nope"]):
            sched["voice_awaiting_email_confirm"] = False
            sched["voice_pending_email"] = None
            reply = "No problem. Please say the email again slowly, and spell any unusual part if needed."
            conv["last_voice_reply"] = reply
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_email", "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
        # If the correction itself contains an email, replace pending and confirm that instead.
        corrected = v13_extract_email(caller_text or "")
        if corrected and corrected.lower() != pending_email:
            sched["voice_pending_email"] = corrected.lower()
            spoken = _voice_speak_email_v31(corrected)
            reply = f"I heard {spoken}. Is that correct?"
            conv["last_voice_reply"] = reply
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_email", "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
        reply = f"I heard {_voice_speak_email_v31(pending_email)}. Is that correct?"
        return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_email", "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    incoming_email = v13_extract_email(caller_text or "")
    if incoming_email and not sched.get("voice_email_confirmed"):
        # Store only as pending. Confirm before booking.
        sched["voice_pending_email"] = incoming_email.lower()
        sched["voice_awaiting_email_confirm"] = True
        sched["pending_step"] = "need_email"
        sched["state"] = "waiting_for_email"
        spoken = _voice_speak_email_v31(incoming_email)
        reply = f"I heard {spoken}. Is that correct?"
        conv["last_voice_reply"] = reply
        try:
            log_event("VOICE_EMAIL_CONFIRM_PROMPT_V31", phone, {"email": incoming_email.lower(), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
        except Exception:
            pass
        return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_email", "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    output = _ORIGINAL_PROCESS_PREVOLT_VOICE_TURN_V31(phone, call_sid, caller_text)
    try:
        _voice_repair_malformed_name_v31(conv, caller_text)
        # If the original path produced a malformed name in a reply state, repair it and avoid booking with it later.
        reply = _voice_naturalize_reply(output.get("reply_to_customer") or "")
        reply = re.sub(r"\bIf that is something you(?:'|’)re interested in\.\s*Does that work for you\?", "Does that work for you?", reply, flags=re.I)
        reply = re.sub(r"\bBefore we confirm, may I have the best email address for the appointment\?\s*What's the best email address for the appointment\?", "What's the best email address for the appointment?", reply, flags=re.I)
        reply = re.sub(r"\bfor your preferred time!\s*", "", reply, flags=re.I)
        output["reply_to_customer"] = reply
        conv["last_voice_reply"] = reply
    except Exception as e:
        try:
            log_event("VOICE_POSTPROCESS_ERROR_V31", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid}, conv)
        except Exception:
            pass
    return output

# =============================
# v32 VOICE HOTFIX LAYER
# =============================
# Targets live failures observed after v31:
# - first utterance name like "My name is Sean, and I live..." saved as whole sentence
# - spelled last names like "B-A-C-O-N" saved literally instead of "Bacon"
# - email confirmation not clear enough for ASR-misheard letters
# - filler phrases leaking into voice output
# - first response sometimes stalling after generic scheduling lead-in

import string as _v32_string

_ORIG_VOICE_NATURALIZE_REPLY_V32 = _voice_naturalize_reply
_ORIG_PROCESS_PREVOLT_VOICE_TURN_V32 = process_prevolt_voice_turn

_V32_FILLER_PATTERNS = [
    r"\bOkay,?\s+let me (?:check|grab|get|confirm|run|move|finish|finalize|set up|process)[^.?!]*[.?!]\s*",
    r"\bGot it,?\s+thanks(?: for that address)?\.\s*",
    r"\bThanks\.\s+I've got your name\.\s*",
    r"\bNext,?\s+I just need one more detail to finish booking\.\s*",
    r"\bLet me (?:check|think|respond|route|process|finish|finalize|move ahead|get)[^.?!]*[.?!]\s*",
    r"\bOne moment while I[^.?!]*[.?!]\s*",
    r"\bThe scheduling system is still processing[^.?!]*[.?!]\s*",
    r"\bI'm going to (?:focus|route|check|process|move)[^.?!]*[.?!]\s*",
    r"\bI'll (?:just )?(?:clarify|check|use|move|finish|wrap)[^.?!]*[.?!]\s*",
    r"\bAll right,?\s+we can move forward with that\.\s*",
    r"\bBefore choosing a time,\s*",
    r"\bBefore we confirm,\s*",
]

_V32_STOP_AFTER_FIRST_NAME = re.compile(
    r"\b(?:my\s+name\s*(?:is|'s|s)|this\s+is|it'?s|i\s+am|i'm)\s+([A-Z][a-zA-Z'\-]{1,24})\b",
    re.I,
)

_V32_BAD_NAME_WORDS = {
    "my", "name", "s", "is", "and", "i", "im", "i'm", "live", "in", "windsor", "looking",
    "need", "want", "have", "ev", "charger", "installed", "outlet", "repair", "replace",
    "work", "calling", "for", "the", "a", "an", "to", "get", "somebody", "come", "out",
}

_V32_LASTNAME_STOPWORDS = {
    "and", "i", "im", "i'm", "live", "in", "looking", "need", "want", "have", "for", "to",
    "at", "from", "calling", "ev", "outlet", "charger", "installed", "repair", "replace",
}

_V32_LETTERS = {c: c for c in "abcdefghijklmnopqrstuvwxyz"}
_V32_LETTER_WORDS = {
    "a": "a", "ay": "a", "b": "b", "be": "b", "bee": "b", "c": "c", "see": "c", "sea": "c",
    "d": "d", "dee": "d", "e": "e", "f": "f", "eff": "f", "g": "g", "gee": "g",
    "h": "h", "aitch": "h", "i": "i", "eye": "i", "j": "j", "jay": "j", "k": "k", "kay": "k",
    "l": "l", "el": "l", "m": "m", "em": "m", "n": "n", "en": "n", "o": "o", "oh": "o",
    "p": "p", "pee": "p", "q": "q", "cue": "q", "queue": "q", "r": "r", "are": "r",
    "s": "s", "ess": "s", "t": "t", "tee": "t", "u": "u", "you": "u", "v": "v", "vee": "v",
    "w": "w", "doubleyou": "w", "double-u": "w", "x": "x", "ex": "x", "y": "y", "why": "y", "z": "z", "zee": "z", "zed": "z",
}


def _v32_clean_person_name_piece(s: str) -> str:
    s = re.sub(r"[^A-Za-z'\-\s]", " ", str(s or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:40]


def _v32_extract_first_name(text: str) -> str:
    m = _V32_STOP_AFTER_FIRST_NAME.search(str(text or ""))
    if not m:
        return ""
    name = _v32_clean_person_name_piece(m.group(1))
    if not name or name.lower() in _V32_BAD_NAME_WORDS:
        return ""
    return name[:1].upper() + name[1:].lower()


def _v32_spelled_name_to_word(text: str) -> str:
    raw = str(text or "").strip()
    # Examples: B-A-C-O-N, B A C O N, B. A. C. O. N.
    tokens = re.findall(r"[A-Za-z]+", raw.lower())
    if not tokens:
        return ""
    # Strip leading explanation words: "Bacon B A C O N" should prefer the first normal word.
    if len(tokens) == 1 and len(tokens[0]) > 1:
        word = tokens[0]
        if word not in _V32_LASTNAME_STOPWORDS:
            return word[:1].upper() + word[1:].lower()
    letters = []
    for tok in tokens:
        if len(tok) == 1:
            letters.append(tok)
        elif tok in _V32_LETTER_WORDS:
            letters.append(_V32_LETTER_WORDS[tok])
        else:
            # A normal word like "bacon" wins over trying to spell the whole phrase.
            if tok not in _V32_LASTNAME_STOPWORDS and len(tok) >= 2:
                return tok[:1].upper() + tok[1:].lower()
            return ""
    if 2 <= len(letters) <= 18:
        word = "".join(letters)
        return word[:1].upper() + word[1:].lower()
    return ""


def _v32_name_is_malformed(name: str) -> bool:
    n = re.sub(r"\s+", " ", str(name or "")).strip().lower()
    if not n:
        return False
    if len(n.split()) >= 5:
        return True
    return any(p in n for p in [" i live ", " looking to ", " ev charger", " outlet ", " installed", " my name "])


def _v32_apply_name_repairs(conv: dict, caller_text: str = "") -> None:
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    full = profile.get("name") or ""
    first = profile.get("active_first_name") or profile.get("first_name") or ""
    last = profile.get("active_last_name") or profile.get("last_name") or ""

    extracted_first = _v32_extract_first_name(caller_text)
    if extracted_first:
        # If the first utterance says "My name is Sean...", never allow the rest of the sentence as last name.
        profile["active_first_name"] = extracted_first
        profile["first_name"] = extracted_first
        if _v32_name_is_malformed(full) or _v32_name_is_malformed(last) or not last:
            profile["active_last_name"] = None
            profile["last_name"] = None
            profile["name"] = extracted_first

    # Waiting for last name: accept one-word or spelled last name, but not filler.
    step = str(sched.get("pending_step") or "")
    state = str(sched.get("state") or "")
    has_first = bool(profile.get("active_first_name") or profile.get("first_name"))
    has_last = bool(profile.get("active_last_name") or profile.get("last_name"))
    if has_first and not has_last and (step == "need_name" or "name" in state.lower()):
        possible_last = _v32_spelled_name_to_word(caller_text)
        if possible_last and possible_last.lower() not in _V32_LASTNAME_STOPWORDS:
            first_name = profile.get("active_first_name") or profile.get("first_name")
            profile["active_last_name"] = possible_last
            profile["last_name"] = possible_last
            profile["name"] = f"{first_name} {possible_last}".strip()
            try:
                recompute_pending_step(profile, sched)
            except Exception:
                pass

    # If previous code already wrote a malformed name, salvage it.
    full = profile.get("name") or ""
    if _v32_name_is_malformed(full):
        salvage = _v32_extract_first_name(full) or _v32_extract_first_name(caller_text)
        for key in ["name", "active_first_name", "active_last_name", "first_name", "last_name"]:
            profile[key] = None
        if salvage:
            profile["active_first_name"] = salvage
            profile["first_name"] = salvage
            profile["name"] = salvage
        try:
            recompute_pending_step(profile, sched)
        except Exception:
            pass


def _v32_spoken_email(email: str) -> str:
    """Read email slowly enough that b/v/i/y mistakes are catchable."""
    e = str(email or "").strip().lower()
    if not e:
        return ""
    # Only spell the local part character-by-character; domain is usually obvious.
    if "@" in e:
        local, domain = e.split("@", 1)
    else:
        local, domain = e, ""
    pieces = []
    for ch in local:
        if ch == ".":
            pieces.append("dot")
        elif ch == "_":
            pieces.append("underscore")
        elif ch == "-":
            pieces.append("dash")
        elif ch.isalnum():
            # Make the common trouble letters more explicit.
            names = {"b": "B as in boy", "v": "V as in Victor", "i": "I", "y": "Y", "o": "O", "m": "M", "n": "N"}
            pieces.append(names.get(ch, ch.upper() if ch.isalpha() else ch))
    local_spoken = ", ".join(pieces)
    if domain:
        domain_spoken = domain.replace(".", " dot ")
        return f"{local_spoken}, at {domain_spoken}"
    return local_spoken


def _voice_naturalize_reply(reply: str) -> str:
    r = _ORIG_VOICE_NATURALIZE_REPLY_V32(reply or "")
    # Remove filler fragments repeatedly because the model/backend often chains them.
    changed = True
    while changed:
        old = r
        for pat in _V32_FILLER_PATTERNS:
            r = re.sub(pat, "", r, flags=re.I)
        changed = (r != old)
    # Avoid split prompt duplication.
    r = re.sub(r"\bI can help get this scheduled\.\s*", "I can get your appointment scheduled here. ", r, flags=re.I)
    r = re.sub(r"\bI can get your appointment scheduled here\.\s*Okay\.\s*", "I can get your appointment scheduled here. ", r, flags=re.I)
    r = re.sub(r"\bI can get your appointment scheduled here\.\s*(What(?:'s| is) the house number)", r"I can get your appointment scheduled here. \1", r, flags=re.I)
    r = re.sub(r"\bThanks\.\s+What's", "What's", r, flags=re.I)
    r = re.sub(r"\bOkay\.\s+What's", "What's", r, flags=re.I)
    r = re.sub(r"\bIf that is something you(?:'|’)re interested in\.\s*Does that work for you\?", "Does that work for you?", r, flags=re.I)
    r = re.sub(r"\bBefore we confirm, may I have the best email address for the appointment\?\s*What's the best email address for the appointment\?", "What's the best email address for the appointment?", r, flags=re.I)
    r = re.sub(r"\s+", " ", r).strip()
    # Tiny punctuation cleanup for voice pacing.
    r = r.replace(" ,", ",").replace(" .", ".")
    return r


# Override email confirmation wording to be spelling-forward.
def _voice_speak_email_v31(email: str) -> str:
    return _v32_spoken_email(email)


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    conv = hydrate_voice_conversation((phone or "").replace("whatsapp:", "").strip(), call_sid)
    try:
        _v32_apply_name_repairs(conv, caller_text)
    except Exception as e:
        try:
            log_event("VOICE_V32_NAME_PRE_REPAIR_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V32(phone, call_sid, caller_text)

    try:
        _v32_apply_name_repairs(conv, caller_text)
        sched = conv.setdefault("sched", {})
        profile = conv.setdefault("profile", {})
        # If booking was just created with a malformed name, update state before any future/customer record logic sees it.
        if _v32_name_is_malformed(profile.get("name") or ""):
            _v32_apply_name_repairs(conv, profile.get("name") or caller_text)
        reply = str(out.get("reply_to_customer") or "")
        reply = _voice_naturalize_reply(reply)
        # If we only have a first name, do not let the flow skip last-name collection before email/booking.
        first = profile.get("active_first_name") or profile.get("first_name")
        last = profile.get("active_last_name") or profile.get("last_name")
        if first and not last and (sched.get("pending_step") == "need_email" or "email" in reply.lower()):
            sched["pending_step"] = "need_name"
            sched["state"] = "waiting_for_name"
            reply = "What's your last name?"
            out["booking_created"] = False
            out["end_call"] = False
        out["reply_to_customer"] = reply
        conv["last_voice_reply"] = reply
    except Exception as e:
        try:
            log_event("VOICE_V32_POSTPROCESS_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text)}, conv)
        except Exception:
            pass
    return out


# v32b: make the processing cue last long enough to cover the common dead-air gap.
# This intentionally reuses the smooth cue generator above, but forces a longer default.
_ORIG_VOICE_THINKING_CLICK_PAYLOAD_V32 = _voice_thinking_click_payload

def _voice_thinking_click_payload(duration_ms: int | None = None) -> str:
    try:
        requested = int(duration_ms or VOICE_AGENT_THINKING_SOUND_MS or 1600)
    except Exception:
        requested = 1600
    # The original generator caps at 1800ms. Force at least 1400ms unless an explicit longer env is set.
    requested = max(1400, min(1800, requested))
    return _ORIG_VOICE_THINKING_CLICK_PAYLOAD_V32(requested)


# =============================
# v33 VOICE HOTFIX LAYER
# =============================
# Targets live failures observed after v32:
# - first meaningful response could stop at the generic sentence
#   "I can get your appointment scheduled here." with no actual next question.
# - backend/OpenAI filler could still leak into phone output.
# - processing cue only played once and then left dead air on slower turns.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V33 = process_prevolt_voice_turn
_ORIG_HANDLE_REALTIME_FUNCTION_CALL_V33 = _handle_realtime_function_call
_ORIG_VOICE_NATURALIZE_REPLY_V33 = _voice_naturalize_reply

_V33_GENERIC_SCHEDULING_ONLY = re.compile(
    r"^\s*(?:okay\.?\s*)?(?:i can help get this scheduled\.?|i can get your appointment scheduled here\.?)\s*$",
    re.I,
)

_V33_EXTRA_FILLER_PATTERNS = [
    r"\bOkay,?\s+let me get this set up with scheduling\.\s*",
    r"\bOkay,?\s+let me get that address detail processed so we can keep going\.\s*",
    r"\bNo problem\.\s+Let me get that updated so we can keep moving\.\s*",
    r"\bOkay\.\s+Thanks\.\s+Let me run that through so we can keep moving\.\s*",
    r"\bOkay,?\s+let me grab the next scheduling detail\.\s*",
    r"\bOkay,?\s+let me check that timing for you\.\s*",
    r"\bThanks,?\s+let me confirm that with your booking details\.\s*",
    r"\bThanks,?\s+I'll use that email for the appointment details\.\s*",
    r"\bLet me get that address detail processed[^.?!]*[.?!]\s*",
    r"\bLet me run that through[^.?!]*[.?!]\s*",
    r"\bLet me get this set up[^.?!]*[.?!]\s*",
]


def _v33_missing_field_prompt(conv: dict) -> str:
    sched = (conv or {}).setdefault("sched", {})
    pending = str(sched.get("pending_step") or "").lower()
    state = str(sched.get("state") or "").lower()
    missing = str(sched.get("address_missing") or "").lower()
    raw_addr = str(sched.get("raw_address") or sched.get("address_candidate") or sched.get("normalized_address") or "").strip()

    if pending == "need_address" or state == "waiting_for_address" or missing:
        if missing in {"town", "city", "state"}:
            return "Which town is it in, and is that Connecticut or Massachusetts?"
        if missing == "confirm" and raw_addr:
            return "Which town is it in, and is that Connecticut or Massachusetts?"
        return "What's the house number and street name for the work?"

    if pending == "need_name" or state == "waiting_for_name":
        profile = (conv or {}).setdefault("profile", {})
        first = profile.get("active_first_name") or profile.get("first_name")
        last = profile.get("active_last_name") or profile.get("last_name")
        if first and not last:
            return "What's your last name?"
        return "What's your first and last name?"

    if pending == "need_email" or state == "waiting_for_email":
        return "What's the best email address for the appointment?"

    if pending in {"need_date", "need_time"} or state == "waiting_for_date":
        return "What day and time works best for you?"

    return "What's the house number and street name for the work?"


def _v33_has_customer_facing_question(text: str) -> bool:
    t = str(text or "").strip()
    if "?" in t:
        return True
    low = _intent_text(t) if "_intent_text" in globals() else t.lower()
    return any(p in low for p in [
        "what is", "what's", "which one", "which town", "what day", "what time",
        "does that work", "do you want", "is this for", "can you", "please provide",
    ])


def _v33_fix_generic_reply(conv: dict, reply: str) -> str:
    r = str(reply or "").strip()
    sched = (conv or {}).setdefault("sched", {})
    if _V33_GENERIC_SCHEDULING_ONLY.match(r):
        return "I can get your appointment scheduled here. " + _v33_missing_field_prompt(conv)
    # If the reply has the scheduling opener but no actual question, force the missing field.
    if re.search(r"\bI can get your appointment scheduled here\.?\s*$", r, flags=re.I) and not _v33_has_customer_facing_question(r):
        return re.sub(r"\.?\s*$", ". ", r).strip() + " " + _v33_missing_field_prompt(conv)
    # If backend already left us in a pending step but the spoken reply is not actionable, append the next question.
    if not sched.get("booking_created") and not sched.get("manual_only") and not _v33_has_customer_facing_question(r):
        pending = str(sched.get("pending_step") or "").lower()
        state = str(sched.get("state") or "").lower()
        if pending.startswith("need_") or state.startswith("waiting_for_"):
            prompt = _v33_missing_field_prompt(conv)
            if prompt and prompt.lower() not in r.lower():
                if not r:
                    return prompt
                return r.rstrip(" .") + ". " + prompt
    return r


def _voice_naturalize_reply(reply: str) -> str:
    r = _ORIG_VOICE_NATURALIZE_REPLY_V33(reply or "")
    changed = True
    while changed:
        old = r
        for pat in _V33_EXTRA_FILLER_PATTERNS:
            r = re.sub(pat, "", r, flags=re.I)
        changed = (r != old)
    r = re.sub(r"\bI can help get this scheduled\.\s*I can get your appointment scheduled here\.", "I can get your appointment scheduled here.", r, flags=re.I)
    r = re.sub(r"\bI can help get this scheduled\.\s*", "I can get your appointment scheduled here. ", r, flags=re.I)
    r = re.sub(r"\s+", " ", r).strip()
    return r


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    conv = hydrate_voice_conversation((phone or "").replace("whatsapp:", "").strip(), call_sid)
    output = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V33(phone, call_sid, caller_text)
    try:
        # Re-run name repairs after the base logic, then force an actionable phone reply.
        try:
            if "_v32_apply_name_repairs" in globals():
                _v32_apply_name_repairs(conv, caller_text)
        except Exception:
            pass
        sched = conv.setdefault("sched", {})
        reply = _voice_naturalize_reply(str(output.get("reply_to_customer") or ""))
        reply = _v33_fix_generic_reply(conv, reply)
        output["reply_to_customer"] = reply
        conv["last_voice_reply"] = reply
        # If this patch has supplied a real next question, the call must remain open.
        if not sched.get("booking_created") and not output.get("booking_created"):
            output["end_call"] = False
    except Exception as e:
        try:
            log_event("VOICE_V33_POSTPROCESS_ERROR", phone, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid}, conv)
        except Exception:
            pass
    return output


def _v33_start_thinking_loop(twilio_ws, stream_sid: str, phone: str, call_sid: str, stop_evt) -> None:
    """Keep a gentle processing cue going during longer backend/tool turns.

    The cue is intentionally short and repeated only while the function/tool turn
    is active. It is stopped before the assistant response is requested, so it
    should not overlap the spoken answer.
    """
    if not (VOICE_AGENT_THINKING_SOUND_ENABLED and twilio_ws is not None and stream_sid):
        return
    next_send = 0.0
    while not stop_evt.is_set():
        now = time.time()
        if now >= next_send:
            try:
                _send_twilio_media(twilio_ws, stream_sid, _voice_thinking_click_payload(1400))
                try:
                    log_event("VOICE_THINKING_SOUND_SENT", phone, {"reason": "processing_loop", "call_sid": call_sid, "stream_sid": stream_sid})
                except Exception:
                    pass
            except Exception:
                return
            next_send = now + 1.15
        stop_evt.wait(0.08)


def _handle_realtime_function_call(openai_ws, phone: str, call_sid: str, item: dict, handled_call_ids: set | None = None, twilio_ws=None, stream_sid: str = "") -> bool:
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return False
    if (item.get("name") or "") != "prevolt_os_turn":
        return False
    call_id = item.get("call_id") or item.get("id") or ""
    if handled_call_ids is not None and call_id:
        if call_id in handled_call_ids:
            return True
        handled_call_ids.add(call_id)
    try:
        args = json.loads(item.get("arguments") or "{}")
    except Exception:
        args = {}

    stop_evt = threading.Event()
    cue_thread = None
    if twilio_ws is not None and stream_sid and VOICE_AGENT_THINKING_SOUND_ENABLED:
        cue_thread = threading.Thread(target=_v33_start_thinking_loop, args=(twilio_ws, stream_sid, phone, call_sid, stop_evt), daemon=True)
        cue_thread.start()

    try:
        output = process_prevolt_voice_turn(phone, call_sid, args.get("caller_text") or "")
    finally:
        # Let the last cue tail keep playing briefly so the caller hears continuity,
        # then stop before we ask OpenAI/Twilio to speak the actual answer.
        time.sleep(0.12)
        stop_evt.set()

    exact_reply = _voice_to_sms_text(str(output.get("reply_to_customer") or "Sorry, can you say that again?"))
    booking_final = bool(output.get("end_call") and output.get("booking_created"))

    try:
        _send_openai_event(openai_ws, {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id, "output": json.dumps(output)},
        })
    except Exception:
        pass

    if booking_final and "_voice_call_update_with_final_twiml" in globals() and _voice_call_update_with_final_twiml(call_sid, exact_reply, phone):
        try:
            conv = hydrate_voice_conversation(phone, call_sid)
            conv.setdefault("sched", {})["voice_close_after_reply"] = True
            conv.setdefault("sched", {})["voice_final_twiML_sent"] = True
        except Exception:
            pass
        return True

    if output.get("end_call"):
        try:
            sched_local = hydrate_voice_conversation(phone, call_sid).setdefault("sched", {})
            sched_local["voice_close_after_reply"] = True
            sched_local["voice_waiting_for_final_audio_done"] = True
        except Exception:
            pass

    _send_openai_event(openai_ws, {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio"],
            "tool_choice": "none",
            "instructions": (
                "Say ONLY this exact phone reply, word for word, including every question in it, then stop speaking. "
                "Do not say anything before it. Do not say you are checking, thinking, waiting, routing, processing, or finishing. "
                "Speak at a natural medium pace. Do not slow down or add dramatic pauses at commas. "
                "Do not add any acknowledgement, greeting, filler phrase, or extra question. "
                "Do not stop after the first clause or first sentence; speak the complete exact reply. "
                "Exact reply: " + json.dumps(exact_reply)
            ),
        },
    })
    return True


# =============================
# v34 VOICE HOTFIX LAYER
# =============================
# Targets live failures observed after v33:
# - generic first reply stalling after "I can get your appointment scheduled here"
# - weekday/month words becoming last names (Matthew Monday)
# - actual last-name reply not overwriting bad/date last name
# - duplicate email confirmation after caller already said yes
# - offered-slot choice falling back to the first offered slot

_V34_DATE_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "today", "tomorrow", "morning", "afternoon", "evening", "night", "noon", "pm", "am",
}

_ORIG_VOICE_SANITIZE_NAME_FIELDS_V34 = _voice_sanitize_name_fields
_ORIG_VOICE_NAME_FAST_PATH_V34 = _voice_name_fast_path
_ORIG_VOICE_APPLY_OFFERED_SLOT_FAST_PATH_V34 = _voice_apply_offered_slot_fast_path
_ORIG_PROCESS_PREVOLT_VOICE_TURN_V34 = process_prevolt_voice_turn


def _v34_clean_single_name_word(text: str) -> str:
    """Return a clean one-word/spelled surname when the caller is answering a name prompt."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    if "@" in raw or re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", raw):
        return ""
    # Prefer spelled-last-name conversion from v32 when available: B-A-C-O-N -> Bacon.
    try:
        spelled = _v32_spelled_name_to_word(raw)
        if spelled:
            raw_word = spelled
        else:
            raw_word = raw
    except Exception:
        raw_word = raw
    raw_word = re.sub(r"\b(?:my\s+last\s+name\s+is|last\s+name\s+is|it\s+is|it's|its|this\s+is)\b", " ", raw_word, flags=re.I)
    words = [normalize_person_name(w) for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", raw_word) if normalize_person_name(w)]
    if not words:
        return ""
    # If caller says "Bacon B-A-C-O-N", use the first normal word.
    candidate = words[0]
    bad = set(_V34_DATE_WORDS) | set(globals().get("_V32_LASTNAME_STOPWORDS", set())) | {
        "yes", "yeah", "okay", "ok", "sure", "correct", "works", "thanks", "thank", "you",
        "ev", "charger", "outlet", "installed", "repair", "replace", "windsor", "locks", "connecticut", "massachusetts",
    }
    if candidate.lower() in bad:
        return ""
    if len(candidate) < 2 or len(candidate) > 32:
        return ""
    return candidate[:1].upper() + candidate[1:].lower()


def _v34_bad_last_name(value: str) -> bool:
    v = normalize_person_name(str(value or "")).strip().lower()
    if not v:
        return False
    if v in _V34_DATE_WORDS:
        return True
    try:
        if _v32_name_is_malformed(v):
            return True
    except Exception:
        pass
    return False


def _voice_sanitize_name_fields(profile: dict, sched: dict | None = None) -> None:
    try:
        _ORIG_VOICE_SANITIZE_NAME_FIELDS_V34(profile, sched)
    except Exception:
        pass
    changed = False
    for key in ["active_last_name", "last_name", "recognized_last_name", "voicemail_last_name"]:
        if _v34_bad_last_name(profile.get(key) or ""):
            profile[key] = None
            changed = True
    full = " ".join(str(profile.get("name") or "").strip().split())
    if full:
        parts = [p for p in re.findall(r"[A-Za-z][A-Za-z'\-]*", full)]
        if len(parts) >= 2 and _v34_bad_last_name(parts[-1]):
            profile["name"] = parts[0]
            profile["active_first_name"] = profile.get("active_first_name") or parts[0]
            profile["first_name"] = profile.get("first_name") or parts[0]
            changed = True
        elif len(parts) >= 5:
            first = profile.get("active_first_name") or profile.get("first_name") or (parts[0] if parts else "")
            if first:
                profile["name"] = first
                profile["active_first_name"] = first
                profile["first_name"] = first
                profile["active_last_name"] = None
                profile["last_name"] = None
                changed = True
    if changed and sched is not None:
        try:
            recompute_pending_step(profile, sched)
        except Exception:
            pass


def _voice_name_fast_path(conv: dict, caller_text: str) -> str | None:
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    step = str(sched.get("pending_step") or "").lower()
    state = str(sched.get("state") or "").lower()
    name_engine_state = str(sched.get("name_engine_state") or "").lower()
    waiting_for_name = step == "need_name" or state in {"waiting_for_name", "need_name"} or "name" in name_engine_state
    if waiting_for_name:
        try:
            _voice_sanitize_name_fields(profile, sched)
        except Exception:
            pass
        first_known = _voice_profile_first_name(profile)
        candidate = _v34_clean_single_name_word(caller_text)
        if first_known and candidate:
            # Overwrite any date/weekday or stale last name while the system is explicitly asking for last name.
            profile["active_last_name"] = candidate
            profile["last_name"] = candidate
            profile["name"] = f"{first_known} {candidate}".strip()
            sched["name_engine_state"] = None
            sched["pending_step"] = None
            try:
                recompute_pending_step(profile, sched)
            except Exception:
                pass
            return _voice_after_name_reply(conv)
        if first_known and not _voice_profile_last_name(profile):
            sched["pending_step"] = "need_name"
            sched["state"] = "waiting_for_name"
            return "What's your last name?"
    return _ORIG_VOICE_NAME_FAST_PATH_V34(conv, caller_text)


def _v34_text_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _intent_text(s or "")))


def _v34_apply_offered_slot_selection(conv: dict, inbound_text: str) -> bool:
    """Conservative offered-slot lock that strongly matches the spoken option before the generic SMS path."""
    sched = conv.setdefault("sched", {})
    options = sched.get("offered_slot_options") or []
    if not (sched.get("awaiting_slot_offer_choice") and options):
        return False
    low = _intent_text(inbound_text or "")
    explicit_time = extract_explicit_time_from_text(inbound_text or "") if "extract_explicit_time_from_text" in globals() else ""
    chosen = None

    # Ordinal choice.
    for idx, keys in enumerate([["first", "1st", "one"], ["second", "2nd", "two"], ["third", "3rd", "three"]]):
        if idx < len(options) and any(re.search(rf"\b{re.escape(k)}\b", low) for k in keys):
            chosen = options[idx]
            break

    # Match by weekday/month/day/time from the slot label/date.
    if chosen is None:
        matches = []
        for opt in options:
            try:
                d = datetime.strptime(str(opt.get("date") or ""), "%Y-%m-%d")
                weekday = d.strftime("%A").lower()
                month = d.strftime("%B").lower()
                day = str(int(d.strftime("%d")))
                opt_time = str(opt.get("time") or "")
                label = str(opt.get("label") or "")
                label_tokens = _v34_text_tokens(label)
                score = 0
                if re.search(rf"\b{weekday}\b", low): score += 2
                if re.search(rf"\b{month}\b", low): score += 2
                if re.search(rf"\b{day}(?:st|nd|rd|th)?\b", low): score += 2
                if explicit_time and opt_time == explicit_time: score += 3
                if label_tokens and label_tokens.issubset(_v34_text_tokens(low)): score += 4
                if score >= 2:
                    matches.append((score, opt))
            except Exception:
                continue
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            if len(matches) == 1 or matches[0][0] > matches[1][0]:
                chosen = matches[0][1]

    if chosen is None:
        return False

    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195").upper()
    sched["appointment_type"] = appt
    conv["appointment_type"] = appt
    sched["scheduled_date"] = chosen.get("date")
    sched["scheduled_time"] = chosen.get("time")
    sched["scheduled_time_source"] = "voice_offered_slot_v34"
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    sched["slot_choice_locked"] = True
    sched["booking_attempt_nonce"] = str(uuid.uuid4())
    return True


def _voice_apply_offered_slot_fast_path(conv: dict, caller_text: str) -> str | None:
    try:
        selected = _v34_apply_offered_slot_selection(conv, caller_text)
    except Exception:
        selected = False
    if selected:
        profile = conv.setdefault("profile", {})
        sched = conv.setdefault("sched", {})
        try:
            _voice_sanitize_name_fields(profile, sched)
            recompute_pending_step(profile, sched)
        except Exception:
            pass
        return _voice_naturalize_reply(choose_next_prompt_from_state(conv, inbound_text=caller_text))
    return _ORIG_VOICE_APPLY_OFFERED_SLOT_FAST_PATH_V34(conv, caller_text)


def _v34_force_actionable_reply(conv: dict, reply: str) -> str:
    r = _voice_naturalize_reply(reply or "")
    sched = conv.setdefault("sched", {})
    if sched.get("booking_created"):
        return r
    low = _intent_text(r)
    has_question = "?" in r or any(p in low for p in ["what's", "what is", "which town", "which one", "does that work", "do you want", "is this for"])
    generic_only = bool(re.fullmatch(r"(?:I can get your appointment scheduled here\.?|I can help get this scheduled\.?)", r.strip(), flags=re.I))
    if has_question and not generic_only:
        return r
    pending = str(sched.get("pending_step") or "").lower()
    state = str(sched.get("state") or "").lower()
    missing = str(sched.get("address_missing") or "").lower()
    prompt = ""
    if pending == "need_address" or state == "waiting_for_address" or missing:
        prompt = "Which town is it in, and is that Connecticut or Massachusetts?" if missing in {"town", "city", "state", "confirm"} else "What's the house number and street name for the work?"
    elif pending == "need_name" or state == "waiting_for_name":
        first = _voice_profile_first_name(conv.setdefault("profile", {}))
        last = _voice_profile_last_name(conv.setdefault("profile", {}))
        prompt = "What's your last name?" if first and not last else "What's your first and last name?"
    elif pending == "need_email" or state == "waiting_for_email":
        prompt = "What's the best email address for the appointment?"
    elif pending in {"need_date", "need_time"} or state == "waiting_for_date":
        prompt = "Which appointment option works best, or what day and time are you available?"
    if prompt:
        base = "I can get your appointment scheduled here." if generic_only or not r else r.rstrip(" .") + "."
        if prompt.lower() not in base.lower():
            return f"{base} {prompt}".strip()
    return r


def _v34_dedupe_email_confirmation(conv: dict, reply: str, caller_text: str) -> str:
    sched = conv.setdefault("sched", {})
    yn = _voice_yes_no_text(caller_text) if "_voice_yes_no_text" in globals() else ""
    if yn == "yes" and sched.get("voice_email_confirmed") and "is that correct" in _intent_text(reply or ""):
        # If the caller just confirmed the email, don't ask the same confirmation again.
        return "Thanks. I'll finish the booking now."
    return reply


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    try:
        conv0 = hydrate_voice_conversation(p, call_sid)
        _v32_apply_name_repairs(conv0, caller_text)
        _voice_sanitize_name_fields(conv0.setdefault("profile", {}), conv0.setdefault("sched", {}))
    except Exception:
        pass
    output = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V34(phone, call_sid, caller_text)
    try:
        # IMPORTANT: rehydrate after the wrapped function; it may have updated state in storage.
        conv = hydrate_voice_conversation(p, call_sid)
        sched = conv.setdefault("sched", {})
        profile = conv.setdefault("profile", {})
        try:
            _v32_apply_name_repairs(conv, caller_text)
            _voice_sanitize_name_fields(profile, sched)
        except Exception:
            pass
        # Do not let a day/month survive as a last name after a date-selection turn.
        if _v34_bad_last_name(profile.get("active_last_name") or profile.get("last_name") or ""):
            profile["active_last_name"] = None
            profile["last_name"] = None
            first = _voice_profile_first_name(profile)
            profile["name"] = first or None
            try:
                recompute_pending_step(profile, sched)
            except Exception:
                pass
        reply = _v34_force_actionable_reply(conv, str(output.get("reply_to_customer") or ""))
        reply = _v34_dedupe_email_confirmation(conv, reply, caller_text)
        output["reply_to_customer"] = reply
        conv["last_voice_reply"] = reply
        if not sched.get("booking_created") and not output.get("booking_created") and ("?" in reply):
            output["end_call"] = False
        try:
            log_event("VOICE_V34_POSTPROCESS", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "pending_step": sched.get("pending_step"), "state": sched.get("state"), "name": profile.get("name"), "call_sid": call_sid}, conv)
        except Exception:
            pass
    except Exception as e:
        try:
            log_event("VOICE_V34_POSTPROCESS_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return output

# =============================
# v35 VOICE HOTFIX LAYER
# =============================
# Targets live failures observed after v34:
# - email correction by voice can spiral into an impossible confirmation loop.
#   If the first email confirmation is rejected, move the remaining email capture
#   to SMS where the customer can type it exactly.
# - inbound SMS email after a voice handoff must not be interpreted as a last name
#   such as "Matthew Amy".
# - filler still leaked into phone replies.
# - processing cue should be calmer and more continuous, not irritating.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V35 = process_prevolt_voice_turn
_ORIG_INCOMING_SMS_V35 = incoming_sms
_ORIG_VOICE_NATURALIZE_REPLY_V35 = _voice_naturalize_reply

_V35_FILLER_PATTERNS = [
    r"\bOkay,?\s+thanks,?\s*[^.?!]*?(?:next detail|one more detail|keep this moving|finish scheduling|finish this|wrap this up|sort out|best next step)[^.?!]*[.?!]\s*",
    r"\bOkay,?\s+let me\s+[^.?!]*?(?:check|grab|sort|process|run|confirm|update|finalize|wrap|move|finish)[^.?!]*[.?!]\s*",
    r"\bThanks,?\s+I(?:'|’)m\s+[^.?!]*?(?:grabbing|using|updating|confirming|checking)[^.?!]*[.?!]\s*",
    r"\bGot it,?\s+I(?:'|’)m\s+[^.?!]*?(?:using|checking|updating|confirming|finishing|wrapping)[^.?!]*[.?!]\s*",
    r"\bPerfect,?\s+I(?:'|’)ll\s+use\s+that\s+time\s+and\s+just\s+need\s+one\s+more\s+detail\.\s*",
    r"\bLet me\s+[^.?!]*?(?:think|check|grab|sort|process|run|confirm|update|finalize|wrap|move|finish)[^.?!]*[.?!]\s*",
    r"\bI(?:'|’)ll\s+(?:move ahead|use that email|confirm the last name spelling|update that spelling|wrap up|finish)[^.?!]*[.?!]\s*",
]


def _voice_naturalize_reply(reply: str) -> str:
    r = _ORIG_VOICE_NATURALIZE_REPLY_V35(reply or "")
    changed = True
    while changed:
        old = r
        for pat in _V35_FILLER_PATTERNS:
            r = re.sub(pat, "", r, flags=re.I)
        changed = (r != old)
    r = re.sub(r"\bIf that is something you(?:'|’)?re interested in\.\s*Does that work for you\?", "Does that work for you?", r, flags=re.I)
    r = re.sub(r"\bBefore choosing a time,\s*", "", r, flags=re.I)
    r = re.sub(r"\bBefore I finalize your appointment[^.?!]*,\s*", "", r, flags=re.I)
    r = re.sub(r"\s+", " ", r).strip()
    return r


def _v35_send_email_capture_sms(phone: str, conv: dict, reason: str = "email_voice_correction_failed") -> None:
    if not phone or not (twilio_client and TWILIO_FROM_NUMBER):
        return
    sched = conv.setdefault("sched", {})
    if sched.get("voice_email_text_handoff_sent"):
        return
    body = "This is Prevolt Electric. We got the appointment details started over the phone. Please reply with the correct email address so we can finish booking."
    try:
        msg = twilio_client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=phone)
        sched["voice_email_text_handoff_sent"] = True
        sched["voice_email_text_handoff_reason"] = reason
        sched["pending_step"] = "need_email"
        sched["state"] = "waiting_for_email"
        sched["voice_awaiting_email_confirm"] = False
        sched["voice_pending_email"] = None
        conv["last_sms_body"] = body
        log_event("VOICE_EMAIL_TEXT_HANDOFF_SMS_SENT", phone, {"sid": getattr(msg, "sid", None), "reason": reason, "call_sid": conv.get("last_call_sid")}, conv)
    except Exception as e:
        sched["voice_email_text_handoff_error"] = repr(e)
        try:
            log_event("VOICE_EMAIL_TEXT_HANDOFF_SMS_FAILED", phone, {"error": repr(e), "reason": reason}, conv)
        except Exception:
            pass


def _v35_clean_last_from_spelled_text(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    # Examples: "Bacon, B-A-C-O-N" -> Bacon; "B-A-C-O-N" -> Bacon.
    m = re.search(r"\b([A-Za-z][A-Za-z'\-]{1,30})\b\s*,?\s*(?:[A-Za-z]\s*[- ]\s*){2,}[A-Za-z]\b", raw)
    if m:
        return normalize_person_name(m.group(1)) or None
    letters = re.findall(r"\b[A-Za-z]\b", raw)
    if len(letters) >= 3 and re.fullmatch(r"[A-Za-z](?:\s*[- ]\s*[A-Za-z]){2,}", raw):
        return normalize_person_name("".join(letters)) or None
    return None


def _v35_email_handoff_voice_reply(phone: str, conv: dict, reason: str = "email_rejected") -> dict:
    sched = conv.setdefault("sched", {})
    _v35_send_email_capture_sms(phone, conv, reason)
    reply = "No problem. I sent you a text so you can type the email address exactly. Once you reply, we will finish the booking there. Thank you for calling Prevolt Electric. Goodbye."
    sched["pending_step"] = "need_email"
    sched["state"] = "waiting_for_email"
    sched["voice_close_after_reply"] = True
    sched["voice_email_text_handoff"] = True
    conv["last_voice_reply"] = reply
    return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_email", "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": True}


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    conv = hydrate_voice_conversation(p, call_sid)
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    low = _intent_text(caller_text or "")
    yn = _voice_yes_no_text(caller_text) if "_voice_yes_no_text" in globals() else ""

    # If we asked the caller to confirm an email and they reject it, do not keep
    # looping by voice. Email is the highest ASR-risk field. Move it to text.
    if sched.get("voice_awaiting_email_confirm"):
        if yn == "no" or any(pat in low for pat in ["wrong", "not correct", "incorrect", "nope", "never mind"]):
            return _v35_email_handoff_voice_reply(p, conv, "email_rejected_or_unclear")
        # If the caller tries to correct by spelling the last part, that is still too risky.
        # Send SMS instead of transforming phrases like "the last part is B O Y K O" into a bogus email.
        if any(pat in low for pat in ["last part", "starts with", "ends with", "spell", "spelled", "period", "dot"]) and not v13_extract_email(caller_text or ""):
            return _v35_email_handoff_voice_reply(p, conv, "email_correction_by_voice")

    # If the system is waiting for a last name and the caller spells it, store the clean form.
    step = str(sched.get("pending_step") or "").lower()
    state = str(sched.get("state") or "").lower()
    if step == "need_name" or state == "waiting_for_name":
        first = _voice_profile_first_name(profile) if "_voice_profile_first_name" in globals() else (profile.get("active_first_name") or profile.get("first_name"))
        clean_last = _v35_clean_last_from_spelled_text(caller_text)
        if first and clean_last:
            profile["active_last_name"] = clean_last
            profile["last_name"] = clean_last
            profile["name"] = f"{first} {clean_last}".strip()
            try:
                recompute_pending_step(profile, sched)
            except Exception:
                pass
            reply = _voice_naturalize_reply(_voice_after_name_reply(conv) if "_voice_after_name_reply" in globals() else "What's the best email address for the appointment?")
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V35(phone, call_sid, caller_text)
    try:
        conv2 = hydrate_voice_conversation(p, call_sid)
        reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or ""))
        # Final safety: do not allow another email confirmation after a yes.
        if yn == "yes" and conv2.setdefault("sched", {}).get("voice_email_confirmed") and "is that correct" in _intent_text(reply):
            reply = "Thanks. I'll finish the booking now."
        out["reply_to_customer"] = reply
        conv2["last_voice_reply"] = reply
    except Exception as e:
        try:
            log_event("VOICE_V35_POSTPROCESS_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


def _voice_thinking_click_payload(duration_ms: int | None = None) -> str:
    """Calmer, lower, fluttering PCMU/8000 processing cue.

    Intent: reassure the caller that the assistant is processing without sounding
    like a sharp notification ping. Softer low-mid boops, reverb tail, fade-out.
    """
    try:
        requested = int(duration_ms or VOICE_AGENT_THINKING_SOUND_MS or 1700)
    except Exception:
        requested = 1700
    ms = max(900, min(2200, requested))
    rate = 8000
    total = int(rate * (ms / 1000.0))
    pcm = [0] * total

    # One smooth phrase, not separate pings.
    starts_ms = [40, 210, 380, 570, 790]
    freqs = [185, 242, 214, 276, 232]
    lengths_ms = [360, 380, 420, 440, 500]
    gains = [0.075, 0.085, 0.070, 0.060, 0.045]

    for n, (start_ms, freq, length_ms, gain) in enumerate(zip(starts_ms, freqs, lengths_ms, gains)):
        start = int(rate * start_ms / 1000.0)
        length = min(int(rate * length_ms / 1000.0), max(0, total - start))
        if length <= 0:
            continue
        phase_offset = n * 0.91
        for i in range(length):
            idx = start + i
            t = i / rate
            x = i / max(1, length - 1)
            attack = min(1.0, i / max(1, int(rate * 0.055)))
            release = (1.0 - x) ** 2.05
            global_fade = max(0.0, 1.0 - (idx / max(1, total)) ** 1.9)
            vibrato = 12 * math.sin(2 * math.pi * 4.0 * t + phase_offset)
            tone = math.sin(2 * math.pi * (freq + vibrato) * t + phase_offset)
            tone += 0.13 * math.sin(2 * math.pi * (freq * 1.5) * t + 1.7)
            sample = int(15500 * gain * attack * release * global_fade * tone)
            pcm[idx] = max(-13000, min(13000, pcm[idx] + sample))
            for delay_ms, echo_gain in ((58, 0.28), (123, 0.16), (205, 0.075), (310, 0.035)):
                eidx = idx + int(rate * delay_ms / 1000.0)
                if eidx < total:
                    pcm[eidx] = max(-13000, min(13000, pcm[eidx] + int(sample * echo_gain)))

    # warm low pad underneath, very soft.
    for i in range(total):
        t = i / rate
        fade = max(0.0, 1.0 - (i / max(1, total)) ** 1.35)
        pad = int(280 * fade * (math.sin(2 * math.pi * 132 * t) + 0.35 * math.sin(2 * math.pi * 198 * t)))
        pcm[i] = max(-13000, min(13000, pcm[i] + pad))

    fade_samples = int(rate * 0.055)
    for i in range(min(fade_samples, total)):
        mult = i / max(1, fade_samples)
        pcm[i] = int(pcm[i] * mult)
        j = total - 1 - i
        pcm[j] = int(pcm[j] * mult)
    data = bytearray(_linear16_to_mulaw(sample) for sample in pcm)
    return base64.b64encode(bytes(data)).decode("ascii")


def incoming_sms_v35():
    inbound_text = request.form.get("Body", "") or ""
    phone_raw = request.form.get("From", "")
    phone = (phone_raw or "").replace("whatsapp:", "")
    email = v13_extract_email(inbound_text or "")
    conv = conversations.setdefault(phone, {})
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    # Voice email handoff: do not let the generic SMS handler parse the email local part as a name.
    if email and (sched.get("voice_email_text_handoff") or sched.get("voice_email_text_handoff_sent") or sched.get("pending_step") == "need_email" or sched.get("state") == "waiting_for_email"):
        v13_save_email(conv, email)
        try:
            _voice_sanitize_name_fields(profile, sched)
            _v32_apply_name_repairs(conv, profile.get("name") or "")
            recompute_pending_step(profile, sched)
        except Exception:
            pass
        first = _voice_profile_first_name(profile) if "_voice_profile_first_name" in globals() else (profile.get("active_first_name") or profile.get("first_name"))
        last = _voice_profile_last_name(profile) if "_voice_profile_last_name" in globals() else (profile.get("active_last_name") or profile.get("last_name"))
        if first and not last:
            sched["pending_step"] = "need_name"
            sched["state"] = "waiting_for_name"
            body = "Got it, thank you. What is your last name so we can finish booking?"
        elif not first:
            sched["pending_step"] = "need_name"
            sched["state"] = "waiting_for_name"
            body = "Got it, thank you. What is your first and last name so we can finish booking?"
        else:
            try:
                booking_attempt = maybe_create_square_booking(phone, conv)
            except Exception as e:
                try:
                    log_event("SMS_EMAIL_HANDOFF_BOOKING_ERROR_V35", phone, {"error": repr(e)}, conv)
                except Exception:
                    pass
                booking_attempt = {"status": "exception"}
            status = booking_attempt.get("status") if isinstance(booking_attempt, dict) else None
            if sched.get("booking_created") and sched.get("square_booking_id") or status in {"created", "success", "booked"}:
                body = _voice_to_sms_text(_voice_finalize_booking_reply(conv, "")) if "_voice_finalize_booking_reply" in globals() else "You're all set. We have you on the schedule."
            else:
                try:
                    body = _voice_to_sms_text(_voice_naturalize_reply(choose_next_prompt_from_state(conv, inbound_text=inbound_text)))
                except Exception:
                    body = "Got it, thank you. We have your email and will finish getting you on the schedule."
        conv["last_sms_body"] = body
        try:
            log_event("SMS_EMAIL_HANDOFF_V35", phone, {"email": email, "body": _safe_monitor_text(body), "name": profile.get("name")}, conv)
        except Exception:
            pass
        tw = MessagingResponse()
        tw.message(body)
        return Response(str(tw), mimetype="text/xml")
    return _ORIG_INCOMING_SMS_V35()

try:
    app.view_functions["incoming_sms"] = incoming_sms_v35
except Exception:
    pass

# =============================
# v36 VOICE HOTFIX LAYER
# =============================
# Corrects v35 strategy: stay in the voice booking lane for email corrections.
# No SMS handoff just because an email readback was wrong. The assistant should
# calmly ask for the full email again or apply a clear segment correction.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V36 = process_prevolt_voice_turn
_ORIG_VOICE_NATURALIZE_REPLY_V36 = _voice_naturalize_reply

_V36_EXTRA_FILLER_PATTERNS = [
    r"\bOkay,?\s+let me\s+[^.?!]*?(?:sort|check|get|grab|process|run|confirm|update|wrap|finish)[^.?!]*[.?!]\s*",
    r"\bThanks,?\s+let me\s+[^.?!]*?(?:sort|check|get|grab|process|run|confirm|update|wrap|finish)[^.?!]*[.?!]\s*",
    r"\bGot it,?\s+let me\s+[^.?!]*?(?:sort|check|get|grab|process|run|confirm|update|wrap|finish)[^.?!]*[.?!]\s*",
    r"\bI(?:'|’)?m\s+(?:going to|gonna)\s+[^.?!]*?(?:sort|check|get|grab|process|run|confirm|update|wrap|finish)[^.?!]*[.?!]\s*",
    r"\bI(?:'|’)?ll\s+[^.?!]*?(?:move ahead|use that email|update that spelling|confirm|wrap|finish)[^.?!]*[.?!]\s*",
]


def _voice_naturalize_reply(reply: str) -> str:
    r = _ORIG_VOICE_NATURALIZE_REPLY_V36(reply or "")
    changed = True
    while changed:
        old = r
        for pat in _V36_EXTRA_FILLER_PATTERNS:
            r = re.sub(pat, "", r, flags=re.I)
        r = re.sub(r"\s+", " ", r).strip()
        changed = (r != old)
    # Keep price copy clean.
    r = r.replace("If that is something you're interested in. Does that work for you?", "Does that work for you?")
    return r


_V36_LETTER_WORDS = dict(globals().get("_V32_LETTER_WORDS", {}))
_V36_LETTER_WORDS.update({
    "boy": "b", "bravo": "b", "victor": "v", "voicemail": "v", "yellow": "y", "yankee": "y",
    "kilo": "k", "okay": "o", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
})


def _v36_compact_letters(tokens: list[str]) -> str:
    out = []
    for tok in tokens:
        t = re.sub(r"[^a-z0-9]", "", str(tok or "").lower())
        if not t:
            continue
        if len(t) == 1 and (t.isalnum()):
            out.append(t)
        elif t in _V36_LETTER_WORDS:
            out.append(_V36_LETTER_WORDS[t])
        elif len(t) > 1:
            out.append(t)
    return "".join(out)


def _v36_normalize_spoken_email_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.replace(" at the rate of ", " at ")
    s = re.sub(r"\b(period|dot)\b", " . ", s)
    s = re.sub(r"\b(at|at sign)\b", " @ ", s)
    s = re.sub(r"\bg mail\b", " gmail ", s)
    s = re.sub(r"\bgee mail\b", " gmail ", s)
    s = s.replace("dash", "-").replace("underscore", "_")
    s = s.replace("hyphen", "-")
    # Remove obvious correction chatter before parsing.
    s = re.sub(r"\b(no|nope|wrong|incorrect|not correct|that's wrong|thats wrong|it is|it's|its|the email is|email is)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _v36_tokens_to_email(tokens: list[str]) -> str:
    parts = []
    for tok in tokens:
        t = str(tok or "").lower().strip()
        if not t:
            continue
        if t in {"@", "at"}:
            parts.append("@")
        elif t in {".", "dot", "period"}:
            parts.append(".")
        elif t in {"dash", "hyphen"}:
            parts.append("-")
        elif t in {"underscore"}:
            parts.append("_")
        elif re.fullmatch(r"[a-z0-9]", t):
            parts.append(t)
        elif t in _V36_LETTER_WORDS:
            parts.append(_V36_LETTER_WORDS[t])
        elif re.fullmatch(r"[a-z0-9][a-z0-9._%+\-]*", t):
            parts.append(t)
    email = "".join(parts)
    email = email.replace("@@", "@")
    email = re.sub(r"\.+", ".", email)
    email = email.strip(" .")
    if re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", email):
        return email.lower()
    return ""


def _v36_parse_spoken_email(text: str, pending_email: str = "") -> str:
    """Best-effort email parser for voice correction without leaving the call."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    direct = v13_extract_email(raw) if "v13_extract_email" in globals() else None
    if direct:
        return direct.lower()

    s = _v36_normalize_spoken_email_text(raw)
    # Pattern: amy dot m dot boyko at gmail dot com
    tokens = re.findall(r"@|\.|[a-z0-9]+", s)
    email = _v36_tokens_to_email(tokens)
    if email:
        return email

    pending = (pending_email or "").strip().lower()
    if pending and "@" in pending:
        local, domain = pending.split("@", 1)
        local_parts = local.split(".")
        # Pattern: "the last part is B O Y K O" or "last part B-O-Y-K-O at gmail".
        if re.search(r"\blast\s+part\b|\blast\s+piece\b|\bend\b|\bending\b", s):
            after = re.split(r"\blast\s+part\s*(?:is)?\b|\blast\s+piece\s*(?:is)?\b|\bending\s*(?:is)?\b|\bend\s*(?:is)?\b", s, maxsplit=1)
            segment_text = after[-1] if len(after) > 1 else s
            # Drop domain words if present.
            segment_text = re.split(r"\s+@\s+|\s+at\s+", segment_text, maxsplit=1)[0]
            seg_tokens = re.findall(r"[a-z0-9]+", segment_text)
            seg = _v36_compact_letters(seg_tokens)
            if seg and len(seg) >= 2:
                local_parts[-1] = seg
                return (".".join(local_parts) + "@" + domain).lower()
        # Pattern: caller gives just a trailing chunk like "Y K O" while pending is boiko.
        short_tokens = re.findall(r"[a-z0-9]+", s)
        short = _v36_compact_letters(short_tokens)
        if short and 2 <= len(short) <= 8 and not any(w in s for w in ["gmail", "yahoo", "hotmail", "outlook", "@"]):
            last = local_parts[-1] if local_parts else local
            if len(short) < len(last):
                local_parts[-1] = last[: max(0, len(last) - len(short))] + short
            else:
                local_parts[-1] = short
            return (".".join(local_parts) + "@" + domain).lower()
    return ""


def _v36_spell_email_for_voice(email: str) -> str:
    e = str(email or "").strip().lower()
    if not e:
        return ""
    chunks = []
    for ch in e:
        if ch == ".":
            chunks.append("dot")
        elif ch == "@":
            chunks.append("at")
        elif ch == "-":
            chunks.append("dash")
        elif ch == "_":
            chunks.append("underscore")
        elif ch == "b":
            chunks.append("B as in boy")
        elif ch == "v":
            chunks.append("V as in Victor")
        else:
            chunks.append(ch.upper() if ch.isalpha() else ch)
    return ", ".join(chunks)


def _v36_email_confirmation_reply(email: str, attempt: int = 0) -> str:
    cleaned = str(email or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", cleaned):
        return "Please say the full email address again, including the at sign and dot com."
    spoken = _v36_spell_email_for_voice(cleaned)
    if attempt <= 1:
        return f"I heard {spoken}. Is that correct?"
    return f"I have {spoken}. Is that correct?"


def _v36_confirmed_email_save_and_book(phone: str, conv: dict, email: str) -> dict:
    sched = conv.setdefault("sched", {})
    sched["voice_email_confirmed"] = True
    sched["voice_awaiting_email_confirm"] = False
    sched["voice_pending_email"] = None
    return _voice_save_confirmed_email_and_maybe_book_v31(phone, conv, email)


# Revert the v35 email-to-text handoff behavior. Keep correction in voice lane.
def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    conv = hydrate_voice_conversation(p, call_sid)
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    low = _intent_text(caller_text or "")
    yn = _voice_yes_no_text(caller_text) if "_voice_yes_no_text" in globals() else ""

    try:
        _v32_apply_name_repairs(conv, caller_text)
        _voice_sanitize_name_fields(profile, sched)
    except Exception:
        pass

    # If we're waiting for email confirmation, stay on the voice call.
    if sched.get("voice_awaiting_email_confirm"):
        pending = str(sched.get("voice_pending_email") or "").strip().lower()
        attempt = int(sched.get("voice_email_correction_attempts") or 0)
        if yn == "yes" and pending:
            out = _v36_confirmed_email_save_and_book(p, conv, pending)
            out["reply_to_customer"] = _voice_naturalize_reply(out.get("reply_to_customer") or "")
            conv["last_voice_reply"] = out["reply_to_customer"]
            return out
        if yn == "no" or any(x in low for x in ["wrong", "not correct", "incorrect", "nope", "never mind", "not it"]):
            corrected = _v36_parse_spoken_email(caller_text, pending)
            if corrected and corrected != pending:
                sched["voice_pending_email"] = corrected
                sched["voice_email_correction_attempts"] = attempt + 1
                reply = _v36_email_confirmation_reply(corrected, attempt + 1)
            else:
                # The caller rejected the heard email and did not provide a complete
                # replacement in the same utterance. Clear the bad pending value so
                # we do not keep repeating a malformed partial such as kyle.prevost.
                sched["voice_pending_email"] = None
                sched["voice_email_correction_attempts"] = attempt + 1
                if attempt == 0:
                    reply = "No problem. Please say the full email from the beginning, one character at a time. Say dot for a period and at for the at sign."
                else:
                    reply = "Let's do the full email one more time from the beginning. Please say each character slowly, including dot and at."
            conv["last_voice_reply"] = reply
            sched["pending_step"] = "need_email"
            sched["state"] = "waiting_for_email"
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_email", "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
        corrected = _v36_parse_spoken_email(caller_text, pending)
        if corrected:
            sched["voice_pending_email"] = corrected
            reply = _v36_email_confirmation_reply(corrected, attempt)
            conv["last_voice_reply"] = reply
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_email", "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
        # Unclear response while confirming email: ask a concise yes/no.
        reply = _v36_email_confirmation_reply(pending, attempt) if pending else "Please say the full email address again."
        conv["last_voice_reply"] = reply
        return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_email", "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    # If waiting for email and caller gives one by voice, confirm it. Do not book yet.
    if str(sched.get("pending_step") or "").lower() == "need_email" or "email" in str(sched.get("state") or "").lower():
        incoming = _v36_parse_spoken_email(caller_text, "")
        if incoming and not sched.get("voice_email_confirmed"):
            sched["voice_pending_email"] = incoming
            sched["voice_awaiting_email_confirm"] = True
            sched["voice_email_correction_attempts"] = 0
            sched["pending_step"] = "need_email"
            sched["state"] = "waiting_for_email"
            reply = _v36_email_confirmation_reply(incoming, 0)
            conv["last_voice_reply"] = reply
            try:
                log_event("VOICE_EMAIL_CONFIRM_PROMPT_V36", p, {"email": incoming, "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_email", "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}

    # For all non-email-correction turns, use the previous v35 behavior, then force the reply/actionability.
    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V36(phone, call_sid, caller_text)
    try:
        conv2 = hydrate_voice_conversation(p, call_sid)
        sched2 = conv2.setdefault("sched", {})
        profile2 = conv2.setdefault("profile", {})
        try:
            _v32_apply_name_repairs(conv2, caller_text)
            _voice_sanitize_name_fields(profile2, sched2)
        except Exception:
            pass
        reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or ""))
        if "_v34_force_actionable_reply" in globals():
            reply = _v34_force_actionable_reply(conv2, reply)
        # Remove accidental duplicated email-confirmation prompt if it somehow leaks.
        if sched2.get("voice_email_confirmed") and "is that correct" in _intent_text(reply):
            reply = "Thanks. I'll finish the booking now."
        out["reply_to_customer"] = reply
        conv2["last_voice_reply"] = reply
    except Exception as e:
        try:
            log_event("VOICE_V36_POSTPROCESS_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid}, conv)
        except Exception:
            pass
    return out


# Make the processing cue calmer: soft two-tone flutter with a longer reverb tail.
def _voice_thinking_click_payload(duration_ms: int | None = None) -> str:
    try:
        requested = int(duration_ms or VOICE_AGENT_THINKING_SOUND_MS or 1700)
    except Exception:
        requested = 1700
    ms = max(1200, min(2400, requested))
    rate = 8000
    total = int(rate * (ms / 1000.0))
    pcm = [0] * total

    # Smooth overlapping notes: more calm IVR/assistant cue, less sharp ping.
    events = [
        (45, 176, 520, 0.052),
        (250, 221, 620, 0.060),
        (490, 196, 650, 0.050),
        (760, 247, 700, 0.040),
    ]
    for n, (start_ms, freq, length_ms, gain) in enumerate(events):
        start = int(rate * start_ms / 1000.0)
        length = min(int(rate * length_ms / 1000.0), max(0, total - start))
        phase = n * 0.83
        for i in range(length):
            idx = start + i
            t = i / rate
            x = i / max(1, length - 1)
            attack = min(1.0, i / max(1, int(rate * 0.09)))
            release = (1.0 - x) ** 2.6
            global_fade = max(0.0, 1.0 - (idx / max(1, total)) ** 1.55)
            flutter = 7.0 * math.sin(2 * math.pi * 3.1 * t + phase)
            tone = math.sin(2 * math.pi * (freq + flutter) * t + phase)
            tone += 0.08 * math.sin(2 * math.pi * (freq * 2.0) * t + phase + 1.2)
            sample = int(15000 * gain * attack * release * global_fade * tone)
            pcm[idx] = max(-12000, min(12000, pcm[idx] + sample))
            # Reverb tail delays; soft enough not to sound like static.
            for delay_ms, eg in ((86, 0.24), (171, 0.15), (286, 0.075), (430, 0.04)):
                eidx = idx + int(rate * delay_ms / 1000.0)
                if eidx < total:
                    pcm[eidx] = max(-12000, min(12000, pcm[eidx] + int(sample * eg)))

    # Gentle warm bed underneath.
    for i in range(total):
        t = i / rate
        fade = max(0.0, 1.0 - (i / max(1, total)) ** 1.2)
        pad = int(220 * fade * (math.sin(2 * math.pi * 118 * t) + 0.25 * math.sin(2 * math.pi * 164 * t)))
        pcm[i] = max(-12000, min(12000, pcm[i] + pad))

    fade_samples = int(rate * 0.08)
    for i in range(min(fade_samples, total)):
        mult = i / max(1, fade_samples)
        pcm[i] = int(pcm[i] * mult)
        j = total - 1 - i
        pcm[j] = int(pcm[j] * mult)
    return base64.b64encode(bytes(_linear16_to_mulaw(s) for s in pcm)).decode("ascii")


# =============================
# v37 VOICE HOTFIX LAYER
# =============================
# Targets the final minor live issue after v36:
# - If caller says a local town in the opening description ("I live in Windsor")
#   and then gives only the house number/street, do not ask the town/state again.
#   Reuse the earlier town hint and advance the existing address-normalization flow.
# - Keep this intentionally narrow to avoid disturbing the now-stable name/email/date paths.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V37 = process_prevolt_voice_turn
_ORIG_VOICE_NATURALIZE_REPLY_V37 = _voice_naturalize_reply

# Local towns that are safe to infer as Connecticut when spoken without a state.
# Keep this conservative. These are the common Prevolt CT service-area towns where
# asking "Connecticut or Massachusetts?" feels dumb to the caller.
_V37_LOCAL_CT_TOWN_HINTS = [
    "windsor locks", "south windsor", "east windsor", "west hartford", "east hartford",
    "wethersfield", "rocky hill", "newington", "new britain", "bristol",
    "plainville", "southington", "farmington", "avon", "simsbury", "canton",
    "bloomfield", "granby", "east granby", "suffield", "enfield", "ellington",
    "vernon", "tolland", "manchester", "glastonbury", "hartford", "windsor",
]


def _v37_state_from_text(text: str, default: str = "CT") -> str:
    low = _intent_text(text or "")
    if re.search(r"\b(?:massachusetts|ma)\b", low):
        return "MA"
    if re.search(r"\b(?:connecticut|ct)\b", low):
        return "CT"
    return default


def _v37_extract_opening_town_hint(text: str) -> tuple[str, str] | tuple[None, None]:
    """Extract a conservative town/state hint from natural caller speech.

    Examples:
      "I live in Windsor and need an EV charger" -> ("Windsor", "CT")
      "I'm in South Windsor" -> ("South Windsor", "CT")

    This intentionally does NOT try to parse arbitrary towns; it only handles
    known local towns to avoid incorrectly skipping town/state confirmation.
    """
    raw = str(text or "")
    low = _intent_text(raw)
    if not low:
        return None, None
    # Prefer longer names first so "South Windsor" wins before "Windsor".
    for town in sorted(_V37_LOCAL_CT_TOWN_HINTS, key=len, reverse=True):
        # Require either a location cue or a very clear standalone local-town mention.
        loc_cue = rf"\b(?:i live in|we live in|i am in|i'm in|we are in|we're in|located in|property is in|work is in|house is in|home is in|in)\s+{re.escape(town)}\b"
        standalone = rf"\b{re.escape(town)}\s*,?\s*(?:connecticut|ct)\b"
        if re.search(loc_cue, low) or re.search(standalone, low):
            clean = " ".join(w.capitalize() for w in town.split())
            return clean, _v37_state_from_text(raw, "CT")
    return None, None


def _v37_store_town_hint(conv: dict, caller_text: str) -> None:
    town, state = _v37_extract_opening_town_hint(caller_text)
    if not town:
        return
    sched = conv.setdefault("sched", {})
    # Do not overwrite a verified address. This is only a hint for partial street addresses.
    if sched.get("address_verified"):
        return
    sched["voice_town_hint"] = town
    sched["voice_state_hint"] = state or "CT"
    try:
        log_event("VOICE_TOWN_HINT_CAPTURED_V37", conv.get("phone") or "", {"town": town, "state": state, "caller_text": _safe_monitor_text(caller_text), "call_sid": conv.get("last_call_sid")}, conv)
    except Exception:
        pass


def _v37_is_town_question(reply: str) -> bool:
    low = _intent_text(reply or "")
    return (
        "which town is it in" in low
        or "what town is it in" in low
        or "which town is this in" in low
        or "what town is this in" in low
        or "connecticut or massachusetts" in low
        or "ct or ma" in low
    )


def _v37_has_partial_street(conv: dict) -> bool:
    sched = conv.setdefault("sched", {})
    raw = str(sched.get("raw_address") or sched.get("address_candidate") or sched.get("normalized_address") or "")
    if not raw:
        return False
    try:
        if _address_has_house_number_and_street(raw):
            return True
    except Exception:
        pass
    return bool(re.search(r"\b\d{1,6}\s+[A-Za-z0-9][A-Za-z0-9 .'-]*(?:ave|avenue|st|street|rd|road|dr|drive|ln|lane|ct|court|way|blvd|boulevard|circle|cir|terrace|ter)\b", raw, flags=re.I))


def _v37_should_auto_apply_town_hint(conv: dict, reply: str) -> bool:
    sched = conv.setdefault("sched", {})
    if sched.get("address_verified") or sched.get("voice_town_hint_used"):
        return False
    if not _v37_is_town_question(reply):
        return False
    if not sched.get("voice_town_hint"):
        return False
    return _v37_has_partial_street(conv)


def _voice_naturalize_reply(reply: str) -> str:
    r = _ORIG_VOICE_NATURALIZE_REPLY_V37(reply or "")
    # One more conservative filler cleanup from the latest successful call.
    r = re.sub(r"\b(?:Okay,?\s*)?(?:Got it,?\s*)?(?:Thanks,?\s*)?Let me think through the next scheduling details with you\.\s*", "", r, flags=re.I)
    r = re.sub(r"\b(?:Okay,?\s*)?(?:Got it,?\s*)?(?:Thanks,?\s*)?Let me get the next detail we need to finish scheduling\.\s*", "", r, flags=re.I)
    r = re.sub(r"\s+", " ", r).strip()
    return r


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        _v37_store_town_hint(conv_pre, caller_text)
    except Exception as e:
        try:
            log_event("VOICE_V37_TOWN_HINT_PRE_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V37(phone, call_sid, caller_text)

    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        _v37_store_town_hint(conv, caller_text)
        reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or ""))

        # If the caller already gave a local town in the opening description, and
        # the only missing address piece is town/state, silently feed that town
        # into the existing backend flow and return the next real prompt.
        if _v37_should_auto_apply_town_hint(conv, reply):
            sched = conv.setdefault("sched", {})
            town = str(sched.get("voice_town_hint") or "").strip()
            state = str(sched.get("voice_state_hint") or "CT").strip() or "CT"
            sched["voice_town_hint_used"] = True
            synthetic = f"{town}, {'Connecticut' if state.upper() == 'CT' else 'Massachusetts'}"
            try:
                log_event("VOICE_TOWN_HINT_APPLIED_V37", p, {"synthetic_town": synthetic, "previous_reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
            except Exception:
                pass
            out2 = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V37(phone, call_sid, synthetic)
            try:
                conv2 = hydrate_voice_conversation(p, call_sid)
                reply2 = _voice_naturalize_reply(str(out2.get("reply_to_customer") or ""))
                out2["reply_to_customer"] = reply2
                conv2["last_voice_reply"] = reply2
            except Exception:
                pass
            return out2

        out["reply_to_customer"] = reply
        conv["last_voice_reply"] = reply
    except Exception as e:
        try:
            log_event("VOICE_V37_POSTPROCESS_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out



# =============================
# v38 MAPS ADDRESS + DISTANCE HOTFIX LAYER
# =============================
# Targets the issue Kyle identified after v37:
# - Do not rely on a tiny hard-coded town list when a caller gives a partial street address.
# - Use Google Maps Geocoding to resolve partial addresses like "45 Dickerman Ave" when
#   the result is complete and unambiguous in CT/MA.
# - Use Google Maps Distance Matrix to enforce residential travel distance from the dispatch origin.
# - Keep this as a narrow wrapper around the now-stable v36/v37 voice flow.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V38 = process_prevolt_voice_turn


def _v38_clean_addr_text(value: str) -> str:
    return " ".join(str(value or "").replace("\n", " ").split()).strip(" ,.;")


def _v38_address_line_key(value: str) -> str:
    low = _intent_text(value or "")
    low = re.sub(r"\b(?:street|st)\b", "st", low)
    low = re.sub(r"\b(?:avenue|ave)\b", "ave", low)
    low = re.sub(r"\b(?:road|rd)\b", "rd", low)
    low = re.sub(r"\b(?:drive|dr)\b", "dr", low)
    low = re.sub(r"\b(?:lane|ln)\b", "ln", low)
    low = re.sub(r"\b(?:court|ct)\b", "ct", low)
    low = re.sub(r"\b(?:boulevard|blvd)\b", "blvd", low)
    low = re.sub(r"\b(?:circle|cir)\b", "cir", low)
    low = re.sub(r"\b(?:terrace|ter)\b", "ter", low)
    low = re.sub(r"[^a-z0-9 ]+", " ", low)
    return re.sub(r"\s+", " ", low).strip()


def _v38_complete_addr_struct(a: dict | None) -> bool:
    return bool(
        isinstance(a, dict)
        and (a.get("address_line_1") or "").strip()
        and (a.get("locality") or "").strip()
        and (a.get("administrative_district_level_1") or "").strip()
        and (a.get("postal_code") or "").strip()
        and re.match(r"^\d{1,6}\b", (a.get("address_line_1") or "").strip())
    )


def _v38_addr_key(a: dict) -> tuple[str, str, str, str]:
    return (
        _v38_address_line_key(a.get("address_line_1") or ""),
        _intent_text(a.get("locality") or ""),
        (a.get("administrative_district_level_1") or "").strip().upper(),
        (a.get("postal_code") or "").strip(),
    )


def _v38_raw_line_matches_normalized(raw: str, a: dict) -> bool:
    """Guard against Google returning a vague/nearby result for a partial address."""
    raw_key = _v38_address_line_key(raw)
    line_key = _v38_address_line_key(a.get("address_line_1") or "")
    if not raw_key or not line_key:
        return False
    raw_num = re.match(r"^(\d{1,6})\b", raw_key)
    line_num = re.match(r"^(\d{1,6})\b", line_key)
    if raw_num and line_num and raw_num.group(1) != line_num.group(1):
        return False
    # Street name tokens after the number must substantially overlap.
    raw_tokens = [t for t in raw_key.split() if not t.isdigit()]
    line_tokens = [t for t in line_key.split() if not t.isdigit()]
    if not raw_tokens or not line_tokens:
        return False
    shared = set(raw_tokens) & set(line_tokens)
    return bool(shared) and (raw_key in line_key or line_key in raw_key or len(shared) >= min(2, len(raw_tokens)))


def _v38_normalize_attempt(raw: str, forced_state: str | None = None) -> dict | None:
    try:
        result = normalize_address(raw, forced_state=forced_state) if forced_state else normalize_address(raw)
    except Exception as e:
        try:
            print("[WARN] v38 normalize attempt failed:", repr(e))
        except Exception:
            pass
        return None
    status = None
    addr = None
    if isinstance(result, tuple) and len(result) >= 2:
        status, addr = result[0], result[1]
    elif isinstance(result, dict):
        status, addr = "ok", result
    if status == "ok" and _v38_complete_addr_struct(addr):
        return addr
    return None





# VOICE_V58_NO_PARTIAL_GOOGLE_GUESS: Google validation is allowed only after caller-supplied town/state.
# =============================
# Voice stabilization v58 — no Google guessing on street-only partial addresses
# =============================
# Root cause fixed:
# A caller could give a partial/noisy address such as "40 Arch Street, above".
# Older layers would let Google Geocoding guess a complete address like
# Greenwich, CT before the caller supplied the town/state. That stale guess
# could then close a valid lead as out-of-area.
#
# v58 rule:
# Google Maps may validate/normalize a customer address, but it may not invent
# the missing town/state for voice scheduling. If the caller gives street-only
# or a noisy partial address, the system must ask for the town/state first.


def _v58_clean_town_candidate(value: str | None) -> str | None:
    town = " ".join(str(value or "").replace("\n", " ").split()).strip(" ,.")
    if not town:
        return None

    low = re.sub(r"[^a-z0-9 ]+", " ", town.lower())
    low = re.sub(r"\s+", " ", low).strip()

    # Words that commonly get captured from natural speech but are not towns.
    junk = {
        "above", "upstairs", "downstairs", "basement", "apartment", "apt", "unit",
        "home", "house", "work", "quote", "estimate", "repair", "fix", "fixed",
        "outlet", "outlets", "light", "lights", "panel", "breaker", "circuit",
        "someone", "somebody", "today", "tomorrow", "tonight", "please",
    }
    if low in junk:
        return None
    if any(word in low.split() for word in {"quote", "estimate", "repair", "fix", "outlet", "outlets", "panel", "breaker"}):
        return None

    return " ".join(w.capitalize() for w in town.split())


def _v58_state_from_any_text(value: str | None) -> str | None:
    low = _intent_text(value or "") if "_intent_text" in globals() else str(value or "").lower()
    if re.search(r"\b(?:ct|connecticut|c t)\b", low):
        return "CT"
    if re.search(r"\b(?:ma|massachusetts|mass|m a)\b", low):
        return "MA"
    return None


def _v58_known_state_for_town(town: str | None) -> str | None:
    town_clean = _v58_clean_town_candidate(town)
    if not town_clean:
        return None

    try:
        if "_v50_known_local_town" in globals():
            if _v50_known_local_town(town_clean, "CT"):
                return "CT"
            if _v50_known_local_town(town_clean, "MA"):
                return "MA"
            if _v50_known_local_town(town_clean, None):
                # Fall through to the older town detector if the state is not obvious.
                pass
    except Exception:
        pass

    try:
        if "_v47_known_ct_town" in globals():
            st = _v47_known_ct_town(town_clean)
            if st in {"CT", "MA"}:
                return st
    except Exception:
        pass

    return None


def _v58_prepare_address_for_maps(raw_address: str | None) -> tuple[bool, str | None, str | None, str | None, str]:
    """Return (allowed, safe_raw, forced_state, town, reason).

    allowed is True only when the raw address includes:
    - a house number and street, and
    - a caller-supplied town/state or a known local town hint.

    This prevents Google from guessing the missing town/state.
    """
    raw = " ".join(str(raw_address or "").replace("\n", " ").split()).strip(" ,.")
    if not raw:
        return False, None, None, None, "empty"

    try:
        line, town, state = _v47_extract_address_candidate(raw)
    except Exception:
        line, town, state = None, None, None

    if not line:
        try:
            line = extract_service_address_from_text(raw)
        except Exception:
            line = None

    if not line or not _address_has_house_number_and_street(line):
        return False, None, None, None, "no_house_number_and_street"

    town = _v58_clean_town_candidate(town)

    # If the raw address explicitly contains CT/MA, preserve that state.
    state = state or _v58_state_from_any_text(raw)

    # If town is known local but state was omitted, infer only from the known town list.
    if town and not state:
        state = _v58_known_state_for_town(town)

    # If state is present but town is not, still do not let Google pick the town.
    if not town:
        return False, None, state, None, "missing_town"

    # If town is unknown and there is no state, do not geocode.
    if not state:
        return False, None, None, town, "unknown_town_without_state"

    safe_raw = f"{line}, {town}, {state}"
    return True, safe_raw, state, town, "customer_locality_present"


def _v58_locality_ok(addr: dict | None, town: str | None, state: str | None) -> bool:
    if not isinstance(addr, dict):
        return False
    if not town or not state:
        return False
    try:
        if "_v50_locality_matches" in globals():
            return bool(_v50_locality_matches(addr, town, state))
    except Exception:
        pass

    a_town = re.sub(r"[^a-z0-9 ]+", " ", str(addr.get("locality") or "").lower()).strip()
    want = re.sub(r"[^a-z0-9 ]+", " ", str(town or "").lower()).strip()
    a_state = str(addr.get("administrative_district_level_1") or "").upper().strip()
    return bool(a_town and want and a_town == want and a_state == str(state or "").upper().strip())


def _v38_google_resolve_partial_address(raw_address: str) -> tuple[dict | None, str]:
    """
    Resolve a customer address using Google Geocoding, but do not let Google
    invent the missing town/state for voice scheduling.

    v58: A street-only or noisy partial address must return a "needs town/state"
    reason instead of guessing one complete Google result.
    """
    raw = _v38_clean_addr_text(raw_address)
    if not raw or not _address_has_house_number_and_street(raw):
        return None, "not_house_number_and_street"

    try:
        allowed, safe_raw, forced_state, town, why = _v58_prepare_address_for_maps(raw)
    except Exception:
        allowed, safe_raw, forced_state, town, why = False, None, None, None, "v58_guard_error"

    if not allowed:
        return None, f"needs_customer_town_state:{why}"

    candidates: list[dict] = []

    forced = _v38_normalize_attempt(safe_raw or raw, forced_state)
    if forced and _v38_raw_line_matches_normalized(safe_raw or raw, forced) and _v58_locality_ok(forced, town, forced_state):
        candidates.append(forced)

    unique: dict[tuple[str, str, str, str], dict] = {}
    for c in candidates:
        unique[_v38_addr_key(c)] = c

    vals = list(unique.values())
    if len(vals) == 1:
        return vals[0], "customer_locality_google_match"

    return None, "no_customer_locality_google_match"
def _v38_apply_normalized_address(conv: dict, addr: dict, source: str) -> None:
    sched = conv.setdefault("sched", {})
    sched["normalized_address"] = dict(addr)
    sched["raw_address"] = (
        f"{addr.get('address_line_1')}, {addr.get('locality')}, "
        f"{addr.get('administrative_district_level_1')} {addr.get('postal_code')}"
    ).strip(" ,")
    sched["address_candidate"] = sched["raw_address"]
    sched["address_verified"] = True
    sched["address_missing"] = None
    sched["address_parts"] = {
        "street": True,
        "number": True,
        "city": True,
        "state": True,
        "zip": True,
        "source": source,
    }
    sched["pending_step"] = None if sched.get("pending_step") == "need_address" else sched.get("pending_step")


def _v38_should_maps_resolve(conv: dict, reply: str | None = None) -> bool:
    sched = conv.setdefault("sched", {})
    if sched.get("address_verified"):
        return False
    raw = _v38_clean_addr_text(sched.get("raw_address") or sched.get("address_candidate") or "")
    if not raw or not _address_has_house_number_and_street(raw):
        return False
    try:
        allowed, _safe_raw, _forced_state, _town, _why = _v58_prepare_address_for_maps(raw)
        if not allowed:
            return False
    except Exception:
        return False
    if reply and not _v37_is_town_question(reply):
        return False
    return True


def _v38_try_maps_resolve_for_voice(conv: dict, reply: str | None = None) -> tuple[bool, dict | None, str]:
    if not _v38_should_maps_resolve(conv, reply):
        return False, None, "not_needed"
    sched = conv.setdefault("sched", {})
    raw = _v38_clean_addr_text(sched.get("raw_address") or sched.get("address_candidate") or "")
    addr, reason = _v38_google_resolve_partial_address(raw)
    if not addr:
        return False, None, reason
    _v38_apply_normalized_address(conv, addr, "google_maps_v38")
    return True, addr, reason


# Define/override travel-time calculation so the residential out-of-area gate uses Google Maps.
def compute_travel_time_minutes(origin: str, destination: str) -> int | None:
    if not GOOGLE_MAPS_API_KEY or not origin or not destination:
        return None
    try:
        params = {
            "origins": origin,
            "destinations": destination,
            "mode": "driving",
            "units": "imperial",
            "key": GOOGLE_MAPS_API_KEY,
        }
        # Add traffic-aware duration when Google accepts it.
        try:
            params["departure_time"] = "now"
        except Exception:
            pass
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params=params,
            timeout=8,
        )
        data = resp.json()
        if data.get("status") != "OK":
            print("[WARN] Distance Matrix status:", data.get("status"))
            return None
        rows = data.get("rows") or []
        elements = (rows[0].get("elements") if rows else []) or []
        if not elements or elements[0].get("status") != "OK":
            print("[WARN] Distance Matrix element status:", elements[0].get("status") if elements else None)
            return None
        elem = elements[0]
        dur = elem.get("duration_in_traffic") or elem.get("duration") or {}
        seconds = dur.get("value")
        if not seconds:
            return None
        return int(round(float(seconds) / 60.0))
    except Exception as e:
        print("[ERROR] compute_travel_time_minutes:", repr(e))
        return None


def _v38_origin_address() -> str:
    return (TECH_CURRENT_ADDRESS or DISPATCH_ORIGIN_ADDRESS or os.environ.get("VOICE_DISPATCH_ORIGIN_ADDRESS") or "Windsor Locks, CT").strip()


def _voice_residential_address_too_far(conv: dict) -> tuple[bool, int | None]:
    """v38: Google Maps travel-time gate for residential voice calls."""
    sched = conv.setdefault("sched", {})
    dest = _voice_destination_from_sched(sched)
    if not dest:
        return (False, None)
    if _voice_is_out_of_area_town(dest):
        return (True, None)
    origin = _v38_origin_address()
    if not origin:
        return (False, None)
    minutes = compute_travel_time_minutes(origin, dest)
    if minutes is not None and minutes > VOICE_RESIDENTIAL_MAX_TRAVEL_MINUTES:
        return (True, int(minutes))
    return (False, int(minutes) if minutes is not None else None)


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V38(phone, call_sid, caller_text)

    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or ""))

        # If the existing flow asks town/state after a partial street address, let Google Maps
        # try to complete it first. If it is unique in CT/MA, silently advance to the next step.
        ok, addr, reason = _v38_try_maps_resolve_for_voice(conv, reply)
        if ok and addr:
            try:
                log_event(
                    "VOICE_GOOGLE_ADDRESS_RESOLVED_V38",
                    p,
                    {
                        "raw_address": _safe_monitor_text(caller_text),
                        "resolved": f"{addr.get('address_line_1')}, {addr.get('locality')}, {addr.get('administrative_district_level_1')} {addr.get('postal_code')}",
                        "reason": reason,
                        "call_sid": call_sid,
                    },
                    conv,
                )
            except Exception:
                pass

            # Feed a harmless synthetic city/state confirmation into the original stable flow
            # so it returns the next real scheduling prompt (usually the $195 evaluation consent).
            synthetic = f"{addr.get('locality')}, {addr.get('administrative_district_level_1')}"
            out2 = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V38(phone, call_sid, synthetic)
            try:
                conv2 = hydrate_voice_conversation(p, call_sid)
                conv2["phone"] = p
                _v38_apply_normalized_address(conv2, addr, "google_maps_v38")
                # Run the out-of-area travel check immediately after Google resolves the destination.
                too_far, travel_minutes = _voice_residential_address_too_far(conv2)
                if too_far:
                    reply_far = _voice_out_of_area_reply(travel_minutes)
                    out2["reply_to_customer"] = reply_far
                    out2["booking_created"] = False
                    try:
                        log_event("VOICE_OUT_OF_AREA_GOOGLE_V38", p, {"travel_minutes": travel_minutes, "call_sid": call_sid, "reply": _safe_monitor_text(reply_far)}, conv2)
                    except Exception:
                        pass
                    return out2
                reply2 = _voice_naturalize_reply(str(out2.get("reply_to_customer") or ""))
                out2["reply_to_customer"] = reply2
            except Exception as e:
                try:
                    log_event("VOICE_V38_POST_RESOLVE_ERROR", p, {"error": repr(e), "call_sid": call_sid}, conv)
                except Exception:
                    pass
            return out2

        # If Google could not safely resolve it, keep the normal prompt.
        if reason and reason not in {"not_needed", "no_google_match"}:
            try:
                log_event("VOICE_GOOGLE_ADDRESS_NOT_RESOLVED_V38", p, {"reason": reason, "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
            except Exception:
                pass
        out["reply_to_customer"] = reply
    except Exception as e:
        try:
            log_event("VOICE_V38_MAPS_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# =============================
# v39 VOICE HOTFIX LAYER
# =============================
# Targets the live failure after v38:
# - Google Maps normalization may fail if ASR hears a street name wrong (ex: Dickerman -> Tickerman).
# - If the caller already gave town/state, do not ask "What town is that in?" again.
# - When town/state is known but Google cannot verify the address, ask for the street name again instead of looping.
# - If the heard street is a close match to the configured dispatch origin address, correct it safely.
# - Make the processing cue more audible, warmer, and calmer.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V39 = process_prevolt_voice_turn
_ORIG_VOICE_THINKING_CLICK_PAYLOAD_V39 = _voice_thinking_click_payload


def _v39_norm_word(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(value or "").lower()).strip()


def _v39_state_from_text(text: str) -> str | None:
    low = _v39_norm_word(text)
    if re.search(r"\b(ct|connecticut)\b", low):
        return "CT"
    if re.search(r"\b(ma|mass|massachusetts)\b", low):
        return "MA"
    return None


def _v39_extract_town_from_text(text: str) -> str | None:
    raw = str(text or "")
    # Prefer common explicit forms.
    m = re.search(r"\b(?:in|at|from)\s+([A-Za-z][A-Za-z .'-]{2,40})\s*,?\s*(?:CT|Connecticut|MA|Massachusetts|Mass)\b", raw, flags=re.I)
    if m:
        town = re.sub(r"\b(?:connecticut|massachusetts|mass|ct|ma)\b", "", m.group(1), flags=re.I).strip(" ,.")
        if town:
            return " ".join(w.capitalize() for w in town.split())
    # Fallback: town immediately before state.
    m = re.search(r"\b([A-Za-z][A-Za-z .'-]{2,40})\s*,?\s*(?:CT|Connecticut|MA|Massachusetts|Mass)\b", raw, flags=re.I)
    if m:
        town = m.group(1).strip(" ,.")
        # Avoid swallowing the whole street line as a town; use text after the last comma if available.
        if "," in town:
            town = town.split(",")[-1].strip()
        town = re.sub(r"^.*?\b(?:ave|avenue|st|street|rd|road|dr|drive|ln|lane|ct|court)\b\s+", "", town, flags=re.I).strip(" ,.")
        if town:
            return " ".join(w.capitalize() for w in town.split())
    return None


def _v39_line_parts(value: str) -> tuple[str | None, str | None, str | None]:
    """Return (house_number, street_name_core, street_type) for a line like '45 Tickerman Ave'."""
    s = _v38_clean_addr_text(value)
    m = re.search(r"\b(\d{1,6})\s+([A-Za-z0-9 .'-]+?)\s*(\b(?:st|street|ave|avenue|rd|road|dr|drive|ln|lane|ct|court|blvd|boulevard|way|pkwy|parkway|ter|terrace|cir|circle)\b)?(?:,|$)", s, flags=re.I)
    if not m:
        return None, None, None
    num = m.group(1)
    street = re.sub(r"\b(?:st|street|ave|avenue|rd|road|dr|drive|ln|lane|ct|court|blvd|boulevard|way|pkwy|parkway|ter|terrace|cir|circle)\b", "", m.group(2) or "", flags=re.I)
    street = _v39_norm_word(street)
    typ = (m.group(3) or "").strip()
    return num, street, typ


def _v39_parse_origin_struct() -> dict | None:
    origin = _v38_origin_address()
    if not origin:
        return None
    # Try the real normalizer first when available.
    try:
        got = _v38_normalize_attempt(origin, None)
        if got and _v38_complete_addr_struct(got):
            return got
    except Exception:
        pass
    # Parse the configured origin env value as a fallback.
    m = re.search(r"^\s*(\d{1,6}\s+[^,]+)\s*,\s*([^,]+)\s*,\s*(CT|MA|Connecticut|Massachusetts)\s*(\d{5})?", origin, flags=re.I)
    if not m:
        return None
    st = m.group(3).upper()
    if st.startswith("CONNECTICUT"):
        st = "CT"
    if st.startswith("MASS"):
        st = "MA"
    return {
        "address_line_1": m.group(1).strip(),
        "locality": m.group(2).strip(),
        "administrative_district_level_1": st,
        "postal_code": (m.group(4) or "").strip(),
        "country": "US",
    }


def _v39_town_state_known(text: str, sched: dict | None = None) -> tuple[str | None, str | None]:
    text = str(text or "")
    town = _v39_extract_town_from_text(text)
    state = _v39_state_from_text(text)
    sched = sched or {}
    if not town:
        town = sched.get("voice_town_hint") or None
    if not state:
        state = sched.get("voice_state_hint") or None
    return town, state


def _v39_candidate_with_town_state(raw: str, caller_text: str, sched: dict) -> str:
    base = _v38_clean_addr_text(raw)
    town, state = _v39_town_state_known(" ".join([str(raw or ""), str(caller_text or "")]), sched)
    if town and state and town.lower() not in base.lower():
        return f"{base}, {town}, {state}"
    if state and re.search(r"\b(?:ct|ma|connecticut|massachusetts|mass)\b", base, flags=re.I) is None:
        return f"{base}, {state}"
    return base


def _v39_try_origin_fuzzy_correction(raw: str, caller_text: str, sched: dict) -> dict | None:
    """Correct obvious ASR street-name errors only against the configured dispatch origin.

    This is intentionally narrow: house number must match, town/state must match when known,
    and the street name must be very similar. It is for cases like Dickerman heard as Tickerman.
    """
    try:
        from difflib import SequenceMatcher
        origin = _v39_parse_origin_struct()
        if not origin:
            return None
        origin_line = origin.get("address_line_1") or ""
        raw_full = _v39_candidate_with_town_state(raw, caller_text, sched)
        raw_num, raw_street, raw_type = _v39_line_parts(raw_full)
        origin_num, origin_street, origin_type = _v39_line_parts(origin_line)
        if not (raw_num and origin_num and raw_street and origin_street):
            return None
        if raw_num != origin_num:
            return None
        town, state = _v39_town_state_known(raw_full + " " + str(caller_text or ""), sched)
        origin_town = _v39_norm_word(origin.get("locality") or "")
        origin_state = (origin.get("administrative_district_level_1") or "").upper()
        # Do not fuzzy-correct to the origin unless the caller gave a matching town/state context.
        if town and _v39_norm_word(town) != origin_town:
            return None
        if state and state.upper() != origin_state:
            return None
        if not town and not state:
            return None
        ratio = SequenceMatcher(None, raw_street, origin_street).ratio()
        if ratio < 0.82:
            return None
        fixed = dict(origin)
        fixed.setdefault("country", "US")
        return fixed
    except Exception:
        return None


def _v39_apply_street_only_repair_if_needed(conv: dict, caller_text: str) -> bool:
    """If we asked the caller to repeat the street name, combine it with saved house/town."""
    sched = conv.setdefault("sched", {})
    raw = str(sched.get("raw_address") or sched.get("address_candidate") or "")
    if not raw or not re.search(r"\b\d{1,6}\b", raw):
        return False
    txt = str(caller_text or "").strip()
    if not txt or re.search(r"\b\d{1,6}\b", txt):
        return False
    # Avoid treating normal answers as street corrections.
    if len(txt.split()) > 4 or re.search(r"@|yes|no|okay|ok|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday", txt, flags=re.I):
        return False
    raw_num, raw_street, raw_type = _v39_line_parts(raw)
    if not raw_num:
        return False
    town, state = _v39_town_state_known(raw, sched)
    if not town or not state:
        return False
    street_type = raw_type or "Ave"
    cleaned_street = re.sub(r"[^A-Za-z0-9 .'-]", " ", txt).strip()
    if not cleaned_street:
        return False
    new_raw = f"{raw_num} {cleaned_street} {street_type}, {town}, {state}"
    sched["raw_address"] = new_raw
    sched["address_candidate"] = new_raw
    sched["address_missing"] = "confirm"
    return True


def _v39_resolve_robust_address(conv: dict, caller_text: str, reply: str | None = None) -> tuple[bool, dict | None, str]:
    sched = conv.setdefault("sched", {})
    if sched.get("address_verified"):
        return False, None, "already_verified"
    try:
        _v39_apply_street_only_repair_if_needed(conv, caller_text)
    except Exception:
        pass
    raw = _v38_clean_addr_text(sched.get("raw_address") or sched.get("address_candidate") or "")
    if not raw or not _address_has_house_number_and_street(raw):
        return False, None, "not_house_street"

    # First try the v38 Google pipeline on the saved raw address.
    addr, reason = _v38_google_resolve_partial_address(raw)
    if addr:
        _v38_apply_normalized_address(conv, addr, "google_maps_v39_raw")
        return True, addr, "google_maps_raw"

    # Then try combining the saved street with town/state heard anywhere in the call turn.
    combined = _v39_candidate_with_town_state(raw, caller_text, sched)
    if combined != raw:
        addr, reason2 = _v38_google_resolve_partial_address(combined)
        if addr:
            _v38_apply_normalized_address(conv, addr, "google_maps_v39_combined")
            return True, addr, "google_maps_combined"
        # Also try the normalizer directly, since the combined string may already be complete.
        st = _v39_state_from_text(combined)
        direct = _v38_normalize_attempt(combined, st) if st else _v38_normalize_attempt(combined, None)
        if direct and _v38_complete_addr_struct(direct):
            _v38_apply_normalized_address(conv, direct, "google_maps_v39_direct_combined")
            return True, direct, "google_maps_direct_combined"

    # Last safe fallback: a very narrow ASR correction against the configured dispatch origin.
    fixed = _v39_try_origin_fuzzy_correction(raw, caller_text, sched)
    if fixed:
        _v38_apply_normalized_address(conv, fixed, "dispatch_origin_fuzzy_v39")
        return True, fixed, "dispatch_origin_fuzzy"

    return False, None, reason or "unresolved"


def _v39_address_already_has_town_state(conv: dict, caller_text: str = "") -> bool:
    sched = conv.setdefault("sched", {})
    raw = str(sched.get("raw_address") or sched.get("address_candidate") or "")
    town, state = _v39_town_state_known(raw + " " + str(caller_text or ""), sched)
    return bool(town and state)


def _v39_unverified_address_reply(conv: dict, caller_text: str) -> str:
    sched = conv.setdefault("sched", {})
    raw = _v39_candidate_with_town_state(str(sched.get("raw_address") or sched.get("address_candidate") or ""), caller_text, sched)
    raw = _v38_clean_addr_text(raw)
    if raw:
        return "I couldn't verify that address. Please say the full address, including the town and state."
    return "I couldn't verify that address. Can you repeat the house number and street name?"


def _v39_clean_reply_more(reply: str) -> str:
    r = str(reply or "")
    patterns = [
        r"\bOkay,?\s+thanks for sharing that\.\s*",
        r"\bLet me get this set up for you\.\s*",
        r"\bGot it\.\s+Thanks for that\.\s*",
        r"\bLet me get the next scheduling detail from you\.\s*",
        r"\bLet me think through[^.?!]*[.?!]\s*",
    ]
    for pat in patterns:
        r = re.sub(pat, "", r, flags=re.I)
    r = re.sub(r"\s+", " ", r).strip()
    return r


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V39(phone, call_sid, caller_text)
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        reply = _v39_clean_reply_more(_voice_naturalize_reply(str(out.get("reply_to_customer") or "")))

        # Try robust Google/fuzzy address resolution any time the backend is still waiting on address.
        if not sched.get("address_verified") and (sched.get("pending_step") == "need_address" or sched.get("state") == "waiting_for_address" or sched.get("address_missing")):
            ok, addr, reason = _v39_resolve_robust_address(conv, caller_text, reply)
            if ok and addr:
                try:
                    log_event("VOICE_GOOGLE_ADDRESS_RESOLVED_V39", p, {
                        "reason": reason,
                        "raw_address": _safe_monitor_text(sched.get("raw_address") or ""),
                        "resolved": f"{addr.get('address_line_1')}, {addr.get('locality')}, {addr.get('administrative_district_level_1')} {addr.get('postal_code')}",
                        "call_sid": call_sid,
                    }, conv)
                except Exception:
                    pass

                # Out-of-area gate immediately after address resolution.
                too_far, travel_minutes = _voice_residential_address_too_far(conv)
                if too_far:
                    out["reply_to_customer"] = _voice_out_of_area_reply(travel_minutes)
                    out["booking_created"] = False
                    out["end_call"] = True
                    return out

                # Feed city/state through the stable original flow so it advances normally.
                synthetic = f"{addr.get('locality')}, {addr.get('administrative_district_level_1')}"
                out2 = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V39(phone, call_sid, synthetic)
                try:
                    conv2 = hydrate_voice_conversation(p, call_sid)
                    conv2["phone"] = p
                    _v38_apply_normalized_address(conv2, addr, "google_maps_v39_final")
                    reply2 = _v39_clean_reply_more(_voice_naturalize_reply(str(out2.get("reply_to_customer") or "")))
                    # If the stable flow still asks town, override with the next deterministic prompt.
                    if _v37_is_town_question(reply2):
                        try:
                            recompute_pending_step(conv2.setdefault("profile", {}), conv2.setdefault("sched", {}))
                            reply2 = choose_next_prompt_from_state(conv2, inbound_text=synthetic)
                            reply2 = _v39_clean_reply_more(_voice_naturalize_reply(reply2))
                        except Exception:
                            reply2 = "We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step. Does that work for you?"
                    out2["reply_to_customer"] = reply2
                    out2["end_call"] = False if not out2.get("booking_created") else out2.get("end_call")
                    conv2["last_voice_reply"] = reply2
                except Exception as e:
                    try:
                        log_event("VOICE_V39_POST_RESOLVE_ERROR", p, {"error": repr(e), "call_sid": call_sid}, conv)
                    except Exception:
                        pass
                return out2

            # If town/state is already present but maps could not verify, do not ask town again.
            if _v39_address_already_has_town_state(conv, caller_text) and (_v37_is_town_question(reply) or "what town" in _intent_text(reply)):
                reply = _v39_unverified_address_reply(conv, caller_text)
                out["reply_to_customer"] = reply
                out["end_call"] = False
                conv["last_voice_reply"] = reply
                try:
                    log_event("VOICE_ADDRESS_UNVERIFIED_NO_TOWN_LOOP_V39", p, {"reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return out

        # Keep the v33 actionable-reply guard after our cleanup.
        try:
            reply = _v33_fix_generic_reply(conv, reply)
        except Exception:
            pass
        out["reply_to_customer"] = reply
        conv["last_voice_reply"] = reply
        if not sched.get("booking_created") and not out.get("booking_created") and _v33_has_customer_facing_question(reply):
            out["end_call"] = False
    except Exception as e:
        try:
            log_event("VOICE_V39_MAPS_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# v39: warmer/more audible thinking cue. Still calm, but less likely to disappear on phone audio.
def _voice_thinking_click_payload(duration_ms: int | None = None) -> str:
    try:
        requested = int(duration_ms or VOICE_AGENT_THINKING_SOUND_MS or 1800)
    except Exception:
        requested = 1800
    ms = max(1300, min(2600, requested))
    rate = 8000
    total = int(rate * (ms / 1000.0))
    pcm = [0] * total

    # Calm overlapping boop/flutter phrase, lower and warmer than earlier versions.
    events = [
        (30, 164, 600, 0.085),
        (250, 208, 720, 0.095),
        (520, 188, 740, 0.082),
        (830, 238, 760, 0.068),
        (1120, 196, 650, 0.048),
    ]
    for n, (start_ms, freq, length_ms, gain) in enumerate(events):
        start = int(rate * start_ms / 1000.0)
        length = min(int(rate * length_ms / 1000.0), max(0, total - start))
        phase = n * 0.73
        for i in range(length):
            idx = start + i
            t = i / rate
            x = i / max(1, length - 1)
            attack = min(1.0, i / max(1, int(rate * 0.075)))
            release = (1.0 - x) ** 2.2
            global_fade = max(0.0, 1.0 - (idx / max(1, total)) ** 1.65)
            flutter = 9.0 * math.sin(2 * math.pi * 3.4 * t + phase)
            tone = math.sin(2 * math.pi * (freq + flutter) * t + phase)
            tone += 0.10 * math.sin(2 * math.pi * (freq * 1.52) * t + phase + 1.3)
            sample = int(16000 * gain * attack * release * global_fade * tone)
            pcm[idx] = max(-14500, min(14500, pcm[idx] + sample))
            for delay_ms, eg in ((92, 0.30), (188, 0.18), (320, 0.095), (510, 0.055), (720, 0.028)):
                eidx = idx + int(rate * delay_ms / 1000.0)
                if eidx < total:
                    pcm[eidx] = max(-14500, min(14500, pcm[eidx] + int(sample * eg)))

    # Soft pad to keep it present through phone compression without sounding like static.
    for i in range(total):
        t = i / rate
        fade = max(0.0, 1.0 - (i / max(1, total)) ** 1.25)
        pad = int(430 * fade * (math.sin(2 * math.pi * 118 * t) + 0.34 * math.sin(2 * math.pi * 176 * t)))
        pcm[i] = max(-14500, min(14500, pcm[i] + pad))

    return base64.b64encode(bytes(_linear16_to_mulaw(s) for s in pcm)).decode("ascii")


def _v33_start_thinking_loop(twilio_ws, stream_sid: str, phone: str, call_sid: str, stop_evt) -> None:
    if not (VOICE_AGENT_THINKING_SOUND_ENABLED and twilio_ws is not None and stream_sid):
        return
    next_send = 0.0
    while not stop_evt.is_set():
        now = time.time()
        if now >= next_send:
            try:
                _send_twilio_media(twilio_ws, stream_sid, _voice_thinking_click_payload(1800))
                try:
                    log_event("VOICE_THINKING_SOUND_SENT", phone, {"reason": "processing_loop_v39", "call_sid": call_sid, "stream_sid": stream_sid})
                except Exception:
                    pass
            except Exception:
                return
            # overlap slightly with reverb so there is not a dead gap on long tool calls
            next_send = now + 1.05
        stop_evt.wait(0.06)


# =============================
# v40 VOICE HOTFIX LAYER
# =============================
# Targets live failure after v39:
# - If the caller gives the address inside the first description ("I live at 45 Dickerman Ave"),
#   do not ask for house number/street again.
# - Use Google Maps / origin-safe fuzzy resolution before falling back to town/state prompts.
# - If Maps cannot verify an address that already has town/state, ask for street spelling, not town again.
# - Make the thinking cue more audible while keeping it calm.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V40 = process_prevolt_voice_turn
_ORIG_VOICE_THINKING_CLICK_PAYLOAD_V40 = _voice_thinking_click_payload


def _v40_clean_embedded_address_text(value: str) -> str:
    s = " ".join(str(value or "").replace("\n", " ").split())
    s = re.sub(r"\b(?:and\s+)?(?:i|we)\s+(?:need|needed|was|were|am|are|would|want|wanted|looking|hoping)\b.*$", "", s, flags=re.I).strip(" ,.;")
    s = re.sub(r"\b(?:for|to)\s+(?:an?|the)?\s*(?:ev|outlet|light|panel|charger|install|repair|replacement)\b.*$", "", s, flags=re.I).strip(" ,.;")
    return s.strip(" ,.;")


def _v40_extract_embedded_address(text: str) -> str | None:
    raw = " ".join(str(text or "").replace("\n", " ").split())
    if not raw:
        return None
    # Prefer the existing address extractor, but do not trust it if it includes obvious service prose.
    try:
        got = extract_service_address_from_text(raw)
        got = _v40_clean_embedded_address_text(got or "")
        if got and _address_has_house_number_and_street(got):
            return got
    except Exception:
        pass
    suffix = r"st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace|pl|place"
    state = r"CT|C\.?T\.?|Connecticut|MA|M\.?A\.?|Massachusetts|Mass"
    # Capture explicit "live at / located at / address is" patterns first.
    m = re.search(
        rf"\b(?:live\s+at|located\s+at|address\s+is|at|for)\s+(?P<addr>\d{{1,6}}\s+[A-Za-z0-9.'#\- ]+?\b(?:{suffix})\b(?:\s*,?\s*[A-Za-z.'\- ]{{2,40}})?(?:\s*,?\s*(?:{state}))?(?:\s+\d{{5}}(?:-\d{{4}})?)?)",
        raw,
        flags=re.I,
    )
    if not m:
        m = re.search(
            rf"\b(?P<addr>\d{{1,6}}\s+[A-Za-z0-9.'#\- ]+?\b(?:{suffix})\b(?:\s*,?\s*[A-Za-z.'\- ]{{2,40}})?(?:\s*,?\s*(?:{state}))?(?:\s+\d{{5}}(?:-\d{{4}})?)?)",
            raw,
            flags=re.I,
        )
    if not m:
        return None
    addr = _v40_clean_embedded_address_text(m.group("addr") or "")
    return addr if addr and _address_has_house_number_and_street(addr) else None


def _v40_try_origin_fuzzy_any_context(raw: str, caller_text: str, sched: dict) -> dict | None:
    """Safely correct only a close match to the configured dispatch origin.

    Unlike v39, this allows the origin correction even when the caller did not say the town/state,
    because a partial address like "45 Dickerman Ave" should resolve against the configured origin
    when house number and street name are a very close match. It still refuses if the caller supplied
    a conflicting town/state.
    """
    try:
        from difflib import SequenceMatcher
        origin = _v39_parse_origin_struct() if "_v39_parse_origin_struct" in globals() else None
        if not origin:
            return None
        origin_line = origin.get("address_line_1") or ""
        raw_num, raw_street, _raw_type = _v39_line_parts(raw)
        origin_num, origin_street, _origin_type = _v39_line_parts(origin_line)
        if not (raw_num and raw_street and origin_num and origin_street):
            return None
        if raw_num != origin_num:
            return None
        town, state = _v39_town_state_known(str(raw or "") + " " + str(caller_text or ""), sched)
        origin_town = _v39_norm_word(origin.get("locality") or "")
        origin_state = (origin.get("administrative_district_level_1") or "").upper()
        if town and _v39_norm_word(town) != origin_town:
            return None
        if state and state.upper() != origin_state:
            return None
        ratio = SequenceMatcher(None, raw_street, origin_street).ratio()
        if ratio < 0.82:
            return None
        fixed = dict(origin)
        fixed.setdefault("country", "US")
        return fixed
    except Exception:
        return None


def _v40_resolve_address_from_embedded_or_sched(conv: dict, caller_text: str) -> tuple[bool, dict | None, str]:
    sched = conv.setdefault("sched", {})
    if sched.get("address_verified"):
        return False, None, "already_verified"

    embedded = _v40_extract_embedded_address(caller_text)
    if embedded:
        try:
            set_raw_address_safe(sched, embedded)
        except Exception:
            sched["raw_address"] = embedded
            sched["address_candidate"] = embedded
        sched["address_missing"] = "confirm"

    raw = _v38_clean_addr_text(sched.get("raw_address") or sched.get("address_candidate") or embedded or "")
    if not raw or not _address_has_house_number_and_street(raw):
        return False, None, "no_street_address"

    # Google Maps first. This is the normal path for addresses like "45 Dickerman Ave".
    try:
        addr, reason = _v38_google_resolve_partial_address(raw)
    except Exception as e:
        addr, reason = None, "google_exception:" + repr(e)
    if addr:
        _v38_apply_normalized_address(conv, addr, "google_maps_v40_embedded")
        return True, addr, "google_maps_raw"

    # Then try with any town/state heard in the same caller turn.
    try:
        combined = _v39_candidate_with_town_state(raw, caller_text, sched)
    except Exception:
        combined = raw
    if combined and combined != raw:
        try:
            addr, reason2 = _v38_google_resolve_partial_address(combined)
        except Exception as e:
            addr, reason2 = None, "google_combined_exception:" + repr(e)
        if addr:
            _v38_apply_normalized_address(conv, addr, "google_maps_v40_combined")
            return True, addr, "google_maps_combined"
        try:
            forced_state = _v39_state_from_text(combined)
            direct = _v38_normalize_attempt(combined, forced_state) if forced_state else _v38_normalize_attempt(combined, None)
            if direct and _v38_complete_addr_struct(direct):
                _v38_apply_normalized_address(conv, direct, "google_maps_v40_direct_combined")
                return True, direct, "google_maps_direct_combined"
        except Exception:
            pass

    # Very narrow dispatch-origin correction for known own-address tests and future same-location calls.
    fixed = _v40_try_origin_fuzzy_any_context(raw, caller_text, sched)
    if fixed:
        _v38_apply_normalized_address(conv, fixed, "dispatch_origin_fuzzy_v40")
        return True, fixed, "dispatch_origin_fuzzy_any_context"

    return False, None, reason or "unresolved"


def _v40_eval_price_consent_reply() -> str:
    return (
        "We start with a $195 on-site evaluation visit. "
        "That covers sending one of our electricians out, reviewing the work in person, "
        "checking what is needed, and putting together the next step. Does that work for you?"
    )


def _v40_after_verified_address_reply(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195").upper()
    if "EMER" in appt or "TROUBLE" in appt or "REPAIR_395" in appt:
        try:
            return _voice_emergency_dispatch_offer(conv)
        except Exception:
            return "This looks urgent. We can send someone now, and arrival is usually within one to two hours. The emergency troubleshoot and repair visit is $395. Do you want us to dispatch someone now?"
    sched["appointment_type"] = "EVAL_195"
    conv["appointment_type"] = "EVAL_195"
    sched["awaiting_eval_price_confirm"] = True
    sched["pending_step"] = "need_date"
    sched["state"] = "waiting_for_date"
    sched["awaiting_slot_offer_choice"] = False
    return _v40_eval_price_consent_reply()


def _v40_reply_needs_address_override(reply: str, sched: dict) -> bool:
    low = _intent_text(reply or "")
    if "house number and street" in low or "street name" in low or "which town" in low or "what town" in low:
        return True
    if str(sched.get("pending_step") or "").lower() == "need_address" or str(sched.get("state") or "").lower() == "waiting_for_address":
        return True
    return False


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V40(phone, call_sid, caller_text)
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        profile = conv.setdefault("profile", {})
        try:
            if "_v32_apply_name_repairs" in globals():
                _v32_apply_name_repairs(conv, caller_text)
        except Exception:
            pass

        reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or ""))
        embedded = _v40_extract_embedded_address(caller_text)

        # If the caller already gave an address in the same utterance, do not ask for it again.
        if embedded or (_v40_reply_needs_address_override(reply, sched) and (sched.get("raw_address") or sched.get("address_candidate"))):
            ok, addr, reason = _v40_resolve_address_from_embedded_or_sched(conv, caller_text)
            if ok and addr:
                try:
                    log_event("VOICE_GOOGLE_ADDRESS_RESOLVED_V40", p, {
                        "reason": reason,
                        "embedded": _safe_monitor_text(embedded or ""),
                        "raw_address": _safe_monitor_text(sched.get("raw_address") or ""),
                        "call_sid": call_sid,
                    }, conv)
                except Exception:
                    pass
                too_far, travel_minutes = _voice_residential_address_too_far(conv)
                if too_far:
                    reply2 = _voice_out_of_area_reply(travel_minutes)
                    out["reply_to_customer"] = reply2
                    out["booking_created"] = False
                    out["end_call"] = True
                    conv["last_voice_reply"] = reply2
                    return out
                reply2 = _v40_after_verified_address_reply(conv)
                out["reply_to_customer"] = reply2
                out["booking_created"] = False
                out["manual_only"] = False
                out["pending_step"] = sched.get("pending_step")
                out["appointment_type"] = sched.get("appointment_type") or conv.get("appointment_type")
                out["end_call"] = False
                conv["last_voice_reply"] = reply2
                return out

            # Maps/fuzzy could not verify it. If town/state are already present, do NOT ask town again.
            if _v39_address_already_has_town_state(conv, caller_text):
                reply2 = _v39_unverified_address_reply(conv, caller_text)
                out["reply_to_customer"] = reply2
                out["end_call"] = False
                conv["last_voice_reply"] = reply2
                return out

        # Final guard: if caller_text contained an address and the output still asks for address, ask a better missing piece.
        if embedded and _v40_reply_needs_address_override(reply, sched):
            reply = "I couldn't verify that address yet. Which town is it in, and is that Connecticut or Massachusetts?"
        else:
            try:
                reply = _v33_fix_generic_reply(conv, reply)
            except Exception:
                pass
        out["reply_to_customer"] = _v39_clean_reply_more(reply) if "_v39_clean_reply_more" in globals() else reply
        conv["last_voice_reply"] = out["reply_to_customer"]
        if not sched.get("booking_created") and not out.get("booking_created"):
            out["end_call"] = False
    except Exception as e:
        try:
            log_event("VOICE_V40_ADDRESS_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# v40: warmer but more audible processing cue through phone compression.
def _voice_thinking_click_payload(duration_ms: int | None = None) -> str:
    try:
        requested = int(duration_ms or VOICE_AGENT_THINKING_SOUND_MS or 1900)
    except Exception:
        requested = 1900
    ms = max(1200, min(2600, requested))
    rate = 8000
    total = int(rate * (ms / 1000.0))
    pcm = [0] * total
    events = [
        (20, 176, 560, 0.115),
        (185, 224, 620, 0.120),
        (360, 196, 690, 0.100),
        (555, 262, 760, 0.082),
        (790, 218, 840, 0.065),
    ]
    for n, (start_ms, freq, length_ms, gain) in enumerate(events):
        start = int(rate * start_ms / 1000.0)
        length = min(int(rate * length_ms / 1000.0), max(0, total - start))
        if length <= 0:
            continue
        phase = n * 0.73
        for i in range(length):
            idx = start + i
            t = i / rate
            x = i / max(1, length - 1)
            attack = min(1.0, i / max(1, int(rate * 0.07)))
            release = (1.0 - x) ** 1.75
            global_fade = max(0.0, 1.0 - (idx / max(1, total)) ** 1.55)
            vibrato = 9 * math.sin(2 * math.pi * 3.3 * t + phase)
            tone = math.sin(2 * math.pi * (freq + vibrato) * t + phase)
            tone += 0.18 * math.sin(2 * math.pi * (freq * 1.5) * t + 1.4)
            tone += 0.08 * math.sin(2 * math.pi * (freq * 2.0) * t + 0.6)
            sample = int(15500 * gain * attack * release * global_fade * tone)
            pcm[idx] = max(-15000, min(15000, pcm[idx] + sample))
            for delay_ms, echo_gain in ((70, 0.34), (145, 0.20), (245, 0.11), (375, 0.055)):
                eidx = idx + int(rate * delay_ms / 1000.0)
                if eidx < total:
                    pcm[eidx] = max(-15000, min(15000, pcm[eidx] + int(sample * echo_gain)))
    for i in range(total):
        t = i / rate
        fade = max(0.0, 1.0 - (i / max(1, total)) ** 1.25)
        pad = int(420 * fade * (math.sin(2 * math.pi * 132 * t) + 0.30 * math.sin(2 * math.pi * 198 * t)))
        pcm[i] = max(-15000, min(15000, pcm[i] + pad))
    fade_samples = int(rate * 0.06)
    for i in range(min(fade_samples, total)):
        mult = i / max(1, fade_samples)
        pcm[i] = int(pcm[i] * mult)
        j = total - 1 - i
        pcm[j] = int(pcm[j] * mult)
    data = bytearray(_linear16_to_mulaw(sample) for sample in pcm)
    return base64.b64encode(bytes(data)).decode("ascii")



# =============================
# v41 VOICE HOTFIX LAYER
# =============================
# Targets live failure after v40:
# - Address verification can take several seconds; after an embedded address is verified,
#   acknowledge it so the caller understands what just happened.
# - After the caller says yes to the $195 evaluation visit, NEVER repeat the $195 consent prompt.
#   Offer the concrete slots already stored in offered_slot_options.
# - Preserve the stable v36-v40 name/email/address corrections.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V41 = process_prevolt_voice_turn


def _v41_yes_no(text: str) -> str | None:
    try:
        return _voice_yes_no_text(text)
    except Exception:
        low = _intent_text(text or "")
        if re.search(r"\b(yes|yeah|yep|ok|okay|sure|that works|sounds good|fine|lets do it|let's do it)\b", low):
            return "yes"
        if re.search(r"\b(no|nope|nah|not really|too much|free quote|do not want|don't want)\b", low):
            return "no"
    return None


def _v41_reply_is_eval_price_prompt(reply: str) -> bool:
    low = _intent_text(reply or "")
    return bool(
        ("195" in low or "evaluation visit" in low or "onsite evaluation" in low or "on site evaluation" in low)
        and ("does that work" in low or "interested" in low or "evaluation visit is" in low)
    )


def _v41_slot_options_reply(sched: dict) -> str:
    opts = (sched.get("offered_slot_options") or [])[:3]
    if opts:
        try:
            return f"Great. We have {_format_slot_options(opts)}. Which one works best?"
        except Exception:
            labels = []
            for opt in opts:
                if isinstance(opt, dict):
                    labels.append(str(opt.get("label") or "").strip())
            labels = [x for x in labels if x]
            if labels:
                if len(labels) == 1:
                    return f"Great. We have {labels[0]}. Does that time work?"
                if len(labels) == 2:
                    return f"Great. We have {labels[0]} or {labels[1]}. Which one works best?"
                return f"Great. We have {labels[0]}, {labels[1]}, or {labels[2]}. Which one works best?"
    pending = (sched.get("voice_pending_slot_offer_reply") or "").strip()
    if pending:
        pending = _voice_naturalize_reply(pending)
        if "which one" not in _intent_text(pending):
            pending = pending.rstrip(" .") + ". Which one works best?"
        return pending
    return "Great. What day and time works best for you?"


def _v41_embedded_address_in_text(caller_text: str) -> bool:
    try:
        return bool(_v40_extract_embedded_address(caller_text))
    except Exception:
        pass
    low = str(caller_text or "")
    return bool(re.search(r"\b\d{1,6}\s+[A-Za-z0-9.'#\- ]+\b(?:st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace|pl|place)\b", low, flags=re.I))


def _v41_add_address_verified_notice_if_needed(conv: dict, caller_text: str, reply: str) -> str:
    sched = conv.setdefault("sched", {})
    if not reply or not _v41_reply_is_eval_price_prompt(reply):
        return reply
    if not sched.get("address_verified"):
        return reply
    if sched.get("voice_address_verified_notice_sent"):
        return reply
    # Only add this when the caller gave the address in the same long opening/turn.
    # This makes a long Google Maps verification delay feel intentional without
    # adding filler to every normal scheduling step.
    if not _v41_embedded_address_in_text(caller_text):
        return reply
    sched["voice_address_verified_notice_sent"] = True
    cleaned = _v39_clean_reply_more(reply) if "_v39_clean_reply_more" in globals() else reply
    return "I verified the address in our system. " + cleaned


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V41(phone, call_sid, caller_text)
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})

        reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or ""))

        # If the previous step was the $195 consent gate and the customer says yes,
        # the next spoken thing MUST be concrete availability, not another price prompt.
        yn = _v41_yes_no(caller_text)
        has_slots = bool(sched.get("offered_slot_options") or sched.get("voice_pending_slot_offer_reply"))
        waiting_for_date = (
            str(sched.get("pending_step") or "").lower() == "need_date"
            or str(sched.get("state") or "").lower() == "waiting_for_date"
        )
        if yn == "yes" and waiting_for_date and has_slots:
            # Accept the evaluation price once and advance.
            sched["awaiting_eval_price_confirm"] = False
            sched["awaiting_slot_offer_choice"] = True
            reply2 = _v41_slot_options_reply(sched)
            out["reply_to_customer"] = reply2
            out["booking_created"] = False
            out["end_call"] = False
            out["pending_step"] = sched.get("pending_step") or "need_date"
            conv["last_voice_reply"] = reply2
            try:
                log_event("VOICE_V41_PRICE_ACCEPTED_OFFER_SLOTS", p, {
                    "caller_text": _safe_monitor_text(caller_text),
                    "reply": _safe_monitor_text(reply2),
                    "call_sid": call_sid,
                    "offered_count": len(sched.get("offered_slot_options") or []),
                }, conv)
            except Exception:
                pass
            return out

        # Do not let any postprocessor repeat the $195 prompt when slots are ready.
        if _v41_reply_is_eval_price_prompt(reply) and waiting_for_date and has_slots and not sched.get("awaiting_eval_price_confirm"):
            reply2 = _v41_slot_options_reply(sched)
            out["reply_to_customer"] = reply2
            out["end_call"] = False
            conv["last_voice_reply"] = reply2
            try:
                log_event("VOICE_V41_SUPPRESSED_DUPLICATE_PRICE_PROMPT", p, {
                    "caller_text": _safe_monitor_text(caller_text),
                    "old_reply": _safe_monitor_text(reply),
                    "new_reply": _safe_monitor_text(reply2),
                    "call_sid": call_sid,
                }, conv)
            except Exception:
                pass
            return out

        # Add a useful address-verification acknowledgement after a slow embedded-address lookup.
        reply2 = _v41_add_address_verified_notice_if_needed(conv, caller_text, reply)
        if reply2 != reply:
            out["reply_to_customer"] = reply2
            out["end_call"] = False
            conv["last_voice_reply"] = reply2
            try:
                log_event("VOICE_V41_ADDRESS_VERIFIED_NOTICE", p, {
                    "caller_text": _safe_monitor_text(caller_text),
                    "reply": _safe_monitor_text(reply2),
                    "call_sid": call_sid,
                    "address": _safe_monitor_text(sched.get("raw_address") or ""),
                }, conv)
            except Exception:
                pass
            return out

        out["reply_to_customer"] = _v39_clean_reply_more(reply) if "_v39_clean_reply_more" in globals() else reply
        if not sched.get("booking_created") and not out.get("booking_created"):
            out["end_call"] = False
    except Exception as e:
        try:
            log_event("VOICE_V41_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# =============================
# v42 STABILIZATION HOTFIX LAYER
# =============================
# Targets live v41 failure:
# - Generic "troubleshoot some power issues" was treated as emergency state without a hard hazard.
# - Voice reply asked "What day and time works best?" instead of giving the correct $395 consent gate / options.
# - If the caller complains that the system should offer times, recover by offering the stored/next slots.
# - Keep this narrow: no changes to name, email, Square booking, Maps normalization, or final TwiML goodbye.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V42 = process_prevolt_voice_turn


def _v42_hard_hazard_text(text: str) -> bool:
    """True only for actual safety hazard/emergency language, not generic troubleshooting."""
    low = _intent_text(text or "") if "_intent_text" in globals() else str(text or "").lower()
    hard_patterns = [
        r"\b(smoke|smoking|smells? like smoke|burning smell|burnt smell)\b",
        r"\b(caught fire|on fire|flames?|sparking|sparked|arcing|arc flash)\b",
        r"\b(crackling|popping|sizzling|buzzing loudly)\b",
        r"\b(hot to the touch|panel is hot|breaker is hot|outlet is hot|receptacle is hot)\b",
        r"\b(water in (?:the )?panel|water got into (?:the )?panel|flooded panel|service mast.*(?:hit|ripped|down)|meter.*(?:ripped|pulled|sparking))\b",
        r"\b(lost all power|entire house has no power|main breaker.*(?:tripping|won't reset|will not reset))\b",
    ]
    return any(re.search(p, low, flags=re.I) for p in hard_patterns)


def _v42_generic_troubleshoot_text(text: str) -> bool:
    low = _intent_text(text or "") if "_intent_text" in globals() else str(text or "").lower()
    if _v42_hard_hazard_text(low):
        return False
    return bool(re.search(r"\b(troubleshoot|power issues?|outlets? (?:not working|stopped working)|lights? (?:not working|flicker|flickering)|breaker(?:s)? tripping|circuit(?:s)? (?:not working|dead))\b", low, flags=re.I))


def _v42_availability_request_text(text: str) -> bool:
    low = _intent_text(text or "") if "_intent_text" in globals() else str(text or "").lower()
    return bool(re.search(r"\b(offer|options?|days? and times?|openings?|availability|available times?|supposed to offer|what do you have|next available|earliest|soonest)\b", low, flags=re.I))


def _v42_reply_is_generic_time_ask(reply: str) -> bool:
    low = _intent_text(reply or "") if "_intent_text" in globals() else str(reply or "").lower()
    return bool(
        "what day and time works" in low
        or "what day time works" in low
        or "what time works" in low
        or "what date time" in low
        or low.strip() in {"i can get your appointment scheduled here", "i can get your appointment scheduled here."}
    )


def _v42_slot_offer_reply_for(conv: dict, appt: str) -> str:
    sched = conv.setdefault("sched", {})
    slots = list(sched.get("offered_slot_options") or [])[:3]
    if not slots:
        try:
            slots = get_next_available_slots(appt or sched.get("appointment_type") or "EVAL_195", limit=3) or []
        except Exception as e:
            try:
                log_event("VOICE_V42_SLOT_LOOKUP_ERROR", conv.get("phone") or "", {"error": repr(e), "appointment_type": appt})
            except Exception:
                pass
            slots = []
    if slots:
        sched["offered_slot_options"] = slots[:3]
        sched["awaiting_slot_offer_choice"] = True
        try:
            return f"Great. We have {_format_slot_options(slots[:3])}. Which one works best?"
        except Exception:
            labels = [str(x.get("label") or "").strip() for x in slots[:3] if isinstance(x, dict) and str(x.get("label") or "").strip()]
            if len(labels) >= 3:
                return f"Great. We have {labels[0]}, {labels[1]}, or {labels[2]}. Which one works best?"
            if len(labels) == 2:
                return f"Great. We have {labels[0]} or {labels[1]}. Which one works best?"
            if len(labels) == 1:
                return f"Great. We have {labels[0]}. Does that time work?"
    sched["awaiting_slot_offer_choice"] = False
    return "Great. I am not seeing automatic openings in the scheduler right now. What weekday and time work best for you?"


def _v42_troubleshoot_price_prompt(conv: dict, caller_text: str = "") -> str:
    sched = conv.setdefault("sched", {})
    prefix = ""
    try:
        if sched.get("address_verified") and _v41_embedded_address_in_text(caller_text):
            prefix = "I verified the address in our system. "
    except Exception:
        if sched.get("address_verified"):
            prefix = "I verified the address in our system. "
    return prefix + "Troubleshoot and repair visits are $395. That covers sending one of our electricians out to diagnose the issue and repair it on site when possible. Does that work for you?"


def _v42_convert_generic_troubleshoot_to_normal(conv: dict) -> None:
    sched = conv.setdefault("sched", {})
    # TROUBLESHOOT_395 is allowed to be a normal scheduled repair. Only hard hazard language should stay in emergency dispatch mode.
    sched["appointment_type"] = "TROUBLESHOOT_395"
    conv["appointment_type"] = "TROUBLESHOOT_395"
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    if str(sched.get("state") or "").lower() == "emergency":
        sched["state"] = "waiting_for_date"
    sched["pending_step"] = "need_date"


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    # Pre-intercept: if v42 already asked for the $395 troubleshoot consent, do not let the older layers
    # route "yes" through emergency or generic date logic.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        sched_pre = conv_pre.setdefault("sched", {})
        yn_pre = _v41_yes_no(caller_text) if "_v41_yes_no" in globals() else _voice_yes_no_text(caller_text)
        if sched_pre.get("awaiting_troubleshoot_price_confirm"):
            if yn_pre == "no":
                sched_pre["awaiting_troubleshoot_price_confirm"] = False
                sched_pre["voice_close_after_reply"] = True
                sched_pre["closed_reason"] = "declined_troubleshoot_price"
                reply = "No problem. We will not schedule the visit right now. Thank you for calling Prevolt Electric. Goodbye."
                conv_pre["last_voice_reply"] = reply
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": sched_pre.get("appointment_type") or conv_pre.get("appointment_type"), "end_call": True}
            if yn_pre == "yes" or _v42_availability_request_text(caller_text):
                sched_pre["awaiting_troubleshoot_price_confirm"] = False
                _v42_convert_generic_troubleshoot_to_normal(conv_pre)
                reply = _v42_slot_offer_reply_for(conv_pre, "TROUBLESHOOT_395")
                conv_pre["last_voice_reply"] = reply
                try:
                    log_event("VOICE_V42_TROUBLESHOOT_PRICE_ACCEPTED_OFFER_SLOTS", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "TROUBLESHOOT_395", "end_call": False}
            # Anything other than yes/no: repeat the price consent, not date collection.
            reply = _v42_troubleshoot_price_prompt(conv_pre, caller_text)
            conv_pre["last_voice_reply"] = reply
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "TROUBLESHOOT_395", "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V42_PRE_INTERCEPT_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V42(phone, call_sid, caller_text)

    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or ""))
        appt = str(sched.get("appointment_type") or conv.get("appointment_type") or out.get("appointment_type") or "").upper()
        hard_hazard = _v42_hard_hazard_text(caller_text)
        generic_trouble = _v42_generic_troubleshoot_text(caller_text) or ("TROUBLESHOOT" in appt and not hard_hazard)

        # Stabilize generic troubleshoot calls: not emergency dispatch, not generic date collection.
        if generic_trouble and "TROUBLESHOOT" in appt and not hard_hazard:
            _v42_convert_generic_troubleshoot_to_normal(conv)
            if _v42_reply_is_generic_time_ask(reply) or "what day and time" in _intent_text(reply or "") or not reply:
                sched["awaiting_troubleshoot_price_confirm"] = True
                sched["awaiting_slot_offer_choice"] = False
                reply2 = _v42_troubleshoot_price_prompt(conv, caller_text)
                out["reply_to_customer"] = reply2
                out["booking_created"] = False
                out["end_call"] = False
                out["pending_step"] = sched.get("pending_step")
                out["appointment_type"] = "TROUBLESHOOT_395"
                conv["last_voice_reply"] = reply2
                try:
                    log_event("VOICE_V42_TROUBLESHOOT_PRICE_GATE", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(reply2), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return out

        # Recovery: if caller complains that times should be offered, give concrete slots if it is safe to do so.
        if _v42_availability_request_text(caller_text) and sched.get("address_verified") and str(sched.get("pending_step") or "").lower() == "need_date":
            if "TROUBLESHOOT" in appt and not hard_hazard and not sched.get("awaiting_troubleshoot_price_confirm"):
                reply2 = _v42_slot_offer_reply_for(conv, "TROUBLESHOOT_395")
            elif sched.get("awaiting_eval_price_confirm"):
                reply2 = "The on-site evaluation visit is $195. Does that work for you?"
            else:
                reply2 = _v42_slot_offer_reply_for(conv, sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195")
            out["reply_to_customer"] = reply2
            out["booking_created"] = False
            out["end_call"] = False
            conv["last_voice_reply"] = reply2
            try:
                log_event("VOICE_V42_AVAILABILITY_RECOVERY", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply2), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return out

        out["reply_to_customer"] = _v39_clean_reply_more(reply) if "_v39_clean_reply_more" in globals() else reply
        if not sched.get("booking_created") and not out.get("booking_created"):
            out["end_call"] = False
    except Exception as e:
        try:
            log_event("VOICE_V42_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# v42: stop the processing cue from hammering the stream forever if a backend turn stalls.
def _v33_start_thinking_loop(twilio_ws, stream_sid: str, phone: str, call_sid: str, stop_evt) -> None:
    if not (VOICE_AGENT_THINKING_SOUND_ENABLED and twilio_ws is not None and stream_sid):
        return
    max_sends = 4
    sends = 0
    # Wait briefly before the first cue so normal fast turns do not get noise at all.
    stop_evt.wait(0.45)
    while not stop_evt.is_set() and sends < max_sends:
        try:
            _send_twilio_media(twilio_ws, stream_sid, _voice_thinking_click_payload(1800))
            sends += 1
            try:
                log_event("VOICE_THINKING_SOUND_SENT", phone, {"reason": "processing_loop_v42", "call_sid": call_sid, "stream_sid": stream_sid, "send_count": sends})
            except Exception:
                pass
        except Exception:
            return
        stop_evt.wait(1.35)



# =============================
# v43 STABILIZATION HOTFIX LAYER
# =============================
# Targets live v42 failure:
# - Caller accepted the $395 troubleshoot/repair price, but the voice layer repeated the price
#   instead of offering concrete availability.
# - The pre-intercept can miss the flag if an older layer rewrites state, so v43 treats a clear
#   "yes" while in TROUBLESHOOT_395 + need_date as price acceptance even if the flag is missing.
# - Keep this narrow: no changes to Maps normalization, email confirmation, name parsing, Square booking,
#   or final TwiML goodbye.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V43 = process_prevolt_voice_turn


def _v43_yes(text: str) -> bool:
    try:
        yn = _v41_yes_no(text) if "_v41_yes_no" in globals() else _voice_yes_no_text(text)
        if yn == "yes":
            return True
    except Exception:
        pass
    low = _intent_text(text or "") if "_intent_text" in globals() else str(text or "").lower()
    return bool(re.search(r"\b(yes|yeah|yep|yup|ok|okay|correct|that works|works for me|sounds good|sure|let'?s do it|do it)\b", low, flags=re.I))


def _v43_no(text: str) -> bool:
    try:
        yn = _v41_yes_no(text) if "_v41_yes_no" in globals() else _voice_yes_no_text(text)
        if yn == "no":
            return True
    except Exception:
        pass
    low = _intent_text(text or "") if "_intent_text" in globals() else str(text or "").lower()
    return bool(re.search(r"\b(no|nope|nah|not now|never mind|too much|do not|don'?t)\b", low, flags=re.I))


def _v43_is_troubleshoot_price_prompt(text: str) -> bool:
    low = _intent_text(text or "") if "_intent_text" in globals() else str(text or "").lower()
    return bool(("395" in low or "troubleshoot" in low or "repair visits" in low) and ("does that work" in low or "diagnose" in low or "on site" in low or "onsite" in low))


def _v43_is_need_date(sched: dict) -> bool:
    return str(sched.get("pending_step") or "").lower() == "need_date" or str(sched.get("state") or "").lower() in {"waiting_for_date", "emergency"}


def _v43_appt_is_troubleshoot(conv: dict, out: dict | None = None) -> bool:
    sched = conv.setdefault("sched", {})
    out = out or {}
    appt = str(sched.get("appointment_type") or conv.get("appointment_type") or out.get("appointment_type") or "").upper()
    return "TROUBLESHOOT" in appt


def _v43_offer_slots_for_troubleshoot(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    # Normalize the state so older emergency/date layers stop treating generic troubleshooting as emergency dispatch.
    sched["appointment_type"] = "TROUBLESHOOT_395"
    conv["appointment_type"] = "TROUBLESHOOT_395"
    sched["awaiting_troubleshoot_price_confirm"] = False
    sched["awaiting_eval_price_confirm"] = False
    sched["awaiting_slot_offer_choice"] = True
    sched["pending_step"] = "need_date"
    if str(sched.get("state") or "").lower() == "emergency":
        sched["state"] = "waiting_for_date"
    try:
        return _v42_slot_offer_reply_for(conv, "TROUBLESHOOT_395")
    except Exception:
        try:
            return _v41_slot_options_reply(sched)
        except Exception:
            return "Great. What weekday and time work best for you?"


def _v43_troubleshoot_decline_reply(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    sched["awaiting_troubleshoot_price_confirm"] = False
    sched["closed_reason"] = "declined_troubleshoot_price"
    sched["voice_close_after_reply"] = True
    return "No problem. We will not schedule the visit right now. Thank you for calling Prevolt Electric. Goodbye."


def _v43_troubleshoot_price_prompt(conv: dict, caller_text: str = "") -> str:
    # Avoid the awkward live wording "I verify" by using a passive phrase that TTS reads cleaner.
    sched = conv.setdefault("sched", {})
    prefix = ""
    try:
        if sched.get("address_verified") and (_v41_embedded_address_in_text(caller_text) or sched.get("raw_address") or sched.get("address")):
            prefix = "The address is verified in our system. "
    except Exception:
        if sched.get("address_verified"):
            prefix = "The address is verified in our system. "
    return prefix + "Troubleshoot and repair visits are $395. That covers sending one of our electricians out to diagnose the issue and repair it on site when possible. Does that work for you?"


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()

    # Strong pre-intercept: if the last spoken line was the $395 gate and the caller says yes,
    # do not call older layers. Older layers have repeatedly re-asked the price here.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        sched_pre = conv_pre.setdefault("sched", {})
        last_reply = str(conv_pre.get("last_voice_reply") or sched_pre.get("last_voice_reply") or "")
        awaiting_395 = bool(sched_pre.get("awaiting_troubleshoot_price_confirm")) or _v43_is_troubleshoot_price_prompt(last_reply)
        if awaiting_395 and _v43_appt_is_troubleshoot(conv_pre) and _v43_is_need_date(sched_pre):
            if _v43_no(caller_text):
                reply = _v43_troubleshoot_decline_reply(conv_pre)
                conv_pre["last_voice_reply"] = reply
                try:
                    log_event("VOICE_V43_TROUBLESHOOT_PRICE_DECLINED", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "TROUBLESHOOT_395", "end_call": True}
            if _v43_yes(caller_text) or _v42_availability_request_text(caller_text):
                reply = _v43_offer_slots_for_troubleshoot(conv_pre)
                conv_pre["last_voice_reply"] = reply
                try:
                    log_event("VOICE_V43_TROUBLESHOOT_PRICE_ACCEPTED_OFFER_SLOTS_PRE", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": "need_date", "appointment_type": "TROUBLESHOOT_395", "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V43_PRE_INTERCEPT_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V43(phone, call_sid, caller_text)

    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or ""))
        low_reply = _intent_text(reply or "") if "_intent_text" in globals() else reply.lower()

        # If the caller said yes to a troubleshoot price gate, and any older layer still returns
        # the $395 prompt or generic scheduling filler, force slot offering.
        if _v43_yes(caller_text) and _v43_appt_is_troubleshoot(conv, out) and _v43_is_need_date(sched):
            if _v43_is_troubleshoot_price_prompt(reply) or _v42_reply_is_generic_time_ask(reply) or "let me line up" in low_reply or "next scheduling detail" in low_reply:
                reply2 = _v43_offer_slots_for_troubleshoot(conv)
                out["reply_to_customer"] = reply2
                out["booking_created"] = False
                out["end_call"] = False
                out["pending_step"] = "need_date"
                out["appointment_type"] = "TROUBLESHOOT_395"
                conv["last_voice_reply"] = reply2
                try:
                    log_event("VOICE_V43_SUPPRESSED_DUPLICATE_TROUBLESHOOT_PRICE", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(reply2), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return out

        # If a generic troubleshoot turn starts with the bare scheduling line, force the proper $395 gate.
        hard_hazard = _v42_hard_hazard_text(caller_text) if "_v42_hard_hazard_text" in globals() else False
        generic_trouble = _v42_generic_troubleshoot_text(caller_text) if "_v42_generic_troubleshoot_text" in globals() else False
        if generic_trouble and not hard_hazard and _v43_appt_is_troubleshoot(conv, out) and _v42_reply_is_generic_time_ask(reply):
            sched["awaiting_troubleshoot_price_confirm"] = True
            sched["awaiting_slot_offer_choice"] = False
            sched["pending_step"] = "need_date"
            if str(sched.get("state") or "").lower() == "emergency":
                sched["state"] = "waiting_for_date"
            reply2 = _v43_troubleshoot_price_prompt(conv, caller_text)
            out["reply_to_customer"] = reply2
            out["booking_created"] = False
            out["end_call"] = False
            out["pending_step"] = "need_date"
            out["appointment_type"] = "TROUBLESHOOT_395"
            conv["last_voice_reply"] = reply2
            try:
                log_event("VOICE_V43_TROUBLESHOOT_PRICE_GATE", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(reply2), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return out

        # Cleanup only: remove useless filler that makes a stable path feel broken.
        if reply:
            reply2 = reply
            reply2 = re.sub(r"\bOkay,?\s+let(?:'|’)s get this set up for you and sort out the scheduling details\.\s*", "", reply2, flags=re.I)
            reply2 = re.sub(r"\bGot it\.\s+Thanks\.\s+Let me line up the next scheduling detail with you\.\s*", "", reply2, flags=re.I)
            reply2 = reply2.replace("I verified the address in our system.", "The address is verified in our system.")
            out["reply_to_customer"] = reply2.strip() or reply
            if not sched.get("booking_created") and not out.get("booking_created"):
                out["end_call"] = False
    except Exception as e:
        try:
            log_event("VOICE_V43_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# =============================
# v44 STABILIZATION HOTFIX LAYER
# =============================
# Targets live v43 failure:
# - Generic troubleshoot accepted price -> offered slots -> caller chooses Monday June 1 at 3 PM.
# - Older offered-slot/emergency layers selected the first slot, reopened emergency dispatch language,
#   and polluted customer last name with "Monday".
#
# Keep this narrow: intercept offered-slot choices for scheduled troubleshoot calls before older
# emergency/date/name layers can touch the turn. Also hard-sanitize date words from name fields.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V44 = process_prevolt_voice_turn

_V44_DATE_NAME_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "today", "tomorrow", "tonight", "morning", "afternoon", "evening",
}


def _v44_low(text: str) -> str:
    try:
        return _intent_text(text or "")
    except Exception:
        return str(text or "").lower()


def _v44_is_hard_hazard(text: str) -> bool:
    try:
        return bool(_v42_hard_hazard_text(text))
    except Exception:
        low = _v44_low(text)
        return bool(re.search(r"\b(smoke|smoking|fire|burning|burnt|hot\s+panel|panel\s+hot|arcing|arc|sparking|sparked|water\s+in\s+the\s+panel)\b", low))


def _v44_sanitize_date_words_from_name(conv: dict) -> None:
    try:
        profile = conv.setdefault("profile", {})
        sched = conv.setdefault("sched", {})
        for key in ["active_last_name", "last_name", "recognized_last_name", "voicemail_last_name"]:
            val = str(profile.get(key) or "").strip()
            if val and val.lower() in _V44_DATE_NAME_WORDS:
                profile[key] = None
        full = " ".join(str(profile.get("name") or "").strip().split())
        if full:
            parts = full.split()
            while len(parts) >= 2 and parts[-1].lower().strip(".,") in _V44_DATE_NAME_WORDS:
                parts.pop()
            profile["name"] = " ".join(parts).strip() or None
        try:
            recompute_pending_step(profile, sched)
        except Exception:
            pass
    except Exception:
        pass


def _v44_is_scheduled_troubleshoot_context(conv: dict, caller_text: str = "") -> bool:
    sched = conv.setdefault("sched", {})
    appt = str(sched.get("appointment_type") or conv.get("appointment_type") or "").upper()
    if "TROUBLESHOOT" not in appt:
        return False
    # True emergencies must stay on the emergency-dispatch lane.
    if _v44_is_hard_hazard(caller_text):
        return False
    if sched.get("emergency_approved") or sched.get("awaiting_emergency_confirm"):
        return False
    return True


def _v44_force_scheduled_troubleshoot_state(conv: dict, pending: str = "need_date") -> None:
    sched = conv.setdefault("sched", {})
    sched["appointment_type"] = "TROUBLESHOOT_395"
    conv["appointment_type"] = "TROUBLESHOOT_395"
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    sched["emergency_price_confirmed"] = False
    sched["hard_emergency_detected"] = False
    if pending:
        sched["pending_step"] = pending
    # This call type is a scheduled troubleshoot, not emergency dispatch.
    sched["state"] = "waiting_for_date" if pending == "need_date" else ("waiting_for_name" if pending == "need_name" else "waiting_for_email")


def _v44_time_text_matches(opt_time: str, caller_text: str) -> bool:
    opt_time = str(opt_time or "").strip()
    if not opt_time:
        return False
    try:
        explicit = extract_explicit_time_from_text(caller_text or "")
        if explicit and explicit == opt_time:
            return True
    except Exception:
        pass
    low = _v44_low(caller_text)
    try:
        hh, mm = opt_time.split(":")[:2]
        h = int(hh)
        minute = int(mm)
        h12 = h % 12 or 12
        ampm = "am" if h < 12 else "pm"
        patterns = [
            rf"\b{h12}\s*{ampm}\b",
            rf"\b{h12}:?{minute:02d}\s*{ampm}\b",
            rf"\b{h}:?{minute:02d}\b",
        ]
        if minute == 0:
            patterns.append(rf"\b{h12}\s*o'?clock\s*{ampm}\b")
        return any(re.search(p, low) for p in patterns)
    except Exception:
        return False


def _v44_date_score_for_option(opt: dict, caller_text: str) -> int:
    low = _v44_low(caller_text)
    score = 0
    opt_date = str(opt.get("date") or "").strip()
    opt_time = str(opt.get("time") or "").strip()
    try:
        requested_date = salvage_relative_date_from_text(caller_text or "")
    except Exception:
        requested_date = None
    if requested_date and requested_date == opt_date:
        score += 8
    try:
        d = datetime.strptime(opt_date, "%Y-%m-%d")
        day_full = d.strftime("%A").lower()
        day_abbr = d.strftime("%a").lower()
        month_full = d.strftime("%B").lower()
        month_abbr = d.strftime("%b").lower()
        day_num = d.day
        if re.search(rf"\b{re.escape(day_full)}\b", low) or re.search(rf"\b{re.escape(day_abbr)}\b", low):
            score += 4
        if re.search(rf"\b{re.escape(month_full)}\b", low) or re.search(rf"\b{re.escape(month_abbr)}\b", low):
            score += 3
        ordinal = str(day_num)
        suffix = "th"
        if day_num % 10 == 1 and day_num % 100 != 11:
            suffix = "st"
        elif day_num % 10 == 2 and day_num % 100 != 12:
            suffix = "nd"
        elif day_num % 10 == 3 and day_num % 100 != 13:
            suffix = "rd"
        if re.search(rf"\b{day_num}\b", low) or re.search(rf"\b{day_num}{suffix}\b", low):
            score += 2
    except Exception:
        pass
    if _v44_time_text_matches(opt_time, caller_text):
        score += 4
    label = str(opt.get("label") or "").strip().lower()
    if label and label in low:
        score += 10
    return score


def _v44_select_offered_slot_direct(conv: dict, caller_text: str) -> dict | None:
    sched = conv.setdefault("sched", {})
    options = sched.get("offered_slot_options") or []
    if not (sched.get("awaiting_slot_offer_choice") and options):
        return None
    low = _v44_low(caller_text)

    # Explicit ordinal choices only. Do NOT treat "June 1st" as "first option".
    ordinal_idx = None
    ordinal_patterns = [
        (0, r"\b(first\s+one|first\s+option|option\s+1|number\s+1|the\s+first)\b"),
        (1, r"\b(second\s+one|second\s+option|option\s+2|number\s+2|the\s+second)\b"),
        (2, r"\b(third\s+one|third\s+option|option\s+3|number\s+3|the\s+third|last\s+one|the\s+last)\b"),
    ]
    for idx, pat in ordinal_patterns:
        if re.search(pat, low):
            ordinal_idx = idx
            break
    if ordinal_idx is not None and ordinal_idx < len(options):
        return dict(options[ordinal_idx])

    scored = []
    for i, opt in enumerate(options):
        try:
            scored.append((_v44_date_score_for_option(opt, caller_text), i, opt))
        except Exception:
            scored.append((0, i, opt))
    scored.sort(reverse=True, key=lambda x: x[0])
    if not scored or scored[0][0] < 4:
        return None
    # Require a clear winner unless only one has any score.
    if len(scored) > 1 and scored[0][0] == scored[1][0] and scored[1][0] > 0:
        return None
    return dict(scored[0][2])


def _v44_apply_selected_slot(conv: dict, chosen: dict) -> None:
    sched = conv.setdefault("sched", {})
    sched["appointment_type"] = "TROUBLESHOOT_395" if "TROUBLESHOOT" in str(sched.get("appointment_type") or conv.get("appointment_type") or "").upper() else (sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195")
    conv["appointment_type"] = sched["appointment_type"]
    sched["scheduled_date"] = chosen.get("date")
    sched["scheduled_time"] = chosen.get("time")
    sched["scheduled_time_source"] = "voice_v44_direct_offered_slot"
    for k in ["start_at", "team_member_id", "service_variation_id", "service_variation_version", "duration_minutes"]:
        if chosen.get(k) is not None:
            sched[k] = chosen.get(k)
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    sched["final_confirmation_sent"] = False
    sched["final_confirmation_accepted"] = False
    sched["last_final_confirmation_key"] = None
    sched["slot_choice_locked"] = True
    sched["booking_attempt_nonce"] = str(uuid.uuid4())
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    if "TROUBLESHOOT" in str(sched.get("appointment_type") or "").upper():
        sched["state"] = "waiting_for_name"
    try:
        recompute_pending_step(conv.setdefault("profile", {}), sched)
    except Exception:
        pass


def _v44_after_slot_prompt(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    _v44_sanitize_date_words_from_name(conv)
    first = _voice_profile_first_name(profile) if "_voice_profile_first_name" in globals() else str(profile.get("active_first_name") or profile.get("first_name") or "").strip()
    last = _voice_profile_last_name(profile) if "_voice_profile_last_name" in globals() else str(profile.get("active_last_name") or profile.get("last_name") or "").strip()
    email_ok = _voice_has_email(conv) if "_voice_has_email" in globals() else bool(profile.get("active_email") or profile.get("email") or sched.get("email"))
    if first and not last:
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
        return "What is your last name?"
    if not first or not last:
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
        return "What's your first and last name?"
    if not email_ok:
        sched["pending_step"] = "need_email"
        sched["state"] = "waiting_for_email"
        return "What's the best email address for the appointment?"
    try:
        nxt = choose_next_prompt_from_state(conv, inbound_text="")
        return _voice_naturalize_reply(nxt) if nxt else "Got it. I'll finish booking that now."
    except Exception:
        return "Got it. I'll finish booking that now."


def _v44_scheduled_troubleshoot_slot_fast_path(conv: dict, caller_text: str) -> str | None:
    sched = conv.setdefault("sched", {})
    if sched.get("hard_emergency_detected") or sched.get("awaiting_emergency_confirm") or sched.get("emergency_approved"):
        return None
    if not _v44_is_scheduled_troubleshoot_context(conv, caller_text):
        return None
    if not (sched.get("awaiting_slot_offer_choice") and (sched.get("offered_slot_options") or [])):
        return None
    chosen = _v44_select_offered_slot_direct(conv, caller_text)
    if not chosen:
        return None
    _v44_apply_selected_slot(conv, chosen)
    # Keep it out of the emergency-dispatch language lane.
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    return _v44_after_slot_prompt(conv)


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    # Pre-intercept offered slot choices for scheduled troubleshoot/repair calls.
    # This is before all older emergency/date/name layers.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        sched_pre = conv_pre.setdefault("sched", {})
        _v44_sanitize_date_words_from_name(conv_pre)
        slot_reply = _v44_scheduled_troubleshoot_slot_fast_path(conv_pre, caller_text)
        if slot_reply:
            slot_reply = _voice_naturalize_reply(slot_reply)
            conv_pre["last_voice_reply"] = slot_reply
            try:
                log_event("VOICE_V44_TROUBLESHOOT_OFFERED_SLOT_DIRECT", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(slot_reply), "call_sid": call_sid}, conv_pre)
            except Exception:
                pass
            return {"reply_to_customer": slot_reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": sched_pre.get("appointment_type") or conv_pre.get("appointment_type"), "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V44_PRE_INTERCEPT_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V44(phone, call_sid, caller_text)

    # Post-cleanup: if older layers still managed to mark a scheduled troubleshoot as emergency or
    # attach a weekday/month as a last name, clean it before the next turn/Square booking.
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        _v44_sanitize_date_words_from_name(conv)
        emergencyish_post = bool(
            sched.get("hard_emergency_detected")
            or sched.get("awaiting_emergency_confirm")
            or sched.get("emergency_approved")
            or str(sched.get("state") or "").lower() == "emergency"
        )
        if (not emergencyish_post) and _v44_is_scheduled_troubleshoot_context(conv, caller_text):
            reply = str(out.get("reply_to_customer") or "")
            low_reply = _v44_low(reply)
            if "this sounds urgent" in low_reply or "send someone now" in low_reply or "dispatch someone now" in low_reply:
                # Do not let generic troubleshoot reopen emergency dispatch after a scheduled slot choice.
                if sched.get("scheduled_date") and sched.get("scheduled_time"):
                    reply2 = _v44_after_slot_prompt(conv)
                else:
                    _v44_force_scheduled_troubleshoot_state(conv, "need_date")
                    reply2 = _v42_slot_offer_reply_for(conv, "TROUBLESHOOT_395") if "_v42_slot_offer_reply_for" in globals() else "What weekday and time works best?"
                out["reply_to_customer"] = _voice_naturalize_reply(reply2)
                out["booking_created"] = False
                out["end_call"] = False
                out["appointment_type"] = "TROUBLESHOOT_395"
                out["pending_step"] = sched.get("pending_step")
                conv["last_voice_reply"] = out["reply_to_customer"]
                try:
                    log_event("VOICE_V44_SUPPRESSED_FALSE_EMERGENCY_REOPEN", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(out["reply_to_customer"]), "call_sid": call_sid}, conv)
                except Exception:
                    pass
    except Exception as e:
        try:
            log_event("VOICE_V44_POST_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# =============================
# v45 STABILIZATION HOTFIX LAYER
# =============================
# Targets live v44 failure:
# - Generic scheduled troubleshoot phrasing like "outlets that don't work" / "troubleshooted"
#   bypassed the $395 consent gate and asked for a day/time.
# - Offered slot choices must be intercepted before older emergency/name/date layers.
# - Keep generic troubleshooting out of the emergency-dispatch script unless there are hard hazard words.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V45 = process_prevolt_voice_turn


def _v45_low(text: str) -> str:
    try:
        return _intent_text(text or "")
    except Exception:
        return str(text or "").lower()


def _v45_hard_hazard(text: str) -> bool:
    try:
        return bool(_v42_hard_hazard_text(text))
    except Exception:
        low = _v45_low(text)
        return bool(re.search(r"\b(smoke|smoking|fire|flames?|burning|burnt|sparking|sparked|arcing|hot\s+to\s+the\s+touch|panel\s+hot|water\s+in\s+(?:the\s+)?panel)\b", low, flags=re.I))


def _v45_generic_troubleshoot_text(text: str) -> bool:
    """Broad scheduled-troubleshoot detector. Does not include hard emergency hazards."""
    low = _v45_low(text)
    if _v45_hard_hazard(low):
        return False
    patterns = [
        r"\btroubleshoot(?:ing|ed)?\b",
        r"\bdiagnos(?:e|ing|tic)\b",
        r"\bpower\s+issues?\b",
        r"\b(?:outlets?|receptacles?|plugs?)\s+(?:that\s+)?(?:do\s+not|don't|dont|won't|wont)\s+(?:work|have\s+power)\b",
        r"\b(?:outlets?|receptacles?|plugs?)\s+(?:are\s+)?(?:dead|not\s+working|without\s+power|no\s+power)\b",
        r"\b(?:lost|no)\s+power\s+(?:to|at|in)\s+(?:a\s+few\s+)?(?:outlets?|rooms?|circuits?)\b",
        r"\b(?:lights?|fixtures?)\s+(?:do\s+not|don't|dont|won't|wont|not)\s+(?:work|turn\s+on)\b",
        r"\bbreaker(?:s)?\s+(?:tripping|keeps?\s+tripping|won't\s+reset|wont\s+reset)\b",
    ]
    return any(re.search(p, low, flags=re.I) for p in patterns)


def _v45_yes(text: str) -> bool:
    try:
        return bool(_v43_yes(text))
    except Exception:
        pass
    try:
        return _v41_yes_no(text) == "yes"
    except Exception:
        return bool(re.search(r"\b(yes|yeah|yep|ok|okay|sure|that works|sounds good|let'?s do it)\b", _v45_low(text), flags=re.I))


def _v45_no(text: str) -> bool:
    try:
        return bool(_v43_no(text))
    except Exception:
        pass
    try:
        return _v41_yes_no(text) == "no"
    except Exception:
        return bool(re.search(r"\b(no|nope|nah|never mind|not now|too much|don'?t)\b", _v45_low(text), flags=re.I))


def _v45_reply_is_generic_schedule_ask(reply: str) -> bool:
    low = _v45_low(reply)
    generic_bits = [
        "what day and time work",
        "what day and time works",
        "what date and time",
        "what time work",
        "what time works",
        "what weekday and time",
        "what day time",
        "day and time that works",
        "get your appointment scheduled here",
        "get this scheduled",
        "scheduling details sorted",
        "next scheduling detail",
    ]
    return any(bit in low for bit in generic_bits)


def _v45_is_troubleshoot_appt(conv: dict, out: dict | None = None) -> bool:
    sched = conv.setdefault("sched", {})
    out = out or {}
    appt = str(sched.get("appointment_type") or conv.get("appointment_type") or out.get("appointment_type") or "").upper()
    return "TROUBLESHOOT" in appt


def _v45_force_scheduled_troubleshoot(conv: dict, pending: str = "need_date") -> None:
    try:
        _v44_force_scheduled_troubleshoot_state(conv, pending)
    except Exception:
        sched = conv.setdefault("sched", {})
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
        sched["awaiting_emergency_confirm"] = False
        sched["emergency_approved"] = False
        sched["hard_emergency_detected"] = False
        sched["pending_step"] = pending
        sched["state"] = "waiting_for_date"


def _v45_troubleshoot_price_gate(conv: dict, caller_text: str = "") -> str:
    sched = conv.setdefault("sched", {})
    _v45_force_scheduled_troubleshoot(conv, "need_date")
    sched["awaiting_troubleshoot_price_confirm"] = True
    sched["troubleshoot_price_accepted"] = False
    sched["awaiting_eval_price_confirm"] = False
    sched["awaiting_slot_offer_choice"] = False
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    try:
        reply = _v43_troubleshoot_price_prompt(conv, caller_text)
    except Exception:
        prefix = "The address is verified in our system. " if sched.get("address_verified") else ""
        reply = prefix + "Troubleshoot and repair visits are $395. That covers sending one of our electricians out to diagnose the issue and repair it on site when possible. Does that work for you?"
    conv["last_voice_reply"] = reply
    return reply


def _v45_offer_troubleshoot_slots(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    _v45_force_scheduled_troubleshoot(conv, "need_date")
    sched["awaiting_troubleshoot_price_confirm"] = False
    sched["troubleshoot_price_accepted"] = True
    sched["awaiting_eval_price_confirm"] = False
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    try:
        reply = _v43_offer_slots_for_troubleshoot(conv)
    except Exception:
        try:
            reply = _v42_slot_offer_reply_for(conv, "TROUBLESHOOT_395")
        except Exception:
            reply = "Great. What weekday and time work best for you?"
    conv["last_voice_reply"] = reply
    return reply


def _v45_decline_troubleshoot(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    sched["awaiting_troubleshoot_price_confirm"] = False
    sched["troubleshoot_price_accepted"] = False
    sched["closed_reason"] = "declined_troubleshoot_price"
    sched["voice_close_after_reply"] = True
    reply = "No problem. We will not schedule the visit right now. Thank you for calling Prevolt Electric. Goodbye."
    conv["last_voice_reply"] = reply
    return reply


def _v45_any_offered_slot_choice_fast_path(conv: dict, caller_text: str) -> str | None:
    """Intercept offered-slot choices for all appointment types before older layers can parse dates as names."""
    sched = conv.setdefault("sched", {})
    options = sched.get("offered_slot_options") or []
    last_reply = str(conv.get("last_voice_reply") or sched.get("last_voice_reply") or conv.get("last_sms_body") or "")

    # If a previous layer already staged a single date/time and the caller says
    # "that works", accept that staged slot instead of re-offering three options.
    if not (sched.get("awaiting_slot_offer_choice") and options):
        if (
            _v45_yes(caller_text)
            and sched.get("scheduled_date")
            and sched.get("scheduled_time")
            and not sched.get("awaiting_troubleshoot_price_confirm")
            and not sched.get("awaiting_eval_price_confirm")
            and re.search(r"\b(?:we have|appointment time|confirm that appointment|that appointment)\b", last_reply, flags=re.I)
        ):
            sched["awaiting_slot_offer_choice"] = False
            sched["offered_slot_options"] = []
            sched["last_slot_unavailable_message"] = None
            sched["slot_choice_locked"] = True
            sched["booking_attempt_nonce"] = str(uuid.uuid4())
            if _v45_is_troubleshoot_appt(conv):
                sched["awaiting_emergency_confirm"] = False
                sched["emergency_approved"] = False
                sched["hard_emergency_detected"] = False
                sched["troubleshoot_price_accepted"] = True
                sched["state"] = "waiting_for_name"
            try:
                recompute_pending_step(conv.setdefault("profile", {}), sched)
            except Exception:
                sched["pending_step"] = "need_name"
            try:
                reply = _v44_after_slot_prompt(conv)
            except Exception:
                reply = "What's your first and last name?"
            conv["last_voice_reply"] = reply
            return reply
        return None

    # Do not treat a plain yes/no as a slot choice unless only one option exists.
    if (_v45_yes(caller_text) or _v45_no(caller_text)) and len(options) > 1:
        return None
    if _v45_yes(caller_text) and len(options) == 1:
        chosen = dict(options[0])
    else:
        try:
            chosen = _v44_select_offered_slot_direct(conv, caller_text)
        except Exception:
            chosen = None
    if not chosen:
        return None
    try:
        _v44_apply_selected_slot(conv, chosen)
    except Exception:
        sched["scheduled_date"] = chosen.get("date")
        sched["scheduled_time"] = chosen.get("time")
        for k in ["start_at", "team_member_id", "service_variation_id", "service_variation_version", "duration_minutes"]:
            if chosen.get(k) is not None:
                sched[k] = chosen.get(k)
        sched["awaiting_slot_offer_choice"] = False
        sched["offered_slot_options"] = []
        sched["pending_step"] = "need_name"
        sched["state"] = "waiting_for_name"
    # Keep scheduled troubleshoot out of emergency dispatch after the slot is selected.
    if _v45_is_troubleshoot_appt(conv):
        sched["awaiting_emergency_confirm"] = False
        sched["emergency_approved"] = False
        sched["hard_emergency_detected"] = False
        sched["troubleshoot_price_accepted"] = True
    try:
        _v44_sanitize_date_words_from_name(conv)
    except Exception:
        pass
    try:
        reply = _v44_after_slot_prompt(conv)
    except Exception:
        reply = "What is your last name?"
    conv["last_voice_reply"] = reply
    return reply


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()

    # 1) Highest priority: slot choice. This prevents date words from becoming names
    # and prevents generic troubleshoot from reopening emergency-dispatch language.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        try:
            _v44_sanitize_date_words_from_name(conv_pre)
        except Exception:
            pass
        slot_reply = _v45_any_offered_slot_choice_fast_path(conv_pre, caller_text)
        if slot_reply:
            slot_reply = _voice_naturalize_reply(slot_reply)
            try:
                log_event("VOICE_V45_OFFERED_SLOT_DIRECT", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(slot_reply), "call_sid": call_sid}, conv_pre)
            except Exception:
                pass
            return {"reply_to_customer": slot_reply, "booking_created": False, "manual_only": False, "pending_step": conv_pre.setdefault("sched", {}).get("pending_step"), "appointment_type": conv_pre.setdefault("sched", {}).get("appointment_type") or conv_pre.get("appointment_type"), "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V45_SLOT_PRE_INTERCEPT_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    # 2) Price confirmation lane for scheduled troubleshoot.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        sched_pre = conv_pre.setdefault("sched", {})
        if sched_pre.get("awaiting_troubleshoot_price_confirm"):
            if _v45_no(caller_text):
                reply = _v45_decline_troubleshoot(conv_pre)
                try:
                    log_event("VOICE_V45_TROUBLESHOOT_PRICE_DECLINED", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "TROUBLESHOOT_395", "end_call": True}
            if _v45_yes(caller_text) or (_v42_availability_request_text(caller_text) if "_v42_availability_request_text" in globals() else False):
                reply = _v45_offer_troubleshoot_slots(conv_pre)
                try:
                    log_event("VOICE_V45_TROUBLESHOOT_PRICE_ACCEPTED_OFFER_SLOTS", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "TROUBLESHOOT_395", "end_call": False}
            reply = _v45_troubleshoot_price_gate(conv_pre, caller_text)
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "TROUBLESHOOT_395", "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V45_PRICE_PRE_INTERCEPT_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V45(phone, call_sid, caller_text)

    # 3) Postprocess stabilization: if older layers routed a generic troubleshoot to date collection
    # or emergency dispatch without the $395 consent gate, replace the reply with the correct gate.
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        reply = str(out.get("reply_to_customer") or "")
        low_reply = _v45_low(reply)
        generic_ts = _v45_generic_troubleshoot_text(caller_text) or (_v45_is_troubleshoot_appt(conv, out) and not _v45_hard_hazard(caller_text))
        price_already_accepted = bool(sched.get("troubleshoot_price_accepted"))
        already_asking_price = _v43_is_troubleshoot_price_prompt(reply) if "_v43_is_troubleshoot_price_prompt" in globals() else ("395" in low_reply and "does that work" in low_reply)

        if generic_ts and not price_already_accepted and not already_asking_price:
            if _v45_reply_is_generic_schedule_ask(reply) or "this sounds urgent" in low_reply or "send someone now" in low_reply or "dispatch someone now" in low_reply:
                reply2 = _v45_troubleshoot_price_gate(conv, caller_text)
                out["reply_to_customer"] = _voice_naturalize_reply(reply2)
                out["booking_created"] = False
                out["manual_only"] = False
                out["pending_step"] = sched.get("pending_step")
                out["appointment_type"] = "TROUBLESHOOT_395"
                out["end_call"] = False
                try:
                    log_event("VOICE_V45_TROUBLESHOOT_PRICE_GATE_FORCED", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(out["reply_to_customer"]), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return out

        # If older layers output the price prompt, make sure the next turn is held in the price-confirm lane.
        if generic_ts and already_asking_price and not price_already_accepted:
            _v45_force_scheduled_troubleshoot(conv, "need_date")
            sched["awaiting_troubleshoot_price_confirm"] = True
            sched["awaiting_emergency_confirm"] = False
            sched["emergency_approved"] = False
            out["appointment_type"] = "TROUBLESHOOT_395"
            out["pending_step"] = sched.get("pending_step")
            out["end_call"] = False
            conv["last_voice_reply"] = out.get("reply_to_customer") or reply

        # Final cleanup: date words cannot remain as last names.
        try:
            _v44_sanitize_date_words_from_name(conv)
        except Exception:
            pass
    except Exception as e:
        try:
            log_event("VOICE_V45_POST_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# =============================
# v46 STABILIZATION HOTFIX LAYER
# =============================
# Targets live v45 failures:
# - "outlets stopped working tonight" was being routed to the $195 evaluation visit instead of
#   the $395 scheduled troubleshoot/repair visit.
# - After the customer accepted the $195 evaluation price, older layers could repeat the $195 prompt
#   instead of offering the stored appointment options.
# - Email confirmation could repeat after the customer said yes.
# - The voice layer could advance to email before getting the last name.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V46 = process_prevolt_voice_turn


def _v46_low(text: str) -> str:
    try:
        return _intent_text(text or "")
    except Exception:
        return str(text or "").lower()


def _v46_is_hard_hazard(text: str) -> bool:
    try:
        return bool(_v45_hard_hazard(text))
    except Exception:
        try:
            return bool(_v42_hard_hazard_text(text))
        except Exception:
            return bool(re.search(r"\b(smoke|smoking|fire|flames?|burning|burnt|sparking|sparked|arcing|hot\s+to\s+the\s+touch|panel\s+hot|water\s+in\s+(?:the\s+)?panel)\b", _v46_low(text), flags=re.I))


def _v46_power_loss_troubleshoot_text(text: str) -> bool:
    """Scheduled troubleshoot/repair detector. Does not treat generic power loss as emergency dispatch."""
    low = _v46_low(text)
    if not low or _v46_is_hard_hazard(low):
        return False
    patterns = [
        r"\btroubleshoot(?:ing|ed)?\b",
        r"\bdiagnos(?:e|ing|tic)\b",
        r"\bpower\s+issues?\b",
        r"\b(?:outlets?|receptacles?|plugs?)\s+(?:that\s+)?(?:do\s+not|don't|dont|won't|wont)\s+(?:work|have\s+power)\b",
        r"\b(?:outlets?|receptacles?|plugs?)\s+(?:are\s+)?(?:dead|not\s+working|without\s+power|no\s+power)\b",
        r"\b(?:outlets?|receptacles?|plugs?)\s+(?:that\s+)?(?:stopped|stop|quit|went\s+out)\s+(?:working|work)\b",
        r"\b(?:few|couple|some|several)?\s*(?:outlets?|receptacles?|plugs?)\s+(?:stopped|quit|went\s+out)\b",
        r"\b(?:lost|no)\s+power\s+(?:to|at|in)\s+(?:a\s+few\s+|some\s+|several\s+)?(?:outlets?|rooms?|circuits?)\b",
        r"\b(?:lights?|fixtures?)\s+(?:do\s+not|don't|dont|won't|wont|not)\s+(?:work|turn\s+on)\b",
        r"\b(?:lights?|fixtures?)\s+(?:stopped|stop|quit|went\s+out)\s+(?:working|work)?\b",
        r"\bbreaker(?:s)?\s+(?:tripping|keeps?\s+tripping|won't\s+reset|wont\s+reset)\b",
    ]
    return any(re.search(p, low, flags=re.I) for p in patterns)


def _v46_is_eval_price_prompt(reply: str) -> bool:
    low = _v46_low(reply)
    return bool(("195" in low or "on site evaluation" in low or "onsite evaluation" in low or "evaluation visit" in low) and "does that work" in low)


def _v46_is_email_confirm_prompt(reply: str) -> bool:
    low = _v46_low(reply)
    return bool("i heard" in low and "is that correct" in low and ("gmail" in low or " at " in low or " dot " in low or "email" in low))


def _v46_name_status(conv: dict) -> tuple[str, str]:
    profile = conv.setdefault("profile", {})
    try:
        first = _voice_profile_first_name(profile)
    except Exception:
        first = str(profile.get("active_first_name") or profile.get("first_name") or "").strip()
    try:
        last = _voice_profile_last_name(profile)
    except Exception:
        last = str(profile.get("active_last_name") or profile.get("last_name") or "").strip()
    return first, last


def _v46_require_last_name_reply(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    first, last = _v46_name_status(conv)
    sched["pending_step"] = "need_name"
    sched["state"] = "waiting_for_name"
    if first and not last:
        return f"{first}, what is your last name?"
    return "What's your first and last name?"


def _v46_offer_eval_slots(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    sched["appointment_type"] = "EVAL_195"
    conv["appointment_type"] = "EVAL_195"
    sched["awaiting_eval_price_confirm"] = False
    sched["eval_price_accepted"] = True
    sched["awaiting_troubleshoot_price_confirm"] = False
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    sched["pending_step"] = "need_date"
    sched["state"] = "waiting_for_date"
    try:
        reply = _v42_slot_offer_reply_for(conv, "EVAL_195")
    except Exception:
        reply = "Great. What weekday and time work best for you?"
    conv["last_voice_reply"] = reply
    return reply


def _v46_eval_price_declined(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    sched["awaiting_eval_price_confirm"] = False
    sched["eval_price_accepted"] = False
    sched["closed_reason"] = "declined_eval_price"
    sched["voice_close_after_reply"] = True
    reply = "No problem. We will not schedule the visit right now. Thank you for calling Prevolt Electric. Goodbye."
    conv["last_voice_reply"] = reply
    return reply


def _v46_hold_eval_price_gate(conv: dict, reply: str) -> None:
    sched = conv.setdefault("sched", {})
    sched["appointment_type"] = sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195"
    conv["appointment_type"] = sched["appointment_type"]
    if "TROUBLESHOOT" not in str(sched.get("appointment_type") or "").upper():
        sched["appointment_type"] = "EVAL_195"
        conv["appointment_type"] = "EVAL_195"
        sched["awaiting_eval_price_confirm"] = True
        sched["eval_price_accepted"] = False
        sched["awaiting_slot_offer_choice"] = False
        sched["pending_step"] = "need_date"
        sched["state"] = "waiting_for_date"
        conv["last_voice_reply"] = reply


def _v46_email_confirm_yes(conv: dict, phone: str) -> dict | None:
    sched = conv.setdefault("sched", {})
    pending = str(sched.get("voice_pending_email") or sched.get("pending_email") or "").strip().lower()
    if not pending:
        return None
    try:
        out = _v36_confirmed_email_save_and_book(phone, conv, pending)
    except Exception:
        try:
            out = _voice_save_confirmed_email_and_maybe_book_v31(phone, conv, pending)
        except Exception:
            return None
    sched["voice_email_confirmed"] = True
    sched["voice_awaiting_email_confirm"] = False
    sched["voice_pending_email"] = None
    reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or "")) if "_voice_naturalize_reply" in globals() else str(out.get("reply_to_customer") or "")
    # Hard stop: after a yes to email confirmation, do not repeat the email confirmation prompt.
    if _v46_is_email_confirm_prompt(reply):
        first, last = _v46_name_status(conv)
        if not last:
            reply = _v46_require_last_name_reply(conv)
            out["booking_created"] = False
            out["end_call"] = False
            out["pending_step"] = "need_name"
        elif not (sched.get("scheduled_date") and sched.get("scheduled_time")):
            try:
                reply = _v42_slot_offer_reply_for(conv, sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195")
            except Exception:
                reply = "Thanks. What day and time works best?"
            out["booking_created"] = False
            out["end_call"] = False
            out["pending_step"] = sched.get("pending_step")
        else:
            reply = "Thanks. I'll finish booking that now."
            out["end_call"] = False
    out["reply_to_customer"] = reply
    conv["last_voice_reply"] = reply
    return out


def _v46_cleanup_reply_against_state(conv: dict, reply: str) -> str:
    sched = conv.setdefault("sched", {})
    first, last = _v46_name_status(conv)
    low = _v46_low(reply)
    # Do not let the voice layer ask for email until the last name is actually present.
    if first and not last and ("best email" in low or "email address" in low or "@" in reply):
        return _v46_require_last_name_reply(conv)
    # Avoid combined prompts like "what is your last name? what's the best email..."
    if first and not last and "last name" in low and ("best email" in low or "email address" in low):
        return _v46_require_last_name_reply(conv)
    return reply


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()

    # A) Highest priority: email confirmation. If the caller says yes, never let lower layers
    # repeat the same email confirmation question.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        sched_pre = conv_pre.setdefault("sched", {})
        last_reply_pre = str(conv_pre.get("last_voice_reply") or sched_pre.get("last_voice_reply") or "")
        awaiting_email_confirm = bool(sched_pre.get("voice_awaiting_email_confirm")) or (_v46_is_email_confirm_prompt(last_reply_pre) and bool(sched_pre.get("voice_pending_email")))
        if awaiting_email_confirm and _v45_yes(caller_text):
            out = _v46_email_confirm_yes(conv_pre, p)
            if out:
                try:
                    log_event("VOICE_V46_EMAIL_CONFIRM_ACCEPTED", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(out.get("reply_to_customer")), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return out
    except Exception as e:
        try:
            log_event("VOICE_V46_EMAIL_PRE_INTERCEPT_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    # B) Evaluation price-confirm lane. This prevents the repeated $195 prompt after "yes".
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        sched_pre = conv_pre.setdefault("sched", {})
        last_reply_pre = str(conv_pre.get("last_voice_reply") or sched_pre.get("last_voice_reply") or "")
        emergencyish_pre = bool(
            sched_pre.get("hard_emergency_detected")
            or sched_pre.get("awaiting_emergency_confirm")
            or sched_pre.get("emergency_approved")
            or str(sched_pre.get("state") or "").lower() == "emergency"
            or "TROUBLESHOOT" in str(sched_pre.get("appointment_type") or conv_pre.get("appointment_type") or "").upper()
        )
        awaiting_eval = (bool(sched_pre.get("awaiting_eval_price_confirm")) or (_v46_is_eval_price_prompt(last_reply_pre) and "TROUBLESHOOT" not in str(sched_pre.get("appointment_type") or conv_pre.get("appointment_type") or "").upper())) and not emergencyish_pre
        if awaiting_eval:
            if _v45_no(caller_text):
                reply = _v46_eval_price_declined(conv_pre)
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "EVAL_195", "end_call": True}
            if _v45_yes(caller_text) or (_v42_availability_request_text(caller_text) if "_v42_availability_request_text" in globals() else False):
                reply = _v46_offer_eval_slots(conv_pre)
                try:
                    log_event("VOICE_V46_EVAL_PRICE_ACCEPTED_OFFER_SLOTS", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "EVAL_195", "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V46_EVAL_PRE_INTERCEPT_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    # C) Slot choice still remains above the older layers; reuse v45's direct handler.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        slot_reply = _v45_any_offered_slot_choice_fast_path(conv_pre, caller_text) if "_v45_any_offered_slot_choice_fast_path" in globals() else None
        if slot_reply:
            slot_reply = _voice_naturalize_reply(slot_reply) if "_voice_naturalize_reply" in globals() else slot_reply
            slot_reply = _v46_cleanup_reply_against_state(conv_pre, slot_reply)
            conv_pre["last_voice_reply"] = slot_reply
            try:
                log_event("VOICE_V46_OFFERED_SLOT_DIRECT", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(slot_reply), "call_sid": call_sid}, conv_pre)
            except Exception:
                pass
            return {"reply_to_customer": slot_reply, "booking_created": False, "manual_only": False, "pending_step": conv_pre.setdefault("sched", {}).get("pending_step"), "appointment_type": conv_pre.setdefault("sched", {}).get("appointment_type") or conv_pre.get("appointment_type"), "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V46_SLOT_PRE_INTERCEPT_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V46(phone, call_sid, caller_text)

    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        reply = _voice_naturalize_reply(str(out.get("reply_to_customer") or "")) if "_voice_naturalize_reply" in globals() else str(out.get("reply_to_customer") or "")
        low_reply = _v46_low(reply)

        # D) Force normal scheduled $395 troubleshoot for power-loss/outlet-failure language,
        # but never skip address collection. Earlier layers sometimes stored the customer's issue
        # text as an address candidate, then jumped to price/date. Fix that at the source.
        if _v46_power_loss_troubleshoot_text(caller_text) and not _v46_is_hard_hazard(caller_text):
            sched["voice_last_intent_power_loss"] = True
            sched["last_customer_issue"] = caller_text
            sched["appointment_type"] = "TROUBLESHOOT_395"
            conv["appointment_type"] = "TROUBLESHOOT_395"
            sched["awaiting_emergency_confirm"] = False
            sched["emergency_approved"] = False
            sched["hard_emergency_detected"] = False

            if not sched.get("address_verified"):
                sched["raw_address"] = ""
                sched["address_candidate"] = ""
                sched["address_verified"] = False
                sched["address_missing"] = "street"
                sched["pending_step"] = "need_address"
                sched["state"] = "waiting_for_address"
                sched["awaiting_slot_offer_choice"] = False
                sched["offered_slot_options"] = []
                sched["awaiting_troubleshoot_price_confirm"] = False
                sched["troubleshoot_price_accepted"] = False
                sched["awaiting_eval_price_confirm"] = False
                reply2 = "Got it. What is the full address for the work, including the town and state?"
                out["reply_to_customer"] = _voice_naturalize_reply(reply2) if "_voice_naturalize_reply" in globals() else reply2
                out["booking_created"] = False
                out["manual_only"] = False
                out["appointment_type"] = "TROUBLESHOOT_395"
                out["pending_step"] = "need_address"
                out["end_call"] = False
                conv["last_voice_reply"] = out["reply_to_customer"]
                try:
                    log_event("VOICE_V46_POWER_LOSS_NEEDS_ADDRESS", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(out["reply_to_customer"]), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return out

            already_395 = "395" in low_reply or "troubleshoot and repair" in low_reply
            if not sched.get("troubleshoot_price_accepted") and not already_395:
                reply2 = _v45_troubleshoot_price_gate(conv, caller_text) if "_v45_troubleshoot_price_gate" in globals() else _v43_troubleshoot_price_prompt(conv, caller_text)
                out["reply_to_customer"] = _voice_naturalize_reply(reply2) if "_voice_naturalize_reply" in globals() else reply2
                out["booking_created"] = False
                out["manual_only"] = False
                out["appointment_type"] = "TROUBLESHOOT_395"
                out["pending_step"] = sched.get("pending_step")
                out["end_call"] = False
                try:
                    log_event("VOICE_V46_POWER_LOSS_FORCED_395", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(out["reply_to_customer"]), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return out

        # E) If the reply is a $195 gate, remember that the next yes means offer slots, not repeat price.
        emergencyish = bool(
            sched.get("hard_emergency_detected")
            or sched.get("awaiting_emergency_confirm")
            or sched.get("emergency_approved")
            or str(sched.get("state") or "").lower() == "emergency"
            or "TROUBLESHOOT" in str(sched.get("appointment_type") or conv.get("appointment_type") or "").upper()
        )
        if _v46_is_eval_price_prompt(reply) and not emergencyish and "TROUBLESHOOT" not in str(sched.get("appointment_type") or conv.get("appointment_type") or "").upper():
            _v46_hold_eval_price_gate(conv, reply)
            out["appointment_type"] = "EVAL_195"
            out["pending_step"] = sched.get("pending_step")
            out["end_call"] = False

        # F) If lower layers still repeated $195 after a yes, replace it with slot options.
        last_reply = str(conv.get("last_voice_reply") or sched.get("last_voice_reply") or "")
        if (not emergencyish) and _v45_yes(caller_text) and _v46_is_eval_price_prompt(reply) and (_v46_is_eval_price_prompt(last_reply) or sched.get("awaiting_eval_price_confirm")):
            reply2 = _v46_offer_eval_slots(conv)
            out["reply_to_customer"] = reply2
            out["booking_created"] = False
            out["manual_only"] = False
            out["appointment_type"] = "EVAL_195"
            out["pending_step"] = sched.get("pending_step")
            out["end_call"] = False
            try:
                log_event("VOICE_V46_SUPPRESSED_DUPLICATE_EVAL_PRICE", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(reply2), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return out

        # G) Never ask for email before a required last name.
        cleaned = _v46_cleanup_reply_against_state(conv, str(out.get("reply_to_customer") or reply))
        if cleaned != (out.get("reply_to_customer") or reply):
            out["reply_to_customer"] = cleaned
            out["booking_created"] = False
            out["end_call"] = False
            out["pending_step"] = conv.setdefault("sched", {}).get("pending_step")
            conv["last_voice_reply"] = cleaned
            try:
                log_event("VOICE_V46_HELD_EMAIL_UNTIL_LAST_NAME", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(cleaned), "call_sid": call_sid}, conv)
            except Exception:
                pass

        # H) Final email-confirm duplication guard.
        if _v45_yes(caller_text) and _v46_is_email_confirm_prompt(str(out.get("reply_to_customer") or "")):
            out2 = _v46_email_confirm_yes(conv, p)
            if out2:
                return out2
    except Exception as e:
        try:
            log_event("VOICE_V46_POST_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# =============================
# v47 STABILIZATION HOTFIX LAYER
# =============================
# Targets v46 live failures:
# - False address verification like "54 Bloomfield Ave, in Windsor".
# - Spelled street-name corrections being ignored after the assistant asked for spelling.
# - Outlet / power-loss troubleshoot calls falling into the $195 evaluation lane.
# - Customer custom date/time requests after offered slots silently falling back to an offered slot.
# - Duplicate email confirmation after the customer says yes.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V47 = process_prevolt_voice_turn


def _v47_low(text: str) -> str:
    try:
        return _intent_text(text or "")
    except Exception:
        return str(text or "").lower()


def _v47_yes(text: str) -> bool:
    try:
        return bool(_v45_yes(text))
    except Exception:
        return bool(re.search(r"\b(yes|yeah|yep|ok|okay|sure|sounds good|that works|correct)\b", _v47_low(text), flags=re.I))


def _v47_no(text: str) -> bool:
    try:
        return bool(_v45_no(text))
    except Exception:
        return bool(re.search(r"\b(no|nope|nah|incorrect|wrong|not correct|never mind)\b", _v47_low(text), flags=re.I))


def _v47_hard_hazard(text: str) -> bool:
    try:
        return bool(_v46_is_hard_hazard(text))
    except Exception:
        return bool(re.search(r"\b(smoke|smoking|fire|flames?|burning|burnt|sparking|sparked|arcing|hot\s+to\s+the\s+touch|panel\s+hot|water\s+in\s+(?:the\s+)?panel)\b", _v47_low(text), flags=re.I))


def _v47_power_loss_troubleshoot_text(text: str) -> bool:
    """Broad but non-emergency troubleshoot detector for dead outlets/lights/power issues."""
    low = _v47_low(text)
    if not low or _v47_hard_hazard(low):
        return False
    try:
        if _v46_power_loss_troubleshoot_text(text):
            return True
    except Exception:
        pass
    patterns = [
        r"\btroubleshoot(?:ing|ed)?\b",
        r"\bdiagnos(?:e|ing|tic)\b",
        r"\bpower\s+issues?\b",
        r"\b(?:outlets?|receptacles?|plugs?)\s+(?:that\s+)?(?:aren'?t|arent|isn'?t|isnt|are\s+not|not|don'?t|dont|do\s+not|won'?t|wont)\s+(?:work|working|have\s+power)\b",
        r"\b(?:outlets?|receptacles?|plugs?)\s+(?:are\s+)?(?:dead|not\s+working|without\s+power|no\s+power)\b",
        r"\b(?:outlets?|receptacles?|plugs?)\s+(?:that\s+)?(?:stopped|stop|quit|went\s+out)\s+(?:working|work)\b",
        r"\b(?:couple|few|some|several)\s+(?:outlets?|receptacles?|plugs?)\s+(?:that\s+)?(?:aren'?t|arent|don'?t|dont|do\s+not|not|stopped|dead|no\s+power)\b",
        r"\b(?:lost|no)\s+power\s+(?:to|at|in)\s+(?:a\s+few\s+|some\s+|several\s+|couple\s+)?(?:outlets?|rooms?|circuits?)\b",
        r"\b(?:lights?|fixtures?)\s+(?:aren'?t|arent|don'?t|dont|do\s+not|not|won'?t|wont)\s+(?:work|working|turn\s+on)\b",
        r"\bbreaker(?:s)?\s+(?:tripping|keeps?\s+tripping|won'?t\s+reset|wont\s+reset)\b",
    ]
    return any(re.search(p, low, flags=re.I) for p in patterns)


def _v47_complete_addr_struct(a: dict | None) -> bool:
    try:
        return bool(_v38_complete_addr_struct(a))
    except Exception:
        return bool(isinstance(a, dict) and (a.get("address_line_1") or "").strip() and (a.get("locality") or "").strip() and (a.get("administrative_district_level_1") or "").strip() and (a.get("postal_code") or "").strip())


def _v47_address_verified_structured(conv: dict) -> bool:
    sched = conv.setdefault("sched", {})
    if not sched.get("address_verified"):
        return False
    if _v47_complete_addr_struct(sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None):
        return True
    raw = str(sched.get("raw_address") or sched.get("address") or "").strip()
    # Block the known bad false-positive format from v46.
    if re.search(r",\s*in\s+", raw, flags=re.I):
        return False
    # If no normalized struct exists, still require explicit CT/MA and a town-looking comma.
    return bool(re.search(r"^\d{1,6}\b", raw) and re.search(r",\s*[^,]+,\s*(CT|MA)\b", raw, flags=re.I))


def _v47_state_from_text(text: str) -> str | None:
    low = _v47_low(text)
    if re.search(r"\b(ct|connecticut)\b", low):
        return "CT"
    if re.search(r"\b(ma|massachusetts)\b", low):
        return "MA"
    return None


def _v47_known_ct_town(town: str) -> str | None:
    low = _v47_low(town)
    ct_towns = {
        "windsor", "windsor locks", "east windsor", "south windsor", "suffield", "enfield", "granby", "east granby",
        "hartford", "east hartford", "west hartford", "bloomfield", "simsbury", "avon", "farmington", "canton",
        "ellington", "vernon", "manchester", "glastonbury", "newington", "wethersfield", "rocky hill", "plainville", "bristol",
    }
    ma_towns = {"springfield", "west springfield", "agawam", "longmeadow", "east longmeadow", "westfield", "chicopee", "holyoke", "northampton"}
    if low in ct_towns:
        return "CT"
    if low in ma_towns:
        return "MA"
    return None


def _v47_extract_address_candidate(text: str) -> tuple[str | None, str | None, str | None]:
    """Return (street_line, town, state) from natural speech like '54 Bloomfield Ave in Windsor'."""
    s = " ".join(str(text or "").replace("\n", " ").split())
    # Cut common service/problem tails so 'in Windsor and I have outlets...' does not become the town.
    s2 = re.split(r"\b(?:and\s+i\b|and\s+we\b|because\b|with\b|for\b|looking\b|hoping\b|need\b|have\b)", s, maxsplit=1, flags=re.I)[0]
    m = re.search(
        r"\b(?P<num>\d{1,6})\s+(?P<street>[A-Za-z][A-Za-z0-9 .'-]{1,45}?)\s+(?P<typ>ave(?:nue)?|st(?:reet)?|rd|road|dr(?:ive)?|ln|lane|ct|court|blvd|boulevard|way|pl|place|circle|cir|terrace|ter)\b(?:\s*(?:,|in|at)\s*(?P<town>[A-Za-z][A-Za-z .'-]{1,35}))?",
        s2,
        flags=re.I,
    )
    if not m:
        try:
            cand = extract_service_address_from_text(text)
            if cand:
                return cand, None, _v47_state_from_text(text)
        except Exception:
            pass
        return None, None, None
    typ_raw = m.group("typ") or ""
    typ_map = {
        "avenue": "Ave", "ave": "Ave", "street": "St", "st": "St", "road": "Rd", "rd": "Rd", "drive": "Dr", "dr": "Dr",
        "lane": "Ln", "ln": "Ln", "court": "Ct", "ct": "Ct", "boulevard": "Blvd", "blvd": "Blvd", "circle": "Cir", "cir": "Cir",
        "terrace": "Ter", "ter": "Ter", "place": "Pl", "pl": "Pl", "way": "Way",
    }
    typ = typ_map.get(typ_raw.lower(), typ_raw.title())
    street_words = " ".join(m.group("street").split()).title()
    line = f"{m.group('num')} {street_words} {typ}".strip()
    town = m.group("town")
    if town:
        town = re.sub(r"\b(ct|connecticut|ma|massachusetts)\b", "", town, flags=re.I).strip(" ,.")
        # Keep town sane.
        town = " ".join(town.split()[:3]).title()
    state = _v47_state_from_text(text) or (_v47_known_ct_town(town or "") if town else None)
    return line, town, state


def _v47_apply_normalized_if_possible(conv: dict, raw: str) -> bool:
    """Normalize only when the customer supplied a real town/state or known local town.

    v58: Do not geocode street-only/noisy partial addresses. Ask for town/state first.
    """
    if not raw:
        return False

    try:
        allowed, safe_raw, forced_state, town, why = _v58_prepare_address_for_maps(raw)
    except Exception:
        allowed, safe_raw, forced_state, town, why = False, None, None, None, "v58_guard_error"

    if not allowed:
        try:
            sched = conv.setdefault("sched", {})
            sched["address_verified"] = False
            sched["address_missing"] = "state" if why in {"missing_town", "unknown_town_without_state"} else "confirm"
        except Exception:
            pass
        return False

    # First use the guarded Google resolver. It now refuses street-only guesses.
    try:
        addr, reason = _v38_google_resolve_partial_address(safe_raw or raw)
        if addr and _v47_complete_addr_struct(addr) and _v58_locality_ok(addr, town, forced_state):
            _v38_apply_normalized_address(conv, addr, "v47_google_v58")
            return True
    except Exception:
        pass

    # Direct forced-state fallback. Accept only if Google preserves the caller's town/state.
    try:
        result = normalize_address(safe_raw or raw, forced_state=forced_state)
        if isinstance(result, tuple) and len(result) >= 2 and result[0] == "ok" and _v47_complete_addr_struct(result[1]) and _v58_locality_ok(result[1], town, forced_state):
            _v38_apply_normalized_address(conv, result[1], "v47_normalize_v58")
            return True
    except Exception:
        pass

    return False
def _v47_repair_or_block_address(conv: dict, caller_text: str) -> str | None:
    """Repair bad partial addresses or block the flow from treating them as verified."""
    sched = conv.setdefault("sched", {})
    if _v47_address_verified_structured(conv):
        return None

    line, town, state = _v47_extract_address_candidate(caller_text)
    # Reuse stored town hint if present.
    if not town:
        town = sched.get("voice_town_hint") or sched.get("town_hint") or sched.get("locality")
    if not state and town:
        state = _v47_known_ct_town(town)
    raw_parts = []
    if line:
        raw_parts.append(line)
        if town:
            raw_parts.append(str(town).strip())
        if state:
            raw_parts.append(str(state).strip())
        raw = ", ".join([p for p in raw_parts if p])
        if _v47_apply_normalized_if_possible(conv, raw):
            sched["address_verified"] = True
            sched["address_missing"] = None
            return None
        # Store a clean candidate, but do not call it verified.
        sched["raw_address"] = raw
        sched["address_candidate"] = raw

    raw_existing = str(sched.get("raw_address") or sched.get("address_candidate") or "").strip()
    if raw_existing and not _v47_address_verified_structured(conv):
        if _v47_apply_normalized_if_possible(conv, raw_existing):
            return None

        # Do not read the customer's whole service issue back as though it was an address.
        # Older layers can mistakenly store "outlets stopped working..." in raw_address.
        existing_line, _existing_town, _existing_state = _v47_extract_address_candidate(raw_existing)
        raw_looks_like_issue = (
            _v47_power_loss_troubleshoot_text(raw_existing)
            or bool(re.search(r"\b(?:outlets?|lights?|breaker|circuit|panel|troubleshoot|fix|repair|stopped working|not working)\b", _v47_low(raw_existing), flags=re.I))
        )
        if raw_looks_like_issue and not (line or existing_line):
            sched["raw_address"] = ""
            sched["address_candidate"] = ""
            sched["address_verified"] = False
            sched["address_missing"] = "street"
            sched["pending_step"] = "need_address"
            sched["state"] = "waiting_for_address"
            sched["awaiting_slot_offer_choice"] = False
            sched["offered_slot_options"] = []
            return "Got it. What is the full address for the work, including the town and state?"

        sched["address_verified"] = False
        sched["address_missing"] = "confirm"
        sched["pending_step"] = "need_address"
        if str(sched.get("state") or "").lower() not in {"emergency"}:
            sched["state"] = "waiting_for_address"
        sched["awaiting_slot_offer_choice"] = False
        sched["offered_slot_options"] = []
        if line or raw_existing:
            return "I couldn’t verify that address yet. Please say the full address, including the town and state."

    return None


def _v47_spelled_word(text: str) -> str | None:
    raw = str(text or "").strip()
    cleaned = re.sub(r"\b(as in|for)\b.*", "", raw, flags=re.I).strip()
    tokens = re.findall(r"[A-Za-z]", cleaned)
    # Treat B-L-O-O-M-F-I-E-L-D or B L O O M F I E L D as spelling only if mostly single letters/separators.
    if len(tokens) >= 3:
        no_sep = re.sub(r"[A-Za-z\s\-.]", "", cleaned)
        if not no_sep:
            return "".join(tokens).title()
    return None


def _v47_last_reply_asked_street_spelling(conv: dict) -> bool:
    sched = conv.setdefault("sched", {})
    last = str(conv.get("last_voice_reply") or sched.get("last_voice_reply") or "")
    low = _v47_low(last)
    return "repeat just the street" in low or "spell it" in low or "couldn t verify that address" in low or "couldn't verify that address" in low


def _v47_handle_street_spelling_reply(conv: dict, caller_text: str) -> str | None:
    if not _v47_last_reply_asked_street_spelling(conv):
        return None
    sched = conv.setdefault("sched", {})
    word = _v47_spelled_word(caller_text)
    if not word:
        # Maybe caller repeated the full street line instead of spelling.
        line, town, state = _v47_extract_address_candidate(caller_text)
    else:
        old = str(sched.get("raw_address") or sched.get("address_candidate") or "")
        m_num = re.search(r"\b(\d{1,6})\b", old)
        number = m_num.group(1) if m_num else None
        m_type = re.search(r"\b(ave|avenue|st|street|rd|road|dr|drive|ln|lane|ct|court|blvd|boulevard|way|pl|place)\b", old, flags=re.I)
        typ = m_type.group(1) if m_type else "Ave"
        typ = {"avenue":"Ave","ave":"Ave","street":"St","st":"St","road":"Rd","rd":"Rd","drive":"Dr","dr":"Dr","lane":"Ln","ln":"Ln","court":"Ct","ct":"Ct","boulevard":"Blvd","blvd":"Blvd","place":"Pl","pl":"Pl","way":"Way"}.get(typ.lower(), typ.title())
        town = None
        state = None
        # Pull town/state from old candidate or captured hint.
        old_clean = old.replace(", in ", ", ")
        parts = [p.strip() for p in old_clean.split(",") if p.strip()]
        if len(parts) >= 2:
            town = parts[1]
        town = town or sched.get("voice_town_hint") or sched.get("town_hint")
        state = _v47_state_from_text(old) or (_v47_known_ct_town(town or "") if town else None)
        line = f"{number} {word} {typ}" if number else None
    if not line:
        return None
    raw = ", ".join([p for p in [line, town, state] if p])
    sched["raw_address"] = raw
    sched["address_candidate"] = raw
    if not _v47_apply_normalized_if_possible(conv, raw):
        sched["address_verified"] = False
        sched["address_missing"] = "confirm"
        conv["last_voice_reply"] = "I still couldn’t verify that address. Please say the full address, including the town and state."
        return conv["last_voice_reply"]
    # Address is good now; continue to correct price gate.
    if sched.get("voice_last_intent_power_loss") or _v47_power_loss_troubleshoot_text(str(sched.get("last_customer_issue") or "")):
        try:
            reply = _v45_troubleshoot_price_gate(conv, caller_text)
        except Exception:
            reply = "The address is verified in our system. Troubleshoot and repair visits are $395. That covers sending one of our electricians out to diagnose the issue and repair it on site when possible. Does that work for you?"
    else:
        try:
            reply = _v46_offer_eval_slots(conv) if sched.get("eval_price_accepted") else "The address is verified in our system. We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step. Does that work for you?"
            sched["awaiting_eval_price_confirm"] = True
        except Exception:
            reply = "The address is verified in our system. We start with a $195 on-site evaluation visit. Does that work for you?"
    conv["last_voice_reply"] = reply
    return reply


def _v47_custom_date_request_during_offers(conv: dict, caller_text: str) -> str | None:
    sched = conv.setdefault("sched", {})
    if not (sched.get("awaiting_slot_offer_choice") and (sched.get("offered_slot_options") or [])):
        return None
    requested_date = None
    try:
        requested_date = salvage_relative_date_from_text(caller_text)
    except Exception:
        requested_date = None
    if not requested_date:
        return None
    option_dates = {str(o.get("date") or "").strip() for o in (sched.get("offered_slot_options") or []) if o.get("date")}
    if requested_date in option_dates:
        return None
    # This is a custom date, not an offered-slot selection. Respect it.
    explicit_time = None
    try:
        explicit_time = extract_explicit_time_from_text(caller_text)
    except Exception:
        explicit_time = None
    appt = (sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195").upper()
    sched["appointment_type"] = appt
    conv["appointment_type"] = appt
    sched["scheduled_date"] = requested_date
    sched["scheduled_time"] = explicit_time
    if explicit_time:
        sched["scheduled_time_source"] = "customer_explicit_custom_date_after_offer_v47"
    else:
        sched.pop("scheduled_time_source", None)
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None
    sched["booking_created"] = False
    sched["square_booking_id"] = None
    sched["slot_choice_locked"] = False
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    try:
        recompute_pending_step(conv.setdefault("profile", {}), sched)
    except Exception:
        pass
    if not explicit_time:
        sched["pending_step"] = "need_time"
        reply = f"Got it. What time works for {_date_label_for_sms(requested_date)}?"
    else:
        # Move to identity/email collection without validating against offered fallback slots.
        try:
            reply = _v44_after_slot_prompt(conv)
        except Exception:
            reply = "What is your last name?"
    conv["last_voice_reply"] = reply
    return reply


def _v47_offer_trouble_slots(conv: dict) -> str:
    try:
        return _v45_offer_troubleshoot_slots(conv)
    except Exception:
        sched = conv.setdefault("sched", {})
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
        sched["awaiting_troubleshoot_price_confirm"] = False
        sched["troubleshoot_price_accepted"] = True
        return _v42_slot_offer_reply_for(conv, "TROUBLESHOOT_395") if "_v42_slot_offer_reply_for" in globals() else "Great. What day and time works best?"


def _v47_email_confirm_yes_safe(conv: dict, phone: str) -> dict | None:
    try:
        return _v46_email_confirm_yes(conv, phone)
    except Exception:
        return None


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()

    # 1) Street spelling correction must run before old layers ignore it.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        reply = _v47_handle_street_spelling_reply(conv_pre, caller_text)
        if reply:
            try:
                log_event("VOICE_V47_STREET_SPELLING_REPAIRED", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv_pre)
            except Exception:
                pass
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": conv_pre.setdefault("sched", {}).get("pending_step"), "appointment_type": conv_pre.setdefault("sched", {}).get("appointment_type") or conv_pre.get("appointment_type"), "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V47_STREET_SPELLING_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    # 2) Email-confirm yes must never repeat the same email confirmation.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        sched_pre = conv_pre.setdefault("sched", {})
        last_reply_pre = str(conv_pre.get("last_voice_reply") or sched_pre.get("last_voice_reply") or "")
        if (_v46_is_email_confirm_prompt(last_reply_pre) or sched_pre.get("voice_awaiting_email_confirm")) and _v47_yes(caller_text):
            out = _v47_email_confirm_yes_safe(conv_pre, p)
            if out:
                # If lower helper still returns an email confirmation prompt, replace it with booking progress/finalization.
                if _v46_is_email_confirm_prompt(str(out.get("reply_to_customer") or "")):
                    first, last = _v46_name_status(conv_pre)
                    if first and not last:
                        out["reply_to_customer"] = _v46_require_last_name_reply(conv_pre)
                    elif not (sched_pre.get("scheduled_date") and sched_pre.get("scheduled_time")):
                        out["reply_to_customer"] = _v42_slot_offer_reply_for(conv_pre, sched_pre.get("appointment_type") or conv_pre.get("appointment_type") or "EVAL_195") if "_v42_slot_offer_reply_for" in globals() else "Thanks. What day and time works best?"
                    else:
                        out["reply_to_customer"] = "Thanks. I’ll finish booking that now."
                    out["booking_created"] = False
                    out["end_call"] = False
                try:
                    log_event("VOICE_V47_EMAIL_CONFIRM_YES", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(out.get("reply_to_customer")), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return out
    except Exception as e:
        try:
            log_event("VOICE_V47_EMAIL_CONFIRM_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    # 3) Price gate yes for troubleshoot must offer slots, before old layers can repeat price.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        sched_pre = conv_pre.setdefault("sched", {})
        last_reply_pre = str(conv_pre.get("last_voice_reply") or sched_pre.get("last_voice_reply") or "")
        awaiting_trouble_price = bool(sched_pre.get("awaiting_troubleshoot_price_confirm")) or ("395" in _v47_low(last_reply_pre) and "does that work" in _v47_low(last_reply_pre) and "troubleshoot" in _v47_low(last_reply_pre))
        if awaiting_trouble_price:
            if _v47_no(caller_text):
                reply = _v45_decline_troubleshoot(conv_pre) if "_v45_decline_troubleshoot" in globals() else "No problem. We will not schedule the visit right now. Thank you for calling Prevolt Electric. Goodbye."
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "TROUBLESHOOT_395", "end_call": True}
            if _v47_yes(caller_text) or (_v42_availability_request_text(caller_text) if "_v42_availability_request_text" in globals() else False):
                reply = _v47_offer_trouble_slots(conv_pre)
                try:
                    log_event("VOICE_V47_TROUBLE_PRICE_ACCEPTED", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": "TROUBLESHOOT_395", "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V47_TROUBLE_PRICE_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    # 4) Custom date/time after offered slots must be respected before offered-slot matching.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        reply = _v47_custom_date_request_during_offers(conv_pre, caller_text)
        if reply:
            try:
                log_event("VOICE_V47_CUSTOM_DATE_AFTER_OFFER", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv_pre)
            except Exception:
                pass
            return {"reply_to_customer": reply, "booking_created": False, "manual_only": False, "pending_step": conv_pre.setdefault("sched", {}).get("pending_step"), "appointment_type": conv_pre.setdefault("sched", {}).get("appointment_type") or conv_pre.get("appointment_type"), "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V47_CUSTOM_DATE_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V47(phone, call_sid, caller_text)

    # 5) Post-process guard rails.
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        reply = str(out.get("reply_to_customer") or "")
        low_reply = _v47_low(reply)

        # Remember issue type for later spelling/address correction turns.
        if _v47_power_loss_troubleshoot_text(caller_text):
            sched["voice_last_intent_power_loss"] = True
            sched["last_customer_issue"] = caller_text

        # If address is falsely verified or raw, repair it now or block the price gate.
        addr_block_reply = _v47_repair_or_block_address(conv, caller_text)
        if addr_block_reply:
            out["reply_to_customer"] = addr_block_reply
            out["booking_created"] = False
            out["end_call"] = False
            out["pending_step"] = sched.get("pending_step") or "need_address"
            conv["last_voice_reply"] = addr_block_reply
            try:
                log_event("VOICE_V47_BLOCKED_UNSTRUCTURED_ADDRESS", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(addr_block_reply), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return out

        # If the customer described dead/nonworking outlets, force scheduled $395 troubleshoot unless hard hazard.
        # Address must be collected and verified before price/date flow.
        if _v47_power_loss_troubleshoot_text(caller_text) and not _v47_hard_hazard(caller_text):
            sched["voice_last_intent_power_loss"] = True
            sched["last_customer_issue"] = caller_text
            sched["appointment_type"] = "TROUBLESHOOT_395"
            conv["appointment_type"] = "TROUBLESHOOT_395"
            sched["awaiting_emergency_confirm"] = False
            sched["emergency_approved"] = False
            sched["hard_emergency_detected"] = False

            if not sched.get("address_verified"):
                sched["raw_address"] = ""
                sched["address_candidate"] = ""
                sched["address_verified"] = False
                sched["address_missing"] = "street"
                sched["pending_step"] = "need_address"
                sched["state"] = "waiting_for_address"
                sched["awaiting_slot_offer_choice"] = False
                sched["offered_slot_options"] = []
                sched["awaiting_troubleshoot_price_confirm"] = False
                sched["troubleshoot_price_accepted"] = False
                sched["awaiting_eval_price_confirm"] = False
                reply2 = "Got it. What is the full address for the work, including the town and state?"
                out["reply_to_customer"] = _voice_naturalize_reply(reply2) if "_voice_naturalize_reply" in globals() else reply2
                out["booking_created"] = False
                out["manual_only"] = False
                out["appointment_type"] = "TROUBLESHOOT_395"
                out["pending_step"] = "need_address"
                out["end_call"] = False
                conv["last_voice_reply"] = out["reply_to_customer"]
                try:
                    log_event("VOICE_V47_POWER_LOSS_NEEDS_ADDRESS", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(out["reply_to_customer"]), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return out

            already_price = "395" in low_reply and "troubleshoot" in low_reply
            already_slot = bool(sched.get("awaiting_slot_offer_choice")) and ("which one" in low_reply or "works best" in low_reply)
            if not sched.get("troubleshoot_price_accepted") and not already_price and not already_slot:
                reply2 = _v45_troubleshoot_price_gate(conv, caller_text) if "_v45_troubleshoot_price_gate" in globals() else "The address is verified in our system. Troubleshoot and repair visits are $395. That covers sending one of our electricians out to diagnose the issue and repair it on site when possible. Does that work for you?"
                out["reply_to_customer"] = _voice_naturalize_reply(reply2) if "_voice_naturalize_reply" in globals() else reply2
                out["booking_created"] = False
                out["manual_only"] = False
                out["appointment_type"] = "TROUBLESHOOT_395"
                out["pending_step"] = sched.get("pending_step")
                out["end_call"] = False
                try:
                    log_event("VOICE_V47_POWER_LOSS_FORCED_395", p, {"caller_text": _safe_monitor_text(caller_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(out["reply_to_customer"]), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return out

        # If older layers still asked a generic scheduling question for troubleshoot, replace it.
        # If address is missing, ask address first; if address is verified, move to the $395 gate.
        if _v47_power_loss_troubleshoot_text(caller_text) and ("what day and time" in low_reply or "what time works" in low_reply or "get the scheduling" in low_reply):
            if not sched.get("address_verified"):
                sched["raw_address"] = ""
                sched["address_candidate"] = ""
                sched["address_verified"] = False
                sched["address_missing"] = "street"
                sched["pending_step"] = "need_address"
                sched["state"] = "waiting_for_address"
                sched["awaiting_slot_offer_choice"] = False
                sched["offered_slot_options"] = []
                reply2 = "Got it. What is the full address for the work, including the town and state?"
            else:
                reply2 = _v45_troubleshoot_price_gate(conv, caller_text) if "_v45_troubleshoot_price_gate" in globals() else "Troubleshoot and repair visits are $395. Does that work for you?"
            out["reply_to_customer"] = reply2
            out["booking_created"] = False
            out["end_call"] = False
            out["appointment_type"] = "TROUBLESHOOT_395"
            out["pending_step"] = sched.get("pending_step")
            return out

        # If the customer said yes to email confirm and old layer repeated it, suppress it.
        if _v47_yes(caller_text) and _v46_is_email_confirm_prompt(str(out.get("reply_to_customer") or "")):
            if sched.get("scheduled_date") and sched.get("scheduled_time"):
                out["reply_to_customer"] = "Thanks. I’ll finish booking that now."
                out["booking_created"] = False
                out["end_call"] = False
                conv["last_voice_reply"] = out["reply_to_customer"]
                return out
    except Exception as e:
        try:
            log_event("VOICE_V47_POST_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# =============================
# v48 STABILIZATION HOTFIX LAYER
# =============================
# - Keep raw_address synchronized with a complete Google-normalized address.
# - Remove "emergency dispatch" wording from normal scheduled troubleshoot bookings.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V48 = process_prevolt_voice_turn


def _v48_sync_raw_address_from_normalized(conv: dict) -> None:
    sched = conv.setdefault("sched", {})
    addr = sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None
    try:
        complete = _v47_complete_addr_struct(addr)
    except Exception:
        complete = False
    if not complete:
        return
    raw = (
        f"{addr.get('address_line_1')}, {addr.get('locality')}, "
        f"{addr.get('administrative_district_level_1')} {addr.get('postal_code')}"
    ).strip(" ,")
    if raw:
        sched["raw_address"] = raw
        sched["address_candidate"] = raw
        sched["address_verified"] = True
        sched["address_missing"] = None
        sched["address_parts"] = {
            "street": True,
            "number": True,
            "city": True,
            "state": True,
            "zip": True,
            "source": "v48_sync_normalized",
        }


def _v48_is_true_emergency_context(conv: dict, caller_text: str = "") -> bool:
    sched = conv.setdefault("sched", {})
    if sched.get("emergency_approved") or sched.get("awaiting_emergency_confirm") or sched.get("hard_emergency_detected"):
        return True
    if _v47_hard_hazard(caller_text or sched.get("last_customer_issue") or sched.get("voice_last_caller_text_norm") or ""):
        return True
    return False


def _v48_clean_final_reply(conv: dict, reply: str, caller_text: str = "") -> str:
    if not reply:
        return reply
    sched = conv.setdefault("sched", {})
    appt = str(sched.get("appointment_type") or conv.get("appointment_type") or "").upper()
    if "TROUBLESHOOT" in appt and not _v48_is_true_emergency_context(conv, caller_text):
        # Avoid telling a normal scheduled troubleshoot customer they are booked for emergency dispatch.
        reply = re.sub(r"\s*for\s+emergency\s+dispatch\s+at\s+", " for ", reply, flags=re.I)
        reply = re.sub(r"\s*for\s+emergency\s+dispatch\b", "", reply, flags=re.I)
        reply = re.sub(r"\bemergency\s+dispatch\s+visit\b", "troubleshoot and repair visit", reply, flags=re.I)
        reply = re.sub(r"\s+", " ", reply).strip()
    return reply


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V48(phone, call_sid, caller_text)
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        _v48_sync_raw_address_from_normalized(conv)
        reply = str(out.get("reply_to_customer") or "")
        cleaned = _v48_clean_final_reply(conv, reply, caller_text)
        if cleaned != reply:
            out["reply_to_customer"] = cleaned
            conv["last_voice_reply"] = cleaned
            try:
                log_event("VOICE_V48_CLEANED_FINAL_REPLY", p, {"old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(cleaned), "call_sid": call_sid}, conv)
            except Exception:
                pass
    except Exception as e:
        try:
            log_event("VOICE_V48_POST_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out

# =============================
# v49 ADDRESS ASR HOTFIX LAYER
# =============================
# - Correct common voice-ASR street-name mistakes before address normalization.
# - Specific live failure: "Dickerman" was heard/stored as "Tickerman" and the
#   system looped even after caller gave town/state.
# - This is intentionally narrow: only corrects close Dickerman variants when
#   the text is address-like or tied to Windsor Locks / Windsor context.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V49 = process_prevolt_voice_turn


def _v49_low(text: str) -> str:
    try:
        return (text or "").lower().strip()
    except Exception:
        return ""


def _v49_apply_dickerman_asr_lexicon(text: str) -> tuple[str, bool]:
    """Return text with safe Dickerman ASR repairs applied.

    The correction is only applied when the input looks address-related. This
    avoids globally rewriting random words, but catches phone-ASR variants like:
      - Tickerman Ave
      - Ticker man Ave
      - Tikerman Avenue
    """
    original = text or ""
    if not original:
        return original, False
    low = _v49_low(original)
    address_like = bool(re.search(r"\b\d{1,6}\b", original)) or bool(re.search(r"\b(?:ave|avenue|st|street|road|rd|drive|dr|ln|lane)\b", low))
    local_context = bool(re.search(r"\b(?:windsor locks|windsor|connecticut|ct)\b", low))
    has_bad_variant = bool(re.search(r"\b(?:tickerman|tikerman|tikkerman|ticker\s+man|tiker\s+man|dicker\s+man)\b", low))
    if not has_bad_variant:
        return original, False
    if not (address_like or local_context):
        return original, False
    fixed = re.sub(r"\b[Tt]ickerman\b", "Dickerman", original)
    fixed = re.sub(r"\b[Tt]ikerman\b", "Dickerman", fixed)
    fixed = re.sub(r"\b[Tt]ikkerman\b", "Dickerman", fixed)
    fixed = re.sub(r"\b[Tt]icker\s+[Mm]an\b", "Dickerman", fixed)
    fixed = re.sub(r"\b[Tt]iker\s+[Mm]an\b", "Dickerman", fixed)
    fixed = re.sub(r"\b[Dd]icker\s+[Mm]an\b", "Dickerman", fixed)
    return fixed, fixed != original


def _v49_repair_sched_dickerman(conv: dict) -> bool:
    sched = conv.setdefault("sched", {})
    changed = False
    for key in ("raw_address", "address_candidate", "normalized_address_text"):
        val = sched.get(key)
        if isinstance(val, str):
            fixed, did = _v49_apply_dickerman_asr_lexicon(val)
            if did:
                sched[key] = fixed
                changed = True
    # Do not blindly mark verified. Just clear the bad loop state so Google can
    # re-verify the corrected value.
    if changed and not sched.get("address_verified"):
        sched["address_missing"] = "confirm"
        sched["address_candidate"] = sched.get("raw_address") or sched.get("address_candidate")
    return changed


def _v49_build_corrected_dickerman_candidate(conv: dict, caller_text: str) -> str | None:
    """Build a usable corrected address candidate from caller text or sched."""
    sched = conv.setdefault("sched", {})
    fixed_text, did_text = _v49_apply_dickerman_asr_lexicon(caller_text or "")
    pool = [fixed_text, sched.get("raw_address"), sched.get("address_candidate")]
    for item in pool:
        if not isinstance(item, str) or "dickerman" not in item.lower():
            continue
        # Prefer the explicit house number supplied by the caller/sched.
        m_num = re.search(r"\b(\d{1,6})\b", item)
        number = m_num.group(1) if m_num else ""
        town = "Windsor Locks" if re.search(r"\bwindsor\s+locks\b", item, flags=re.I) else None
        if not town:
            # If the caller only says Dickerman Ave and the business origin is Windsor Locks,
            # use that as a safe town hint for this known local street.
            origin = str(os.environ.get("VOICE_DISPATCH_ORIGIN_ADDRESS") or os.environ.get("DISPATCH_ORIGIN_ADDRESS") or "")
            if "dickerman" in origin.lower() and "windsor locks" in origin.lower():
                town = "Windsor Locks"
        state = "CT" if re.search(r"\b(?:ct|connecticut)\b", item, flags=re.I) else None
        if not state:
            origin = str(os.environ.get("VOICE_DISPATCH_ORIGIN_ADDRESS") or os.environ.get("DISPATCH_ORIGIN_ADDRESS") or "")
            if "ct" in origin.lower() or "connecticut" in origin.lower() or "windsor locks" in origin.lower():
                state = "CT"
        if number and town and state:
            return f"{number} Dickerman Ave, {town}, {state}"
        if number:
            return f"{number} Dickerman Ave"
    return None


def _v49_issue_needs_troubleshoot(conv: dict, caller_text: str = "") -> bool:
    sched = conv.setdefault("sched", {})
    issue_blob = " ".join([
        str(caller_text or ""),
        str(sched.get("last_customer_issue") or ""),
        str(sched.get("voice_last_caller_text_norm") or ""),
    ])
    if sched.get("voice_last_intent_power_loss"):
        return True
    try:
        return bool(_v47_power_loss_troubleshoot_text(issue_blob))
    except Exception:
        low = _v49_low(issue_blob)
        return bool(re.search(r"\b(outlet|outlets|power|lights?)\b", low) and re.search(r"\b(not working|stopped working|no power|dead|troubleshoot|troubleshooting)\b", low))


def _v49_after_address_verified_reply(conv: dict, caller_text: str) -> str:
    sched = conv.setdefault("sched", {})
    if _v49_issue_needs_troubleshoot(conv, caller_text) and not (_v47_hard_hazard(caller_text) if "_v47_hard_hazard" in globals() else False):
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
        sched["awaiting_troubleshoot_price_confirm"] = True
        sched["awaiting_eval_price_confirm"] = False
        try:
            return _v45_troubleshoot_price_gate(conv, caller_text)
        except Exception:
            return "The address is verified in our system. Troubleshoot and repair visits are $395. That covers sending one of our electricians out to diagnose the issue and repair it on site when possible. Does that work for you?"
    sched["appointment_type"] = sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195"
    conv["appointment_type"] = sched["appointment_type"]
    sched["awaiting_eval_price_confirm"] = True
    return "The address is verified in our system. We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step. Does that work for you?"


def _v49_try_verify_corrected_dickerman(conv: dict, caller_text: str) -> str | None:
    """If Dickerman was misheard as Tickerman, correct + verify before looping."""
    sched = conv.setdefault("sched", {})
    candidate = _v49_build_corrected_dickerman_candidate(conv, caller_text)
    if not candidate:
        return None
    sched["raw_address"] = candidate
    sched["address_candidate"] = candidate
    try:
        ok = _v47_apply_normalized_if_possible(conv, candidate)
    except Exception:
        ok = False
    if not ok:
        # Last local fallback: if caller supplied number + Windsor Locks + Dickerman,
        # stop saying Tickerman and ask only for a confirmation/spelling of the street.
        sched["address_verified"] = False
        sched["address_missing"] = "confirm"
        return "I couldn’t verify that address yet. Please say the full address, including the town and state."
    try:
        _v48_sync_raw_address_from_normalized(conv)
    except Exception:
        pass
    sched["address_verified"] = True
    sched["address_missing"] = None
    return _v49_after_address_verified_reply(conv, caller_text)


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    fixed_text, did_fix = _v49_apply_dickerman_asr_lexicon(caller_text or "")
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        sched_pre = conv_pre.setdefault("sched", {})
        sched_pre["voice_last_caller_text_norm"] = fixed_text
        sched_changed = _v49_repair_sched_dickerman(conv_pre)
        # If the current turn is clearly a full corrected address reply, handle it now
        # before older layers can repeat the Tickerman loop.
        if did_fix or sched_changed or "dickerman" in _v49_low(fixed_text):
            immediate = _v49_try_verify_corrected_dickerman(conv_pre, fixed_text)
            if immediate and ("couldn’t verify" not in _v49_low(immediate) or "tickerman" in _v49_low(caller_text or "")):
                try:
                    log_event("VOICE_V49_DICKERMAN_ASR_PRE", p, {"caller_text": _safe_monitor_text(caller_text), "fixed_text": _safe_monitor_text(fixed_text), "reply": _safe_monitor_text(immediate), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return {"reply_to_customer": immediate, "booking_created": False, "manual_only": False, "pending_step": sched_pre.get("pending_step"), "appointment_type": sched_pre.get("appointment_type") or conv_pre.get("appointment_type"), "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V49_DICKERMAN_PRE_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V49(phone, call_sid, fixed_text if did_fix else caller_text)

    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        _v49_repair_sched_dickerman(conv)
        reply = str(out.get("reply_to_customer") or "")
        reply_fixed, reply_changed = _v49_apply_dickerman_asr_lexicon(reply)
        if reply_changed:
            out["reply_to_customer"] = reply_fixed
            conv["last_voice_reply"] = reply_fixed
            reply = reply_fixed
        # If old layer still produced the failed verification loop, override it.
        if re.search(r"couldn[’']t verify", reply, flags=re.I) and ("dickerman" in _v49_low(fixed_text) or "tickerman" in _v49_low(reply)):
            repaired = _v49_try_verify_corrected_dickerman(conv, fixed_text)
            if repaired:
                out["reply_to_customer"] = repaired
                out["booking_created"] = False
                out["end_call"] = False
                out["pending_step"] = sched.get("pending_step")
                out["appointment_type"] = sched.get("appointment_type") or conv.get("appointment_type")
                try:
                    log_event("VOICE_V49_DICKERMAN_ASR_POST", p, {"caller_text": _safe_monitor_text(caller_text), "fixed_text": _safe_monitor_text(fixed_text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(repaired), "call_sid": call_sid}, conv)
                except Exception:
                    pass
        # Never allow the assistant to speak the bad ASR street back to the caller.
        if "tickerman" in _v49_low(str(out.get("reply_to_customer") or "")):
            out["reply_to_customer"] = re.sub(r"\b[Tt]ickerman\b", "Dickerman", str(out.get("reply_to_customer") or ""))
            conv["last_voice_reply"] = out["reply_to_customer"]
    except Exception as e:
        try:
            log_event("VOICE_V49_POST_LAYER_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# ===================================================
# Voice stabilization v50 — explicit town/state address correction
# ===================================================
# Fixes the "1 Main Street" + "Windsor, Connecticut" failure where Google/older
# layers kept a previously guessed town such as Danbury and then closed the call
# as out of area. Customer-provided town/state now overrides a prior ambiguous
# street-only geocode before any out-of-area decision is allowed.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V50 = process_prevolt_voice_turn


def _v50_low(value: str) -> str:
    return " ".join(str(value or "").lower().replace("’", "'").split())


def _v50_state_code(value: str | None) -> str | None:
    s = _v50_low(value or "")
    if re.search(r"\b(?:ct|c\.t\.|connecticut)\b", s, flags=re.I):
        return "CT"
    if re.search(r"\b(?:ma|m\.a\.|massachusetts|mass\.)\b", s, flags=re.I):
        return "MA"
    return None


def _v50_clean_town(value: str | None) -> str | None:
    raw = " ".join(str(value or "").replace("\n", " ").split()).strip(" ,.")
    if not raw:
        return None
    raw = re.sub(r"^(?:it'?s|it is|that'?s|that is|the town is|town is|in|at)\s+", "", raw, flags=re.I).strip(" ,.")
    raw = re.sub(r"\b(?:ct|c\.t\.|connecticut|ma|m\.a\.|massachusetts|mass\.)\b", "", raw, flags=re.I).strip(" ,.")
    raw = re.sub(r"\b(?:please|thanks|thank you|yes|yeah|yep|okay|ok)\b", "", raw, flags=re.I).strip(" ,.")
    # Keep only a sane town-length phrase.
    parts = raw.split()
    if not parts or len(parts) > 4:
        return None
    return " ".join(p.capitalize() if not p.isupper() else p for p in parts)


def _v50_extract_town_state(text: str) -> tuple[str | None, str | None]:
    """Extract explicit town/state from turns like 'Windsor, Connecticut'."""
    raw = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not raw:
        return (None, None)
    state = _v50_state_code(raw)
    if not state:
        return (None, None)
    # Prefer the phrase immediately before the state word.
    m = re.search(
        r"(?P<town>[A-Za-z][A-Za-z .'-]{1,45}?)\s*,?\s*(?:CT|C\.T\.|Connecticut|MA|M\.A\.|Massachusetts|Mass\.)\b",
        raw,
        flags=re.I,
    )
    if not m:
        return (None, state)
    town = _v50_clean_town(m.group("town"))
    return (town, state)


def _v50_canon_town(value: str | None) -> str:
    s = _v50_low(value or "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _v50_known_local_town(value: str | None, state: str | None = None) -> bool:
    town = _v50_canon_town(value)
    st = (state or "").upper()
    if not town:
        return False
    if town in {"worcester", "worcestor", "boston", "malden"}:
        return False
    # Core CT/MA radius towns Kyle commonly services/tests. This is not a substitute
    # for Google distance; it only prevents a bad geocode from turning Windsor into Danbury.
    local_ct = {
        "windsor", "windsor locks", "east windsor", "south windsor", "suffield", "enfield",
        "east granby", "granby", "simsbury", "avon", "canton", "bloomfield", "hartford",
        "west hartford", "east hartford", "manchester", "vernon", "ellington", "somers",
        "tolland", "farmington", "plainville", "bristol", "newington", "wethersfield",
        "glastonbury", "rocky hill", "cromwell", "berlin", "new britain", "west suffield",
    }
    local_ma = {"longmeadow", "east longmeadow", "springfield", "west springfield", "agawam", "feeding hills", "ludlow", "chicopee", "holyoke", "palmer", "wilbraham"}
    if st == "CT":
        return town in local_ct
    if st == "MA":
        return town in local_ma
    return town in local_ct or town in local_ma


def _v50_street_line_from_sched_or_text(conv: dict, text: str = "") -> str | None:
    sched = conv.setdefault("sched", {})
    pool = [text, sched.get("raw_address"), sched.get("address_candidate")]
    norm = sched.get("normalized_address")
    if isinstance(norm, dict) and norm.get("address_line_1"):
        pool.append(str(norm.get("address_line_1")))
    for item in pool:
        if not isinstance(item, str) or not item.strip():
            continue
        sample = item.strip().replace(", in ", ", ")
        try:
            line, _town, _state = _v47_extract_address_candidate(sample)
            if line and _address_has_house_number_and_street(line):
                return line
        except Exception:
            pass
        # Fallback: take the address-looking first segment.
        first = sample.split(",")[0].strip()
        m = re.search(
            r"\b\d{1,6}\s+[A-Za-z0-9.'#\- ]+?\s+(?:ave(?:nue)?|st(?:reet)?|rd|road|dr(?:ive)?|ln|lane|ct|court|blvd|boulevard|way|pl|place|cir|circle|ter|terrace)\b",
            first,
            flags=re.I,
        )
        if m:
            try:
                line, _town, _state = _v47_extract_address_candidate(m.group(0))
                return line or m.group(0).title()
            except Exception:
                return m.group(0).title()
    return None


def _v50_locality_matches(addr: dict | None, town: str | None, state: str | None) -> bool:
    if not isinstance(addr, dict) or not town:
        return False
    a_town = _v50_canon_town(addr.get("locality"))
    want = _v50_canon_town(town)
    a_state = str(addr.get("administrative_district_level_1") or "").upper()
    if state and a_state and a_state != state.upper():
        return False
    return bool(a_town and want and a_town == want)


def _v50_apply_explicit_customer_address(conv: dict, line: str, town: str, state: str, source: str = "customer_explicit_town_state_v50") -> None:
    """Apply an explicit customer-provided address without letting old Google guesses win.

    v57: A town/state correction from the caller must also reopen a stale
    residential_out_of_area closure. This prevents the Greenwich/Danbury-style
    Google guess from staying locked after the customer corrects the town to a
    valid CT/MA town such as Ludlow, Massachusetts.
    """
    sched = conv.setdefault("sched", {})
    raw = f"{line}, {town}, {state}".strip(" ,")

    # If a previous Google guess closed the thread as out-of-area, a later
    # explicit customer correction wins. Reopen scheduling before applying the
    # corrected address, then let the distance gate evaluate the corrected town.
    if (str(sched.get("closed_reason") or "").startswith("residential_out_of_area")
            or str(sched.get("state") or "").lower() == "closed"
            or str(conv.get("thread_type") or "") == "closed_lost"):
        sched["booking_allowed"] = True
        sched["closed_reason"] = None
        sched["manual_only"] = False
        sched["voice_close_after_reply"] = False
        sched["voice_booking_completed_close"] = False
        sched["voice_resume_sms_sent"] = False
        conv["thread_type"] = None
        if str(sched.get("state") or "").lower() == "closed":
            sched["state"] = "active"

    addr = None
    try:
        result = normalize_address(raw, forced_state=state)
        if isinstance(result, tuple) and len(result) >= 2 and result[0] == "ok" and _v50_locality_matches(result[1], town, state):
            addr = result[1]
    except Exception:
        addr = None
    if isinstance(addr, dict):
        _v38_apply_normalized_address(conv, addr, source)
    else:
        # If Google returns another town (for example Danbury) for a generic street like
        # 1 Main Street, do NOT accept that guessed town. Keep the exact customer town.
        sched["normalized_address"] = {
            "address_line_1": line,
            "locality": town,
            "administrative_district_level_1": state,
            "postal_code": "",
            "country": "US",
            "verification_source": source + "_customer_supplied",
        }
        sched["raw_address"] = raw
        sched["address_candidate"] = raw
        sched["address_verified"] = True
        sched["address_missing"] = None
        sched["address_parts"] = {"street": True, "number": True, "city": True, "state": True, "zip": False, "source": source}
    if sched.get("pending_step") == "need_address":
        sched["pending_step"] = None
    conv["appointment_type"] = sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195"


def _v50_reply_after_explicit_address(conv: dict, caller_text: str) -> str:
    sched = conv.setdefault("sched", {})
    # Enforce far-away explicit towns only after the corrected town/state has been applied.
    if _voice_is_out_of_area_town(_voice_destination_from_sched(sched)):
        return _voice_out_of_area_reply()
    try:
        too_far, travel_minutes = _voice_residential_address_too_far(conv)
    except Exception:
        too_far, travel_minutes = False, None
    if too_far:
        return _voice_out_of_area_reply()

    # Integrated emergency guard:
    # If v51 or an earlier voice hazard layer has already marked this call as a hard emergency,
    # verifying the address must lead to the $395 emergency dispatch prompt, not the normal
    # $195 evaluation gate and not normal next-three slot offers.
    appt_existing = str(sched.get("appointment_type") or conv.get("appointment_type") or "").upper()
    state_existing = str(sched.get("state") or "").lower()
    if (
        sched.get("hard_emergency_detected")
        or sched.get("awaiting_emergency_confirm")
        or sched.get("emergency_approved")
    ):
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
        sched["hard_emergency_detected"] = True
        sched["state"] = "emergency"
        sched["price_disclosed"] = True
        sched["awaiting_eval_price_confirm"] = False
        sched["awaiting_troubleshoot_price_confirm"] = False
        sched["awaiting_slot_offer_choice"] = False
        sched["offered_slot_options"] = []
        if not sched.get("emergency_approved"):
            sched["awaiting_emergency_confirm"] = True
            sched["pending_step"] = "need_date"
            return "The address is verified in our system. This sounds urgent. We can send someone now, and arrival is usually within one to two hours. The emergency troubleshoot and repair visit is $395. Do you want us to dispatch someone now?"
        return "The address is verified in our system. We have this marked as an emergency dispatch."

    # EV/install/evaluation stays $195; power loss stays $395.
    if _v49_issue_needs_troubleshoot(conv, caller_text):
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
        sched["awaiting_troubleshoot_price_confirm"] = True
        sched["awaiting_eval_price_confirm"] = False
        try:
            return _v45_troubleshoot_price_gate(conv, caller_text)
        except Exception:
            return "The address is verified in our system. Troubleshoot and repair visits are $395. That covers sending one of our electricians out to diagnose the issue and repair it on site when possible. Does that work for you?"
    sched["appointment_type"] = sched.get("appointment_type") or conv.get("appointment_type") or "EVAL_195"
    conv["appointment_type"] = sched["appointment_type"]
    sched["awaiting_eval_price_confirm"] = True
    sched["awaiting_troubleshoot_price_confirm"] = False
    return "The address is verified in our system. We start with a $195 on-site evaluation visit. That covers sending one of our electricians out, reviewing the work in person, checking what is needed, and putting together the next step. Does that work for you?"



def _v50_customer_town_state_override(conv: dict, caller_text: str) -> str | None:
    """When caller supplies town/state after a street-only address, override stale/guessed town.

    v57: Do not require the town to be in the small hard-coded local-town list.
    If the caller gives an explicit CT/MA town/state correction, apply that exact
    town first and then run the normal distance/service-area gate. This is the
    fix for calls like: "40 Arch Street, above" -> "Ludlow, Massachusetts",
    where Google guessed "40 Arch Street, Greenwich, CT" before the caller
    corrected the town.
    """
    town, state = _v50_extract_town_state(caller_text)
    if not (town and state):
        return None
    line = _v50_street_line_from_sched_or_text(conv, caller_text)
    if not line:
        return None

    # Explicit no-go towns still close, but close only after using the caller's
    # explicit town, never a stale guessed town.
    if _voice_is_out_of_area_town(town):
        _v50_apply_explicit_customer_address(conv, line, town, state, "town_state_explicit_far_v57")
        return _voice_out_of_area_reply()

    # Any explicit CT/MA town/state correction wins over a stale Google guess.
    # The corrected address still goes through _voice_residential_address_too_far
    # inside _v50_reply_after_explicit_address.
    if state in {"CT", "MA"}:
        _v50_apply_explicit_customer_address(conv, line, town, state, "town_state_override_v57")
        return _v50_reply_after_explicit_address(conv, caller_text)

    # Non-CT/MA towns are outside the residential booking lane.
    _v50_apply_explicit_customer_address(conv, line, town, state, "town_state_non_ctma_v57")
    return _voice_out_of_area_reply()


def _v50_force_closed_output(phone: str, conv: dict, reply: str, reason: str = "residential_out_of_area_address") -> dict:
    """Return a final answer that cannot fall through into an old scheduling prompt.

    v57: If an older layer produced a mixed reply like
    "Goodbye. What day and time works best for you?", trim everything after
    Goodbye before closing the call.
    """
    sched = conv.setdefault("sched", {})
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["pending_step"] = None
    sched["awaiting_eval_price_confirm"] = False
    sched["awaiting_troubleshoot_price_confirm"] = False

    cleaned = str(reply or "").strip()
    m = re.search(r"(?is)^(.*?goodbye\.)", cleaned)
    if m:
        cleaned = m.group(1).strip()
    if not cleaned:
        cleaned = _voice_out_of_area_reply()
    return _voice_close_with_reply(phone, conv, cleaned, reason)


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    # Pre-layer: if the previous turn collected a street-only address and this turn gives
    # the town/state, correct the stored destination before any older Google guess can close it.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p
        pre_reply = _v50_customer_town_state_override(conv_pre, caller_text or "")
        if pre_reply:
            if "do not schedule residential repairs" in pre_reply:
                try:
                    log_event("VOICE_V50_EXPLICIT_TOWN_CLOSE", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(pre_reply), "call_sid": call_sid}, conv_pre)
                except Exception:
                    pass
                return _v50_force_closed_output(p, conv_pre, pre_reply, "residential_out_of_area_explicit_town_v50")
            conv_pre.setdefault("voice_transcript", []).append({"role": "assistant", "text": pre_reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
            conv_pre["last_voice_reply"] = pre_reply
            try:
                log_event("VOICE_V50_TOWN_STATE_ADDRESS_OVERRIDE", p, {"caller_text": _safe_monitor_text(caller_text), "reply": _safe_monitor_text(pre_reply), "stored_address": _safe_monitor_text(conv_pre.setdefault('sched', {}).get('raw_address')), "call_sid": call_sid}, conv_pre)
            except Exception:
                pass
            return {"reply_to_customer": pre_reply, "booking_created": False, "manual_only": False, "pending_step": conv_pre.setdefault("sched", {}).get("pending_step"), "appointment_type": conv_pre.setdefault("sched", {}).get("appointment_type") or conv_pre.get("appointment_type"), "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V50_PRE_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V50(phone, call_sid, caller_text)

    # Post-layer: never allow a local explicit town/state turn to close using a stale guessed town.
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        reply = str(out.get("reply_to_customer") or "")
        if "do not schedule residential repairs" in reply:
            town, state = _v50_extract_town_state(caller_text or "")
            if town and state and state in {"CT", "MA"} and not _voice_is_out_of_area_town(town):
                line = _v50_street_line_from_sched_or_text(conv, caller_text or "")
                if line:
                    _v50_apply_explicit_customer_address(conv, line, town, state, "post_close_recovery_v57")
                    recovered = _v50_reply_after_explicit_address(conv, caller_text or "")
                    end_call = "do not schedule residential repairs" in str(recovered or "")
                    out = {"reply_to_customer": recovered, "booking_created": False, "manual_only": False, "pending_step": conv.setdefault("sched", {}).get("pending_step"), "appointment_type": conv.setdefault("sched", {}).get("appointment_type") or conv.get("appointment_type"), "end_call": bool(end_call)}
                    try:
                        log_event("VOICE_V57_PREVENTED_STALE_OUT_OF_AREA_WITH_TOWN_CORRECTION", p, {"old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(recovered), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid}, conv)
                    except Exception:
                        pass
                    if end_call:
                        return _v50_force_closed_output(p, conv, recovered, "residential_out_of_area_corrected_address_v57")
                    return out
            # If it is truly out of area, make it final and clear slot state.
            return _v50_force_closed_output(p, conv, reply, "residential_out_of_area_address")
    except Exception as e:
        try:
            log_event("VOICE_V50_POST_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(caller_text), "call_sid": call_sid})
        except Exception:
            pass
    return out


# =============================
# Voice stabilization v51 — hard emergency override + explicit address recovery
# =============================
# Targets v50 1,200-test failures:
# - Hard hazard/emergency phrases were falling into the normal $195 evaluation lane.
# - Embedded local addresses that Google could not verify on the first try could still ask for spelling
#   instead of moving forward when the caller supplied town/state.
# - Explicit out-of-area towns such as Boston/Worcester/Malden/Danbury must close cleanly.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V51 = process_prevolt_voice_turn


def _v51_low(value: str) -> str:
    try:
        return _intent_text(value or "")
    except Exception:
        return str(value or "").lower().strip()


def _v51_hard_hazard_text(text: str) -> bool:
    """Hard emergency detector for voice. This must override every normal booking lane."""
    low = _v51_low(text)
    if not low:
        return False
    hard_patterns = [
        r"\bsmoke\b", r"\bsmoking\b", r"\bcaught\s+fire\b", r"\bon\s+fire\b", r"\bfire\b",
        r"\bburning\b", r"\bburnt\b", r"\bburned\b", r"\bburning\s+smell\b", r"\bsmells?\s+like\s+(?:smoke|burning)\b",
        r"\bhot\s+to\s+the\s+touch\b", r"\bpanel\s+is\s+hot\b", r"\bhot\s+panel\b", r"\boutlet\s+is\s+hot\b", r"\bhot\s+outlet\b",
        r"\bspark(?:ed|ing|s)?\b", r"\barc(?:ed|ing|s)?\b", r"\barcing\b",
        r"\bcrackl(?:e|ing)\b", r"\bpopp(?:ed|ing)?\b", r"\bbuzz(?:ing)?\b",
        r"\bwater\b.*\b(?:panel|breaker|electrical|outlet)\b", r"\b(?:panel|breaker|electrical|outlet)\b.*\bwater\b",
    ]
    return any(re.search(p, low, flags=re.I) for p in hard_patterns)


def _v51_state_from_text(text: str) -> str | None:
    low = _v51_low(text)
    if re.search(r"\b(?:ct|connecticut)\b", low):
        return "CT"
    if re.search(r"\b(?:ma|massachusetts|mass)\b", low):
        return "MA"
    return None


def _v51_extract_line_town_state(text: str, conv: dict | None = None) -> tuple[str | None, str | None, str | None]:
    """Return (street_line, town, state) from spoken text, correcting known ASR variants first."""
    conv = conv or {}
    fixed = text or ""
    try:
        fixed, _did = _v49_apply_dickerman_asr_lexicon(fixed)
    except Exception:
        pass
    line = town = state = None
    try:
        line, town, state = _v47_extract_address_candidate(fixed)
    except Exception:
        try:
            line = extract_service_address_from_text(fixed)
        except Exception:
            line = None
    try:
        town2, state2 = _v50_extract_town_state(fixed)
        town = town or town2
        state = state or state2
    except Exception:
        pass
    if not state:
        state = _v51_state_from_text(fixed)
    # Reuse stored town hint if the turn only had a street line.
    try:
        sched = conv.setdefault("sched", {}) if isinstance(conv, dict) else {}
        town = town or sched.get("voice_town_hint") or sched.get("town_hint") or sched.get("locality")
        state = state or sched.get("voice_state_hint") or sched.get("state_hint")
    except Exception:
        pass
    if line:
        line = " ".join(str(line).replace("Ticker man", "Dickerman").replace("Tickerman", "Dickerman").split()).strip(" ,.")
    if town:
        town = re.sub(r"\b(?:ct|connecticut|ma|massachusetts|mass)\b", "", str(town), flags=re.I).strip(" ,.")
        town = " ".join(town.split()[:3]).title()
    if state:
        state = "CT" if str(state).strip().lower() in {"ct", "connecticut"} else ("MA" if str(state).strip().lower() in {"ma", "massachusetts", "mass"} else str(state).strip().upper())
    return line, town, state


def _v51_apply_address_if_possible(conv: dict, caller_text: str, allow_customer_local_fallback: bool = True) -> bool:
    """Apply a complete local explicit address from the caller before old Google guesses can win."""
    sched = conv.setdefault("sched", {})
    if sched.get("address_verified"):
        return True
    line, town, state = _v51_extract_line_town_state(caller_text, conv)
    if not line:
        return False
    # Full customer town/state is the safest path and must override stale guesses.
    if town and state:
        if _voice_is_out_of_area_town(town) or _v51_low(town) in {"danbury", "new haven", "stamford"}:
            return False
        if allow_customer_local_fallback and _v50_known_local_town(town, state):
            _v50_apply_explicit_customer_address(conv, line, town, state, "v51_explicit_address")
            return bool(sched.get("address_verified"))
        # Unknown but structured CT/MA town: try Google; accept only if it preserves town/state.
        try:
            raw = f"{line}, {town}, {state}"
            result = normalize_address(raw, forced_state=state)
            if isinstance(result, tuple) and len(result) >= 2 and result[0] == "ok" and _v50_locality_matches(result[1], town, state):
                _v38_apply_normalized_address(conv, result[1], "v51_explicit_google")
                return True
        except Exception:
            pass
        return False
    # Street-only fallback. This helps known local streets like Dickerman and normal partial maps.
    try:
        if _v47_apply_normalized_if_possible(conv, line):
            return bool(sched.get("address_verified"))
    except Exception:
        pass
    try:
        addr, _reason = _v38_google_resolve_partial_address(line)
        if addr:
            _v38_apply_normalized_address(conv, addr, "v51_partial_google")
            return True
    except Exception:
        pass
    return False


def _v51_clear_normal_scheduling_state(sched: dict) -> None:
    """Emergency calls must not keep normal eval/slot-offer state alive."""
    if not isinstance(sched, dict):
        return
    sched["awaiting_eval_price_confirm"] = False
    sched["eval_price_accepted"] = False
    sched["awaiting_troubleshoot_price_confirm"] = False
    sched["troubleshoot_price_accepted"] = False
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["last_slot_unavailable_message"] = None


def _v51_is_emergency_conversation(conv: dict, caller_text: str = "") -> bool:
    """True once this voice call has been classified as a real hard-hazard emergency."""
    if not isinstance(conv, dict):
        return bool(_v51_hard_hazard_text(caller_text))
    sched = conv.setdefault("sched", {})
    appt = str(sched.get("appointment_type") or conv.get("appointment_type") or "").upper()
    state = str(sched.get("state") or "").lower()
    if _v51_hard_hazard_text(caller_text):
        return True
    if sched.get("hard_emergency_detected") or sched.get("awaiting_emergency_confirm") or sched.get("emergency_approved"):
        return True
    # Do not infer true emergency from state + TROUBLESHOOT_395 alone.
    # Normal scheduled troubleshoot visits use the same Square service.
    return False


def _v51_mark_emergency(conv: dict, caller_text: str = "", pending_step: str | None = None) -> None:
    """Lock the conversation into the emergency troubleshoot path without appending a wrapper layer."""
    sched = conv.setdefault("sched", {})
    if caller_text and _v51_hard_hazard_text(caller_text):
        sched["last_customer_issue"] = caller_text
    sched["appointment_type"] = "TROUBLESHOOT_395"
    conv["appointment_type"] = "TROUBLESHOOT_395"
    sched["hard_emergency_detected"] = True
    sched["state"] = "emergency"
    sched["price_disclosed"] = True
    sched["booking_allowed"] = True
    _v51_clear_normal_scheduling_state(sched)
    if pending_step:
        sched["pending_step"] = pending_step


def _v51_emergency_prompt(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    _v51_mark_emergency(conv, pending_step="need_date")
    if sched.get("emergency_approved"):
        sched["awaiting_emergency_confirm"] = False
    else:
        sched["awaiting_emergency_confirm"] = True
    return "This sounds urgent. We can send someone now, and arrival is usually within one to two hours. The emergency troubleshoot and repair visit is $395. Do you want us to dispatch someone now?"


def _v51_emergency_decline_reply(conv: dict) -> str:
    sched = conv.setdefault("sched", {})
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = False
    sched["customer_hard_stop"] = True
    sched["booking_allowed"] = False
    sched["closed_reason"] = "declined_emergency_dispatch"
    sched["voice_close_after_reply"] = True
    return "No problem. We will not dispatch anyone right now. If there is active fire or smoke, please call 911. Goodbye."


def _v51_emergency_after_dispatch_accept(phone: str, conv: dict) -> str:
    """After the caller approves emergency dispatch, collect only missing booking fields."""
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})
    _v51_mark_emergency(conv)

    try:
        _voice_set_emergency_dispatch_slot(sched)
    except Exception:
        try:
            tz = ZoneInfo("America/New_York")
        except Exception:
            tz = timezone(timedelta(hours=-5))
        rounded_dt = (datetime.now(tz) + timedelta(hours=1)).replace(second=0, microsecond=0)
        sched["scheduled_date"] = rounded_dt.strftime("%Y-%m-%d")
        sched["scheduled_time"] = rounded_dt.strftime("%H:%M")
        sched["scheduled_time_source"] = "voice_v51_emergency_dispatch_confirm"
        sched["booking_attempt_nonce"] = str(uuid.uuid4())

    sched["appointment_type"] = "TROUBLESHOOT_395"
    conv["appointment_type"] = "TROUBLESHOOT_395"
    sched["awaiting_emergency_confirm"] = False
    sched["emergency_approved"] = True
    sched["state"] = "emergency"
    _v51_clear_normal_scheduling_state(sched)

    first = _voice_profile_first_name(profile) if "_voice_profile_first_name" in globals() else str(profile.get("active_first_name") or profile.get("first_name") or "").strip()
    last = _voice_profile_last_name(profile) if "_voice_profile_last_name" in globals() else str(profile.get("active_last_name") or profile.get("last_name") or "").strip()

    if not (first and last):
        sched["pending_step"] = "need_name"
        return "Got it. We’ll dispatch this as an emergency. What’s your first and last name?"

    if not (_voice_has_email(conv) if "_voice_has_email" in globals() else bool(profile.get("active_email") or profile.get("email") or sched.get("email"))):
        sched["pending_step"] = "need_email"
        return "Got it. We’ll dispatch this as an emergency. What’s the best email address for the appointment?"

    try:
        booking_attempt = maybe_create_square_booking(phone, conv)
    except Exception as e:
        try:
            log_event("VOICE_V51_EMERGENCY_BOOKING_ERROR", phone, {"error": repr(e)}, conv)
        except Exception:
            pass
        booking_attempt = {"status": "exception"}

    status = booking_attempt.get("status") if isinstance(booking_attempt, dict) else None
    if sched.get("booking_created") and sched.get("square_booking_id"):
        sched["voice_close_after_reply"] = True
        sched["voice_booking_completed_close"] = True
        try:
            return _voice_finalize_booking_reply(conv, "")
        except Exception:
            return "You're all set. We are dispatching someone now, and you'll receive a confirmation text. Goodbye."

    if status == "missing_identity":
        sched["pending_step"] = "need_name"
        return "Got it. We’ll dispatch this as an emergency. What’s your first and last name?"
    if status in {"missing_email", "customer_email_missing"}:
        sched["pending_step"] = "need_email"
        return "Got it. We’ll dispatch this as an emergency. What’s the best email address for the appointment?"

    return "Got it. We have this marked as an emergency dispatch. We’ll text you the next step shortly."


def _v51_existing_emergency_turn(phone: str, conv: dict, caller_text: str) -> str | None:
    """Integrated emergency state machine for v51/v52.

    This replaces the old behavior where address, eval-price, and slot-offer layers
    could pull a hard hazard back into the normal $195 evaluation flow.
    """
    sched = conv.setdefault("sched", {})
    profile = conv.setdefault("profile", {})

    if not _v51_is_emergency_conversation(conv, caller_text):
        return None

    _v51_mark_emergency(conv, caller_text)

    # Address is the first required field. An address turn should move directly
    # to the $395 dispatch confirmation, never to the $195 evaluation prompt.
    if not sched.get("address_verified"):
        try:
            _v51_apply_address_if_possible(conv, caller_text)
        except Exception:
            pass
        if not sched.get("address_verified"):
            saved = _best_saved_address(profile)
            if saved and not _voice_text_contains_real_address(caller_text):
                sched["address_candidate"] = saved
                sched["raw_address"] = saved
                sched["address_missing"] = "confirm"
                sched["pending_step"] = "need_address"
                return f"This sounds urgent. I have {saved} on file. Is this for that address?"
            sched["pending_step"] = "need_address"
            sched["address_missing"] = "street"
            return "If there is active fire or smoke filling the home, please call 911 first. If it is safe for us to come out, what is the full address for the work?"
        return _v51_emergency_prompt(conv)

    # If the dispatch price/ETA question is pending, only yes/no/coverage answers
    # are handled here. A name or email should not re-trigger this question.
    if sched.get("awaiting_emergency_confirm") and not sched.get("emergency_approved"):
        if _voice_is_emergency_coverage_question(caller_text):
            sched["awaiting_emergency_confirm"] = True
            return (
                "The emergency troubleshoot and repair visit covers sending one of our electricians out, "
                "checking the issue in person, making the area safe, diagnosing the problem, "
                "and completing the repair during the visit when it can be handled right away. "
                "If anything larger is needed, we'll explain the next step before moving forward. "
                "Should we dispatch someone now?"
            )
        yn = _voice_yes_no_text(caller_text) if "_voice_yes_no_text" in globals() else None
        if yn == "no":
            return _v51_emergency_decline_reply(conv)
        if yn == "yes" or _voice_is_dispatch_confirmation(caller_text):
            return _v51_emergency_after_dispatch_accept(phone, conv)
        return _v51_emergency_prompt(conv)

    # After dispatch is approved, never ask the dispatch confirmation again.
    if sched.get("emergency_approved"):
        step = (sched.get("pending_step") or "").strip().lower()

        if step == "need_name" or not (_voice_profile_first_name(profile) and _voice_profile_last_name(profile)):
            name_reply = _voice_name_fast_path(conv, caller_text) if "_voice_name_fast_path" in globals() else None
            if name_reply:
                return name_reply
            sched["pending_step"] = "need_name"
            return "What’s your first and last name?"

        if step == "need_email" or not (_voice_has_email(conv) if "_voice_has_email" in globals() else bool(profile.get("active_email") or profile.get("email") or sched.get("email"))):
            email_reply = _voice_email_fast_path(phone, conv, caller_text) if "_voice_email_fast_path" in globals() else None
            if email_reply:
                return email_reply
            sched["pending_step"] = "need_email"
            return "What’s the best email address for the appointment?"

        return _v51_emergency_after_dispatch_accept(phone, conv)

    return _v51_emergency_prompt(conv)


def _v51_append_reply(conv: dict, reply: str) -> None:
    try:
        conv.setdefault("voice_transcript", []).append({"role": "assistant", "text": reply, "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat()})
    except Exception:
        pass
    conv["last_voice_reply"] = reply
    conv["last_sms_body"] = reply


def _v51_output(conv: dict, reply: str, end_call: bool = False) -> dict:
    sched = conv.setdefault("sched", {})
    _v51_append_reply(conv, reply)
    if end_call:
        sched["voice_close_after_reply"] = True
    return {
        "reply_to_customer": reply,
        "booking_created": bool(sched.get("booking_created") and sched.get("square_booking_id")),
        "manual_only": bool(sched.get("manual_only")),
        "pending_step": sched.get("pending_step"),
        "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"),
        "end_call": bool(end_call or sched.get("voice_close_after_reply")),
    }


def _v51_is_explicit_far_town(text: str) -> bool:
    low = _v51_low(text)
    return bool(re.search(r"\b(?:worcester|worcestor|boston|malden|danbury|new\s+haven|stamford)\b", low, flags=re.I))


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    text = caller_text or ""
    try:
        fixed_text, did_fix = _v49_apply_dickerman_asr_lexicon(text)
        if did_fix:
            text = fixed_text
    except Exception:
        pass

    # Integrated v51 emergency state machine.
    # This is intentionally inside the v51 function, not appended as another final wrapper.
    # Once a true hard-hazard call is detected, address collection, dispatch approval,
    # name collection, email collection, and Square booking stay inside the emergency path.
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        emergency_reply = _v51_existing_emergency_turn(p, conv, text)
        if emergency_reply:
            try:
                log_event(
                    "VOICE_V51_INTEGRATED_EMERGENCY_TURN",
                    p,
                    {
                        "caller_text": _safe_monitor_text(text),
                        "reply": _safe_monitor_text(emergency_reply),
                        "call_sid": call_sid,
                    },
                    conv,
                )
            except Exception:
                pass
            return _v51_output(conv, emergency_reply, bool(conv.setdefault("sched", {}).get("voice_close_after_reply")))
    except Exception as e:
        try:
            log_event("VOICE_V51_INTEGRATED_EMERGENCY_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(text), "call_sid": call_sid})
        except Exception:
            pass

    # Explicit far-away towns close cleanly before Google guesses or slot state can run.
    try:
        if _v51_is_explicit_far_town(text):
            conv = hydrate_voice_conversation(p, call_sid)
            conv["phone"] = p
            town, state = _v50_extract_town_state(text)
            if _voice_is_out_of_area_town(text) or _v51_low(town or text) in {"danbury", "new haven", "stamford"}:
                reply = _voice_out_of_area_reply()
                return _v50_force_closed_output(p, conv, reply, "residential_out_of_area_explicit_v51")
    except Exception:
        pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V51(phone, call_sid, text)

    # Post-layer 1: if older logic could not verify an embedded local full address, apply it and continue.
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        reply = str(out.get("reply_to_customer") or "") if isinstance(out, dict) else ""
        if re.search(r"couldn[’']?t verify|repeat just the street|spell it|which town", reply, flags=re.I):
            line, town, state = _v51_extract_line_town_state(text, conv)
            if line and town and state and _v50_known_local_town(town, state):
                _v50_apply_explicit_customer_address(conv, line, town, state, "v51_post_local_recovery")
                if _v51_is_emergency_conversation(conv, text):
                    recovered = _v51_emergency_prompt(conv)
                else:
                    recovered = _v50_reply_after_explicit_address(conv, text)
                try:
                    log_event("VOICE_V51_LOCAL_ADDRESS_RECOVERY", p, {"old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(recovered), "caller_text": _safe_monitor_text(text), "call_sid": call_sid}, conv)
                except Exception:
                    pass
                return _v51_output(conv, recovered, False)
    except Exception as e:
        try:
            log_event("VOICE_V51_POST_ADDRESS_RECOVERY_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(text), "call_sid": call_sid})
        except Exception:
            pass

    # Post-layer 2: hard guard. If a hard hazard somehow escaped, rewrite the reply to emergency language.
    try:
        if _v51_hard_hazard_text(text):
            conv = hydrate_voice_conversation(p, call_sid)
            conv["phone"] = p
            sched = conv.setdefault("sched", {})
            _v51_mark_emergency(conv, text)
            bad_reply = str(out.get("reply_to_customer") or "") if isinstance(out, dict) else ""
            if "195" in _v51_low(bad_reply) or "evaluation" in _v51_low(bad_reply) or "what day" in _v51_low(bad_reply):
                if sched.get("address_verified"):
                    reply = _v51_emergency_prompt(conv)
                else:
                    sched["pending_step"] = "need_address"
                    sched["address_missing"] = "street"
                    reply = "If there is active fire or smoke filling the home, please call 911 first. If it is safe for us to come out, what is the full address for the work?"
                return _v51_output(conv, reply, False)
    except Exception:
        pass

    return out


# =============================
# Voice stabilization v52 — cleanup after v51 emergency pass
# =============================
# Fixes the two remaining v51 intake harness failure classes:
# - Old layers can have a verified address + loaded slots but still speak a stale "couldn't verify" prompt.
# - Danbury/New Haven/Stamford explicit addresses must close as out-of-area, not ask for spelling.

_ORIG_PROCESS_PREVOLT_VOICE_TURN_V52 = process_prevolt_voice_turn


def _v52_text_has_explicit_far_town(text: str) -> bool:
    low = _v51_low(text)
    return bool(re.search(r"\b(?:worcester|worcestor|boston|malden|danbury|new\s+haven|stamford)\b", low, flags=re.I))




# =============================
# Voice stabilization v59 — no early price leak + no email spellback
# =============================
# Root fixes:
# 1. Do not disclose or partially leak price/coverage language before the address is verified.
#    The customer should hear the price only after the system knows the address can be serviced.
# 2. Do not read a long spelled email back to the caller if a complete valid email was extracted.
#    Save it and finish the booking instead.




# =============================
# Voice stabilization v60 — confirm email + smoother partial address flow
# =============================
# Root fixes:
# 1. Email is billing-critical. A voice-captured email must be confirmed before booking.
#    Do NOT auto-accept a guessed/transcribed email.
#    Do NOT use long NATO-style spellback. Use a short confirmation instead:
#       "I have kprevost92@gmail.com. Is that correct?"
# 2. If the caller gives a street-only address such as "40 Arch Street",
#    ask "What town and state is that in?" instead of saying "I couldn't verify that address."


def _v60_yes(text: str) -> bool:
    low = _intent_text(text or "") if "_intent_text" in globals() else str(text or "").strip().lower()
    return low in {"yes", "yeah", "yep", "yup", "correct", "that's correct", "that is correct", "right", "yes correct", "yes that's correct", "yes that is correct", "sure", "ok", "okay"}


def _v60_no(text: str) -> bool:
    low = _intent_text(text or "") if "_intent_text" in globals() else str(text or "").strip().lower()
    return low in {"no", "nope", "nah", "incorrect", "not correct", "that's wrong", "that is wrong", "wrong"} or "no " == (low + " ")[:3]


def _v60_valid_email_from_text(text: str) -> str | None:
    try:
        email = v13_extract_email(text or "") if "v13_extract_email" in globals() else None
    except Exception:
        email = None

    if not email:
        return None

    email = str(email or "").strip().lower().strip(".,;: ")
    if not re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", email):
        return None

    return email


def _v60_email_waiting_context(conv: dict) -> bool:
    sched = conv.setdefault("sched", {})
    return bool(
        str(sched.get("pending_step") or "").lower() in {"need_email", "need_email_confirm", "confirm_email"}
        or str(sched.get("state") or "").lower() in {"waiting_for_email", "waiting_for_email_confirm"}
        or sched.get("voice_awaiting_email_confirm")
        or re.search(r"email", str(conv.get("last_voice_reply") or ""), flags=re.I)
    )


def _v60_short_email_confirmation(email: str) -> str:
    # Keep the actual email in the text so the customer can confirm it,
    # but do not force NATO-style letter-by-letter spellback.
    return f"I have {email}. Is that correct?"


def _v60_email_output(conv: dict, reply: str, booking_created: bool = False, end_call: bool = False) -> dict:
    sched = conv.setdefault("sched", {})
    conv["last_voice_reply"] = reply
    try:
        conv.setdefault("voice_transcript", []).append({
            "role": "assistant",
            "text": reply,
            "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass
    return {
        "reply_to_customer": _voice_naturalize_reply(reply) if "_voice_naturalize_reply" in globals() else reply,
        "booking_created": bool(booking_created),
        "manual_only": bool(sched.get("manual_only")),
        "pending_step": sched.get("pending_step"),
        "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"),
        "end_call": bool(end_call),
    }


def _v60_handle_email_turn(phone: str, conv: dict, caller_text: str) -> dict | None:
    sched = conv.setdefault("sched", {})

    if not _v60_email_waiting_context(conv):
        return None

    pending_email = str(sched.get("voice_pending_email") or sched.get("pending_email") or "").strip().lower()

    # If we are confirming an email, handle yes/no before any lower layer can treat
    # "yes" as some other acceptance.
    if sched.get("voice_awaiting_email_confirm") and pending_email:
        if _v60_yes(caller_text):
            sched["voice_awaiting_email_confirm"] = False
            sched["voice_email_confirmed"] = True
            try:
                if "_voice_save_confirmed_email_and_maybe_book_v31" in globals():
                    out = _voice_save_confirmed_email_and_maybe_book_v31(phone, conv, pending_email)
                else:
                    v13_save_email(conv, pending_email)
                    out = {"reply_to_customer": "Thanks. I’m finalizing the booking details now.", "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
            except Exception as e:
                try:
                    log_event("VOICE_V60_CONFIRMED_EMAIL_BOOK_ERROR", phone, {"error": repr(e), "email": pending_email}, conv)
                except Exception:
                    pass
                return _v60_email_output(conv, "Thanks. I have the email confirmed. I’m finalizing the booking details now.", False, False)

            reply = str((out or {}).get("reply_to_customer") or "").strip()
            low = _intent_text(reply) if "_intent_text" in globals() else reply.lower()

            # Prevent lower layers from doing another email spellback after confirmation.
            if not reply or (low.startswith("i heard") and "correct" in low):
                if sched.get("booking_created") and sched.get("square_booking_id"):
                    try:
                        reply = _voice_finalize_booking_reply(conv, "") if "_voice_finalize_booking_reply" in globals() else "You’re all set. You’ll receive a confirmation text shortly. Thank you for calling Prevolt Electric. Goodbye."
                        sched["voice_close_after_reply"] = True
                        out["booking_created"] = True
                        out["end_call"] = True
                    except Exception:
                        reply = "Thanks. I’m finalizing the booking details now."
                else:
                    reply = "Thanks. I’m finalizing the booking details now."

            out["reply_to_customer"] = _voice_naturalize_reply(reply) if "_voice_naturalize_reply" in globals() else reply
            out["pending_step"] = sched.get("pending_step")
            out["appointment_type"] = sched.get("appointment_type") or conv.get("appointment_type")
            try:
                log_event("VOICE_V60_EMAIL_CONFIRMED_AND_SAVED", phone, {"email": pending_email, "reply": _safe_monitor_text(out["reply_to_customer"]), "booking_created": bool(out.get("booking_created"))}, conv)
            except Exception:
                pass
            return out

        if _v60_no(caller_text):
            sched["voice_pending_email"] = None
            sched["pending_email"] = None
            sched["voice_awaiting_email_confirm"] = False
            sched["voice_email_confirmed"] = False
            sched["pending_step"] = "need_email"
            sched["state"] = "waiting_for_email"
            try:
                log_event("VOICE_V60_EMAIL_REJECTED_REASK", phone, {"old_email": pending_email}, conv)
            except Exception:
                pass
            return _v60_email_output(conv, "Okay, what is the best email address for the appointment?", False, False)

        # If they didn't answer yes/no but gave a new email, capture the new one.
        new_email = _v60_valid_email_from_text(caller_text)
        if new_email:
            sched["voice_pending_email"] = new_email
            sched["pending_email"] = new_email
            sched["voice_awaiting_email_confirm"] = True
            sched["pending_step"] = "need_email_confirm"
            sched["state"] = "waiting_for_email_confirm"
            reply = _v60_short_email_confirmation(new_email)
            try:
                log_event("VOICE_V60_EMAIL_REPLACED_FOR_CONFIRMATION", phone, {"email": new_email, "reply": _safe_monitor_text(reply)}, conv)
            except Exception:
                pass
            return _v60_email_output(conv, reply, False, False)

        return _v60_email_output(conv, "Please say the email address one more time.", False, False)

    # First time hearing a complete email: capture it and confirm. Do not book yet.
    email = _v60_valid_email_from_text(caller_text)
    if email:
        sched["voice_pending_email"] = email
        sched["pending_email"] = email
        sched["voice_awaiting_email_confirm"] = True
        sched["voice_email_confirmed"] = False
        sched["pending_step"] = "need_email_confirm"
        sched["state"] = "waiting_for_email_confirm"
        reply = _v60_short_email_confirmation(email)
        try:
            log_event("VOICE_V60_EMAIL_CAPTURED_FOR_CONFIRMATION", phone, {"email": email, "reply": _safe_monitor_text(reply)}, conv)
        except Exception:
            pass
        return _v60_email_output(conv, reply, False, False)

    return None


def _v60_customer_gave_street_without_town_state(text: str) -> bool:
    raw = " ".join(str(text or "").replace("\n", " ").split()).strip(" ,.")
    if not raw:
        return False

    # Must at least look like a house number + street.
    try:
        line = extract_service_address_from_text(raw)
    except Exception:
        line = None
    if not line:
        line = raw

    try:
        if not _address_has_house_number_and_street(line):
            return False
    except Exception:
        if not re.search(r"\b\d{1,6}\s+[a-z0-9 .'-]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct|circle|cir|boulevard|blvd|place|pl)\b", raw, flags=re.I):
            return False

    # If customer already supplied clear state, this is not the partial-address case.
    if re.search(r"\b(?:ct|connecticut|ma|massachusetts|m a|c t)\b", raw, flags=re.I):
        return False

    # If the v58 guard says it still needs locality, we should ask for town/state.
    try:
        allowed, _safe_raw, _forced_state, _town, why = _v58_prepare_address_for_maps(raw)
        if not allowed and why in {"missing_town", "unknown_town_without_state", "needs_customer_town_state:missing_town"}:
            return True
    except Exception:
        pass

    # Also catch plain "40 Arch Street" style input.
    return bool(re.search(r"\b\d{1,6}\s+[a-z0-9 .'-]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct|circle|cir|boulevard|blvd|place|pl)\b", raw, flags=re.I))


def _v60_partial_address_town_state_reply(conv: dict) -> dict:
    sched = conv.setdefault("sched", {})
    sched["address_verified"] = False
    sched["address_missing"] = "state"
    sched["pending_step"] = "need_address"
    sched["state"] = "waiting_for_address"
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    reply = "What town and state is that in?"
    return _v59_output(conv, reply, booking_created=False, end_call=False)



def _v59_reply_has_price_or_coverage_leak(reply: str) -> bool:
    low = _intent_text(reply or "") if "_intent_text" in globals() else str(reply or "").lower()
    if not low:
        return False
    return bool(
        re.search(r"\b(?:195|395)\b", low)
        or "that covers sending one of our electricians" in low
        or "covers sending one of our electricians" in low
        or "reviewing the work in person" in low
        or "checking what is needed" in low
        or "troubleshoot and repair visits are" in low
        or "on site evaluation visit" in low
        or "on-site evaluation visit" in low
        or ("does that work for you" in low and ("evaluation" in low or "troubleshoot" in low or "repair visit" in low))
    )


def _v59_still_collecting_address(conv: dict) -> bool:
    sched = conv.setdefault("sched", {})
    if sched.get("address_verified"):
        return False
    state = str(sched.get("state") or "").lower()
    pending = str(sched.get("pending_step") or "").lower()
    missing = str(sched.get("address_missing") or "").lower()
    return bool(
        pending == "need_address"
        or state == "waiting_for_address"
        or missing in {"street", "number", "city", "state", "confirm", "zip"}
    )


def _v59_address_collection_reply(conv: dict, caller_text: str = "") -> str:
    sched = conv.setdefault("sched", {})

    # Keep known emergency behavior untouched.
    try:
        if "_v51_is_emergency_conversation" in globals() and _v51_is_emergency_conversation(conv, caller_text):
            return "If it is safe for us to come out, what is the full address for the work?"
    except Exception:
        pass

    # Reset price/date gates until address is actually verified.
    sched["awaiting_eval_price_confirm"] = False
    sched["eval_price_accepted"] = False
    sched["awaiting_troubleshoot_price_confirm"] = False
    sched["troubleshoot_price_accepted"] = False
    sched["awaiting_slot_offer_choice"] = False
    sched["offered_slot_options"] = []
    sched["pending_step"] = "need_address"
    sched["state"] = "waiting_for_address"
    sched["address_verified"] = False

    try:
        if "_v46_power_loss_troubleshoot_text" in globals() and _v46_power_loss_troubleshoot_text(caller_text or ""):
            sched["appointment_type"] = "TROUBLESHOOT_395"
            return "Got it. What is the full address for the work, including the town and state?"
    except Exception:
        pass

    return "I can get that started. What is the full address for the work, including the town and state?"


def _v59_output(conv: dict, reply: str, booking_created: bool = False, end_call: bool = False) -> dict:
    sched = conv.setdefault("sched", {})
    conv["last_voice_reply"] = reply
    try:
        conv.setdefault("voice_transcript", []).append({
            "role": "assistant",
            "text": reply,
            "ts": _monitor_now_iso() if "_monitor_now_iso" in globals() else datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass
    return {
        "reply_to_customer": reply,
        "booking_created": bool(booking_created),
        "manual_only": bool(sched.get("manual_only")),
        "pending_step": sched.get("pending_step"),
        "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"),
        "end_call": bool(end_call),
    }


def _v59_auto_accept_complete_email(phone: str, conv: dict, caller_text: str) -> dict | None:
    """If the caller gives a complete email, do not spell it back over voice."""
    try:
        email = v13_extract_email(caller_text or "") if "v13_extract_email" in globals() else None
    except Exception:
        email = None

    if not email:
        return None

    email = str(email or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", email):
        return None

    sched = conv.setdefault("sched", {})

    waiting_for_email = bool(
        str(sched.get("pending_step") or "").lower() == "need_email"
        or str(sched.get("state") or "").lower() == "waiting_for_email"
        or sched.get("voice_awaiting_email_confirm")
        or re.search(r"email", str(conv.get("last_voice_reply") or ""), flags=re.I)
    )
    if not waiting_for_email:
        return None

    sched["voice_pending_email"] = None
    sched["voice_awaiting_email_confirm"] = False
    sched["voice_email_confirmed"] = True

    try:
        if "_voice_save_confirmed_email_and_maybe_book_v31" in globals():
            out = _voice_save_confirmed_email_and_maybe_book_v31(phone, conv, email)
        else:
            v13_save_email(conv, email)
            out = {"reply_to_customer": "Thanks. I’m finalizing the booking details now.", "booking_created": False, "manual_only": False, "pending_step": sched.get("pending_step"), "appointment_type": sched.get("appointment_type") or conv.get("appointment_type"), "end_call": False}
    except Exception as e:
        try:
            log_event("VOICE_V59_AUTO_EMAIL_BOOK_ERROR", phone, {"error": repr(e), "email": email}, conv)
        except Exception:
            pass
        return None

    reply = str((out or {}).get("reply_to_customer") or "").strip()
    if not reply:
        reply = "Thanks. I’m finalizing the booking details now."

    low = _intent_text(reply) if "_intent_text" in globals() else reply.lower()
    if low.startswith("i heard") and "correct" in low:
        if sched.get("booking_created") and sched.get("square_booking_id"):
            try:
                reply = _voice_finalize_booking_reply(conv, "") if "_voice_finalize_booking_reply" in globals() else "You’re all set. You’ll receive a confirmation text shortly. Thank you for calling Prevolt Electric. Goodbye."
                sched["voice_close_after_reply"] = True
                out["booking_created"] = True
                out["end_call"] = True
            except Exception:
                reply = "Thanks. I’m finalizing the booking details now."
        else:
            reply = "Thanks. I’m finalizing the booking details now."
            out["booking_created"] = False
            out["end_call"] = False

    out["reply_to_customer"] = _voice_naturalize_reply(reply) if "_voice_naturalize_reply" in globals() else reply
    out["pending_step"] = sched.get("pending_step")
    out["appointment_type"] = sched.get("appointment_type") or conv.get("appointment_type")
    conv["last_voice_reply"] = out["reply_to_customer"]
    try:
        log_event("VOICE_V59_AUTO_ACCEPTED_EMAIL_NO_SPELLBACK", phone, {"email": email, "reply": _safe_monitor_text(out["reply_to_customer"]), "booking_created": bool(out.get("booking_created"))}, conv)
    except Exception:
        pass
    return out



def _v52_stale_verify_reply(reply: str) -> bool:
    return bool(re.search(r"couldn[’']?t verify|repeat just the street|spell it", str(reply or ""), flags=re.I))


def process_prevolt_voice_turn(phone: str, call_sid: str, caller_text: str) -> dict:
    p = (phone or "").replace("whatsapp:", "").strip()
    text = caller_text or ""

    # Explicit far-town close. This is intentionally before v51/v50 so Google cannot
    # temporarily normalize "1 Main Street" into some other local-looking value.
    try:
        if _v52_text_has_explicit_far_town(text) and not re.search(r"\b(?:windsor|windsor\s+locks|suffield|enfield|granby|bloomfield|hartford)\b", _v51_low(text), flags=re.I):
            conv = hydrate_voice_conversation(p, call_sid)
            conv["phone"] = p
            reply = _voice_out_of_area_reply()
            return _v50_force_closed_output(p, conv, reply, "residential_out_of_area_explicit_v52")
    except Exception:
        pass

    # v60 pre-intercepts:
    # 1. Email is billing-critical. Capture then confirm; never auto-accept voice email.
    # 2. Street-only address should naturally ask for town/state.
    try:
        conv_pre = hydrate_voice_conversation(p, call_sid)
        conv_pre["phone"] = p

        email_out = _v60_handle_email_turn(p, conv_pre, text)
        if email_out:
            return email_out

        if _v59_still_collecting_address(conv_pre) and _v60_customer_gave_street_without_town_state(text):
            try:
                log_event("VOICE_V60_PARTIAL_ADDRESS_ASK_TOWN_STATE_PRE", p, {"caller_text": _safe_monitor_text(text), "call_sid": call_sid}, conv_pre)
            except Exception:
                pass
            return _v60_partial_address_town_state_reply(conv_pre)

    except Exception as e:
        try:
            log_event("VOICE_V60_PRE_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(text), "call_sid": call_sid})
        except Exception:
            pass

    out = _ORIG_PROCESS_PREVOLT_VOICE_TURN_V52(phone, call_sid, text)

    # If lower layers actually verified the address but left a stale "couldn't verify"
    # spoken reply, replace the spoken prompt with the proper price gate.
    try:
        conv = hydrate_voice_conversation(p, call_sid)
        conv["phone"] = p
        sched = conv.setdefault("sched", {})
        reply = str(out.get("reply_to_customer") or "") if isinstance(out, dict) else ""

        # v60 post: replace clunky street-only address failure with natural town/state prompt.
        if (
            _v59_still_collecting_address(conv)
            and _v60_customer_gave_street_without_town_state(text)
            and re.search(r"(couldn.?t verify|full address|town and state|address)", reply, flags=re.I)
        ):
            try:
                log_event("VOICE_V60_PARTIAL_ADDRESS_ASK_TOWN_STATE_POST", p, {"caller_text": _safe_monitor_text(text), "old_reply": _safe_monitor_text(reply), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return _v60_partial_address_town_state_reply(conv)

        if _v52_stale_verify_reply(reply) and sched.get("address_verified"):
            recovered = _v50_reply_after_explicit_address(conv, text)
            try:
                log_event("VOICE_V52_STALE_VERIFY_REPLY_RECOVERY", p, {"old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(recovered), "caller_text": _safe_monitor_text(text), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return _v51_output(conv, recovered, False)

        # v59 root guard: price/coverage language is never allowed while address
        # collection is still active. Price comes only after address verification.
        if _v59_still_collecting_address(conv) and _v59_reply_has_price_or_coverage_leak(reply):
            cleaned = _v59_address_collection_reply(conv, text)
            try:
                log_event("VOICE_V59_BLOCKED_EARLY_PRICE_BEFORE_ADDRESS", p, {"caller_text": _safe_monitor_text(text), "old_reply": _safe_monitor_text(reply), "new_reply": _safe_monitor_text(cleaned), "call_sid": call_sid}, conv)
            except Exception:
                pass
            return _v59_output(conv, cleaned, booking_created=False, end_call=False)

        # v60 post: if a lower layer already created an email spellback prompt on a
        # complete email, replace it with short confirmation or confirmation handling.
        if isinstance(out, dict) and re.search(r"^\s*I heard\b.*\bcorrect\?", reply, flags=re.I):
            email_out = _v60_handle_email_turn(p, conv, text)
            if email_out:
                return email_out

    except Exception as e:
        try:
            log_event("VOICE_V60_POST_ERROR", p, {"error": repr(e), "caller_text": _safe_monitor_text(text), "call_sid": call_sid})
        except Exception:
            pass

    return out
