import os
import re
import json
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta, timezone, time as dt_time
from zoneinfo import ZoneInfo

import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from openai import OpenAI


# ---------------------------------------------------
# Environment
# ---------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")

SQUARE_ACCESS_TOKEN = os.environ.get("SQUARE_ACCESS_TOKEN")
SQUARE_LOCATION_ID = os.environ.get("SQUARE_LOCATION_ID")
SQUARE_TEAM_MEMBER_ID = os.environ.get("SQUARE_TEAM_MEMBER_ID")

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
DISPATCH_ORIGIN_ADDRESS = os.environ.get("DISPATCH_ORIGIN_ADDRESS")
TECH_CURRENT_ADDRESS = os.environ.get("TECH_CURRENT_ADDRESS")
RULES_FILE = os.environ.get("PREVOLT_RULES_FILE") or os.environ.get("PREVOLT_RULES_PATH")

SERVICE_VARIATION_EVAL_ID = "IPCUF6EPOYGWJUEFUZOXL2AZ"
SERVICE_VARIATION_EVAL_VERSION = 1764725435505

SERVICE_VARIATION_INSPECTION_ID = "LYK646AH4NAESCFUZL6PUTZ2"
SERVICE_VARIATION_INSPECTION_VERSION = 1764725393938

SERVICE_VARIATION_TROUBLESHOOT_ID = "64IQNJYO3H6XNTLPIHABDJOQ"
SERVICE_VARIATION_TROUBLESHOOT_VERSION = 1762464315698

BOOKING_START_HOUR = 9
BOOKING_END_HOUR = 16
MAX_TRAVEL_MINUTES = 60
LOCAL_TZ = ZoneInfo("America/New_York")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

app = Flask(__name__)
conversations: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------
# Rules text loader
# ---------------------------------------------------
def load_rule_matrix_text() -> str:
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
        except Exception as exc:
            print(f"[WARN] rule load failed for {candidate}: {exc!r}")
    return ""


RULE_MATRIX_TEXT = load_rule_matrix_text()


# ---------------------------------------------------
# Conversation schema
# ---------------------------------------------------
def now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

STREET_SUFFIXES = (
    "st", "street", "ave", "avenue", "rd", "road", "ln", "lane",
    "dr", "drive", "ct", "court", "cir", "circle", "blvd", "boulevard",
    "way", "pkwy", "parkway", "ter", "terrace", "pl", "place"
)

AMBIGUOUS_TIME_PHRASES = {
    "any time", "anytime", "whenever", "later", "sometime", "around",
    "as soon as possible", "asap", "i'm around", "im around",
    "i'm home today", "im home today", "i'm here all day", "im here all day",
    "it doesn't matter", "it doesnt matter", "whenever works", "whenever you can",
    "sometime today", "later today", "this morning", "this afternoon",
    "this evening", "today works", "i'm available today", "im available today",
}

EMERGENCY_KEYWORDS = [
    "no power", "power outage", "partial power", "half the house", "lights flicker",
    "burning", "burnt", "sparking", "arcing", "smoke", "fire", "hot outlet",
    "hot panel", "water in panel", "meter ripped", "service ripped", "tree on line",
    "tree fell", "line down", "lines down", "weatherhead", "breaker won't reset",
    "breaker wont reset", "main breaker", "melted", "loud pop", "boom", "bang",
    "urgent", "emergency", "asap", "right now", "immediately"
]

QUESTION_FALLBACKS = {
    "need_issue": "What’s going on there today?",
    "need_date": "What day works best for you?",
    "need_time": "What time works best?",
    "need_address": "What’s the address for the visit?",
    "need_name": "What is your first and last name?",
    "need_email": "What is the best email address for the appointment?",
    "need_state": "Just to confirm, is this address in Connecticut or Massachusetts?",
}


def empty_conversation() -> Dict[str, Any]:
    return {
        "profile": {
            "customer_type": "residential",
            "first_name": None,
            "last_name": None,
            "email": None,
            "square_customer_id": None,
            "square_lookup_done": False,
            "addresses": [],
            "upcoming_appointment": None,
            "past_jobs": [],
        },
        "current_job": {
            "job_type": None,
            "raw_description": None,
        },
        "sched": {
            "appointment_type": None,
            "scheduled_date": None,
            "scheduled_time": None,
            "raw_address": None,
            "normalized_address": None,
            "address_verified": False,
            "address_missing": None,
            "booking_created": False,
            "square_booking_id": None,
            "awaiting_emergency_confirm": False,
            "emergency_approved": False,
            "pending_step": None,
            "last_prompt": None,
            "last_prompt_count": 0,
            "price_disclosed": False,
            "reassurance_used": False,
            "post_booking_open": False,
            "address_state_ask_count": 0,
            "time_ask_count": 0,
            "date_ask_count": 0,
            "name_ask_count": 0,
            "email_ask_count": 0,
            "last_user_message": None,
        },
        "cleaned_transcript": None,
        "category": None,
        "initial_sms": None,
        "last_sms_body": None,
    }


def get_convo_key(is_call: bool = False) -> str:
    frm = (request.values.get("From") or "").replace("whatsapp:", "").strip()
    if frm:
        return frm
    if is_call:
        return request.values.get("CallSid") or "call:unknown"
    return request.values.get("MessageSid") or request.values.get("SmsSid") or "sms:unknown"


def ensure_conv(key: str) -> Dict[str, Any]:
    conv = conversations.setdefault(key, empty_conversation())
    base = empty_conversation()
    for top_key, top_val in base.items():
        if isinstance(top_val, dict):
            conv.setdefault(top_key, {})
            for k, v in top_val.items():
                conv[top_key].setdefault(k, v)
        else:
            conv.setdefault(top_key, top_val)
    return conv


# ---------------------------------------------------
# Generic helpers
# ---------------------------------------------------
def norm_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def lower_text(text: str) -> str:
    return norm_text(text).lower()


def humanize_time(value: Optional[str]) -> str:
    if not value:
        return ""
    t = value.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", t)
    if not m:
        return t
    hh = int(m.group(1))
    mm = int(m.group(2))
    suffix = "AM" if hh < 12 else "PM"
    hh12 = hh % 12 or 12
    return f"{hh12}:{mm:02d} {suffix}"


def humanize_date(date_str: Optional[str]) -> str:
    if not date_str:
        return ""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.strftime("%A, %B %d").replace(" 0", " ")
    except Exception:
        return date_str


def strip_ai_punctuation(text: str) -> str:
    text = (text or "").replace("—", ".").replace("–", ".").replace(" - ", " ")
    text = re.sub(r"\s+\.", ".", text)
    return norm_text(text)


def shorten_for_texting(text: str, max_chars: int = 240) -> str:
    text = norm_text(text)
    if len(text) <= max_chars:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(parts[:2]).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip(" ,;:") + "."
    return norm_text(text)


def deterministic_pick(key: str, options: list[str]) -> str:
    if not options:
        return ""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(options)
    return options[idx]


def is_ack_only(text: str) -> bool:
    t = lower_text(text)
    ack_set = {"ok", "okay", "k", "kk", "sure", "yep", "yeah", "yes", "thanks", "thank you", "thx", "perfect", "sounds good", "that works"}
    if t in ack_set:
        return True
    return len(t) <= 20 and any(x in t for x in ["ok", "okay", "thanks", "thank", "thx"])


def is_ct_or_ma(text: str) -> Optional[str]:
    low = lower_text(text)
    if re.search(r"\b(ct|connecticut)\b", low):
        return "CT"
    if re.search(r"\b(ma|mass|massachusetts)\b", low):
        return "MA"
    return None


def clean_name_piece(value: str) -> str:
    value = re.sub(r"[^A-Za-z\-\' ]", " ", value)
    return " ".join(part.capitalize() for part in value.split())


def extract_email(text: str) -> Optional[str]:
    m = re.search(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", text or "", flags=re.I)
    return m.group(1).strip() if m else None


def extract_name(text: str) -> Tuple[Optional[str], Optional[str]]:
    text = norm_text(text)
    low = lower_text(text)
    if not text or "@" in text or any(ch.isdigit() for ch in text):
        return None, None
    if any(x in low for x in ["connecticut", "massachusetts", " ct", " ma", "tomorrow", "today", "friday", "monday", "tuesday", "wednesday", "thursday", "saturday", "sunday", "any time", "asap"]):
        return None, None
    text = re.sub(r"^(my name is|name is|this is|it's|its)\s+", "", text, flags=re.I)
    parts = [clean_name_piece(p) for p in text.split() if clean_name_piece(p)]
    if len(parts) < 2:
        return None, None
    return parts[0], " ".join(parts[1:])


def likely_address_text(text: str) -> bool:
    low = lower_text(text)
    has_num = bool(re.search(r"\b\d{1,6}\b", low))
    has_suffix = bool(re.search(r"\b(" + "|".join(STREET_SUFFIXES) + r")\b", low))
    return has_num and has_suffix


def split_multi_messages(text: str) -> list[str]:
    text = norm_text(text)
    if not text:
        return []
    parts = re.split(r"\s*(?:\n+|\.{3,}|\|+)\s*", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------
# Intent and extraction
# ---------------------------------------------------
def detect_emergency(text: str) -> bool:
    low = lower_text(text)
    return any(k in low for k in EMERGENCY_KEYWORDS)


def classify_appointment_type(text: str, existing: Optional[str] = None) -> str:
    low = lower_text(text)
    if detect_emergency(text):
        return "TROUBLESHOOT_395"
    if any(k in low for k in ["inspection", "inspect", "home inspection", "whole home"]):
        return "WHOLE_HOME_INSPECTION"
    if any(k in low for k in ["troubleshoot", "repair", "diagnose", "problem", "issue", "not working", "breaker", "outlet", "light"]):
        return "TROUBLESHOOT_395" if "repair" in low or "troubleshoot" in low else (existing or "EVAL_195")
    if any(k in low for k in ["quote", "estimate", "panel", "generator", "ev charger", "install", "upgrade", "look at"]):
        return existing or "EVAL_195"
    return existing or "EVAL_195"


def extract_weekday_date(text: str, base_dt: Optional[datetime] = None) -> Optional[str]:
    base_dt = base_dt or now_local()
    low = lower_text(text)

    if re.search(r"\btoday\b", low):
        return base_dt.strftime("%Y-%m-%d")
    if re.search(r"\btomorrow\b", low):
        return (base_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    for word, idx in WEEKDAYS.items():
        if re.search(rf"\b{re.escape(word)}s?\b", low):
            days_ahead = (idx - base_dt.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            if re.search(rf"\bthis\s+{re.escape(word)}\b", low):
                days_ahead = (idx - base_dt.weekday()) % 7
                days_ahead = 0 if days_ahead == 0 else days_ahead
            if re.search(rf"\bnext\s+{re.escape(word)}\b", low):
                if days_ahead == 0:
                    days_ahead = 7
            return (base_dt + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    return None


def extract_explicit_date(text: str, base_dt: Optional[datetime] = None) -> Optional[str]:
    base_dt = base_dt or now_local()
    low = lower_text(text)

    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", low)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else base_dt.year
        if year < 100:
            year += 2000
        try:
            d = datetime(year, month, day, tzinfo=LOCAL_TZ)
            if d.date() < base_dt.date():
                d = datetime(year + 1, month, day, tzinfo=LOCAL_TZ)
            return d.strftime("%Y-%m-%d")
        except Exception:
            pass

    m2 = re.search(r"\b(" + "|".join(MONTHS.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b", low)
    if m2:
        month = MONTHS[m2.group(1)]
        day = int(m2.group(2))
        year = int(m2.group(3)) if m2.group(3) else base_dt.year
        try:
            d = datetime(year, month, day, tzinfo=LOCAL_TZ)
            if d.date() < base_dt.date() and not m2.group(3):
                d = datetime(year + 1, month, day, tzinfo=LOCAL_TZ)
            return d.strftime("%Y-%m-%d")
        except Exception:
            pass

    return None


def extract_date(text: str) -> Optional[str]:
    return extract_explicit_date(text) or extract_weekday_date(text)


def parse_time_components(hour: int, minute: int, meridiem: Optional[str]) -> Optional[str]:
    if meridiem:
        mer = meridiem.lower()
        if hour == 12:
            hour = 0
        if mer == "pm":
            hour += 12
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    return None


def round_up_to_next_half_hour(base_dt: Optional[datetime] = None) -> str:
    base_dt = base_dt or now_local()
    minute = 30 if base_dt.minute > 0 and base_dt.minute <= 30 else 0
    hour = base_dt.hour
    if base_dt.minute > 30:
        hour += 1
        minute = 0
    if base_dt.minute == 0:
        minute = 30
    future = base_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=(hour - base_dt.hour))
    future = future.replace(hour=hour % 24, minute=minute)
    if future <= base_dt:
        future = base_dt + timedelta(minutes=30)
        future = future.replace(minute=30 if future.minute <= 30 and future.minute != 0 else 0, second=0, microsecond=0)
    return future.strftime("%H:%M")


def extract_time(text: str, appointment_type: Optional[str] = None) -> Tuple[Optional[str], bool]:
    low = lower_text(text)

    for phrase in sorted(AMBIGUOUS_TIME_PHRASES, key=len, reverse=True):
        if phrase in low:
            return None, True

    if appointment_type == "TROUBLESHOOT_395" and re.search(r"\b(now|right now|immediately|asap)\b", low):
        return round_up_to_next_half_hour(), False

    range_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|\-|through|thru)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", low)
    if range_match:
        first_hour = int(range_match.group(1))
        first_min = int(range_match.group(2) or "00")
        first_mer = range_match.group(3) or range_match.group(6)
        return parse_time_components(first_hour, first_min, first_mer), False

    noon_midnight = {
        "noon": "12:00",
        "midnight": "00:00",
    }
    for phrase, val in noon_midnight.items():
        if re.search(rf"\b{phrase}\b", low):
            return val, False

    exact = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", low)
    if exact:
        hour = int(exact.group(1))
        minute = int(exact.group(2) or "00")
        return parse_time_components(hour, minute, exact.group(3)), False

    military = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", low)
    if military:
        return f"{int(military.group(1)):02d}:{int(military.group(2)):02d}", False

    parts = {
        "morning": "09:00",
        "afternoon": "14:00",
        "evening": "16:00",
        "early": "09:00",
        "midday": "12:00",
        "tonight": "16:00" if appointment_type != "TROUBLESHOOT_395" else "19:00",
    }
    for phrase, val in parts.items():
        if re.search(rf"\b{phrase}\b", low):
            return val, False

    # Bare hour like "at 3" or "3 works"
    loose = re.search(r"(?:\bat\s+|\bafter\s+|\baround\s+|\bby\s+)?\b(\d{1,2})\b", low)
    if loose:
        hour = int(loose.group(1))
        # Avoid treating address numbers as times when a street suffix is nearby.
        nearby = low[max(0, loose.start() - 10): min(len(low), loose.end() + 18)]
        if not re.search(r"\b(" + "|".join(STREET_SUFFIXES) + r")\b", nearby):
            if 1 <= hour <= 7:
                return parse_time_components(hour, 0, "pm"), False
            if 8 <= hour <= 11:
                return parse_time_components(hour, 0, "am"), False
            if hour == 12:
                return "12:00", False

    return None, False


def extract_address_candidate(text: str) -> Optional[str]:
    text = norm_text(text)
    if not text:
        return None
    m = re.search(
        r"\b\d{1,6}\s+[A-Za-z0-9 .'-]+?\s(?:" + "|".join(STREET_SUFFIXES) + r")\b(?:\s+(?:apt|apartment|unit|suite|ste)\s*[A-Za-z0-9-]+)?(?:,?\s+[A-Za-z .'-]+)?(?:,?\s+(?:CT|MA|Connecticut|Massachusetts))?",
        text,
        flags=re.I,
    )
    if m:
        return m.group(0).strip(" ,")
    return None


def extract_city_only(text: str) -> Optional[str]:
    text = norm_text(text).replace(",", " ")
    if not text or any(ch.isdigit() for ch in text):
        return None
    state = is_ct_or_ma(text)
    if state:
        text = re.sub(r"\b(ct|connecticut|ma|mass|massachusetts)\b", "", text, flags=re.I).strip()
    if not text:
        return None
    if re.search(r"\b(" + "|".join(STREET_SUFFIXES) + r")\b", text, flags=re.I):
        return None
    return " ".join(part.capitalize() for part in text.split())


def update_address_state(sched: Dict[str, Any]) -> None:
    sched.setdefault("raw_address", None)
    sched.setdefault("normalized_address", None)
    sched.setdefault("address_verified", False)
    sched.setdefault("address_missing", None)

    norm = sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None
    if norm:
        line1 = (norm.get("address_line_1") or "").strip()
        city = (norm.get("locality") or "").strip()
        state = (norm.get("administrative_district_level_1") or "").strip()
        zip_code = (norm.get("postal_code") or "").strip()
        if line1 and city and state and zip_code and re.match(r"^\d{1,6}\b", line1):
            sched["address_verified"] = True
            sched["address_missing"] = None
            return
        if line1 and city and state and zip_code and not re.match(r"^\d{1,6}\b", line1):
            sched["address_verified"] = False
            sched["address_missing"] = "number"
            return

    raw = (sched.get("raw_address") or "").strip()
    if not raw:
        sched["address_verified"] = False
        sched["address_missing"] = "street"
        return

    low = raw.lower()
    has_num = bool(re.search(r"^\d{1,6}\b", raw))
    has_suffix = bool(re.search(r"\b(" + "|".join(STREET_SUFFIXES) + r")\b", low))
    has_state = bool(is_ct_or_ma(raw))

    if has_suffix and not has_num:
        sched["address_verified"] = False
        sched["address_missing"] = "number"
        return

    if has_num and has_suffix:
        # Needs town/state confirmation if not normalized yet.
        if has_state:
            sched["address_missing"] = "confirm"
        else:
            sched["address_missing"] = "confirm"
        sched["address_verified"] = False
        return

    if extract_city_only(raw):
        sched["address_verified"] = False
        sched["address_missing"] = "street"
        return

    sched["address_verified"] = False
    sched["address_missing"] = "street"


def merge_partial_address_reply(sched: Dict[str, Any], inbound_text: str) -> bool:
    inbound = norm_text(inbound_text)
    if not inbound or not sched.get("raw_address"):
        return False
    if extract_email(inbound):
        return False
    first_name_guess, last_name_guess = extract_name(inbound)
    if first_name_guess and last_name_guess and not is_ct_or_ma(inbound):
        return False

    update_address_state(sched)
    missing = sched.get("address_missing")
    last_prompt = lower_text(sched.get("last_prompt") or "")

    if missing == "state":
        state = is_ct_or_ma(inbound)
        if not state:
            return False
        raw = sched["raw_address"]
        if re.search(r",\s*(CT|MA)\b", raw, flags=re.I):
            raw = re.sub(r",\s*(CT|MA)\b", f", {state}", raw, flags=re.I)
        else:
            raw = f"{raw}, {state}"
        sched["raw_address"] = raw
        return True

    if missing in {"confirm", "street"}:
        city = extract_city_only(inbound)
        state = is_ct_or_ma(inbound)
        allow_city_merge = bool(state) or ("what town" in last_prompt or "which town" in last_prompt)
        if city and allow_city_merge:
            raw = sched["raw_address"]
            if city.lower() not in raw.lower():
                raw = f"{raw}, {city}"
            if state and state.lower() not in raw.lower():
                raw = f"{raw}, {state}"
            sched["raw_address"] = raw
            return True
    return False


def recompute_pending_step(conv: Dict[str, Any]) -> None:
    profile = conv["profile"]
    sched = conv["sched"]
    update_address_state(sched)

    if sched.get("booking_created") and sched.get("square_booking_id"):
        sched["pending_step"] = None
        return

    if not conv["current_job"].get("raw_description") and not sched.get("appointment_type"):
        sched["pending_step"] = "need_issue"
        return

    if sched.get("appointment_type") == "TROUBLESHOOT_395" and sched.get("scheduled_time") and not sched.get("scheduled_date"):
        sched["scheduled_date"] = now_local().strftime("%Y-%m-%d")

    if not sched.get("appointment_type"):
        sched["pending_step"] = None
        return

    if not sched.get("scheduled_date"):
        sched["pending_step"] = "need_date"
        return

    if not sched.get("scheduled_time"):
        sched["pending_step"] = "need_time"
        return

    if not sched.get("address_verified"):
        missing = sched.get("address_missing")
        sched["pending_step"] = "need_state" if missing == "state" else "need_address"
        return

    if not ((profile.get("first_name") or "").strip() and (profile.get("last_name") or "").strip()):
        sched["pending_step"] = "need_name"
        return

    if not (profile.get("email") or "").strip():
        sched["pending_step"] = "need_email"
        return

    sched["pending_step"] = None


def build_address_prompt(sched: Dict[str, Any], convo_key: str) -> str:
    update_address_state(sched)
    missing = sched.get("address_missing") or "street"
    raw = (sched.get("raw_address") or "").strip()

    if missing == "number" and raw:
        return deterministic_pick(f"{convo_key}:addr:number:{raw}", [
            f"What’s the house number on {raw}?",
            f"What number is the place on {raw}?",
            f"What’s the street number on {raw}?",
        ])

    if missing == "state":
        return "Just to confirm, is this address in Connecticut or Massachusetts?"

    if missing == "confirm" and raw:
        return deterministic_pick(f"{convo_key}:addr:confirm:{raw}", [
            f"What town is {raw} in?",
            f"Which town is {raw} in?",
            f"What town is that address in?",
        ])

    return deterministic_pick(f"{convo_key}:addr:street", [
        "What’s the address for the visit?",
        "What address are we heading to?",
        "What’s the service address?",
    ])


# ---------------------------------------------------
# Booking and post-booking responses
# ---------------------------------------------------
def apply_price_injection(appointment_type: str, message: str) -> str:
    if not message:
        return message
    appt = (appointment_type or "").upper()
    if appt == "TROUBLESHOOT_395":
        return f"Troubleshoot and repair visits are $395. {message}" if "$395" not in message else message
    if appt == "WHOLE_HOME_INSPECTION":
        return f"Whole-home inspections run $375 to $650 depending on house size. {message}" if "$375" not in message else message
    return f"The service visit is $195. {message}" if "$195" not in message else message


def post_booking_reply(conv: Dict[str, Any], inbound_text: str) -> str:
    sched = conv["sched"]
    appt = sched.get("appointment_type") or "EVAL_195"
    low = lower_text(inbound_text)

    if is_ack_only(inbound_text):
        return ""

    if any(x in low for x in ["what time", "when am i booked", "are we good", "am i booked", "are we all set"]):
        return f"You’re all set for {humanize_date(sched.get('scheduled_date'))} at {humanize_time(sched.get('scheduled_time'))}."

    if any(x in low for x in ["eta", "on the way", "how close", "when will they arrive"]):
        return "We’ll text when we’re on the way."

    if any(x in low for x in ["reschedule", "change", "move it", "different time", "different day"]):
        sched["booking_created"] = False
        sched["square_booking_id"] = None
        sched["post_booking_open"] = False
        sched["scheduled_date"] = None
        sched["scheduled_time"] = None
        sched["pending_step"] = "need_date"
        return "No problem. What day works best for the new appointment?"

    if any(x in low for x in ["cancel", "never mind", "dont come", "don't come"]):
        return "Okay. If you want to cancel it fully, just reply cancel appointment and I’ll treat this thread as closed."

    if any(x in low for x in ["how much", "price", "cost", "do you charge", "cash", "card", "pay"]):
        if appt == "TROUBLESHOOT_395":
            return "It is still $395 for the troubleshoot visit. Card or cash after the visit is fine."
        if appt == "WHOLE_HOME_INSPECTION":
            return "Whole-home inspections run $375 to $650 depending on house size. Payment is handled after the visit."
        return "It is still $195 for the visit. Payment is handled after the visit."

    if any(x in low for x in ["how long", "how long does it take", "how long will it be"]):
        if appt == "TROUBLESHOOT_395":
            return "Most minor issues are handled during the troubleshoot visit, and we’re usually able to diagnose within the first hour."
        return "The visit is usually about an hour, depending on what you have going on."

    if any(x in low for x in ["licensed", "insured", "service area", "do you service", "do you work in"]):
        return "Yes, we’re licensed and insured, and we service Connecticut and nearby Massachusetts areas we cover."

    return "You’re all set. If anything changes before the visit, just text me here."


def should_treat_as_new_booking_after_booked(text: str) -> bool:
    low = lower_text(text)
    if detect_emergency(text):
        return True
    if likely_address_text(text):
        return True
    if extract_date(text) or extract_time(text)[0]:
        return True
    return any(x in low for x in ["new appointment", "another appointment", "different property"])


# ---------------------------------------------------
# Twilio/voice helpers
# ---------------------------------------------------
def send_sms(to_number: str, body: str) -> None:
    if not body:
        return
    if not twilio_client:
        print("[WARN] Twilio not configured. SMS not sent:", body)
        return
    try:
        from_number = TWILIO_FROM_NUMBER or "whatsapp:+14155238886"
        to_number = to_number if to_number.startswith("whatsapp:") else to_number
        twilio_client.messages.create(body=body, from_=from_number, to=to_number)
    except Exception as exc:
        print("[ERROR] send_sms failed:", repr(exc))


def transcribe_recording(recording_url: str) -> str:
    if not recording_url or not openai_client:
        return ""
    wav_url = recording_url + ".wav"
    mp3_url = recording_url + ".mp3"

    def download(url: str):
        try:
            resp = requests.get(url, stream=True, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=12)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except Exception:
            return None

    resp = download(wav_url) or download(mp3_url)
    if not resp:
        return ""

    tmp_path = "/tmp/prevolt_voicemail.wav"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return (transcript.text or "").strip()
    except Exception as exc:
        print("[ERROR] Whisper transcription failed:", repr(exc))
        return ""


def clean_transcript_text(raw_text: str) -> str:
    raw_text = norm_text(raw_text)
    if not raw_text or not openai_client:
        return raw_text
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Clean up voicemail transcription errors without changing meaning. Return plain text only."},
                {"role": "user", "content": raw_text},
            ],
        )
        return norm_text(completion.choices[0].message.content or raw_text)
    except Exception:
        return raw_text


# ---------------------------------------------------
# Square and Maps helpers
# ---------------------------------------------------
def square_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def normalize_address(raw_address: str, forced_state: Optional[str] = None) -> Tuple[str, Optional[dict]]:
    if not GOOGLE_MAPS_API_KEY or not raw_address:
        return "error", None
    try:
        params = {"address": raw_address, "key": GOOGLE_MAPS_API_KEY}
        params["components"] = f"country:US|administrative_area:{forced_state}" if forced_state else "country:US"
        resp = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=8)
        data = resp.json() or {}
        if data.get("status") != "OK" or not data.get("results"):
            return "error", None

        comps = data["results"][0].get("address_components", [])
        line1 = None
        city = None
        state = None
        zip_code = None
        for comp in comps:
            types = comp.get("types", [])
            if "street_number" in types:
                line1 = comp["long_name"]
            if "route" in types:
                line1 = f"{line1} {comp['long_name']}" if line1 else comp["long_name"]
            if "locality" in types:
                city = comp["long_name"]
            if "administrative_area_level_1" in types:
                state = comp["short_name"]
            if "postal_code" in types:
                zip_code = comp["long_name"]

        final_state = forced_state or state
        if not final_state or final_state not in {"CT", "MA"}:
            return "needs_state", None
        if not (line1 and city and zip_code):
            return "error", None

        addr = {
            "address_line_1": line1,
            "locality": city,
            "administrative_district_level_1": final_state,
            "postal_code": zip_code,
            "country": "US",
        }
        return "ok", addr
    except Exception as exc:
        print("[ERROR] normalize_address:", repr(exc))
        return "error", None


def try_normalize_now(sched: Dict[str, Any]) -> None:
    update_address_state(sched)
    raw = (sched.get("raw_address") or "").strip()
    if not raw or sched.get("address_verified"):
        return
    forced_state = is_ct_or_ma(raw)
    status, addr = normalize_address(raw, forced_state=forced_state)
    if status == "ok" and addr:
        sched["normalized_address"] = addr
    elif status == "needs_state":
        sched["address_missing"] = "state"
    update_address_state(sched)


def compute_travel_time_minutes(origin: str, destination: str) -> Optional[int]:
    if not GOOGLE_MAPS_API_KEY or not origin or not destination:
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params={
                "origins": origin,
                "destinations": destination,
                "key": GOOGLE_MAPS_API_KEY,
                "units": "imperial",
            },
            timeout=8,
        )
        data = resp.json() or {}
        rows = data.get("rows") or []
        elements = rows[0].get("elements") if rows else []
        if not elements or elements[0].get("status") != "OK":
            return None
        seconds = int(elements[0]["duration"]["value"])
        return max(1, round(seconds / 60))
    except Exception as exc:
        print("[WARN] compute_travel_time_minutes failed:", repr(exc))
        return None


def square_lookup_customer_by_phone(phone: str) -> Optional[dict]:
    if not SQUARE_ACCESS_TOKEN:
        return None
    try:
        payload = {"query": {"filter": {"phone_number": {"exact": phone}}}}
        resp = requests.post("https://connect.squareup.com/v2/customers/search", json=payload, headers=square_headers(), timeout=10)
        if resp.status_code not in (200, 201):
            return None
        customers = (resp.json() or {}).get("customers") or []
        return customers[0] if customers else None
    except Exception as exc:
        print("[WARN] square_lookup_customer_by_phone failed:", repr(exc))
        return None


def square_create_or_get_customer(phone: str, profile: dict, addr_struct: Optional[dict]) -> Optional[str]:
    existing = square_lookup_customer_by_phone(phone)
    if existing and existing.get("id"):
        cid = existing["id"]
        if not profile.get("first_name"):
            profile["first_name"] = existing.get("given_name")
        if not profile.get("last_name"):
            profile["last_name"] = existing.get("family_name")
        if not profile.get("email"):
            profile["email"] = existing.get("email_address")
        profile["square_customer_id"] = cid
        return cid

    try:
        payload = {
            "idempotency_key": str(uuid.uuid4()),
            "phone_number": phone,
            "given_name": (profile.get("first_name") or "Prevolt").strip() or "Prevolt",
            "family_name": (profile.get("last_name") or "Lead").strip() or "Lead",
        }
        if profile.get("email"):
            payload["email_address"] = profile["email"]
        if addr_struct:
            payload["address"] = addr_struct

        resp = requests.post("https://connect.squareup.com/v2/customers", json=payload, headers=square_headers(), timeout=10)
        if resp.status_code not in (200, 201):
            print("[ERROR] square customer create failed:", resp.status_code, resp.text)
            return None
        cid = ((resp.json() or {}).get("customer") or {}).get("id")
        if cid:
            profile["square_customer_id"] = cid
        return cid
    except Exception as exc:
        print("[ERROR] square_create_or_get_customer failed:", repr(exc))
        return None


def parse_local_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        t = datetime.strptime(time_str, "%H:%M").time()
        combined = datetime.combine(d, t).replace(tzinfo=LOCAL_TZ)
        if combined < now_local():
            return None
        return combined.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def map_appointment_type_to_variation(appt: str) -> Tuple[Optional[str], Optional[int]]:
    appt = (appt or "").upper()
    if "INSPECTION" in appt:
        return SERVICE_VARIATION_INSPECTION_ID, SERVICE_VARIATION_INSPECTION_VERSION
    if "TROUBLESHOOT" in appt or "REPAIR" in appt:
        return SERVICE_VARIATION_TROUBLESHOOT_ID, SERVICE_VARIATION_TROUBLESHOOT_VERSION
    if "EVAL" in appt:
        return SERVICE_VARIATION_EVAL_ID, SERVICE_VARIATION_EVAL_VERSION
    return None, None


def maybe_create_square_booking(phone: str, conv: Dict[str, Any]) -> bool:
    sched = conv["sched"]
    profile = conv["profile"]

    if sched.get("booking_created") and sched.get("square_booking_id"):
        return True

    recompute_pending_step(conv)
    if sched.get("pending_step") is not None:
        return False

    if not (SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID and SQUARE_TEAM_MEMBER_ID):
        print("[ERROR] Square not configured")
        return False

    if not sched.get("address_verified"):
        return False

    addr_struct = sched.get("normalized_address")
    if not isinstance(addr_struct, dict):
        return False

    origin = TECH_CURRENT_ADDRESS or DISPATCH_ORIGIN_ADDRESS
    if origin:
        destination = f"{addr_struct.get('address_line_1')}, {addr_struct.get('locality')}, {addr_struct.get('administrative_district_level_1')} {addr_struct.get('postal_code')}"
        travel_minutes = compute_travel_time_minutes(origin, destination)
        if travel_minutes and travel_minutes > MAX_TRAVEL_MINUTES:
            print("[BLOCKED] Travel too long:", travel_minutes)
            return False

    variation_id, variation_version = map_appointment_type_to_variation(sched.get("appointment_type"))
    if not variation_id:
        return False

    start_at_utc = parse_local_datetime(sched["scheduled_date"], sched["scheduled_time"])
    if not start_at_utc:
        return False

    customer_id = square_create_or_get_customer(phone, profile, addr_struct)
    if not customer_id:
        return False

    booking_address = dict(addr_struct)
    booking_address.pop("country", None)

    payload = {
        "idempotency_key": f"prevolt-{phone}-{sched['scheduled_date']}-{sched['scheduled_time']}-{variation_id}",
        "booking": {
            "location_id": SQUARE_LOCATION_ID,
            "customer_id": customer_id,
            "start_at": start_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location_type": "CUSTOMER_LOCATION",
            "address": booking_address,
            "appointment_segments": [{
                "duration_minutes": 60,
                "service_variation_id": variation_id,
                "service_variation_version": variation_version,
                "team_member_id": SQUARE_TEAM_MEMBER_ID,
            }],
            "customer_note": f"Auto-booked by Prevolt OS. Raw address: {sched.get('raw_address') or '[none]'}",
        },
    }

    try:
        resp = requests.post("https://connect.squareup.com/v2/bookings", json=payload, headers=square_headers(), timeout=12)
        if resp.status_code not in (200, 201):
            print("[ERROR] Square booking failed:", resp.status_code, resp.text)
            return False
        booking = (resp.json() or {}).get("booking") or {}
        booking_id = booking.get("id")
        if not booking_id:
            return False

        get_resp = requests.get(f"https://connect.squareup.com/v2/bookings/{booking_id}", headers=square_headers(), timeout=12)
        if get_resp.status_code not in (200, 201):
            return False
        booking_obj = (get_resp.json() or {}).get("booking") or {}
        version = booking_obj.get("version")
        status = (booking_obj.get("status") or "").upper()

        if status not in {"ACCEPTED", "CONFIRMED"}:
            accept_payload = {"booking": {"version": version, "status": "ACCEPTED"}}
            put_resp = requests.put(f"https://connect.squareup.com/v2/bookings/{booking_id}", json=accept_payload, headers=square_headers(), timeout=12)
            if put_resp.status_code not in (200, 201):
                print("[ERROR] Square accept failed:", put_resp.status_code, put_resp.text)
                return False

        sched["booking_created"] = True
        sched["square_booking_id"] = booking_id
        sched["post_booking_open"] = True
        profile["upcoming_appointment"] = {
            "date": sched["scheduled_date"],
            "time": sched["scheduled_time"],
            "type": sched["appointment_type"],
            "square_id": booking_id,
        }
        return True
    except Exception as exc:
        print("[ERROR] maybe_create_square_booking exception:", repr(exc))
        return False


# ---------------------------------------------------
# Core deterministic engine
# ---------------------------------------------------
def ingest_user_message(conv: Dict[str, Any], inbound_text: str, convo_key: str) -> None:
    profile = conv["profile"]
    job = conv["current_job"]
    sched = conv["sched"]

    inbound_text = norm_text(inbound_text)
    low = lower_text(inbound_text)
    sched["last_user_message"] = inbound_text

    if not inbound_text:
        return

    # Reset / start over
    if low in {"start over", "reset", "nvm let’s start over", "nvm let's start over", "new thread"}:
        appointment_type = sched.get("appointment_type")
        preserved_profile = profile.copy()
        new_conv = empty_conversation()
        new_conv["profile"].update(preserved_profile)
        new_conv["sched"]["appointment_type"] = appointment_type
        conv.clear()
        conv.update(new_conv)
        return

    # If already booked, keep post-booking open unless message clearly starts a new booking.
    if sched.get("booking_created") and sched.get("square_booking_id") and should_treat_as_new_booking_after_booked(inbound_text):
        existing_type = sched.get("appointment_type")
        existing_profile = profile.copy()
        conv.clear()
        conv.update(empty_conversation())
        conv["profile"].update(existing_profile)
        conv["sched"]["appointment_type"] = existing_type
        profile = conv["profile"]
        job = conv["current_job"]
        sched = conv["sched"]

    if detect_emergency(inbound_text):
        sched["appointment_type"] = "TROUBLESHOOT_395"
        sched["emergency_approved"] = True
        sched["awaiting_emergency_confirm"] = False
        if re.search(r"\b(now|right now|immediately|asap)\b", low) and not sched.get("scheduled_date"):
            sched["scheduled_date"] = now_local().strftime("%Y-%m-%d")
    else:
        sched["appointment_type"] = classify_appointment_type(inbound_text, sched.get("appointment_type"))

    if not job.get("raw_description"):
        job["raw_description"] = inbound_text
        job["job_type"] = sched.get("appointment_type")

    # Name / email capture
    if not (profile.get("email") or "").strip():
        email = extract_email(inbound_text)
        if email:
            profile["email"] = email

    if not ((profile.get("first_name") or "").strip() and (profile.get("last_name") or "").strip()):
        if sched.get("pending_step") == "need_name" or re.search(r"\b(my name is|name is|this is)\b", low) or (len(inbound_text.split()) in {2, 3} and not likely_address_text(inbound_text) and not extract_email(inbound_text)):
            first, last = extract_name(inbound_text)
            if first and last:
                profile["first_name"] = first
                profile["last_name"] = last

    # Address capture
    merged = False
    if sched.get("pending_step") in {"need_address", "need_state"}:
        merged = merge_partial_address_reply(sched, inbound_text)
    addr = extract_address_candidate(inbound_text)
    if addr:
        sched["raw_address"] = addr
    elif merged:
        pass
    elif sched.get("pending_step") in {"need_address", "need_state"} and sched.get("address_missing") in {"confirm", "state"}:
        first_name_guess, last_name_guess = extract_name(inbound_text)
        if not extract_email(inbound_text) and not (first_name_guess and last_name_guess and not is_ct_or_ma(inbound_text)):
            city = extract_city_only(inbound_text)
            state = is_ct_or_ma(inbound_text)
            if city and sched.get("raw_address"):
                sched["raw_address"] = f"{sched['raw_address']}, {city}" + (f", {state}" if state else "")
            elif state and sched.get("raw_address") and state.lower() not in sched["raw_address"].lower():
                sched["raw_address"] = f"{sched['raw_address']}, {state}"

    # Date capture
    date_value = extract_date(inbound_text)
    if date_value:
        sched["scheduled_date"] = date_value

    # Time capture
    time_value, ambiguous_time = extract_time(inbound_text, sched.get("appointment_type"))
    if time_value:
        sched["scheduled_time"] = time_value
    elif ambiguous_time:
        # Keep the date if user gave a weekday/date but did not give a real time.
        if sched.get("pending_step") != "need_date":
            pass

    # Window enforcement for non-emergency
    if sched.get("scheduled_time") and sched.get("appointment_type") != "TROUBLESHOOT_395":
        try:
            hh = int(sched["scheduled_time"].split(":")[0])
            if hh < BOOKING_START_HOUR:
                sched["scheduled_time"] = f"{BOOKING_START_HOUR:02d}:00"
            elif hh >= BOOKING_END_HOUR:
                sched["scheduled_time"] = f"{BOOKING_END_HOUR:02d}:00"
        except Exception:
            pass

    # Early normalize whenever we have enough address text.
    try_normalize_now(sched)
    recompute_pending_step(conv)


def build_reply(conv: Dict[str, Any], inbound_text: str, convo_key: str) -> str:
    profile = conv["profile"]
    sched = conv["sched"]
    job = conv["current_job"]
    recompute_pending_step(conv)

    if sched.get("booking_created") and sched.get("square_booking_id"):
        msg = post_booking_reply(conv, inbound_text)
        return shorten_for_texting(strip_ai_punctuation(msg))

    # Price reveal once, early but not spammy.
    reveal_price_now = not sched.get("price_disclosed") and (
        sched.get("appointment_type") is not None or any(x in lower_text(inbound_text) for x in ["price", "cost", "quote", "estimate"])
    )

    step = sched.get("pending_step")
    if step == "need_issue":
        msg = QUESTION_FALLBACKS[step]
    elif step == "need_address":
        msg = build_address_prompt(sched, convo_key)
    elif step == "need_state":
        msg = QUESTION_FALLBACKS[step]
    elif step == "need_date":
        sched["date_ask_count"] += 1
        msg = "What day works best for you?"
    elif step == "need_time":
        sched["time_ask_count"] += 1
        if sched.get("appointment_type") == "TROUBLESHOOT_395" and sched.get("scheduled_date") == now_local().strftime("%Y-%m-%d"):
            part = "morning" if now_local().hour < 12 else ("afternoon" if now_local().hour < 17 else "evening")
            msg = f"We can come today. What time later this {part} works for you?"
        else:
            msg = "What time works best?"
    elif step == "need_name":
        sched["name_ask_count"] += 1
        msg = QUESTION_FALLBACKS[step]
    elif step == "need_email":
        sched["email_ask_count"] += 1
        msg = QUESTION_FALLBACKS[step]
    else:
        booked = maybe_create_square_booking(get_convo_key(), conv)
        if booked:
            msg = f"You’re all set for {humanize_date(sched.get('scheduled_date'))} at {humanize_time(sched.get('scheduled_time'))}. If anything changes before the visit, just text me here."
        else:
            # If everything is present but booking did not complete, fail safely without fake confirmation.
            recompute_pending_step(conv)
            if sched.get("pending_step"):
                if sched["pending_step"] == "need_address":
                    msg = build_address_prompt(sched, convo_key)
                else:
                    msg = QUESTION_FALLBACKS.get(sched["pending_step"], "What time works best?")
            else:
                # Postpone if Square unavailable.
                msg = f"I have {humanize_date(sched.get('scheduled_date'))} at {humanize_time(sched.get('scheduled_time'))} and the service address noted. Please reply with yes to keep that appointment request."

    if reveal_price_now:
        msg = apply_price_injection(sched.get("appointment_type") or "EVAL_195", msg)
        sched["price_disclosed"] = True

    # Troubleshoot reassurance, once only, if user asks or seems hesitant.
    if sched.get("appointment_type") == "TROUBLESHOOT_395" and not sched.get("reassurance_used"):
        low = lower_text(inbound_text)
        if any(x in low for x in ["how long", "what happens", "what do you do", "worried", "nervous", "is it bad"]):
            msg = f"Most minor issues are handled during the troubleshoot visit, and we’re usually able to diagnose within the first hour. {msg}"
            sched["reassurance_used"] = True

    # Avoid repeating the same prompt forever.
    final_msg = shorten_for_texting(strip_ai_punctuation(msg))
    if final_msg == sched.get("last_prompt"):
        sched["last_prompt_count"] += 1
    else:
        sched["last_prompt"] = final_msg
        sched["last_prompt_count"] = 1

    if sched.get("last_prompt_count", 0) >= 3 and sched.get("pending_step") == "need_time":
        if sched.get("appointment_type") == "TROUBLESHOOT_395":
            default_time = round_up_to_next_half_hour()
        else:
            default_time = "09:00"
        sched["scheduled_time"] = default_time
        recompute_pending_step(conv)
        final_msg = f"I’ll use {humanize_time(default_time)} unless you want a different time."

    if sched.get("last_prompt_count", 0) >= 3 and sched.get("pending_step") == "need_date":
        next_day = (now_local() + timedelta(days=1)).strftime("%Y-%m-%d")
        sched["scheduled_date"] = next_day
        recompute_pending_step(conv)
        final_msg = f"I’ll use {humanize_date(next_day)} unless you want a different day."

    conv["last_sms_body"] = final_msg
    return final_msg


# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/", methods=["GET", "HEAD"])
def home():
    return "Prevolt OS running", 200


@app.route("/incoming-call", methods=["GET", "POST"])
def incoming_call():
    response = VoiceResponse()
    gather = Gather(num_digits=1, action="/handle-call-selection", method="POST", timeout=6)
    gather.say(
        '<speak><prosody rate="95%">Thanks for calling Prevolt Electric. If you are a residential customer, press 1. If you are a commercial, government, or facility customer, press 2.</prosody></speak>',
        voice="Polly.Matthew-Neural",
    )
    response.append(gather)
    response.redirect("/incoming-call")
    return Response(str(response), mimetype="text/xml")


@app.route("/handle-call-selection", methods=["POST"])
def handle_call_selection():
    digit = request.form.get("Digits", "")
    convo_key = get_convo_key(is_call=True)
    conv = ensure_conv(convo_key)
    profile = conv["profile"]

    response = VoiceResponse()
    if digit == "1":
        profile["customer_type"] = "residential"
        response.say(
            '<speak><prosody rate="95%">Please leave your name, address, and a brief description of what you need help with. We will text you shortly.</prosody></speak>',
            voice="Polly.Matthew-Neural",
        )
        response.record(max_length=60, play_beep=True, trim="do-not-trim", action="/voicemail-complete", method="POST")
        response.hangup()
        return Response(str(response), mimetype="text/xml")

    if digit == "2":
        profile["customer_type"] = "commercial"
        response.say('<speak><prosody rate="90%">Connecting you now.</prosody></speak>', voice="Polly.Matthew-Neural")
        if TWILIO_FROM_NUMBER:
            response.dial(TWILIO_FROM_NUMBER)
        else:
            response.hangup()
        return Response(str(response), mimetype="text/xml")

    response.redirect("/incoming-call")
    return Response(str(response), mimetype="text/xml")


@app.route("/voicemail-complete", methods=["POST"])
def voicemail_complete():
    recording_url = request.form.get("RecordingUrl")
    phone = (request.form.get("From", "") or "").replace("whatsapp:", "")
    conv = ensure_conv(phone)

    transcript = clean_transcript_text(transcribe_recording(recording_url or ""))
    conv["cleaned_transcript"] = transcript
    conv["initial_sms"] = transcript
    if transcript:
        ingest_user_message(conv, transcript, phone)
        reply = build_reply(conv, transcript, phone)
        if phone and reply:
            send_sms(phone, reply)

    response = VoiceResponse()
    response.say("Thank you. Your message has been recorded.")
    response.hangup()
    return Response(str(response), mimetype="text/xml")


@app.route("/incoming-sms", methods=["POST"])
def incoming_sms():
    inbound_text = request.form.get("Body", "") or ""
    convo_key = get_convo_key()

    if lower_text(inbound_text) == "mobius1":
        conversations[convo_key] = empty_conversation()
        tw = MessagingResponse()
        tw.message("Memory reset complete for this number.")
        return Response(str(tw), mimetype="text/xml")

    conv = ensure_conv(convo_key)
    ingest_user_message(conv, inbound_text, convo_key)
    reply = build_reply(conv, inbound_text, convo_key)

    tw = MessagingResponse()
    if reply:
        tw.message(reply)
    else:
        tw.message("")
    return Response(str(tw), mimetype="text/xml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
