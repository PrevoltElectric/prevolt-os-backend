import os
import json
import sqlite3
import time
import uuid
from pathlib import Path
import requests
from datetime import datetime, timezone, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather, Dial
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from openai import OpenAI



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


# ---------------------------------------------------
# Google LSA relay sanitation + price firewall
# ---------------------------------------------------
def extract_lsa_payload_fields(text: str) -> dict:
    """Extract stable metadata from Google Local Services Ads relay wrappers."""
    raw = str(text or "")
    fields = {}
    patterns = {
        "customer_name": r"\bCustomer Name:\s*(.+?)(?=\s+(?:Location|Service|Message):|$)",
        "location": r"\bLocation:\s*(.+?)(?=\s+(?:Service|Message):|$)",
        "service": r"\bService:\s*(.+?)(?=\s+Message:|$)",
        "message": r"\bMessage:\s*(.+)$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, raw, flags=re.I | re.S)
        if m:
            value = " ".join(m.group(1).split()).strip()
            if value:
                fields[key] = value
    return fields


def strip_lsa_customer_text_noise(text: str) -> str:
    """
    Remove Google LSA wrapper fragments, notes, quoted email history, and obvious
    email-signature debris before the text enters the normal booking brain.
    """
    raw = str(text or "").replace("\r", "\n").strip()
    if not raw:
        return ""

    # Remove Google/LSA notes appended to the customer-authored sentence.
    raw = re.sub(r"\s*\[Notes?\s+from\s+LSA:.*?\]\s*", " ", raw, flags=re.I | re.S)

    # Cut off relay/dashboard instructions and quoted email boilerplate.
    cut_patterns = [
        r"\bTo respond,\s*just reply to this email\b",
        r"\bReply here or respond via\b",
        r"\bReplies to this number\b",
        r"\brespond via your LSA dashboard\b",
        r"https://g\.co/homeservices\S*",
        r"\bNeed help\?\s*We are here for you\b",
        r"\bGoogle LLC\b",
        r"\bYou received this mandatory service announcement\b",
        r"\bCONFIDENTIAL\s*:",
        r"\bThis email and any files transmitted\b",
    ]
    for pat in cut_patterns:
        raw = re.split(pat, raw, maxsplit=1, flags=re.I | re.S)[0].strip()

    # Drop common signature-ish lines while preserving the customer's message.
    kept = []
    for line in raw.splitlines():
        line_clean = " ".join(line.split()).strip()
        low = line_clean.lower()
        if not line_clean:
            continue
        if low in {"thanks,", "thanks", "prevolt", "sent from my iphone"}:
            continue
        if low.startswith(("kyle prevost", "owner", "office:", "website:", "now accepting online bookings")):
            continue
        if "prevoltelectric@gmail.com" in low or "prevoltllc.com" in low:
            continue
        kept.append(line_clean)

    return " ".join(" ".join(kept).split()).strip()


def is_lsa_self_echo_or_dirty_business_reply(text: str) -> bool:
    """
    True when Google relays Prevolt's own dashboard/email response back into
    Twilio. These must never generate another automated reply.
    """
    low = _intent_text(text)
    if not low:
        return False
    return (
        "prevolt sent you a response" in low
        or "prevolt wrote" in low
        or (
            "kyle prevost" in low
            and ("prevoltelectric gmail com" in low or "prevoltllc com" in low)
        )
        or (
            "office 860 758 0707" in low
            and "prevolt" in low
        )
    )


def is_google_lsa_thread(conv: dict | None, text: str = "") -> bool:
    conv = conv or {}
    return (
        (conv.get("source") or "").strip() == "google_lsa"
        or is_google_lsa_platform_notice(text)
        or bool(extract_lsa_customer_message(text))
    )


def apply_lsa_customer_name_to_profile(conv: dict, customer_name: str | None) -> None:
    """Store the LSA form name, but only trust a full non-generic name for booking identity."""
    name = " ".join(str(customer_name or "").split()).strip()
    if not name:
        return
    profile = conv.setdefault("profile", {})
    profile["lsa_customer_name"] = name

    # Do not overwrite a customer-provided name later in the booking flow.
    if profile.get("identity_source") == "customer_provided":
        return

    cleaned = re.sub(r"[^A-Za-z\-\'\s]", " ", name)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return

    parts = cleaned.split()
    low = cleaned.lower().strip()
    generic_names = {
        "john doe", "jane doe", "test test", "test user", "testing testing",
        "first last", "first name", "last name", "na na", "n a"
    }

    # LSA often gives only a first name. Do not lock that into active identity,
    # because a typo in Google's form becomes an awkward prompt like
    # "Kile, what is your last name?" Store it only as metadata until the
    # customer confirms a full name in the booking flow.
    if len(parts) < 2 or low in generic_names:
        return

    if not (profile.get("active_first_name") or profile.get("first_name")):
        profile["active_first_name"] = parts[0]
        profile["first_name"] = parts[0]
        profile["identity_source"] = profile.get("identity_source") or "lsa_form_name"

    if not (profile.get("active_last_name") or profile.get("last_name")):
        last = " ".join(parts[1:])
        profile["active_last_name"] = last
        profile["last_name"] = last
        profile["identity_source"] = profile.get("identity_source") or "lsa_form_name"


def maybe_apply_direct_name_correction(conv: dict, inbound_text: str) -> str | None:
    """Handle messages like "it's Toby not John" without restarting intake."""
    raw = " ".join(str(inbound_text or "").split()).strip()
    if not raw:
        return None

    # Keep this narrow so normal service text does not get stolen.
    patterns = [
        r"\b(?:it\s*is|it's|its|this\s+is|my\s+name\s+is|name\s+is|i\s+am|i'm)\s+([A-Za-z][A-Za-z'\-]{1,})\s+(?:not|not\s+the\s+name)\s+([A-Za-z][A-Za-z'\-]{1,})\b",
        r"\b([A-Za-z][A-Za-z'\-]{1,})\s+not\s+([A-Za-z][A-Za-z'\-]{1,})\b",
    ]
    candidate = None
    for pat in patterns:
        m = re.search(pat, raw, flags=re.I)
        if m:
            candidate = normalize_person_name(m.group(1))
            break

    if not candidate:
        return None

    # Guard against capturing scheduling/service words as names.
    if candidate.lower() in {
        "tomorrow", "today", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "address", "email", "appointment",
        "service", "quote", "estimate"
    }:
        return None

    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})

    old_first = (profile.get("active_first_name") or profile.get("first_name") or "").strip()
    old_last = (profile.get("active_last_name") or profile.get("last_name") or "").strip()
    old_source = (profile.get("identity_source") or "").strip()

    profile["active_first_name"] = candidate
    profile["first_name"] = candidate
    profile["identity_source"] = "customer_provided"

    # If the old identity came from a generic/test/Square/voicemail source and
    # the customer is correcting only the first name, do not keep a stale last
    # name attached to the wrong person.
    if old_first and old_first.lower() != candidate.lower():
        if old_last.lower() in {"doe", "test", "user"} or old_source not in {"customer_provided", "customer_provided_full_name"}:
            profile["active_last_name"] = None
            profile["last_name"] = None

    sched["intro_sent"] = True
    sched["name_engine_state"] = None
    sched["name_engine_prompted"] = False
    return candidate


def lsa_has_emergency_signal(text: str) -> bool:
    low = _intent_text(text)
    emergency_terms = [
        "no power", "power out", "outage", "burning", "smoke", "sparking", "sparks",
        "arcing", "melted", "hot outlet", "hot panel", "breaker wont reset",
        "breaker won't reset", "breaker not resetting", "tree ripped", "line down",
        "wire down", "service ripped", "meter ripped", "main breaker"
    ]
    return any(t in low for t in emergency_terms)


def looks_like_lsa_quote_or_service_lead(text: str) -> bool:
    """
    LSA 'Get quote' leads often begin with price/scope language instead of
    scheduling language. Treat those as bookable evaluation leads before the
    general LLM can improvise pricing.
    """
    low = _intent_text(text)
    if not low:
        return False
    if lsa_has_emergency_signal(low):
        return False

    service_terms = [
        "install", "installation", "replace", "replacement", "repair", "rewire",
        "remodel", "charger", "tesla", "ev charger", "wall charger", "outlet",
        "switch", "panel", "service", "meter", "breaker", "dishwasher",
        "garbage disposal", "disposal", "light", "fan", "fixture", "quote",
        "estimate", "electrical"
    ]
    quote_terms = [
        "how much", "cost", "price", "quote", "estimate", "looking for an estimate",
        "looking for a quote", "requested a quote", "looking to", "need", "needs",
        "can you help", "help with", "looking for"
    ]
    scheduling_terms = [
        "availability", "available", "appointment", "schedule", "come out",
        "take a look", "site visit", "visit"
    ]
    return (
        any(t in low for t in service_terms)
        and (any(t in low for t in quote_terms) or any(t in low for t in scheduling_terms))
    )


def build_lsa_eval_entry_reply(conv: dict, inbound_text: str) -> str:
    """
    Safe first response for Google LSA quote/message leads. This should sound
    like the normal Prevolt intake opener. Do NOT pre-argue that $195 is not the
    total job price unless the customer specifically asks that trap question.
    """
    sched = conv.setdefault("sched", {})
    sched["appointment_type"] = sched.get("appointment_type") or "EVAL_195"
    conv["appointment_type"] = sched["appointment_type"]
    sched["price_disclosed"] = True
    sched["intro_sent"] = True
    sched["manual_only"] = False
    sched["non_service_thread"] = False

    try:
        absorb_address_from_mixed_text(conv, inbound_text)
    except Exception:
        pass

    if sched.get("raw_address") or sched.get("address_verified"):
        sched["pending_step"] = "need_date"
        return (
            "Hello, you've reached Prevolt Electric. I'll help you here by text. "
            "Our evaluation visit is $195. What day works best for the evaluation?"
        )

    sched["pending_step"] = "need_address"
    return (
        "Hello, you've reached Prevolt Electric. I'll help you here by text. "
        "Our evaluation visit is $195. What is the service address?"
    )


def build_lsa_safety_reply(conv: dict, inbound_text: str) -> str | None:
    """
    Deterministic firewall for Google LSA price/material/duration questions.
    This runs before the LLM and before normal interruption handling.
    """
    if not is_google_lsa_thread(conv, inbound_text):
        return None

    low = _intent_text(inbound_text)
    if not low:
        return None

    sched = conv.setdefault("sched", {})
    # A deterministic LSA safety reply is customer-facing. Mark the intro/pricing
    # context as already established so Step 4 cannot prepend a fresh greeting on
    # the next relay message.
    sched["intro_sent"] = True
    sched["manual_only"] = False
    sched["non_service_thread"] = False

    def _next_lsa_question(default_eval: str = "What day works best for the evaluation?") -> str:
        if not (sched.get("raw_address") or sched.get("address_verified")):
            sched["pending_step"] = "need_address"
            return "What is the service address?"
        if not sched.get("scheduled_date"):
            sched["pending_step"] = "need_date"
            return default_eval
        if not sched.get("scheduled_time"):
            sched["pending_step"] = "need_time"
            return "What time works best?"
        recompute_pending_step(conv.setdefault("profile", {}), sched)
        return ""

    is_emergency = lsa_has_emergency_signal(inbound_text) or "TROUBLESHOOT" in (sched.get("appointment_type") or "").upper()

    # Only fire the defensive price firewall on trap/follow-up questions.
    # A normal opener like "looking for a quote" or "how much to install" should
    # receive the standard $195 evaluation opener from build_lsa_eval_entry_reply(),
    # not an over-explained correction.
    material_terms = [
        "material", "materials", "include material", "include materials",
        "includes material", "includes materials", "included", "parts"
    ]
    duration_terms = [
        "how long", "how long does it take", "install time", "how many hours",
        "take to install", "duration"
    ]
    total_trap_terms = [
        "total", "total cost", "full price", "full cost", "final price",
        "all in", "all-in", "out the door", "complete price", "complete cost",
        "is that everything", "is that all", "is 195", "is $195",
        "195 the total", "$195 the total", "195 total", "$195 total"
    ]
    money_mentioned = "$195" in low or "195" in low or "$395" in low or "395" in low

    asks_material = any(t in low for t in material_terms) and (
        money_mentioned or any(v in low for v in ["include", "included", "includes", "come with", "cover", "covers"])
    )
    asks_duration = any(t in low for t in duration_terms)
    asks_price = money_mentioned and any(t in low for t in total_trap_terms)

    if not (asks_price or asks_material or asks_duration):
        return None

    if is_emergency:
        sched["appointment_type"] = "TROUBLESHOOT_395"
        conv["appointment_type"] = "TROUBLESHOOT_395"
    else:
        sched["appointment_type"] = sched.get("appointment_type") or "EVAL_195"
        conv["appointment_type"] = sched["appointment_type"]

    if asks_material:
        sched["price_disclosed"] = True
        if is_emergency:
            return (
                "Materials for larger repairs are not included in the $395. "
                "The $395 is the troubleshoot and repair visit, and anything larger is reviewed on site first. "
                f"{_next_lsa_question('What day and time work best?')}"
            )
        return (
            "No. Materials for the installation are not included in the $195. "
            "The $195 is the evaluation and quote visit so we can review everything on site and give you a firm number. "
            f"{_next_lsa_question()}"
        )

    if asks_duration:
        return (
            "Install time depends on the panel, wire path, distance, and site conditions. "
            "We review that during the evaluation visit before giving the final installation price. "
            f"{_next_lsa_question()}"
        )

    if asks_price:
        sched["price_disclosed"] = True
        if is_emergency:
            return (
                "The $395 is for the troubleshoot and repair visit, not a guaranteed total for larger repair work. "
                f"Anything larger is reviewed on site first. {_next_lsa_question('What day and time work best?')}"
            )
        return (
            "The $195 is the evaluation and quote visit, not the total installation price. "
            "Final pricing depends on the panel, wire path, breaker, distance, and materials after we review it on site. "
            f"{_next_lsa_question()}"
        )

    return None


def enforce_lsa_outbound_safety(conv: dict, inbound_text: str, outbound_text: str) -> str:
    """
    Last-chance guard for any LSA response generated outside the deterministic
    path. If a forbidden claim slips through, replace it with the safe answer.
    """
    body = " ".join(str(outbound_text or "").split()).strip()
    if not body or not is_google_lsa_thread(conv, inbound_text):
        return body

    low = _intent_text(body)
    forbidden = (
        ("total cost" in low and ("195" in low or "$195" in body))
        or "includes both the installation and the materials" in low
        or "includes materials" in low
        or "materials are included" in low
        or "usually about an hour" in low
        or "takes about an hour" in low
    )
    if forbidden:
        safe = build_lsa_safety_reply(conv, inbound_text)
        if safe:
            return safe
        return (
            "The $195 is the evaluation and quote visit, not the total installation price. "
            "Final pricing is reviewed on site before any installation work moves forward."
        )

    # Do not re-open an active Google relay thread with a new greeting.
    body = re.sub(
        r"^(?:hello|hi)[^.!?]{0,80}prevolt electric[.!?]\s*",
        "",
        body,
        flags=re.I,
    ).strip()
    body = re.sub(r"^let'?s get this lined up[.!?]?\s*", "", body, flags=re.I).strip()
    return body

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
    if existing in {"employment_inquiry", "commercial_bid_contact", "manual_only"}:
        return existing

    if is_pure_lsa_platform_notice(inbound_text):
        return "platform_wrapper"

    if looks_like_employment_inquiry(history):
        return "employment_inquiry"

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
    sched["manual_only"] = thread_type in {"manual_only", "employment_inquiry", "commercial_bid_contact"}
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
        f"{EMPLOYMENT_RESUME_EMAIL}, and Kyle can review it."
    )

def build_commercial_bid_reply(inbound_text: str = "") -> str:
    low = _intent_text(inbound_text)
    if "email" in low or "sent" in low or "questions" in low:
        return "Got it, thank you. Kyle will review the email and get back to you."
    if "personal" in low or "save this number" in low:
        return "Sounds good, thank you. Kyle appreciates it."
    return "Got it, thank you. Kyle will review this and get back to you."

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
- If they say they sent an email, acknowledge that Kyle will review it.
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
    """Start a normal $195 evaluation booking flow from an inbound SMS/GLS lead."""
    sched = conv.setdefault("sched", {})
    sched["appointment_type"] = "EVAL_195"
    conv["appointment_type"] = "EVAL_195"
    sched["price_disclosed"] = True
    # This helper itself sends the first customer-facing intro line.
    # Mark it sent so later slot/name/email prompts do not prepend
    # "Hello, you've reached Prevolt Electric..." again mid-thread.
    sched["intro_sent"] = True
    sched["pending_step"] = "need_date"
    sched["manual_only"] = False
    sched["non_service_thread"] = False

    try:
        absorb_address_from_mixed_text(conv, inbound_text)
    except Exception:
        pass

    low = _intent_text(inbound_text)
    if "meter bank" in low or "6-gang" in low or "six gang" in low or "gang meter" in low:
        return (
            "Hello, you've reached Prevolt Electric. I'll help you here by text. "
            "For the meter bank replacement, we start with a $195 evaluation visit so we can review everything in person and give you a firm number. "
            "What day works best for you?"
        )

    return (
        "Hello, you've reached Prevolt Electric. I'll help you here by text. "
        "Our evaluation visit is $195. What day works best for you?"
    )


def build_complex_commercial_coordination_reply(inbound_text: str = "") -> str:
    low = _intent_text(inbound_text)
    if "meter bank" in low or "6-gang" in low or "six gang" in low or "gang meter" in low:
        return (
            "Got it, thank you. Kyle will review the walkthrough details and follow up directly to coordinate access and timing."
        )
    if "walk" in low or "site visit" in low:
        return (
            "Got it, thank you. Kyle will review the details and follow up directly to coordinate the commercial walkthrough."
        )
    return (
        "Thanks for reaching out. This type of commercial project is handled directly by Kyle so the scope and walkthrough "
        "are coordinated correctly. Kyle will follow up directly."
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
    suffix = r"st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace|pl|place"
    m = re.search(
        rf"\b(?P<addr>\d{{1,6}}[A-Za-z]?\s+[A-Za-z0-9.'\- ]+?\b(?:{suffix})\b(?:\s+[A-Za-z.'\- ]{{2,40}})?)",
        raw,
        flags=re.I,
    )
    if not m:
        return None
    addr = m.group("addr").strip(" ,.;")
    addr = re.split(
        r"\s*,\s*(?:please|and|we|woodgate|association|condominium|hoa|message|service)\b",
        addr,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" ,.;")
    return addr or None


def absorb_address_from_mixed_text(conv: dict, inbound_text: str) -> bool:
    """Save a service address embedded in a longer customer message."""
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    if sched.get("raw_address"):
        return False
    addr = extract_service_address_from_text(inbound_text)
    if not addr:
        return False
    sched["raw_address"] = addr
    try:
        if addr not in profile.setdefault("addresses", []):
            profile["addresses"].append(addr)
    except Exception:
        pass
    try:
        try_early_address_normalize(sched)
    except Exception:
        pass
    update_address_assembly_state(sched)
    return True


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
        slots = get_next_available_slots(appointment_type, limit=3)
        return _offer_slots_response(conv, slots, "I’m not seeing openings for that day.")

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

@app.route("/", methods=["GET", "HEAD"])
def home():
    return "Prevolt OS running", 200


# ---------------------------------------------------
# In-Memory Conversation Store
# ---------------------------------------------------
conversations = {}

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
    if sched.get("booking_created") and sched.get("square_booking_id"):
        return "booked"
    if "TROUBLESHOOT" in appt or sched.get("emergency_approved") or sched.get("awaiting_emergency_confirm"):
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
    address = (sched.get("raw_address") or "").strip()
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
        "updated_at": _monitor_now_iso(),
    }

def log_event(event_type: str, phone: str = "", payload: dict | None = None, conv: dict | None = None) -> None:
    try:
        init_monitor_db()
        base_payload = dict(payload or {})
        if conv is not None:
            base_payload.setdefault("state", _monitor_state_label(conv))
            base_payload.setdefault("snapshot", _conversation_snapshot(phone, conv))
        with _monitor_connect() as conn:
            conn.execute(
                "INSERT INTO monitor_events (ts, event_type, phone, payload_json) VALUES (?, ?, ?, ?)",
                (_monitor_now_iso(), event_type, phone or "", json.dumps(base_payload, ensure_ascii=False)),
            )
    except Exception as e:
        print("[WARN] monitor log_event failed:", repr(e))

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

    # A bare hour is safe when the whole reply is the hour, or when the user
    # says "at 1" / "at 2" in a scheduling phrase. Do not grab random numbers.
    m = re.fullmatch(r"\s*(?:at\s+)?(\d{1,2})\s*", s)
    if not m:
        m = re.search(r"\bat\s+(\d{1,2})\b", s)
    if m:
        hh = int(m.group(1))
        if 1 <= hh <= 12:
            if hh == 12:
                return "12:00"
            return f"{hh:02d}:00"

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



def recompute_pending_step(profile: dict, sched: dict) -> None:
    if sched.get("non_service_thread") or sched.get("manual_only"):
        sched["pending_step"] = None
        return

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
def looks_like_transcription_placeholder(text: str) -> bool:
    """
    Guard against LLM/meta placeholder text being treated like a real voicemail.
    When Whisper or cleanup has no usable audio/text, the cleanup model can
    otherwise answer with 'Please provide the voicemail transcription...', which
    then causes the classifier to hallucinate a fake customer/name/address.
    """
    low = _intent_text(text)
    if not low:
        return True
    placeholder_phrases = [
        "please provide the voicemail transcription",
        "provide the voicemail transcription",
        "voicemail transcription you need cleaned up",
        "voicemail transcription you'd like me to clean up",
        "transcription you'd like me to clean up",
        "transcription you would like me to clean up",
        "please provide the transcription",
        "no transcription provided",
        "i need the transcription",
        "send the transcription",
    ]
    return any(p in low for p in placeholder_phrases)


def clean_transcript_text(raw_text: str) -> str:
    """
    Minor cleanup only; does NOT change meaning. Never let the cleanup model
    invent a placeholder prompt when the recording/transcription is empty.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text or looks_like_transcription_placeholder(raw_text):
        return ""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You clean up voicemail transcriptions for an electrical contractor. "
                        "Fix transcription errors, preserve meaning exactly. No embellishments. "
                        "If the input is empty or not a voicemail, return an empty string."
                    ),
                },
                {"role": "user", "content": raw_text},
            ],
        )
        cleaned = (completion.choices[0].message.content or "").strip()
        if looks_like_transcription_placeholder(cleaned):
            return ""
        return cleaned
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
    cleaned_text = (cleaned_text or "").strip()
    if not cleaned_text or looks_like_transcription_placeholder(cleaned_text):
        return {
            "category": "EMPTY_VOICEMAIL",
            "appointment_type": "none",
            "detected_first_name": None,
            "detected_last_name": None,
            "detected_address": None,
            "detected_date": None,
            "detected_time": None,
            "intent": "other",
            "task_topics": [],
        }

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
def hydrate_square_profile_by_phone(profile: dict, phone: str) -> None:
    """Best-effort one-time Square hydrate for repeat callers before the first text goes out."""
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

    if profile.get("square_lookup_done"):
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
    if thread_type == "employment_inquiry":
        clear_service_booking_state_for_non_service(conv, "employment_inquiry")
        sched["intro_sent"] = True
        sched["price_disclosed"] = False
        return build_employment_inquiry_reply()
    if thread_type == "commercial_bid_contact":
        clear_service_booking_state_for_non_service(conv, "commercial_bid_contact")
        sched["intro_sent"] = True
        sched["price_disclosed"] = False
        return build_commercial_bid_reply(conv.get("cleaned_transcript") or "")

    first_name = (profile.get("active_first_name") or profile.get("recognized_first_name") or profile.get("voicemail_first_name") or "").strip()
    intro = f"Hello {first_name}, you've reached Prevolt Electric. I'll help you here by text." if first_name else "You've reached Prevolt Electric. I'll help you here by text."

    topics = conv.get("task_topics") or classification.get("task_topics") or []
    topics = [str(t).strip() for t in topics if str(t).strip()]
    task_line = ""
    if topics:
        topics = topics[:4]
        if len(topics) == 1:
            joined = topics[0]
        elif len(topics) == 2:
            joined = f"{topics[0]} and {topics[1]}"
        else:
            joined = ", ".join(topics[:-1]) + f", and {topics[-1]}"
        task_line = f" It sounds like you're looking for help with {joined}."

    def starts_with_house_number(value: str) -> bool:
        value = (value or "").strip()
        return bool(re.match(r"^\d{1,6}\b", value))

    saved_full = None
    for a in profile.get("addresses") or []:
        a = (a or "").strip()
        if starts_with_house_number(a):
            saved_full = a
            break

    raw_hint = (sched.get("raw_address") or classification.get("detected_address") or "").strip()
    appt_type = (sched.get("appointment_type") or conv.get("appointment_type") or classification.get("appointment_type") or "EVAL_195").upper()

    address_line = ""
    if raw_hint and starts_with_house_number(raw_hint):
        address_line = f" I have {raw_hint} for the visit. Is that correct?"
    elif saved_full:
        address_line = f" I have {saved_full} on file. Is this for that address?"
    elif raw_hint:
        address_line = f" What is the house number and street name in {raw_hint}?"
    else:
        address_line = " What is the address for the visit?"

    # Emergency first text should gather the missing address only.
    # Do not disclose price yet and do not mark it as disclosed.
    if appt_type == "TROUBLESHOOT_395":
        sms = (intro + task_line + address_line).strip()
        sched["intro_sent"] = True
        sched["price_disclosed"] = False
        return re.sub(r"\s+", " ", sms).strip()

    price_line = " Home inspections are $395." if appt_type == "WHOLE_HOME_INSPECTION" else " Our evaluation visit is $195."
    sms = (intro + task_line + address_line + price_line).strip()
    sched["intro_sent"] = True
    sched["price_disclosed"] = True
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
                'Thanks for calling PREE-volt Electric.<break time="0.7s"/>'
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

        response.say(
            '<speak>'
                '<prosody rate="95%">'
                    'Welcome to PREE-volt Electric’s premium residential service desk.<break time="0.7s"/>'
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
        return Response(str(response), mimetype="text/xml")

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

    # If transcription/cleanup failed or produced a meta-placeholder, do not let
    # the classifier hallucinate a fake customer, address, or work scope.
    if not cleaned or looks_like_transcription_placeholder(cleaned):
        try:
            log_event("VOICEMAIL_TRANSCRIPT_EMPTY_SUPPRESSED", from_number, {
                "recording_url": recording_url,
                "transcript": _safe_monitor_text(cleaned, 500),
            }, conversations.setdefault(from_number, {}))
        except Exception:
            pass
        try:
            send_sms(
                from_number,
                "Thanks for calling Prevolt Electric. I couldn't clearly read the voicemail. Please text the service address and a brief description of what you need help with."
            )
        except Exception as e:
            print("[WARN] empty voicemail fallback SMS failed:", repr(e))
        resp.say("Thank you. Your message has been recorded.")
        resp.hangup()
        return Response(str(resp), mimetype="text/xml")

    # 2) Classification
    classification = generate_initial_sms(cleaned)

    # 3) Save to memory
    conv = conversations.setdefault(from_number, {})

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

    # Hard pre-routing lock before any address/date scheduling state is saved.
    thread_type = detect_non_service_thread_type(conv, "", classification.get("category"), cleaned)
    if thread_type in {"employment_inquiry", "commercial_bid_contact"}:
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
    if classification.get("intent") == "emergency" and conv.get("thread_type") not in {"employment_inquiry", "commercial_bid_contact", "manual_only"}:
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
- NEVER say “one moment”, “please wait”, “hold on”, “securing your appointment”, or anything implying background processing.
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
        rf"^(?P<line1>\d{{1,6}}\s+.+?),\s*(?P<city>[A-Za-z .'\-]+?),\s*(?P<state>{state_token})\s+(?P<zip>\d{{5}}(?:-\d{{4}})?)$",
        rf"^(?P<line1>\d{{1,6}}\s+.+?)\s+(?P<city>[A-Za-z .'\-]+?)\s+(?P<state>{state_token})\s+(?P<zip>\d{{5}}(?:-\d{{4}})?)$",
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

        if not re.match(r"^\d{1,6}\b", line1):
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
        return bool(re.match(r"^\d{1,6}\b", line))

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
        " way", " pkwy", " parkway", " ter", " terrace"
    )
    has_street_word = any(suf in low for suf in street_suffixes)

    # Has a house number at the start?
    starts_with_number = low[:1].isdigit()

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
            "Send the house number and street name for the address.",
            "What’s the house number and street for the visit?",
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
        "What’s the address?",
        "What address are we heading to?",
        "What’s the address for the visit?",
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
    street_suffixes = (" st", " street", " ave", " avenue", " rd", " road", " ln", " lane", " dr", " drive", " blvd", " boulevard", " way", " ct", " court", " cir", " circle", " ter", " terrace", " pkwy", " parkway")
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





def extract_city_state_from_reply(text: str) -> tuple[str | None, str | None]:
    """Parse compact city/state replies like 'Windsor CT', 'Brockton', or 'Massachusetts'."""
    txt = " ".join((text or "").strip().replace(",", " ").split())
    if not txt:
        return None, None

    low = txt.lower()
    state = None
    if re.search(r"\bct\b|\bconnecticut\b", low):
        state = "CT"
        txt = re.sub(r"\bct\b|\bconnecticut\b", "", txt, flags=re.I).strip(" ,")
    elif re.search(r"\bma\b|\bmass\b|\bmassachusetts\b", low):
        state = "MA"
        txt = re.sub(r"\bma\b|\bmass\b|\bmassachusetts\b", "", txt, flags=re.I).strip(" ,")

    # Reject obvious non-city inputs.
    if re.search(r"\d", txt):
        return None, state
    if re.search(r"\b(st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace)\b", txt, flags=re.I):
        return None, state

    city = " ".join(w.capitalize() for w in txt.split()) if txt else None
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

    sched["raw_address"] = re.sub(r"\s+,", ",", merged).strip(" ,")

    # Best effort normalization immediately so the thread advances instead of re-asking.
    forced_state = state if state in {"CT", "MA"} else None
    try:
        result = normalize_address(sched["raw_address"], forced_state=forced_state)
        if isinstance(result, tuple) and len(result) >= 2:
            status, addr_struct = result[0], result[1]
            if status == "ok" and isinstance(addr_struct, dict):
                sched["normalized_address"] = addr_struct
        elif isinstance(result, dict):
            sched["normalized_address"] = result
    except Exception as e:
        print("[WARN] apply_partial_address_reply normalize failed:", repr(e))

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
        if is_emergency and wants_immediate_dispatch and sched.get("address_verified"):
            return "This looks urgent. We can send someone now and arrival is usually within 1 to 2 hours. Troubleshoot and repair visits are $395. Do you want us to dispatch someone now?"
        if not sched.get("scheduled_time"):
            return humanize_question("What day and time work best for you?")
        return humanize_question("What day works best for you?")
    if step == "need_time":
        if is_emergency and wants_immediate_dispatch and sched.get("address_verified"):
            return "This looks urgent. We can send someone now and arrival is usually within 1 to 2 hours. Troubleshoot and repair visits are $395. Do you want us to dispatch someone now?"
        if not sched.get("scheduled_date"):
            return humanize_question("What day and time work best for you?")
        if provided_ambiguous_time:
            if is_emergency and sched.get("address_verified"):
                return "This looks urgent. We can send someone now and arrival is usually within 1 to 2 hours. Troubleshoot and repair visits are $395. Do you want us to dispatch someone now?"
            if is_emergency:
                return humanize_question("What is the address for the visit?")
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
    # Extract customer-authored payloads, strip relay/email debris, and suppress
    # Google/Prevolt echo messages before they can enter the normal SMS brain.
    raw_inbound_text = inbound_text
    lsa_payload_fields = extract_lsa_payload_fields(raw_inbound_text)
    lsa_customer_message = extract_lsa_customer_message(raw_inbound_text)

    if is_lsa_self_echo_or_dirty_business_reply(raw_inbound_text):
        try:
            log_event("SMS_LSA_SELF_ECHO_SUPPRESSED", phone, {"sid": inbound_sid, "body": _safe_monitor_text(raw_inbound_text)})
        except Exception:
            pass
        return Response(str(MessagingResponse()), mimetype="text/xml")

    if is_pure_lsa_platform_notice(raw_inbound_text):
        try:
            log_event("SMS_PLATFORM_SUPPRESSED", phone, {"sid": inbound_sid, "body": _safe_monitor_text(raw_inbound_text), "source": "google_lsa"})
        except Exception:
            pass
        return Response(str(MessagingResponse()), mimetype="text/xml")

    if lsa_customer_message:
        inbound_text = strip_lsa_customer_text_noise(lsa_customer_message)
        inbound_low = inbound_text.lower().strip()
    else:
        # Follow-up relay messages usually do not repeat the full LSA wrapper.
        # If this relay number is already known as google_lsa, clean it before
        # logging so the monitor does not show CONFIDENTIAL blocks/signatures.
        existing_conv = conversations.get(phone) or {}
        if (existing_conv.get("source") or "").strip() == "google_lsa":
            cleaned_existing_lsa = strip_lsa_customer_text_noise(inbound_text)
            if cleaned_existing_lsa:
                inbound_text = cleaned_existing_lsa
                inbound_low = inbound_text.lower().strip()
            if not inbound_text or is_lsa_self_echo_or_dirty_business_reply(inbound_text):
                try:
                    log_event("SMS_LSA_SELF_ECHO_SUPPRESSED", phone, {"sid": inbound_sid, "body": _safe_monitor_text(raw_inbound_text), "reason": "pre_log_existing_lsa"}, existing_conv)
                except Exception:
                    pass
                return Response(str(MessagingResponse()), mimetype="text/xml")

    # Google LSA email replies can arrive as bare follow-up relay texts after a
    # deploy or memory reset, with no wrapper and no known google_lsa state.
    # Strip obvious email signature debris globally before the monitor sees it.
    if re.search(r"\bCONFIDENTIAL\s*:", inbound_text or "", flags=re.I) or re.search(r"\bThis email and any files transmitted\b", inbound_text or "", flags=re.I):
        cleaned_signature_text = strip_lsa_customer_text_noise(inbound_text)
        if cleaned_signature_text:
            inbound_text = cleaned_signature_text
            inbound_low = inbound_text.lower().strip()
        else:
            try:
                log_event("SMS_SIGNATURE_ONLY_SUPPRESSED", phone, {"sid": inbound_sid, "body": _safe_monitor_text(raw_inbound_text)})
            except Exception:
                pass
            return Response(str(MessagingResponse()), mimetype="text/xml")

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
    if lsa_customer_message or is_google_lsa_platform_notice(raw_inbound_text):
        conv["source"] = "google_lsa"
        if lsa_payload_fields:
            conv["lsa_payload"] = {**(conv.get("lsa_payload") or {}), **lsa_payload_fields}
        try:
            if lsa_payload_fields.get("customer_name"):
                apply_lsa_customer_name_to_profile(conv, lsa_payload_fields.get("customer_name"))
            if lsa_payload_fields.get("location"):
                conv.setdefault("sched", {})["lsa_location"] = lsa_payload_fields.get("location")
            if lsa_payload_fields.get("service"):
                conv.setdefault("current_job", {})["lsa_service"] = lsa_payload_fields.get("service")
        except Exception:
            pass

    is_lsa_thread_now = is_google_lsa_thread(conv, raw_inbound_text)

    # Once a relay number is known to be an LSA thread, every follow-up through
    # that relay must be cleaned too. Follow-ups often arrive without the full
    # "Message:" wrapper, but still carry email signatures, CONFIDENTIAL blocks,
    # quoted history, or dashboard debris.
    if is_lsa_thread_now:
        cleaned_lsa_followup = strip_lsa_customer_text_noise(inbound_text)
        if cleaned_lsa_followup:
            inbound_text = cleaned_lsa_followup
            inbound_low = inbound_text.lower().strip()
        if not inbound_text or is_lsa_self_echo_or_dirty_business_reply(inbound_text):
            try:
                log_event("SMS_LSA_SELF_ECHO_SUPPRESSED", phone, {"sid": inbound_sid, "body": _safe_monitor_text(raw_inbound_text), "reason": "post_clean"}, conv)
            except Exception:
                pass
            return Response(str(MessagingResponse()), mimetype="text/xml")

    # Twilio and WhatsApp can occasionally retry the same inbound webhook.
    # Guard both by inbound SID and by a short body fingerprint window.
    inbound_fingerprint = re.sub(r"\s+", " ", inbound_low).strip()
    now_ts = time.time()
    if inbound_sid and conv.get("last_inbound_sid") == inbound_sid and conv.get("last_sms_body"):
        if is_lsa_thread_now:
            try:
                log_event("SMS_LSA_DUPLICATE_SUPPRESSED", phone, {"sid": inbound_sid, "body": _safe_monitor_text(inbound_text), "reason": "same_sid"}, conv)
            except Exception:
                pass
            return Response(str(MessagingResponse()), mimetype="text/xml")
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
        if is_lsa_thread_now:
            try:
                log_event("SMS_LSA_DUPLICATE_SUPPRESSED", phone, {"sid": inbound_sid, "body": _safe_monitor_text(inbound_text), "reason": "same_body_window"}, conv)
            except Exception:
                pass
            return Response(str(MessagingResponse()), mimetype="text/xml")
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

    name_correction_first = maybe_apply_direct_name_correction(conv, inbound_text)
    if name_correction_first:
        sched = conv.setdefault("sched", {})
        profile = conv.setdefault("profile", {})
        if sched.get("appointment_type"):
            try:
                recompute_pending_step(profile, sched)
                next_prompt = choose_next_prompt_from_state(conv, inbound_text="")
            except Exception:
                next_prompt = "What is your last name?"
            body = next_prompt or f"Got it, {name_correction_first}. What is your last name?"
        else:
            body = f"Got it, {name_correction_first}. What can I help you with?"
        conv["last_sms_body"] = body
        try:
            log_event("SMS_NAME_CORRECTION", phone, {"body": _safe_monitor_text(body), "first_name": name_correction_first}, conv)
        except Exception:
            pass
        resp = MessagingResponse()
        resp.message(body)
        return Response(str(resp), mimetype="text/xml")

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

    # Google LSA message leads are dirty relay traffic. Before any general
    # interruption/LLM route can answer, enforce pricing/material/duration rules.
    try:
        if is_lsa_thread_now:
            lsa_safe_reply = build_lsa_safety_reply(conv, inbound_text)
            if lsa_safe_reply:
                conv["last_inbound_sid"] = inbound_sid
                conv["last_inbound_fingerprint"] = inbound_fingerprint
                conv["last_inbound_fingerprint_ts"] = now_ts
                conv["last_sms_body"] = lsa_safe_reply
                try:
                    log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(lsa_safe_reply), "source": "google_lsa", "lsa_safety": True}, conv)
                except Exception:
                    pass
                tw = MessagingResponse()
                tw.message(lsa_safe_reply)
                return Response(str(tw), mimetype="text/xml")

            if not sched.get("appointment_type") and looks_like_lsa_quote_or_service_lead(inbound_text):
                lsa_entry_reply = build_lsa_eval_entry_reply(conv, inbound_text)
                conv["last_inbound_sid"] = inbound_sid
                conv["last_inbound_fingerprint"] = inbound_fingerprint
                conv["last_inbound_fingerprint_ts"] = now_ts
                conv["last_sms_body"] = lsa_entry_reply
                try:
                    log_event("SMS_FLOW_REPLY", phone, {"body": _safe_monitor_text(lsa_entry_reply), "source": "google_lsa", "lsa_entry": True}, conv)
                except Exception:
                    pass
                tw = MessagingResponse()
                tw.message(lsa_entry_reply)
                return Response(str(tw), mimetype="text/xml")
    except Exception as e:
        print("[WARN] incoming_sms LSA safety/entry guard failed:", repr(e))

    # Non-service / thread-type hard guard before normal booking state can run.
    thread_type = detect_non_service_thread_type(conv, inbound_text, conv.get("category"), conv.get("cleaned_transcript"))
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
            # Preserve the address for Kyle's review without treating it as a bookable Square visit.
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

    # First-message GLS/SMS service leads should establish the $195 evaluation
    # context before availability handling offers slots. This includes commercial
    # or multifamily equipment work like a meter-bank replacement.
    try:
        if (not is_lsa_thread_now) and looks_like_initial_service_booking_request(conv, inbound_text):
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
        if sched.get("awaiting_slot_offer_choice") and is_rejecting_offered_slots(inbound_text):
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

        fast_interrupt = enforce_lsa_outbound_safety(conv, inbound_text, fast_interrupt).strip()
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = fast_interrupt
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
        candidate_address = str(reply["address"] or "").strip()
        existing_address = str(sched.get("raw_address") or "").strip()
        if not (
            yes_text(inbound_text)
            and existing_address
            and (candidate_address == f"{existing_address}, Yes" or candidate_address == f"{existing_address} Yes")
        ):
            sched["raw_address"] = candidate_address
            # addresses list is guaranteed above
            if candidate_address and candidate_address not in profile["addresses"]:
                profile["addresses"].append(candidate_address)

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

    # Final LSA outbound safety net before duplicate/frustration handling.
    sms_body = enforce_lsa_outbound_safety(conv, inbound_text, sms_body)

    # Frustration and duplicate-prompt protection.
    if is_frustrated_with_bot(inbound_text):
        sms_body = "Sorry about that. I have the information you sent. Kyle will review this manually."
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
        "from", "at", "in", "the", "a", "an"
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

    # Google LSA relay threads must never let the LLM improvise price,
    # material-inclusion, or install-duration answers.
    lsa_safe = build_lsa_safety_reply(conv, inbound_text)
    if lsa_safe:
        return lsa_safe

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
            answer = "We can provide insurance documentation when the visit is moving forward."
        else:
            answer = "Yes, we're licensed and insured."
    elif any(x in low for x in ["where are you located", "where are you guys located", "where are you based", "where are you out of"]):
        answer = "We service Connecticut and Massachusetts."
    elif any(x in low for x in ["call when", "text when", "on the way", "arrival window", "when close", "when you re close", "when you're close", "when youre close"]):
        answer = "Yes, you'll get a text when we're on the way."
    elif any(x in low for x in ["do i need to buy", "bring anything", "materials", "should i buy", "do i need anything"]):
        answer = "No, you do not need to buy anything ahead of time for the visit."
    elif any(x in low for x in ["permit", "permit required"]):
        answer = "If anything needs a permit, we'll go over that during the visit."
    elif any(x in low for x in ["card", "cash", "check", "payment", "pay by", "how do i pay"]):
        answer = "Card or cash after the visit is fine."
    elif any(x in low for x in [
        "how much", "price", "cost", "$195", "$395", "195", "395",
        "just to come out", "just to come", "service fee", "trip fee", "diagnostic fee",
        "quote", "estimate", "free estimate", "ballpark", "rough price", "firm number",
        "what do you charge", "what does the visit include", "what does that include",
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
                answer = "The $195 covers the evaluation visit itself. If you decide to move forward after that, we go over the next step in person."
        elif any(x in low for x in ["quote", "estimate", "free estimate", "ballpark", "firm number", "rough price"]):
            answer = "For quote requests, we handle that with a $195 evaluation visit so we can see everything in person and give you a firm number."
        elif "TROUBLESHOOT" in appt:
            answer = "The $395 is the troubleshoot and repair visit to come out, diagnose the issue, and handle minor repairs if it makes sense on site."
        elif "INSPECTION" in appt:
            answer = "Whole-home inspections are $395, and larger homes can run higher depending on square footage."
        else:
            answer = "The $195 is the service visit to come out, evaluate the issue, and go over the next step."
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


def salvage_relative_date_from_text(inbound_text: str) -> str | None:
    low = _loose_text(inbound_text)
    if not low:
        return None
    now_local = datetime.now(ZoneInfo("America/New_York")) if ZoneInfo else datetime.now()
    today = now_local.date()

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

    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", low)
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
        return "You're all set for the visit."

    if any(x in low for x in ["who is coming", "who's coming", "whos coming", "on the way", "arrival window", "will they call"]):
        return "You'll get a text when we're on the way."

    if any(x in low for x in ["price", "how much", "cost"]):
        return "You're all set for the visit."

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

        # If the customer picks one of the offered fallback slots, lock it in before any LLM work.
        maybe_apply_offered_slot_selection(conv, inbound_text)

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
        IS_EMERGENCY = any(k in inbound_lower for k in EMERGENCY_KEYWORDS)

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
                if not inbound_text:
                    topics_text = _task_topics_text()
                    if topics_text:
                        intro = f"{intro} It sounds like you're looking for help with {topics_text}."
                s = f"{intro} {s}".strip()
                sched["intro_sent"] = True
            return _norm(s)

        def _maybe_price_once(s: str, appt_type_local: str) -> str:
            # Emergency pricing is disclosed manually at the dispatch prompt, not on the opener
            # and not on every follow-up.
            if appt_type_local == "TROUBLESHOOT_395":
                return _norm(s)
            if not sched.get("price_disclosed"):
                try:
                    s = apply_price_injection(appt_type_local, s)
                except Exception:
                    pass
                sched["price_disclosed"] = True
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
                " way", " pkwy", " parkway", " ter", " terrace"
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
                        sched["raw_address"] = f"{inbound_clean}, {city_hint}".strip(" ,")
                    else:
                        sched["raw_address"] = inbound_clean
                    sched["normalized_address"] = None
                    update_address_assembly_state(sched)

                elif inbound_starts_with_number and inbound_has_street_word:
                    city_hint = raw.strip()
                    if city_hint and city_hint.lower() not in inbound_clean.lower():
                        sched["raw_address"] = f"{inbound_clean}, {city_hint}".strip(" ,")
                    else:
                        sched["raw_address"] = inbound_clean
                    sched["normalized_address"] = None
                    update_address_assembly_state(sched)

            elif missing_atom == "number":
                if inbound_starts_with_number and inbound_has_street_word:
                    sched["raw_address"] = inbound_clean
                    sched["normalized_address"] = None
                    update_address_assembly_state(sched)

                elif num:
                    base = raw or norm_line1
                    if base and not re.match(r"^\d{1,6}\b", base):
                        sched["raw_address"] = f"{num} {base}".strip()
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

        if isinstance(model_addr, str) and len(model_addr.strip()) > 3 and is_plausible_address_text(model_addr):
            sched["raw_address"] = model_addr.strip()

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
def apply_price_injection(appt_type: str, body: str) -> str:
    if "$" in body:
        return body

    if "TROUBLESHOOT" in appt_type:
        return f"{body} Troubleshooting and repair visits are $395."
    if "INSPECTION" in appt_type:
        return f"{body} Whole-home electrical inspections range from $375–$650 depending on square footage."
    return f"{body} Our evaluation visit is $195."



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


def square_lookup_customer_by_phone(phone: str) -> dict | None:
    """Return the first Square customer match for this phone, or None."""
    try:
        payload = {"query": {"filter": {"phone_number": {"exact": phone}}}}
        resp = requests.post(
            "https://connect.squareup.com/v2/customers/search",
            json=payload,
            headers=square_headers(),
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            return None
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
    appointment_type = (appointment_type or "EVAL_195").upper()
    tz = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
    now_local = datetime.now(tz)
    collected = []
    seen = set()

    non_emergency = "TROUBLESHOOT" not in appointment_type

    for offset in range(days_ahead + 1):
        target_day = now_local.date() + timedelta(days=offset)
        if non_emergency and target_day.weekday() >= 5:
            continue
        target_date = target_day.strftime("%Y-%m-%d")
        day_slots = search_square_availability_for_day(target_date, appointment_type)
        for slot in day_slots:
            key = (slot.get("date"), slot.get("time"))
            if key in seen:
                continue
            try:
                slot_dt = datetime.strptime(f"{slot.get('date')} {slot.get('time')}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            except Exception:
                continue
            if slot_dt <= now_local:
                continue
            seen.add(key)
            collected.append(slot)
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
    sched = conv.setdefault("sched", {})
    if not sched.get("awaiting_slot_offer_choice"):
        return False

    options = sched.get("offered_slot_options") or []
    if not options:
        sched["awaiting_slot_offer_choice"] = False
        return False

    txt = (inbound_text or "").strip()
    low = txt.lower()
    chosen = None

    ordinal_map = [
        ("first", 0), ("1st", 0), ("one", 0),
        ("second", 1), ("2nd", 1), ("two", 1),
        ("third", 2), ("3rd", 2), ("three", 2),
    ]
    for key, idx in ordinal_map:
        if re.search(rf"\b{re.escape(key)}\b", low):
            if idx < len(options):
                chosen = options[idx]
                break

    if chosen is None and txt.isdigit():
        idx = int(txt) - 1
        if 0 <= idx < len(options):
            chosen = options[idx]

    explicit_time = extract_explicit_time_from_text(txt)
    if chosen is None and explicit_time:
        matches = [opt for opt in options if opt.get("time") == explicit_time]
        if matches:
            chosen = matches[0]

    if chosen is None:
        for opt in options:
            label = (opt.get("label") or "").lower()
            if label and label in low:
                chosen = opt
                break

    if not chosen:
        return False

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
        return 9 <= dt.hour < 16
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


@app.route("/monitor/events", methods=["GET"])
def monitor_events():
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


# ---------------------------------------------------
# Local Development Entrypoint
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
