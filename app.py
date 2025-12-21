import os
import json
import time
import uuid
import requests
from datetime import datetime, timezone, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather, Dial
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from openai import OpenAI


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

app = Flask(__name__)

# ---------------------------------------------------
# In-Memory Conversation Store
# ---------------------------------------------------
conversations = {}


# ---------------------------------------------------
# WhatsApp SMS Helper (Testing Path Only)
# ---------------------------------------------------
def send_sms(to_number: str, body: str) -> None:
    """
    Outbound always routes to WhatsApp sandbox for safe testing.
    """
    if not twilio_client:
        print("[WARN] Twilio not configured. SMS not sent.")
        print("Message:", body)
        return

    try:
        whatsapp_from = "whatsapp:+14155238886"
        whatsapp_to   = "whatsapp:+18609701727"  # your cell

        msg = twilio_client.messages.create(
            body=body,
            from_=whatsapp_from,
            to=whatsapp_to
        )
        print("[WhatsApp] Sent SID:", msg.sid)
    except Exception as e:
        print("[ERROR] WhatsApp send failed:", repr(e))


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
    """
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
                        "You extract structured info from voicemail. "
                        "You DO NOT generate SMS replies.\n\n"
                        "Return JSON:\n"
                        "{\n"
                        "  'category': str,\n"
                        "  'appointment_type': str,\n"
                        "  'detected_address': str|null,\n"
                        "  'detected_date': 'YYYY-MM-DD'|null,\n"
                        "  'detected_time': 'HH:MM'|null,\n"
                        "  'intent': 'schedule'|'quote'|'emergency'|'other'\n"
                        "}"
                    ),
                },
                {"role": "user", "content": cleaned_text},
            ],
        )

        data = json.loads(completion.choices[0].message.content)

        return {
            "category": data.get("category"),
            "appointment_type": data.get("appointment_type"),
            "detected_address": data.get("detected_address"),
            "detected_date": data.get("detected_date"),
            "detected_time": data.get("detected_time"),
            "intent": data.get("intent"),
        }

    except Exception as e:
        print("[ERROR] Voicemail classifier failed:", repr(e))
        return {
            "category": "OTHER",
            "appointment_type": "EVAL_195",
            "detected_address": None,
            "detected_date": None,
            "detected_time": None,
            "intent": "other",
        }


# ---------------------------------------------------
# Voice: Incoming Call (IVR + Spam Filter)
# ---------------------------------------------------
@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    from twilio.twiml.voice_response import Gather, VoiceResponse

    response = VoiceResponse()

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

    conv = conversations.setdefault(phone, {})

    # ---------------------------------------------------
    # ✅ CRITICAL FIX:
    # conv.setdefault("profile", {}) leaves profile as {} forever,
    # which breaks /incoming-sms when it expects profile["addresses"].
    # So we "hydrate" required keys even if profile already exists.
    # ---------------------------------------------------
    profile = conv.setdefault("profile", {})
    profile.setdefault("name", None)
    profile.setdefault("addresses", [])
    profile.setdefault("upcoming_appointment", None)
    profile.setdefault("past_jobs", [])

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
        response.dial("+15555555555")  # replace with real number
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
    conv.setdefault("initial_sms", cleaned)

    if classification.get("detected_date"):
        sched["scheduled_date"] = classification["detected_date"]
    if classification.get("detected_time"):
        sched["scheduled_time"] = classification["detected_time"]
    if classification.get("detected_address"):
        sched["raw_address"] = classification["detected_address"]

    # Refresh address assembly state (safe if normalized/raw changed)
    update_address_assembly_state(sched)

    # --------------------------------------
    # HARD OVERRIDE: Voicemail Emergency Pre-Flag
    # --------------------------------------
    if classification.get("intent") == "emergency":
        sched["appointment_type"] = "TROUBLESHOOT_395"
        sched["awaiting_emergency_confirm"] = True
        sched["emergency_approved"] = False

    # 4) Trigger First SMS (Step 4 call)
    try:
        outbound = generate_reply_for_inbound(
            cleaned,
            conv.get("category"),
            conv.get("appointment_type"),
            conv.get("initial_sms"),
            "",  # no user inbound yet
            sched.get("scheduled_date"),
            sched.get("scheduled_time"),
            sched.get("raw_address")
        )

        initial_msg = outbound.get("sms_body") or "Thanks for your voicemail — how can we help?"
        send_sms(from_number, initial_msg)
    except Exception as e:
        print("[ERROR] voicemail_complete → Step4:", repr(e))

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
    ]

    if any(p in low for p in banned_phrases) and not booking_created:
        return "Thanks — got it."

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
    return f"""
You are Prevolt OS — a deterministic scheduling engine.
You ALWAYS return strict JSON with fields:
  sms_body, scheduled_date, scheduled_time, address.

NEVER forget known values. NEVER reset fields unless the user changes them.

Known transcript: {cleaned_transcript}
Category: {category}
Initial appointment type: {appointment_type}
Initial SMS: {initial_sms}

Current stored values:
  scheduled_date = {scheduled_date}
  scheduled_time = {scheduled_time}
  address = {address}

Today's date: {today_date_str} ({today_weekday})

Rules:
- NEVER ask questions already answered.
- If only one field is missing, ask ONLY for that field.
- If all fields are present, confirm appointment.
- Use simple, direct language.
- NEVER say “one moment”, “please wait”, “hold on”, “securing your appointment”, or anything implying background processing.
- Only confirm a booking if it is ACTUALLY booked with Square (real booking id exists).
"""


# ---------------------------------------------------
# Address Assembly State Helper (Step 2) — FIXED (requires house number)
# ---------------------------------------------------
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
                f"What street is it on in {candidate}?",
                f"Which street in {candidate} is this on?",
                f"What’s the street name in {candidate}?",
            ]
            return pick_variant_once(sched, "addr_missing_street_with_city", options)

        options = [
            "What street is it on?",
            "What’s the street name?",
            "Which street is this on?",
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



# ---------------------------------------------------
# Incoming SMS (B-3 State Machine, Option A)
# ---------------------------------------------------
@app.route("/incoming-sms", methods=["POST"])
def incoming_sms():
    inbound_text = request.form.get("Body", "") or ""
    phone        = request.form.get("From", "").replace("whatsapp:", "")
    inbound_low  = inbound_text.lower().strip()

    # SECRET RESET COMMAND
    if inbound_low == "mobius1":
        conversations[phone] = {
            "profile": {"name": None, "addresses": [], "upcoming_appointment": None, "past_jobs": []},
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
            }
        }
        resp = MessagingResponse()
        resp.message("✔ Memory reset complete for this number.")
        return Response(str(resp), mimetype="text/xml")

    # ---------------------------------------------------
    # Initialize layers (HARDENED: never assume dict shape)
    # ---------------------------------------------------
    conv = conversations.setdefault(phone, {})

    profile = conv.setdefault("profile", {})
    profile.setdefault("name", None)
    profile.setdefault("addresses", [])
    profile.setdefault("upcoming_appointment", None)
    profile.setdefault("past_jobs", [])
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

    # Address Assembly State
    sched.setdefault("address_candidate", None)
    sched.setdefault("address_verified", False)
    sched.setdefault("address_missing", None)
    sched.setdefault("address_parts", {})

    # Emergency flags (used by incoming_sms patch logic too)
    sched.setdefault("awaiting_emergency_confirm", False)
    sched.setdefault("emergency_approved", False)

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
            import re

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
            else:
                sched["pending_step"] = None

    # Keep address state fresh (pre-Step4)
    update_address_assembly_state(sched)

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
    sched["scheduled_date"] = reply.get("scheduled_date")
    sched["scheduled_time"] = reply.get("scheduled_time")

    if reply.get("address"):
        sched["raw_address"] = reply["address"]
        # addresses list is guaranteed above
        if reply["address"] not in profile["addresses"]:
            profile["addresses"].append(reply["address"])

    # Re-derive address assembly state after Step 4 updates
    update_address_assembly_state(sched)

    # POST-Step4 pending_step (Address complete only when verified)
    if not sched.get("scheduled_date"):
        sched["pending_step"] = "need_date"
    elif not sched.get("scheduled_time"):
        sched["pending_step"] = "need_time"
    elif not sched.get("address_verified"):
        sched["pending_step"] = "need_address"
    else:
        sched["pending_step"] = None

    tw = MessagingResponse()
    tw.message(reply["sms_body"])
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
            sms = build_address_prompt(sched)
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
                sms = build_address_prompt(sched)
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


# ---------------------------------------------------
# Step 4 — Generate Replies (Hybrid Logic + Deterministic State Machine)
# PATCHED (HUMAN + HOUSE# + NO WAIT-TEXT + ACK MEMORY + SLOT SUGGESTIONS)
#   ✅ Longer, more human intros + prompts (deterministic per conversation)
#   ✅ No em dash "—" and no " - " telltales
#   ✅ No "Got it", "Thanks" filler
#   ✅ No "one moment / securing" messages (ever)
#   ✅ House-number patch: merges "45" into stored street
#   ✅ Re-check address state AFTER model output (prevents booking without house number)
#   ✅ If Square did NOT book, never "confirm"; always ask next missing atom
#   ✅ If Square slot is taken, offers available time slots same day; if none, next day
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
        conv  = conversations.setdefault(phone, {})

        profile = conv.setdefault("profile", {})
        profile.setdefault("addresses", [])
        profile.setdefault("past_jobs", [])
        profile.setdefault("upcoming_appointment", None)

        sched = conv.setdefault("sched", {})

        # Core state
        sched.setdefault("raw_address", None)
        sched.setdefault("normalized_address", None)
        sched.setdefault("pending_step", None)
        sched.setdefault("intro_sent", False)
        sched.setdefault("price_disclosed", False)
        sched.setdefault("awaiting_emergency_confirm", False)
        sched.setdefault("emergency_approved", False)

        # Booking flags
        sched.setdefault("booking_created", False)
        sched.setdefault("square_booking_id", None)

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

        # Time suggestion memory (set by maybe_create_square_booking when slot is taken)
        # Expected shape: {"date": "YYYY-MM-DD", "starts_utc": ["...Z", ...], "requested_start_utc": "...Z"}
        sched.setdefault("time_suggestions", None)

        inbound_text  = (inbound_text or "").strip()
        inbound_lower = inbound_text.lower().strip()

        # --------------------------------------
        # Local helpers
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

        def pick_variant_once(label: str, options: list[str]) -> str:
            if not options:
                return ""
            store = sched.setdefault("prompt_variants", {})
            if label in store and store[label] in options:
                return store[label]
            h = _stable_choice_key(label)
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
            options = [
                "Hi, this is Prevolt Electric. I can help you right here by text.",
                "Hey, this is Prevolt Electric. Quick question so I can get this lined up.",
                "Hi, you’ve reached Prevolt Electric. I’ll get this set up for you here.",
            ]
            return pick_variant_once("intro_line", options)

        def humanize_question(core_question: str) -> str:
            core_question = _norm(core_question)
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
                s = f"{build_human_intro_line()} {s}".strip()
                sched["intro_sent"] = True
            return _norm(s)

        def _maybe_price_once(s: str, appt_type_local: str) -> str:
            if not sched.get("price_disclosed"):
                try:
                    s = apply_price_injection(appt_type_local, s)
                except Exception:
                    pass
                sched["price_disclosed"] = True
            return _norm(s)

        def _format_time_suggestions_for_sms() -> str | None:
            sugg = sched.get("time_suggestions")
            if not isinstance(sugg, dict):
                return None
            starts = sugg.get("starts_utc")
            if not isinstance(starts, list) or not starts:
                return None

            # Convert UTC strings into local human times
            pretty = []
            for s_utc in starts[:3]:
                try:
                    dt_utc = datetime.strptime(s_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    dt_local = dt_utc.astimezone(tz)
                    pretty.append(dt_local.strftime("%I:%M %p").lstrip("0"))
                except Exception:
                    continue

            if not pretty:
                return None

            day = (sugg.get("date") or "").strip()
            if day:
                return f"That time just got taken. I can do {', '.join(pretty)} on {day}. Which one works best?"
            return f"That time just got taken. I can do {', '.join(pretty)}. Which one works best?"

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
                    elif not sched.get("scheduled_date"):
                        s = humanize_question("What day works best for you?")
                        s = _apply_intro_once(s)
                    elif not sched.get("scheduled_time"):
                        # If we have slot suggestions, use them
                        sug_txt = _format_time_suggestions_for_sms()
                        s = sug_txt if sug_txt else humanize_question("What time works best?")
                        s = _apply_intro_once(s)
                    else:
                        s = "Okay."

            # Extra safety: if not booked, strip any confirmation language
            if not booking_created:
                conf_markers = ["confirmation number", "confirmed", "your appointment is", "booked for", "scheduled for"]
                if any(m in s.lower() for m in conf_markers):
                    update_address_assembly_state(sched)
                    if not sched.get("address_verified"):
                        s = build_address_prompt(sched)
                        s = _apply_intro_once(s)
                    elif not sched.get("scheduled_date"):
                        s = humanize_question("What day works best for you?")
                        s = _apply_intro_once(s)
                    elif not sched.get("scheduled_time"):
                        sug_txt = _format_time_suggestions_for_sms()
                        s = sug_txt if sug_txt else humanize_question("What time works best?")
                        s = _apply_intro_once(s)
                    else:
                        s = "Okay."

            try:
                s = sanitize_sms_body(s, booking_created=bool(booking_created))
            except Exception:
                pass

            s = s.replace("—", ".").replace("–", ".").replace(" - ", " ")
            s = _norm(s)

            # Prevent repeating same ack text
            if _is_ack_only(inbound_text):
                if sched.get("last_ack_text") and sched["last_ack_text"].lower() == s.lower():
                    s = "Okay."
                sched["last_ack_text"] = s
                sched["last_ack_ts"] = datetime.utcnow().isoformat()

            return s

        # --------------------------------------
        # Emergency flow (2-step confirmation)
        # --------------------------------------
        EMERGENCY_KEYWORDS = [
            "tree fell", "tree down", "power line", "lines down",
            "service ripped", "sparking", "burning", "fire",
            "smoke", "no power", "power outage",
            "urgent", "emergency"
        ]

        IS_EMERGENCY = any(k in inbound_lower for k in EMERGENCY_KEYWORDS)
        EMERGENCY = IS_EMERGENCY

        if IS_EMERGENCY and not sched["awaiting_emergency_confirm"] and not sched["emergency_approved"]:
            sched["appointment_type"] = "TROUBLESHOOT_395"
            sched["awaiting_emergency_confirm"] = True
            sched["pending_step"] = None

            if not sched.get("raw_address"):
                msg = "This sounds urgent. " + build_address_prompt(sched)
                msg = _finalize_sms(msg, "TROUBLESHOOT_395", booking_created=False)
                return {"sms_body": msg, "scheduled_date": None, "scheduled_time": None, "address": None, "booking_complete": False}

            msg = (
                f"This looks urgent at {sched.get('raw_address')}. "
                "Troubleshoot and repair visits are $395. "
                "Do you want us to dispatch someone now?"
            )
            msg = _finalize_sms(msg, "TROUBLESHOOT_395", booking_created=False)
            return {"sms_body": msg, "scheduled_date": None, "scheduled_time": None, "address": sched.get("raw_address"), "booking_complete": False}

        CONFIRM_PHRASES = ["yes", "yeah", "yup", "ok", "okay", "sure", "book", "send", "do it"]

        if sched["awaiting_emergency_confirm"] and any(p in inbound_lower for p in CONFIRM_PHRASES):
            sched["emergency_approved"] = True
            sched["awaiting_emergency_confirm"] = False
            sched["appointment_type"] = "TROUBLESHOOT_395"
            sched["scheduled_date"] = today_date_str
            sched["scheduled_time"] = now_local.strftime("%H:%M")
            sched["pending_step"] = None

            scheduled_date = sched["scheduled_date"]
            scheduled_time = sched["scheduled_time"]

        # --------------------------------------
        # Appointment type fallback
        # --------------------------------------
        appt_type = sched.get("appointment_type") or appointment_type
        if not appt_type:
            if any(w in inbound_lower for w in ["not working", "no power", "dead", "sparking", "burning", "breaker keeps", "gfci", "outlet not", "troubleshoot"]):
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

        # ✅ HOUSE NUMBER PATCH (pre-LLM)
        try:
            update_address_assembly_state(sched)
            missing_atom = (sched.get("address_missing") or "").strip().lower()

            if missing_atom == "number":
                inbound_clean = inbound_text.strip()

                m_num = re.search(r"\b(\d{1,6})\b", inbound_clean)
                num = m_num.group(1) if m_num else None

                low = inbound_clean.lower()
                street_suffixes = (
                    " st", " street", " ave", " avenue", " rd", " road", " ln", " lane",
                    " dr", " drive", " ct", " court", " cir", " circle", " blvd", " boulevard",
                    " way", " pkwy", " parkway", " ter", " terrace"
                )
                inbound_has_street_word = any(suf in f" {low} " for suf in street_suffixes)
                inbound_starts_with_number = bool(re.match(r"^\s*\d{1,6}\b", inbound_clean))

                raw = (sched.get("raw_address") or "").strip()
                norm = sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None
                norm_line1 = (norm.get("address_line_1") or "").strip() if norm else ""

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

        # RESET-LOCK
        if sched.get("scheduled_date") and not model_date:
            model_date = sched["scheduled_date"]
        if sched.get("scheduled_time") and not model_time:
            model_time = sched["scheduled_time"]
        if sched.get("raw_address") and not model_addr:
            model_addr = sched["raw_address"]

        if isinstance(model_addr, str) and len(model_addr.strip()) > 3:
            sched["raw_address"] = model_addr.strip()

        # Re-run address state AFTER model changes
        update_address_assembly_state(sched)
        try:
            try_early_address_normalize(sched)
        except Exception as e:
            print("[WARN] try_early_address_normalize post-LLM failed:", repr(e))
        update_address_assembly_state(sched)

        model_addr = sched.get("raw_address")

        # Human-readable time (cross-platform safe)
        human_time = None
        if model_time:
            try:
                dt = datetime.strptime(model_time, "%H:%M")
                human_time = dt.strftime("%I:%M %p").lstrip("0")
            except Exception:
                human_time = model_time

        if model_time and human_time:
            sms_body = sms_body.replace(model_time, human_time)

        # Save new values
        if model_date:
            sched["scheduled_date"] = model_date
        if model_time:
            sched["scheduled_time"] = model_time

        # If address STILL not verified, override immediately
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
        ready_for_booking = (
            bool(sched.get("scheduled_date")) and
            bool(sched.get("scheduled_time")) and
            bool(sched.get("address_verified")) and
            bool(sched.get("appointment_type")) and
            not sched.get("pending_step") and
            not sched.get("booking_created")
        )

        if ready_for_booking or EMERGENCY:
            try:
                maybe_create_square_booking(phone, conv)

                if sched.get("booking_created") and sched.get("square_booking_id"):
                    booked_sms = (
                        f"Booked for {sched['scheduled_date']} at {human_time} "
                        f"at {model_addr}. Confirmation {sched.get('square_booking_id')}."
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

            if not sched.get("address_verified"):
                sms_body = build_address_prompt(sched)
            elif not sched.get("scheduled_date"):
                sms_body = humanize_question("What day works best for you?")
            elif not sched.get("scheduled_time"):
                # ✅ NEW: show slot suggestions if present
                sug_txt = _format_time_suggestions_for_sms()
                if sug_txt:
                    sms_body = sug_txt
                    # Clear after use so we don't repeat endlessly
                    sched["time_suggestions"] = None
                else:
                    sms_body = humanize_question("What time works best?")
            else:
                sms_body = pick_variant_once("neutral_no_book", [
                    "Okay. If anything changes, just text me here.",
                    "All set. If you need to adjust anything, just message me here.",
                    "Okay. Want to keep that same day and time?",
                ])

        booking_created = bool(sched.get("booking_created") and sched.get("square_booking_id"))
        sms_body = _finalize_sms(sms_body, appt_type, booking_created=booking_created)

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


def square_create_or_get_customer(phone: str, addr_struct: dict | None = None):
    """
    Searches by phone, else creates new.
    """
    # Search
    try:
        payload = {
            "query": {"filter": {"phone_number": {"exact": phone}}}
        }
        resp = requests.post(
            "https://connect.squareup.com/v2/customers/search",
            json=payload,
            headers=square_headers(),
            timeout=10,
        )
        data = resp.json()
        custs = data.get("customers", [])
        if custs:
            return custs[0]["id"]
    except Exception as e:
        print("[WARN] customer search failed:", repr(e))

    # Create
    try:
        payload = {
            "idempotency_key": str(uuid.uuid4()),
            "given_name": "Prevolt Lead",
            "phone_number": phone,
        }
        if addr_struct:
            payload["address"] = addr_struct

        resp = requests.post(
            "https://connect.squareup.com/v2/customers",
            json=payload,
            headers=square_headers(),
            timeout=10,
        )
        data = resp.json()
        return data.get("customer", {}).get("id")

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
    # ✅ Availability + Suggestions
    #   - If requested slot is free: proceed
    #   - If not: store same-day suggestions; if none, store next-day suggestions
    #   - Do NOT send SMS from here (Step 4 will message using sched["time_suggestions"])
    # ---------------------------------------------------
    duration_minutes = 60
    end_at_utc = start_at_utc + timedelta(minutes=duration_minutes)

    def _availability_search(start_dt_utc, end_dt_utc):
        payload = {
            "query": {
                "filter": {
                    "start_at_range": {
                        "start_at": start_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "end_at": end_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                    "location_id": SQUARE_LOCATION_ID,
                    "segment_filters": [
                        {
                            "service_variation_id": variation_id,
                            "team_member_id_filter": {"any": [SQUARE_TEAM_MEMBER_ID]},
                        }
                    ],
                }
            }
        }
        r = requests.post(
            "https://connect.squareup.com/v2/bookings/availability/search",
            headers=square_headers(),
            json=payload,
            timeout=12,
        )
        if r.status_code not in (200, 201):
            print("[ERROR] availability search failed:", r.status_code, r.text)
            return None
        return (r.json() or {}).get("availabilities") or []

    def _extract_start_times(av_list, limit=3):
        starts = []
        for a in (av_list or []):
            sa = (a.get("start_at") or "").strip()
            if sa:
                starts.append(sa)
            if len(starts) >= limit:
                break
        return starts

    # 1) Check the exact requested slot
    try:
        availabilities = _availability_search(start_at_utc, end_at_utc)
        if availabilities is None:
            return

        want_start = start_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        slot_ok = any((a.get("start_at") or "") == want_start for a in availabilities)

        if not slot_ok:
            # 2) Suggestions same day: search a wider window after requested start
            same_day_av = _availability_search(start_at_utc, start_at_utc + timedelta(hours=10))
            if same_day_av is None:
                return

            suggestions_utc = _extract_start_times(same_day_av, limit=3)
            suggestion_date = scheduled_date

            # 3) If none same day: search next day
            if not suggestions_utc:
                next_day_start = start_at_utc + timedelta(days=1)
                next_day_end = next_day_start + timedelta(hours=10)

                next_day_av = _availability_search(next_day_start, next_day_end)
                if next_day_av is None:
                    return

                suggestions_utc = _extract_start_times(next_day_av, limit=3)

                # If we found next-day suggestions, set the suggested date
                if suggestions_utc:
                    try:
                        d = datetime.strptime(scheduled_date, "%Y-%m-%d")
                        suggestion_date = (d + timedelta(days=1)).strftime("%Y-%m-%d")
                    except Exception:
                        suggestion_date = scheduled_date

            # Store for Step 4 to present
            sched["time_suggestions"] = {
                "date": suggestion_date,
                "starts_utc": suggestions_utc,  # list[str] like "2025-12-22T14:00:00Z"
                "requested_start_utc": want_start
            }

            print("[BLOCKED] Slot not available; saved suggestions:", sched["time_suggestions"])

            # Force Step 4 to ask for a new time (and guide it toward suggestion_date)
            sched["scheduled_time"] = None
            sched["pending_step"] = "need_time"

            # If we have next-day suggestions, move the date forward so Step 4 naturally stays aligned
            if suggestion_date and suggestion_date != scheduled_date:
                sched["scheduled_date"] = suggestion_date

            return

    except Exception as e:
        print("[ERROR] availability exception:", repr(e))
        return



# ---------------------------------------------------
# Local Development Entrypoint
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
