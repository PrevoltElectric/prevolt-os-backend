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
# build_system_prompt (RESTORED)
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
"""


# ---------------------------------------------------
# Address Assembly State Helper (Step 2)
# ---------------------------------------------------
def update_address_assembly_state(sched: dict) -> None:
    """
    Derives address state atoms from raw_address / normalized_address.
    This does NOT change booking or messaging behavior yet.
    """
    # Defaults (never assume keys exist)
    sched.setdefault("address_candidate", None)
    sched.setdefault("address_verified", False)
    sched.setdefault("address_missing", None)
    sched.setdefault("address_parts", {})

    raw = (sched.get("raw_address") or "").strip()
    norm = sched.get("normalized_address")

    # If we already have a normalized struct, we consider it bookable.
    if (
        isinstance(norm, dict)
        and norm.get("address_line_1")
        and norm.get("locality")
        and norm.get("administrative_district_level_1")
        and norm.get("postal_code")
    ):
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

    # No normalized address yet → evaluate raw candidate quality
    sched["address_verified"] = False
    if raw:
        sched["address_candidate"] = raw

    # If nothing at all
    if not raw:
        sched["address_missing"] = "street"
        sched["address_parts"] = {"street": False, "number": False, "city": False, "state": False, "zip": False, "source": "none"}
        return

    low = raw.lower()

    # Heuristic: detect street words (helps distinguish "Windsor Locks" vs "Dickerman Ave")
    street_suffixes = (
        " st", " street", " ave", " avenue", " rd", " road", " ln", " lane", " dr", " drive",
        " ct", " court", " cir", " circle", " blvd", " boulevard", " way", " pkwy", " parkway",
        " ter", " terrace"
    )
    has_street_word = any(suf in low for suf in street_suffixes)

    # Has a house number at the start? (leading digits)
    starts_with_number = low[:1].isdigit()

    # Has explicit CT/MA?
    has_state = (" ct" in f" {low} ") or (" connecticut" in low) or (" ma" in f" {low} ") or (" massachusetts" in low)

    # Very common case: "Dickerman Ave" (street, no number)
    if has_street_word and not starts_with_number:
        sched["address_missing"] = "number"
        sched["address_parts"] = {"street": True, "number": False, "city": False, "state": has_state, "zip": False, "source": "raw_address"}
        return

    # Common case: "24 Main St" (number + street, likely needs town/state/zip via normalization/confirm)
    if has_street_word and starts_with_number:
        sched["address_missing"] = "confirm"
        sched["address_parts"] = {"street": True, "number": True, "city": False, "state": has_state, "zip": False, "source": "raw_address"}
        return

    # Town-only like "Windsor Locks"
    sched["address_missing"] = "street"
    sched["address_parts"] = {"street": False, "number": False, "city": True, "state": has_state, "zip": False, "source": "raw_address"}


# ---------------------------------------------------
# Address Prompt Builder (Step 4, improved)
# ---------------------------------------------------
def build_address_prompt(sched: dict) -> str:
    """
    Human-friendly prompts that ask for the missing address atom.
    Avoids phrases like "full service address".
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

    if missing == "street":
        # If user previously gave a town-only value, keep it contextual.
        if parts.get("city") and candidate and not parts.get("street"):
            return f"Got it — what street is the job on in/near “{candidate}”? House number helps too."
        return "What street is the job at? House number + street is perfect."

    if missing == "number":
        # If we have route-level normalization (street + town/state/zip), ask ONLY for house number with context.
        if norm_line1 and norm_city and norm_state:
            tail = f"{norm_city}, {norm_state}"
            if norm_zip:
                tail += f" {norm_zip}"
            return f"Got it — what’s the house number on {norm_line1} in {tail}?"
        if candidate:
            return f"Got it — what’s the house number on {candidate}?"
        return "Got it — what’s the house number?"

    if missing == "confirm":
        # We have a street (and maybe a number). Ask town (and state if they didn’t say CT/MA).
        need_state = not bool(parts.get("state"))
        if need_state:
            return "What town is that in — and is it in Connecticut or Massachusetts?"
        return "What town is that in?"

    return "What’s the street address for the job? House number + street is perfect."


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


# ---------------------------------------------------
# Step 4 — Generate Replies (Hybrid Logic + Deterministic State Machine)
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

        # Ensure profile never breaks downstream code that assumes addresses exists
        profile = conv.setdefault("profile", {})
        profile.setdefault("addresses", [])
        profile.setdefault("past_jobs", [])
        profile.setdefault("upcoming_appointment", None)

        sched = conv.setdefault("sched", {})

        sched.setdefault("raw_address", None)
        sched.setdefault("normalized_address", None)
        sched.setdefault("pending_step", None)
        sched.setdefault("intro_sent", False)
        sched.setdefault("price_disclosed", False)
        sched.setdefault("awaiting_emergency_confirm", False)
        sched.setdefault("emergency_approved", False)

        # Ensure booking flags exist
        sched.setdefault("booking_created", False)
        sched.setdefault("square_booking_id", None)

        # Ensure address assembly keys always exist (safe for prompts)
        sched.setdefault("address_candidate", None)
        sched.setdefault("address_verified", False)
        sched.setdefault("address_missing", None)
        sched.setdefault("address_parts", {})

        inbound_text  = inbound_text or ""
        inbound_lower = inbound_text.lower()

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
                return {
                    "sms_body": (
                        "This is Prevolt Electric — this sounds like an emergency. "
                        + build_address_prompt(sched)
                    ),
                    "scheduled_date": None,
                    "scheduled_time": None,
                    "address": None,
                    "booking_complete": False
                }

            return {
                "sms_body": (
                    f"This is Prevolt Electric — this appears to be an emergency at "
                    f"{sched['raw_address']}. Emergency troubleshooting visits are $395. "
                    "Would you like us to dispatch a technician immediately?"
                ),
                "scheduled_date": None,
                "scheduled_time": None,
                "address": sched.get("raw_address"),
                "booking_complete": False
            }

        CONFIRM_PHRASES = [
            "yes", "yeah", "yup", "ok", "okay",
            "sure", "that works", "book", "send", "do it"
        ]

        if sched["awaiting_emergency_confirm"] and any(p in inbound_lower for p in CONFIRM_PHRASES):
            sched["emergency_approved"] = True
            sched["awaiting_emergency_confirm"] = False

            sched["appointment_type"] = "TROUBLESHOOT_395"
            sched["scheduled_date"] = today_date_str
            sched["scheduled_time"] = now_local.strftime("%H:%M")
            sched["pending_step"] = None

            # Sync locals so downstream sees them
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
            elif any(w in inbound_lower for w in [
                "inspection", "whole home inspection", "electrical inspection"
            ]):
                appt_type = "WHOLE_HOME_INSPECTION"
            else:
                appt_type = "EVAL_195"
            sched["appointment_type"] = appt_type

        # --------------------------------------
        # Missing-info resolver (Step 3 + Step 5B)
        # --------------------------------------
        update_address_assembly_state(sched)

        # Step 5B: attempt early normalization once we have a plausible street
        try:
            try_early_address_normalize(sched)
        except Exception as e:
            print("[WARN] try_early_address_normalize failed:", repr(e))

        # Refresh state after normalization attempt
        update_address_assembly_state(sched)

        missing_date = not (scheduled_date or sched.get("scheduled_date"))
        missing_time = not (scheduled_time or sched.get("scheduled_time"))

        # raw_address ≠ complete address
        missing_addr = not bool(sched.get("address_verified"))

        if missing_addr:
            sched["pending_step"] = "need_address"
        elif missing_date:
            sched["pending_step"] = "need_date"
        elif missing_time:
            sched["pending_step"] = "need_time"
        else:
            sched["pending_step"] = None

        # Deterministic address ask (no loops, no "full service address" wording)
        if sched.get("pending_step") == "need_address":
            return {
                "sms_body": build_address_prompt(sched),
                "scheduled_date": sched.get("scheduled_date") or scheduled_date,
                "scheduled_time": sched.get("scheduled_time") or scheduled_time,
                "address": sched.get("raw_address"),
                "booking_complete": False
            }

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

        # --------------------------------------
        # RESET-LOCK
        # --------------------------------------
        if sched.get("scheduled_date") and not model_date:
            model_date = sched["scheduled_date"]
        if sched.get("scheduled_time") and not model_time:
            model_time = sched["scheduled_time"]
        if sched.get("raw_address") and not model_addr:
            model_addr = sched["raw_address"]

        if isinstance(model_addr, str) and len(model_addr) > 5:
            sched["raw_address"] = model_addr
        model_addr = sched.get("raw_address")

        # --------------------------------------
        # ONE-TIME BRANDING
        # --------------------------------------
        if not sched["intro_sent"]:
            sms_body = f"This is Prevolt Electric — {sms_body}"
            sched["intro_sent"] = True

        # --------------------------------------
        # ONE-TIME PRICE DISCLOSURE
        # --------------------------------------
        if not sched["price_disclosed"]:
            sms_body = apply_price_injection(appt_type, sms_body)
            sched["price_disclosed"] = True

        # --------------------------------------
        # Human-readable time
        # --------------------------------------
        try:
            human_time = datetime.strptime(model_time, "%H:%M").strftime("%-I:%M %p") if model_time else None
        except Exception:
            human_time = model_time

        if model_time and human_time:
            sms_body = sms_body.replace(model_time, human_time)

        # --------------------------------------
        # Save new values
        # --------------------------------------
        if model_date:
            sched["scheduled_date"] = model_date
        if model_time:
            sched["scheduled_time"] = model_time

        # --------------------------------------
        # AUTOBOOKING (Step 5)
        # Gate booking on address_verified, NOT raw_address.
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

                # Never claim "booked" unless Square returned a booking id.
                if sched.get("booking_created") and sched.get("square_booking_id"):
                    return {
                        "sms_body": (
                            f"You're all set — your appointment is booked for "
                            f"{sched['scheduled_date']} at {human_time} at {model_addr}. "
                            f"Your confirmation number is {sched.get('square_booking_id')}."
                        ),
                        "scheduled_date": sched["scheduled_date"],
                        "scheduled_time": sched["scheduled_time"],
                        "address": model_addr,
                        "booking_complete": True
                    }
            except Exception as e:
                print("[ERROR] Autobooking:", repr(e))

        # ---------------------------------------------------
        # HARD SAFETY: Never confirm appointment unless Square booked it
        # ---------------------------------------------------
        if not (sched.get("booking_created") and sched.get("square_booking_id")):
            confirmation_markers = [
                "is scheduled",
                "has been scheduled",
                "scheduled for",
                "booked for",
                "confirmation number",
                "you're all set",
                "your appointment"
            ]
            lowered = sms_body.lower()
            if any(m in lowered for m in confirmation_markers):
                if not sched.get("address_verified"):
                    sms_body = build_address_prompt(sched)
                elif not sched.get("scheduled_date"):
                    sms_body = "What date would you like to schedule the appointment?"
                elif not sched.get("scheduled_time"):
                    sms_body = "What time works best for you?"
                else:
                    sms_body = "One moment while I finish securing your appointment."

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
            "sms_body": "Sorry — can you say that again?",
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
# Create Square Booking (Corrected + Step 5B Compatible)
# ---------------------------------------------------
def maybe_create_square_booking(phone: str, convo: dict):
    sched = convo.setdefault("sched", {})
    profile = convo.setdefault("profile", {})
    current_job = convo.setdefault("current_job", {})

    # Never double-book
    if sched.get("booking_created") and sched.get("square_booking_id"):
        return

    scheduled_date   = sched.get("scheduled_date")
    scheduled_time   = sched.get("scheduled_time")
    raw_address      = (sched.get("raw_address") or "").strip()
    appointment_type = sched.get("appointment_type")

    # ✅ Step 5 hard gate: do NOT attempt Square booking until address is VERIFIED
    # This prevents "it thinks it booked" when the address is partial.
    if not sched.get("address_verified"):
        return

    if not (scheduled_date and scheduled_time and appointment_type):
        return

    variation_id, variation_version = map_appointment_type_to_variation(appointment_type)
    if not variation_id:
        print("[ERROR] Unknown appt_type:", appointment_type)
        return

    # weekend rule
    if is_weekend(scheduled_date) and appointment_type != "TROUBLESHOOT_395":
        print("[BLOCKED] Weekend not allowed for non-emergency.")
        return

    # time-window rule
    if appointment_type != "TROUBLESHOOT_395":
        if not is_within_normal_hours(scheduled_time):
            print("[BLOCKED] Time outside 9–4")
            return

    if not (SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID and SQUARE_TEAM_MEMBER_ID):
        print("[ERROR] Square not configured.")
        return

    # ---------------------------------------------------
    # Address normalization precedence (Step 5B)
    # 1) Use sched["normalized_address"] if it is a dict and complete
    # 2) Otherwise attempt normalize_address(raw_address)
    # ---------------------------------------------------
    addr_struct = sched.get("normalized_address") if isinstance(sched.get("normalized_address"), dict) else None

    def _is_complete_norm(a: dict) -> bool:
        return bool(
            isinstance(a, dict)
            and (a.get("address_line_1") or "").strip()
            and (a.get("locality") or "").strip()
            and (a.get("administrative_district_level_1") or "").strip()
            and (a.get("postal_code") or "").strip()
        )

    if not _is_complete_norm(addr_struct):
        # If we somehow reached here with address_verified True but no good normalized struct, re-normalize
        if not raw_address:
            # Nothing to normalize; bail safely
            sched["address_verified"] = False
            sched["address_missing"] = "street"
            return

        try:
            status, fresh = normalize_address(raw_address)
        except Exception as e:
            print("[ERROR] normalize_address exception:", repr(e))
            return

        # Preserve your existing "needs_state" flow
        if status == "needs_state":
            send_sms(phone, "Just to confirm, is this address in Connecticut or Massachusetts?")
            # Mark not verified until we have a full normalized address
            sched["address_verified"] = False
            sched["address_missing"] = "state"
            return

        # If normalization fails, do not book
        if status != "ok" or not isinstance(fresh, dict):
            print("[ERROR] Address normalization failed. status=", status)
            sched["address_verified"] = False
            sched["address_missing"] = "confirm"
            return

        addr_struct = fresh
        sched["normalized_address"] = addr_struct

    # Final check: must be complete
    if not _is_complete_norm(addr_struct):
        print("[ERROR] Normalized address missing required fields.")
        sched["address_verified"] = False
        sched["address_missing"] = "confirm"
        return

    # ---------------------------------------------------
    # Square booking.address FIX (remove unsupported field)
    # ---------------------------------------------------
    booking_address = dict(addr_struct)
    booking_address.pop("country", None)  # Square booking.address rejects country in some setups

    # ---------------------------------------------------
    # Travel check (uses normalized destination)
    # ---------------------------------------------------
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
            return

    # ---------------------------------------------------
    # Square customer (customer CAN include country)
    # ---------------------------------------------------
    customer_id = square_create_or_get_customer(phone, addr_struct)
    if not customer_id:
        print("[ERROR] Can't create/fetch customer.")
        return

    start_at_utc = parse_local_datetime(scheduled_date, scheduled_time)
    if not start_at_utc:
        print("[ERROR] Invalid start time.")
        return

    # Use normalized address in the idempotency key so partial text changes don't accidentally create dupes
    idempotency_key = (
        f"prevolt-{phone}-{scheduled_date}-{scheduled_time}-{appointment_type}-"
        f"{(addr_struct.get('address_line_1') or '').strip()}-"
        f"{(addr_struct.get('postal_code') or '').strip()}"
    )

    booking_payload = {
        "idempotency_key": idempotency_key,
        "booking": {
            "location_id": SQUARE_LOCATION_ID,
            "customer_id": customer_id,
            "start_at": start_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location_type": "CUSTOMER_LOCATION",
            "address": booking_address,
            "appointment_segments": [
                {
                    "duration_minutes": 60,
                    "service_variation_id": variation_id,
                    "service_variation_version": variation_version,
                    "team_member_id": SQUARE_TEAM_MEMBER_ID
                }
            ],
            "customer_note": (
                "Auto-booked by Prevolt OS. "
                f"Raw address: {raw_address or '[none]'} | "
                f"Normalized: {addr_struct.get('address_line_1')}, {addr_struct.get('locality')}, "
                f"{addr_struct.get('administrative_district_level_1')} {addr_struct.get('postal_code')}"
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
            return

        data = resp.json()
        booking = data.get("booking") or {}
        booking_id = booking.get("id")

        if not booking_id:
            print("[ERROR] booking_id missing.")
            return

        # ✅ Only mark booked when we have a real Square booking id
        sched["booking_created"] = True
        sched["square_booking_id"] = booking_id

        profile["upcoming_appointment"] = {
            "date": scheduled_date,
            "time": scheduled_time,
            "type": appointment_type,
            "square_id": booking_id
        }

        if current_job.get("job_type"):
            profile.setdefault("past_jobs", []).append({
                "type": current_job["job_type"],
                "date": scheduled_date
            })

        print("[SUCCESS] Booking created:", booking_id)

    except Exception as e:
        print("[ERROR] Square exception:", repr(e))


# ---------------------------------------------------
# Local Development Entrypoint
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
