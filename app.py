import os
import json
import re
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

app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def home():
    return "Prevolt OS running", 200


# ---------------------------------------------------
# In-Memory Conversation Store
# ---------------------------------------------------
conversations = {}


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
    Pull an explicit time out of a mixed customer message.
    Examples:
      - "2pm and I have dogs is that ok" -> "14:00"
      - "around 2:30 pm" -> "14:30"
      - "1500" -> "15:00"
    Returns None for vague phrases like "this afternoon" or "later".
    """
    import re

    s = (text or "").strip().lower()
    if not s:
        return None

    m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*([ap])\s*m', s, flags=re.I)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or '00')
        ap = m.group(3).lower()
        if hh == 12:
            hh = 0
        if ap == 'p':
            hh += 12
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    m = re.search(r'([01]?\d|2[0-3]):([0-5]\d)', s)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"

    m = re.search(r'(\d{3,4})', s)
    if m:
        raw = m.group(1).zfill(4)
        hh = int(raw[:2])
        mm = int(raw[2:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
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
    except Exception as e:
        print("[ERROR] SMS send failed:", repr(e))



def recompute_pending_step(profile: dict, sched: dict) -> None:
    active_first = (profile.get("active_first_name") or profile.get("first_name") or "").strip()
    active_last = (profile.get("active_last_name") or profile.get("last_name") or "").strip()
    active_email = (profile.get("active_email") or profile.get("email") or "").strip()

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
@app.route("/incoming-call", methods=["GET", "POST"])
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





def extract_city_state_from_reply(text: str) -> tuple[str | None, str | None]:
    """Parse compact city/state replies like 'Windsor CT' or 'Windsor, Connecticut'."""
    txt = " ".join((text or "").strip().replace(",", " ").split())
    if not txt:
        return None, None

    low = txt.lower()
    state = None
    if re.search(r"ct|connecticut", low):
        state = "CT"
        txt = re.sub(r"ct|connecticut", "", txt, flags=re.I).strip(" ,")
    elif re.search(r"ma|massachusetts", low):
        state = "MA"
        txt = re.sub(r"ma|massachusetts", "", txt, flags=re.I).strip(" ,")

    # Reject obvious non-city inputs
    if re.search(r"\d", txt):
        return None, state
    if re.search(r"(st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace)", txt, flags=re.I):
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

    update_address_assembly_state(sched)
    missing = (sched.get("address_missing") or "").strip().lower()
    if missing not in {"confirm", "state"}:
        return False

    raw = (sched.get("raw_address") or "").strip()
    if not raw:
        return False

    city, state = extract_city_state_from_reply(inbound)

    # State-only reply like 'CT'
    if not city and state:
        if re.search(r",\s*[A-Za-z .'-]+$", raw) and not re.search(r",\s*[A-Za-z .'-]+,\s*(CT|MA)", raw, flags=re.I):
            merged = f"{raw}, {state}"
        else:
            merged = f"{raw}, {state}"
    # City + optional state reply
    elif city:
        base = raw
        # If raw already ends with the same city/state, do nothing.
        if re.search(rf",\s*{re.escape(city)}(?:,\s*(CT|MA))?", raw, flags=re.I):
            merged = raw if not state else re.sub(r",\s*([A-Za-z .'-]+)(?:,\s*(CT|MA))?$", rf", {city}, {state}", raw, flags=re.I)
        else:
            merged = f"{base}, {city}" + (f", {state}" if state else "")
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



def force_capture_address_from_inbound(sched: dict, inbound_text: str) -> bool:
    """
    Deterministically capture address fragments from natural SMS replies.
    Handles:
      - "Bloomfield Ave" -> route only, asks for house number next
      - "45 Bloomfield Ave" -> full street+number, asks for town/state next
      - "54" while waiting on a house number -> merges into existing street
      - "It's 54 Bloomfield Ave sorry" -> overwrites the prior street candidate
    """
    txt = (inbound_text or "").strip()
    if not txt:
        return False

    low = txt.lower()
    # Skip obvious non-address informational questions.
    if any(x in low for x in ["how much", "price", "cost", "licensed", "insured", "card", "cash", "check", "payment", "permit", "dog", "dogs", "pet", "pets", "call when", "text when", "on the way"]):
        # But do not skip if there is a clear street address embedded.
        pass

    update_address_assembly_state(sched)
    current_missing = (sched.get("address_missing") or "").strip().lower()
    raw_existing = (sched.get("raw_address") or "").strip()

    cleaned = txt
    cleaned = re.sub(r"^(ok|okay|its|it's|it is|im at|i'm at|my address is|address is|it\s+is)[\s,:-]*", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"(sorry|thanks|thank you)", "", cleaned, flags=re.I).strip(' ,.-')

    street_pat = re.compile(
        r"(\d{1,6}\s+[A-Za-z0-9.'\- ]+?\s(?:st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace))",
        flags=re.I
    )
    route_only_pat = re.compile(
        r"([A-Za-z][A-Za-z0-9.'\- ]+?\s(?:st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace))",
        flags=re.I
    )

    # Full numbered street in one message.
    m_full = street_pat.search(cleaned)
    if m_full:
        candidate = " ".join(m_full.group(1).split()).strip(' ,')
        if candidate and candidate != raw_existing:
            sched["raw_address"] = candidate
            sched["normalized_address"] = None
            update_address_assembly_state(sched)
            return True
        return False

    # Customer only sent the missing house number while we already know the street.
    if current_missing == "number" and raw_existing:
        m_num = re.search(r"(\d{1,6})", cleaned)
        if m_num and not route_only_pat.search(cleaned):
            street_only = re.sub(r"^\d{1,6}\s+", "", raw_existing).strip()
            if street_only:
                sched["raw_address"] = f"{m_num.group(1)} {street_only}"
                sched["normalized_address"] = None
                update_address_assembly_state(sched)
                return True

    # Route-only street name, no number yet.
    m_route = route_only_pat.search(cleaned)
    if m_route:
        candidate = " ".join(m_route.group(1).split()).strip(' ,')
        if candidate:
            sched["raw_address"] = candidate
            sched["normalized_address"] = None
            sched["address_candidate"] = candidate
            sched["address_verified"] = False
            sched["address_missing"] = "number"
            sched["address_parts"] = {"street": True, "number": False, "city": False, "state": False, "zip": False, "source": "raw_address"}
            return True

    return False

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
        return humanize_question("What day works best for you?")
    if step == "need_time":
        if provided_ambiguous_time:
            if is_emergency:
                now_hour = datetime.now(ZoneInfo("America/New_York")).hour if ZoneInfo else datetime.utcnow().hour
                part = "morning" if now_hour < 12 else ("afternoon" if now_hour < 17 else "evening")
                return humanize_question(f"We can come today. What time later this {part} works for you?")
            return humanize_question("What time works for you?")
        return humanize_question("What time works best for you?")
    if step == "need_name":
        return humanize_question("What is your first and last name?")
    if step == "need_email":
        return humanize_question("What is the best email address for the appointment?")

    if step is None and not (sched.get("booking_created") and sched.get("square_booking_id")):
        if sched.get("scheduled_date") and sched.get("scheduled_time"):
            final_key = f"{sched.get('scheduled_date')}|{sched.get('scheduled_time')}"
            if sched.get("final_confirmation_accepted") and sched.get("last_final_confirmation_key") == final_key:
                return "Everything looks good here."
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
            }
        }
        resp = MessagingResponse()
        resp.message("✔ Memory reset complete for this number.")
        return Response(str(resp), mimetype="text/xml")

    # ---------------------------------------------------
    # Initialize layers (HARDENED: never assume dict shape)
    # ---------------------------------------------------
    conv = conversations.setdefault(phone, {})

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

    # Deterministic address capture for raw street / house-number replies.
    try:
        force_capture_address_from_inbound(sched, inbound_text)
    except Exception as e:
        print("[WARN] incoming_sms direct address capture failed:", repr(e))

    # If the customer just corrected or completed the address, refresh immediately.
    try:
        try_early_address_normalize(sched)
    except Exception as e:
        print("[WARN] incoming_sms early normalize failed:", repr(e))

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
        conv["last_inbound_sid"] = inbound_sid
        conv["last_inbound_fingerprint"] = inbound_fingerprint
        conv["last_inbound_fingerprint_ts"] = now_ts
        conv["last_sms_body"] = fast_interrupt.strip()
        tw = MessagingResponse()
        tw.message(fast_interrupt.strip())
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
    sched["scheduled_date"] = reply.get("scheduled_date")
    sched["scheduled_time"] = reply.get("scheduled_time")

    if reply.get("address"):
        sched["raw_address"] = reply["address"]
        # addresses list is guaranteed above
        if reply["address"] not in profile["addresses"]:
            profile["addresses"].append(reply["address"])

    # Re-derive address assembly state after Step 4 updates
    update_address_assembly_state(sched)

    # One more deterministic address pass in case the inbound text corrected the house number
    # but the model did not lock it cleanly.
    try:
        force_capture_address_from_inbound(sched, inbound_text)
        try_early_address_normalize(sched)
    except Exception as e:
        print("[WARN] incoming_sms post-step4 address capture failed:", repr(e))

    recompute_pending_step(profile, sched)

    sms_body = (reply.get("sms_body") or "").strip()
    next_prompt = choose_next_prompt_from_state(conv, inbound_text=inbound_text)

    # Route-level guardrail: only override when Step 4 returned a stall / generic filler.
    generic_fillers = {"", "Okay.", "Okay", "ok", "ok.", "sure.", "Sure."}
    if sms_body in generic_fillers:
        sms_body = next_prompt

    conv["last_inbound_sid"] = inbound_sid
    conv["last_inbound_fingerprint"] = inbound_fingerprint
    conv["last_inbound_fingerprint_ts"] = now_ts
    conv["last_sms_body"] = sms_body

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



def get_active_first_name(profile: dict) -> str:
    return (profile.get("active_first_name") or profile.get("first_name") or "").strip()

def get_active_last_name(profile: dict) -> str:
    return (profile.get("active_last_name") or profile.get("last_name") or "").strip()

def get_active_email(profile: dict) -> str:
    return (profile.get("active_email") or profile.get("email") or "").strip()

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

def extract_possible_person_name(text: str) -> tuple[str | None, str | None]:
    txt = " ".join((text or "").strip().split())
    if not txt:
        return None, None

    patterns = [
        r"\b(?:my name is|this is|i am|i'm)\s+([A-Za-z][A-Za-z'\-]{1,})(?:\s+([A-Za-z][A-Za-z'\-]{1,}))?",
        r"\bname\s+is\s+([A-Za-z][A-Za-z'\-]{1,})(?:\s+([A-Za-z][A-Za-z'\-]{1,}))?",
    ]
    for pat in patterns:
        m = re.search(pat, txt, flags=re.I)
        if m:
            first = normalize_person_name(m.group(1) or "") or None
            last = normalize_person_name(m.group(2) or "") or None
            return first, last
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
    yes_markers = ["yes", "yeah", "yep", "correct", "right", "thats right", "that's right"]
    if any(marker in low for marker in yes_markers):
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

    if not known_names and recognized_first:
        upsert_known_person(
            profile,
            first_name=recognized_first,
            last_name=profile.get("recognized_last_name") or "",
            email=profile.get("recognized_email") or "",
            square_customer_id=profile.get("square_customer_id") or None,
        )
        known_names = list_known_first_names(profile)

    if sched.get("name_engine_state"):
        return None

    # If voicemail first name matches a known person, lock onto that person.
    if voicemail_first:
        known = find_known_person_by_first(profile, voicemail_first)
        if known:
            apply_known_person_to_active(profile, known, source="known_person_match")
            return None

        # One known person on file but voicemail says a different first name -> clarify.
        if len(known_names) == 1:
            sched["name_engine_state"] = "awaiting_new_person_confirmation"
            sched["name_engine_candidate_first"] = voicemail_first
            sched["name_engine_candidate_last"] = voicemail_last
            sched["name_engine_expected_known_first"] = known_names[0]
            return wrap_name_engine_message(sched, f"I have {known_names[0]} on file from a past visit, but the voicemail said {voicemail_first}. Is this {voicemail_first}?", voicemail_first)

        # Multiple known people and voicemail gives a new first name -> clarify against the known names.
        if len(known_names) >= 2:
            joined = ", ".join(known_names[:-1]) + f" or {known_names[-1]}" if len(known_names) > 1 else known_names[0]
            sched["name_engine_state"] = "awaiting_new_person_confirmation_multi"
            sched["name_engine_candidate_first"] = voicemail_first
            sched["name_engine_candidate_last"] = voicemail_last
            return wrap_name_engine_message(sched, f"I have {joined} on file for this number, but the voicemail said {voicemail_first}. Is this {voicemail_first}?", voicemail_first)

        # No known people yet -> use voicemail first name as active first name, but still collect last/email later.
        profile["active_first_name"] = voicemail_first
        profile["first_name"] = voicemail_first
        profile["identity_source"] = "voicemail_first_name"
        return None

    # No voicemail name, multiple known people on the number -> ask who is calling.
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
            corrected_first, _ = extract_possible_person_name(inbound_text)
            chosen_first = candidate_first
            if corrected_first:
                corrected_first = normalize_person_name(corrected_first)
                if corrected_first and corrected_first.lower() != candidate_first.lower():
                    chosen_first = corrected_first
            profile["active_first_name"] = chosen_first
            profile["first_name"] = chosen_first
            profile["identity_source"] = "new_person_confirmed_from_voicemail"
            sched["name_engine_candidate_first"] = chosen_first
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

def interruption_answer_and_return_prompt(conv: dict, inbound_text: str) -> str | None:
    profile = conv.setdefault("profile", {})
    sched = conv.setdefault("sched", {})
    low = (inbound_text or "").lower().strip()
    if not low:
        return None

    # Only use this during active booking, not after booking.
    if sched.get("booking_created") and sched.get("square_booking_id"):
        return None

    answer = None
    if any(x in low for x in ["dog", "dogs", "pet", "pets"]):
        answer = "Yes, please make sure we can safely get to the panel when we arrive."
    elif any(x in low for x in ["licensed", "insured"]):
        answer = "Yes, we're licensed and insured."
    elif any(x in low for x in ["call when", "text when", "on the way", "arrival window"]):
        answer = "Yes, you'll get a text when we're on the way."
    elif any(x in low for x in ["do i need to buy", "bring anything", "materials"]):
        answer = "Nope, we bring what we need for the visit."
    elif any(x in low for x in ["permit", "permit required"]):
        answer = "If anything needs a permit, we'll go over that during the visit."
    elif any(x in low for x in ["card", "cash", "check", "payment"]):
        answer = "Card or cash after the visit is fine."
    elif any(x in low for x in ["how much", "price", "cost", "$195", "$395"]):
        appt = (sched.get("appointment_type") or "").upper()
        if "TROUBLESHOOT" in appt:
            answer = "The $395 is the troubleshoot and repair visit to come out and diagnose the issue."
        elif "INSPECTION" in appt:
            answer = "Whole-home inspections run $395, and larger homes can range higher depending on square footage."
        else:
            answer = "The $195 is the service visit to come out, evaluate the issue, and go over the next step."
        sched["price_disclosed"] = True

    if not answer:
        return None

    recompute_pending_step(profile, sched)

    # Do not restate the final confirmation while answering an interrupting question.
    if sched.get("final_confirmation_sent") or sched.get("final_confirmation_accepted"):
        if not sched.get("pending_step") and sched.get("scheduled_date") and sched.get("scheduled_time"):
            return answer

    next_prompt = choose_next_prompt_from_state(conv, inbound_text="")
    if next_prompt and next_prompt not in {"Okay.", "Okay", answer, "Everything looks good here."}:
        return f"{answer} {next_prompt}"
    return answer


def looks_like_slot_payload(inbound_text: str) -> bool:
    txt = (inbound_text or "").strip()
    low = txt.lower()
    if not txt:
        return False
    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", txt, flags=re.I):
        return True
    if re.search(r"\d{1,6}", txt) and re.search(r"(st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace)", low, flags=re.I):
        return True
    if re.search(r"\d{1,2}(:\d{2})?\s*(am|pm)", low, flags=re.I):
        return True
    if re.search(r"^\d{3,4}$", txt):
        return True
    if any(day in low for day in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday","tomorrow","today","next monday","next tuesday","next wednesday","next thursday","next friday","next saturday","next sunday"]):
        return True
    if len(txt.split()) <= 3 and re.fullmatch(r"[A-Za-z'\- ]+", txt):
        return True
    return False


def should_short_circuit_interrupt(conv: dict, inbound_text: str) -> bool:
    sched = conv.setdefault("sched", {})
    if sched.get("booking_created") and sched.get("square_booking_id"):
        return False
    if looks_like_slot_payload(inbound_text):
        return False
    # If the message contains a street/address fragment, do not short-circuit.
    if re.search(r"\b(?:\d{1,6}\s+)?[A-Za-z][A-Za-z0-9.'\- ]+\s(?:st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|court|cir|circle|blvd|boulevard|way|pkwy|parkway|ter|terrace)\b", (inbound_text or ""), flags=re.I):
        return False
    low = (inbound_text or "").lower().strip()
    interrupt_markers = [
        "how much", "price", "cost", "$195", "$395", "licensed", "insured",
        "card", "cash", "check", "payment", "permit", "permit required",
        "dog", "dogs", "pet", "pets", "call when", "text when", "on the way",
        "arrival window", "bring anything", "materials"
    ]
    return any(x in low for x in interrupt_markers)

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

    closed_thread = bool(
        (sched.get("booking_created") and sched.get("square_booking_id"))
        or sched.get("final_confirmation_accepted")
        or sched.get("final_confirmation_sent")
        or profile.get("upcoming_appointment")
    )
    if not closed_thread:
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
        sched["final_confirmation_sent"] = False
        sched["final_confirmation_accepted"] = False
        return None

    if any(x in low for x in ["picture", "pictures", "photo", "photos", "image", "images"]):
        return "Yes, you can send them over if you'd like, but we'll still evaluate everything in person."

    if any(x in low for x in ["prepare", "prep", "anything i should do", "what should i do"]):
        return "Nothing special. Just make sure we can safely get to the panel and work area when we arrive."

    if any(x in low for x in ["card", "cash", "check", "payment", "pay", "forms of payment"]):
        return "Card or cash after the visit is totally fine."

    if any(x in low for x in ["dog", "dogs", "pet", "pets"]):
        return "That is fine. Just make sure we can safely get to the panel when we arrive."

    if any(x in low for x in ["gate", "code", "lock", "locked", "access", "doorbell", "call when outside"]):
        return "That is fine. Just make sure we can get to the panel when we arrive."

    if any(x in low for x in ["what time", "when are you coming", "what day", "when is my appointment", "are we good"]):
        date_txt = (appt.get("date") or sched.get("scheduled_date") or "").strip()
        time_txt = humanize_time(appt.get("time") or sched.get("scheduled_time") or "")
        if date_txt and time_txt:
            return f"You're all set for {date_txt} at {time_txt}."
        if date_txt:
            return f"You're all set for {date_txt}."
        return "You're all set for the visit."

    if any(x in low for x in ["who is coming", "who's coming", "whos coming", "on the way", "arrival window", "will they call", "text when"]):
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

        # Hazard detection must override post-booking mode.
        EMERGENCY_KEYWORDS = [
            "tree fell", "tree down", "power line", "lines down",
            "service ripped", "sparking", "burning", "fire",
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
        if sched.get("pending_step") == "need_name" or "my name" in inbound_lower or inbound_lower.startswith("name is"):
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

        # --------------------------------------
        # Emergency flow (2-step confirmation)
        # --------------------------------------
        EMERGENCY = IS_EMERGENCY

        if IS_EMERGENCY and not sched["awaiting_emergency_confirm"] and not sched["emergency_approved"]:
            sched["appointment_type"] = "TROUBLESHOOT_395"
            sched["awaiting_emergency_confirm"] = True
            sched["pending_step"] = None

            if not sched.get("raw_address"):
                msg = "This sounds urgent. " + build_address_prompt(sched)
                msg = _finalize_sms(msg, "TROUBLESHOOT_395", booking_created=False)
                return {
                    "sms_body": msg,
                    "scheduled_date": None,
                    "scheduled_time": None,
                    "address": None,
                    "booking_complete": False
                }

            msg = (
                f"This looks urgent at {sched.get('raw_address')}. "
                "Troubleshoot and repair visits are $395. "
                "Do you want us to dispatch someone now?"
            )
            msg = _finalize_sms(msg, "TROUBLESHOOT_395", booking_created=False)
            return {
                "sms_body": msg,
                "scheduled_date": None,
                "scheduled_time": None,
                "address": sched.get("raw_address"),
                "booking_complete": False
            }

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
        if not model_time:
            salvaged_time = extract_explicit_time_from_text(inbound_text)
            if salvaged_time:
                model_time = salvaged_time

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
            sched["scheduled_time"] = model_time

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
        ready_for_booking = (
            bool(sched.get("scheduled_date")) and
            bool(sched.get("scheduled_time")) and
            bool(sched.get("address_verified")) and
            bool(sched.get("appointment_type")) and
            bool(get_active_first_name(profile)) and
            bool(get_active_last_name(profile)) and
            bool(get_active_email(profile)) and
            not sched.get("pending_step") and
            not sched.get("booking_created")
        )

        if ready_for_booking or (EMERGENCY and bool(get_active_first_name(profile)) and bool(get_active_last_name(profile)) and bool(get_active_email(profile))):
            try:
                maybe_create_square_booking(phone, conv)

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
    import re

    sched = convo.setdefault("sched", {})
    profile = convo.setdefault("profile", {})
    current_job = convo.setdefault("current_job", {})

    # Harden profile keys (prevents KeyError elsewhere)
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

    active_first = (profile.get("active_first_name") or profile.get("first_name") or "").strip()
    active_last = (profile.get("active_last_name") or profile.get("last_name") or "").strip()
    active_email = (profile.get("active_email") or profile.get("email") or "").strip()

    profile["first_name"] = active_first
    profile["last_name"] = active_last
    profile["email"] = active_email

    if not (active_first and active_last and active_email):
        return


    # Never double-book
    if sched.get("booking_created") and sched.get("square_booking_id"):
        return

    # Clear stale booking state before a fresh attempt
    sched["booking_created"] = False
    sched["square_booking_id"] = None

    scheduled_date   = sched.get("scheduled_date")
    scheduled_time   = sched.get("scheduled_time")
    raw_address      = (sched.get("raw_address") or "").strip()
    appointment_type = sched.get("appointment_type")

    # ✅ Step 5 hard gate: do NOT attempt Square booking until address is VERIFIED
    # (Verified now MUST mean "has a house number" per your patched address state helper.)
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

    def _has_house_number(line1: str) -> bool:
        line1 = (line1 or "").strip()
        return bool(re.match(r"^\d{1,6}\b", line1))

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
    # ✅ HARD BLOCK if no house number in normalized address_line_1
    # ---------------------------------------------------
    line1 = (addr_struct.get("address_line_1") or "").strip()
    if not _has_house_number(line1):
        sched["address_verified"] = False
        sched["address_missing"] = "number"
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
    customer_id = square_create_or_get_customer(phone, profile, addr_struct)
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
                f"{(addr_struct.get('administrative_district_level_1') or '').strip()} {(addr_struct.get('postal_code') or '').strip()}"
            )
        }
    }

    try:
        # -----------------------------
        # 1) Create booking
        # -----------------------------
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
            print("[ERROR] booking_id missing. payload=", data)
            return

        # -----------------------------
        # 2) Retrieve booking to confirm status/version
        # -----------------------------
        r2 = requests.get(
            f"https://connect.squareup.com/v2/bookings/{booking_id}",
            headers=square_headers(),
            timeout=12,
        )
        if r2.status_code not in (200, 201):
            print("[ERROR] retrieve booking failed:", r2.status_code, r2.text)
            return

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

        # -----------------------------
        # 3) Accept booking if not already accepted/confirmed
        # -----------------------------
        if status not in ("ACCEPTED", "CONFIRMED"):
            if version is None:
                print("[ERROR] booking version missing; cannot accept.")
                return

            accept_payload = {
                "booking": {
                    "version": version,
                    "status": "ACCEPTED",
                }
            }

            r3 = requests.put(
                f"https://connect.squareup.com/v2/bookings/{booking_id}",
                headers=square_headers(),
                json=accept_payload,
                timeout=12,
            )
            if r3.status_code not in (200, 201):
                print("[ERROR] accept booking failed:", r3.status_code, r3.text)
                return

            b3 = (r3.json() or {}).get("booking") or {}
            status2 = (b3.get("status") or "").upper()
            print("[DEBUG] booking accepted:", {"id": booking_id, "status": status2})

            if status2 not in ("ACCEPTED", "CONFIRMED"):
                print("[ERROR] booking did not end in ACCEPTED/CONFIRMED. status=", status2)
                return

        # ✅ Only mark booked AFTER create + retrieve + accept succeeds
        sched["booking_created"] = True
        sched["square_booking_id"] = booking_id

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

    except Exception as e:
        print("[ERROR] Square exception:", repr(e))



# ---------------------------------------------------
# Local Development Entrypoint
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
