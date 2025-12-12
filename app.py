import os
import json
import time
import uuid
import requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather, Dial
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from openai import OpenAI



try:
    # Python 3.9+
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
SQUARE_TEAM_MEMBER_ID = os.environ.get("SQUARE_TEAM_MEMBER_ID")  # required for bookings

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
DISPATCH_ORIGIN_ADDRESS = os.environ.get("DISPATCH_ORIGIN_ADDRESS")  # e.g. "Granby, CT"
TECH_CURRENT_ADDRESS = os.environ.get("TECH_CURRENT_ADDRESS")        # optional dynamic origin

# ---------------------------------------------------
# Square service variation data (final and verified)
# ---------------------------------------------------
SERVICE_VARIATION_EVAL_ID = "IPCUF6EPOYGWJUEFUZOXL2AZ"
SERVICE_VARIATION_EVAL_VERSION = 1764725435505

SERVICE_VARIATION_INSPECTION_ID = "LYK646AH4NAESCFUZL6PUTZ2"
SERVICE_VARIATION_INSPECTION_VERSION = 1764725393938

SERVICE_VARIATION_TROUBLESHOOT_ID = "64IQNJYO3H6XNTLPIHABDJOQ"
SERVICE_VARIATION_TROUBLESHOOT_VERSION = 1762464315698

# Non-emergency booking window (local time)
BOOKING_START_HOUR = 9   # 9:00
BOOKING_END_HOUR = 16    # 16:00 (4pm)
MAX_TRAVEL_MINUTES = 60  # max 1 hour travel

openai_client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = (
    Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
    else None
)

app = Flask(__name__)

# Conversation memory (in-memory for now)
conversations = {}
# conversations[phone] = {
#   "cleaned_transcript": ...,
#   "category": ...,
#   "appointment_type": ...,
#   "initial_sms": ...,
#   "first_sms_time": ...,
#   "replied": ...,
#   "followup_sent": ...,
#   "scheduled_date": ...,
#   "scheduled_time": ...,
#   "address": ... (raw freeform from customer),
#   "normalized_address": {...} or None,
#   "booking_created": bool,
#   "square_booking_id": str | None,
#   "state_prompt_sent": bool,
# }


# ---------------------------------------------------
# Utility: Send outbound message via WhatsApp (testing)
# ---------------------------------------------------
def send_sms(to_number: str, body: str) -> None:
    """
    For now, force all outbound messages to WhatsApp sandbox to your cell
    so you can test end-to-end without A2P headaches.
    """
    if not twilio_client:
        print("Twilio not configured; WhatsApp message not sent.")
        print("Intended WhatsApp message:", body)
        return

    try:
        whatsapp_from = "whatsapp:+14155238886"  # Twilio Sandbox
        whatsapp_to = "whatsapp:+18609701727"    # <-- YOUR CELL

        msg = twilio_client.messages.create(
            body=body,
            from_=whatsapp_from,
            to=whatsapp_to,
        )
        print("WhatsApp sent. SID:", msg.sid)

    except Exception as e:
        print("Failed to send WhatsApp message:", repr(e))


# ---------------------------------------------------
# Step 1 — Transcription (Twilio → Whisper)
# ---------------------------------------------------
def transcribe_recording(recording_url: str) -> str:
    """
    Downloads the voicemail from Twilio (wav or mp3) and sends it to Whisper.
    Auto-fallback: try .wav first, then .mp3.
    """

    # --- 1) Try WAV first ---
    wav_url = recording_url + ".wav"
    mp3_url = recording_url + ".mp3"

    def download(url):
        resp = requests.get(
            url,
            stream=True,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp

    resp = download(wav_url)
    if resp is None:
        resp = download(mp3_url)
    if resp is None:
        print("Voicemail download FAILED for:", recording_url)
        return ""

    # --- 2) Save temp file ---
    tmp_path = "/tmp/prevolt_voicemail.wav"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    # --- 3) Send to Whisper ---
    with open(tmp_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

    return transcript.text.strip()

# ---------------------------------------------------
# Step 2 — Cleanup (improve clarity)
# ---------------------------------------------------
def clean_transcript_text(raw_text: str) -> str:
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You clean up voicemail transcriptions for an electrical contractor. "
                        "Fix obvious transcription mistakes and electrical terminology, "
                        "improve grammar slightly, but preserve the customer's meaning EXACTLY. "
                        "Do NOT add details. Do NOT change the problem description."
                    ),
                },
                {"role": "user", "content": raw_text},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("Cleanup FAILED:", repr(e))
        return raw_text



# ---------------------------------------------------
# Step 3 — Voicemail Classifier + Info Extraction (NO SMS GENERATION)
# ---------------------------------------------------
def generate_initial_sms(cleaned_text: str) -> dict:
    """
    NEW STEP 3 (Option C):
    • Does NOT generate SMS.
    • Only classifies voicemail + extracts address/date/time/intent.
    • Step 4 will generate ALL outgoing text.
    """
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Prevolt OS, the voicemail classifier for Prevolt Electric.\n"
                        "You DO NOT write SMS messages.\n"
                        "Your ONLY job is to extract structured data from the voicemail.\n\n"

                        "========================================\n"
                        "APPOINTMENT TYPE RULES\n"
                        "========================================\n"
                        "• TROUBLESHOOT_395 = ANY outage, burning smell, dangerous issue, "
                        "  breaker problems, sparking, fire risk, wires down, tree damage, etc.\n"
                        "• WHOLE_HOME_INSPECTION = ONLY if they explicitly say: "
                        "  'inspection', 'whole home inspection', 'electrical inspection'.\n"
                        "• EVAL_195 = Everything else: installs, upgrades, quotes, EV chargers, "
                        "  generators, renovations, pricing questions, consultations.\n"
                        "• When unsure between eval and troubleshoot → choose TROUBLESHOOT.\n\n"

                        "========================================\n"
                        "NATURAL LANGUAGE DATE/TIME EXTRACTION\n"
                        "========================================\n"
                        "Interpret natural language dates such as:\n"
                        "• 'next Tuesday'\n"
                        "• 'this Friday'\n"
                        "• 'on the 21st'\n"
                        "• 'tomorrow'\n"
                        "• 'Sunday morning'\n\n"
                        "Convert them into real calendar values:\n"
                        "• scheduled_date = 'YYYY-MM-DD' or null\n"
                        "• scheduled_time = 'HH:MM' in 24-hour time or null\n\n"

                        "========================================\n"
                        "ADDRESS EXTRACTION\n"
                        "========================================\n"
                        "If voicemail mentions a street, road, ave, lane, etc., extract it.\n"
                        "Otherwise return null.\n\n"

                        "========================================\n"
                        "INTENT DETECTION\n"
                        "========================================\n"
                        "intent must be ONE of:\n"
                        "• 'schedule' — caller clearly wants an appointment\n"
                        "• 'quote' — caller is pricing or exploring options\n"
                        "• 'emergency' — outage, burning smell, wires, tree damage, anything urgent\n"
                        "• 'other' — cannot determine\n\n"

                        "========================================\n"
                        "REQUIRED STRICT JSON OUTPUT\n"
                        "========================================\n"
                        "{\n"
                        "  'category': 'string',\n"
                        "  'appointment_type': 'string',\n"
                        "  'detected_address': 'string or null',\n"
                        "  'detected_date': 'YYYY-MM-DD or null',\n"
                        "  'detected_time': 'HH:MM or null',\n"
                        "  'intent': 'schedule' | 'quote' | 'emergency' | 'other'\n"
                        "}\n"
                        "NO OTHER FIELDS. NO SMS BODY. NO FLUFF."
                    ),
                },
                {"role": "user", "content": cleaned_text},
            ],
        )

        data = json.loads(completion.choices[0].message.content)

        # Return classification + extracted fields ONLY
        return {
            "category": data.get("category"),
            "appointment_type": data.get("appointment_type"),
            "detected_address": data.get("detected_address"),
            "detected_date": data.get("detected_date"),
            "detected_time": data.get("detected_time"),
            "intent": data.get("intent")
        }

    except Exception as e:
        print("Voicemail classifier FAILED:", repr(e))
        # Fail-safe: still NO SMS — Step 4 will handle fallback messaging
        return {
            "category": "OTHER",
            "appointment_type": "EVAL_195",
            "detected_address": None,
            "detected_date": None,
            "detected_time": None,
            "intent": "other"
        }



# ---------------------------------------------------
# NEW — Incoming SMS Webhook (B-3 State Machine Version, Option A Applied)
# ---------------------------------------------------
@app.route("/incoming-sms", methods=["POST"])
def incoming_sms():
    from twilio.twiml.messaging_response import MessagingResponse

    inbound_text = request.form.get("Body", "") or ""
    phone        = request.form.get("From", "").replace("whatsapp:", "")
    inbound_low  = inbound_text.lower().strip()

    # ==========================================================
    # SECRET MEMORY WIPE COMMAND
    # ==========================================================
    if inbound_low == "mobius1":
        conversations[phone] = {
            "profile": {
                "name": None,
                "addresses": [],
                "upcoming_appointment": None,
                "past_jobs": []
            },
            "current_job": {
                "job_type": None,
                "raw_description": None
            },
            "sched": {
                "pending_step": None,
                "scheduled_date": None,
                "scheduled_time": None,
                "appointment_type": None,
                "normalized_address": None,
                "raw_address": None,
                "booking_created": False
            }
        }
        resp = MessagingResponse()
        resp.message("✔ Memory reset complete for this number.")
        return Response(str(resp), mimetype="text/xml")

    # ==========================================================
    # LAYER INITIALIZATION
    # ==========================================================
    conv = conversations.setdefault(phone, {})

    profile = conv.setdefault("profile", {
        "name": None,
        "addresses": [],
        "upcoming_appointment": None,
        "past_jobs": []
    })

    current_job = conv.setdefault("current_job", {
        "job_type": None,
        "raw_description": None
    })

    sched = conv.setdefault("sched", {
        "pending_step": None,
        "scheduled_date": None,
        "scheduled_time": None,
        "appointment_type": None,
        "normalized_address": None,
        "raw_address": None,
        "booking_created": False
    })

    # ==========================================================
    # INITIAL VARIABLE LOAD
    # ==========================================================
    cleaned_transcript = conv.get("cleaned_transcript")
    category           = conv.get("category")
    appointment_type   = sched.get("appointment_type")
    initial_sms        = conv.get("initial_sms")

    scheduled_date = sched.get("scheduled_date")
    scheduled_time = sched.get("scheduled_time")

    address = (
        sched.get("raw_address")
        or sched.get("normalized_address")
        or None
    )

    # ==========================================================
    # PATCH A1 — ADDRESS EXTRACTION FROM INITIAL SMS
    # ==========================================================
    try:
        if initial_sms:
            import re

            # Parentheses extraction
            m = re.search(r"\((.*?)\)", initial_sms)
            if m:
                extracted = m.group(1).strip()
                if extracted and not sched.get("raw_address"):
                    sched["raw_address"] = extracted
                    address = extracted
                    if extracted not in profile["addresses"]:
                        profile["addresses"].append(extracted)

            # Street-pattern extraction
            if not sched.get("raw_address"):
                street_pattern = r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+(st|street|ave|avenue|rd|road|ln|lane|dr|drive|ct|circle|blvd|way)\b"
                m2 = re.search(street_pattern, initial_sms, flags=re.IGNORECASE)
                if m2:
                    extracted2 = m2.group(0).strip()
                    sched["raw_address"] = extracted2
                    address = extracted2
                    if extracted2 not in profile["addresses"]:
                        profile["addresses"].append(extracted2)
    except Exception as e:
        print("Initial SMS address extraction failed:", repr(e))

    # ==========================================================
    # STATE MACHINE — pending_step (Option A naming)
    # ==========================================================
    if not scheduled_date:
        sched["pending_step"] = "need_date"
    elif not scheduled_time:
        sched["pending_step"] = "need_time"
    elif not address:
        sched["pending_step"] = "need_address"
    else:
        sched["pending_step"] = None  # ready-to-book (if Step 4 agrees)

    # ==========================================================
    # PASS CONTEXT TO STEP 4 ENGINE
    # ==========================================================
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

    # ==========================================================
    # APPLY STEP 4 UPDATES
    # ==========================================================
    sched["scheduled_date"] = reply.get("scheduled_date")
    sched["scheduled_time"] = reply.get("scheduled_time")

    new_addr = reply.get("address")
    if new_addr:
        sched["raw_address"] = new_addr
        if new_addr not in profile["addresses"]:
            profile["addresses"].append(new_addr)

    # ==========================================================
    # RECHECK STATE AFTER STEP 4 (Option A naming)
    # ==========================================================
    if not sched["scheduled_date"]:
        sched["pending_step"] = "need_date"
    elif not sched["scheduled_time"]:
        sched["pending_step"] = "need_time"
    elif not sched["raw_address"]:
        sched["pending_step"] = "need_address"
    else:
        sched["pending_step"] = None

    # ==========================================================
    # RETURN TWILIO RESPONSE
    # ==========================================================
    twilio_reply = MessagingResponse()
    twilio_reply.message(reply["sms_body"])
    return Response(str(twilio_reply), mimetype="text/xml")


    # ==========================================================
    # PASS CONTEXT TO STEP 4
    # ==========================================================
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

    # ==========================================================
    # APPLY STEP 4 UPDATES
    # ==========================================================
    # Date/time always inherited or updated safely
    sched["scheduled_date"] = reply.get("scheduled_date")
    sched["scheduled_time"] = reply.get("scheduled_time")

    # Address update SAFE HANDLING
    new_addr = reply.get("address")
    if new_addr:
        # treat as raw address first
        sched["raw_address"] = new_addr
        if new_addr not in profile["addresses"]:
            profile["addresses"].append(new_addr)

    # ==========================================================
    # STATE RE-EVALUATE AFTER STEP 4
    # ==========================================================
    if not sched["scheduled_date"]:
        sched["pending_step"] = "ask_date"
    elif not sched["scheduled_time"]:
        sched["pending_step"] = "ask_time"
    elif not sched["raw_address"]:
        sched["pending_step"] = "ask_address"
    else:
        sched["pending_step"] = None

    # ==========================================================
    # RETURN SMS
    # ==========================================================
    twilio_reply = MessagingResponse()
    twilio_reply.message(reply["sms_body"])
    return Response(str(twilio_reply), mimetype="text/xml")


    

# ---------------------------------------------------
# Build System Prompt (Prevolt Rules Engine)
# With Reset-Lock + Pending-Step Logic (B-4)
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
    convo
):

    global PREVOLT_RULES_CACHE

    # Load rules only once
    if PREVOLT_RULES_CACHE is None:
        with open("prevolt_rules.json", "r", encoding="utf-8") as f:
            PREVOLT_RULES_CACHE = json.load(f)

    rules_text = PREVOLT_RULES_CACHE.get("rules", "")

    # ---------------------------------------------------
    # Voicemail context (optional, safe to include)
    # ---------------------------------------------------
    voicemail_intent = convo.get("voicemail_intent")
    voicemail_town = convo.get("voicemail_town")
    voicemail_partial_address = convo.get("voicemail_partial_address")

    voicemail_context = ""
    if voicemail_intent or voicemail_town or voicemail_partial_address:
        voicemail_context += (
            "\n\n===================================================\n"
            "VOICEMAIL INSIGHTS (PRE-EXTRACTED)\n"
            "===================================================\n"
        )
        if voicemail_intent:
            voicemail_context += f"Intent mentioned in voicemail: {voicemail_intent}\n"
        if voicemail_town:
            voicemail_context += f"Town detected: {voicemail_town}\n"
        if voicemail_partial_address:
            voicemail_context += f"Partial address detected: {voicemail_partial_address}\n"

    # ---------------------------------------------------
    # Reset-Lock (C1)
    # ---------------------------------------------------
    llm_reset_lock = f"""
===================================================
LLM RESET-LOCK RULES (MANDATORY)
===================================================
You MUST obey these rules with zero exceptions:

1. If a value is already known, YOU MUST NOT change it:
   - Scheduled date: {scheduled_date}
   - Scheduled time: {scheduled_time}
   - Address: {address}

2. You MUST NOT output null for any of these fields if they already have a value.

3. If the customer's message does NOT provide a new date/time/address,
   you MUST reuse the stored value above.

4. If the stored value is present, you may NOT reinterpret or modify it.
   You must return it exactly.

5. If the customer's message contains a new address/date/time,
   ONLY THEN are you allowed to update it — never otherwise.

6. If uncertain, ALWAYS inherit the previous value.

7. NEVER re-ask for information that already exists in memory.
"""

    # ---------------------------------------------------
    # Pending-Step Logic (B-4 Injection)
    # ---------------------------------------------------
    pending_step = convo.get("sched", {}).get("pending_step")

    pending_step_rules = f"""
===================================================
PENDING-STEP LOGIC (MANDATORY)
===================================================
Current pending_step: {pending_step}

You MUST follow these rules:

1. NEVER ask for a value that already exists in memory.
   If date/time/address exists, DO NOT ask for it again.

2. ALWAYS ask ONLY for the next required value based on pending_step:
   • If pending_step == "need_date": ask ONLY for a date.
   • If pending_step == "need_time": ask ONLY for a time.
   • If pending_step == "need_address": ask ONLY for the address.
   • If pending_step is None and all values exist: DO NOT ask questions — finalize.

3. ADVANCING STEPS:
   When the user provides the correct missing value:
   • Providing a date → next step is "need_time".
   • Providing a time → next step is "need_address".
   • Providing an address → pending_step becomes null.

4. NEVER advance multiple steps at once.
   Ignore values that are not part of the current step.

5. DO NOT modify or reinterpret any known value.

6. When all values exist and pending_step is None,
   you MUST produce a final confirmation message.

7. NEVER output contradictory guidance.
"""

    # ---------------------------------------------------
    # Required JSON output spec
    # ---------------------------------------------------
    output_block = (
        "{\n"
        '  "sms_body": "string",\n'
        '  "scheduled_date": "YYYY-MM-DD or null",\n'
        '  "scheduled_time": "HH:MM or null",\n'
        '  "address": "string or null"\n'
        "}"
    )

    # ---------------------------------------------------
    # Assemble final system prompt
    # ---------------------------------------------------
    system_prompt = (
        "You are Prevolt OS, the SMS assistant for Prevolt Electric.\n"
        "You MUST respond ONLY in strict JSON.\n\n"
        f"Today is {today_date_str}, a {today_weekday}.\n\n"
        f"{rules_text}"
        f"{voicemail_context}\n"
        f"{llm_reset_lock}\n"
        f"{pending_step_rules}\n"
        "===================================================\n"
        "CURRENT CONTEXT\n"
        "===================================================\n"
        f"Original voicemail: {cleaned_transcript}\n"
        f"Category: {category}\n"
        f"Appointment type: {appointment_type}\n"
        f"Initial SMS: {initial_sms}\n"
        f"Stored date: {scheduled_date}\n"
        f"Stored time: {scheduled_time}\n"
        f"Stored address: {address}\n\n"
        "===================================================\n"
        "REQUIRED JSON OUTPUT FORMAT\n"
        "===================================================\n"
        f"{output_block}\n"
    )

    return system_prompt

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
        # Conversation + scheduler layers
        # --------------------------------------
        phone = request.form.get("From", "").replace("whatsapp:", "")
        conv  = conversations.setdefault(phone, {})
        sched = conv.setdefault("sched", {})

        sched.setdefault("raw_address", None)
        sched.setdefault("normalized_address", None)
        sched.setdefault("pending_step", None)

        inbound_text  = inbound_text or ""
        inbound_lower = inbound_text.lower()

        # --------------------------------------
        # Appointment type fallback
        # --------------------------------------
        appt_type = sched.get("appointment_type") or appointment_type
        if not appt_type:
            if any(w in inbound_lower for w in [
                "not working","no power","dead","sparking","burning",
                "breaker keeps","gfci","outlet not","troubleshoot"
            ]):
                appt_type = "TROUBLESHOOT_395"
            elif any(w in inbound_lower for w in [
                "inspection","whole home inspection","electrical inspection"
            ]):
                appt_type = "WHOLE_HOME_INSPECTION"
            else:
                appt_type = "EVAL_195"
            sched["appointment_type"] = appt_type

        # --------------------------------------
        # HARD ADDRESS EXTRACTION (A1)
        # --------------------------------------
        address_markers = [
            " st", " street", " ave", " avenue", " rd", " road",
            " ln", " lane", " dr", " drive", " ct", " circle",
            " blvd", " way"
        ]
        if any(m in inbound_lower for m in address_markers) and len(inbound_text) > 6:
            sched["raw_address"] = inbound_text.strip()
            address = sched["raw_address"]

        # --------------------------------------
        # Hybrid Missing-Info Resolver (Option B)
        # --------------------------------------
        missing_date = not (scheduled_date or sched.get("scheduled_date"))
        missing_time = not (scheduled_time or sched.get("scheduled_time"))
        missing_addr = not (address or sched.get("raw_address"))

        # LLM decides phrasing → Step 4 enforces correctness
        expected_next_step = None
        if missing_addr:
            expected_next_step = "need_address"
        elif missing_date:
            expected_next_step = "need_date"
        elif missing_time:
            expected_next_step = "need_time"

        sched["pending_step"] = expected_next_step

        # --------------------------------------
        # Build System Prompt (RESET-LOCK)
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
        # LLM CALL
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
        # RESET-LOCK — Never wipe known values
        # --------------------------------------
        if sched.get("scheduled_date") and not model_date:
            model_date = sched["scheduled_date"]

        if sched.get("scheduled_time") and not model_time:
            model_time = sched["scheduled_time"]

        if sched.get("raw_address") and not model_addr:
            model_addr = sched["raw_address"]

        # --------------------------------------
        # ADDRESS FINALIZATION
        # --------------------------------------
        if isinstance(model_addr, str) and len(model_addr) > 5:
            sched["raw_address"] = model_addr
        model_addr = sched.get("raw_address")

        # --------------------------------------
        # PRICE INJECTION
        # --------------------------------------
        sms_body = apply_price_injection(appt_type, sms_body)

        # --------------------------------------
        # Human-readable time
        # --------------------------------------
        try:
            human_time = datetime.strptime(model_time, "%H:%M").strftime("%-I:%M %p") \
                if model_time else None
        except Exception:
            human_time = model_time

        if model_time and human_time:
            sms_body = sms_body.replace(model_time, human_time)

        # --------------------------------------
        # Save new scheduler values
        # --------------------------------------
        if model_date:
            sched["scheduled_date"] = model_date
        if model_time:
            sched["scheduled_time"] = model_time

        # --------------------------------------
        # HARD-GATE AUTOBOOKING
        # --------------------------------------
        ready_for_booking = (
            bool(sched.get("scheduled_date")) and
            bool(sched.get("scheduled_time")) and
            bool(sched.get("raw_address")) and
            bool(sched.get("appointment_type")) and
            not sched.get("pending_step") and
            not sched.get("booking_created")
        )

        if ready_for_booking:
            try:
                maybe_create_square_booking(phone, conv)

                if sched.get("booking_created"):
                    booking_id = sched.get("square_booking_id")
                    return {
                        "sms_body": (
                            f"You're all set — your appointment is booked for "
                            f"{sched['scheduled_date']} at {human_time} at {model_addr}. "
                            f"Your confirmation number is {booking_id}."
                        ),
                        "scheduled_date": sched["scheduled_date"],
                        "scheduled_time": sched["scheduled_time"],
                        "address": model_addr,
                        "booking_complete": True
                    }

            except Exception as e:
                print("AUTO-BOOKING ERROR:", repr(e))
                sms_body += " (Auto-booking failed internally — but you're almost set.)"

        # --------------------------------------
        # NORMAL RETURN
        # --------------------------------------
        return {
            "sms_body": sms_body,
            "scheduled_date": model_date,
            "scheduled_time": model_time,
            "address": model_addr,
            "booking_complete": False
        }

    except Exception as e:
        print("Inbound reply FAILED:", repr(e))
        return {
            "sms_body": "Sorry — can you say that again?",
            "scheduled_date": scheduled_date,
            "scheduled_time": scheduled_time,
            "address": address,
            "booking_complete": False
        }


















# ---------------------------------------------------
# Google Maps helper (travel time)
# ---------------------------------------------------
def compute_travel_time_minutes(origin: str, destination: str) -> float | None:
    """
    Uses Google Maps Distance Matrix API to estimate travel time in minutes.
    Returns None on failure.
    """
    if not GOOGLE_MAPS_API_KEY:
        return None
    if not origin or not destination:
        return None

    try:
        params = {
            "origins": origin,
            "destinations": destination,
            "key": GOOGLE_MAPS_API_KEY,
        }
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params=params,
            timeout=8,
        )
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            return None
        elements = rows[0].get("elements", [])
        if not elements:
            return None
        elem = elements[0]
        if elem.get("status") != "OK":
            return None
        seconds = elem["duration"]["value"]
        return seconds / 60.0
    except Exception as e:
        print("Travel time computation failed:", repr(e))
        return None


# ---------------------------------------------------
# Google Maps — Address Normalization (CT/MA-aware)
# 3-Layer Memory Compatible (NO direct writes to convo)
# ---------------------------------------------------
def normalize_address(raw_address: str, forced_state: str | None = None) -> tuple[str, dict | None]:
    """
    Normalize a freeform address into a Square-ready structure.

    Returns:
      ("ok", {address_struct})
      ("needs_state", None)
      ("error", None)
    """

    if not GOOGLE_MAPS_API_KEY or not raw_address:
        print("normalize_address: missing API key or raw address")
        return "error", None

    try:
        params = {
            "address": raw_address,
            "key": GOOGLE_MAPS_API_KEY,
        }

        # Constrain to US; bias if forced_state provided
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
            print("normalize_address: geocode status not OK:", status, "for", raw_address)
            return "error", None

        result = data["results"][0]
        components = result.get("address_components", [])

        line1 = None
        city = None
        state = None
        zipcode = None

        for comp in components:
            types = comp.get("types", [])
            if "street_number" in types:
                if line1 is None:
                    line1 = comp["long_name"]
            if "route" in types:
                if line1 is None:
                    line1 = comp["long_name"]
                else:
                    line1 = f"{line1} {comp['long_name']}"
            if "locality" in types:
                city = comp["long_name"]
            if "postal_town" in types and city is None:
                city = comp["long_name"]
            if "administrative_area_level_1" in types:
                state = comp["short_name"]
            if "postal_code" in types:
                zipcode = comp["long_name"]

        # Missing state → prompt CT/MA
        if not state and not forced_state:
            print("normalize_address: missing state for", raw_address)
            return "needs_state", None

        # Non-CT/MA → ask CT/MA
        if state and state not in ("CT", "MA") and not forced_state:
            print("normalize_address: geocoded state not CT/MA:", state, "for", raw_address)
            return "needs_state", None

        final_state = forced_state or state

        if not (line1 and city and final_state and zipcode):
            print("normalize_address: incomplete components",
                  "line1:", line1, "city:", city, "state:", final_state, "zip:", zipcode)
            return "error", None

        addr_struct = {
            "address_line_1": line1,
            "locality": city,
            "administrative_district_level_1": final_state,
            "postal_code": zipcode,
            "country": "US",
        }

        print("normalize_address: success for", raw_address, "->", addr_struct)
        return "ok", addr_struct

    except Exception as e:
        print("normalize_address exception:", repr(e))
        return "error", None



# ---------------------------------------------------
# Square helpers
# ---------------------------------------------------
def square_headers() -> dict:
    if not SQUARE_ACCESS_TOKEN:
        raise RuntimeError("SQUARE_ACCESS_TOKEN not configured")
    return {
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def square_create_or_get_customer(phone: str, address_struct: dict | None = None) -> str | None:
    """
    Create or return an existing Square customer.
    Stores NOTHING in scheduler state.
    Profile layer is updated only by caller, not here.
    """
    if not SQUARE_ACCESS_TOKEN:
        print("Square not configured; skipping customer create.")
        return None

    # ---------------------------------------------------
    # SEARCH BY PHONE NUMBER
    # ---------------------------------------------------
    try:
        search_payload = {
            "query": {
                "filter": {
                    "phone_number": {"exact": phone}
                }
            }
        }
        resp = requests.post(
            "https://connect.squareup.com/v2/customers/search",
            headers=square_headers(),
            json=search_payload,
            timeout=10,
        )
        data = resp.json()
        customers = data.get("customers", [])
        if customers:
            cid = customers[0]["id"]
            print("square_create_or_get_customer: found existing", cid)
            return cid
    except Exception as e:
        print("Square search customer failed:", repr(e))

    # ---------------------------------------------------
    # CREATE NEW CUSTOMER (address permitted)
    # ---------------------------------------------------
    try:
        customer_payload = {
            "idempotency_key": str(uuid.uuid4()),
            "given_name": "Prevolt Lead",
            "phone_number": phone,
        }

        if address_struct:
            customer_payload["address"] = {
                "address_line_1": address_struct.get("address_line_1"),
                "locality": address_struct.get("locality"),
                "administrative_district_level_1": address_struct.get("administrative_district_level_1"),
                "postal_code": address_struct.get("postal_code"),
                "country": address_struct.get("country", "US"),
            }

        resp = requests.post(
            "https://connect.squareup.com/v2/customers",
            headers=square_headers(),
            json=customer_payload,
            timeout=10,
        )
        resp.raise_for_status()

        data = resp.json()
        cid = data["customer"]["id"]

        print("square_create_or_get_customer: created", cid)
        return cid

    except Exception as e:
        print("Square create customer failed:", repr(e))
        return None


# ---------------------------------------------------
# REPLACED parse_local_datetime — BULLETPROOF VERSION
# ---------------------------------------------------
def parse_local_datetime(date_str: str, time_str: str) -> datetime | None:
    """
    Robust parser for Prevolt OS.
    Handles:
        - "now"
        - "any time"/"anytime"
        - "asap"
        - "tonight"
        - emergency time snapping
        - past-time correction
    Always returns a FUTURE datetime in UTC for Square.
    """

    try:
        tz = ZoneInfo("America/New_York")
    except:
        tz = timezone(timedelta(hours=-5))

    now_local = datetime.now(tz)

    if not date_str:
        print("No date_str provided to parse_local_datetime")
        return None

    normalized = (time_str or "").lower().strip()

    # ----------------------------------
    # Natural-language time interpretations
    # ----------------------------------

    # Emergency immediate scheduling
    if normalized in ("now", "asap", "right now", "immediately"):
        minute = (now_local.minute + 4) // 5 * 5
        if minute >= 60:
            # Snap to start of next hour
            now_local = now_local.replace(minute=0) + timedelta(hours=1)
        else:
            now_local = now_local.replace(minute=minute)
        t = now_local.time()

    # "anytime" — book 1 hour from now
    elif normalized in ("any time", "anytime", "whenever"):
        future = now_local + timedelta(hours=1)
        t = time(future.hour, future.minute)

    # Tonight — default to 7pm unless already past 7pm
    elif normalized == "tonight":
        tonight = now_local.replace(hour=19, minute=0, second=0, microsecond=0)
        if tonight < now_local:
            tonight = now_local + timedelta(minutes=30)
        t = tonight.time()

    # Try strict HH:MM
    else:
        try:
            t = datetime.strptime(normalized, "%H:%M").time()
        except Exception:
            print("Failed strict time parse; invalid:", time_str)
            return None

    # ----------------------------------
    # Build full local datetime
    # ----------------------------------
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        combined = datetime.combine(d, t).replace(tzinfo=tz)
    except Exception as e:
        print("Failed combining date/time:", repr(e))
        return None

    # ----------------------------------
    # Prevent appointments in the past
    # ----------------------------------
    if combined < now_local:
        print("Correcting past time to next available future slot.")
        combined = now_local + timedelta(minutes=15)
        combined = combined.replace(second=0, microsecond=0)

    # ----------------------------------
    # Convert to UTC for Square
    # ----------------------------------
    return combined.astimezone(timezone.utc).replace(tzinfo=None)


def map_appointment_type_to_variation(appointment_type: str):
    if not appointment_type:
        return None, None

    appt = appointment_type.upper().strip()

    if appt in ("EVAL_195", "EVALUATION", "EVAL"):
        return SERVICE_VARIATION_EVAL_ID, SERVICE_VARIATION_EVAL_VERSION

    if appt in ("WHOLE_HOME_INSPECTION", "INSPECTION", "HOME_INSPECTION"):
        return SERVICE_VARIATION_INSPECTION_ID, SERVICE_VARIATION_INSPECTION_VERSION

    if appt in ("TROUBLESHOOT_395", "TROUBLESHOOT", "REPAIR"):
        return SERVICE_VARIATION_TROUBLESHOOT_ID, SERVICE_VARIATION_TROUBLESHOOT_VERSION

    return None, None


# ---------------------------------------------------
# Time Window Helper — REQUIRED FOR BOOKING LOGIC
# ---------------------------------------------------
def is_within_normal_hours(time_str: str) -> bool:
    """
    Normal booking hours are 9:00–16:00 (4 PM). Input is 'HH:MM'.
    """
    try:
        dt = datetime.strptime(time_str, "%H:%M")
        return 9 <= dt.hour < 16
    except:
        return False


# ---------------------------------------------------
# Weekend Helper — REQUIRED FOR BOOKING LOGIC
# ---------------------------------------------------
def is_weekend(date_str: str) -> bool:
    """
    Returns True if the given YYYY-MM-DD date falls on a Saturday or Sunday.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday
    except:
        return False


# ---------------------------------------------------
# Create Square Booking (Option A — Hard Verification)
# ---------------------------------------------------
def maybe_create_square_booking(phone: str, convo: dict) -> None:
    """
    Creates a Square booking ONLY if all scheduler fields exist
    AND Square returns a REAL booking_id.
    """
    sched = convo.setdefault("sched", {})
    profile = convo.setdefault("profile", {})
    current_job = convo.setdefault("current_job", {})

    # Already booked?
    if sched.get("booking_created"):
        return

    # Required fields
    scheduled_date = sched.get("scheduled_date")
    scheduled_time = sched.get("scheduled_time")
    raw_address    = sched.get("raw_address")
    appointment_type = sched.get("appointment_type")

    if not (scheduled_date and scheduled_time and raw_address and appointment_type):
        return

    # Map service to Square Variation
    variation_id, variation_version = map_appointment_type_to_variation(appointment_type)
    if not variation_id:
        print("ERROR: Unknown appointment_type →", appointment_type)
        return

    # Weekend rule
    if is_weekend(scheduled_date) and appointment_type != "TROUBLESHOOT_395":
        print("Weekend blocked:", phone, scheduled_date)
        return

    # Time window rule
    if appointment_type != "TROUBLESHOOT_395":
        if not is_within_normal_hours(scheduled_time):
            print("Time outside 9–4, blocking:", phone, scheduled_time)
            return

    # Validate Square config
    if not (SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID and SQUARE_TEAM_MEMBER_ID):
        print("Square config missing; cannot create booking.")
        return

    # Normalize address if needed
    addr_struct = sched.get("normalized_address")
    if not addr_struct:
        status, addr_struct = normalize_address(raw_address)
        if status == "needs_state":
            send_sms(phone, "Just to confirm, is this address in Connecticut or Massachusetts?")
            return
        elif status != "ok":
            print("Address normalization failed:", raw_address)
            return
        sched["normalized_address"] = addr_struct

    # Resolve travel feasibility
    origin = TECH_CURRENT_ADDRESS or DISPATCH_ORIGIN_ADDRESS
    if origin:
        destination = (
            f"{addr_struct['address_line_1']}, "
            f"{addr_struct['locality']}, "
            f"{addr_struct['administrative_district_level_1']} "
            f"{addr_struct['postal_code']}"
        )
        travel_minutes = compute_travel_time_minutes(origin, destination)
        if travel_minutes and travel_minutes > MAX_TRAVEL_MINUTES:
            print("Travel too long → aborting.")
            return

    # Create or fetch Square customer
    customer_id = square_create_or_get_customer(phone, addr_struct)
    if not customer_id:
        print("Customer lookup failed.")
        return

    # Build UTC start time
    start_at_utc = parse_local_datetime(scheduled_date, scheduled_time)
    if not start_at_utc:
        print("Time conversion failed:", scheduled_date, scheduled_time)
        return

    idempotency_key = f"prevolt-{phone}-{scheduled_date}-{scheduled_time}-{appointment_type}"

    booking_payload = {
        "idempotency_key": idempotency_key,
        "booking": {
            "location_id": SQUARE_LOCATION_ID,
            "customer_id": customer_id,
            "start_at": start_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location_type": "CUSTOMER_LOCATION",
            "address": {
                "address_line_1": addr_struct["address_line_1"],
                "address_line_2": addr_struct.get("address_line_2"),
                "locality": addr_struct["locality"],
                "administrative_district_level_1": addr_struct["administrative_district_level_1"],
                "postal_code": addr_struct["postal_code"]
            },
            "appointment_segments": [
                {
                    "duration_minutes": 60,
                    "service_variation_id": variation_id,
                    "service_variation_version": variation_version,
                    "team_member_id": SQUARE_TEAM_MEMBER_ID
                }
            ],
            "customer_note": f"Auto-booked by Prevolt OS. Raw address: {raw_address}"
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
            print("Square booking failed:", resp.status_code, resp.text)
            return

        data = resp.json()
        booking = data.get("booking")
        booking_id = booking.get("id") if booking else None

        # HARD VERIFICATION — only mark created if ID exists
        if not booking_id:
            print("Square returned no booking_id — aborting success state.")
            return

        # Save booking state
        sched["booking_created"] = True
        sched["square_booking_id"] = booking_id

        print("BOOKING CREATED:", booking_id, scheduled_date, scheduled_time)

        # Save to profile
        profile["upcoming_appointment"] = {
            "date": scheduled_date,
            "time": scheduled_time,
            "type": appointment_type,
            "square_id": booking_id
        }

        # Save job history
        if current_job.get("job_type"):
            profile.setdefault("past_jobs", []).append({
                "type": current_job["job_type"],
                "date": scheduled_date
            })

    except Exception as e:
        print("Square exception:", repr(e))



# ---------------------------------------------------
# Local Dev
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
