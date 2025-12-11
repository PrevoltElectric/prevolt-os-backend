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
# NEW — Incoming SMS Webhook (Rebuilt for 3-Layer Memory)
# ---------------------------------------------------
@app.route("/incoming-sms", methods=["POST"])
def incoming_sms():
    from twilio.twiml.messaging_response import MessagingResponse

    inbound_text = request.form.get("Body", "")
    phone = request.form.get("From", "").replace("whatsapp:", "")

    # ==========================================================
    # SECRET MEMORY WIPE COMMAND (Tester Only)
    # ==========================================================
    if inbound_text.strip().lower() == "mobius1":
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
                "booking_created": False
            }
        }

        resp = MessagingResponse()
        resp.message("✔ Memory reset complete for this number.")
        return Response(str(resp), mimetype="text/xml")

    # ==========================================================
    # 3-LAYER MEMORY INITIALIZATION (NO OTHER FILES REQUIRED)
    # ==========================================================
    conv = conversations.setdefault(phone, {})

    # Create profile layer if missing
    if "profile" not in conv:
        conv["profile"] = {
            "name": None,
            "addresses": [],
            "upcoming_appointment": None,
            "past_jobs": []
        }

    # Create current job layer if missing
    if "current_job" not in conv:
        conv["current_job"] = {
            "job_type": None,
            "raw_description": None
        }

    # Create scheduler layer if missing
    if "sched" not in conv:
        conv["sched"] = {
            "pending_step": None,
            "scheduled_date": None,
            "scheduled_time": None,
            "appointment_type": None,
            "normalized_address": None,
            "booking_created": False
        }

    profile = conv["profile"]
    current_job = conv["current_job"]
    sched = conv["sched"]

    # ==========================================================
    # LEGACY VARIABLES (KEEPING THEM TO PREVENT BREAKING FLOW)
    # ==========================================================
    cleaned_transcript = conv.get("cleaned_transcript")
    category = conv.get("category")
    appointment_type = sched.get("appointment_type")
    initial_sms = conv.get("initial_sms")
    scheduled_date = sched.get("scheduled_date")
    scheduled_time = sched.get("scheduled_time")
    address = sched.get("normalized_address")

    # ==========================================================
    # SNAPSHOT BEFORE REPLY (FOR DEBUG ONLY)
    # ==========================================================
    conv["_scan_snapshot_before"] = {
        "scheduled_date": scheduled_date,
        "scheduled_time": scheduled_time,
        "address": address,
        "appointment_type": appointment_type,
        "category": category
    }

    # ==========================================================
    # GENERATE AI REPLY (NO STRUCTURE CHANGE TO DOWNSTREAM CODE)
    # ==========================================================
    reply = generate_reply_for_inbound(
        cleaned_transcript,
        category,
        appointment_type,
        initial_sms,
        inbound_text,
        scheduled_date,
        scheduled_time,
        address,
    )

    # ==========================================================
    # APPLY REPLY FIELDS TO **SCHEDULER LAYER ONLY**
    # ==========================================================
    sched["scheduled_date"] = reply.get("scheduled_date")
    sched["scheduled_time"] = reply.get("scheduled_time")
    sched["normalized_address"] = reply.get("address")

    # ==========================================================
    # IF BOOKING COMPLETED (DETECTED BY LLM), MOVE INTO PROFILE
    # ==========================================================
    if reply.get("booking_complete"):

        # Store upcoming appointment in profile memory
        profile["upcoming_appointment"] = {
            "date": sched.get("scheduled_date"),
            "time": sched.get("scheduled_time"),
            "type": sched.get("appointment_type")
        }

        # Add to job history if job exists
        if current_job.get("job_type"):
            profile["past_jobs"].append({
                "type": current_job["job_type"],
                "date": sched.get("scheduled_date")
            })

        # FULL scheduler reset (core fix)
        conv["sched"] = {
            "pending_step": None,
            "scheduled_date": None,
            "scheduled_time": None,
            "appointment_type": None,
            "normalized_address": None,
            "booking_created": True
        }

    # ==========================================================
    # SNAPSHOT AFTER APPLYING REPLY
    # ==========================================================
    conv["_scan_snapshot_after"] = {
        "scheduled_date": conv["sched"]["scheduled_date"],
        "scheduled_time": conv["sched"]["scheduled_time"],
        "address": conv["sched"]["normalized_address"],
    }

    # ==========================================================
    # RETURN TWILIO REPLY
    # ==========================================================
    twilio_reply = MessagingResponse()
    twilio_reply.message(reply["sms_body"])

    return Response(str(twilio_reply), mimetype="text/xml")




# ---------------------------------------------------
# REQUIRED IMPORTS FOR STEP 4
# ---------------------------------------------------
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import json


# ---------------------------------------------------
# STATE HELPERS (MINIMAL SAFE VERSIONS)
# ---------------------------------------------------
def get_current_state(conv: dict) -> str:

    if conv.get("scheduled_date") and conv.get("scheduled_time") and conv.get("address"):
        return "ready_for_confirmation"

    if not conv.get("address"):
        return "awaiting_address"

    if not conv.get("scheduled_date") and not conv.get("scheduled_time"):
        return "awaiting_date_or_time"

    if conv.get("scheduled_date") and not conv.get("scheduled_time"):
        return "awaiting_time"

    if conv.get("scheduled_time") and not conv.get("scheduled_date"):
        return "awaiting_date"

    return "unknown"


def enforce_state_lock(state, conv, inbound_lower, address, scheduled_date, scheduled_time):
    return {}



# ---------------------------------------------------
# GLOBAL RULES CACHE (REQUIRED FOR SYSTEM PROMPT LOADER)
# ---------------------------------------------------
PREVOLT_RULES_CACHE = None



# ---------------------------------------------------
# Build System Prompt (Prevolt Rules Engine) — RESTORED
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

    # Load once
    if PREVOLT_RULES_CACHE is None:
        with open("prevolt_rules.json", "r", encoding="utf-8") as f:
            PREVOLT_RULES_CACHE = json.load(f)

    rules_text = PREVOLT_RULES_CACHE.get("rules", "")

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

    output_block = (
        "{\n"
        '  "sms_body": "string",\n'
        '  "scheduled_date": "YYYY-MM-DD or null",\n'
        '  "scheduled_time": "HH:MM or null",\n'
        '  "address": "string or null"\n'
        "}"
    )

    system_prompt = (
        "You are Prevolt OS, the SMS assistant for Prevolt Electric.\n"
        "You MUST respond ONLY in strict JSON.\n\n"
        f"Today is {today_date_str}, a {today_weekday}.\n\n"
        f"{rules_text}"
        f"{voicemail_context}\n\n"

        "===================================================\n"
        "STATE HANDLING RULES\n"
        "===================================================\n"
        "• NEVER ask again for information the customer already provided.\n"
        "• ALWAYS inherit previously known values.\n"
        "• NEVER output null if the value is already known.\n\n"

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
# Step 4 — Generate Replies (THE BRAIN) + AUTO-BOOKING
# COMPLETE + RESTORED + PATCH 3 APPLIED
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
        except:
            tz = timezone(timedelta(hours=-5))

        now_local      = datetime.now(tz)
        today_date_str = now_local.strftime("%Y-%m-%d")
        today_weekday  = now_local.strftime("%A")

        # --------------------------------------
        # Load conversation
        # --------------------------------------
        phone = request.form.get("From", "").replace("whatsapp:", "")
        conv  = conversations.setdefault(phone, {})

        conv.setdefault("profile", {})
        conv.setdefault("current_job", {})
        conv.setdefault("sched", {})
        sched = conv["sched"]

        inbound_lower = inbound_text.lower()

        # -------------------------------------------------
        # Ensure raw address bucket exists
        # -------------------------------------------------
        sched.setdefault("raw_address", None)

        # --------------------------------------
        # Appointment type fallback
        # --------------------------------------
        appt_type = sched.get("appointment_type")
        if not appt_type:
            if any(word in inbound_lower for word in [
                "not working", "no power", "dead", "sparking", "burning",
                "breaker keeps", "gfci", "outlet not", "troubleshoot"
            ]):
                appt_type = "TROUBLESHOOT_395"
            elif any(word in inbound_lower for word in [
                "inspection", "whole home inspection", "electrical inspection"
            ]):
                appt_type = "WHOLE_HOME_INSPECTION"
            else:
                appt_type = "EVAL_195"
            sched["appointment_type"] = appt_type

        # --------------------------------------
        # HARD ADDRESS CAPTURE — raw only
        # --------------------------------------
        address_markers = [
            "st", "street", "ave", "avenue", "rd", "road",
            "ln", "lane", "dr", "drive", "ct", "circle",
            "blvd", "way"
        ]

        if any(marker in inbound_lower for marker in address_markers) and len(inbound_text) > 6:
            sched["raw_address"] = inbound_text.strip()
            address = sched["raw_address"]

        # --------------------------------------
        # Build LLM System Prompt
        # --------------------------------------
        system_prompt = build_system_prompt(
            cleaned_transcript,
            category,
            appt_type,
            initial_sms,
            sched.get("scheduled_date"),
            sched.get("scheduled_time"),
            sched.get("raw_address"),
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
                {"role": "user",   "content": inbound_text},
            ],
        )

        raw_json = completion.choices[0].message.content.strip()
        raw_json = raw_json.replace("None", "null").replace("none", "null")
        ai_raw = json.loads(raw_json)

        sms_body   = ai_raw.get("sms_body", "").strip()
        model_date = ai_raw.get("scheduled_date")
        model_time = ai_raw.get("scheduled_time")
        model_addr = ai_raw.get("address")

        # -------------------------------------------------
        # PATCH 1 — DATE/TIME LOCK
        # Ensures LLM cannot remove known date/time
        # -------------------------------------------------
        if sched.get("scheduled_date") and not model_date:
            model_date = sched["scheduled_date"]

        if sched.get("scheduled_time") and not model_time:
            model_time = sched["scheduled_time"]

        # -------------------------------------------------
        # PATCH 3 — Always capture raw address
        # -------------------------------------------------
        if isinstance(model_addr, str) and len(model_addr) > 5:
            sched["raw_address"] = model_addr

        # -------------------------------------------------
        # SYNC LOCK — Cannot erase known values
        # -------------------------------------------------
        if not model_date and scheduled_date:
            model_date = scheduled_date

        if not model_time and scheduled_time:
            model_time = scheduled_time

        if not model_addr and address:
            model_addr = address

        # State cannot regress
        if sched.get("scheduled_date") and not model_date:
            model_date = sched["scheduled_date"]

        if sched.get("scheduled_time") and not model_time:
            model_time = sched["scheduled_time"]

        if sched.get("raw_address") and not model_addr:
            model_addr = sched["raw_address"]

        # -------------------------------------------------
        # RAW ADDRESS FINALIZATION
        # -------------------------------------------------
        if isinstance(model_addr, str):
            sched["raw_address"] = model_addr

        # -------------------------------------------------
        # PRICE INJECTION
        # -------------------------------------------------
        price_map = {
            "eval_195": " The visit is a $195 consultation.",
            "troubleshoot_395": " The visit is a $395 troubleshoot and repair.",
            "whole_home_inspection": " Home inspections range from $375–$650 depending on size."
        }
        phrase = price_map.get(appt_type.lower(), "")
        if phrase and phrase not in sms_body:
            sms_body += phrase

        # -------------------------------------------------
        # TIME FORMATTING (12-hour)
        # -------------------------------------------------
        try:
            human_time = datetime.strptime(model_time, "%H:%M").strftime("%-I:%M %p") \
                if model_time else None
        except:
            human_time = model_time

        final_sms = (
            sms_body.replace(model_time, human_time)
            if (sms_body and model_time and human_time)
            else sms_body
        )

        # -------------------------------------------------
        # PATCH 2 — SAVE DATE/TIME BACK INTO STATE
        # -------------------------------------------------
        if model_date:
            sched["scheduled_date"] = model_date

        if model_time:
            sched["scheduled_time"] = model_time

        # -------------------------------------------------
        # AUTO-BOOKING CHECK
        # -------------------------------------------------
        ready_for_booking = (
            bool(model_date)
            and bool(model_time)
            and bool(sched.get("raw_address"))
            and not sched.get("booking_created")
        )

        if ready_for_booking:
            sched["scheduled_date"] = model_date
            sched["scheduled_time"] = model_time
            sched["normalized_address"] = None
            sched["raw_address"] = model_addr

            try:
                maybe_create_square_booking(phone, conv)

                if sched.get("booking_created"):
                    booking_id = sched.get("square_booking_id")

                    return {
                        "sms_body": (
                            f"You're all set — your appointment is booked for {model_date} "
                            f"at {human_time} at {model_addr}. "
                            f"Your confirmation number is {booking_id}."
                        ),
                        "scheduled_date": model_date,
                        "scheduled_time": model_time,
                        "address": model_addr,
                        "booking_complete": True
                    }

            except Exception as e:
                print("AUTO-BOOKING ERROR:", repr(e))
                final_sms += (
                    " (We couldn't auto-book, but you're almost set — we'll confirm manually.)"
                )

        # -------------------------------------------------
        # NORMAL RETURN
        # -------------------------------------------------
        return {
            "sms_body": final_sms,
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
# Create Square Booking (3-Layer Memory Compatible)
# ---------------------------------------------------
def maybe_create_square_booking(phone: str, convo: dict) -> None:
    """
    Square bookings created only when scheduler layer has full info.
    """
    sched = convo.setdefault("sched", {})
    profile = convo.setdefault("profile", {})
    current_job = convo.setdefault("current_job", {})

    if sched.get("booking_created"):
        return

    scheduled_date = sched.get("scheduled_date")
    scheduled_time = sched.get("scheduled_time")
    raw_address    = sched.get("normalized_address")
    appointment_type = sched.get("appointment_type")

    if not (scheduled_date and scheduled_time and raw_address):
        return

    variation_id, variation_version = map_appointment_type_to_variation(appointment_type)
    if not variation_id:
        print("Unknown appointment_type; cannot map:", appointment_type)
        return

    # Weekend rule
    if is_weekend(scheduled_date) and appointment_type != "TROUBLESHOOT_395":
        print("Weekend non-emergency booking blocked:", phone, scheduled_date)
        return

    # Time window rule
    if appointment_type != "TROUBLESHOOT_395" and not is_within_normal_hours(scheduled_time):
        print("Non-emergency time outside 9–4; booking not auto-created:", phone, scheduled_time)
        return

    if not (SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID and SQUARE_TEAM_MEMBER_ID):
        print("Square configuration incomplete; skipping booking creation.")
        return

    # Normalize or reuse
    addr_struct = sched.get("normalized_address")
    if not addr_struct:
        status, addr_struct = normalize_address(raw_address)
        if status == "ok":
            sched["normalized_address"] = addr_struct
        elif status == "needs_state":
            if not sched.get("state_prompt_sent"):
                send_sms(phone, "Just to confirm, is this address in Connecticut or Massachusetts?")
                sched["state_prompt_sent"] = True
            print("Address needs CT/MA confirmation:", raw_address)
            return
        else:
            print("Address normalization failed:", raw_address)
            return

    # Travel time
    origin = TECH_CURRENT_ADDRESS or DISPATCH_ORIGIN_ADDRESS
    if origin:
        destination_for_travel = (
            f"{addr_struct['address_line_1']}, "
            f"{addr_struct['locality']}, "
            f"{addr_struct['administrative_district_level_1']} "
            f"{addr_struct['postal_code']}"
        )
        travel_minutes = compute_travel_time_minutes(origin, destination_for_travel)
        if travel_minutes is not None:
            print(
                f"Estimated travel: {travel_minutes:.1f} minutes for {phone} "
                f"→ {destination_for_travel}"
            )
            if travel_minutes > MAX_TRAVEL_MINUTES:
                print("Travel exceeds limit; skipping booking.")
                return

    # Customer ID
    customer_id = square_create_or_get_customer(phone, addr_struct)
    if not customer_id:
        print("No customer_id; cannot create booking.")
        return

    start_at_utc = parse_local_datetime(scheduled_date, scheduled_time)
    if not start_at_utc:
        print("Time parse failed; skipping booking.")
        return

    idempotency_key = f"prevolt-{phone}-{scheduled_date}-{scheduled_time}-{appointment_type}"

    booking_address = {
        "address_line_1": addr_struct["address_line_1"],
        "locality": addr_struct["locality"],
        "administrative_district_level_1": addr_struct["administrative_district_level_1"],
        "postal_code": addr_struct["postal_code"],
    }
    if addr_struct.get("address_line_2"):
        booking_address["address_line_2"] = addr_struct["address_line_2"]

    booking_payload = {
        "idempotency_key": idempotency_key,
        "booking": {
            "location_id": SQUARE_LOCATION_ID,
            "customer_id": customer_id,
            "start_at": start_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location_type": "CUSTOMER_LOCATION",
            "address": booking_address,
            "customer_note": f"Auto-booked by Prevolt OS. Raw address: {raw_address}",
            "appointment_segments": [
                {
                    "duration_minutes": 60,
                    "service_variation_id": variation_id,
                    "service_variation_version": variation_version,
                    "team_member_id": SQUARE_TEAM_MEMBER_ID,
                }
            ],
        },
    }

    try:
        resp = requests.post(
            "https://connect.squareup.com/v2/bookings",
            headers=square_headers(),
            json=booking_payload,
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            print("Square booking create failed:", resp.status_code, resp.text)
            return

        data = resp.json()
        booking = data.get("booking")
        booking_id = booking.get("id") if booking else None

        # Update scheduler layer
        sched["booking_created"] = True
        sched["square_booking_id"] = booking_id

        print(
            f"Square booking created for {phone}: {booking_id} "
            f"{scheduled_date} {scheduled_time} ({appointment_type})"
        )

        # ---------------------------------------------------
        # Save upcoming appointment into PROFILE (not scheduler)
        # ---------------------------------------------------
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

        # Scheduler will be wiped by incoming_sms() after booking_complete return

    except Exception as e:
        print("Square booking exception:", repr(e))


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
        method="POST"
    )

    # FULL SSML — Slowed down + pauses + Matthew voice
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

    # No input → replay menu
    response.say(
        '<speak><prosody rate="95%">Sorry, I did not get that. Let me repeat the options.</prosody></speak>',
        voice="Polly.Matthew-Neural"
    )
    response.redirect("/incoming-call")

    return Response(str(response), mimetype="text/xml")


# ---------------------------------------------------
# Handle Residential vs Commercial
# ---------------------------------------------------
@app.route("/handle-call-selection", methods=["POST"])
def handle_call_selection():
    from twilio.twiml.voice_response import VoiceResponse

    digit = request.form.get("Digits", "")
    response = VoiceResponse()

    # -----------------------------
    # OPTION 1 → RESIDENTIAL FLOW
    # -----------------------------
    if digit == "1":
        response.say(
            '<speak>'
                '<prosody rate="95%">'
                    'Welcome to PREE-volt Electric’s premium residential service desk.<break time="0.7s"/>'
                    'You’ll leave a quick message, and our team will text you right away to assist.<break time="0.8s"/>'
                    'Please leave your name,<break time="0.4s"/> your address,<break time="0.4s"/> '
                    'and a brief description of what you need help with.<break time="0.6s"/>'
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
        )

        response.hangup()
        return Response(str(response), mimetype="text/xml")

    # -----------------------------
    # OPTION 2 → COMMERCIAL / GOVERNMENT ROUTING
    # -----------------------------
    elif digit == "2":
        response.say(
            '<speak><prosody rate="90%">Connecting you now.</prosody></speak>',
            voice="Polly.Matthew-Neural"
        )
        response.dial("+15555555555")  # Replace with your real number
        return Response(str(response), mimetype="text/xml")

    # -----------------------------
    # INVALID INPUT → Replay Menu
    # -----------------------------
    else:
        response.say(
            '<speak><prosody rate="90%">Sorry, I didn’t understand that.</prosody></speak>',
            voice="Polly.Matthew-Neural"
        )
        response.redirect("/incoming-call")
        return Response(str(response), mimetype="text/xml")


# ---------------------------------------------------
# Voice → Voicemail Completion (FULL INTELLIGENT PIPELINE)
# ---------------------------------------------------
@app.route("/voicemail-complete", methods=["POST"])
def voicemail_complete():
    from twilio.twiml.voice_response import VoiceResponse

    recording_url = request.form.get("RecordingUrl", "")
    from_number = request.form.get("From", "").replace("whatsapp:", "")

    # ---------------------------------------------------
    # 1. Retrieve or initialize conversation state
    # ---------------------------------------------------
    conv = conversations.get(from_number, {})

    new_conv = {
        "voicemail_url": recording_url,

        # Preserve existing values
        "initial_sms": conv.get("initial_sms"),
        "cleaned_transcript": conv.get("cleaned_transcript"),
        "category": conv.get("category"),
        "appointment_type": conv.get("appointment_type"),
        "first_sms_time": time.time(),
        "replied": conv.get("replied", False),
        "followup_sent": conv.get("followup_sent", False),
        "scheduled_date": conv.get("scheduled_date"),
        "scheduled_time": conv.get("scheduled_time"),
        "address": conv.get("address"),
        "normalized_address": conv.get("normalized_address"),
        "booking_created": conv.get("booking_created", False),
        "square_booking_id": conv.get("square_booking_id"),
        "state_prompt_sent": conv.get("state_prompt_sent", False),

        # NEW extracted values
        "voicemail_intent": conv.get("voicemail_intent"),
        "voicemail_town": conv.get("voicemail_town"),
        "voicemail_partial_address": conv.get("voicemail_partial_address"),
    }

    conversations[from_number] = new_conv

    # ---------------------------------------------------
    # 2. TRANSCRIBE THE VOICEMAIL
    # ---------------------------------------------------
    try:
        raw_transcript = transcribe_recording(recording_url)
    except Exception as e:
        print("Whisper transcription failed:", repr(e))
        raw_transcript = ""

    if not raw_transcript:
        raw_transcript = ""

    # ---------------------------------------------------
    # 3. CLEAN IT (LLM minor correction)
    # ---------------------------------------------------
    try:
        cleaned = clean_transcript_text(raw_transcript)
    except Exception as e:
        print("Transcript cleanup failed:", repr(e))
        cleaned = raw_transcript

    new_conv["cleaned_transcript"] = cleaned

    # ---------------------------------------------------
    # 4. EXTRACT: intent, town, partial street
    # ---------------------------------------------------
    try:
        extract = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract EXACTLY THREE fields from this voicemail:\n"
                        "1. intent: short phrase of what work they want\n"
                        "2. town: CT/MA town mentioned\n"
                        "3. partial_address: any street number/name\n\n"
                        "If unknown, return null.\n"
                        "Return STRICT JSON: {intent, town, partial_address}"
                    )
                },
                {"role": "user", "content": cleaned}
            ]
        )
        extracted = json.loads(extract.choices[0].message.content)
    except Exception as e:
        print("Extractor failed:", repr(e))
        extracted = {"intent": None, "town": None, "partial_address": None}

    # Store extraction values (only if empty)
    if new_conv.get("voicemail_intent") is None:
        new_conv["voicemail_intent"] = extracted.get("intent")

    if new_conv.get("voicemail_town") is None:
        new_conv["voicemail_town"] = extracted.get("town")

    if new_conv.get("voicemail_partial_address") is None:
        new_conv["voicemail_partial_address"] = extracted.get("partial_address")

    # ---------------------------------------------------
    # 4B. *** NEW FIX *** — AUTO-STORE ADDRESS FROM VOICEMAIL
    # ---------------------------------------------------
    partial = new_conv.get("voicemail_partial_address")
    town = new_conv.get("voicemail_town")

    # Case 1 — full street + town present → store as full address
    if partial and town and not new_conv.get("address"):
        new_conv["address"] = f"{partial} {town}"

    # Case 2 — full address already stored → nothing to override
    # Case 3 — partial exists but no town → OS later asks: "What town is that in?"
    # Case 4 — town exists but no street → OS asks for street name/number

    # ---------------------------------------------------
    # 5. CLASSIFY + BUILD INITIAL SMS BODY
    # ---------------------------------------------------
    try:
        initial = generate_initial_sms(cleaned)
        sms_body = initial.get("sms_body", "")
        category = initial.get("category")
        appt_type = initial.get("appointment_type")
    except Exception as e:
        print("Initial SMS classification failed:", repr(e))
        sms_body = "Hi, this is Prevolt Electric — we received your message. What day works for you?"
        category = "OTHER"
        appt_type = "EVAL_195"

    new_conv["initial_sms"] = sms_body
    new_conv["category"] = category
    new_conv["appointment_type"] = appt_type

    # ---------------------------------------------------
    # 6. BUILD INTELLIGENT FIRST SMS (Address-Aware)
    # ---------------------------------------------------
    intent = new_conv.get("voicemail_intent")
    full_address = new_conv.get("address")

    msg = "Hi, this is Prevolt Electric — "

    if intent:
        msg += f"I saw your message about {intent}. "

    if town:
        msg += f"Looks like you're in {town}. "

    # *** DO NOT ASK FOR ADDRESS AGAIN IF WE ALREADY HAVE ONE ***
    if full_address:
        msg += f"Thanks for confirming your address ({full_address}). "
        msg += "What day works for the visit?"
    else:
        # Ask only for missing pieces
        if partial and not town:
            msg += f"I caught the street ({partial}). What town is that in?"
        elif town and not partial:
            msg += f"I heard {town}. What’s the street address?"
        else:
            # No usable address extracted
            msg += "What’s the full street address for the visit?"

    # ---------------------------------------------------
    # 7. SEND FIRST SMS
    # ---------------------------------------------------
    try:
        send_sms(from_number, msg)
    except Exception as e:
        print("Error sending SMS:", repr(e))

    # ---------------------------------------------------
    # 8. Voice confirmation to caller
    # ---------------------------------------------------
    response = VoiceResponse()
    response.say(
        '<speak><prosody rate="90%">Thanks, we received your message.</prosody></speak>',
        voice="Polly.Matthew-Neural"
    )
    response.hangup()
    return str(response)



# ---------------------------------------------------
# Local Dev
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
