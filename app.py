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
# Step 3 — Generate Initial SMS (Ultra-Deterministic Classifier)
# ---------------------------------------------------
def generate_initial_sms(cleaned_text: str) -> dict:
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Prevolt OS, the SMS assistant for Prevolt Electric.\n"
                        "Your job is to classify the voicemail into EXACTLY ONE appointment type.\n\n"
                        "APPOINTMENT TYPE RULES (DO NOT GUESS):\n"
                        "1. TROUBLESHOOT_395 = ANY active problem, urgent issue, outage, burning smell, fire, "
                        "   wires down, tree damage, main wires pulled, breaker tripping, no power, partial power, "
                        "   anything dangerous, anything affecting existing equipment.\n"
                        "2. WHOLE_HOME_INSPECTION = ONLY when caller explicitly says: "
                        "   'inspection', 'whole home inspection', 'electrical inspection', "
                        "   or requests a safety inspection for insurance or buying a home.\n"
                        "3. EVAL_195 = Quotes, upgrades, installs, adding circuits, panel upgrades, EV chargers, "
                        "   generator installs, renovations, pricing requests, or non-emergency consultations.\n\n"
                        "NEVER misclassify outages as inspections or evals. When in doubt between eval and troubleshoot → choose TROUBLESHOOT.\n"
                        "NEVER soften or second-guess. Use hard classification logic ONLY based on the voicemail.\n\n"
                        "SMS RULES:\n"
                        "• Must begin with: 'Hi, this is Prevolt Electric —'\n"
                        "• Mention the price ONCE.\n"
                        "• No small talk, no emojis, no fluff.\n"
                        "• Do NOT ask them to repeat the voicemail.\n\n"
                        "Return STRICT JSON: {sms_body, category, appointment_type}."
                    ),
                },
                {"role": "user", "content": cleaned_text},
            ],
        )

        data = json.loads(completion.choices[0].message.content)

        return {
            "sms_body": data["sms_body"].strip(),
            "category": data["category"],
            "appointment_type": data["appointment_type"],
        }

    except Exception as e:
        print("Initial SMS FAILED:", repr(e))
        return {
            "sms_body": (
                "Hi, this is Prevolt Electric — I received your message. "
                "What day works for a visit?"
            ),
            "category": "OTHER",
            "appointment_type": "EVAL_195",
        }



# ---------------------------------------------------
# NEW — Incoming SMS Webhook (Enables ALL Scheduling Logic)
# SCANNING METADATA ATTACHED — SAFE, NON-EXECUTING
# ---------------------------------------------------
@app.route("/incoming-sms", methods=["POST"])
def incoming_sms():
    """
    SCAN_TARGET: incoming_sms
    SCAN_FIELDS:
        - scheduled_date
        - scheduled_time
        - address
        - normalized_address
        - appointment_type
        - category
    SCAN_NOTES:
        The scanner should analyze how state progresses AFTER the reply
        and confirm nothing is overwritten incorrectly.
    """
    from twilio.twiml.messaging_response import MessagingResponse

    inbound_text = request.form.get("Body", "")
    phone = request.form.get("From", "").replace("whatsapp:", "")

    # Pull conversation state
    conv = conversations.setdefault(phone, {})

    cleaned_transcript = conv.get("cleaned_transcript")
    category = conv.get("category")
    appointment_type = conv.get("appointment_type")
    initial_sms = conv.get("initial_sms")
    scheduled_date = conv.get("scheduled_date")
    scheduled_time = conv.get("scheduled_time")
    address = conv.get("address")

    # ---------------------------------------------------
    # SCANNER_HOOK: PRE-REPLY STATE SNAPSHOT
    # ---------------------------------------------------
    conv["_scan_snapshot_before"] = {
        "scheduled_date": scheduled_date,
        "scheduled_time": scheduled_time,
        "address": address,
        "appointment_type": appointment_type,
        "category": category,
    }

    # ---------------------------------------------------
    # Generate AI reply (Step 4)
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # SAVE UPDATED STATE
    # ---------------------------------------------------
    conv["scheduled_date"] = reply.get("scheduled_date")
    conv["scheduled_time"] = reply.get("scheduled_time")
    conv["address"] = reply.get("address")

    # ---------------------------------------------------
    # SCANNER_HOOK: POST-REPLY STATE SNAPSHOT
    # ---------------------------------------------------
    conv["_scan_snapshot_after"] = {
        "scheduled_date": conv["scheduled_date"],
        "scheduled_time": conv["scheduled_time"],
        "address": conv["address"],
    }

    # ---------------------------------------------------
    # Build Twilio reply
    # ---------------------------------------------------
    twilio_reply = MessagingResponse()
    twilio_reply.message(reply["sms_body"])

    return Response(str(twilio_reply), mimetype="text/xml")




# ---------------------------------------------------
# Load + Build System Prompt (Prevolt Rules Engine)
# ---------------------------------------------------

import json

# Cache rules so we do not re-read the file every SMS
PREVOLT_RULES_CACHE = None


# ---------------------------------------------------
# Build System Prompt (loads JSON rules dynamically)
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

    # Load JSON rules once
    if PREVOLT_RULES_CACHE is None:
        with open("prevolt_rules.json", "r", encoding="utf-8") as f:
            PREVOLT_RULES_CACHE = json.load(f)

    rules_text = PREVOLT_RULES_CACHE.get("rules", "")

    # Optional voicemail context
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

    # SYSTEM PROMPT WITH STABILITY RULES
    system_prompt = (
        f"You are Prevolt OS, the SMS assistant for Prevolt Electric.\n"
        f"Continue the conversation naturally. Do NOT repeat yourself.\n\n"
        f"Today is {today_date_str}, a {today_weekday}, local time America/New_York.\n\n"
        f"{rules_text}"
        f"{voicemail_context}\n\n"

        "===================================================\n"
        "STATE HANDLING RULES\n"
        "===================================================\n"
        "• NEVER ask again for information the customer already provided.\n"
        "• If customer provides ONLY a time → use the previously stored date.\n"
        "• If customer provides ONLY a date → use the previously stored time.\n"
        "• If customer provides an address → NEVER ask for address again.\n"
        "• ALWAYS inherit previously known values unless customer changes them.\n"
        "• NEVER return null for a field already known.\n"
        "• Avoid phrases like 'Got it —'. Vary your confirmations.\n"
        "• Include correct pricing once ONLY based on appointment type.\n\n"

        "===================================================\n"
        "CONTEXT\n"
        "===================================================\n"
        f"Original voicemail: {cleaned_transcript}\n"
        f"Category: {category}\n"
        f"Appointment type: {appointment_type}\n"
        f"Initial SMS: {initial_sms}\n"
        f"Stored date/time/address: {scheduled_date}, {scheduled_time}, {address}\n\n"

        "===================================================\n"
        "OUTPUT FORMAT (STRICT JSON)\n"
        "===================================================\n"
        "{\n"
        '  "sms_body": "...",\n'
        '  "scheduled_date": "YYYY-MM-DD or null",\n'
        '  "scheduled_time": "HH:MM or null",\n'
        '  "address": "string or null"\n'
        "}\n"
    )

    return system_prompt



# ---------------------------------------------------
# REQUIRED IMPORTS FOR STEP 4
# ---------------------------------------------------
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


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
# SQUARE BOOKING CANCELLATION
# ---------------------------------------------------
def cancel_square_booking(booking_id: str):
    """Cancel/delete a Square booking."""
    try:
        resp = requests.delete(
            f"https://connect.squareup.com/v2/bookings/{booking_id}",
            headers=square_headers(),
            timeout=10,
        )
        if resp.status_code in (200, 204):
            print("Square booking cancelled:", booking_id)
            return True

        print("Square cancel failed:", resp.status_code, resp.text)
        return False

    except Exception as e:
        print("Square cancel exception:", repr(e))
        return False


# ---------------------------------------------------
# SQUARE BOOKING ADDRESS UPDATE (NEW)
# ---------------------------------------------------
def update_square_booking_address(booking_id: str, new_address: str):
    """
    Update the booking's address using Square PATCH.
    Booking 'address' CANNOT include a country field.
    """
    try:
        # Attempt to re-normalize
        status, addr = normalize_address(new_address)
        if status != "ok":
            print("Address normalize failed for update:", new_address)
            return False

        booking_address = {
            "address_line_1": addr["address_line_1"],
            "locality": addr["locality"],
            "administrative_district_level_1": addr["administrative_district_level_1"],
            "postal_code": addr["postal_code"],
        }
        if addr.get("address_line_2"):
            booking_address["address_line_2"] = addr["address_line_2"]

        payload = {
            "booking": {
                "location_type": "CUSTOMER_LOCATION",
                "address": booking_address,
            }
        }

        resp = requests.patch(
            f"https://connect.squareup.com/v2/bookings/{booking_id}",
            headers=square_headers(),
            json=payload,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            print("Square address updated:", booking_id)
            return True

        print("Square address update failed:", resp.status_code, resp.text)
        return False

    except Exception as e:
        print("Square address update exception:", repr(e))
        return False


# ---------------------------------------------------
# SQUARE BOOKING NOTES UPDATE (NEW)
# ---------------------------------------------------
def update_square_booking_notes(booking_id: str, append_text: str):
    """
    Adds text to the existing appointment notes.
    """
    try:
        # Fetch existing booking (Square requires full booking object updates sometimes)
        resp = requests.get(
            f"https://connect.squareup.com/v2/bookings/{booking_id}",
            headers=square_headers(),
            timeout=10,
        )

        if resp.status_code not in (200, 201):
            print("Square fetch failed for notes:", resp.status_code, resp.text)
            return False

        data = resp.json().get("booking", {})
        old_note = data.get("customer_note", "") or ""
        new_note = (old_note + " | " + append_text).strip()

        payload = {
            "booking": {
                "customer_note": new_note
            }
        }

        resp2 = requests.patch(
            f"https://connect.squareup.com/v2/bookings/{booking_id}",
            headers=square_headers(),
            json=payload,
            timeout=10,
        )

        if resp2.status_code in (200, 201):
            print("Square notes updated:", booking_id)
            return True

        print("Square note update failed:", resp2.status_code, resp2.text)
        return False

    except Exception as e:
        print("Square notes update exception:", repr(e))
        return False


# ---------------------------------------------------
# Step 4 — Generate Replies (THE BRAIN) + AUTO-BOOKING + HANDLERS
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
        # -------------------------------
        # Timezone setup
        # -------------------------------
        try:
            tz = ZoneInfo("America/New_York")
        except:
            tz = timezone(timedelta(hours=-5))

        now_local      = datetime.now(tz)
        today_date_str = now_local.strftime("%Y-%m-%d")
        today_weekday  = now_local.strftime("%A")

        # -------------------------------
        # Load conversation
        # -------------------------------
        phone = request.form.get("From", "").replace("whatsapp:", "")
        conv  = conversations.setdefault(phone, {})
        inbound_lower = inbound_text.lower()

        booking_id = conv.get("square_booking_id")

        # =====================================================
        # 1️⃣ CANCELATION HANDLER
        # =====================================================
        cancel_keywords = [
            "cancel", "cancel appointment", "cancel it",
            "i can't make it", "need to cancel",
            "please cancel", "don't come"
        ]

        if booking_id and any(k in inbound_lower for k in cancel_keywords):
            success = cancel_square_booking(booking_id)

            conversations[phone] = {
                "cleaned_transcript": None,
                "category": None,
                "appointment_type": None,
                "initial_sms": None,
                "first_sms_time": None,
                "replied": False,
                "followup_sent": False,
                "scheduled_date": None,
                "scheduled_time": None,
                "address": None,
                "normalized_address": None,
                "booking_created": False,
                "square_booking_id": None,
                "state_prompt_sent": False,
            }

            if success:
                return {
                    "sms_body": "Your appointment has been cancelled. If you'd like to reschedule, just let me know.",
                    "scheduled_date": None,
                    "scheduled_time": None,
                    "address": None,
                }
            return {
                "sms_body": "I wasn't able to cancel it in the system, but I flagged it for manual cancellation. Let me know if you'd like to reschedule.",
                "scheduled_date": None,
                "scheduled_time": None,
                "address": None,
            }

        # =====================================================
        # 2️⃣ ETA HANDLER
        # =====================================================
        eta_keywords = [
            "eta", "how long", "how far", "on your way",
            "when will you", "are you coming", "when are you"
        ]

        if booking_id and any(k in inbound_lower for k in eta_keywords):
            appt_date = conv.get("scheduled_date")
            appt_time = conv.get("scheduled_time")

            if appt_date and appt_time:
                human_time = datetime.strptime(appt_time, "%H:%M").strftime("%-I:%M %p")
                return {
                    "sms_body": f"Your visit is scheduled for {appt_date} at {human_time}. We'll send an arrival text when we're en route.",
                    "scheduled_date": appt_date,
                    "scheduled_time": appt_time,
                    "address": conv.get("address"),
                }

        # =====================================================
        # 3️⃣ RESCHEDULE HANDLER
        # =====================================================
        res_keywords = ["reschedule", "different time", "another time", "can we move"]

        if booking_id and any(k in inbound_lower for k in res_keywords):
            cancel_square_booking(booking_id)

            conv["booking_created"] = False
            conv["square_booking_id"] = None
            conv["scheduled_time"] = None
            conv["scheduled_date"] = None

            return {
                "sms_body": "No problem — what day works better for you?",
                "scheduled_date": None,
                "scheduled_time": None,
                "address": conv.get("address"),
            }

        # =====================================================
        # 4️⃣ ADDRESS CHANGE HANDLER
        # =====================================================
        if booking_id and any(x in inbound_lower for x in [" st", " ave", " rd", " road", " ln", " lane", "dr", "drive"]):
            conv["address"] = inbound_text.strip()
            success = update_square_booking_address(booking_id, conv["address"])

            if success:
                return {
                    "sms_body": f"Updated your appointment address to: {conv['address']}.",
                    "scheduled_date": conv.get("scheduled_date"),
                    "scheduled_time": conv.get("scheduled_time"),
                    "address": conv["address"],
                }
            return {
                "sms_body": "I couldn't update the address in the system, but I saved it internally.",
                "scheduled_date": conv.get("scheduled_date"),
                "scheduled_time": conv.get("scheduled_time"),
                "address": conv["address"],
            }

        # =====================================================
        # 5️⃣ ADDITIONAL NOTES HANDLER
        # =====================================================
        note_keywords = ["note", "code", "gate", "also", "by the way"]

        if booking_id and any(k in inbound_lower for k in note_keywords):
            update_square_booking_notes(booking_id, inbound_text.strip())
            return {
                "sms_body": "Got it — I added that to your appointment notes.",
                "scheduled_date": conv.get("scheduled_date"),
                "scheduled_time": conv.get("scheduled_time"),
                "address": conv.get("address"),
            }

        # -----------------------------------------------------
        # HARD ADDRESS CAPTURE
        # -----------------------------------------------------
        if any(x in inbound_lower for x in [" st", " ave", " rd", "road", " ln", " lane", "dr", "drive"]) and len(inbound_text) > 6:
            conv["address"] = inbound_text.strip()
            address = conv["address"]

        # -----------------------------------------------------
        # BUILD SYSTEM PROMPT
        # -----------------------------------------------------
        system_prompt = build_system_prompt(
            cleaned_transcript,
            category,
            appointment_type,
            initial_sms,
            conv.get("scheduled_date"),
            conv.get("scheduled_time"),
            conv.get("address"),
            today_date_str,
            today_weekday,
            conv
        )

        # -----------------------------------------------------
        # LLM CALL
        # -----------------------------------------------------
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": inbound_text},
            ],
        )

        ai_raw     = json.loads(completion.choices[0].message.content)
        sms_body   = ai_raw.get("sms_body", "").strip()
        model_date = ai_raw.get("scheduled_date")
        model_time = ai_raw.get("scheduled_time")
        model_addr = ai_raw.get("address")

        # =====================================================
        #  S2 FALLBACK (Option A)
        #  Immediately after LLM call — before price injection
        # =====================================================
        s2 = maybe_create_square_booking(phone, conv)
        if s2 and s2.get("sms_body"):
            return s2

        # -----------------------------------------------------
        # PRICE INJECTION
        # -----------------------------------------------------
        appt = str(appointment_type).lower()
        price_map = {
            "eval_195": " The visit is a $195 consultation.",
            "troubleshoot_395": " The visit is a $395 troubleshoot and repair.",
            "whole_home_inspection": " Home inspections range from $375–$650 depending on size."
        }
        price_phrase = price_map.get(appt, "")
        if price_phrase and price_phrase not in sms_body:
            sms_body += price_phrase

        # -----------------------------------------------------
        # INHERIT KNOWN VALUES
        # -----------------------------------------------------
        if scheduled_date and not model_date:
            model_date = scheduled_date
        if scheduled_time and not model_time:
            model_time = scheduled_time
        if address and not model_addr:
            model_addr = address

        # -----------------------------------------------------
        # TIME INFERENCE
        # -----------------------------------------------------
        if model_time and not model_date:
            model_date = today_date_str

        # -----------------------------------------------------
        # HUMAN TIME
        # -----------------------------------------------------
        try:
            human_time = datetime.strptime(model_time, "%H:%M").strftime("%-I:%M %p")
        except:
            human_time = model_time

        final_sms = sms_body.replace(model_time, human_time) if (sms_body and model_time and human_time) else sms_body

        # =====================================================
        # AUTO-BOOKING ENGINE
        # =====================================================
        if model_date and model_time and model_addr and not conv.get("booking_created"):
            conv["scheduled_date"] = model_date
            conv["scheduled_time"] = model_time
            conv["address"]        = model_addr

            try:
                maybe_create_square_booking(phone, conv)

                if conv.get("booking_created"):
                    booking_id = conv.get("square_booking_id")
                    return {
                        "sms_body": f"You're all set — your appointment is booked for {model_date} at {human_time} at {model_addr}. Your confirmation number is {booking_id}.",
                        "scheduled_date": model_date,
                        "scheduled_time": model_time,
                        "address": model_addr,
                    }

            except Exception as e:
                print("AUTO-BOOKING ERROR:", repr(e))
                final_sms += " (Couldn't auto-book — we'll confirm manually.)"

        # -----------------------------------------------------
        # NORMAL RETURN
        # -----------------------------------------------------
        return {
            "sms_body": final_sms,
            "scheduled_date": model_date,
            "scheduled_time": model_time,
            "address": model_addr,
        }

    except Exception as e:
        print("Inbound reply FAILED:", repr(e))
        return {
            "sms_body": "Sorry — can you say that again?",
            "scheduled_date": scheduled_date,
            "scheduled_time": scheduled_time,
            "address": address,
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
# ---------------------------------------------------
def normalize_address(raw_address: str, forced_state: str | None = None) -> tuple[str, dict | None]:
    """
    Normalize a freeform address like:
      '45 Dickerman Ave Windsor Locks'
    into a Square-ready structure.

    Returns:
      ("ok", {address_struct})
      ("needs_state", None)  -> ask customer CT or MA
      ("error", None)        -> unable to normalize
    """
    if not GOOGLE_MAPS_API_KEY or not raw_address:
        print("normalize_address: missing API key or raw address")
        return "error", None

    try:
        params = {
            "address": raw_address,
            "key": GOOGLE_MAPS_API_KEY,
        }

        # Constrain to US; if forced_state is provided, bias to that state
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

        # If we don't have a state at all (and no forced_state), we need to ask CT vs MA
        if not state and not forced_state:
            print("normalize_address: missing state for", raw_address)
            return "needs_state", None

        # If the geocoder picked a state that is NOT CT/MA and we didn't force it, ask CT vs MA
        if state and state not in ("CT", "MA") and not forced_state:
            print("normalize_address: geocoded state is not CT/MA:", state, "for", raw_address)
            return "needs_state", None

        # If we explicitly forced_state, trust that state if geocoder cooperates
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
    Very simple customer create-or-get by phone number.
    """
    if not SQUARE_ACCESS_TOKEN:
        print("Square not configured; skipping customer create.")
        return None

    # Try search by phone
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

    # Create new customer
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
    if appointment_type == "EVAL_195":
        return SERVICE_VARIATION_EVAL_ID, SERVICE_VARIATION_EVAL_VERSION
    if appointment_type == "WHOLE_HOME_INSPECTION":
        return SERVICE_VARIATION_INSPECTION_ID, SERVICE_VARIATION_INSPECTION_VERSION
    if appointment_type == "TROUBLESHOOT_395":
        return SERVICE_VARIATION_TROUBLESHOOT_ID, SERVICE_VARIATION_TROUBLESHOOT_VERSION
    return None, None


def is_weekend(date_str: str) -> bool:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return d.weekday() >= 5
    except Exception:
        return False


def is_within_normal_hours(time_str: str) -> bool:
    try:
        t = datetime.strptime(time_str, "%H:%M").time()
        return BOOKING_START_HOUR <= t.hour <= BOOKING_END_HOUR
    except Exception:
        return False



# ---------------------------------------------------
# Cancel an existing Square booking
# ---------------------------------------------------
def cancel_square_booking(booking_id: str) -> bool:
    """
    Cancels a Square booking. Returns True if successful.
    """
    try:
        url = f"https://connect.squareup.com/v2/bookings/{booking_id}/cancel"

        payload = {
            "idempotency_key": f"cancel-{booking_id}-{uuid.uuid4()}",
        }

        resp = requests.post(
            url,
            headers=square_headers(),
            json=payload,
            timeout=10
        )

        if resp.status_code in (200, 201):
            print("Booking cancelled:", booking_id)
            return True

        print("Square cancel failed:", resp.status_code, resp.text)
        return False

    except Exception as e:
        print("Square cancel exception:", repr(e))
        return False



# ---------------------------------------------------
# Update a booking's address AFTER it has already been created
# ---------------------------------------------------
def update_square_booking_address(booking_id: str, new_address: str) -> bool:
    """
    Updates the booking.address field. Square booking objects CANNOT contain 'country'.
    """
    try:
        status, addr_struct = normalize_address(new_address)

        if status != "ok":
            print("Address normalization failed; update aborted.")
            return False

        booking_address = {
            "address_line_1": addr_struct["address_line_1"],
            "locality": addr_struct["locality"],
            "administrative_district_level_1": addr_struct["administrative_district_level_1"],
            "postal_code": addr_struct["postal_code"],
        }

        if addr_struct.get("address_line_2"):
            booking_address["address_line_2"] = addr_struct["address_line_2"]

        payload = {
            "idempotency_key": f"update-addr-{booking_id}-{uuid.uuid4()}",
            "booking": { "address": booking_address },
        }

        resp = requests.put(
            f"https://connect.squareup.com/v2/bookings/{booking_id}",
            headers=square_headers(),
            json=payload,
            timeout=10
        )

        if resp.status_code in (200, 201):
            print("Booking address updated:", booking_id, booking_address)
            return True

        print("Square update-address failed:", resp.status_code, resp.text)
        return False

    except Exception as e:
        print("Square update-address exception:", repr(e))
        return False



# ---------------------------------------------------
# Append additional customer notes to an existing booking
# ---------------------------------------------------
def update_square_booking_notes(booking_id: str, note_text: str) -> bool:
    """
    Appends note_text to the booking.customer_note field.
    """
    try:
        # Fetch current booking
        resp = requests.get(
            f"https://connect.squareup.com/v2/bookings/{booking_id}",
            headers=square_headers(),
            timeout=10
        )

        if resp.status_code not in (200, 201):
            print("Failed to fetch booking for note update:", resp.text)
            return False

        booking = resp.json().get("booking", {})
        current_note = booking.get("customer_note") or ""

        updated_note = (current_note + "\n" + note_text).strip()

        payload = {
            "idempotency_key": f"update-note-{booking_id}-{uuid.uuid4()}",
            "booking": { "customer_note": updated_note },
        }

        resp2 = requests.put(
            f"https://connect.squareup.com/v2/bookings/{booking_id}",
            headers=square_headers(),
            json=payload,
            timeout=10
        )

        if resp2.status_code in (200, 201):
            print("Booking notes updated:", booking_id)
            return True

        print("Square update-notes failed:", resp2.status_code, resp2.text)
        return False

    except Exception as e:
        print("Square update-notes exception:", repr(e))
        return False




# ---------------------------------------------------
# Create Square Booking (with address normalization + duration + conflict check)
# ---------------------------------------------------
def maybe_create_square_booking(phone: str, convo: dict) -> dict | None:
    """
    Attempts to create a Square booking.

    NOW INCLUDES:
    • Option C duration logic (Eval 60 / Troubleshoot 120 / Inspection 120)
    • Option S2 conflict fallback: "We're booked then, but I can do X or Y"
    • Hard conflict detection BEFORE booking
    • Safe return dict when booking cannot proceed
    """

    # Prevent double-book
    if convo.get("booking_created"):
        print("Booking already exists — skipping Square create.")
        return None

    scheduled_date = convo.get("scheduled_date")
    scheduled_time = convo.get("scheduled_time")
    raw_address    = convo.get("address")
    appointment_type = convo.get("appointment_type")

    if not (scheduled_date and scheduled_time and raw_address):
        print("Missing scheduling fields — cannot create booking.")
        return None

    # ---------------------------------------------------
    # DURATION LOGIC — OPTION C
    # ---------------------------------------------------
    appt_type = (appointment_type or "").upper()

    if appt_type == "EVAL_195":
        duration_minutes = 60
    elif appt_type == "TROUBLESHOOT_395":
        duration_minutes = 120
    elif appt_type == "WHOLE_HOME_INSPECTION":
        duration_minutes = 120
    else:
        duration_minutes = 60  # fallback safeguard

    variation_id, variation_version = map_appointment_type_to_variation(appointment_type)
    if not variation_id:
        print("Appointment type missing mapping:", appointment_type)
        return None

    # Weekend rules
    if is_weekend(scheduled_date) and appointment_type != "TROUBLESHOOT_395":
        print("Weekend blocked for non-emergency.")
        return None

    # Hours rule
    if appointment_type != "TROUBLESHOOT_395" and not is_within_normal_hours(scheduled_time):
        print("Requested time outside normal hours.")
        return None

    if not (SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID and SQUARE_TEAM_MEMBER_ID):
        print("Missing Square config — cannot book.")
        return None

    # ---------------------------------------------------
    # Normalize Address
    # ---------------------------------------------------
    addr_struct = convo.get("normalized_address")
    if not addr_struct:
        status, addr_struct = normalize_address(raw_address)

        if status == "ok":
            convo["normalized_address"] = addr_struct

        elif status == "needs_state":
            if not convo.get("state_prompt_sent"):
                send_sms(phone, "Just to confirm, is this address in Connecticut or Massachusetts?")
                convo["state_prompt_sent"] = True
            print("Needs CT/MA clarification before booking.")
            return None

        else:
            print("Address normalization failed:", raw_address)
            return None

    # ---------------------------------------------------
    # Travel time screening
    # ---------------------------------------------------
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
            print(f"Travel estimate: {travel_minutes:.1f} minutes")
            if travel_minutes > MAX_TRAVEL_MINUTES:
                print("Travel exceeds max — skip booking.")
                return None

    # ---------------------------------------------------
    # Convert Local Time → UTC
    # ---------------------------------------------------
    start_at_utc = parse_local_datetime(scheduled_date, scheduled_time)
    if not start_at_utc:
        print("Unable to convert datetime to UTC.")
        return None

    # ---------------------------------------------------
    # PRE-CHECK AVAILABILITY — prevents double booking!
    # ---------------------------------------------------
    availability = get_square_availability(
        scheduled_date,
        scheduled_time,
        duration_minutes
    )

    if availability == "blocked":
        print("Requested time is unavailable — sending S2 fallback.")

        return {
            "sms_body": "We’re booked then, but I can do 2:30 PM or 4:30 PM — which works?",
            "scheduled_date": None,
            "scheduled_time": None,
            "address": raw_address,
        }

    if availability != "open":
        print("Unknown availability response:", availability)
        return None

    # ---------------------------------------------------
    # Build Square booking payload
    # ---------------------------------------------------
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
            "customer_id": square_create_or_get_customer(phone, addr_struct),
            "start_at": start_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location_type": "CUSTOMER_LOCATION",
            "address": booking_address,
            "customer_note": f"Auto-booked by Prevolt OS. Raw address: {raw_address}",
            "appointment_segments": [
                {
                    "duration_minutes": duration_minutes,
                    "service_variation_id": variation_id,
                    "service_variation_version": variation_version,
                    "team_member_id": SQUARE_TEAM_MEMBER_ID,
                }
            ],
        },
    }

    # ---------------------------------------------------
    # POST to Square API
    # ---------------------------------------------------
    try:
        resp = requests.post(
            "https://connect.squareup.com/v2/bookings",
            headers=square_headers(),
            json=booking_payload,
            timeout=10,
        )

        if resp.status_code not in (200, 201):
            print("Square create FAILED:", resp.status_code, resp.text)
            return None

        booking_id = resp.json().get("booking", {}).get("id")

        convo["booking_created"] = True
        convo["square_booking_id"] = booking_id

        print(f"Square booking created successfully → {booking_id}")

        return {
            "sms_body": None,
            "scheduled_date": scheduled_date,
            "scheduled_time": scheduled_time,
            "address": raw_address,
        }

    except Exception as e:
        print("Square booking exception:", repr(e))
        return None


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
# Handle Residential vs Commercial
# ---------------------------------------------------
@app.route("/handle-call-selection", methods=["POST"])
def handle_call_selection():
    from twilio.twiml.voice_response import VoiceResponse

    digit = request.form.get("Digits", "")
    response = VoiceResponse()

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

    elif digit == "2":
        response.say(
            '<speak><prosody rate="90%">Connecting you now.</prosody></speak>',
            voice="Polly.Matthew-Neural"
        )
        response.dial("+15555555555")
        return Response(str(response), mimetype="text/xml")

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
