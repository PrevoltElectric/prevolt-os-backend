import os
import json
import time
import uuid
import requests
from datetime import datetime, timezone, timedelta

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from openai import OpenAI

# Python 3.9+ zone handling
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ---------------------------------------------------
# Environment Variables
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


# ---------------------------------------------------
# Square Service Variation IDs (Final Verified)
# ---------------------------------------------------
# $195 On-Site Electrical Evaluation & Quote Visit
SERVICE_VARIATION_EVAL_ID = "IPCUF6EPOYGWJUEFUZOXL2AZ"
SERVICE_VARIATION_EVAL_VERSION = 1764725435505

# Whole-Home Electrical Safety Inspection
SERVICE_VARIATION_INSPECTION_ID = "LYK646AH4NAESCFUZL6PUTZ2"
SERVICE_VARIATION_INSPECTION_VERSION = 1764725393938

# $395 24/7 Electrical Troubleshooting & Diagnostics
SERVICE_VARIATION_TROUBLESHOOT_ID = "64IQNJYO3H6XNTLPIHABDJOQ"
SERVICE_VARIATION_TROUBLESHOOT_VERSION = 1762464315698


# ---------------------------------------------------
# Initialize Clients
# ---------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

twilio_client = (
    Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
    else None
)

app = Flask(__name__)


# ---------------------------------------------------
# In-Memory Conversation State
# ---------------------------------------------------
# Example structure:
# conversations[caller] = {
#     "cleaned_transcript": "",
#     "category": "",
#     "appointment_type": "",
#     "initial_sms": "",
#     "first_sms_time": 0,
#     "replied": False,
#     "followup_sent": False,
#     "scheduled_date": None,
#     "scheduled_time": None,
#     "address": None,
#     "normalized_address": None,
#     "booking_created": False,
#     "square_booking_id": None,
#     "state_prompt_sent": False,
# }
conversations = {}


# ---------------------------------------------------
# Utility: Forced WhatsApp Outbound for Testing
# ---------------------------------------------------
def send_sms(to_number: str, body: str) -> None:
    """
    All outbound messages go to WhatsApp sandbox for testing.
    Avoids A2P registration or SMS compliance issues.
    """
    if not twilio_client:
        print("Twilio not configured. Intended message:", body)
        return

    try:
        whatsapp_from = "whatsapp:+14155238886"  # Twilio Sandbox
        whatsapp_to = "whatsapp:+18609701727"    # <-- Your personal WhatsApp

        msg = twilio_client.messages.create(
            body=body,
            from_=whatsapp_from,
            to=whatsapp_to,
        )
        print("WhatsApp sent. SID:", msg.sid)

    except Exception as e:
        print("Failed to send WhatsApp message:", repr(e))
# ---------------------------------------------------
# Step 1 — Transcription (Twilio Recording → Whisper)
# ---------------------------------------------------
def transcribe_recording(recording_url: str) -> str:
    """
    Downloads the Twilio voicemail audio and sends it to Whisper for transcription.
    """
    audio_url = recording_url + ".wav"

    resp = requests.get(
        audio_url,
        stream=True,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
    )
    resp.raise_for_status()

    tmp_path = "/tmp/prevolt_voicemail.wav"

    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    with open(tmp_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

    return transcript.text


# ---------------------------------------------------
# Step 2 — Cleanup (Fix transcription noise)
# ---------------------------------------------------
def clean_transcript_text(raw_text: str) -> str:
    """
    Corrects obvious transcription mistakes but preserves meaning.
    """
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You clean up voicemail transcriptions for an electrical contractor. "
                        "Fix transcription errors, fix electrical terminology, fix grammar lightly. "
                        "Do NOT add meaning or embellish details."
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
# Step 3 — Generate Initial SMS
# ---------------------------------------------------
def generate_initial_sms(cleaned_text: str) -> dict:
    """
    Creates the FIRST outbound SMS to the customer after listening to voicemail.
    """
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Prevolt OS, the SMS assistant for Prevolt Electric.\n"
                        "Generate the FIRST outbound SMS after reading voicemail.\n\n"
                        "Rules:\n"
                        "• MUST start with: 'Hi, this is Prevolt Electric —'\n"
                        "• NEVER ask them to repeat their voicemail.\n"
                        "• Determine appointment type:\n"
                        "     - Installs/quotes/upgrades → EVAL_195\n"
                        "     - Active problems → TROUBLESHOOT_395\n"
                        "     - Whole-home inspection → WHOLE_HOME_INSPECTION\n"
                        "• Mention price once.\n"
                        "• No photos. No AI mentions. No Kyle.\n\n"
                        "Return STRICT JSON: {sms_body, category, appointment_type}"
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
                "The next step is a $195 on-site consultation and quote visit. What day works for you?"
            ),
            "category": "OTHER",
            "appointment_type": "EVAL_195",
        }


# ---------------------------------------------------
# Step 4 — The Brain (Inbound SMS → AI reply)
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
    """
    This is the decision engine for ALL inbound SMS replies.
    Handles:
    - Date parsing
    - Time parsing
    - Address parsing
    - Emergency vs non-emergency rules
    - Tenant rule
    - Final confirmations
    """

    try:
        # Local timezone
        if ZoneInfo:
            tz = ZoneInfo("America/New_York")
        else:
            tz = timezone(timedelta(hours=-5))

        now_local = datetime.now(tz)
        today_date_str = now_local.strftime("%Y-%m-%d")
        today_weekday = now_local.strftime("%A")

        system_prompt = f"""
You are Prevolt OS, the SMS assistant for Prevolt Electric.

===================================================
TODAY'S CONTEXT
===================================================
Today is {today_date_str}, a {today_weekday}, America/New_York.

===================================================
STRICT FLOW RULES
===================================================
1. NEVER repeat a question already asked.
2. NEVER restart conversation.
3. NEVER re-ask for date, time, or address if already stored.
4. NEVER restate prices.
5. ALWAYS move conversation forward.
6. If customer gives date + time → accept once.
7. If customer gives address → accept once.
8. Once all 3 (date, time, address) exist → send FINAL confirmation with NO '?'
9. After customer says “yes/confirmed/sounds good” → send NOTHING further.
10. Keep replies short and human. No AI mentions.

===================================================
SCHEDULING RULES
===================================================
NON-EMERGENCY (EVAL_195, INSPECTION):
• Must schedule 9am–4pm.
• If time outside window: ask once and NEVER ask again.

EMERGENCY (TROUBLESHOOT_395):
• Ignore 9–4 rule entirely.
• Accept any time.
• If they say “now / ASAP / immediately”:
    Respond: “We can prioritize this. What’s the earliest time today you can meet us at the property?”
• Only say that ONCE.

===================================================
TENANT RULE
===================================================
If customer says “my tenant will schedule,” reply EXACTLY:
“For scheduling and service details, we can only coordinate directly with you as the property owner.”

===================================================
VALUE MESSAGE (TROUBLESHOOT ONLY)
===================================================
Use ONCE:
“Most minor issues are handled during the troubleshoot visit, and we’re usually able to diagnose within the first hour.”

===================================================
YOU MUST OUTPUT STRICT JSON:
{{
  "sms_body": "...",
  "scheduled_date": "YYYY-MM-DD or null",
  "scheduled_time": "HH:MM or null",
  "address": "string or null"
}}

===================================================
CONTEXT TO USE:
Voicemail: {cleaned_transcript}
Category: {category}
Appointment type: {appointment_type}
Initial SMS: {initial_sms}
Stored date/time/address: {scheduled_date}, {scheduled_time}, {address}
"""

        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": inbound_text},
            ],
        )

        return json.loads(completion.choices[0].message.content)

    except Exception as e:
        print("Inbound reply FAILED:", repr(e))
        return {
            "sms_body": "Got it.",
            "scheduled_date": scheduled_date,
            "scheduled_time": scheduled_time,
            "address": address,
        }
# ---------------------------------------------------
# Google Maps — Travel Time Helper
# ---------------------------------------------------
def compute_travel_time_minutes(origin: str, destination: str) -> float | None:
    """
    Uses Google Maps Distance Matrix API to estimate travel time.
    Returns None if unavailable.
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

        elem = rows[0].get("elements", [])[0]
        if elem.get("status") != "OK":
            return None

        return elem["duration"]["value"] / 60.0

    except Exception as e:
        print("Travel time computation failed:", repr(e))
        return None


# ---------------------------------------------------
# Google Maps — Address Normalization (CT/MA aware)
# ---------------------------------------------------
def normalize_address(raw_address: str, forced_state: str | None = None) -> tuple[str, dict | None]:
    """
    Normalize a freeform address like:
        "45 Dickerman Ave Windsor Locks"
    → into structured components for Square.

    Returns:
        ("ok", struct)
        ("needs_state", None)
        ("error", None)
    """

    if not GOOGLE_MAPS_API_KEY or not raw_address:
        return "error", None

    try:
        params = {"address": raw_address, "key": GOOGLE_MAPS_API_KEY}

        # Restrict to US. If user later says CT/MA, force the state.
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

        if data.get("status") != "OK" or not data.get("results"):
            print("Geocode failure:", raw_address)
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
                line1 = comp["long_name"]
            if "route" in types:
                if line1:
                    line1 = f"{line1} {comp['long_name']}"
                else:
                    line1 = comp["long_name"]
            if "locality" in types:
                city = comp["long_name"]
            if "postal_town" in types and not city:
                city = comp["long_name"]
            if "administrative_area_level_1" in types:
                state = comp["short_name"]
            if "postal_code" in types:
                zipcode = comp["long_name"]

        # Still missing state → ask CT/MA
        if not state and not forced_state:
            return "needs_state", None

        # Invalid state (not CT or MA) → ask CT/MA
        if state and state not in ("CT", "MA") and not forced_state:
            return "needs_state", None

        final_state = forced_state or state

        # If still missing required fields → error
        if not (line1 and city and final_state and zipcode):
            return "error", None

        addr_struct = {
            "address_line_1": line1,
            "locality": city,
            "administrative_district_level_1": final_state,
            "postal_code": zipcode,
            "country": "US",  # IMPORTANT: this is removed later for Square bookings
        }

        return "ok", addr_struct

    except Exception as e:
        print("normalize_address exception:", repr(e))
        return "error", None


# ---------------------------------------------------
# Square API Helpers
# ---------------------------------------------------
def square_headers() -> dict:
    if not SQUARE_ACCESS_TOKEN:
        raise RuntimeError("Missing SQUARE_ACCESS_TOKEN")

    return {
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def square_create_or_get_customer(phone: str, address_struct: dict | None = None) -> str | None:
    """
    Lookup by phone. If customer exists → return ID.
    Otherwise create new one.
    """
    # Search first
    try:
        payload = {
            "query": {
                "filter": {
                    "phone_number": {"exact": phone}
                }
            }
        }
        resp = requests.post(
            "https://connect.squareup.com/v2/customers/search",
            headers=square_headers(),
            json=payload,
            timeout=10,
        )
        data = resp.json()
        customers = data.get("customers", [])
        if customers:
            return customers[0]["id"]
    except Exception as e:
        print("Square search customer failed:", repr(e))

    # Create new customer
    try:
        create_payload = {
            "idempotency_key": str(uuid.uuid4()),
            "given_name": "Prevolt Lead",
            "phone_number": phone,
        }

        # Customers CAN include "country".
        if address_struct:
            create_payload["address"] = {
                "address_line_1": address_struct["address_line_1"],
                "locality": address_struct["locality"],
                "administrative_district_level_1": address_struct["administrative_district_level_1"],
                "postal_code": address_struct["postal_code"],
                "country": "US",
            }

        resp = requests.post(
            "https://connect.squareup.com/v2/customers",
            headers=square_headers(),
            json=create_payload,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["customer"]["id"]

    except Exception as e:
        print("Square create customer failed:", repr(e))
        return None
# ---------------------------------------------------
# Create Square Booking (with address normalization)
# ---------------------------------------------------
def maybe_create_square_booking(phone: str, convo: dict) -> None:
    """
    Auto-create Square booking when:
      • date exists
      • time exists
      • address exists + normalized
    """
    if convo.get("booking_created"):
        return

    scheduled_date = convo.get("scheduled_date")
    scheduled_time = convo.get("scheduled_time")
    raw_address = convo.get("address")
    appointment_type = convo.get("appointment_type")

    if not (scheduled_date and scheduled_time and raw_address):
        return

    # Map type → variation ID/version
    variation_id, variation_version = map_appointment_type_to_variation(appointment_type)
    if not variation_id:
        print("Unknown appointment type:", appointment_type)
        return

    # Weekend rule (non-emergency)
    if is_weekend(scheduled_date) and appointment_type != "TROUBLESHOOT_395":
        print("Weekend non-emergency blocked:", phone)
        return

    # Time rule (non-emergency)
    if appointment_type != "TROUBLESHOOT_395" and not is_within_normal_hours(scheduled_time):
        print("Time outside 9–4 window:", scheduled_time)
        return

    # Address normalization
    addr_struct = convo.get("normalized_address")
    if not addr_struct:
        status, addr_struct = normalize_address(raw_address)

        if status == "ok":
            convo["normalized_address"] = addr_struct

        elif status == "needs_state":
            if not convo.get("state_prompt_sent"):
                send_sms(
                    phone,
                    "Just to confirm, is this address in Connecticut or Massachusetts?"
                )
                convo["state_prompt_sent"] = True
            return

        else:
            print("Address normalization failed:", raw_address)
            return

    # Travel time (optional, MAX_TRAVEL_MINUTES must be defined)
    origin = TECH_CURRENT_ADDRESS or DISPATCH_ORIGIN_ADDRESS
    if origin:
        destination = (
            f"{addr_struct['address_line_1']}, "
            f"{addr_struct['locality']}, "
            f"{addr_struct['administrative_district_level_1']} "
            f"{addr_struct['postal_code']}"
        )
        travel_minutes = compute_travel_time_minutes(origin, destination)

        if travel_minutes is not None:
            print(f"Travel estimate: ~{travel_minutes:.1f} minutes → {phone}")
            if travel_minutes > MAX_TRAVEL_MINUTES:
                print("Travel exceeds allowed threshold.")
                return

    # Create or retrieve customer
    customer_id = square_create_or_get_customer(phone, addr_struct)
    if not customer_id:
        print("No customer ID; cannot book.")
        return

    # Convert to UTC
    start_at_utc = parse_local_datetime(scheduled_date, scheduled_time)
    if not start_at_utc:
        print("Date/time parse failure:", scheduled_date, scheduled_time)
        return

    idempotency_key = f"prevolt-{phone}-{scheduled_date}-{scheduled_time}-{appointment_type}"

    # ---------------- CRITICAL FIX ----------------
    # DO NOT include "country" inside booking.address
    booking_address = {
        "address_line_1": addr_struct["address_line_1"],
        "locality": addr_struct["locality"],
        "administrative_district_level_1": addr_struct["administrative_district_level_1"],
        "postal_code": addr_struct["postal_code"],
    }
    if addr_struct.get("address_line_2"):
        booking_address["address_line_2"] = addr_struct["address_line_2"]
    # ------------------------------------------------

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
            print("Square booking FAILED:", resp.status_code, resp.text)
            return

        booking_id = resp.json().get("booking", {}).get("id")
        convo["booking_created"] = True
        convo["square_booking_id"] = booking_id

        print(f"BOOKED → {phone} / ID: {booking_id}")

    except Exception as e:
        print("Square booking exception:", repr(e))
# ---------------------------------------------------
# Voice: Incoming Call
# ---------------------------------------------------
@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    response = VoiceResponse()
    response.say(
        "Thanks for calling Prevolt Electric. "
        "Please leave your name, address, and a brief description of your project. "
        "We will text you shortly."
    )
    response.record(
        max_length=60,
        play_beep=True,
        trim="do-not-trim",
        action="/voicemail-complete",
    )
    response.hangup()
    return Response(str(response), mimetype="text/xml")


# ---------------------------------------------------
# Voicemail → Transcription → Cleanup → First SMS
# ---------------------------------------------------
@app.route("/voicemail-complete", methods=["POST"])
def voicemail_complete():
    recording_url = request.form.get("RecordingUrl")
    caller = request.form.get("From")

    try:
        raw = transcribe_recording(recording_url)
        cleaned = clean_transcript_text(raw)
        sms_info = generate_initial_sms(cleaned)

        send_sms(caller, sms_info["sms_body"])

        conversations[caller] = {
            "cleaned_transcript": cleaned,
            "category": sms_info["category"],
            "appointment_type": sms_info["appointment_type"],
            "initial_sms": sms_info["sms_body"],
            "first_sms_time": time.time(),
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

    except Exception as e:
        print("Voicemail fail:", repr(e))

    response = VoiceResponse()
    response.hangup()
    return Response(str(response), mimetype="text/xml")


# ---------------------------------------------------
# Incoming SMS / WhatsApp
# ---------------------------------------------------
@app.route("/incoming-sms", methods=["POST"])
def incoming_sms():
    from_number = request.form.get("From", "")
    body = request.form.get("Body", "").strip()

    # Normalize WhatsApp prefix
    if from_number.startswith("whatsapp:"):
        from_number = from_number.replace("whatsapp:", "")

    convo = conversations.get(from_number)

    # Cold inbound with no voicemail history
    if not convo:
        resp = MessagingResponse()
        resp.message(
            "Hi, this is Prevolt Electric — thanks for reaching out. "
            "What electrical work are you looking to have done?"
        )
        return Response(str(resp), mimetype="text/xml")

    # --------------------------
    # HANDLE CT/MA CONFIRMATION
    # --------------------------
    if convo.get("state_prompt_sent") and not convo.get("normalized_address"):
        upper = body.upper()

        if "CT" in upper or "CONNECTICUT" in upper:
            chosen_state = "CT"
        elif "MA" in upper or "MASS" in upper or "MASSACHUSETTS" in upper:
            chosen_state = "MA"
        else:
            resp = MessagingResponse()
            resp.message("Please reply with either CT or MA so we can confirm the address.")
            return Response(str(resp), mimetype="text/xml")

        raw_address = convo.get("address")
        status, addr_struct = normalize_address(raw_address, forced_state=chosen_state)

        if status != "ok":
            resp = MessagingResponse()
            resp.message(
                "I still couldn't verify the address. "
                "Please reply with the full street, town, state, and ZIP code."
            )
            convo["state_prompt_sent"] = False
            return Response(str(resp), mimetype="text/xml")

        convo["normalized_address"] = addr_struct
        convo["state_prompt_sent"] = False

        try:
            maybe_create_square_booking(from_number, convo)
        except Exception as e:
            print("maybe_create_square_booking after CT/MA reply failed:", repr(e))

        resp = MessagingResponse()
        resp.message("Thanks — that helps. We have everything we need for your visit.")
        return Response(str(resp), mimetype="text/xml")

    # --------------------------
    # NORMAL CONVERSATION FLOW
    # --------------------------
    convo["replied"] = True

    ai_reply = generate_reply_for_inbound(
        cleaned_transcript=convo["cleaned_transcript"],
        category=convo["category"],
        appointment_type=convo["appointment_type"],
        initial_sms=convo["initial_sms"],
        inbound_text=body,
        scheduled_date=convo.get("scheduled_date"),
        scheduled_time=convo.get("scheduled_time"),
        address=convo.get("address"),
    )

    sms_body = ai_reply.get("sms_body", "").strip()

    # Persist values
    if ai_reply.get("scheduled_date"):
        convo["scheduled_date"] = ai_reply["scheduled_date"]
    if ai_reply.get("scheduled_time"):
        convo["scheduled_time"] = ai_reply["scheduled_time"]
    if ai_reply.get("address"):
        convo["address"] = ai_reply["address"]

    # Attempt auto-book
    try:
        maybe_create_square_booking(from_number, convo)
    except Exception as e:
        print("maybe_create_square_booking failed:", repr(e))

    # Stop responding after final confirmation
    if sms_body == "":
        return Response(str(MessagingResponse()), mimetype="text/xml")

    resp = MessagingResponse()
    resp.message(sms_body)
    return Response(str(resp), mimetype="text/xml")


# ---------------------------------------------------
# 10-Minute Follow-Up
# ---------------------------------------------------
@app.route("/cron-followups", methods=["GET"])
def cron_followups():
    now = time.time()
    sent_count = 0

    for phone, convo in conversations.items():
        if convo.get("replied"):
            continue
        if convo.get("followup_sent"):
            continue

        if now - convo.get("first_sms_time", 0) >= 600:
            send_sms(phone, "Just checking in — still interested?")
            convo["followup_sent"] = True
            sent_count += 1

    return f"Sent {sent_count} follow-up(s)."


# ---------------------------------------------------
# Helper: Weekend Detection
# ---------------------------------------------------
def is_weekend(date_str: str) -> bool:
    try:
        y, m, d = map(int, date_str.split("-"))
        return datetime(y, m, d).weekday() >= 5
    except:
        return False


# ---------------------------------------------------
# Helper: 9am–4pm Rule (Non-emergency)
# ---------------------------------------------------
def is_within_normal_hours(time_str: str) -> bool:
    try:
        hour = int(time_str.split(":")[0])
        return 9 <= hour < 16
    except:
        return False


# ---------------------------------------------------
# Map Appointment Type → Square Variation
# ---------------------------------------------------
def map_appointment_type_to_variation(a_type: str):
    if a_type == "EVAL_195":
        return SERVICE_VARIATION_EVAL_ID, SERVICE_VARIATION_EVAL_VERSION
    if a_type == "WHOLE_HOME_INSPECTION":
        return SERVICE_VARIATION_INSPECTION_ID, SERVICE_VARIATION_INSPECTION_VERSION
    if a_type == "TROUBLESHOOT_395":
        return SERVICE_VARIATION_TROUBLESHOOT_ID, SERVICE_VARIATION_TROUBLESHOOT_VERSION
    return None, None


# ---------------------------------------------------
# Convert Local Date+Time → UTC datetime
# ---------------------------------------------------
def parse_local_datetime(date_str: str, time_str: str) -> datetime | None:
    try:
        if ZoneInfo:
            tz = ZoneInfo("America/New_York")
        else:
            tz = timezone(timedelta(hours=-5))

        dt_local = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        dt_local = dt_local.replace(tzinfo=tz)
        return dt_local.astimezone(timezone.utc)
    except Exception as e:
        print("parse_local_datetime failed:", repr(e))
        return None





# ---------------------------------------------------
# Flask Boot
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

