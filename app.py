import os
import json
import time
import uuid
import requests
from datetime import datetime
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
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

# Prevolt service IDs (from your booking URLs)
SERVICE_ID_EVAL = "2AANQPVPZDC5KK32LIA24MKW"          # On-Site Electrical Evaluation & Quote
SERVICE_ID_HOME_INSPECTION = "WA4NC3LKHU2JM2K4EWOYFOFZ"  # Full-Home Electrical Safety Inspection
SERVICE_ID_TROUBLESHOOT = "5FMVM7VONJ6SVGVZN6AKZFNW"     # 24/7 Electrical Troubleshooting & Diagnostics

# Non-emergency booking window (local time)
BOOKING_START_HOUR = 9   # 9:00
BOOKING_END_HOUR = 16    # 16:00 (4pm)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = (
    Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
    else None
)

app = Flask(__name__)

# Conversation memory
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
#   "address": ...,
#   "booking_created": bool,
#   "square_booking_id": str | None
# }


# ---------------------------------------------------
# Utility: Send outbound message via WhatsApp (testing)
# ---------------------------------------------------
def send_sms(to_number: str, body: str) -> None:
    """
    Force all outbound messages to WhatsApp sandbox to bypass A2P filtering.
    Currently always sends to your cell via WhatsApp sandbox.
    """
    if not twilio_client:
        print("Twilio not configured; WhatsApp message not sent.")
        print("Intended WhatsApp:", body)
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
                        "improve grammar slightly, but preserve meaning. Do NOT add info."
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
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Prevolt OS, the SMS assistant for Prevolt Electric. "
                        "Generate the FIRST outbound SMS after reading voicemail.\n\n"
                        "Rules:\n"
                        "• MUST start with: 'Hi, this is Prevolt Electric —'\n"
                        "• NEVER ask them to repeat their voicemail.\n"
                        "• Determine correct appointment type:\n"
                        "   - Installs/quotes/upgrades → EVAL_195\n"
                        "   - Active problems → TROUBLESHOOT_395\n"
                        "   - Whole house inspection → WHOLE_HOME_INSPECTION\n"
                        "• Mention price once only.\n"
                        "• No photos. No AI mentions. No Kyle.\n\n"
                        "Return STRICT JSON with keys: sms_body, category, appointment_type."
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
# Step 4 — Generate Replies (THE BRAIN)
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
        system_prompt = f"""
You are Prevolt OS, the SMS assistant for Prevolt Electric. Continue the conversation naturally.

===================================================
STRICT CONVERSATION FLOW RULES
===================================================
1. NEVER repeat a question already asked.
2. NEVER restart the conversation.
3. NEVER ask again for date, time, or address if already collected.
4. NEVER restate prices.
5. ALWAYS move the conversation forward.
6. If customer gives date AND time → accept once.
7. If customer gives address → accept once.
8. When date + time + address are all collected → send FINAL confirmation with NO question mark.
9. After customer replies “yes / sounds good / confirmed” → send NOTHING further.
10. No AI mentions. No quoting their text.
11. Keep messages short and human.

===================================================
SCHEDULING RULES (9am–4pm ONLY, LOCAL TIME)
===================================================
• Normal, non-emergency appointments are scheduled between 9:00 and 16:00 local time.
• If the customer proposes outside that window for a non-emergency:
  “We typically schedule between 9am and 4pm. What time in that window works for you?”
• Troubleshoot/emergency appointments (TROUBLESHOOT_395) can be outside that window if needed.

===================================================
EMERGENCY RULE
===================================================
If voicemail indicates an active issue AND customer says “now / ASAP / immediately”:
Use:
“We can prioritize this. What’s the earliest time today you can meet us at the property?”

===================================================
TENANT RULE
===================================================
If: “my tenant will schedule” → ONLY say:
“For scheduling and service details, we can only coordinate directly with you as the property owner.”

===================================================
VALUE / REASSURANCE (TROUBLESHOOT ONLY)
===================================================
Use ONCE:
“Most minor issues are handled during the troubleshoot visit, and we’re usually able to diagnose within the first hour.”

===================================================
AUTO-DETECTION
===================================================
You must detect and store:
• scheduled_date
• scheduled_time
• address

FORMAT REQUIREMENTS:
• scheduled_date MUST be 'YYYY-MM-DD' (example: 2025-12-03)
• scheduled_time MUST be 'HH:MM' in 24-hour local time (example: 14:30)
If you don't know yet, use null.

If a customer changes date, time, or address later → update the stored value.

===================================================
CONTEXT
===================================================
Original voicemail: {cleaned_transcript}
Category: {category}
Appointment type: {appointment_type}
Initial SMS: {initial_sms}
Stored date/time/address: {scheduled_date}, {scheduled_time}, {address}

===================================================
OUTPUT FORMAT (STRICT JSON)
===================================================
{{
  "sms_body": "...",
  "scheduled_date": "YYYY-MM-DD or null",
  "scheduled_time": "HH:MM or null",
  "address": "string or null"
}}
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
# Google Maps helper (optional)
# ---------------------------------------------------
def compute_travel_time_minutes(origin: str, destination: str) -> float | None:
    """
    Uses Google Maps Distance Matrix API to estimate travel time in minutes.
    This is a hook for future use with dynamic tech locations.
    For now, you can set DISPATCH_ORIGIN_ADDRESS or pass a tech address.
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


def square_create_or_get_customer(phone: str, address: str | None = None) -> str | None:
    """
    Very simple customer create-or-get by phone number.
    """
    if not SQUARE_ACCESS_TOKEN:
        print("Square not configured; skipping customer create.")
        return None

    try:
        # Try search by phone
        search_payload = {
            "query": {
                "filter": {
                    "phone_number": {"exact": phone.replace("whatsapp:", "")}
                }
            }
        }
        resp = requests.post(
            "https://connect.squareup.com/v2/customers/search",
            headers=square_headers(),
            json=search_payload,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            customers = data.get("customers", [])
            if customers:
                return customers[0]["id"]
    except Exception as e:
        print("Square search customer failed:", repr(e))

    # Create new customer
    try:
        customer_payload = {
            "idempotency_key": str(uuid.uuid4()),
            "given_name": "Prevolt Lead",
            "phone_number": phone.replace("whatsapp:", ""),
        }
        if address:
            customer_payload["address"] = {"address_line_1": address}

        resp = requests.post(
            "https://connect.squareup.com/v2/customers",
            headers=square_headers(),
            json=customer_payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["customer"]["id"]
    except Exception as e:
        print("Square create customer failed:", repr(e))
        return None


def parse_local_datetime(date_str: str, time_str: str) -> datetime | None:
    """
    Parse 'YYYY-MM-DD' and 'HH:MM' into aware datetime in America/New_York,
    then convert to UTC for Square.
    """
    if not date_str or not time_str:
        return None
    try:
        local_naive = datetime.strptime(
            f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
        )
        if ZoneInfo:
            local = local_naive.replace(tzinfo=ZoneInfo("America/New_York"))
        else:
            # Fallback: assume -05:00 (no DST handling)
            from datetime import timezone, timedelta as _td
            local = local_naive.replace(tzinfo=timezone(_td(hours=-5)))
        return local.astimezone(datetime.utc).replace(tzinfo=None)
    except Exception as e:
        print("Failed to parse local datetime:", repr(e))
        return None


def map_appointment_type_to_service_id(appointment_type: str) -> str | None:
    if appointment_type == "EVAL_195":
        return SERVICE_ID_EVAL
    if appointment_type == "WHOLE_HOME_INSPECTION":
        return SERVICE_ID_HOME_INSPECTION
    if appointment_type == "TROUBLESHOOT_395":
        return SERVICE_ID_TROUBLESHOOT
    return None


def is_weekend(date_str: str) -> bool:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        # Monday=0 ... Sunday=6
        return d.weekday() >= 5
    except Exception:
        return False


def is_within_normal_hours(time_str: str) -> bool:
    try:
        t = datetime.strptime(time_str, "%H:%M").time()
        return BOOKING_START_HOUR <= t.hour <= BOOKING_END_HOUR
    except Exception:
        return False


def maybe_create_square_booking(phone: str, convo: dict) -> None:
    """
    Create a Square booking once we have date, time, and address.
    - No duplicates (idempotency by phone+datetime+appt_type)
    - Weekend rule: only TROUBLESHOOT_395 on weekends
    - Non-emergency: 9–4 only
    """
    if convo.get("booking_created"):
        return

    scheduled_date = convo.get("scheduled_date")
    scheduled_time = convo.get("scheduled_time")
    address = convo.get("address")
    appointment_type = convo.get("appointment_type")

    if not (scheduled_date and scheduled_time and address):
        return

    service_id = map_appointment_type_to_service_id(appointment_type)
    if not service_id:
        print("Unknown appointment_type; cannot map to service:", appointment_type)
        return

    # Weekend rule: only TROUBLESHOOT_395 allowed Sat/Sun
    if is_weekend(scheduled_date) and appointment_type != "TROUBLESHOOT_395":
        print("Weekend non-emergency booking blocked:", phone, scheduled_date)
        return

    # Non-emergency time window rule (9–4)
    if appointment_type != "TROUBLESHOOT_395" and not is_within_normal_hours(
        scheduled_time
    ):
        print("Non-emergency time outside 9–4; booking not auto-created:", phone, scheduled_time)
        return

    if not SQUARE_ACCESS_TOKEN or not SQUARE_LOCATION_ID or not SQUARE_TEAM_MEMBER_ID:
        print("Square configuration incomplete; skipping booking creation.")
        return

    start_at_utc = parse_local_datetime(scheduled_date, scheduled_time)
    if not start_at_utc:
        print("Could not parse scheduled date/time; skipping booking.")
        return

    # Optional travel-time hook (currently using dispatch origin as best-effort)
    if DISPATCH_ORIGIN_ADDRESS:
        travel_minutes = compute_travel_time_minutes(DISPATCH_ORIGIN_ADDRESS, address)
        if travel_minutes is not None:
            print(
                f"Estimated travel from dispatch to job: ~{travel_minutes:.1f} minutes "
                f"for {phone} at {address}"
            )

    customer_id = square_create_or_get_customer(phone, address)
    if not customer_id:
        print("No customer_id; cannot create booking.")
        return

    idempotency_key = f"prevolt-{phone}-{scheduled_date}-{scheduled_time}-{appointment_type}"

    booking_payload = {
        "idempotency_key": idempotency_key,
        "booking": {
            "location_id": SQUARE_LOCATION_ID,
            "customer_id": customer_id,
            "start_at": start_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location_type": "CUSTOMER_LOCATION",
            "customer_note": f"Auto-booked by Prevolt OS. Customer address: {address}",
            "appointment_segments": [
                {
                    "duration_minutes": 60,  # All three are effectively 60-minute segments
                    "service_variation_id": service_id,
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
        convo["booking_created"] = True
        convo["square_booking_id"] = booking_id
        print(
            f"Square booking created for {phone}: {booking_id} "
            f"{scheduled_date} {scheduled_time} ({appointment_type})"
        )
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
# Voicemail Complete → Transcribe → Initial SMS
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
            "booking_created": False,
            "square_booking_id": None,
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

    if from_number.startswith("whatsapp:"):
        from_number = from_number.replace("whatsapp:", "")

    convo = conversations.get(from_number)

    if not convo:
        resp = MessagingResponse()
        resp.message(
            "Hi, this is Prevolt Electric — thanks for reaching out. "
            "What electrical work are you looking to have done?"
        )
        return Response(str(resp), mimetype="text/xml")

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

    if ai_reply.get("scheduled_date"):
        convo["scheduled_date"] = ai_reply["scheduled_date"]
    if ai_reply.get("scheduled_time"):
        convo["scheduled_time"] = ai_reply["scheduled_time"]
    if ai_reply.get("address"):
        convo["address"] = ai_reply["address"]

    # Attempt booking once we have a complete date + time + address
    try:
        maybe_create_square_booking(from_number, convo)
    except Exception as e:
        print("maybe_create_square_booking failed:", repr(e))

    # If final confirmation matched → stop responding
    if sms_body == "":
        return Response(str(MessagingResponse()), mimetype="text/xml")

    resp = MessagingResponse()
    resp.message(sms_body)
    return Response(str(resp), mimetype="text/xml")


# ---------------------------------------------------
# Follow-up Cron (10 minutes)
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
# Local Dev
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
