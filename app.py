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
SQUARE_TEAM_MEMBER_ID = os.environ.get("SQUARE_TEAM_MEMBER_ID")  # for bookings

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
DISPATCH_ORIGIN_ADDRESS = os.environ.get("DISPATCH_ORIGIN_ADDRESS")
TECH_CURRENT_ADDRESS = os.environ.get("TECH_CURRENT_ADDRESS")

# ---------------------------------------------------
# Square service variation data (final and verified)
# ---------------------------------------------------
SERVICE_VARIATION_EVAL_ID = "IPCUF6EPOYGWJUEFUZOXL2AZ"
SERVICE_VARIATION_EVAL_VERSION = 1764725435505

SERVICE_VARIATION_INSPECTION_ID = "EGGYZF6JRHFBWKRWEKWB2WYI"
SERVICE_VARIATION_INSPECTION_VERSION = 1764719028312

SERVICE_VARIATION_TROUBLESHOOT_ID = "I6XYKSUWBOQJ3WPNES4LG5WG"
SERVICE_VARIATION_TROUBLESHOOT_VERSION = 1764718988109

# Non-emergency booking window
BOOKING_START_HOUR = 9
BOOKING_END_HOUR = 16
MAX_TRAVEL_MINUTES = 60

openai_client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = (
    Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
    else None
)

app = Flask(__name__)

# Stored conversations
conversations = {}


# ---------------------------------------------------
# Send WhatsApp SMS
# ---------------------------------------------------
def send_sms(to_number: str, body: str) -> None:
    if not twilio_client:
        print("Twilio not configured; skipping WhatsApp send.")
        print("Intended:", body)
        return
    try:
        msg = twilio_client.messages.create(
            body=body,
            from_="whatsapp:+14155238886",
            to="whatsapp:+18609701727",
        )
        print("WhatsApp sent. SID:", msg.sid)
    except Exception as e:
        print("SMS send error:", repr(e))


# ---------------------------------------------------
# Step 1 — Transcription
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
        for chunk in resp.iter_content(8192):
            f.write(chunk)

    with open(tmp_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
    return transcript.text


# ---------------------------------------------------
# Step 2 — Cleanup transcription
# ---------------------------------------------------
def clean_transcript_text(raw_text: str) -> str:
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Clean up voicemail text for an electrical contractor. "
                        "Fix wording, transcription mistakes, and electrical terms but do NOT add new info."
                    ),
                },
                {"role": "user", "content": raw_text},
            ],
        )
        return completion.choices[0].message.content.strip()
    except:
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
                        "You are Prevolt OS. Generate the FIRST SMS.\n"
                        "Rules:\n"
                        "- Must start with: 'Hi, this is Prevolt Electric —'\n"
                        "- NEVER ask them to repeat voicemail\n"
                        "- Detect appointment type:\n"
                        "   Installs/quotes → EVAL_195\n"
                        "   Active issues → TROUBLESHOOT_395\n"
                        "   Whole home requests → WHOLE_HOME_INSPECTION\n"
                        "- Mention price once\n"
                        "- No photos. No AI mentions. No Kyle.\n"
                        "Return strict JSON: sms_body, category, appointment_type."
                    ),
                },
                {"role": "user", "content": cleaned_text},
            ],
        )
        data = json.loads(completion.choices[0].message.content)
        return {
            "sms_body": data["sms_body"],
            "category": data["category"],
            "appointment_type": data["appointment_type"],
        }
    except Exception:
        return {
            "sms_body": (
                "Hi, this is Prevolt Electric — I received your message. "
                "The next step is a $195 evaluation visit. What day works for you?"
            ),
            "category": "OTHER",
            "appointment_type": "EVAL_195",
        }


# ---------------------------------------------------
# Step 4 — Generate Replies (Brain)
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
):
    try:
        # local date for Option A date conversion
        if ZoneInfo:
            tz = ZoneInfo("America/New_York")
        else:
            tz = timezone(timedelta(hours=-5))
        now_local = datetime.now(tz)

        today_date_str = now_local.strftime("%Y-%m-%d")
        today_weekday = now_local.strftime("%A")

        system_prompt = f"""
You are Prevolt OS. Continue SMS conversation naturally.

Today is {today_date_str}, a {today_weekday}, local time America/New_York.

STRICT RULES:
1. Never repeat a question.
2. Never restart conversation.
3. Never re-ask for date/time/address already collected.
4. Never restate prices.
5. If customer gives date+time → accept once.
6. When date+time+address complete → final confirmation (NO question mark).
7. After customer confirms → send nothing further.
8. No AI mentions.

SCHEDULING RULES:
• Non-emergency visits: 9am–4pm
• TROUBLESHOOT_395 can be outside the window & weekends
• If they propose outside window on non-emergency:
  “We typically schedule between 9am and 4pm. What time in that window works for you?”

OPTION A DATE CONVERSION:
Convert natural phrases (“this Thursday”, “next Tuesday”, “tomorrow at 10”) into:
- scheduled_date = YYYY-MM-DD
- scheduled_time = HH:MM 24-hour
Only ask for missing part.

EMERGENCY RULE:
If active issue & customer says ASAP/now:
“We can prioritize this. What’s the earliest time today you can meet us at the property?”

TENANT RULE:
If they say “my tenant will schedule”:
“For scheduling and service details, we can only coordinate directly with you as the property owner.”

REASSURANCE (troubleshoot only, ONCE):
“Most minor issues are handled during the troubleshoot visit, and we’re usually able to diagnose within the first hour.”

AUTO-DETECTION REQUIRED:
Detect and store:
- scheduled_date (YYYY-MM-DD)
- scheduled_time (HH:MM)
- address

CONTEXT:
Original voicemail: {cleaned_transcript}
Category: {category}
Appointment type: {appointment_type}
Initial: {initial_sms}
Stored values: date={scheduled_date}, time={scheduled_time}, address={address}

RETURN STRICT JSON:
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

    except Exception:
        return {
            "sms_body": "Got it.",
            "scheduled_date": scheduled_date,
            "scheduled_time": scheduled_time,
            "address": address,
        }


# ---------------------------------------------------
# Google Maps
# ---------------------------------------------------
def compute_travel_time_minutes(origin: str, destination: str) -> float | None:
    if not GOOGLE_MAPS_API_KEY or not origin or not destination:
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
        elem = data["rows"][0]["elements"][0]
        if elem["status"] != "OK":
            return None
        return elem["duration"]["value"] / 60.0
    except:
        return None


# ---------------------------------------------------
# Square Helpers
# ---------------------------------------------------
def square_headers():
    return {
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def square_create_or_get_customer(phone, address=None) -> str | None:
    if not SQUARE_ACCESS_TOKEN:
        return None

    # search by phone
    try:
        resp = requests.post(
            "https://connect.squareup.com/v2/customers/search",
            headers=square_headers(),
            json={"query": {"filter": {"phone_number": {"exact": phone}}}},
            timeout=10,
        )
        data = resp.json()
        if "customers" in data and data["customers"]:
            return data["customers"][0]["id"]
    except:
        pass

    # create
    try:
        payload = {
            "idempotency_key": str(uuid.uuid4()),
            "given_name": "Prevolt Lead",
            "phone_number": phone,
        }
        if address:
            payload["address"] = {"address_line_1": address}

        resp = requests.post(
            "https://connect.squareup.com/v2/customers",
            headers=square_headers(),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["customer"]["id"]
    except:
        return None


def parse_local_datetime(date_str: str, time_str: str) -> datetime | None:
    if not date_str or not time_str:
        return None
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        if ZoneInfo:
            dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
        else:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=-5)))
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except:
        return None


def map_service_variation(appointment_type: str):
    if appointment_type == "EVAL_195":
        return SERVICE_VARIATION_EVAL_ID, SERVICE_VARIATION_EVAL_VERSION
    if appointment_type == "WHOLE_HOME_INSPECTION":
        return SERVICE_VARIATION_INSPECTION_ID, SERVICE_VARIATION_INSPECTION_VERSION
    if appointment_type == "TROUBLESHOOT_395":
        return SERVICE_VARIATION_TROUBLESHOOT_ID, SERVICE_VARIATION_TROUBLESHOOT_VERSION
    return None, None


def is_weekend(date_str: str) -> bool:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() >= 5
    except:
        return False


def is_within_normal_hours(time_str: str) -> bool:
    try:
        hour = int(time_str.split(":")[0])
        return BOOKING_START_HOUR <= hour <= BOOKING_END_HOUR
    except:
        return False


# ---------------------------------------------------
# Create Square Booking (FINAL)
# ---------------------------------------------------
def maybe_create_square_booking(phone: str, convo: dict) -> None:
    if convo.get("booking_created"):
        return

    scheduled_date = convo.get("scheduled_date")
    scheduled_time = convo.get("scheduled_time")
    address = convo.get("address")
    appointment_type = convo.get("appointment_type")

    if not (scheduled_date and scheduled_time and address):
        return

    variation_id, variation_version = map_service_variation(appointment_type)
    if not variation_id:
        print("Unknown appointment type, cannot map:", appointment_type)
        return

    # Weekend rule
    if is_weekend(scheduled_date) and appointment_type != "TROUBLESHOOT_395":
        print("Weekend blocked:", scheduled_date)
        return

    # Weekday time rule
    if appointment_type != "TROUBLESHOOT_395" and not is_within_normal_hours(scheduled_time):
        print("Outside 9–4 window:", scheduled_time)
        return

    # Travel time limit
    origin = TECH_CURRENT_ADDRESS or DISPATCH_ORIGIN_ADDRESS
    if origin:
        travel = compute_travel_time_minutes(origin, address)
        if travel and travel > MAX_TRAVEL_MINUTES:
            print("Travel too long:", travel, "minutes")
            return

    customer_id = square_create_or_get_customer(phone, address)
    if not customer_id:
        print("Customer missing; cannot create booking.")
        return

    start_utc = parse_local_datetime(scheduled_date, scheduled_time)
    if not start_utc:
        print("Bad datetime; cannot book.")
        return

    # REQUIRED BY SQUARE FOR CUSTOMER_LOCATION
    booking_payload = {
        "idempotency_key": f"prevolt-{phone}-{scheduled_date}-{scheduled_time}-{appointment_type}",
        "booking": {
            "location_id": SQUARE_LOCATION_ID,
            "customer_id": customer_id,
            "start_at": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location_type": "CUSTOMER_LOCATION",

            # SQUARE REQUIRED:
            "address": {
                "address_line_1": address
            },

            "customer_note": f"Auto-booked by Prevolt OS. Address: {address}",
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
            print("Square booking failed:", resp.status_code, resp.text)
            return

        booking_id = resp.json().get("booking", {}).get("id")
        convo["booking_created"] = True
        convo["square_booking_id"] = booking_id
        print(f"Square booking created:", booking_id)

    except Exception as e:
        print("Booking exception:", repr(e))


# ---------------------------------------------------
# Voice Call Handler
# ---------------------------------------------------
@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    resp = VoiceResponse()
    resp.say(
        "Thanks for calling Prevolt Electric. "
        "Please leave your name, address, and a brief description of your project. "
        "We will text you shortly."
    )
    resp.record(
        max_length=60,
        trim="do-not-trim",
        play_beep=True,
        action="/voicemail-complete",
    )
    resp.hangup()
    return Response(str(resp), mimetype="text/xml")


# ---------------------------------------------------
# Voicemail → Transcribe → SMS
# ---------------------------------------------------
@app.route("/voicemail-complete", methods=["POST"])
def voicemail_complete():
    caller = request.form.get("From")
    recording_url = request.form.get("RecordingUrl")

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
        print("Voicemail error:", repr(e))

    resp = VoiceResponse()
    resp.hangup()
    return Response(str(resp), mimetype="text/xml")


# ---------------------------------------------------
# Incoming SMS Handler
# ---------------------------------------------------
@app.route("/incoming-sms", methods=["POST"])
def incoming_sms():
    from_number = request.form.get("From", "")
    inbound = request.form.get("Body", "").strip()

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
        convo["cleaned_transcript"],
        convo["category"],
        convo["appointment_type"],
        convo["initial_sms"],
        inbound,
        convo.get("scheduled_date"),
        convo.get("scheduled_time"),
        convo.get("address"),
    )

    sms_body = ai_reply.get("sms_body", "")

    if ai_reply.get("scheduled_date"):
        convo["scheduled_date"] = ai_reply["scheduled_date"]
    if ai_reply.get("scheduled_time"):
        convo["scheduled_time"] = ai_reply["scheduled_time"]
    if ai_reply.get("address"):
        convo["address"] = ai_reply["address"]

    try:
        maybe_create_square_booking(from_number, convo)
    except Exception as e:
        print("Booking creation failed:", repr(e))

    if sms_body == "":
        return Response(str(MessagingResponse()), mimetype="text/xml")

    resp = MessagingResponse()
    resp.message(sms_body)
    return Response(str(resp), mimetype="text/xml")


# ---------------------------------------------------
# 10-minute Follow-up
# ---------------------------------------------------
@app.route("/cron-followups", methods=["GET"])
def cron_followups():
    now = time.time()
    sent = 0
    for phone, convo in conversations.items():
        if not convo.get("replied") and not convo.get("followup_sent"):
            if now - convo.get("first_sms_time", 0) >= 600:
                send_sms(phone, "Just checking in — still interested?")
                convo["followup_sent"] = True
                sent += 1
    return f"Sent {sent} follow-ups."


# ---------------------------------------------------
# Local Run
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


