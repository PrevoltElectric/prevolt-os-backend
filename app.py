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
SQUARE_TEAM_MEMBER_ID = os.environ.get("SQUARE_TEAM_MEMBER_ID")  # required for bookings

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
DISPATCH_ORIGIN_ADDRESS = os.environ.get("DISPATCH_ORIGIN_ADDRESS")  # e.g. "Granby, CT"
TECH_CURRENT_ADDRESS = os.environ.get("TECH_CURRENT_ADDRESS")        # optional dynamic origin

# ---------------------------------------------------
# Square service variation data (final and verified)
# ---------------------------------------------------
SERVICE_VARIATION_EVAL_ID = "IPCUF6EPOYGWJUEFUZOXL2AZ"
SERVICE_VARIATION_EVAL_VERSION = 1764725435505

SERVICE_VARIATION_INSPECTION_ID = "EGGYZF6JRHFBWKRWEKWB2WYI"
SERVICE_VARIATION_INSPECTION_VERSION = 1764719028312

SERVICE_VARIATION_TROUBLESHOOT_ID = "I6XYKSUWBOQJ3WPNES4LG5WG"
SERVICE_VARIATION_TROUBLESHOOT_VERSION = 1764718988109

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
        # Local date/time for Option A date conversion
        if ZoneInfo:
            tz = ZoneInfo("America/New_York")
        else:
            tz = timezone(timedelta(hours=-5))
        now_local = datetime.now(tz)
        today_date_str = now_local.strftime("%Y-%m-%d")
        today_weekday = now_local.strftime("%A")

        system_prompt = f"""
You are Prevolt OS, the SMS assistant for Prevolt Electric. Continue the conversation naturally.

Today is {today_date_str}, a {today_weekday}, local time America/New_York.

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
SCHEDULING RULES
===================================================
NON-EMERGENCY APPOINTMENTS:
• Must be scheduled between 9am and 4pm local time.
• If customer suggests outside that window → ask once:
  “We typically schedule between 9am and 4pm. What time in that window works for you?”

EMERGENCY APPOINTMENTS (TROUBLESHOOT_395):
• Ignore the 9–4 rule entirely.
• Accept ANY time the customer gives.
• If the customer gives an impossible time (e.g., 1am, 3am):
    Ask ONCE:
    “We can come today. What time later this morning works for you?”
• Never repeat this message.
• Never revert back to non-emergency scheduling language.


===================================================
DATE CONVERSION (OPTION A)
===================================================
Convert natural language like “tomorrow at 10”, “this Thursday afternoon”,
“next Tuesday at 1” into:
• scheduled_date = YYYY-MM-DD
• scheduled_time = HH:MM (24-hour)
If only one is provided, ask once for the missing piece (date OR time).

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
• scheduled_date — in 'YYYY-MM-DD' (example: 2025-12-03)
• scheduled_time — in 'HH:MM' 24-hour format (example: 14:30)
• address — freeform, customer-typed address. Do NOT worry about ZIP; we handle that.

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
# Create Square Booking (with address normalization)
# ---------------------------------------------------
def maybe_create_square_booking(phone: str, convo: dict) -> None:
    """
    Create a Square booking once we have date, time, and address.
    Square bookings CANNOT include the 'country' field inside booking.address.
    Customer profiles CAN include country, but bookings cannot.
    """

    # Refresh service debug log on every booking attempt
    square_dump_services_debug()

    if convo.get("booking_created"):
        return

    scheduled_date = convo.get("scheduled_date")
    scheduled_time = convo.get("scheduled_time")
    raw_address = convo.get("address")
    appointment_type = convo.get("appointment_type")

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

    # Time window rule (non-emergency)
    if appointment_type != "TROUBLESHOOT_395" and not is_within_normal_hours(scheduled_time):
        print("Non-emergency time outside 9–4; booking not auto-created:", phone, scheduled_time)
        return

    if not (SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID and SQUARE_TEAM_MEMBER_ID):
        print("Square configuration incomplete; skipping booking creation.")
        return

    # Normalize or reuse
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
            print("Address needs CT/MA confirmation for:", raw_address)
            return
        else:
            print("Address normalization failed; cannot create booking for:", raw_address)
            return

    # Travel time check
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
                f"Estimated travel from origin to job: ~{travel_minutes:.1f} minutes "
                f"for {phone} at {destination_for_travel}"
            )
            if travel_minutes > MAX_TRAVEL_MINUTES:
                print("Travel exceeds max; skipping auto-book.")
                return

    # Create or find customer (customers CAN include country)
    customer_id = square_create_or_get_customer(phone, addr_struct)
    if not customer_id:
        print("No customer_id; cannot create booking.")
        return

    # Convert local → UTC
    start_at_utc = parse_local_datetime(scheduled_date, scheduled_time)
    if not start_at_utc:
        print("Could not parse scheduled date/time; skipping booking.")
        return

    idempotency_key = f"prevolt-{phone}-{scheduled_date}-{scheduled_time}-{appointment_type}"

    # ---------- *** CRITICAL FIX *** ----------
    # Square booking.address MUST NOT include "country"
    # Only these fields are permitted:
    #   address_line_1, address_line_2, locality,
    #   administrative_district_level_1, postal_code
    booking_address = {
        "address_line_1": addr_struct["address_line_1"],
        "locality": addr_struct["locality"],
        "administrative_district_level_1": addr_struct["administrative_district_level_1"],
        "postal_code": addr_struct["postal_code"],
    }
    # (Optional) Only include line_2 if present
    if "address_line_2" in addr_struct and addr_struct["address_line_2"]:
        booking_address["address_line_2"] = addr_struct["address_line_2"]
    # -----------------------------------------

    booking_payload = {
        "idempotency_key": idempotency_key,
        "booking": {
            "location_id": SQUARE_LOCATION_ID,
            "customer_id": customer_id,
            "start_at": start_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location_type": "CUSTOMER_LOCATION",
            "address": booking_address,   # <-- FIXED
            "customer_note": (
                f"Auto-booked by Prevolt OS. Raw address from customer: {raw_address}"
            ),
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

        convo["booking_created"] = True
        convo["square_booking_id"] = booking_id

        print(
            f"Square booking created for {phone}: {booking_id} "
            f"{scheduled_date} {scheduled_time} ({appointment_type})"
        )

    except Exception as e:
        print("Square booking exception:", repr(e))

# ---------------------------------------------------
# Square Debug Route — Find Prevolt Service Variations
# ---------------------------------------------------
@app.route("/debug/prevolt-services", methods=["GET"])
def debug_prevolt_services():
    """
    Returns ONLY the 3 Prevolt service variations we rely on:
      • On-Site Electrical Evaluation & Quote Visit
      • Full-Home Electrical Safety Inspection
      • 24/7 Electrical Troubleshooting & Diagnostics

    URL on Render:
        https://prevolt-os-backend.onrender.com/debug/prevolt-services
    """
    if not SQUARE_ACCESS_TOKEN:
        return ("Square credentials missing", 500)

    url = "https://connect.squareup.com/v2/catalog/search"

    payload = {
        "object_types": ["ITEM", "ITEM_VARIATION"]
    }

    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=12,
        )

        data = resp.json()
        objects = data.get("objects", [])

        TARGET_NAMES = {
            "On-Site Electrical Evaluation & Quote Visit": "EVALUATION",
            "Full-Home Electrical Safety Inspection": "INSPECTION",
            "24/7 Electrical Troubleshooting & Diagnostics": "TROUBLESHOOT",
        }

        results = []

        for obj in objects:
            if obj.get("type") != "ITEM":
                continue

            item_name = obj.get("item_data", {}).get("name", "")
            if item_name not in TARGET_NAMES:
                continue

            matched_as = TARGET_NAMES[item_name]
            item_id = obj.get("id")
            item_version = obj.get("version")
            variations = obj.get("item_data", {}).get("variations", [])

            results.append({
                "matched_as": matched_as,
                "service_name": item_name,
                "service_id": item_id,
                "service_version": item_version,
                "variations": variations,
            })

        return (
            json.dumps(results, indent=4),
            200,
            {"Content-Type": "application/json"},
        )

    except Exception as e:
        return (f"Error: {repr(e)}", 500)



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
[
  {
    "matched_as": "EVALUATION",
    "service_name": "On-site Electrical Evaluation",
    "service_id": "XXXXXX",
    "service_version": 1234,
    "variations": [...]
  },
  {
    "matched_as": "TROUBLESHOOT",
    "service_name": "24/7 Troubleshooting & Diagnostic",
    "service_id": "YYYYYY",
    "service_version": 5678,
    "variations": [...]
  },
  {
    "matched_as": "INSPECTION",
    "service_name": "Full Home Electrical Safety Inspection",
    "service_id": "ZZZZZZ",
    "service_version": 1357,
    "variations": [...]
  }
]



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

    # Normalize Twilio's WhatsApp prefix
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

    # Handle CT/MA reply after we asked specifically
    if convo.get("state_prompt_sent") and not convo.get("normalized_address"):
        upper = body.upper()
        if "CT" in upper or "CONNECTICUT" in upper:
            chosen_state = "CT"
        elif "MA" in upper or "MASS" in upper or "MASSACHUSETTS" in upper:
            chosen_state = "MA"
        else:
            # Not clearly CT or MA; ask again
            resp = MessagingResponse()
            resp.message("Please reply with either CT or MA so we can confirm the address.")
            return Response(str(resp), mimetype="text/xml")

        raw_address = convo.get("address")
        status, addr_struct = normalize_address(raw_address, forced_state=chosen_state)
        if status != "ok" or not addr_struct:
            # Still no good; ask for full address details
            resp = MessagingResponse()
            resp.message(
                "I still couldn't verify the address. "
                "Please reply with the full street, town, state, and ZIP code."
            )
            convo["state_prompt_sent"] = False
            return Response(str(resp), mimetype="text/xml")

        convo["normalized_address"] = addr_struct
        convo["state_prompt_sent"] = False

        # Attempt booking now that we have a fully normalized address
        try:
            maybe_create_square_booking(from_number, convo)
        except Exception as e:
            print("maybe_create_square_booking after CT/MA reply failed:", repr(e))

        resp = MessagingResponse()
        resp.message("Thanks — that helps. We have everything we need for your visit.")
        return Response(str(resp), mimetype="text/xml")

    # Normal conversational flow
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

