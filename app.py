import os
import json
import time
import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from openai import OpenAI

# ---------------------------------------------------
# Environment & Clients
# ---------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")

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
#   "address": ...
# }


# ---------------------------------------------------
# Utility: Send outbound message via WhatsApp (testing)
# ---------------------------------------------------
def send_sms(to_number: str, body: str) -> None:
    """
    Force all outbound messages to WhatsApp sandbox to bypass A2P filtering.
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
                        "Return STRICT JSON."
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
                "The next step is a $195 evaluation visit. What day works for you?"
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
SCHEDULING RULES (9am–5pm ONLY)
===================================================
If customer proposes outside window:
“ We typically schedule between 9am and 5pm. What time in that window works for you?”

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
Detect and store:
• date
• time
• address

If changed → update.

===================================================
CONTEXT
===================================================
Original voicemail: {cleaned_transcript}
Category: {category}
Appointment type: {appointment_type}
Initial SMS: {initial_sms}
Stored date/time/address: {scheduled_date}, {scheduled_time}, {address}

===================================================
OUTPUT FORMAT
===================================================
{{
  "sms_body": "...",
  "scheduled_date": "... or null",
  "scheduled_time": "... or null",
  "address": "... or null"
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
