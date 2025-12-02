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

# In-memory conversation store (per caller)
# In production you’d replace this with a real database.
conversations = {}
# conversations[phone] = {
#   "cleaned_transcript": str,
#   "category": str,
#   "appointment_type": str,
#   "initial_sms": str,
#   "first_sms_time": float,
#   "replied": bool,
#   "followup_sent": bool,
#   "scheduled_date": str | None,
#   "scheduled_time": str | None,
#   "address": str | None,
# }


@app.route("/")
def home():
    return "Prevolt OS running"


# ---------------------------------------------------
# Utility: Send outbound message via WhatsApp (for testing)
# ---------------------------------------------------
def send_sms(to_number: str, body: str) -> None:
    """
    For now, all outbound messages go to your WhatsApp
    so we can bypass A2P carrier filtering.
    """
    if not twilio_client:
        print("Twilio not configured; WhatsApp message not sent.")
        print("Intended WhatsApp to your phone:", body)
        return

    try:
        whatsapp_from = "whatsapp:+14155238886"  # Twilio Sandbox number
        whatsapp_to = "whatsapp:+18609701727"    # <-- YOUR CELL HERE

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
    Downloads the Twilio WAV recording securely using Basic Auth,
    saves it to /tmp, then submits it to OpenAI Whisper (whisper-1).
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
            if chunk:
                f.write(chunk)

    with open(tmp_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

    return transcript.text


# ---------------------------------------------------
# Step 2 — Cleanup (fix mishears, polish text)
# ---------------------------------------------------
def clean_transcript_text(raw_text: str) -> str:
    """
    Uses AI to correct minor transcription errors, especially for electrical terms,
    and improve readability while preserving meaning.
    """
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You clean up voicemail transcriptions for an electrical contractor. "
                        "Fix obvious transcription mistakes and electrical terminology, "
                        "improve grammar slightly, but do not change the caller's intent "
                        "or add new information."
                    ),
                },
                {"role": "user", "content": raw_text},
            ],
        )
        cleaned = completion.choices[0].message.content
        return cleaned.strip()
    except Exception as e:
        print("Cleanup FAILED:", repr(e))
        return raw_text


# ---------------------------------------------------
# Step 3 — Generate Initial SMS (dynamic, personalized)
# ---------------------------------------------------
def generate_initial_sms(cleaned_text: str) -> dict:
    """
    Generates the FIRST outbound SMS to a new caller based on their voicemail.
    Must follow Prevolt rules:
      - Start with 'Hi, this is Prevolt Electric —' (only for this first message).
      - Do NOT ask them to repeat the voicemail.
      - Choose the correct appointment type:
          * $195 evaluation visit for installs / upgrades / quotes.
          * $395 troubleshoot/repair for active issues / problems.
          * Whole-home inspection (price band $375–$600) for full-house requests.
      - For whole-home inspection: ask square footage and mention range, but do NOT
        give an exact price yet.
      - Be short, professional, and decisive.
      - No mention of being automated or AI.
      - No Kyle, just Prevolt Electric.
    Returns a dict:
      {
         "sms_body": str,
         "category": str,
         "appointment_type": "EVAL_195" | "TROUBLESHOOT_395" | "WHOLE_HOME_INSPECTION" | "OTHER"
      }
    """
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Prevolt OS, the SMS assistant for Prevolt Electric, a high-end "
                        "electrical contractor in Connecticut and Massachusetts. You write the FIRST "
                        "outbound text message to a new customer after reviewing their voicemail transcript.\n\n"
                        "Rules:\n"
                        "1. The message MUST start with: 'Hi, this is Prevolt Electric —'.\n"
                        "2. Do NOT ask the customer to repeat what they said in the voicemail.\n"
                        "3. Use the transcript to understand what they want (EV charger, panel upgrade, "
                        "transfer switch, active problem, whole-home inspection, etc.).\n"
                        "4. Choose an appointment type:\n"
                        "   - Use 'EVAL_195' when they want an install, upgrade, panel work, generator/transfer switch, "
                        "     EV charger, or need a quote / evaluation for a specific project. Clearly state:\n"
                        "       'The first step is a $195 on-site evaluation visit.'\n"
                        "   - Use 'TROUBLESHOOT_395' when there is an active problem: burning smell, flickering, "
                        "     tripping breakers, partial power loss, overheating, or clear symptoms. State:\n"
                        "       'For active issues, we schedule a $395 troubleshoot/repair visit.'\n"
                        "   - Use 'WHOLE_HOME_INSPECTION' if they clearly want the entire home inspected or mention "
                        "     a full-house electrical inspection. Ask for square footage and mention:\n"
                        "       'Whole-home inspections range from $375 to $600 depending on the size of the home.'\n"
                        "     Do NOT give an exact price yet.\n"
                        "   - Use 'OTHER' if the request is clearly outside scope or not a good fit. In that case, "
                        "     politely indicate it's not a service you offer or not a good fit, and do not push to schedule.\n"
                        "5. Do not ask for photos. Do not give detailed quotes over text.\n"
                        "6. Keep the tone professional, confident, and concise.\n"
                        "7. Do NOT mention that you are automated or AI.\n"
                        "8. Do NOT use any personal names. Just Prevolt Electric.\n\n"
                        "Output STRICT JSON ONLY in this format:\n"
                        "{\n"
                        '  \"sms_body\": \"...\",\n'
                        '  \"category\": \"EV_CHARGER\" | \"PANEL\" | \"GENERATOR\" | \"TROUBLESHOOT\" | \"WHOLE_HOME\" | \"OTHER\",\n'
                        '  \"appointment_type\": \"EVAL_195\" | \"TROUBLESHOOT_395\" | \"WHOLE_HOME_INSPECTION\" | \"OTHER\"\n'
                        "}\n"
                    ),
                },
                {"role": "user", "content": cleaned_text},
            ],
        )
        content = completion.choices[0].message.content
        data = json.loads(content)
        return {
            "sms_body": data.get("sms_body", "").strip(),
            "category": data.get("category", "OTHER"),
            "appointment_type": data.get("appointment_type", "OTHER"),
        }
    except Exception as e:
        print("Initial SMS generation FAILED:", repr(e))
        # Fallback generic text if something breaks
        fallback = (
            "Hi, this is Prevolt Electric — I received your message. "
            "The next step is scheduling a visit so we can evaluate the work properly. "
            "The standard evaluation visit is $195. What day works for you?"
        )
        return {
            "sms_body": fallback,
            "category": "OTHER",
            "appointment_type": "EVAL_195",
        }


# ---------------------------------------------------
# Step 4 — Generate Reply for Inbound SMS / WhatsApp
# ---------------------------------------------------
def generate_reply_for_inbound(
    cleaned_transcript: str,
    category: str,
    appointment_type: str,
    initial_sms: str,
    inbound_text: str,
    scheduled_date: str | None,
    scheduled_time: str | None,
    address: str | None,
) -> dict:
    """
    Generates the NEXT message in an ongoing thread.

    Returns dict:
    {
      "sms_body": "...",
      "scheduled_date": "... or null",
      "scheduled_time": "... or null",
      "address": "... or null"
    }
    """
    try:
        system_prompt = (
            "You are Prevolt OS, the SMS assistant for Prevolt Electric. "
            "You are continuing an existing text conversation with a customer.\n\n"
            "CRITICAL FLOW RULES (DO ALL OF THESE):\n"
            "1. NEVER repeat the same question twice.\n"
            "2. NEVER restart the conversation.\n"
            "3. NEVER ask again for information the customer already provided.\n"
            "4. NEVER repeat big chunks of earlier messages.\n"
            "5. ALWAYS move the conversation forward.\n"
            "6. If the customer gives a specific time, CONFIRM it and do NOT ask for time again.\n"
            "7. If the customer gives a date (like 'next Tuesday'), CONFIRM it and do NOT ask again.\n"
            "8. If the customer gives an address, do NOT ask for address again.\n"
            "9. If you already have date, time, and address, send a clear final confirmation and stop asking questions.\n"
            "10. Keep responses short, natural, and human.\n"
            "11. Do NOT start with 'Hi, this is Prevolt Electric —' (that is ONLY for the first message).\n"
            "12. Do NOT mention automation, AI, or that you are a bot.\n\n"
            "CONTEXT FROM THE VOICEMAIL AND FIRST MESSAGE:\n"
            f"- Original voicemail transcript: {cleaned_transcript}\n"
            f"- Classified category: {category}\n"
            f"- Appointment type: {appointment_type}\n"
            f"- First outbound SMS already sent: {initial_sms}\n\n"
            "CURRENT STORED DETAILS (THESE ARE ALREADY KNOWN):\n"
            f"- scheduled_date: {scheduled_date}\n"
            f"- scheduled_time: {scheduled_time}\n"
            f"- address: {address}\n\n"
            "APPOINTMENT LOGIC (KEEP ALL OF THESE RULES):\n"
            "• If this is a simple question (service area, licensing, availability, etc.), "
            "  answer the question directly and DO NOT push scheduling in that same message.\n"
            "• If they’re ready to move forward or ask what’s next, use the correct appointment type:\n"
            "   - EVAL_195 → say: 'The first step is a $195 on-site evaluation visit.'\n"
            "   - TROUBLESHOOT_395 → say: 'For active issues, we schedule a $395 troubleshoot/repair visit.'\n"
            "   - WHOLE_HOME_INSPECTION →\n"
            "       * If square footage is provided, calculate price:\n"
            "           - Under 1500 sq ft → $375\n"
            "           - 1500–2400 sq ft → $475\n"
            "           - Over 2400 sq ft → $600\n"
            "       * If square footage is NOT provided and you need it, ask for it ONCE.\n"
            "• No photos. No detailed project quotes over text.\n"
            "• No mentioning Kyle. No mentioning automation. No repeating the voicemail.\n\n"
            "AUTO-DETECTION & MEMORY RULES:\n"
            "• Read the customer's latest message and auto-detect if they gave:\n"
            "   - a date (example: 'next Tuesday', '12/6', 'tomorrow').\n"
            "   - a time (example: '2pm', '9:30 AM', 'tomorrow morning at 8').\n"
            "   - an address (anything that clearly looks like a street address).\n"
            "• If they provide a new date / time / address, include it in your JSON output.\n"
            "• If they change the time or date, update to the NEW value.\n"
            "• If all three are present (date, time, address), send a short final confirmation:\n"
            "   something like: 'Perfect, we’ll see you on [date] at [time] at [address].'\n\n"
            "OUTPUT FORMAT (STRICT JSON ONLY):\n"
            "{\n"
            '  \"sms_body\": \"your message back to the customer\",\n'
            '  \"scheduled_date\": \"date you detected or null if none\",\n'
            '  \"scheduled_time\": \"time you detected or null if none\",\n'
            '  \"address\": \"address you detected or null if none\"\n'
            "}\n"
            "Do NOT include any extra keys. Do NOT include explanations. JSON ONLY."
        )

        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": inbound_text},
            ],
        )
        content = completion.choices[0].message.content
        data = json.loads(content)

        # Normalise empty strings to None
        sms_body = (data.get("sms_body") or "").strip()
        scheduled_date_out = data.get("scheduled_date")
        scheduled_time_out = data.get("scheduled_time")
        address_out = data.get("address")

        if isinstance(scheduled_date_out, str) and not scheduled_date_out.strip():
            scheduled_date_out = None
        if isinstance(scheduled_time_out, str) and not scheduled_time_out.strip():
            scheduled_time_out = None
        if isinstance(address_out, str) and not address_out.strip():
            address_out = None

        return {
            "sms_body": sms_body or "Got it.",
            "scheduled_date": scheduled_date_out,
            "scheduled_time": scheduled_time_out,
            "address": address_out,
        }

    except Exception as e:
        print("Inbound reply generation FAILED:", repr(e))
        return {
            "sms_body": "Got it.",
            "scheduled_date": None,
            "scheduled_time": None,
            "address": None,
        }


# ---------------------------------------------------
# Voice: Incoming Call → Send to Voicemail
# ---------------------------------------------------
@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    response = VoiceResponse()

    response.say(
        "Thanks for calling Prevolt Electric. "
        "Please leave your name, address, and a brief description of your project. "
        "We will review your message and text you shortly."
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
# Voice: Voicemail Complete → Transcribe → SMS #1
# ---------------------------------------------------
@app.route("/voicemail-complete", methods=["POST"])
def voicemail_complete():
    recording_url = request.form.get("RecordingUrl")
    caller = request.form.get("From")
    duration = request.form.get("RecordingDuration")

    print("\n----- NEW VOICEMAIL -----")
    print("From:", caller)
    print("Recording URL:", recording_url)
    print("Duration (sec):", duration)

    try:
        # Step 1: Transcribe
        raw_text = transcribe_recording(recording_url)
        print("\nRaw Transcript:")
        print(raw_text)

        # Step 2: Clean transcript
        cleaned_text = clean_transcript_text(raw_text)
        print("\nCleaned Transcript:")
        print(cleaned_text)

        # Step 3: Generate initial SMS
        sms_info = generate_initial_sms(cleaned_text)
        sms_body = sms_info["sms_body"]
        category = sms_info["category"]
        appointment_type = sms_info["appointment_type"]

        print("\nInitial SMS Info:")
        print(json.dumps(sms_info, indent=2))

        # Send first SMS (currently via WhatsApp sandbox)
        send_sms(caller, sms_body)

        # Store conversation state
        conversations[caller] = {
            "cleaned_transcript": cleaned_text,
            "category": category,
            "appointment_type": appointment_type,
            "initial_sms": sms_body,
            "first_sms_time": time.time(),
            "replied": False,
            "followup_sent": False,
            "scheduled_date": None,
            "scheduled_time": None,
            "address": None,
        }

    except Exception as e:
        print("Voicemail pipeline FAILED:", repr(e))

    response = VoiceResponse()
    response.hangup()
    return Response(str(response), mimetype="text/xml")


# ---------------------------------------------------
# SMS / WhatsApp: Incoming Message Webhook (Twilio Messaging)
# ---------------------------------------------------
@app.route("/incoming-sms", methods=["POST"])
def incoming_sms():
    from_number = request.form.get("From", "")
    body = request.form.get("Body", "").strip()

    # Normalise WhatsApp numbers so conversations map correctly
    # WhatsApp sends "whatsapp:+1860xxxxxxx"
    if from_number.startswith("whatsapp:"):
        from_number = from_number.replace("whatsapp:", "")

    print("\n----- INCOMING SMS/WA MESSAGE -----")
    print("From:", from_number)
    print("Body:", body)

    convo = conversations.get(from_number)

    if not convo:
        # No voicemail context — treat as generic inbound text
        resp = MessagingResponse()
        resp.message(
            "Hi, this is Prevolt Electric — thanks for reaching out. "
            "What electrical work are you looking to have done?"
        )
        return Response(str(resp), mimetype="text/xml")

    # Mark that the customer has replied
    convo["replied"] = True

    cleaned_transcript = convo["cleaned_transcript"]
    category = convo["category"]
    appointment_type = convo["appointment_type"]
    initial_sms = convo["initial_sms"]
    scheduled_date = convo.get("scheduled_date")
    scheduled_time = convo.get("scheduled_time")
    address = convo.get("address")

    # Generate a context-aware reply + updated state
    ai_reply = generate_reply_for_inbound(
        cleaned_transcript=cleaned_transcript,
        category=category,
        appointment_type=appointment_type,
        initial_sms=initial_sms,
        inbound_text=body,
        scheduled_date=scheduled_date,
        scheduled_time=scheduled_time,
        address=address,
    )

    reply_text = ai_reply.get("sms_body", "Got it.").strip() or "Got it."
    new_date = ai_reply.get("scheduled_date")
    new_time = ai_reply.get("scheduled_time")
    new_address = ai_reply.get("address")

    # Update stored state if model extracted anything new
    if new_date:
        convo["scheduled_date"] = new_date
    if new_time:
        convo["scheduled_time"] = new_time
    if new_address:
        convo["address"] = new_address

    print("Reply SMS:", reply_text)
    print(
        "Stored state -> date:", convo.get("scheduled_date"),
        "| time:", convo.get("scheduled_time"),
        "| address:", convo.get("address"),
    )

    # Respond via TwiML so Twilio sends the SMS / WhatsApp message
    resp = MessagingResponse()
    resp.message(reply_text)
    return Response(str(resp), mimetype="text/xml")


# ---------------------------------------------------
# Follow-up Cron: 10-minute auto check-in
# ---------------------------------------------------
@app.route("/cron-followups", methods=["GET"])
def cron_followups():
    """
    Simple cron-style endpoint:
    Call this from an external scheduler (e.g. every minute).
    For any caller who received an initial SMS, has not replied in 10+ minutes,
    and has not already received a follow-up, send:

        'Just checking in — still interested?'
    """
    now = time.time()
    sent_count = 0

    for phone, convo in conversations.items():
        if convo.get("replied"):
            continue
        if convo.get("followup_sent"):
            continue

        first_time = convo.get("first_sms_time", 0)
        if first_time and (now - first_time) >= 600:  # 10 minutes
            followup = "Just checking in — still interested?"
            send_sms(phone, followup)
            convo["followup_sent"] = True
            sent_count += 1

    return f"Follow-up check complete. Sent {sent_count} follow-up message(s)."


# ---------------------------------------------------
# Local Dev Fallback
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


