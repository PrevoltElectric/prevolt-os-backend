import os
import json
import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from openai import OpenAI

# -------------------------------
# Global Clients & Config
# -------------------------------
openai_api_key = os.environ.get("OPENAI_API_KEY")
twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
twilio_from_number = os.environ.get("TWILIO_FROM_NUMBER")

client = OpenAI(api_key=openai_api_key)
twilio_client = (
    Client(twilio_account_sid, twilio_auth_token)
    if twilio_account_sid and twilio_auth_token
    else None
)

app = Flask(__name__)


@app.route("/")
def home():
    return "Prevolt OS running"


# ---------------------------------------------------------
# STEP 1 — Transcription Helper (Twilio → Whisper-1)
# ---------------------------------------------------------
def transcribe_recording(recording_url: str) -> str:
    """
    Downloads the Twilio WAV recording securely using Basic Auth,
    saves it to /tmp, then submits it to OpenAI Whisper (whisper-1).
    """
    audio_url = recording_url + ".wav"

    # Auth with Twilio (Required to get the audio file)
    resp = requests.get(
        audio_url,
        stream=True,
        auth=(twilio_account_sid, twilio_auth_token),
    )
    resp.raise_for_status()

    tmp_path = "/tmp/prevolt_voicemail.wav"

    # Save audio to temporary file
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # Transcribe with Whisper
    with open(tmp_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

    return transcript.text


# ---------------------------------------------------------
# STEP 2 — Cleanup Layer (Fix mistakes, misheard words)
# ---------------------------------------------------------
def clean_transcript_text(raw_text: str) -> str:
    """
    Uses AI to correct misheard electrical terms and improve readability.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that cleans up voicemail transcriptions "
                        "for an electrical contractor. Fix misheard words (especially "
                        "electrical terminology), refine grammar, and preserve meaning. "
                        "Do NOT add new information."
                    ),
                },
                {"role": "user", "content": raw_text},
            ],
        )
        cleaned = completion.choices[0].message.content
        return cleaned.strip()
    except Exception as e:
        print("Cleanup FAILED:", repr(e))
        return raw_text  # fallback


# ---------------------------------------------------------
# STEP 3 — Lead Analysis Engine (Core Prevolt Logic)
# ---------------------------------------------------------
def analyze_lead(cleaned_text: str) -> dict:
    """
    Evaluates the cleaned transcript and returns a structured JSON dict:
    {
      "summary": str,
      "work_type": str,
      "complexity_score": int,
      "seriousness_score": int,
      "price_sensitivity_score": int,
      "red_flags": [str],
      "final_classification": "HIGH" | "MEDIUM" | "LOW" | "REJECT",
      "recommended_action": "SEND_TEXT_1" | "SEND_TEXT_2" | "IGNORE" | "BLOCK" | "HUMAN_REVIEW"
    }
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Prevolt OS, an expert electrical lead evaluator for "
                        "Prevolt Electric. Analyze the voicemail and return ONLY a JSON "
                        "object with these exact fields:\n"
                        "{\n"
                        '  "summary": str,\n'
                        '  "work_type": str,\n'
                        '  "complexity_score": int,\n'
                        '  "seriousness_score": int,\n'
                        '  "price_sensitivity_score": int,\n'
                        '  "red_flags": [str],\n'
                        '  "final_classification": "HIGH" | "MEDIUM" | "LOW" | "REJECT",\n'
                        '  "recommended_action": "SEND_TEXT_1" | "SEND_TEXT_2" | "IGNORE" | "BLOCK" | "HUMAN_REVIEW"\n'
                        "}\n"
                        "Rules:\n"
                        "- Prevolt does NOT chase work or beg.\n"
                        "- Be strict about price shoppers, rebate-capped jobs, people asking for free quotes, "
                        "and low-value work unless clearly urgent.\n"
                        "- Mention of sending photos or asking you to review photos is a red flag.\n"
                        "- Use REJECT when the lead is clearly not worth pursuing.\n"
                        "- Recommended_action should reflect your final judgment."
                    ),
                },
                {"role": "user", "content": cleaned_text},
            ],
        )
        content = completion.choices[0].message.content
        analysis = json.loads(content)
        return analysis
    except Exception as e:
        print("Lead Analysis FAILED:", repr(e))
        # If parsing failed, log raw content if available
        try:
            print("Raw analysis content:", content)
        except Exception:
            pass
        return {}


# ---------------------------------------------------------
# STEP 4 — SMS Follow-Up Engine (Text #1 / #2)
# ---------------------------------------------------------
def maybe_send_followup_sms(caller: str, cleaned_text: str, analysis: dict) -> None:
    """
    Uses the analysis dict to decide whether to send an SMS follow-up
    and which template to use.
    """
    if not twilio_client:
        print("Twilio client not configured; SMS not sent.")
        return

    if not twilio_from_number:
        print("TWILIO_FROM_NUMBER not set; SMS not sent.")
        return

    if not analysis:
        print("No analysis available; defaulting to HUMAN_REVIEW (no SMS).")
        return

    final_class = analysis.get("final_classification", "").upper()
    action = analysis.get("recommended_action", "").upper()
    summary = analysis.get("summary", "your electrical project")

    print(f"\nDecision: class={final_class}, action={action}")

    # Hard filters
    if final_class == "REJECT" or action in ("IGNORE", "BLOCK"):
        print("Lead rejected or set to ignore/block. No SMS sent.")
        return

    # Decide message body
    if action == "SEND_TEXT_1":
        body = (
            "Hi, this is Prevolt Electric. We received your message about "
            f"{summary}. Our first step is a $195 on-site visit to diagnose and "
            "quote your project. Would you like to schedule that visit?"
        )
    elif action == "SEND_TEXT_2":
        # Slightly softer follow-up template placeholder
        body = (
            "Hi, this is Prevolt Electric following up on your message about "
            f"{summary}. If you’d like to move forward, our standard process is "
            "a $195 on-site visit to evaluate and provide a written quote."
        )
    else:
        # HUMAN_REVIEW or unknown → for now, do not auto-text
        print("Action is HUMAN_REVIEW or unknown. No SMS sent automatically.")
        return

    try:
        msg = twilio_client.messages.create(
            body=body,
            from_=twilio_from_number,
            to=caller,
        )
        print(f"Follow-up SMS sent. SID: {msg.sid}")
    except Exception as e:
        print("Failed to send SMS:", repr(e))


# ---------------------------------------------------------
# Incoming Call → Direct to Voicemail
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Voicemail Complete → Transcribe → Clean → Analyze → SMS
# ---------------------------------------------------------
@app.route("/voicemail-complete", methods=["POST"])
def voicemail_complete():
    recording_url = request.form.get("RecordingUrl")
    caller = request.form.get("From")
    duration = request.form.get("RecordingDuration")

    print("\n----- NEW VOICEMAIL -----")
    print("From:", caller)
    print("Recording URL:", recording_url)
    print("Duration:", duration)

    try:
        # Step 1 — Raw transcription
        raw_text = transcribe_recording(recording_url)
        print("\nRaw Transcript:")
        print(raw_text)

        # Step 2 — Cleaned transcript
        cleaned_text = clean_transcript_text(raw_text)
        print("\nCleaned Transcript:")
        print(cleaned_text)

        # Step 3 — Lead analysis
        analysis = analyze_lead(cleaned_text)
        print("\nLead Analysis (JSON):")
        print(json.dumps(analysis, indent=2))

        # Step 4 — Optional SMS follow-up
        maybe_send_followup_sms(caller, cleaned_text, analysis)

    except Exception as e:
        print("Voicemail Pipeline FAILED:")
        print(repr(e))

    response = VoiceResponse()
    response.hangup()
    return Response(str(response), mimetype="text/xml")


# ---------------------------------------------------------
# Local Dev Mode Fallback
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

