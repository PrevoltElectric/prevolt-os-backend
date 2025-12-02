import os
import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
        auth=(
            os.environ.get("TWILIO_ACCOUNT_SID"),
            os.environ.get("TWILIO_AUTH_TOKEN")
        )
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
            file=audio_file
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
        cleaned = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that cleans up voicemail transcriptions. "
                        "Fix misheard words (especially electrical terminology), refine grammar, "
                        "and preserve the caller's meaning. Be concise and accurate."
                    )
                },
                {"role": "user", "content": raw_text}
            ]
        ).choices[0].message.content

        return cleaned.strip()

    except Exception as e:
        print("Cleanup FAILED:", repr(e))
        return raw_text  # fallback


# ---------------------------------------------------------
# STEP 3 — Lead Analysis Engine (Core Prevolt Logic)
# ---------------------------------------------------------
def analyze_lead(cleaned_text: str) -> dict:
    """
    Evaluates the cleaned transcript and assigns:
    - Work type
    - Complexity score
    - Customer seriousness
    - Price sensitivity
    - Red flags
    - Final classification (HIGH / MEDIUM / LOW / REJECT)
    - Recommended action
    """

    try:
        analysis = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Prevolt OS, an expert electrical lead evaluator. "
                        "Analyze the voicemail and determine customer intent, seriousness, "
                        "price sensitivity, potential red flags, and classify the lead as "
                        "HIGH VALUE, MEDIUM VALUE, LOW VALUE, or REJECT. "
                        "Follow Kyle Prevost’s filtering rules strictly: "
                        "No price shoppers, no tire kickers, no rebate-based budgets, "
                        "no 'just looking for quotes', no customers dictating process, "
                        "no photo requests (automatic rejection), and no low-value work unless urgent. "
                        "Recommend one action: SEND_TEXT_1, SEND_TEXT_2, IGNORE, BLOCK, or HUMAN_REVIEW."
                    )
                },
                {"role": "user", "content": cleaned_text}
            ]
        ).choices[0].message.content

        return {"analysis": analysis}

    except Exception as e:
        print("Lead Analysis FAILED:", repr(e))
        return {"analysis": None}


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
        action="/voicemail-complete"
    )

    response.hangup()
    return Response(str(response), mimetype="text/xml")


# ---------------------------------------------------------
# Voicemail Complete → Transcribe → Clean → Analyze
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
        # Step 1 — Raw Whisper transcription
        raw_text = transcribe_recording(recording_url)
        print("\nRaw Transcript:")
        print(raw_text)

        # Step 2 — Clean corrected transcript
        cleaned_text = clean_transcript_text(raw_text)
        print("\nCleaned Transcript:")
        print(cleaned_text)

        # Step 3 — Lead evaluation
        analysis = analyze_lead(cleaned_text)
        print("\nLead Analysis:")
        print(analysis["analysis"])

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
