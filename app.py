import os
import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from openai import OpenAI

# Initialize OpenAI client using environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = Flask(__name__)

@app.route("/")
def home():
    return "Prevolt OS running"


# -------------------------------
# Transcription Helper (Twilio → Whisper-1)
# -------------------------------
def transcribe_recording(recording_url: str) -> str:
    """
    Downloads the Twilio WAV recording to /tmp,
    then sends it to OpenAI Whisper (whisper-1) for transcription.
    """

    audio_url = recording_url + ".wav"

    # Authenticate with Twilio to download audio
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

    # Save WAV audio to disk
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


# -------------------------------
# Clean-up Layer (Fix misheard words)
# -------------------------------
def clean_transcript_text(raw_text: str) -> str:
    """
    Uses AI to correct transcription errors, misheard electrical terms,
    and normalize grammar while preserving meaning.
    """

    try:
        cleaned = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that cleans up voicemail transcriptions. "
                        "Fix misheard words (especially electrical terminology), grammar, "
                        "and produce a corrected version without changing meaning. "
                        "Example: 'illogical work' → 'electrical work'."
                    )
                },
                {"role": "user", "content": raw_text}
            ]
        ).choices[0].message.content

        return cleaned.strip()

    except Exception as e:
        print("Cleanup FAILED:", repr(e))
        return raw_text  # fallback


# -------------------------------
# Incoming Call → Voicemail Recording
# -------------------------------
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


# -------------------------------
# Voicemail Complete → Transcribe + Clean + Log
# -------------------------------
@app.route("/voicemail-complete", methods=["POST"])
def voicemail_complete():
    recording_url = request.form.get("RecordingUrl")
    caller = request.form.get("From")
    duration = request.form.get("RecordingDuration")

    print("\n----- NEW VOICEMAIL -----")
    print("From:", caller)
    print("Recording:", recording_url)
    print("Duration:", duration)

    try:
        # Step 1: raw transcription
        raw_text = transcribe_recording(recording_url)
        print("\nRaw Transcript:")
        print(raw_text)

        # Step 2: cleaned transcription
        cleaned_text = clean_transcript_text(raw_text)
        print("\nCleaned Transcript:")
        print(cleaned_text)

    except Exception as e:
        print("\nTranscription FAILED:")
        print(repr(e))

    response = VoiceResponse()
    response.hangup()
    return Response(str(response), mimetype="text/xml")


# -------------------------------
# Render Fallback (for local dev)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

