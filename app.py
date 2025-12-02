import os
import requests
from io import BytesIO
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
# Transcription Helper
# -------------------------------
def transcribe_recording(recording_url):
    """
    Downloads WAV audio from Twilio, wraps it in BytesIO,
    and sends it to OpenAI Whisper for transcription.
    """

    # Twilio default recording format = WAV
    audio_url = recording_url + ".wav"

    # Download audio bytes from Twilio
    audio_bytes = requests.get(audio_url).content

    # Wrap audio in a file-like object
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "voicemail.wav"  # Required for OpenAI to detect format

    # Transcribe using OpenAI
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file
    )

    return transcript.text


# -------------------------------
# Incoming Call → Send to Voicemail
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
# Voicemail Complete → Transcribe & Log
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
        transcript_text = transcribe_recording(recording_url)
        print("Transcript:")
        print(transcript_text)
    except Exception as e:
        print("Transcription FAILED:", str(e))
        print("Error:", str(e))
        transcript_text = None

    response = VoiceResponse()
    response.hangup()
    return Response(str(response), mimetype="text/xml")


# -------------------------------
# Render Fallback (local dev mode)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

