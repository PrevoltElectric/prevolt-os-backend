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
# Transcription Helper (Twilio -> Whisper-1)
# -------------------------------
def transcribe_recording(recording_url: str) -> str:
    """
    Downloads the Twilio WAV recording to /tmp,
    then sends it to OpenAI Whisper (whisper-1) for transcription.
    """

    # Twilio default recording format = WAV
    audio_url = recording_url + ".wav"

    # Download audio stream safely
    resp = requests.get(audio_url, stream=True)
    resp.raise_for_status()

    tmp_path = "/tmp/prevolt_voicemail.wav"

    # Save to a temporary file on disk
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # Now call the OpenAI Speech-to-Text API (whisper-1)
    with open(tmp_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",   # Stable transcription model
            file=audio_file,
        )

    # transcript.text is the actual text string
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

    transcript_text = None
    try:
        transcript_text = transcribe_recording(recording_url)
        print("Transcript:")
        print(transcript_text)
    except Exception as e:
        print("Transcription FAILED:")
        print(repr(e))

    response = VoiceResponse()
    response.hangup()
    return Response(str(response), mimetype="text/xml")


# -------------------------------
# Render Fallback (local dev mode)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

