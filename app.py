import os
import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import openai

# Load OpenAI API key from Render environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)

@app.route("/")
def home():
    return "Prevolt OS running"

# -------------------------------
# Transcription Helper
# -------------------------------
def transcribe_recording(recording_url):
    """
    Downloads voicemail audio from Twilio and sends it
    to OpenAI Whisper (gpt-4o-transcribe) for transcription.
    """

    # Twilio requires ".mp3" appended to get the audio file
    audio_url = recording_url + ".mp3"

    # Download audio bytes
    audio_data = requests.get(audio_url).content

    # Send to OpenAI transcription model
    transcript = openai.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=("voicemail.mp3", audio_data),
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

    # Attempt transcription
    try:
        transcript_text = transcribe_recording(recording_url)
        print("Transcript:")
        print(transcript_text)
    except Exception as e:
        print("Transcription FAILED:", str(e))
        transcript_text = None

    # Placeholder for next steps (scoring / SMS automation)
    # Step 4 and Step 5 will plug in here.

    response = VoiceResponse()
    response.hangup()
    return Response(str(response), mimetype="text/xml")


# -------------------------------
# Render Fallback (if gunicorn fails)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
