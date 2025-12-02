from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

@app.route("/")
def home():
    return "Prevolt OS running"

@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    # Create Twilio VoiceResponse
    response = VoiceResponse()

    # Voicemail greeting (you can customize later)
    response.say(
        "Thanks for calling Prevolt Electric. "
        "Please leave your name, address, and a brief description of your project. "
        "We will review your message and text you shortly."
    )

    # Record voicemail
    response.record(
        max_length=60,          # 1 minute maximum voicemail
        play_beep=True,
        trim="do-not-trim",
        action="/voicemail-complete"  # Twilio calls this when recording finishes
    )

    # If caller hangs up before recording
    response.hangup()

    return Response(str(response), mimetype="text/xml")

@app.route("/voicemail-complete", methods=["POST"])
def voicemail_complete():
    # Capture Twilio recording data
    recording_url = request.form.get("RecordingUrl")
    caller = request.form.get("From")
    duration = request.form.get("RecordingDuration")

    print("New voicemail received:")
    print("From:", caller)
    print("Recording URL:", recording_url)
    print("Duration:", duration)

    # TODO: Add transcription step (Step 3)
    # TODO: Add lead scoring logic (Step 4)
    # TODO: Add SMS response logic (Step 5)

    response = VoiceResponse()
    response.hangup()
    return Response(str(response), mimetype="text/xml")
